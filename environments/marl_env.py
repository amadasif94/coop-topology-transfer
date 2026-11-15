"""Maintained MARL environment: the ORIGINAL DESSEnv (primary MARL protocol, hash-pinned) plus protocol bookkeeping.

MarlDESSEnv subclasses environments.dess_env.DESSEnv without changing its physics, power flow, backup or reward:
  * reset(): records the start-of-day SoC s_i0 of every battery (fixed for the whole daily episode);
  * step(): BEFORE the action is executed, computes the pre-action feasible normalised bounds (l, h) from the
    authoritative battery state (environments.battery_feasibility.normalized_bounds); runs the original step unchanged (hard SoC
    projection, DOE (off), directional network backup, reward); and, on a genuinely completed daily episode
    (last of the 96 steps, no failed dispatch), adds the protocol's terminal term exactly once, using the SoC after the
    executed (post-backup) battery power, before any reset. Protocol 'b3' (historical): team-wide quadratic penalty
        r_i += -lambda_T * (1/N) * sum_j d_j^2,   d_j = max(0, s_j0 - s_jT) / (s_j0 - s_j_min)
Info keys (exported to step_metrics by the original evaluation_fields 'marl_' / 'reward_term' prefixes):
  marl_feasible_action_lower / _upper, marl_start_of_day_soc, marl_terminal_completed,
  marl_terminal_soc_shortfall, marl_terminal_penalty, reward_term_terminal_soc (and reward_terms_total).

Protocols 'b4' and 'soc_feasible' (config['experiment_protocol'], configs.experiments.apply_experiment; absent for 'b3'):
  * reset() in mode 'train' draws SoC_i0 ~ U(lo, hi) independently per battery from the env's own seeded generator
    (soc_rng, seeded from the env seed and re-seeded by reset(seed=...)), sets the batteries to it and rebuilds the
    reset observation; 'val'/'test' episodes keep the configured soc_init (0.50);
  * lambda_T must be 0 (no quadratic shortfall penalty); instead, on the completed day's last transition, every agent
    receives the team terminal valuation  w_grid * phi_cost * sum_j V_j  (V_j: terminal_energy_value, $), the same
    weighting and team distribution as the original feeder energy-cost term, so that
        sum_t (-w_grid*phi_cost*price_t*import_t) + w_grid*phi_cost*sum_j V_j = -w_grid*phi_cost * adjusted cost.
  Extra info keys: marl_day_mean_price, marl_stored_energy_start_kwh / _end_kwh, marl_terminal_energy_value_usd.
  * the completed day's last transition returns terminated=True (truncated stays True, so episode completion and
    evaluation are unchanged): the trainers store done = 1 there and no critic target bootstraps beyond the day,
    which the b4 discount gamma = 1.0 requires (finite-horizon undiscounted return).

Action mapping (protocol 'action_mapping'; 'rated' for b3/b4, 'soc_feasible' for the soc_feasible experiment):
  _prepare_action() (the original DESSEnv validation/clip/float32, then, under soc_feasible,
  environments.battery_feasibility.soc_feasible_action at the PRE-action SoC) turns the policy action a into the
  execution action a'. soc_feasible uses the AFFINE mapping (version 'affine_v2'):
  P_mapped = -P_ch_avail + (a + 1)/2 * (P_ch_avail + P_dis_avail), so a = 0 is the midpoint of the available interval,
  not 0 kW (0 kW only when the available charge and discharge powers are equal). Extra info keys:
  marl_action_mapping_version, marl_available_charge_kw / marl_available_discharge_kw (pre-action P_ch_avail, P_dis_avail). The original step records a (marl_raw_action) BEFORE this call and derives everything else from
  a' (marl_prepared_action; marl_requested_power_kw = a' * rating = P_mapped). Under soc_feasible the policy's feasible
  interval is the whole [-1, 1], so marl_feasible_action_lower/_upper are -1/+1. Every step also records, separately:
    marl_action_policy_normalized      a        (actor output after exploration noise and clipping; stored in replay)
    marl_action_execution_normalized   a'       (passed to Battery.step)
    marl_power_soc_mapped_kw           P_mapped (request of a', before the physical limiter and network backup)
    marl_power_limiter_kw              physical-limiter output
    marl_power_executed_kw             final dispatch after DOE (off) and network backup
    marl_soc_limiter_correction_kw     |P_mapped - limiter output|   (SoC/power correction; ~0 under soc_feasible)
    marl_backup_correction_kw          |limiter output - executed|   (network intervention, reported separately)

Dispatch mode (set_dispatch_mode; default 'policy', which is the unchanged RL path above):
  'direct_kw' (optimisation references only, e.g. evaluation.oracle PF-MPC): step() receives battery powers in kW
  (+ discharge / injection, - charge), converts them to the rated fraction a' = P / P_rated with the env's own
  DESSEnv._power_to_action (the unit conversion the network backup already uses) and SKIPS the policy action mapping
  (soc_feasible_action). Everything after that is unchanged: Battery.step's physical SoC/power limiter, DOE (off),
  directional network backup, AC power flow, curtailment, reward and accounting. Extra info keys in this mode only:
    marl_dispatch_source = 'direct_kw', marl_action_mapping = marl_action_mapping_version = 'direct_kw' (labels),
    marl_planned_power_kw            the commanded (planned) kW
    marl_planned_minus_limiter_kw    planned - physical-limiter output   (signed, kW)
    marl_planned_minus_executed_kw   planned - final executed power      (signed, kW; includes the backup)
"""
from environments.dess_env import DESSEnv
from environments.primary_marl import validate_primary_env
from experiments.load_stress.stress_env import check_scale, scenario, stress_config
import numpy as np

from environments.battery_feasibility import (SOC_FEASIBLE_MAPPING_VERSION, check_bounds, feasible_powers_kw,
                                              normalized_bounds, soc_feasible_action, terminal_penalty,
                                              terminal_shortfall)
from configs.experiments import action_mapping_of, action_mapping_version_of

SOC_STREAM = 0x50C0          # child stream of the env seed reserved for the b4 initial-SoC draw
DISPATCH_MODES = ('policy', 'direct_kw')


def terminal_energy_value(day_mean_price, soc_start, soc_end, capacity_kwh):
    """Two-sided terminal stored-energy valuation of each battery ($; positive = credit, negative = charge):

        V_i = p_day * (E_iT - E_i0) / 1000,    E_i = SoC_i * capacity_i  [kWh],   p_day [$/MWh]

    p_day = mean of the episode's 96 scaled prices. Exactly the convention of the adjusted net daily cost
    (experiments.no_shared_kpi.protocol.inventory_adjusted_cost and evaluation.reporting.metrics:
    adjusted = raw - p_day * (E_T - E_0) / 1000): stored-side kWh, charge/discharge efficiencies already act on SoC
    inside the battery model, so no extra efficiency factor (an extra factor would double-count them).
    """
    s0, sT, cap = (np.asarray(v, dtype=float) for v in (soc_start, soc_end, capacity_kwh))
    return float(day_mean_price) * (sT * cap - s0 * cap) / 1000.0


class MarlDESSEnv(DESSEnv):
    def __init__(self, config, mode='train', seed=None, lambda_terminal=None):
        if lambda_terminal is None or not np.isfinite(lambda_terminal) or lambda_terminal < 0:
            raise ValueError('MarlDESSEnv requires an explicit finite nonnegative lambda_terminal')
        self.lambda_terminal = float(lambda_terminal)
        self.start_of_day_soc = None
        self.protocol = config.get('experiment_protocol')
        if self.protocol is not None and self.lambda_terminal != 0.0:
            raise ValueError(f"experiment {self.protocol['name']} has no quadratic terminal penalty; lambda_T must be 0")
        self.soc_rng = np.random.default_rng(None if seed is None else [int(seed), SOC_STREAM])
        self.day_mean_price = None
        self.action_mapping = action_mapping_of(config)
        self.action_mapping_version = action_mapping_version_of(config)
        if self.action_mapping == 'soc_feasible' and self.action_mapping_version != SOC_FEASIBLE_MAPPING_VERSION:
            raise ValueError(f'this code implements soc_feasible mapping {SOC_FEASIBLE_MAPPING_VERSION}, '
                             f'configuration asks for {self.action_mapping_version}')
        self.dispatch_mode = 'policy'
        super().__init__(config=config, mode=mode, seed=seed)

    def set_dispatch_mode(self, mode):
        """'policy' (default; RL controllers) or 'direct_kw' (step() takes battery kW; optimisation references only)."""
        if mode not in DISPATCH_MODES:
            raise ValueError(f'dispatch mode must be one of {DISPATCH_MODES}, got {mode!r}')
        self.dispatch_mode = mode

    def _prepare_action(self, action):
        """Original validation/clip/float32, then (soc_feasible policy dispatch only) the SOC-feasible execution action
        a'. In direct_kw mode the action is already the rated fraction P / P_rated and is not mapped again."""
        prepared = super()._prepare_action(action)
        if self.action_mapping == 'soc_feasible' and self.dispatch_mode == 'policy':
            prepared = soc_feasible_action(prepared, self.batteries)
        return prepared

    def reset(self, seed=None, options=None):
        output = super().reset(seed=seed, options=options)
        if self.protocol is not None:
            ts = self.time_series
            prices = ts['df'][ts['price_col']].to_numpy(dtype=float)[self.episode_start_index:
                                                                     self.episode_start_index + self.episode_length]
            prices = prices * self.price_scaling_factor
            if len(prices) != self.episode_length or not np.isclose(
                    prices[0], self._get_profiles(self.episode_start_index)[3], rtol=0, atol=1e-9):
                raise ValueError('Day price extraction differs from DESSEnv._get_profiles')
            self.day_mean_price = float(prices.mean())
            if self.mode == 'train':
                output = self._randomize_initial_soc(output, seed)
        self.start_of_day_soc = np.array([b.get_soc() for b in self.batteries], dtype=float)
        return output

    def _randomize_initial_soc(self, output, seed):
        """b4 training only: SoC_i0 ~ U(lo, hi) per battery, once per episode; the reset observation is rebuilt so the
        observed SoC, the battery state, the feasible bounds and the terminal reference all use the drawn value."""
        if seed is not None:
            self.soc_rng = np.random.default_rng([int(seed), SOC_STREAM])
        lo, hi = self.protocol['initial_soc']['train_uniform']
        draw = self.soc_rng.uniform(lo, hi, self.num_agents)
        for battery, soc in zip(self.batteries, draw):
            if not battery.config.soc_min <= soc <= battery.config.soc_max:
                raise ValueError('Initial SoC draw outside the battery SoC limits')
            battery.soc = float(soc)
        obs, info = output
        if not getattr(self, '_reset_pf_failed', False):
            # Same inputs as DESSEnv.reset (zero battery power at reset, so the reset power flow does not depend on SoC).
            load_kw, _, pv_kw, price, _ = self._get_profiles(self.current_index)
            pv_used_kw, _, _ = self._apply_curtailment(load_kw=load_kw, pv_kw=pv_kw, dess_power_kw=self.last_dess_power_kw)
            obs = self._build_observation(load_kw=load_kw, pv_kw=pv_used_kw, price=price,
                                          voltage_pu=self.last_pf_result['voltage_pu'])
        return obs, info

    def step(self, action):
        if self.start_of_day_soc is None:
            raise RuntimeError('MarlDESSEnv.step before reset')
        if self.action_mapping == 'soc_feasible':      # every policy action in [-1, 1] is executable as mapped
            lower, upper = np.full(self.num_agents, -1.0, np.float32), np.full(self.num_agents, 1.0, np.float32)
        else:
            lower, upper = check_bounds(*normalized_bounds(self), self.num_agents)
        step_index = int(self.current_step)
        available_charge, available_discharge = feasible_powers_kw(self)       # pre-action, authoritative battery state
        planned = None
        if self.dispatch_mode == 'direct_kw':
            planned = np.asarray(action, dtype=float).reshape(-1)
            if planned.shape != (self.num_agents,) or not np.all(np.isfinite(planned)):
                raise ValueError(f'direct_kw dispatch needs {self.num_agents} finite kW values, got {action!r}')
            action = self._power_to_action(planned)          # rated fraction; Battery.step applies the physical limits
        next_obs, rewards, terminated, truncated, info = super().step(action)
        info.update(marl_available_charge_kw=available_charge, marl_available_discharge_kw=available_discharge)
        self._record_action_quantities(info)
        if planned is not None:
            self._record_direct_dispatch(info, planned)
        completed = bool(step_index == self.episode_length - 1 and not info.get('episode_failed', False))
        soc_end = np.array([b.get_soc() for b in self.batteries], dtype=float)
        shortfall = terminal_shortfall(self.start_of_day_soc, soc_end, [b.config.soc_min for b in self.batteries])
        penalty = terminal_penalty(self.lambda_terminal, shortfall) if completed else 0.0
        if self.protocol is not None:
            # b4: lambda_T = 0 (no quadratic term); the team terminal valuation replaces it, once, on the last transition.
            capacity = [b.config.capacity_kwh for b in self.batteries]
            value = terminal_energy_value(self.day_mean_price, self.start_of_day_soc, soc_end, capacity)
            weight = float(self.reward_cfg.get('w_grid', 0.60)) * float(self.reward_cfg.get('phi_cost', 1.0))
            penalty = weight * float(value.sum()) if completed else 0.0
            info.update(marl_day_mean_price=self.day_mean_price,
                        marl_stored_energy_start_kwh=self.start_of_day_soc * np.asarray(capacity),
                        marl_stored_energy_end_kwh=soc_end * np.asarray(capacity),
                        marl_terminal_energy_value_usd=value if completed else np.full(self.num_agents, np.nan))
            if completed and self.protocol['discount']['completed_day_terminal']:
                terminated = True                     # finite horizon: replay done = 1, no bootstrap past the day
                info['terminated'] = True
        if completed:
            rewards = (np.asarray(rewards, dtype=np.float32) + np.float32(penalty)).astype(np.float32)
        terms = np.full(self.num_agents, penalty)
        info.update(marl_feasible_action_lower=lower.copy(), marl_feasible_action_upper=upper.copy(),
                    marl_start_of_day_soc=self.start_of_day_soc.copy(), marl_terminal_completed=completed,
                    marl_terminal_soc_shortfall=shortfall if completed else np.full(self.num_agents, np.nan),
                    marl_terminal_penalty=penalty, reward_term_terminal_soc=terms)
        if 'reward_terms_total' in info:
            info['reward_terms_total'] = np.asarray(info['reward_terms_total'], dtype=float) + terms
        return next_obs, rewards, terminated, truncated, info

    @property
    def b3_start_soc(self):
        """Historical attribute name of start_of_day_soc (read-only)."""
        return self.start_of_day_soc

    def _record_action_quantities(self, info):
        """The four action quantities plus the limiter output, under explicit names (copies; nothing is overwritten)."""
        info['marl_action_mapping'] = self.action_mapping
        info['marl_action_mapping_version'] = self.action_mapping_version or 'rated'
        keys = ('marl_raw_action', 'marl_prepared_action', 'marl_requested_power_kw', 'physical_limited_power_kw',
                'final_battery_power_kw')
        if not all(k in info and info[k] is not None for k in keys):
            return                                          # failed dispatch: the original info has no dispatch
        mapped = np.asarray(info['marl_requested_power_kw'], dtype=float).copy()
        limiter = np.asarray(info['physical_limited_power_kw'], dtype=float).copy()
        executed = np.asarray(info['final_battery_power_kw'], dtype=float).copy()
        info.update(marl_action_policy_normalized=np.asarray(info['marl_raw_action'], dtype=float).copy(),
                    marl_action_execution_normalized=np.asarray(info['marl_prepared_action'], dtype=float).copy(),
                    marl_power_soc_mapped_kw=mapped, marl_power_limiter_kw=limiter, marl_power_executed_kw=executed,
                    marl_soc_limiter_correction_kw=np.abs(mapped - limiter),
                    marl_backup_correction_kw=np.abs(limiter - executed))

    @staticmethod
    def _record_direct_dispatch(info, planned):
        """direct_kw mode only: planned kW and its signed differences to the limiter output and the executed power."""
        info.update(marl_dispatch_source='direct_kw', marl_action_mapping='direct_kw',
                    marl_action_mapping_version='direct_kw', marl_planned_power_kw=planned.copy())
        if info.get('physical_limited_power_kw') is not None and info.get('final_battery_power_kw') is not None:
            limiter = np.asarray(info['physical_limited_power_kw'], dtype=float)
            executed = np.asarray(info['final_battery_power_kw'], dtype=float)
            info.update(marl_planned_minus_limiter_kw=planned - limiter, marl_planned_minus_executed_kw=planned - executed)


def make_marl_env(config, mode, seed, lambda_terminal, scale=1.0, env_class=MarlDESSEnv):
    """Original primary_config (+ load scale for stress tests only), protocol bookkeeping, original validations.

    env_class: MarlDESSEnv (A0, A1, A2, GNNs) or a subclass (A3: environments.local_reward_env.A3DESSEnv)."""
    if not issubclass(env_class, MarlDESSEnv):
        raise TypeError('env_class must be MarlDESSEnv or a subclass')
    env = env_class(config=stress_config(config, check_scale(scale)), mode=mode, seed=seed,
                    lambda_terminal=lambda_terminal)
    validate_primary_env(env, scenario(scale))
    if env.num_buses == 69:
        from experiments.no_shared_kpi.ieee69 import protocol as p69
        from experiments.no_shared_kpi.ieee69.tp10_provenance import runtime_evidence
        if env.num_agents != 9 or list(env.dess_buses) != p69.DESS_BUSES or env.agent_obs_dim != 10:
            raise ValueError('Wrong IEEE69 feeder/agent/interface')
        # TP10 line-file extension proof for TP1 train/val environments (None for every other case).
        env.tp10_extension_equivalence = runtime_evidence(env)
    if env.episode_length != 96 or abs(env.delta_t_hours - 0.25) > 1e-12:
        raise ValueError('MarlDESSEnv expects the original daily horizon of 96 fifteen-minute steps')
    return env


def terminal_coefficient(env):
    """Default lambda_T: training-data value of re-buying every battery's usable start-of-day energy.

        lambda_T = w_grid * phi_cost * mean_train_price[$/MWh] * sum_i (s_i0 - s_i_min) * E_i / eta_ch_i / 1000

    i.e. the reward-weighted energy-cost term for importing the energy that a full one-sided deficit (all d_i = 1)
    removes. Uses ONLY the training split's prices (never validation/test) and the env's own battery and reward
    configuration, so MLP and GNN controllers of one feeder get the identical value.
    """
    reward = env.reward_cfg
    train = np.asarray(env.splits['train_indices'], dtype=int)
    ts = env.time_series
    prices = ts['df'][ts['price_col']].to_numpy(dtype=float)[train] * env.price_scaling_factor
    for p in (0, len(train) // 2, len(train) - 1):      # same values as the env's own row reader
        if not np.isclose(env._get_profiles(int(train[p]))[3], prices[p], rtol=0, atol=1e-9):
            raise ValueError('Training price extraction differs from DESSEnv._get_profiles')
    usable_mwh = sum((b.config.soc_init - b.config.soc_min) * b.config.capacity_kwh / b.config.charge_efficiency
                     for b in env.batteries) / 1000.0
    w_grid, phi_cost = float(reward.get('w_grid', 0.60)), float(reward.get('phi_cost', 1.0))
    value = w_grid * phi_cost * float(prices.mean()) * usable_mwh
    return value, dict(rule='w_grid * phi_cost * mean_train_price * sum_i (s0-smin)E/eta_ch', w_grid=w_grid,
                       phi_cost=phi_cost, mean_train_price_per_mwh=float(prices.mean()),
                       train_price_samples=int(len(prices)), usable_replacement_energy_mwh=usable_mwh,
                       lambda_terminal=value)
