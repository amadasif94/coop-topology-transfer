"""A3 environment: the maintained MARL environment (original DESSEnv physics, power flow, physical limiter, directional
backup and protocol bookkeeping, unchanged) returning a strictly LOCAL, individual reward for every battery.

For battery i at bus b_i (P_i > 0 discharge, executed post-backup power; v_ref and all coefficients from the config):

    r_i(t) = w_grid * [ -phi_cost * dC_i(t) - phi_voltage_deviation * |V_{b_i}(t) - v_ref| ]
           + w_local * R_local,i(t)                                   (the original A0 local term, unchanged)
           + infeasible_action_penalty * 1[|P_i^exec - P_i^phys| > OVERRIDE_TOL_KW]   (own battery moved by backup)
           - kappa_i * d_i^2                                          (last step of a completed day only)

    dC_i(t)  = price(t) * dt / 1000 * ( max(0, n_i(t) - P_i(t)) - max(0, n_i(t)) )        E3 own-bus tariff
    n_i(t)   = load_{b_i}(t) - PV_{b_i}(t)       own-bus net load before the battery (available, uncurtailed PV)
    kappa_i  = lambda_T * R_i / sum_j R_j,       R_i = (s_i,init - s_i,min) * E_i / eta_ch,i
    d_i      = max(0, s_i0 - s_iT) / (s_i0 - s_i,min)              (environments.battery_feasibility.terminal_shortfall)

dC_i is the change of battery i's own-bus purchase cost caused by battery i under the SAME tariff convention as the
feeder cost of A0 (energy imported at the price, export uncompensated), measured at the battery's own bus instead of
the feeder head. Inputs per battery: price, own-bus load and available PV, own executed and physically limited power,
own-bus voltage, own SoC. Never used: feeder import, feeder KPIs (voltage-deviation sum, curtailment, grid stress),
other batteries' quantities, or the shared A0 reward. The A0 team reward computed by the parent is kept in info
(marl_a0_team_reward) for magnitude diagnostics only.

Failed dispatches keep the original reward: such transitions are never stored in replay (training.transition_semantics).

Experiment 'b4' (config['experiment_protocol']): the per-battery terminal term is battery i's OWN two-sided stored-energy
valuation instead of -kappa_i * d_i^2, weighted like its E3 economic term:
    terminal_i = w_grid * phi_cost * V_i,   V_i = p_day * (E_iT - E_i0) / 1000   (marl_env.terminal_energy_value)
p_day is the day's mean price (the same exogenous price signal every battery already receives); no other battery's
quantity enters. Summed over batteries this equals the cooperative team terminal valuation of A0-A2.
"""
import numpy as np

from environments.battery_feasibility import terminal_shortfall
from environments.marl_env import MarlDESSEnv

REWARD_DEFINITION = 'a3_local_e3_own_bus_tariff'
OVERRIDE_TOL_KW = 1e-6     # same tolerance as DESSEnv's physical_clipped flag


def own_bus_cost_change(price, net_load_kw, power_kw, dt_hours):
    """E3: change of the own-bus purchase cost caused by the battery ($); negative = saving."""
    net_load_kw, power_kw = np.asarray(net_load_kw, float), np.asarray(power_kw, float)
    return float(price) * dt_hours / 1000.0 * (np.maximum(0.0, net_load_kw - power_kw) - np.maximum(0.0, net_load_kw))


def terminal_coefficients(batteries, lambda_terminal):
    """kappa_i = lambda_T * R_i / sum_j R_j: each battery's share of the quadratic-terminal (protocol b3) replacement-energy value."""
    replacement = np.array([(b.config.soc_init - b.config.soc_min) * b.config.capacity_kwh / b.config.charge_efficiency
                            for b in batteries], dtype=float)
    return float(lambda_terminal) * replacement / replacement.sum()


def local_reward_terms(cfg, price, dt_hours, v_ref, net_load_kw, power_kw, physical_kw, voltage_pu, local_term,
                       shortfall, kappa, completed, terminal_value_usd=None):
    """Per-battery A3 reward terms from that battery's own quantities only (arrays indexed by battery).

    terminal_value_usd (b4): each battery's own terminal valuation V_i; replaces -kappa_i * d_i^2 when given."""
    w_grid = float(cfg.get('w_grid', 0.60))
    economic = -w_grid * float(cfg.get('phi_cost', 1.0)) * own_bus_cost_change(price, net_load_kw, power_kw, dt_hours)
    voltage = -w_grid * float(cfg.get('phi_voltage_deviation', 1.0)) * np.abs(np.asarray(voltage_pu, float) - v_ref)
    overridden = np.abs(np.asarray(power_kw, float) - np.asarray(physical_kw, float)) > OVERRIDE_TOL_KW
    backup = np.where(overridden, float(cfg.get('infeasible_action_penalty', -50.0)), 0.0)
    if not completed:
        terminal = np.zeros(len(economic))
    elif terminal_value_usd is not None:
        terminal = w_grid * float(cfg.get('phi_cost', 1.0)) * np.asarray(terminal_value_usd, float)
    else:
        terminal = -np.asarray(kappa, float) * np.asarray(shortfall, float) ** 2
    return dict(economic=economic, voltage=voltage, local=np.asarray(local_term, float), backup=backup,
                terminal_soc=terminal)


class A3DESSEnv(MarlDESSEnv):
    reward_definition = REWARD_DEFINITION

    def __init__(self, config, mode='train', seed=None, lambda_terminal=None):
        super().__init__(config=config, mode=mode, seed=seed, lambda_terminal=lambda_terminal)
        if self.doe_enabled or self.backup_mode != 'directional_incremental' or \
                self.backup_cfg.get('residual_penalty_enabled', False):
            raise ValueError('A3 requires the primary protocol: DOE off, directional backup, no residual penalty')

    @property
    def terminal_coefficients(self):
        return terminal_coefficients(self.batteries, self.lambda_terminal)

    def step(self, action):
        load_kw, _, pv_kw, price, _ = self._get_profiles(self.current_index)    # this step's row, as DESSEnv.step
        buses = list(self.dess_buses)
        next_obs, team, terminated, truncated, info = super().step(action)
        info['marl_a0_team_reward'] = np.asarray(team, dtype=float).copy()
        if info.get('episode_failed', False):
            return next_obs, team, terminated, truncated, info
        if not np.isclose(info['price'], price, rtol=0, atol=1e-9):
            raise RuntimeError('A3 reward row differs from the executed step')
        completed = bool(info['marl_terminal_completed'])
        soc_end = np.array([b.get_soc() for b in self.batteries], dtype=float)
        shortfall = terminal_shortfall(self.start_of_day_soc, soc_end, [b.config.soc_min for b in self.batteries])
        net_load = load_kw[buses] - pv_kw[buses]
        voltage = np.asarray(info['final_voltage_pu'], float)[buses]
        own_value = info['marl_terminal_energy_value_usd'] if self.protocol is not None else None
        terms = local_reward_terms(self.reward_cfg, price, self.delta_t_hours, self.v_ref, net_load,
                                   info['final_battery_power_kw'], info['physical_limited_power_kw'], voltage,
                                   info['reward_term_local'], shortfall, self.terminal_coefficients, completed,
                                   terminal_value_usd=own_value)
        rewards = np.sum(list(terms.values()), axis=0).astype(np.float32)
        # A3 decomposition replaces the team decomposition; the team terms stay available under reference_a0_*.
        for key in [k for k in info if k.startswith('reward_term_')]:
            info['reference_a0_' + key] = info.pop(key)
        info['reference_a0_reward_terms_total'] = info.pop('reward_terms_total', None)
        for name, value in terms.items():
            info['reward_term_a3_' + name] = value
        info['reward_terms_total'] = rewards.astype(float)
        info.update(marl_reward_definition=REWARD_DEFINITION, marl_own_bus_net_load_kw=net_load,
                    marl_own_bus_cost_change=own_bus_cost_change(price, net_load, info['final_battery_power_kw'],
                                                                 self.delta_t_hours),
                    marl_terminal_coefficients=self.terminal_coefficients if self.protocol is None else None,
                    marl_terminal_penalty=float(terms['terminal_soc'].sum()) if completed else 0.0)
        return next_obs, rewards, terminated, truncated, info


def make_a3_env(config, mode, seed, lambda_terminal, scale=1.0):
    from environments.marl_env import make_marl_env
    return make_marl_env(config, mode, seed, lambda_terminal, scale, env_class=A3DESSEnv)
