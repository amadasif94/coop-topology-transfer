"""
Distributed energy storage coordination environment for graph-aware
cooperative multi-agent reinforcement learning.

This environment combines:

    - DESS battery dynamics,
    - static distribution-network data from the provided config,
    - SMART-DS / NYISO node-level 15-minute time-series data,
    - tensor-based fixed-point power-flow evaluation,
    - graph construction for GNN-based policies,
    - normalized neural-network observations,
    - hard feasibility handling for operational constraints,
    - cooperative reward calculation.

Action convention:
    action > 0  : DESS discharge / active-power injection
    action < 0  : DESS charge / active-power absorption
    action = 0  : idle

Important design choice:
    This environment does NOT run the GNN encoder. It only returns:

        x, edge_index, edge_attr, agent_obs

    The GNN encoder should be used later inside the model/training stack:

        H_t = GNNEncoder(x, edge_index, edge_attr)
        h_i(t) = H_t[dess_bus_i]
        graph_aware_agent_obs_i = concat(agent_obs_i, h_i(t))

All accepted environment transitions are evaluated through the power-flow
model. Under hard feasibility handling, voltage and line-current limits are
checked before the transition is accepted. The original zero-action fallback
is preserved; it is not guaranteed to restore feasibility.

Pre-correction diagnostics:
    info["requested_*"] describes the candidate entering the AC backup,
    AFTER battery SOC/power clipping and, if enabled, the local DOE.
    info["marl_*"] identifies the policy proposal separately. With DOE disabled,
    the requested_* fields retain their original policy-candidate meaning.
    All diagnostics are read-only and do not alter physical dispatch, rewards,
    observations, or correction behavior.

Correction / penalty convention:
    raw_infeasible_requested:
        True when the candidate entering the hard backup was not feasible.

    minor_corrected_action:
        True when the original candidate action became feasible after a small
        number of correction iterations. In directional_incremental mode this
        is a diagnostic only and does not affect the penalty.

    major_infeasible_action:
        Legacy field. In directional_incremental mode "major" means "backup
        activated", independent of correction size. It can coexist with minor.
        Use backup_penalty_applied to determine whether reward was penalized.

    Penalty rule (directional_incremental):
        [-1, 1] clipping and battery power/SOC limits: never penalized
        (logged as physical_clipped / physical_clipped_kw).
        DOE: optional -doe_penalty_coef * |physical - doe| / rated per agent
        (default coefficient 0.0).
        AC backup: existing infeasible_action_penalty on every activation.

    info["infeasible_action"]:
        Kept for backward compatibility with existing evaluation scripts.
        It is now equal to major_infeasible_action, not raw_infeasible_requested.

Reward convention:
    r_i(t) = w_grid  * R_grid(t)
           + w_local * R_local,i(t)
           + w_kpi   * R_KPI(t)

where w_grid, w_local, and w_kpi are defined in the provided config.
"""

import copy
import time

import numpy as np
from environments.directional_backup import backup_attribution

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover
    import gym
    from gym import spaces

from environments.battery import Battery, BatteryConfig
from environments.voltage_doe import apply_voltage_doe, validate_parameters, validate_doe_config
from environments.directional_backup import settings, valid_pf, violation_metrics, search, legacy_diagnostics
from utility.feasibility import VOLTAGE_TOLERANCE_PU, current_overloaded, voltage_excess
from utility.data_loader import (
    load_all_data,
    select_time_series_row,
)
from utility.grid import build_grid_from_static_data
from utility.graph_builder import build_graph_builder
from utility.power_flow import build_power_flow_solver
from utility.normalization import ObservationNormalizer
from utility.obs_interface import AGENT_OBS_DIM, AGENT_OBS_FEATURES


class DESSEnv(gym.Env):
    """
    Graph-ready cooperative MARL environment for DESS coordination.

    The environment returns graph-structured observations but does not apply
    any neural network internally. This keeps simulation, feasibility handling,
    and learning-model code cleanly separated.
    """

    metadata = {"render_modes": []}

    def __init__(self, config=None, mode="train", seed=None):
        super(DESSEnv, self).__init__()

        if config is None:
            raise ValueError(
                "DESSEnv requires a configuration dictionary. "
                "Pass IEEE33_CONFIG, IEEE69_CONFIG, or another compatible config."
            )

        self.config = copy.deepcopy(config)
        doe = self.config.get("doe", {})
        validate_doe_config(doe, config["network"]["v_min"], config["network"]["v_max"])
        self.doe_enabled = bool(doe.get("doe_enabled", False))
        self.doe_w = float(doe.get("doe_w", 0.015))
        self.doe_delta_c = float(doe.get("doe_delta_c", 0.010))
        if self.doe_enabled:
            validate_parameters(self.doe_w, self.doe_delta_c,
                                config["network"]["v_min"], config["network"]["v_max"])
        # Optional per-agent penalty on the DOE's own change, measured after the
        # physical power/SOC limits: -coef * |physical - doe| / rated. Off at 0.
        self.doe_penalty_coef = float(doe.get("doe_penalty_coef", 0.0))
        if not np.isfinite(self.doe_penalty_coef) or self.doe_penalty_coef < 0:
            raise ValueError("doe_penalty_coef must be finite and nonnegative")
        self.backup_cfg = settings(self.config.get("backup", {}))
        self.backup_mode = self.backup_cfg["backup_mode"]
        self.doe_pf_failure_mode = doe.get("pf_failure_mode", "strict")
        if self.doe_pf_failure_mode not in ("strict", "logged_fallback"):
            raise ValueError("Unknown DOE pf_failure_mode")
        self._robust_pf = (self.backup_mode == "directional_incremental" or
                           (self.doe_enabled and self.doe_pf_failure_mode == "logged_fallback"))
        self.mode = mode

        if self.mode not in ["train", "val", "test", "all"]:
            raise ValueError("mode must be one of: train, val, test, all.")
        # F2: the December test split ('test', and 'all' which contains it) is reachable only
        # through an explicit final-test authorization (utility.evaluation_protocol).
        if self.mode in ("test", "all") and (self.config.get("evaluation_protocol") or {}).get("final_test") is not True:
            raise ValueError(f"mode={self.mode!r} includes the December test split and requires explicit "
                             "final-test authorization (evaluator --final); development runs use train/val")

        self.rng = np.random.default_rng(seed)

        # F8 instrumentation only (timing, optional evaluation diagnostics). Never read by
        # the dynamics, rewards or control path. evaluation_diagnostics enables the optional
        # pre-DOE proposal PF; it is off by default (training).
        self.evaluation_diagnostics = False
        self._pf_time_s = 0.0
        self._pf_solves = 0
        self._step_timing = {}

        # ------------------------------------------------------------
        # Core network and time settings
        # ------------------------------------------------------------
        self.num_buses = int(self.config["network"]["num_buses"])
        self.slack_bus = int(self.config["network"]["slack_bus"])

        self.v_ref = float(self.config["network"]["v_ref"])
        self.v_min = float(self.config["network"]["v_min"])
        self.v_max = float(self.config["network"]["v_max"])

        self.grid_export_limit_kw = float(
            self.config["network"].get("grid_export_limit_kw", 0.0)
        )

        self.episode_length = int(self.config["time"]["episode_length"])
        self.delta_t_hours = float(self.config["time"]["delta_t_hours"])

        # ------------------------------------------------------------
        # Reward and constraint settings
        # ------------------------------------------------------------
        self.reward_cfg = self.config["reward"]
        self.constraint_cfg = self.config["constraints"]

        self.constraint_handling = self.constraint_cfg.get(
            "handling",
            "hard_feasibility",
        )
        if self.backup_mode == "directional_incremental" and self.constraint_handling != "hard_feasibility":
            raise ValueError("directional_incremental requires hard_feasibility handling")
        self.hard_constraint_action = self.constraint_cfg.get(
            "hard_constraint_action",
            "clip_or_correct",
        )
        self.max_correction_attempts = int(
            self.constraint_cfg.get("max_correction_attempts", 20)
        )
        self.action_correction_factor = float(
            self.constraint_cfg.get("action_correction_factor", 0.90)
        )

        # A correction that succeeds within this many iterations is treated
        # as a minor boundary correction. By default, minor corrections are
        # logged but not penalized.
        self.minor_correction_max_iterations = int(
            self.constraint_cfg.get("minor_correction_max_iterations", 1)
        )
        self.penalize_minor_corrections = bool(
            self.constraint_cfg.get("penalize_minor_corrections", False)
        )

        # ------------------------------------------------------------
        # Data scaling
        # ------------------------------------------------------------
        data_scaling = self.config.get("data_scaling", {})
        self.load_scaling_factor = float(data_scaling.get("load_scaling_factor", 1.0))
        self.pv_scaling_factor = float(data_scaling.get("pv_scaling_factor", 1.0))
        self.price_scaling_factor = float(data_scaling.get("price_scaling_factor", 1.0))

        self.normalize_observations = bool(
            self.config.get("normalization", {}).get("normalize_observations", True)
        )

        # ------------------------------------------------------------
        # Data, grid, graph builder, and power flow solver
        # ------------------------------------------------------------
        all_data = load_all_data(
            self.config,
            require_time_series=True,
        )

        self.static_data = all_data["static"]
        self.time_series = all_data["processed_time_series"]
        self.splits = all_data["splits"]

        self.grid = build_grid_from_static_data(
            self.static_data,
            self.config,
        )

        self.normalizer = self._build_normalizer()

        self.graph_builder = build_graph_builder(
            self.grid,
            self.config,
        )

        self.power_flow = build_power_flow_solver(
            self.grid,
            self.config,
        )

        self.dess_buses = list(self.grid.get_dess_buses())
        self.num_agents = len(self.dess_buses)

        self._initialize_line_current_limits()
        self._build_batteries()

        self.episode_start_indices = self._build_episode_start_indices()

        if len(self.episode_start_indices) == 0:
            raise ValueError(
                "No valid episode start indices were found for mode '{}'.".format(
                    self.mode
                )
            )

        # ------------------------------------------------------------
        # Action and observation spaces
        # ------------------------------------------------------------
        self.action_space = spaces.Box(
            low=float(self.config["action"]["low"]),
            high=float(self.config["action"]["high"]),
            shape=(self.num_agents,),
            dtype=np.float32,
        )

        # Base local observation only. GNN embeddings are NOT included here.
        # Later, in the model stack:
        #     graph_aware_obs_i = concat(agent_obs_i, h_i)
        # Feature order: utility.obs_interface.AGENT_OBS_FEATURES (10; grid_stress
        # is not an actor input, it remains in the KPI reward).
        self.agent_obs_features = AGENT_OBS_FEATURES
        self.agent_obs_dim = AGENT_OBS_DIM

        self.observation_space = spaces.Dict(
            {
                "x": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(
                        self.num_buses,
                        int(self.config["graph"]["node_feature_dim"]),
                    ),
                    dtype=np.float32,
                ),
                "edge_index": spaces.Box(
                    low=0,
                    high=self.num_buses - 1,
                    shape=self.grid.edge_index.shape,
                    dtype=np.int64,
                ),
                "edge_attr": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=self.grid.edge_attr.shape,
                    dtype=np.float32,
                ),
                "agent_obs": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.num_agents, self.agent_obs_dim),
                    dtype=np.float32,
                ),
            }
        )

        self.reset(seed=seed)

    # ============================================================
    # Normalization utilities
    # ============================================================

    def _build_normalizer(self):
        """
        Build observation normalizer from config.

        If selected values are None in config, infer safe defaults from
        static grid data.
        """

        norm_cfg = self.config.get("normalization", {})

        total_base_load_kw = max(float(np.sum(self.grid.get_base_load_kw())), 1.0)
        total_base_pv_kw = max(float(np.sum(self.grid.get_base_pv_kw())), 1.0)

        max_time_index = norm_cfg.get(
            "max_time_index",
            float(self.episode_length - 1),
        )
        max_price = norm_cfg.get("max_price", 300.0)

        max_load_kw = norm_cfg.get("max_load_kw", None)
        if max_load_kw is None:
            max_load_kw = total_base_load_kw

        # F3: PV and curtailment references must be explicit. The former fallback (sum of
        # bus_data base PV = 0 -> 1 kW floor) silently left PV and curtailment un-normalized.
        max_pv_kw = norm_cfg.get("max_pv_kw", None)
        if max_pv_kw is None:
            raise ValueError("normalization.max_pv_kw must be configured explicitly (F3)")
        max_curtailment_kw = norm_cfg.get("max_curtailment_kw", None)
        if max_curtailment_kw is None:
            raise ValueError("normalization.max_curtailment_kw must be configured explicitly (F3)")

        max_grid_import_kw = norm_cfg.get("max_grid_import_kw", None)
        if max_grid_import_kw is None:
            max_grid_import_kw = total_base_load_kw

        max_voltage_deviation = norm_cfg.get(
            "max_voltage_deviation",
            float(self.num_buses) * 0.10,
        )
        max_line_current_pu = norm_cfg.get("max_line_current_pu", 5.0)

        return ObservationNormalizer(
            max_time_index=float(max_time_index),
            max_price=float(max_price),
            max_load_kw=max(float(max_load_kw), 1.0),
            max_pv_kw=max(float(max_pv_kw), 1.0),
            max_grid_import_kw=max(float(max_grid_import_kw), 1.0),
            max_voltage_deviation=max(float(max_voltage_deviation), 1e-6),
            max_line_current_pu=max(float(max_line_current_pu), 1e-6),
            max_curtailment_kw=max(float(max_curtailment_kw), 1.0),
        )

    # ============================================================
    # Initialization utilities
    # ============================================================

    def _initialize_line_current_limits(self):
        """
        Initialize or calibrate line-current limits.
        """

        mode = self.config["network"].get(
            "line_current_limit_mode",
            "baseline_margin",
        )

        if self.grid.line_current_limits_pu is not None:
            return

        if mode != "baseline_margin":
            return

        candidate_indices = self.splits["train_indices"]

        if len(candidate_indices) == 0:
            raise ValueError(
                "Cannot calibrate line-current limits because the training split "
                "contains no indices."
            )

        rho = float(
            self.config["network"].get("line_current_security_margin_rho", 1.20)
        )

        self._calibrate_line_current_limits_from_scaled_baseline(
            indices=candidate_indices,
            rho=rho,
        )

    def _calibrate_line_current_limits_from_scaled_baseline(self, indices, rho):
        """
        Calibrate line-current limits from no-control baseline operation.

        I_line,max = rho * max_t I_line,baseline(t)
        """

        if rho <= 1.0:
            raise ValueError("rho must be greater than 1.0.")

        max_currents = None

        for idx in indices:
            load_kw, load_kvar, pv_kw, _, _ = self._get_profiles(idx)

            result = self.power_flow.run_power_flow(
                load_kw=load_kw,
                load_kvar=load_kvar,
                pv_kw=pv_kw,
                dess_power_kw=np.zeros(self.num_buses, dtype=np.float64),
            )

            currents = result["line_current_pu"].astype(np.float64)

            if max_currents is None:
                max_currents = currents.copy()
            else:
                max_currents = np.maximum(max_currents, currents)

        limits = rho * max_currents
        self.grid.set_line_current_limits_pu(limits)

    def _build_batteries(self):
        """
        Create one DESS battery model for each DESS bus.
        """

        dess_cfg = self.config["dess"]

        battery_config = BatteryConfig(
            capacity_kwh=dess_cfg["capacity_kwh"],
            max_charge_kw=dess_cfg["max_charge_kw"],
            max_discharge_kw=dess_cfg["max_discharge_kw"],
            efficiency=dess_cfg.get("efficiency", 0.95),
            charge_efficiency=dess_cfg.get("charge_efficiency", 0.95),
            discharge_efficiency=dess_cfg.get("discharge_efficiency", 0.95),
            soc_min=dess_cfg["soc_min"],
            soc_max=dess_cfg["soc_max"],
            soc_init=dess_cfg["soc_init"],
        )

        self.batteries = [
            Battery(battery_config, delta_t_hours=self.delta_t_hours)
            for _ in range(self.num_agents)
        ]

    # ============================================================
    # Episode indexing
    # ============================================================

    def _build_episode_start_indices(self):
        """
        Build valid episode-start indices for the selected data split.

        For 15-minute data, an episode starts exactly at 00:00.
        """

        df = self.time_series["df"]
        datetime_col = self.time_series["datetime_col"]

        if self.mode == "train":
            candidate_indices = self.splits["train_indices"]
        elif self.mode == "val":
            candidate_indices = self.splits["val_indices"]
        elif self.mode == "test":
            candidate_indices = self.splits["test_indices"]
        else:
            candidate_indices = list(range(len(df)))

        candidate_set = set(candidate_indices)
        starts = []

        for idx in candidate_indices:
            if idx + self.episode_length - 1 >= len(df):
                continue

            dt = df.loc[idx, datetime_col]

            if dt.hour != 0 or dt.minute != 0:
                continue

            episode_indices = range(idx, idx + self.episode_length)

            if all(j in candidate_set for j in episode_indices):
                starts.append(idx)

        return starts

    # ============================================================
    # Reset
    # ============================================================

    def reset(self, seed=None, options=None):
        """
        Reset the environment at the beginning of an episode.
        """

        del options

        try:
            super().reset(seed=seed)
        except TypeError:
            pass

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.episode_start_index = int(self.rng.choice(self.episode_start_indices))
        self.current_step = 0
        self.failed_episode_count = getattr(self, "failed_episode_count", 0)
        self._episode_failure_reported = False
        self.pf_failure_count = 0
        self.pf_failed_solve_count = 0
        self.pf_failure_reason = ""
        self._failed_pf_indices = set()
        self._reset_pf_failed = False
        self.current_index = self.episode_start_index

        for battery in self.batteries:
            battery.reset()

        self.previous_actions = np.zeros(self.num_agents, dtype=np.float32)
        self.last_dess_power_kw = np.zeros(self.num_buses, dtype=np.float64)

        self.previous_kpis = None
        self.current_kpis = None

        load_kw, load_kvar, pv_kw, price, date_time = self._get_profiles(
            self.current_index
        )

        pv_used_kw, curtailment_by_bus, curtailment_kw = self._apply_curtailment(
            load_kw=load_kw,
            pv_kw=pv_kw,
            dess_power_kw=self.last_dess_power_kw,
        )

        pf_result = self._run_pf(
            load_kw=load_kw,
            load_kvar=load_kvar,
            pv_kw=pv_used_kw,
            dess_power_kw=self.last_dess_power_kw,
        )

        self.last_pf_result = pf_result
        self.last_curtailment_by_bus = curtailment_by_bus.copy()
        self.last_curtailment_kw = float(curtailment_kw)

        self.current_kpis = self._compute_kpis(
            pf_result=pf_result,
            price=price,
            curtailment_kw=curtailment_kw,
        )
        self.previous_kpis = dict(self.current_kpis)
        self.last_date_time = date_time

        if not valid_pf(pf_result):
            if self.doe_enabled and self.doe_pf_failure_mode == "strict":
                raise ValueError("Invalid/nonconverged reset power flow")
            self._record_pf_failure(pf_result)
            self._reset_pf_failed = True
            obs = self._failure_observation(load_kw, pv_used_kw, price)
            info = self._build_info()
            info.update(episode_failed=True, observation_valid=False,
                        transition_valid=False, pf_failure_stage="reset",
                        terminated=False, truncated=True,
                        feasible=False, converged=False,
                        observation_pf_evaluations=1,
                        pf_failure_count=self.pf_failure_count,
                        pf_failed_solve_count=self.pf_failed_solve_count,
                        pf_failure_reason=self.pf_failure_reason)
            info.update(self._mark_episode_failed("reset"))
            return obs, info

        obs = self._build_observation(
            load_kw=load_kw,
            pv_kw=pv_used_kw,
            price=price,
            voltage_pu=pf_result["voltage_pu"],
        )

        info = self._build_info()
        info["observation_pf_evaluations"] = 1
        return obs, info

    # ============================================================
    # Observation construction
    # ============================================================

    def _build_observation(self, load_kw, pv_kw, price, voltage_pu):
        """
        Build graph-level and agent-level observations.
        """

        soc_values = np.array(
            [battery.get_soc() for battery in self.batteries],
            dtype=np.float32,
        )

        soc_by_bus = self.graph_builder.build_soc_vector(soc_values)

        graph = self.graph_builder.build_graph(
            time_index=self.current_step,
            price=price,
            load_kw=load_kw,
            pv_kw=pv_kw,
            soc_by_bus=soc_by_bus,
            voltage_pu=voltage_pu,
        )

        agent_obs = self._build_agent_observation(
            load_kw=load_kw,
            pv_kw=pv_kw,
            price=price,
            voltage_pu=voltage_pu,
            soc_values=soc_values,
        )

        return {
            "x": graph["x"].astype(np.float32),
            "edge_index": graph["edge_index"].astype(np.int64),
            "edge_attr": graph["edge_attr"].astype(np.float32),
            "agent_obs": agent_obs.astype(np.float32),
        }

    def _build_agent_observation(self, load_kw, pv_kw, price, voltage_pu, soc_values):
        """
        Build local per-agent observations.
        """

        agent_obs = np.zeros((self.num_agents, self.agent_obs_dim), dtype=np.float32)

        # grid_stress (max voltage + max line-current violation) is deliberately
        # NOT an actor input; it is used only by the KPI reward.
        grid_import = float(self.current_kpis["grid_import_kw"])
        voltage_deviation = float(self.current_kpis["voltage_deviation"])
        curtailment = float(self.current_kpis["curtailment_kw"])

        if self.normalize_observations:
            time_value = float(self.normalizer.normalize_time_index(self.current_step))
            price_value = float(self.normalizer.normalize_price(price))
            grid_import_value = float(
                self.normalizer.normalize_grid_import_kw(grid_import)
            )
            voltage_deviation_value = float(
                self.normalizer.normalize_voltage_deviation(voltage_deviation)
            )
            # F3: feeder-wide curtailment uses its own feeder-level scale (was the per-bus PV scale).
            curtailment_value = float(self.normalizer.normalize_curtailment_kw(curtailment))
        else:
            time_value = float(self.current_step)
            price_value = float(price)
            grid_import_value = grid_import
            voltage_deviation_value = voltage_deviation
            curtailment_value = curtailment

        for agent_idx, bus in enumerate(self.dess_buses):
            if self.normalize_observations:
                load_value = float(self.normalizer.normalize_load_kw(load_kw[bus]))
                pv_value = float(self.normalizer.normalize_pv_kw(pv_kw[bus]))
                voltage_value = float(
                    self.normalizer.normalize_voltage_pu(voltage_pu[bus])
                )
                soc_value = float(self.normalizer.normalize_soc(soc_values[agent_idx]))
            else:
                load_value = float(load_kw[bus])
                pv_value = float(pv_kw[bus])
                voltage_value = float(voltage_pu[bus])
                soc_value = float(soc_values[agent_idx])

            agent_obs[agent_idx, :] = np.array(
                [
                    time_value,
                    price_value,
                    soc_value,
                    voltage_value,
                    load_value,
                    pv_value,
                    float(self.previous_actions[agent_idx]),
                    grid_import_value,
                    voltage_deviation_value,
                    curtailment_value,
                ],
                dtype=np.float32,
            )

        if not np.all(np.isfinite(agent_obs)):
            raise ValueError("Agent observation contains non-finite values.")

        return agent_obs

    # ============================================================
    # Environment transition
    # ============================================================

    def step(self, action):
        """
        Apply DESS actions and advance the environment by one step.

        F8 timing wrapper around _step_impl (instrumentation only). Non-overlapping fields,
        milliseconds, all measured inside env.step:
          time_doe_ms              battery-limit preview + DOE envelope (no PF inside)
          time_backup_search_ms    directional search bookkeeping EXCLUDING its PF solves
          time_pf_ms               every control-path AC PF solve this step (initial
                                   candidate, backup trials, next-observation PF)
          time_diagnostic_pf_ms    optional pre-DOE proposal diagnostic PF (evaluation only)
          time_env_other_ms        remainder of env.step
          time_env_step_ms         total = sum of the five above
        Actor inference is timed by the caller (evaluators: time_policy_ms); optimizer
        update time is reported by MADDPG.update (update_time_ms).
        """
        t0, pf_t0, pf_n0 = time.perf_counter(), self._pf_time_s, self._pf_solves
        self._step_timing = dict(doe=0.0, backup_total=0.0, backup_pf=0.0, diagnostic_pf=0.0)
        output = self._step_impl(action)
        info = output[-1]
        st = self._step_timing
        total = time.perf_counter() - t0
        pf = self._pf_time_s - pf_t0
        backup_excl = max(st["backup_total"] - st["backup_pf"], 0.0)
        info.update(time_env_step_ms=1000 * total, time_doe_ms=1000 * st["doe"],
                    time_backup_search_ms=1000 * backup_excl, time_pf_ms=1000 * pf,
                    time_diagnostic_pf_ms=1000 * st["diagnostic_pf"],
                    time_env_other_ms=1000 * max(total - st["doe"] - backup_excl - pf - st["diagnostic_pf"], 0.0),
                    pf_solves_in_step=int(self._pf_solves - pf_n0))
        return output

    def _step_impl(self, action):
        marl_raw_action = np.asarray(action).copy()
        action = self._prepare_action(action)
        marl_action = action.copy()
        # reset() and the previous step's time advance already solved this
        # timestep with previous executed battery powers held. No extra PF.
        pre_voltage = np.asarray(self.last_pf_result["voltage_pu"], dtype=float)[self.dess_buses].copy()
        pre_converged = bool(self.last_pf_result["converged"])
        failed_solves_before = self.pf_failed_solve_count
        pre_failure = not valid_pf(self.last_pf_result)
        if pre_failure:
            self._record_pf_failure(self.last_pf_result)
        doe_skipped = bool(self.doe_enabled and pre_failure and self.doe_pf_failure_mode == "logged_fallback")
        doe_diagnostics = {}
        physical_power = None
        doe_power = None
        if self.doe_enabled:
            if pre_failure and self.doe_pf_failure_mode == "strict":
                raise ValueError("DOE requires a finite, converged pre-action power flow.")
            t_doe = time.perf_counter()  # F8 timing
            # Preview the existing battery physics only, without a PF or
            # permanent SOC transition. Restore even if validation fails.
            saved = self._snapshot_battery_states()
            charge = np.array([b.get_available_charge_kw() for b in self.batteries])
            discharge = np.array([b.get_available_discharge_kw() for b in self.batteries])
            cr = np.array([b.config.max_charge_kw for b in self.batteries])
            dr = np.array([b.config.max_discharge_kw for b in self.batteries])
            try:
                physical_power = np.array([b.step(a)["power_kw"] for b, a in zip(self.batteries, action)])
            finally:
                self._restore_battery_states(saved)
            if doe_skipped:
                doe_power = physical_power.copy()
            else:
                doe_power, doe_diagnostics = apply_voltage_doe(
                    physical_power, pre_voltage, cr, dr, charge, discharge,
                    w=self.doe_w, delta_c=self.doe_delta_c,
                    v_min=self.v_min, v_max=self.v_max, pf_converged=pre_converged)
            ratings = np.where(doe_power >= 0, dr, cr)
            action = np.divide(doe_power, ratings, out=np.zeros_like(doe_power),
                               where=ratings > 0)
            self._step_timing["doe"] = time.perf_counter() - t_doe

        load_kw, load_kvar, pv_kw, price, date_time = self._get_profiles(
            self.current_index
        )

        soc_before = np.array(
            [battery.get_soc() for battery in self.batteries],
            dtype=np.float32,
        )

        candidate = self._evaluate_candidate_action(
            action=action,
            load_kw=load_kw,
            load_kvar=load_kvar,
            pv_kw=pv_kw,
        )

        # Save the ORIGINAL candidate before the feasibility-correction loop.
        # The candidate carries its own PF result and battery outputs, while
        # _evaluate_candidate_action has already restored the battery states.
        requested_candidate = candidate

        raw_infeasible_requested = False
        minor_corrected_action = False
        major_infeasible_action = False

        backup_diagnostics = {}
        failed_dispatch = False
        # F8.1: optional evaluation-only PF of the physically limited, pre-DOE proposal.
        proposal_diag = self._physical_proposal_diagnostic(
            physical_power, doe_power, doe_skipped, requested_candidate, load_kw, load_kvar, pv_kw)

        if self.backup_mode == "directional_incremental":
            initial_power = candidate["dess_power_kw"][self.dess_buses].copy()
            t_backup, pf_backup0 = time.perf_counter(), self._pf_time_s  # F8 timing
            candidate, backup_diagnostics = search(
                candidate, initial_power,
                [b.get_available_charge_kw() for b in self.batteries],
                [b.get_available_discharge_kw() for b in self.batteries],
                lambda power: self._evaluate_candidate_action(
                    self._power_to_action(power), load_kw, load_kvar, pv_kw),
                self._measure_violation, self.backup_cfg)
            self._step_timing["backup_total"] = time.perf_counter() - t_backup
            self._step_timing["backup_pf"] = self._pf_time_s - pf_backup0
            raw_infeasible_requested = not self._measure_violation(requested_candidate["pf_result"])["feasible"]
            if candidate is not None and raw_infeasible_requested:
                # Diagnostic only: a small feasible correction. The penalty rule
                # does not depend on move size; every activation is penalized.
                minor_corrected_action = bool(candidate["feasible"] and
                    not candidate["correction_fallback_to_zero"] and
                    candidate["correction_iterations"] <= self.minor_correction_max_iterations)
                major_infeasible_action = True
            elif candidate is None:
                failed_dispatch = True
                major_infeasible_action = True
                candidate = dict(requested_candidate)
        elif self.constraint_handling == "hard_feasibility":
            if not candidate["feasible"]:
                raw_infeasible_requested = True

                candidate = self._handle_infeasible_action(
                    original_action=action,
                    load_kw=load_kw,
                    load_kvar=load_kvar,
                    pv_kw=pv_kw,
                )

                correction_iterations = int(candidate.get("correction_iterations", 0))
                fallback_to_zero = bool(
                    candidate.get("correction_fallback_to_zero", False)
                )

                if (
                    candidate["feasible"]
                    and not fallback_to_zero
                    and correction_iterations <= self.minor_correction_max_iterations
                ):
                    minor_corrected_action = True
                    major_infeasible_action = bool(self.penalize_minor_corrections)
                else:
                    major_infeasible_action = True

        else:
            candidate["accepted_action"] = action.copy()
            candidate["correction_applied"] = False
            candidate["correction_iterations"] = 0
            candidate["correction_fallback_to_zero"] = False

        failed_dispatch = bool(failed_dispatch or not valid_pf(candidate["pf_result"]))
        if self._reset_pf_failed:
            # reset has no terminal flag; the first step exposes terminal failure.
            failed_dispatch = True
        if failed_dispatch:
            self._record_pf_failure(candidate["pf_result"])
            candidate = dict(candidate)
            if self.backup_mode == "directional_incremental":
                candidate["correction_applied"] = bool(raw_infeasible_requested)
            candidate["feasible"] = False
            candidate["pf_result"] = dict(candidate["pf_result"], converged=False, feasible=False)
            candidate["dess_power_kw"] = np.zeros(self.num_buses)
            candidate["accepted_action"] = np.zeros(self.num_agents)
            candidate["battery_outputs"] = [dict(b, power_kw=0.0, energy_kwh=0.0,
                soc=self.batteries[i].get_soc()) for i, b in enumerate(candidate["battery_outputs"])]
        else:
            # Commit exactly one accepted battery transition.
            self._restore_battery_states(candidate["battery_states_after"])

        accepted_action = candidate["accepted_action"]
        dess_power_kw = candidate["dess_power_kw"]
        battery_outputs = candidate["battery_outputs"]
        pv_used_kw = candidate["pv_used_kw"]
        curtailment_by_bus = candidate["curtailment_by_bus"]
        curtailment_kw = candidate["curtailment_kw"]
        pf_result = candidate["pf_result"]

        self.previous_kpis = dict(self.current_kpis)

        self.last_pf_result = pf_result
        self.last_dess_power_kw = dess_power_kw.copy()
        self.last_curtailment_by_bus = curtailment_by_bus.copy()
        self.last_curtailment_kw = float(curtailment_kw)

        self.current_kpis = self._compute_kpis(
            pf_result=pf_result,
            price=price,
            curtailment_kw=curtailment_kw,
        )

        if pre_failure:
            # No KPI improvement can be inferred from an invalid measurement.
            self.previous_kpis = dict(self.current_kpis)

        soc_after = np.array(
            [battery.get_soc() for battery in self.batteries],
            dtype=np.float32,
        )

        # Preserve the existing nonconvergence early-return semantics.
        backup_penalty_applied = bool(major_infeasible_action and pf_result["converged"])
        rewards = self._calculate_rewards(
            price=price,
            dess_power_kw=dess_power_kw,
            soc_before=soc_before,
            soc_after=soc_after,
            infeasible_action=major_infeasible_action,  # legacy call compatibility
            backup_penalty_applied=backup_penalty_applied,
        )

        if self.backup_mode == "directional_incremental" and not failed_dispatch:
            rewards = rewards + float(backup_diagnostics["backup_residual_penalty"])

        # DOE's own change only: physically limited power -> DOE power.
        rated_kw = np.array([max(b.config.max_charge_kw, b.config.max_discharge_kw)
                             for b in self.batteries])
        if self.doe_enabled:
            doe_change_kw = np.abs(physical_power - doe_power)
        else:
            doe_change_kw = np.zeros(self.num_agents)
        doe_change_normalized = np.divide(doe_change_kw, rated_kw, out=np.zeros(self.num_agents),
                                          where=rated_kw > 0)
        doe_penalty = -self.doe_penalty_coef * doe_change_normalized
        if not failed_dispatch:
            rewards = (rewards + doe_penalty).astype(np.float32)
        else:
            doe_penalty = np.zeros(self.num_agents)

        self.previous_actions = accepted_action.copy()
        self.last_date_time = date_time

        terminated = False
        truncated = failed_dispatch or self.current_step >= self.episode_length - 1

        info = self._build_info()
        info["battery_outputs"] = battery_outputs
        info["final_voltage_pu"] = np.asarray(pf_result.get("voltage_pu_raw", pf_result["voltage_pu"])).copy()
        info["final_line_current_pu"] = np.asarray(pf_result.get("line_current_pu_raw", pf_result["line_current_pu"])).copy()
        info["dess_power_kw"] = dess_power_kw.copy()
        info["curtailment_by_bus_kw"] = curtailment_by_bus.copy()
        info["requested_action"] = action.copy()
        info["accepted_action"] = accepted_action.copy()

        # Diagnostic ONLY: candidate before the action-correction loop.
        # Crucially, do not run a second PF or battery transition here.
        requested_pf = requested_candidate["pf_result"]
        requested_batteries = requested_candidate["battery_outputs"]
        requested_voltages = np.asarray(requested_pf["voltage_pu"], dtype=np.float64)

        info["requested_feasible"] = bool(requested_candidate["feasible"])
        info["requested_pf_converged"] = bool(requested_pf["converged"])
        info["requested_min_voltage_pu"] = float(np.min(requested_voltages))
        info["requested_max_voltage_pu"] = float(np.max(requested_voltages))
        info["requested_max_voltage_violation"] = float(
            requested_pf["max_voltage_violation"]
        )
        info["requested_max_line_current_violation"] = float(
            requested_pf["max_line_current_violation"]
        )
        info["requested_line_current_pu"] = np.asarray(
            requested_pf["line_current_pu"],
            dtype=np.float64,
        ).copy()

        info["requested_line_current_pu_raw"] = np.asarray(
            requested_pf.get("line_current_pu_raw", requested_pf["line_current_pu"]), dtype=np.float64).copy()

        # Battery.step reports requested kW and actual kW after limiting by
        # available SOC/energy. The action passed to Battery.step has already
        # been clipped to the normalized action bounds in _prepare_action.
        info["requested_battery_limited_count"] = int(
            sum(bool(b["was_limited"]) for b in requested_batteries)
        )
        info["requested_battery_clipping_kw"] = float(
            sum(
                abs(float(b["requested_power_kw"]) - float(b["power_kw"]))
                for b in requested_batteries
            )
        )

        # Optional agent-level detail (not automatically serialized by the
        # topology-evaluation scripts unless explicitly added there).
        info["requested_battery_requested_power_kw"] = np.array(
            [float(b["requested_power_kw"]) for b in requested_batteries],
            dtype=np.float64,
        )
        info["requested_battery_actual_power_kw"] = np.array(
            [float(b["power_kw"]) for b in requested_batteries],
            dtype=np.float64,
        )
        info["requested_battery_was_limited"] = np.array(
            [bool(b["was_limited"]) for b in requested_batteries],
            dtype=bool,
        )

        # Backward compatible field:
        # Existing scripts summarize "infeasible_action"; now this means
        # major/penalized infeasibility, not minor one-step correction.
        info["infeasible_action"] = bool(major_infeasible_action)

        # New diagnostic fields:
        info["raw_infeasible_requested"] = bool(raw_infeasible_requested)
        info["minor_corrected_action"] = bool(minor_corrected_action)
        info["major_infeasible_action"] = bool(major_infeasible_action)
        info["correction_applied"] = bool(candidate.get("correction_applied", False))
        info["correction_iterations"] = int(candidate.get("correction_iterations", 0))
        info["correction_fallback_to_zero"] = bool(
            candidate.get("correction_fallback_to_zero", False)
        )

        info["price"] = float(price)
        info["date_time"] = date_time
        info["dess_buses"] = list(self.dess_buses)
        # requested_* remains pre-backup candidate diagnostics. With DOE on,
        # that candidate is DOE-conditioned; marl_* always identifies policy.
        if physical_power is None:
            physical_power = info["requested_battery_actual_power_kw"].copy()
            doe_power = physical_power.copy()
        info.update(doe_diagnostics)
        for key in ("doe_preventive_clipped", "doe_corrective_active", "doe_corrective_intervened"):
            info.setdefault(key, np.zeros(self.num_agents, dtype=bool))
        for key in ("doe_corrective_required_kw", "doe_corrective_shortfall_kw"):
            info.setdefault(key, np.zeros(self.num_agents))
        info["doe_enabled"] = self.doe_enabled
        info["marl_raw_action"] = marl_raw_action
        info["marl_prepared_action"] = marl_action
        info["marl_requested_power_kw"] = np.array([
            a * (b.config.max_discharge_kw if a >= 0 else b.config.max_charge_kw)
            for a, b in zip(marl_action, self.batteries)])
        info["pre_action_local_voltage_pu"] = pre_voltage
        info["pre_action_pf_converged"] = pre_converged
        info["soc_before"] = soc_before.copy()
        info["soc_after"] = soc_after.copy()
        info["physical_limited_power_kw"] = physical_power
        info["doe_power_kw"] = doe_power
        info["doe_conditioned_action"] = action.copy()
        info["final_battery_power_kw"] = np.array([b["power_kw"] for b in battery_outputs])
        # Physical power/SOC-limit clipping of the [-1, 1] policy proposal (not penalized).
        physical_clipped_kw = np.abs(info["marl_requested_power_kw"] - physical_power)
        info["physical_clipped_kw"] = physical_clipped_kw
        info["physical_clipped"] = physical_clipped_kw > 1e-6
        info["physical_clipping_any"] = bool(np.any(info["physical_clipped"]))
        info["physical_clipped_kw_total"] = float(np.sum(physical_clipped_kw))
        info["doe_change_kw"] = doe_change_kw.copy()
        info["doe_change_normalized"] = doe_change_normalized.copy()
        info["doe_penalty_coef"] = self.doe_penalty_coef
        info["doe_penalty_amount"] = np.asarray(doe_penalty, dtype=float).copy()
        # F8.5: reward components (per agent); their sum reconstructs the returned reward.
        terms = {k: np.asarray(v, dtype=float).copy() for k, v in getattr(self, "_last_reward_terms", {}).items()}
        residual = (float(backup_diagnostics.get("backup_residual_penalty", 0.0))
                    if self.backup_mode == "directional_incremental" and not failed_dispatch else 0.0)
        terms["residual_penalty"] = np.full(self.num_agents, residual)
        terms["doe_penalty"] = np.asarray(doe_penalty, dtype=float).copy()
        for name, value in terms.items():
            info["reward_term_" + name] = value
        info["reward_terms_total"] = np.sum(list(terms.values()), axis=0)
        # F8.1-F8.4: read-only classification of PF results already computed this step.
        info.update(proposal_diag)
        info.update(self._violation_fields("doe_candidate_", requested_candidate["pf_result"]))
        info.update(self._violation_fields("executed_", None if failed_dispatch else pf_result))
        info["backup_activated"] = bool(candidate.get("correction_applied", False))
        if self.backup_mode == "directional_incremental":
            info["backup_activated"] = bool(raw_infeasible_requested)
        info["final_network_feasible"] = bool(candidate["feasible"])
        if self.backup_mode == "directional_incremental":
            info["feasible"] = bool(candidate["feasible"])
        # F12: feasibility of a zero fallback is known only if its power flow was actually solved;
        # a failed dispatch (zero placeholder, no valid PF) is unknown, never "infeasible zero".
        info["zero_fallback_feasible"] = (
            bool(candidate["feasible"]) if info["correction_fallback_to_zero"] and not failed_dispatch else None)
        info["zero_fallback_evaluated"] = bool(info["correction_fallback_to_zero"] and not failed_dispatch)

        info.update(backup_diagnostics)
        info.update(backup_mode=self.backup_mode, doe_skipped=doe_skipped,
                    pre_action_pf_failure=pre_failure, pf_failure_count=self.pf_failure_count,
                    pf_failure_reason=getattr(self, "pf_failure_reason", ""),
                    pf_failed_solve_count=self.pf_failed_solve_count,
                    step_pf_failed_solves=self.pf_failed_solve_count-failed_solves_before,
                    episode_failed=failed_dispatch, failed_episode_count=self.failed_episode_count,
                    dispatch_executed=not failed_dispatch,
                    observation_valid=not failed_dispatch, observation_pf_evaluations=0,
                    transition_valid=not (failed_dispatch or pre_failure),
                    terminated=terminated, truncated=truncated, pf_failure_stage="",
                    backup_pf_budget=self.backup_cfg["pf_budget"])
        if self.backup_mode == "legacy_scaling":
            additional = info["correction_iterations"] + int(info["correction_fallback_to_zero"])
            info.update(backup_pf_evaluations=1+additional,
                        backup_additional_pf_evaluations=additional)
            # Diagnostic scoring never changes legacy acceptance or rewards.
            # Legacy users can have no effective ratings with current checking
            # disabled; _measure_violation handles that configuration.
            info.update(legacy_diagnostics(requested_candidate, candidate, self._measure_violation))
            info["backup_initial_power_kw"] = physical_power.copy() if not self.doe_enabled else doe_power.copy()
            info["backup_adjustment_kw"] = info["final_battery_power_kw"] - info["backup_initial_power_kw"]
        info["backup_penalty_applied"] = backup_penalty_applied
        penalty_applied = backup_penalty_applied
        info["major_backup_penalty_applied"] = penalty_applied
        info["major_backup_penalty_amount"] = (float(self.reward_cfg.get(
            "infeasible_action_penalty", -50.0)) if penalty_applied else 0.0)
        zero_valid = bool(info.get("backup_zero_evaluated", False) and
                          np.isfinite(info.get("backup_zero_score", np.nan)))
        info.update(backup_attribution(info))
        info["backup_zero_valid"] = zero_valid
        info["backup_zero_infeasible"] = (bool(info["backup_zero_score"] > 0)
                                           if zero_valid else None)
        if self.pf_failed_solve_count > failed_solves_before:
            info["pf_failure_stage"] = "candidate_search"
        if pre_failure:
            info["pf_failure_stage"] = "pre_action"
        current_base_a = self.config["network"]["s_base_kva"] / (np.sqrt(3)*self.config["network"]["v_base_kv"])
        info["step_total_pf_evaluations"] = info["backup_pf_evaluations"]
        for stage in ("initial", "final"):
            info["backup_"+stage+"_max_line_current_a"] = info["backup_"+stage+"_max_line_current"] * current_base_a
        if failed_dispatch:
            info["feasible"] = info["final_network_feasible"] = False
            info.update(backup_fully_feasible=False, backup_best_effort=False,
                        backup_outcome="not resolved", backup_final_score=float('inf'),
                        backup_residual_penalty=0.0)
            info["backup_failure_reason"] = "no_valid_dispatch" if not self._reset_pf_failed else "reset_pf_failed"
            info.update(transition_valid=False, terminated=False, truncated=True,
                        pf_failure_stage="reset" if self._reset_pf_failed else "candidate_search")
            info.update(self._mark_episode_failed(info["pf_failure_stage"]))
            return self._failure_observation(load_kw, pv_used_kw, price), rewards, False, True, info

        # If episode ends here, return the current post-action observation.
        if truncated and (self.current_index + 1 >= len(self.time_series["df"])
                          or self.current_index + 1 not in self.splits.get(self.mode + "_indices", range(len(self.time_series["df"])))):
            info["transition_valid"] = False
            info["bootstrap_unavailable_reason"] = "end_of_data_split"
            next_obs = self._build_observation(
                load_kw=load_kw,
                pv_kw=pv_used_kw,
                price=price,
                voltage_pu=pf_result["voltage_pu"],
            )
            return next_obs, rewards, terminated, truncated, info

        # Advance time.
        self.current_step += 1
        self.current_index += 1

        next_load_kw, next_load_kvar, next_pv_kw, next_price, next_date_time = (
            self._get_profiles(self.current_index)
        )

        next_pv_used_kw, next_curtailment_by_bus, next_curtailment_kw = (
            self._apply_curtailment(
                load_kw=next_load_kw,
                pv_kw=next_pv_kw,
                dess_power_kw=self.last_dess_power_kw,
            )
        )

        next_pf_result = self._run_pf(
            load_kw=next_load_kw,
            load_kvar=next_load_kvar,
            pv_kw=next_pv_used_kw,
            dess_power_kw=self.last_dess_power_kw,
        )
        info["observation_pf_evaluations"] = 1
        info["pf_failed_solve_count"] = self.pf_failed_solve_count
        info["step_pf_failed_solves"] = self.pf_failed_solve_count-failed_solves_before
        info["step_total_pf_evaluations"] += 1

        if not valid_pf(next_pf_result):
            self._record_pf_failure(next_pf_result)
            self.current_step -= 1
            self.current_index -= 1
            if self.doe_enabled and self.doe_pf_failure_mode == "strict":
                raise ValueError("Invalid/nonconverged next-observation power flow")
            info.update(episode_failed=True, pf_failure_count=self.pf_failure_count,
                        pf_failure_reason="next_observation: " + self.pf_failure_reason,
                        observation_valid=False, next_observation_pf_failed=True,
                        transition_valid=False, terminated=False, truncated=True,
                        pf_failure_stage="next_observation")
            terminal_obs = self._build_observation(load_kw=load_kw, pv_kw=pv_used_kw,
                price=price, voltage_pu=pf_result["voltage_pu"])
            info.update(self._mark_episode_failed("next_observation"))
            return terminal_obs, rewards, False, True, info

        self.last_pf_result = next_pf_result
        self.last_curtailment_by_bus = next_curtailment_by_bus.copy()
        self.last_curtailment_kw = float(next_curtailment_kw)

        self.current_kpis = self._compute_kpis(
            pf_result=next_pf_result,
            price=next_price,
            curtailment_kw=next_curtailment_kw,
        )

        self.last_date_time = next_date_time

        next_obs = self._build_observation(
            load_kw=next_load_kw,
            pv_kw=next_pv_used_kw,
            price=next_price,
            voltage_pu=next_pf_result["voltage_pu"],
        )

        return next_obs, rewards, terminated, truncated, info

    # ============================================================
    # Candidate action evaluation
    # ============================================================

    def _power_to_action(self, powers):
        powers = np.asarray(powers, dtype=float)
        ratings = np.array([b.config.max_discharge_kw if p >= 0 else b.config.max_charge_kw
                            for b, p in zip(self.batteries, powers)])
        return np.divide(powers, ratings, out=np.zeros_like(powers), where=ratings > 0)

    def _measure_violation(self, pf):
        if "voltage_complex" in pf and np.all(np.isfinite(pf["voltage_complex"])):
            # Solver's public magnitudes are float32. Reuse the saved complex
            # solution for full-precision metrics under the shared feasibility convention.
            # This is algebraic postprocessing, NOT another AC PF solve.
            pf = dict(pf)
            pf["voltage_pu"] = np.abs(pf["voltage_complex"])
            pf["line_current_pu"] = self.power_flow._compute_branch_quantities(
                pf["voltage_complex"])["line_current_pu"]
        return violation_metrics(pf, self.grid.get_line_current_limits_pu(),
            v_min=self.v_min, v_max=self.v_max,
            voltage_enabled=self.constraint_cfg.get("enforce_voltage_limits", True),
            current_enabled=self.constraint_cfg.get("enforce_line_current_limits", True),
            voltage_scale_pu=self.backup_cfg["voltage_scale_pu"],
            relative_current_scale=self.backup_cfg["relative_current_scale"])

    def _violation_fields(self, prefix, pf):
        """F8.3/F8.4 read-only classification of an already-solved PF (no new solve).

        Same precision and tolerances as _measure_violation. Battery presence at a violating
        bus is reported as a location fact, not as evidence of causation. Line loading is
        |I| / effective rating of EACH branch, in percent (production ratings: 255 A).
        """
        keys = ("voltage_violation", "current_violation", "network_violation", "undervoltage_buses",
                "overvoltage_buses", "overloaded_branches", "violating_buses_with_dess",
                "violating_buses_without_dess", "violating_bus_has_dess", "line_loading_pct",
                "max_line_loading_pct", "max_line_loading_branch")
        if pf is None or not valid_pf(pf):
            return {prefix + k: None for k in keys}
        if "voltage_complex" in pf and np.all(np.isfinite(pf["voltage_complex"])):
            v = np.abs(pf["voltage_complex"])
            current = self.power_flow._compute_branch_quantities(pf["voltage_complex"])["line_current_pu"]
        else:
            v = np.asarray(pf.get("voltage_pu_raw", pf["voltage_pu"]), dtype=float)
            current = np.asarray(pf.get("line_current_pu_raw", pf["line_current_pu"]), dtype=float)
        current = np.abs(np.asarray(current, dtype=float))
        limits = np.asarray(self.grid.get_line_current_limits_pu(), dtype=float)
        low, high = voltage_excess(v, self.v_min, self.v_max)
        under = np.flatnonzero(low > VOLTAGE_TOLERANCE_PU)
        over = np.flatnonzero(high > VOLTAGE_TOLERANCE_PU)
        overloaded = np.flatnonzero(current_overloaded(current, limits))
        violating = np.union1d(under, over)
        dess = np.asarray(self.dess_buses)
        with_dess = violating[np.isin(violating, dess)]
        loading = 100.0 * current / limits
        values = dict(voltage_violation=bool(violating.size), current_violation=bool(overloaded.size),
                      network_violation=bool(violating.size or overloaded.size),
                      undervoltage_buses=under, overvoltage_buses=over, overloaded_branches=overloaded,
                      violating_buses_with_dess=with_dess,
                      violating_buses_without_dess=violating[~np.isin(violating, dess)],
                      violating_bus_has_dess=bool(with_dess.size), line_loading_pct=loading,
                      max_line_loading_pct=float(loading.max()) if loading.size else 0.0,
                      max_line_loading_branch=int(np.argmax(loading)) if loading.size else -1)
        return {prefix + k: values[k] for k in keys}

    def _physical_proposal_diagnostic(self, physical_power, doe_power, doe_skipped, requested_candidate,
                                      load_kw, load_kvar, pv_kw):
        """F8.1: is the physically limited (pre-DOE) proposal feasible?

        Stage definitions: policy proposal -> physically limited proposal (this check) ->
        DOE-conditioned candidate (doe_candidate_*; the backup-needed flag) -> executed
        (executed_*). If the DOE did not change the action, the physically limited proposal IS
        the DOE candidate and its existing PF is reused. Otherwise one extra PF is solved only
        when evaluation_diagnostics is True (off by default / in training). That solve calls the
        solver directly: outside the backup's PF budget, no battery state, failure counters,
        dispatch, reward or replay are touched.
        """
        req = self._measure_violation(requested_candidate["pf_result"])
        same = dict(physical_proposal_feasible=bool(req["feasible"]) if req["valid"] else None,
                    physical_proposal_score=float(req["score"]) if req["valid"] else None)
        if not self.doe_enabled or physical_power is None or doe_skipped:
            return dict(same, physical_proposal_source="no_doe_stage_same_as_candidate")
        if not np.any(np.abs(np.asarray(physical_power) - np.asarray(doe_power)) > 1e-9):
            return dict(same, physical_proposal_source="unchanged_by_doe_same_as_candidate")
        if not self.evaluation_diagnostics:
            return dict(physical_proposal_feasible=None, physical_proposal_score=None,
                        physical_proposal_source="diagnostic_disabled")
        t0 = time.perf_counter()
        dess_kw = np.zeros(self.num_buses, dtype=np.float64)
        dess_kw[self.dess_buses] = np.asarray(physical_power, dtype=float)
        pv_used, _, _ = self._apply_curtailment(load_kw=load_kw, pv_kw=pv_kw, dess_power_kw=dess_kw)
        try:
            result = self.power_flow.run_power_flow(load_kw=load_kw, load_kvar=load_kvar,
                                                    pv_kw=pv_used, dess_power_kw=dess_kw)
        except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError):
            result = None
        self._step_timing["diagnostic_pf"] = time.perf_counter() - t0
        if result is None or not valid_pf(result):
            return dict(physical_proposal_feasible=None, physical_proposal_score=None,
                        physical_proposal_source="diagnostic_pf_failed")
        m = self._measure_violation(result)
        return dict(physical_proposal_feasible=bool(m["feasible"]), physical_proposal_score=float(m["score"]),
                    physical_proposal_source="diagnostic_pf")

    def _run_pf(self, **kwargs):
        t0 = time.perf_counter()  # F8: PF wall time, accumulated per step (instrumentation only)
        try:
            result = self.power_flow.run_power_flow(**kwargs)
            self._pf_time_s += time.perf_counter() - t0
            self._pf_solves += 1
            if not valid_pf(result):
                self.pf_failed_solve_count += 1
                self._record_pf_failure(result)
            return result
        except (ValueError, RuntimeError, ArithmeticError, np.linalg.LinAlgError) as exc:
            self._pf_time_s += time.perf_counter() - t0
            self._pf_solves += 1
            if not self._robust_pf:
                raise
            self.pf_failed_solve_count += 1
            self._record_pf_failure({"failure_reason": type(exc).__name__ + ": " + str(exc)})
            # Invalid results are marked, never promoted to a verified state.
            return dict(converged=False, feasible=False, iterations=0,
                voltage_pu=np.full(self.num_buses, np.nan),
                line_current_pu=np.full(len(self.grid.in_service_line_df), np.nan),
                grid_import_kw=np.nan, grid_import_kvar=np.nan,
                max_voltage_violation=float('inf'), max_line_current_violation=float('inf'),
                failure_reason=type(exc).__name__ + ": " + str(exc))

    def _mark_episode_failed(self, stage):
        if not self._episode_failure_reported:
            self.failed_episode_count += 1
            self._episode_failure_reported = True
        return dict(episode_failed=True, failed_episode_count=self.failed_episode_count,
                    pf_failure_stage=stage, terminated=False, truncated=True,
                    transition_valid=False)

    def _record_pf_failure(self, pf):
        if self.current_index not in self._failed_pf_indices:
            self.pf_failure_count += 1
            self._failed_pf_indices.add(self.current_index)
        self.pf_failure_reason = pf.get("failure_reason", "nonconverged_or_nonfinite_power_flow")

    def _failure_observation(self, load_kw, pv_kw, price):
        # Terminal/reset-failure API placeholder, NOT a measured state. Callers
        # receive observation_valid=False and episode_failed=True. Never DOE it.
        saved = self.current_kpis
        try:
            self.current_kpis = {k: float(v) if np.isfinite(v) else 0.0 for k, v in saved.items()}
            return self._build_observation(load_kw=load_kw, pv_kw=pv_kw, price=price,
                                           voltage_pu=np.ones(self.num_buses))
        finally:
            self.current_kpis = saved

    def _prepare_action(self, action):
        """
        Validate and clip the normalized action vector.
        """

        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.shape != (self.num_agents,):
            raise ValueError(
                "Action must have shape ({},), got {}.".format(
                    self.num_agents,
                    action.shape,
                )
            )

        return np.clip(
            action,
            float(self.config["action"]["low"]),
            float(self.config["action"]["high"]),
        ).astype(np.float32)

    def _evaluate_candidate_action(self, action, load_kw, load_kvar, pv_kw):
        """
        Evaluate a candidate DESS action without permanently committing it.
        """

        battery_states_before = self._snapshot_battery_states()

        try:
            dess_power_kw = np.zeros(self.num_buses, dtype=np.float64)
            battery_outputs = []

            for agent_idx, bus in enumerate(self.dess_buses):
                battery_result = self.batteries[agent_idx].step(action[agent_idx])
                dess_power_kw[bus] = battery_result["power_kw"]
                battery_outputs.append(battery_result)

            battery_states_after = self._snapshot_battery_states()

            pv_used_kw, curtailment_by_bus, curtailment_kw = self._apply_curtailment(
                load_kw=load_kw,
                pv_kw=pv_kw,
                dess_power_kw=dess_power_kw,
            )

            pf_result = self._run_pf(
                load_kw=load_kw,
                load_kvar=load_kvar,
                pv_kw=pv_used_kw,
                dess_power_kw=dess_power_kw,
            )

            feasible = self._is_feasible_pf_result(pf_result)


        finally:
            self._restore_battery_states(battery_states_before)

        return {
            "accepted_action": action.copy(),
            "dess_power_kw": dess_power_kw,
            "battery_outputs": battery_outputs,
            "battery_states_after": battery_states_after,
            "pv_used_kw": pv_used_kw,
            "curtailment_by_bus": curtailment_by_bus,
            "curtailment_kw": float(curtailment_kw),
            "pf_result": pf_result,
            "feasible": bool(feasible),
            "correction_applied": False,
            "correction_iterations": 0,
            "correction_fallback_to_zero": False,
        }

    def _handle_infeasible_action(self, original_action, load_kw, load_kvar, pv_kw):
        """
        Handle an infeasible action according to the configured rule.

        Minor correction:
            action becomes feasible after a small number of scaling attempts.

        Major correction:
            multiple correction attempts are needed or zero-action fallback is used.
        """

        if self.hard_constraint_action == "reject_action":
            zero_action = np.zeros(self.num_agents, dtype=np.float32)
            candidate = self._evaluate_candidate_action(
                action=zero_action,
                load_kw=load_kw,
                load_kvar=load_kvar,
                pv_kw=pv_kw,
            )
            candidate["accepted_action"] = zero_action.copy()
            candidate["correction_applied"] = True
            candidate["correction_iterations"] = 0
            candidate["correction_fallback_to_zero"] = True
            return candidate

        if self.hard_constraint_action == "clip_or_correct":
            corrected_action = original_action.copy()

            for attempt in range(1, self.max_correction_attempts + 1):
                corrected_action = corrected_action * self.action_correction_factor

                candidate = self._evaluate_candidate_action(
                    action=corrected_action,
                    load_kw=load_kw,
                    load_kvar=load_kvar,
                    pv_kw=pv_kw,
                )

                if candidate["feasible"]:
                    candidate["accepted_action"] = corrected_action.copy()
                    candidate["correction_applied"] = True
                    candidate["correction_iterations"] = int(attempt)
                    candidate["correction_fallback_to_zero"] = False
                    return candidate

            zero_action = np.zeros(self.num_agents, dtype=np.float32)
            candidate = self._evaluate_candidate_action(
                action=zero_action,
                load_kw=load_kw,
                load_kvar=load_kvar,
                pv_kw=pv_kw,
            )
            candidate["accepted_action"] = zero_action.copy()
            candidate["correction_applied"] = True
            candidate["correction_iterations"] = int(self.max_correction_attempts)
            candidate["correction_fallback_to_zero"] = True
            return candidate

        if self.hard_constraint_action == "terminate_episode":
            zero_action = np.zeros(self.num_agents, dtype=np.float32)
            candidate = self._evaluate_candidate_action(
                action=zero_action,
                load_kw=load_kw,
                load_kvar=load_kvar,
                pv_kw=pv_kw,
            )
            candidate["accepted_action"] = zero_action.copy()
            candidate["correction_applied"] = True
            candidate["correction_iterations"] = 0
            candidate["correction_fallback_to_zero"] = True
            return candidate

        raise ValueError(
            "Unsupported hard_constraint_action: {}".format(
                self.hard_constraint_action
            )
        )

    def _is_feasible_pf_result(self, pf_result):
        """
        Check whether a power-flow result satisfies enabled feasibility limits.
        """

        return self._measure_violation(pf_result)["feasible"]


    # ============================================================
    # Battery-state snapshots
    # ============================================================

    def _snapshot_battery_states(self):
        """
        Save current battery states.
        """

        return [
            {
                "soc": float(battery.soc),
                "last_power_kw": float(battery.last_power_kw),
                "last_energy_kwh": float(battery.last_energy_kwh),
                "last_requested_power_kw": float(
                    getattr(battery, "last_requested_power_kw", 0.0)
                ),
                "last_action": float(getattr(battery, "last_action", 0.0)),
                "last_was_limited": bool(
                    getattr(battery, "last_was_limited", False)
                ),
            }
            for battery in self.batteries
        ]

    def _restore_battery_states(self, states):
        """
        Restore battery states.
        """

        for battery, state in zip(self.batteries, states):
            battery.soc = float(state["soc"])
            battery.last_power_kw = float(state["last_power_kw"])
            battery.last_energy_kwh = float(state["last_energy_kwh"])
            battery.last_requested_power_kw = float(state["last_requested_power_kw"])
            battery.last_action = float(state["last_action"])
            battery.last_was_limited = bool(state["last_was_limited"])

    # ============================================================
    # Time-series profiles
    # ============================================================

    def _get_profiles(self, row_index):
        """
        Load one scaled node-level time-series row.
        """

        row = select_time_series_row(self.time_series, row_index)

        load_kw = row["load_kw"].astype(np.float64) * self.load_scaling_factor
        pv_kw = row["pv_kw"].astype(np.float64) * self.pv_scaling_factor
        price = float(row["price"]) * self.price_scaling_factor
        date_time = row["date_time"]

        base_load_kw = self.grid.get_base_load_kw().astype(np.float64)
        base_load_kvar = self.grid.get_base_load_kvar().astype(np.float64)

        total_base_load = float(np.sum(base_load_kw))

        if total_base_load <= 0.0:
            raise ValueError("Total base active load must be positive.")

        load_scale = float(np.sum(load_kw)) / total_base_load
        load_kvar = base_load_kvar * load_scale

        return load_kw, load_kvar, pv_kw, price, date_time

    # ============================================================
    # Curtailment calculation
    # ============================================================

    def _apply_curtailment(self, load_kw, pv_kw, dess_power_kw):
        """
        Apply feeder-level renewable curtailment using the configured export limit.
        """

        total_load_kw = float(np.sum(load_kw))
        total_pv_kw = float(np.sum(pv_kw))

        total_dess_discharge_kw = float(np.sum(np.maximum(0.0, dess_power_kw)))
        total_dess_charge_kw = float(np.sum(np.maximum(0.0, -dess_power_kw)))

        surplus_kw = (
            total_pv_kw
            + total_dess_discharge_kw
            - total_load_kw
            - total_dess_charge_kw
            - self.grid_export_limit_kw
        )

        curtailment_kw = max(0.0, surplus_kw)
        curtailment_kw = min(curtailment_kw, total_pv_kw)

        if curtailment_kw <= 0.0 or total_pv_kw <= 0.0:
            return pv_kw.copy(), np.zeros_like(pv_kw), 0.0

        curtailment_fraction = curtailment_kw / total_pv_kw
        curtailment_by_bus = pv_kw * curtailment_fraction
        pv_used_kw = np.maximum(0.0, pv_kw - curtailment_by_bus)

        return pv_used_kw, curtailment_by_bus, float(np.sum(curtailment_by_bus))

    # ============================================================
    # KPI and reward calculation
    # ============================================================

    def _compute_kpis(self, pf_result, price, curtailment_kw):
        """
        Compute grid-level performance indicators.
        """

        voltage_pu = pf_result["voltage_pu"]

        voltage_deviation = float(np.sum(np.abs(voltage_pu - self.v_ref)))
        voltage_violation_max = float(pf_result["max_voltage_violation"])
        line_current_violation = float(pf_result["max_line_current_violation"])

        grid_stress = voltage_violation_max + line_current_violation

        return {
            "grid_import_kw": float(pf_result["grid_import_kw"]),
            "grid_import_kvar": float(pf_result["grid_import_kvar"]),
            "voltage_deviation": voltage_deviation,
            "voltage_violation_max": voltage_violation_max,
            "curtailment_kw": float(curtailment_kw),
            "line_current_violation": line_current_violation,
            "grid_stress": grid_stress,
            "price": float(price),
        }

    def _calculate_rewards(
        self,
        price,
        dess_power_kw,
        soc_before,
        soc_after,
        infeasible_action=False,
        backup_penalty_applied=None,
    ):
        """
        Compute per-agent cooperative rewards.

        Research_Summer-consistent structure:

            r_i(t) = w_grid  * R_grid(t)
                   + w_local * R_local,i(t)
                   + w_kpi   * R_KPI(t)
        """

        cfg = self.reward_cfg

        if not bool(self.last_pf_result["converged"]):
            penalty = float(cfg.get("nonconvergence_penalty", -10000.0))
            self._last_reward_terms = dict(nonconvergence_penalty=np.full(self.num_agents, penalty))  # F8.5
            return np.ones(self.num_agents, dtype=np.float32) * penalty

        w_grid = float(cfg.get("w_grid", 0.60))
        w_local = float(cfg.get("w_local", 0.15))
        w_kpi = float(cfg.get("w_kpi", 0.25))

        grid_import_kw = float(self.current_kpis["grid_import_kw"])
        voltage_deviation = float(self.current_kpis["voltage_deviation"])
        curtailment_kw = float(self.current_kpis["curtailment_kw"])

        grid_import_mwh = max(0.0, grid_import_kw) * self.delta_t_hours / 1000.0
        curtailment_mwh = curtailment_kw * self.delta_t_hours / 1000.0

        energy_cost = float(price) * grid_import_mwh

        grid_reward = -(
            float(cfg.get("phi_cost", 1.0)) * energy_cost
            + float(cfg.get("phi_voltage_deviation", 1.0)) * voltage_deviation
            + float(cfg.get("lambda_curtailment", 1.0)) * curtailment_mwh
        )

        kpi_reward = 0.0
        if bool(cfg.get("use_kpi_reward", True)):
            kpi_reward = self._calculate_kpi_reward()

        rewards = np.zeros(self.num_agents, dtype=np.float32)
        local_terms = np.zeros(self.num_agents)  # F8.5 record only

        for agent_idx, bus in enumerate(self.dess_buses):
            power_abs_kw = abs(float(dess_power_kw[bus]))
            power_abs_mwh = power_abs_kw * self.delta_t_hours / 1000.0
            delta_soc = float(soc_after[agent_idx] - soc_before[agent_idx])

            local_reward = -(
                float(cfg.get("phi_cycling", 0.05)) * power_abs_mwh
                + float(cfg.get("phi_soc_reserve", 0.05)) * (delta_soc ** 2)
            )

            rewards[agent_idx] = (
                w_grid * grid_reward
                + w_local * local_reward
                + w_kpi * kpi_reward
            )
            local_terms[agent_idx] = w_local * local_reward

        # Compatibility for existing direct callers; step passes the explicit bool.
        if backup_penalty_applied is None:
            backup_penalty_applied = bool(infeasible_action)
        if backup_penalty_applied:
            rewards += float(cfg.get("infeasible_action_penalty", -50.0))
        # F8.5: weighted reward components exactly as summed above (record only).
        n = self.num_agents
        self._last_reward_terms = dict(
            grid_energy_cost=np.full(n, -w_grid * float(cfg.get("phi_cost", 1.0)) * energy_cost),
            grid_voltage_deviation=np.full(n, -w_grid * float(cfg.get("phi_voltage_deviation", 1.0)) * voltage_deviation),
            grid_curtailment=np.full(n, -w_grid * float(cfg.get("lambda_curtailment", 1.0)) * curtailment_mwh),
            local=local_terms, kpi=np.full(n, w_kpi * kpi_reward),
            backup_penalty=np.full(n, float(cfg.get("infeasible_action_penalty", -50.0)) if backup_penalty_applied else 0.0))

        return rewards.astype(np.float32)

    def _calculate_kpi_reward(self):
        """
        Compute KPI-improvement reward using consecutive grid indicators.
        """

        if self.previous_kpis is None:
            return 0.0

        cfg = self.reward_cfg

        current = self.current_kpis
        previous = self.previous_kpis

        delta_import = (
            previous["grid_import_kw"] - current["grid_import_kw"]
        ) * self.delta_t_hours / 1000.0

        delta_curtailment = (
            previous["curtailment_kw"] - current["curtailment_kw"]
        ) * self.delta_t_hours / 1000.0

        delta_vdev = previous["voltage_deviation"] - current["voltage_deviation"]
        delta_stress = previous["grid_stress"] - current["grid_stress"]

        kpi_reward = (
            float(cfg.get("beta_import", 1.0)) * delta_import
            + float(cfg.get("beta_voltage_dev", 1.0)) * delta_vdev
            + float(cfg.get("beta_curtailment", 1.0)) * delta_curtailment
            + float(cfg.get("beta_grid_stress", 1.0)) * delta_stress
        )

        return float(kpi_reward)

    # ============================================================
    # Diagnostics
    # ============================================================

    def _build_info(self):
        """
        Build diagnostic information dictionary.
        """

        min_voltage = float(np.min(self.last_pf_result["voltage_pu"]))
        max_voltage = float(np.max(self.last_pf_result["voltage_pu"]))

        return {
            "mode": self.mode,
            "episode_start_index": self.episode_start_index,
            "current_step": self.current_step,
            "current_index": self.current_index,
            "date_time": getattr(self, "last_date_time", None),
            "dess_buses": list(self.dess_buses),
            "num_agents": int(self.num_agents),
            "agent_obs_dim": int(self.agent_obs_dim),
            "kpis": dict(self.current_kpis) if self.current_kpis is not None else None,
            "feasible": bool(self._is_feasible_pf_result(self.last_pf_result)),
            "converged": bool(self.last_pf_result["converged"]),
            "iterations": int(self.last_pf_result["iterations"]),
            "min_voltage": min_voltage,
            "max_voltage": max_voltage,
            "min_voltage_pu": min_voltage,
            "max_voltage_pu": max_voltage,
            "max_voltage_violation": float(
                self.last_pf_result["max_voltage_violation"]
            ),
            "max_line_current_pu": float(
                np.max(self.last_pf_result["line_current_pu"])
            ),
            "max_line_current_violation": float(
                self.last_pf_result["max_line_current_violation"]
            ),
            "curtailment_kw": float(self.current_kpis["curtailment_kw"]),
            "load_scaling_factor": self.load_scaling_factor,
            "pv_scaling_factor": self.pv_scaling_factor,
            "price_scaling_factor": self.price_scaling_factor,
            "normalize_observations": self.normalize_observations,
        }

    def render(self):
        """
        Print compact environment diagnostics.
        """

        info = self._build_info()
        kpis = info["kpis"]

        print(
            "mode={}, step={}, date_time={}, feasible={}, minV={:.4f}, "
            "maxV={:.4f}, grid_import={:.2f} kW, curtailment={:.2f} kW, "
            "max_line_current={:.4f} pu".format(
                info["mode"],
                info["current_step"],
                info["date_time"],
                info["feasible"],
                info["min_voltage_pu"],
                info["max_voltage_pu"],
                kpis["grid_import_kw"],
                kpis["curtailment_kw"],
                info["max_line_current_pu"],
            )
        )
