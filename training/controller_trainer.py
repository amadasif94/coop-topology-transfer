"""Maintained controller trainers: the ORIGINAL MLPMADDPGTrainer / GraphMADDPGTrainer loops (warm-up, exploration, replay validity,
reset handling, update cadence, periodic validation, checkpoint cadence), with only two bindings replaced:
the model factory (-> models.controllers) and the replay factory (-> training.bounded_replay). This is the
repository's existing idiom for ablation trainers (experiments/no_shared_kpi/runner.trainer_class): a private
globals dictionary, no monkey-patching of the original modules, no copied loop.

Added behaviour (training environment only; periodic validation is untouched):
  * _step_env hands the pre-action feasible bounds recorded by MarlDESSEnv for this step to the replay buffer, keyed to
    the step's next_obs object, so add() can only attach them to the same transition;
  * training_days.csv, one row per training day (written when the day ends): phase (warmup = uniform random actions,
    policy = actor + exploration noise), steps, exploratory/raw-action infeasible fraction (submitted action vs
    pre-action bounds), battery SoC projection (physical clipping steps and kW), network backup activations,
    energy cost, voltage/current violation steps, final network feasibility, terminal SoC shortfall and penalty,
    failed day;
  * training_updates.csv, one row every `diagnostics_every` updates: the agent's update diagnostics
    (training.feasibility_supervision: pre-tanh statistics, saturation, deterministic actor infeasible fraction,
    feasibility loss, component and pre/post-clip actor gradient norms) plus the exploratory infeasible fraction
    accumulated since the previous row.
  * training_gradient_balance.csv (only with config gradient_balance_log, on for A3): per-agent actor-gradient norms
    of the RL term and of the lambda_f-weighted feasibility term and their ratio, on the same diagnostic updates.
The deterministic actor infeasible fraction (current actor, no noise, on sampled replay states) and the exploratory
infeasible fraction (submitted actions incl. noise or uniform warm-up) are kept separate.
"""
import csv
from functools import partial
import json
from pathlib import Path
from types import FunctionType

import numpy as np

from environments.battery_feasibility import violation
from configs.experiments import CONTROLLER_KEYS, LEGACY_LABELS
from models.controllers import GNN_TYPES, build_model
from training.bounded_replay import build_bounded_replay_buffer
from experiments.no_gnn_maddpg.trainer_mlp import MLPMADDPGTrainer

DAY_FIELDS = ('day', 'phase', 'global_step_end', 'start_index', 'steps', 'failed_day', 'completed_day',
              'raw_action_infeasible_fraction', 'physical_clipping_steps', 'physical_clipped_kw_total',
              'backup_activations', 'energy_cost', 'grid_import_mwh', 'voltage_violation_steps',
              'current_violation_steps', 'network_infeasible_steps', 'terminal_shortfall_mean', 'terminal_shortfall_max',
              'terminal_penalty')
UPDATE_FIELDS = ('update_count', 'global_step', 'critic_loss', 'actor_loss', 'mean_q', 'feasibility_formulation',
                 'lambda_feasibility', 'feasibility_loss', 'feasibility_loss_weighted', 'action_space_violation_sq',
                 'actor_infeasible_fraction', 'raw_action_infeasible_fraction_window', 'preact_abs_mean', 'preact_std',
                 'preact_abs_p50', 'preact_abs_p90', 'preact_abs_p99', 'preact_abs_max', 'saturation_fraction_099',
                 'saturation_fraction_0999', 'actor_grad_norm_pre_clip', 'actor_grad_norm_post_clip', 'q_param_grad_norm',
                 'feasibility_param_grad_norm', 'feasibility_param_grad_norm_unweighted',
                 'q_z_grad_median_abs_infeasible', 'feas_z_grad_median_abs_infeasible',
                 'update_time_ms')
# Optional (config gradient_balance_log; on for A3): per-agent RL vs lambda_f-weighted feasibility actor-gradient norms.
BALANCE_FIELDS = ('update_count', 'global_step', 'lambda_feasibility', 'q_param_grad_norm', 'feasibility_param_grad_norm',
                  'gradient_balance', 'q_param_grad_norm_per_agent', 'feasibility_param_grad_norm_per_agent',
                  'gradient_balance_per_agent')


def _rebound_init(base, bindings):
    original = base.__init__
    constructor = FunctionType(original.__code__, dict(original.__globals__, **bindings), original.__name__,
                               original.__defaults__, original.__closure__)
    constructor.__kwdefaults__ = original.__kwdefaults__
    return constructor


class _CsvLog:
    """Fixed-column CSV, created on first write; appended to only when `append` (a resumed run) is set."""

    def __init__(self, path, fields, append=False):
        self.path, self.fields, self.append, self.started = Path(path), list(fields), append, False

    def write(self, row):
        if not self.started:
            if self.path.exists() and not self.append:
                raise FileExistsError(f'{self.path} exists; diagnostics are never overwritten')
            if not self.path.exists():
                with self.path.open('x', newline='') as f:
                    csv.DictWriter(f, self.fields).writeheader()
            self.started = True
        with self.path.open('a', newline='') as f:
            csv.DictWriter(f, self.fields, extrasaction='ignore').writerow({k: row.get(k) for k in self.fields})


def trainer_class(controller, lambda_feasibility, action_mapping='soc_feasible', feasibility_supervision=False,
                  legacy_names=False):
    """action_mapping / feasibility_supervision / legacy_names are passed to every model this trainer builds (models.controllers);
    the trainer refuses an environment whose action mapping differs (replay always stores the policy action a)."""
    if controller not in CONTROLLER_KEYS:
        raise ValueError(f'Unknown controller {controller}; choose from {CONTROLLER_KEYS}')
    label = LEGACY_LABELS[controller] if legacy_names else controller
    graph = controller in GNN_TYPES
    if graph:
        from training.trainer import GraphMADDPGTrainer     # needs PyTorch Geometric
    base = GraphMADDPGTrainer if graph else MLPMADDPGTrainer
    factory = partial(build_model, controller, lambda_feasibility, action_mapping=action_mapping,
                      feasibility_supervision=feasibility_supervision, legacy_names=legacy_names)
    bindings = {('GraphAwareMADDPG' if graph else 'MLPMADDPG'): lambda **kw: factory(**kw),
                'build_replay_buffer_from_env_specs': build_bounded_replay_buffer}
    original_init = _rebound_init(base, bindings)

    def __init__(self, env, eval_env=None, config=None, device=None):
        for e in (env, eval_env):
            if e is not None and getattr(e, 'action_mapping', 'rated') != action_mapping:
                raise ValueError(f'environment action_mapping {getattr(e, "action_mapping", "rated")!r} differs from the '
                                 f'trainer action_mapping {action_mapping!r}')
        original_init(self, env, eval_env, config, device)
        folder = Path(self.config.get('diagnostics_dir', Path(self.checkpoint_dir).parent))
        append = bool(self.config.get('diagnostics_append', False))
        self._day_log = _CsvLog(folder / 'training_days.csv', DAY_FIELDS, append)
        self._update_log = _CsvLog(folder / 'training_updates.csv', UPDATE_FIELDS, append)
        self._balance_log = (_CsvLog(folder / 'training_gradient_balance.csv', BALANCE_FIELDS, append)
                             if self.config.get('gradient_balance_log', False) else None)
        self.agent.diagnostics_every = int(self.config.get('diagnostics_every', 100))
        self.agent.diagnostics_callback = self._record_update
        self._day, self._day_count, self._window = None, 0, [0, 0]

    def _step_env(self, env, action):
        phase = 'warmup' if len(self.replay_buffer) < self.warmup_steps else 'policy'
        output = base._step_env(self, env, action)
        if env is self.env:
            next_obs, reward, done, terminated, truncated, info = output
            self.replay_buffer.set_pending_bounds(info['marl_feasible_action_lower'],
                                                  info['marl_feasible_action_upper'], next_obs)
            self._record_step(phase, info, done)
        return output

    def _record_step(self, phase, info, done):
        if self._day is None:
            self._day = dict(day=self._day_count, phase=phase, start_index=getattr(self.env, 'episode_start_index', None),
                             steps=0, raw_out=0, raw_total=0, physical_clipping_steps=0, physical_clipped_kw_total=0.0,
                             backup_activations=0, energy_cost=0.0, grid_import_mwh=0.0, voltage_violation_steps=0,
                             current_violation_steps=0, network_infeasible_steps=0, failed_day=False,
                             completed_day=False, terminal_shortfall_mean=None, terminal_shortfall_max=None,
                             terminal_penalty=0.0)
        d = self._day
        d['steps'] += 1
        raw = np.asarray(info.get('marl_raw_action', []), dtype=float)
        if raw.size:
            outside = int(np.sum(violation(raw, info['marl_feasible_action_lower'], info['marl_feasible_action_upper']) != 0))
            d['raw_out'] += outside
            d['raw_total'] += raw.size
            if phase == 'policy':
                self._window[0] += outside
                self._window[1] += raw.size
        d['physical_clipping_steps'] += int(bool(info.get('physical_clipping_any', False)))
        d['physical_clipped_kw_total'] += float(info.get('physical_clipped_kw_total', 0.0) or 0.0)
        d['backup_activations'] += int(bool(info.get('backup_activated', False)))
        kpis = info.get('kpis') or {}
        if not info.get('episode_failed', False) and 'price' in kpis:
            mwh = max(0.0, float(kpis['grid_import_kw'])) * self.env.delta_t_hours / 1000.0
            d['grid_import_mwh'] += mwh
            d['energy_cost'] += float(kpis['price']) * mwh
        d['voltage_violation_steps'] += int(bool(info.get('executed_voltage_violation') or False))
        d['current_violation_steps'] += int(bool(info.get('executed_current_violation') or False))
        d['network_infeasible_steps'] += int(not bool(info.get('final_network_feasible', info.get('feasible', True))))
        d['failed_day'] = d['failed_day'] or bool(info.get('episode_failed', False))
        if info.get('marl_terminal_completed', False):
            shortfall = np.asarray(info['marl_terminal_soc_shortfall'], dtype=float)
            d.update(completed_day=True, terminal_shortfall_mean=float(shortfall.mean()),
                     terminal_shortfall_max=float(shortfall.max()), terminal_penalty=float(info['marl_terminal_penalty']))
        if done:
            d['global_step_end'] = self.global_step + 1        # collect_step increments global_step after _step_env
            d['raw_action_infeasible_fraction'] = d['raw_out'] / d['raw_total'] if d['raw_total'] else None
            self._day_log.write({k: d.get(k) for k in DAY_FIELDS})
            self._day, self._day_count = None, self._day_count + 1

    def _record_update(self, logs):
        row = dict(logs, global_step=self.global_step)
        row['raw_action_infeasible_fraction_window'] = self._window[0] / self._window[1] if self._window[1] else None
        self._window = [0, 0]
        self._update_log.write(row)
        if self._balance_log is not None and 'gradient_balance' in row:
            self._balance_log.write({k: json.dumps(v) if isinstance(v, list) else v for k, v in row.items()})

    def save_checkpoint(self, path):
        self.agent.experiment_metadata.update(training_step=self.global_step, total_steps=self.total_steps,
                                              exploration=self.exploration_config(), controller=label,
                                              feasibility_formulation=self.agent.feasibility_formulation)
        return base.save_checkpoint(self, path)

    def exploration_config(self):
        return dict(base.exploration_config(self), controller=label)

    name = ('GraphController' if graph else 'MLPController') + 'Trainer'   # 'Graph' in the name selects the GNN validation label
    return type(name, (base,), dict(__init__=__init__, _step_env=_step_env, _record_step=_record_step,
                                    _record_update=_record_update, save_checkpoint=save_checkpoint,
                                    exploration_config=exploration_config, controller=label))
