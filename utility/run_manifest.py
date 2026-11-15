"""Owned, non-overwriting run directories and manifests of instantiated environments."""
from datetime import datetime, timezone
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from environments.primary_marl import validate_primary_env, validate_network, effective_current_ratings_a
from utility.feasibility import VOLTAGE_TOLERANCE_PU, CURRENT_RELATIVE_TOLERANCE
from utility.provenance import file_record, input_files, normalization_identity, interface_identity, source_files


def serial(value):
    if isinstance(value, dict): return {str(k): serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [serial(v) for v in value]
    if isinstance(value, np.ndarray): return serial(value.tolist())
    if isinstance(value, np.generic): return serial(value.item())
    if isinstance(value, Path): return str(value)
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def reserve_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'Refusing to reuse nonempty result directory: {path}')
    path.mkdir(parents=True, exist_ok=True)
    # Exclusive reservation protects simultaneous launches with the same output.
    with (path/'.run_reserved').open('x') as f:
        f.write(datetime.now(timezone.utc).isoformat())
    return path


def validate_optimization_compatibility(env, baseline):
    """Reject physical drift; do not change optimization equations/configuration."""
    if list(env.dess_buses) != list(baseline.DESS_BUSES):
        raise ValueError('Optimization battery bus ordering differs from DESSEnv')
    names = dict(capacity_kwh='BATTERY_CAPACITY_KWH', max_charge_kw='P_CH_MAX_KW',
        max_discharge_kw='P_DIS_MAX_KW', charge_efficiency='ETA_CH',
        discharge_efficiency='ETA_DIS', soc_min='SOC_MIN', soc_max='SOC_MAX', soc_init='SOC0')
    for battery in env.batteries:
        for field, constant in names.items():
            if not np.isclose(getattr(battery.config, field), getattr(baseline, constant), rtol=0, atol=1e-9):
                raise ValueError(f'Optimization {constant} differs from DESSEnv battery {field}')
    for actual, expected, label in ((env.delta_t_hours, baseline.DT_HOURS, 'timestep'),
        (env.v_min, baseline.V_MIN, 'Vmin'), (env.v_max, baseline.V_MAX, 'Vmax'),
        (env.grid_export_limit_kw, baseline.GRID_EXPORT_LIMIT_KW, 'export limit')):
        if not np.isclose(actual, expected, rtol=0, atol=1e-9):
            raise ValueError(f'Optimization {label} differs from DESSEnv')


def source_revision():
    """(revision or None, status). Only a missing/unlaunchable/hung git is tolerated."""
    try:
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f'WARNING: git unavailable ({type(exc).__name__}: {exc}); recording source_revision=null',
              file=sys.stderr, flush=True)
        return None, 'git_unavailable'
    if revision.returncode != 0:
        print('WARNING: git rev-parse failed (exit %d); recording source_revision=null' % revision.returncode,
              file=sys.stderr, flush=True)
        return None, 'git_failed'
    return revision.stdout.strip(), 'ok'


class RunManifest:
    def __init__(self, output_dir, controller, seed, mode, checkpoint=None, arguments=None):
        self.directory = reserve_directory(output_dir)
        self.path = self.directory/'run_manifest.json'
        revision, revision_status = source_revision()
        self.data = dict(schema_version=1, controller=str(controller).upper(), seed=int(seed) if seed is not None else None,
            mode=mode, checkpoint=str(checkpoint) if checkpoint else None,
            source_revision=revision, source_revision_status=revision_status,
            started_at_utc=datetime.now(timezone.utc).isoformat(), result_directory=str(self.directory.resolve()),
            arguments=serial(arguments or {}), environments=[], status='started')
        if checkpoint:
            # Identity of the evaluated weights (F14): sha256/size/mtime of the checkpoint file.
            self.data['checkpoint_file'] = file_record(checkpoint)
        self._write()

    def _write(self):
        # Only this RunManifest instance owns this reserved path.
        temp = self.path.with_suffix('.tmp')
        temp.write_text(json.dumps(serial(self.data), indent=2, allow_nan=False), encoding='utf-8')
        temp.replace(self.path)

    def add_environment(self, env, topology='TP1', primary=True, scenario='production', intended_episodes=None):
        (validate_primary_env if primary else validate_network)(env, scenario)
        # The stated topology must agree with the environment actually built (F14).
        actual_topology = (env.config.get('topology_case') or {}).get('case_name', 'TP1')
        if str(topology).upper() != str(actual_topology).upper():
            raise ValueError(f'Manifest topology {topology!r} contradicts the environment topology {actual_topology!r}')
        amps = effective_current_ratings_a(env)
        cfg = serial(env.config)
        # sha256 of EVERY configured input file: network CSVs incl. the PF line file
        # (gridtensor_line.csv), DESS placement, processed time series, graph data.
        inputs = input_files(env)
        entry = dict(feeder=f'IEEE{env.num_buses}', topology=topology, mode=env.mode,
            intended_episodes=intended_episodes, expected_steps_per_episode=env.episode_length,
            available_episode_start_indices=[int(i) for i in env.episode_start_indices],
            scenario=scenario, primary_marl=primary, input_files=inputs,
            configuration_sha256=hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest(),
            data_scaling=cfg.get('data_scaling'),
            agent_obs_dim=int(env.agent_obs_dim),
            agent_obs_features=list(getattr(env, 'agent_obs_features', [])),
            node_feature_dim=int(env.observation_space['x'].shape[-1]),
            node_features=list(cfg.get('graph', {}).get('node_features', [])),
            normalization_identity=normalization_identity(env),
            interface_identity=interface_identity(env),
            source_files=source_files(),
            doe=dict(enabled=env.doe_enabled, w=env.doe_w, delta_c=env.doe_delta_c,
                     pf_failure_mode=env.doe_pf_failure_mode,
                     penalty_coef=getattr(env, 'doe_penalty_coef', 0.0)),
            penalty_rule='physical limits: none; DOE: -coef*|physical-doe|/rated; backup: penalty on every activation',
            backup_mode=env.backup_mode, backup=env.backup_cfg,
            voltage_tolerance_pu=VOLTAGE_TOLERANCE_PU,
            current_relative_tolerance=CURRENT_RELATIVE_TOLERANCE,
            network=cfg.get('network'), reward=env.reward_cfg,
            battery_constraints=[asdict(b.config) if is_dataclass(b.config) else vars(b.config) for b in env.batteries],
            active_current_ratings_a=amps.tolist(), min_current_rating_a=float(amps.min()),
            max_current_rating_a=float(amps.max()),
            action_processing='policy/controller -> physical limits -> optional DOE -> configured AC backup -> execution')
        self.data['environments'].append(entry)
        self._write()
        print('RUN_MANIFEST', self.path, 'environment:', entry['feeder'], topology, env.mode, flush=True)

    def record_schedule(self, schedule):
        """F16: the fixed validation episode schedule (ids, start indices, seeds, sha256)."""
        from utility.validation_schedule import schedule_summary
        self.data['validation_schedule'] = schedule_summary(schedule)
        self._write()

    def record_protocol(self, protocol, policies=()):
        """F2: run purpose, data split, final-test flag and per-policy selection provenance."""
        from utility.evaluation_protocol import selection_provenance
        self.data['evaluation_protocol'] = dict(protocol, selection_by_policy=selection_provenance(protocol, policies))
        self._write()

    def record_exploration(self, exploration):
        """F9: effective exploration settings and seed of a training run."""
        self.data['exploration'] = exploration
        self._write()

    def finish(self):
        self.data.update(status='finished', finished_at_utc=datetime.now(timezone.utc).isoformat())
        self._write()
