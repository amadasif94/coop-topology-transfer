"""IEEE33 PURE zero-action reference: all five batteries at exactly 0 kW, no backup intervention.

One load scale per call, TP1-TP7, all 31 December test days, with the A0 final-test episode plan.

    python -u -m experiments.load_stress.evaluate_pure_zero_ieee33 --scale {1.0,1.1,1.2,1.3} \
        --job-map <job_map.json> [--pilot]

Separate from the existing zero-action + backup runs (results/load_stress/ieee33/zero_action/), which stay valid
and untouched. Environment: pure_zero_env.make_pure_zero_env (backup off, dispatch executed as-is). Checks:
  - the job map, A0 final manifest and episode plan exactly as the zero-action load-stress run;
  - per topology, a control environment WITH backup passes the original compare_stress_environment against A0,
    and the pure environment's record differs from that control only in the backup settings;
  - every dispatch step executes exactly 0 kW on every battery, unchanged, with one control power flow.
Evaluation is the original evaluate_one_episode(env, None, 'zero', ...) with summarize and runner.energy_rows.
--pilot: TP1, first 2 days, into load_x<pct>_pilot (informational only).
Outputs: <output-root>/load_x<pct>[_pilot]/ ; directories that are not empty are refused.
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from experiments.load_stress import evaluate_load_stress_ieee33 as E
from experiments.load_stress.pure_zero_env import CHANGED_KEYS, LABEL, NOTE, make_pure_zero_env
from experiments.load_stress.summarize_load_stress import as_bool
from experiments.load_stress.stress_env import (ALLOWED_SCALES, PV_SCALE, check_scale, compare_stress_environment,
    make_stress_env, scale_label, scenario)
from experiments.no_gnn_maddpg.run_tp_eval_mlp import evaluate_one_episode, summarize
from experiments.no_shared_kpi.protocol import read_json, sha
from experiments.no_shared_kpi.runner import energy_rows
from experiments.topology_generalization.topology_utils import build_topology_config
from utility.evaluation_protocol import authorize, episode_plan, protocol_config
from utility.run_manifest import RunManifest

TOPOLOGIES = E.TOPOLOGIES
DAYS, STEPS = 31, 96
DEFAULT_OUTPUT_ROOT = E.DEFAULT_OUTPUT_ROOT / LABEL
PILOT_TOPOLOGY, PILOT_EPISODES = 'TP1', 2
# Record keys allowed to differ between the pure environment and its backup-enabled control.
RECORD_DIFFERENCES = {'primary_marl', 'configuration_sha256', 'backup_mode', 'backup'}


class PureZeroError(RuntimeError):
    pass


def run_dir(output_root, scale, pilot=False):
    return Path(output_root) / (scale_label(scale) + ('_pilot' if pilot else ''))


def check_output_root(output_root, protected):
    """IEEE33 protections plus every existing load-stress model tree (zero_action included)."""
    labels = [E.DEFAULT_OUTPUT_ROOT / label for label, _ in E.MODELS.values()]
    return E.check_output_root(output_root, [*protected, *labels])


def compare_to_control(record, control):
    diff = sorted(k for k in set(record) | set(control) if record.get(k) != control.get(k))
    if [k for k in diff if k not in RECORD_DIFFERENCES]:
        raise PureZeroError(f'Pure zero environment differs from its backup-enabled control in {diff}')
    backup = sorted(k for k in set(record['backup']) | set(control['backup'])
                    if record['backup'].get(k) != control['backup'].get(k))
    if backup != ['backup_mode']:
        raise PureZeroError(f'Backup settings differ beyond backup_mode: {backup}')
    return dict(record_differences=diff, backup_differences=backup,
                control_configuration_sha256=control['configuration_sha256'],
                pure_configuration_sha256=record['configuration_sha256'])


def check_steps(frame, num_agents):
    """Every dispatch step: exactly 0 kW on every battery, requested == accepted == 0, one control PF."""
    steps = frame[frame['record_type'] == 'dispatch_attempt'] if 'record_type' in frame else frame
    cols = [f'{k}_agent_{i}' for k in ('requested_action', 'accepted_action', 'dess_power_kw') for i in range(num_agents)]
    values = steps[cols].to_numpy(dtype=float)
    problems = dict(nonzero_battery_values=int(np.count_nonzero(values != 0)),
                    backup_activations=int(as_bool(steps['backup_activated']).sum()),
                    corrected_steps=int((steps['backup_direction'] != 'unchanged').sum()),
                    extra_control_pf=int((pd.to_numeric(steps['backup_pf_evaluations']) != 1).sum()))
    if any(problems.values()):
        raise PureZeroError(f'Pure zero dispatch was not exactly 0 kW and unchanged: {problems}')
    return dict(problems, dispatch_steps=int(len(steps)))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scale', required=True, type=float, choices=ALLOWED_SCALES)
    p.add_argument('--job-map', required=True, type=Path)
    p.add_argument('--pilot', action='store_true', help=f'{PILOT_TOPOLOGY}, first {PILOT_EPISODES} days (informational)')
    p.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    a = p.parse_args(argv)
    a.scale = check_scale(a.scale)
    return a


def main(argv=None):
    args = parse_args(argv)
    job = read_json(args.job_map)
    # The zero baseline does not depend on the training run: run 0's A0 final test defines the plan.
    baseline, entry, _, a0_path, info = E.resolve_references(
        SimpleNamespace(run=0, model='zero', reference_manifest=None), job)
    ref = read_json(a0_path)
    protocol = authorize(mode='test', final=True, run_purpose='final_test', learned=False,
                         topologies=TOPOLOGIES, selection=None)
    E.verify_reference(ref, protocol, 'zero', 0, None)
    output_root = check_output_root(args.output_root, {Path(info['a0_evaluation_manifest']['path']).parent})
    seed = int(baseline['seed'])
    topologies = [PILOT_TOPOLOGY] if args.pilot else TOPOLOGIES

    out = run_dir(output_root, args.scale, args.pilot)
    run = RunManifest(out, 'ZERO', seed, 'test', None, vars(args))
    run.record_protocol(protocol, ['zero'])
    run.data['pure_zero'] = dict(info, feeder='IEEE33', label=LABEL, policy='zero', load_scaling_factor=args.scale,
        pv_scaling_factor=PV_SCALE, scenario=scenario(args.scale), pilot=bool(args.pilot), topologies=topologies,
        backup='disabled for this reference only', changed_settings=[list(k) for k in CHANGED_KEYS],
        robust_pf_instance_override=True,
        failure_rule='a failed power flow ends that day; the day is counted as failed, never dropped',
        note=NOTE, post_hoc=E.POST_HOC, environment_comparisons={}, control_comparisons={}, step_checks={})
    run._write()
    print('PURE_ZERO', json.dumps(dict(scale=args.scale, pilot=args.pilot, out=str(out))), flush=True)

    frames, energy = [], []
    for tp in topologies:
        cfg = protocol_config(build_topology_config(case_name=tp, output_root=out / 'network_inputs', validate=True), protocol)
        control = make_stress_env(cfg, 'test', seed, args.scale)
        run.add_environment(control, topology=tp, scenario=scenario(args.scale), intended_episodes=DAYS)
        control_record = run.data['environments'].pop()
        run.data['pure_zero']['environment_comparisons'][tp] = compare_stress_environment(control_record, baseline, args.scale)
        env = make_pure_zero_env(cfg, 'test', seed, args.scale)
        starts, seeds, schedule = episode_plan(env, 'all', seed, protocol)
        if schedule is not None or len(starts) != DAYS or env.episode_length != STEPS:
            raise PureZeroError('Expected 31 December days x 96 steps')
        ref_starts, ref_plan = E.reference_plan(ref, tp)
        plan = [dict(start_index=int(s), seed=int(z)) for s, z in zip(starts, seeds)]
        if [int(s) for s in starts] != ref_starts or (ref_plan is not None and ref_plan != plan):
            raise PureZeroError(f'{tp}: episode plan differs from the A0 final test')
        run.add_environment(env, topology=tp, primary=False, scenario=scenario(args.scale), intended_episodes=len(starts))
        run.data['pure_zero']['control_comparisons'][tp] = compare_to_control(run.data['environments'][-1], control_record)
        if args.pilot:
            plan = plan[:PILOT_EPISODES]
        run.data.setdefault('episode_plans', {})[tp] = plan
        run._write()
        rows = []
        for ep, item in enumerate(plan):
            rows.extend(evaluate_one_episode(env, None, 'zero', tp, ep, item['start_index'], item['seed']))
        frame = pd.DataFrame(rows)
        run.data['pure_zero']['step_checks'][tp] = check_steps(frame, env.num_agents)
        frame['run_id'], frame['model'], frame['load_scaling_factor'] = 0, LABEL, args.scale
        frames.append(frame)
        energy.append(energy_rows(frame, env))

    steps = pd.concat(frames, ignore_index=True)
    ep, agg = summarize(steps)
    energies = pd.concat(energy, ignore_index=True)
    ep = ep.merge(energies, on=['topology_case', 'episode_id', 'start_index'], validate='one_to_one')
    extra = ep.groupby('topology_case')[['raw_energy_cost', 'adjusted_energy_cost', 'terminal_stored_energy_kwh',
        'mean_backup_adjustment_kw', 'mean_physical_clipping_kw']].mean().add_prefix('mean_').reset_index()
    agg = agg.merge(extra, on='topology_case', validate='one_to_one')
    for df in (ep, agg, energies):
        df['run_id'], df['model'], df['load_scaling_factor'] = 0, LABEL, args.scale
    for name, df in [('step_metrics', steps), ('episode_summary', ep), ('aggregate_summary', agg),
                     ('terminal_soc_summary', energies), ('adjusted_cost_summary', energies)]:
        df.to_csv(out / f'{name}.csv', index=False)

    complete = ep['complete'].astype(bool)
    run.data['pure_zero']['episodes'] = dict(planned=int(sum(len(v) for v in run.data['episode_plans'].values())),
        evaluated=int(len(ep)), complete=int(complete.sum()), failed_days=int((~complete).sum()))
    run.finish()
    print('PURE_ZERO_DONE', out, json.dumps(run.data['pure_zero']['episodes']), flush=True)
    return out


if __name__ == '__main__':
    main()
