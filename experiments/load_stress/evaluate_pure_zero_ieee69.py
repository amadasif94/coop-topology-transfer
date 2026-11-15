"""IEEE69 PURE zero-action reference: all nine batteries at exactly 0 kW, no backup intervention.

One load scale per call, TP1-TP10, all 31 December test days, with the A0 final-test references.

    python -u -m experiments.load_stress.evaluate_pure_zero_ieee69 --scale {1.0,1.1,1.2,1.3} \
        --job-map <cluster_job_map_final.json> [--smoke]

References exactly as the IEEE69 load-stress evaluator: ORIGINAL ieee69 protocol.verify_map, then
final_evaluation.references() (one A0 manifest per topology group); every topology TP1-TP10 must be present, all
groups share one checkpoint and selection. The zero reference does not depend on training, so run 0's A0
references define the plan; each topology reuses its archived A0 network bytes, start indices, seeds
(manifest seed + episode id) and December timestamps.
Checks per topology:
  - a backup-enabled control environment passes the ORIGINAL IEEE69 comparison against A0 (only
    load_scaling_factor may differ), and the pure environment's record differs from it only in the backup settings;
  - every dispatch step executes exactly 0 kW on all nine batteries, unchanged, with one control power flow.
--smoke: 1.3x, TP1 and TP10, first 2 days. Also runs the same days in the backup-enabled control environment and
         requires the control's pre-backup (0 kW) power flow to equal the pure run's on every step; writes
         smoke_check.json. Full runs record its status: absent -> NOT_PERFORMED; an existing FAIL is refused.
Outputs: results/load_stress/ieee69/pure_zero_no_backup/load_x<pct>[_smoke]/ ; nonempty directories are refused.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd

from configs.ieee69_config import IEEE69_CONFIG
from experiments.load_stress import evaluate_load_stress_ieee69 as E69
from experiments.load_stress.evaluate_pure_zero_ieee33 import PureZeroError, check_steps, compare_to_control
from experiments.load_stress.pure_zero_env import CHANGED_KEYS, LABEL, NOTE
from experiments.load_stress.pure_zero_env_ieee69 import make_pure_zero_env_ieee69
from experiments.load_stress.stress_env import ALLOWED_SCALES, PV_SCALE, check_scale, scale_label, scenario
from experiments.load_stress.stress_env_ieee69 import compare_stress_environment_ieee69, make_stress_env_ieee69
from experiments.load_stress.validate_pure_zero_ieee33 import compare as compare_initial_pf
from experiments.no_gnn_maddpg.run_tp_eval_mlp_ieee69 import evaluate_one_episode, summarize
from experiments.no_shared_kpi.ieee69 import protocol as p69
from experiments.no_shared_kpi.ieee69.runner import RunManifest, energy_rows
from experiments.no_shared_kpi.protocol import ROOT, read_json, sha, write_json
from utility.evaluation_protocol import authorize, protocol_config

TOPOLOGIES, DAYS, STEPS = E69.TOPOLOGIES, E69.DAYS, E69.STEPS
DEFAULT_OUTPUT_ROOT = E69.OUTPUT_BASE / LABEL
SMOKE_SCALE, SMOKE_TOPOLOGIES, SMOKE_EPISODES = 1.3, ['TP1', 'TP10'], 2
REFERENCE_RUN = 0


def run_dir(output_root, scale, smoke=False):
    return Path(output_root) / (scale_label(scale) + ('_smoke' if smoke else ''))


def check_output_root(output_root, protected):
    """IEEE69 load-stress protections plus every A0/A1/A2 IEEE69 stress tree."""
    labels = [ROOT / E69.OUTPUT_BASE / label for label, _ in E69.MODELS.values()]
    return E69.check_output_root_ieee69(output_root, [*protected, *labels])


def smoke_status(output_root):
    path = run_dir(output_root, SMOKE_SCALE, smoke=True) / 'smoke_check.json'
    if not path.is_file():
        print(f'WARNING: smoke check not performed ({path} absent); recorded as NOT_PERFORMED', flush=True)
        return dict(performed=False, status='NOT_PERFORMED', path=None, sha256=None)
    report = read_json(path)
    if report.get('status') != 'PASS':
        raise PureZeroError(f'Smoke check {path} has status {report.get("status")}; resolve it before full runs')
    if report.get('evaluation_manifest_sha256') != sha(path.parent / 'run_manifest.json'):
        raise PureZeroError('Smoke check: its run manifest changed after the check')
    return dict(performed=True, status='PASS', path=str(path.resolve()), sha256=sha(path))


def topology_config(ref, protocol):
    cfg = deepcopy(IEEE69_CONFIG)
    cfg['paths'].update(ref['files'])            # read-only exact archived A0 network bytes
    cfg['topology_case'] = dict(case_name=ref['entry']['topology'], **ref['case'])
    return protocol_config(cfg, protocol)


def run_days(env, tp, plan):
    rows = []
    for ep, start, seed in plan:
        rows.extend(evaluate_one_episode(env, None, 'zero', tp, ep, start, seed))
    return pd.DataFrame(rows)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scale', required=True, type=float, choices=ALLOWED_SCALES)
    p.add_argument('--job-map', required=True, type=Path)
    p.add_argument('--smoke', action='store_true', help=f'{SMOKE_SCALE}x, {SMOKE_TOPOLOGIES}, first {SMOKE_EPISODES} days')
    p.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    a = p.parse_args(argv)
    a.scale = check_scale(a.scale)
    if a.smoke and a.scale != SMOKE_SCALE:
        p.error(f'--smoke uses scale {SMOKE_SCALE}')
    return a


def main(argv=None):
    args = parse_args(argv)
    entry = p69.verify_map(args.job_map, REFERENCE_RUN)
    refs, a0_selection, groups = E69.a0_references(entry, REFERENCE_RUN)
    topologies = SMOKE_TOPOLOGIES if args.smoke else TOPOLOGIES
    protocol = authorize(mode='test', final=True, run_purpose='final_test', learned=False,
                         topologies=topologies, selection=None)
    output_root = check_output_root(args.output_root, {Path(g).parent for g in groups})
    gates = {} if args.smoke else dict(smoke=smoke_status(output_root))

    out = run_dir(output_root, args.scale, args.smoke)
    run = RunManifest(out, 'ZERO', refs['TP1']['manifest']['seed'], 'test', None, vars(args))
    run.record_protocol(protocol, ['zero'])
    run.data['pure_zero'] = dict(feeder='IEEE69', label=LABEL, policy='zero', load_scaling_factor=args.scale,
        pv_scaling_factor=PV_SCALE, scenario=scenario(args.scale), smoke=bool(args.smoke), topologies=topologies,
        gates=gates, reference_run=REFERENCE_RUN, a0_topology_groups={g: tps for g, tps in groups.items()},
        a0_references={tp: dict(path=str(Path(ref['path']).resolve()), sha256=sha(ref['path'])) for tp, ref in refs.items()},
        a0_selection_sha256=a0_selection.get('selection_sha256'),
        job_map=dict(path=str(args.job_map.resolve()), sha256=sha(args.job_map)),
        backup='disabled for this reference only', changed_settings=[list(k) for k in CHANGED_KEYS],
        doe='forced off by the ORIGINAL primary_config (raw IEEE69 config has doe_enabled=True), as in A0',
        robust_pf_instance_override=True,
        failure_rule='a failed power flow ends that day; the day is counted as failed, never dropped',
        note=NOTE, post_hoc=E69.POST_HOC, environment_comparisons={}, control_comparisons={}, step_checks={},
        smoke_initial_pf_checks={})
    run._write()
    print('PURE_ZERO_IEEE69', json.dumps(dict(scale=args.scale, smoke=args.smoke, out=str(out))), flush=True)

    frames, energies, controls = [], [], []
    for tp in topologies:
        ref = refs[tp]
        cfg = topology_config(ref, protocol)
        seed = ref['manifest']['seed']
        control = make_stress_env_ieee69(cfg, 'test', seed, args.scale)
        run.add_environment(control, topology=tp, scenario=scenario(args.scale), intended_episodes=DAYS)
        control_record = run.data['environments'].pop()
        run.data['pure_zero']['environment_comparisons'][tp] = compare_stress_environment_ieee69(
            control_record, ref['manifest'], args.scale)
        env = make_pure_zero_env_ieee69(cfg, 'test', seed, args.scale)
        if [s for _, s, _ in ref['plan']] != list(env.episode_start_indices) or len(ref['plan']) != DAYS:
            raise PureZeroError(f'{tp}: A0 episode indices differ from the current December dataset')
        run.add_environment(env, topology=tp, primary=False, scenario=scenario(args.scale), intended_episodes=DAYS)
        run.data['pure_zero']['control_comparisons'][tp] = compare_to_control(run.data['environments'][-1], control_record)
        plan = ref['plan'][:SMOKE_EPISODES] if args.smoke else ref['plan']
        run.data.setdefault('episode_plans', {})[tp] = [dict(episode_id=e, start_index=s, seed=z) for e, s, z in plan]
        run._write()
        frame = run_days(env, tp, plan)
        run.data['pure_zero']['step_checks'][tp] = check_steps(frame, env.num_agents)
        expected = pd.to_datetime(ref['timestamps'][:len(plan) * STEPS], utc=True)
        if frame.episode_completed.all() and not np.array_equal(pd.to_datetime(frame.date_time, utc=True), expected):
            raise PureZeroError(f'{tp}: timestamps differ from the A0 final reference')
        if args.smoke:
            reference = run_days(control, tp, plan)
            table, coverage = compare_initial_pf(frame, reference, 1e-9)
            ok = (table['status'] == 'PASS').all() and coverage['pure_violating_steps'] == coverage['reference_backup_activations'] \
                and not coverage['only_in_pure'] and not coverage['only_in_reference']
            run.data['pure_zero']['smoke_initial_pf_checks'][tp] = dict(coverage, status='PASS' if ok else 'FAIL',
                failing_metrics=table.loc[table.status != 'PASS', 'metric'].tolist())
            controls.append(reference.assign(load_scaling_factor=args.scale))
        frame['run_id'], frame['model'], frame['load_scaling_factor'] = REFERENCE_RUN, LABEL, args.scale
        frames.append(frame)
        energies.append(energy_rows(frame, env))

    steps = pd.concat(frames, ignore_index=True)
    ep, agg = summarize(steps)
    energy = pd.concat(energies, ignore_index=True)
    ep = ep.merge(energy, on=['topology_case', 'episode_id', 'start_index'], validate='one_to_one')
    extra = ep.groupby('topology_case')[['raw_energy_cost', 'adjusted_energy_cost', 'terminal_stored_energy_kwh',
        'mean_backup_adjustment_kw', 'mean_physical_clipping_kw']].mean().add_prefix('mean_').reset_index()
    agg = agg.merge(extra, on='topology_case', validate='one_to_one')
    for df in (ep, agg, energy):
        df['run_id'], df['model'], df['load_scaling_factor'] = REFERENCE_RUN, LABEL, args.scale
    tables = [('step_metrics', steps), ('episode_summary', ep), ('aggregate_summary', agg),
              ('terminal_soc_summary', energy), ('adjusted_cost_summary', energy)]
    if controls:
        tables.append(('smoke_control_step_metrics', pd.concat(controls, ignore_index=True)))
    for name, df in tables:
        df.to_csv(out / f'{name}.csv', index=False)

    complete = ep['complete'].astype(bool)
    run.data['pure_zero']['episodes'] = dict(planned=int(sum(len(v) for v in run.data['episode_plans'].values())),
        evaluated=int(len(ep)), complete=int(complete.sum()), failed_days=int((~complete).sum()))
    run.finish()
    if args.smoke:
        checks = run.data['pure_zero']['smoke_initial_pf_checks']
        status = 'PASS' if len(ep) == SMOKE_EPISODES * len(topologies) and all(c['status'] == 'PASS' for c in checks.values()) else 'FAIL'
        write_json(out / 'smoke_check.json', dict(status=status, scale=args.scale, topologies=topologies,
            episodes=int(len(ep)), complete_episodes=int(complete.sum()), failed_days=int((~complete).sum()),
            initial_pf_checks=checks, step_checks=run.data['pure_zero']['step_checks'],
            evaluation_manifest_sha256=sha(out / 'run_manifest.json')))
        print('SMOKE', status, flush=True)
        if status != 'PASS':
            raise SystemExit(1)
    print('PURE_ZERO_IEEE69_DONE', out, json.dumps(run.data['pure_zero']['episodes']), flush=True)
    return out


if __name__ == '__main__':
    main()
