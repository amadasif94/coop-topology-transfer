"""IEEE69 load-stress evaluation of the FROZEN A0/A1/A2 checkpoints (no zero-action baseline).

One controller x run x load scale per call, on TP1-TP10 and all 31 December test days, reusing the existing
frozen IEEE69 selections. Post-hoc sensitivity analysis: never used for model choice.

    python -u -m experiments.load_stress.evaluate_load_stress_ieee69 \
        --model {a0,a1,a2} --run {0,1,2} --scale {1.0,1.1,1.2,1.3} [--pilot] --job-map <cluster_job_map_final.json>

References: the job map is checked by the ORIGINAL ieee69 protocol.verify_map (sha256 of every A0 manifest),
then the ORIGINAL final_evaluation.references() reads the A0 final manifests (one per topology group, e.g.
TP1-TP7, TP8-TP9, TP10). Every topology TP1-TP10 must be present with 31 complete December days; its archived
A0 network snapshot, episode plan (start index, seed + episode id) and timestamps are reused exactly, as the
IEEE69 ablation final evaluation does. All A0 groups must share one checkpoint and one frozen selection.

Checks (all original): authorize/verify_selection, selection model/run/feeder, checkpoint specification,
run id and A0 training identity, A0 MLPMADDPG.load or the IEEE69 A1/A2 evaluation loader (sidecar sha,
source inventory, provenance), the IEEE69 compare_environment against A0 (only load_scaling_factor may
differ), A0 start indices and timestamps.

Optional, never blocking:
  scale 1.0   identity rerun, checked by validate_load_stress_identity_ieee69.py. Absent -> stress runs record
              identity NOT_PERFORMED / identity_verified=false. An existing report that is not a PASS
              (or PASS_PARTIAL) for this checkpoint and 1.0x manifest is refused, never ignored.
  --pilot     TP1, first 2 days, a stressed scale; writes pilot_check.json; full runs only record its status.
Outputs: results/load_stress/ieee69/<label>/run_<r>/load_x<pct>[_pilot]/ ; nonempty directories are refused.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from configs.ieee69_config import IEEE69_CONFIG
from experiments.load_stress.evaluate_load_stress_ieee33 import (NORMALIZATION_NOTE, POST_HOC, ObservationRecorder,
    StressError, check_output_root)
from experiments.load_stress.stress_env import ALLOWED_SCALES, PV_SCALE, check_scale, scale_label, scenario
from experiments.load_stress.stress_env_ieee69 import compare_stress_environment_ieee69, make_stress_env_ieee69
from experiments.no_gnn_maddpg.run_tp_eval_mlp_ieee69 import MLP_POLICY_NAME, build_agent, evaluate_one_episode, summarize
from experiments.no_shared_kpi.ieee69 import protocol as p69
from experiments.no_shared_kpi.ieee69.final_evaluation import references
from experiments.no_shared_kpi.ieee69.models import model_class
from experiments.no_shared_kpi.ieee69.runner import RunManifest, energy_rows, namespace
from experiments.no_shared_kpi.protocol import ROOT, VARIANTS, read_json, sha, write_json
from utility.evaluation_protocol import authorize, protocol_config

TOPOLOGIES = [f'TP{i}' for i in range(1, 11)]
MODELS = dict(a0=('a0_original', None), a1=('a1_no_shared_kpi', VARIANTS[0]),
              a2=('a2_no_centralized_critic', VARIANTS[1]))
OUTPUT_BASE = Path('results/load_stress/ieee69')
PILOT_EPISODES, PILOT_TOPOLOGY = 2, 'TP1'
DAYS, STEPS = 31, 96
IDENTITY_ACCEPTED = ('PASS', 'PASS_PARTIAL')


def run_dir(output_root, model, run, scale, pilot=False):
    return Path(output_root) / MODELS[model][0] / f'run_{run}' / (scale_label(scale) + ('_pilot' if pilot else ''))


def check_output_root_ieee69(output_root, protected, base=None):
    """IEEE33 protections plus: only under results/load_stress/ieee69, never inside the IEEE33 stress tree."""
    base = Path(base or ROOT / OUTPUT_BASE).resolve()
    out = check_output_root(output_root, [*protected, ROOT / 'results/load_stress/ieee33'])
    if not (out == base or out.is_relative_to(base)):
        raise StressError(f'IEEE69 load-stress outputs must be under {base}; got {out}')
    return out


# ------------------------------------------------------------------------- optional checks
def identity_status(output_root, model, run, checkpoint_sha):
    d = run_dir(output_root, model, run, 1.0)
    path = d / 'identity_report.json'
    if not path.is_file():
        print(f'WARNING: optional 1.0x identity check not performed ({path} absent); recorded as NOT_PERFORMED', flush=True)
        return dict(performed=False, status='NOT_PERFORMED', path=None, sha256=None)
    report = read_json(path)
    if report.get('status') not in IDENTITY_ACCEPTED:
        raise StressError(f'Identity report {path} has status {report.get("status")}; resolve it before stress runs')
    if report.get('evaluation_manifest_sha256') != sha(d / 'run_manifest.json'):
        raise StressError('Identity report: the 1.0x run manifest changed after validation')
    if report.get('checkpoint_sha256') != checkpoint_sha:
        raise StressError('Identity report was produced for a different checkpoint')
    return dict(performed=True, status=report['status'], path=str(path.resolve()), sha256=sha(path),
                not_observable_topologies=report.get('not_observable_topologies', []))


def pilot_status(output_root, model, run, checkpoint_sha):
    """Informational only: the latest pilot (any stressed scale) for this checkpoint, or NOT_PERFORMED."""
    found = []
    for s in ALLOWED_SCALES[1:]:
        path = run_dir(output_root, model, run, s, pilot=True) / 'pilot_check.json'
        if path.is_file():
            r = read_json(path)
            found.append(dict(scale=s, status=r.get('status'), path=str(path.resolve()), sha256=sha(path),
                              same_checkpoint=r.get('checkpoint_sha256') == checkpoint_sha))
    return found or [dict(status='NOT_PERFORMED')]


# ------------------------------------------------------------------------------ references
def a0_references(entry, run):
    records = entry.get('baseline_evaluation_manifests') or []
    if not records:
        raise StressError('Job map has no IEEE69 A0 final evaluation manifests (baseline_evaluation_manifests)')
    refs = references([r['path'] for r in records], run)
    missing = [tp for tp in TOPOLOGIES if tp not in refs]
    if missing:
        raise StressError(f'A0 final references lack {missing}; all of TP1-TP10 are required')
    selections = {json.dumps(ref['manifest']['evaluation_protocol'].get('selection'), sort_keys=True) for ref in refs.values()}
    if len(selections) != 1:
        raise StressError('A0 topology groups were evaluated with different frozen selections')
    groups = {}
    for tp, ref in refs.items():
        groups.setdefault(str(Path(ref['path']).resolve()), []).append(tp)
    return {tp: refs[tp] for tp in TOPOLOGIES}, json.loads(selections.pop()), groups


def verify_ablation_reference(ref, protocol, variant, run, plans):
    p = ref.get('evaluation_protocol') or {}
    if ref.get('status') != 'finished' or p.get('data_split') != 'test' or p.get('final_test') is not True:
        raise StressError('A1/A2 reference is not a finished authorized December final test')
    if (p.get('selection') or {}).get('selection_sha256') != protocol['selection']['selection_sha256']:
        raise StressError('A1/A2 reference used a different frozen selection record')
    if (ref.get('checkpoint_file') or {}).get('sha256') != protocol['selection']['checkpoint_sha256']:
        raise StressError('A1/A2 reference evaluated a different checkpoint')
    if ref.get('ablation') != variant or int(ref.get('run_id', -1)) != run:
        raise StressError('A1/A2 reference belongs to a different ablation/run')
    checks = {}
    for tp, plan in plans.items():
        recorded = (ref.get('episode_plans') or {}).get(tp)
        if recorded is None:
            checks[tp] = 'not_in_reference'
        elif recorded != plan:
            raise StressError(f'{tp}: A1/A2 reference episode plan differs from the A0 reference plan')
        else:
            checks[tp] = 'matches'
    return checks


# --------------------------------------------------------------------------------- models
def load_a0(env, checkpoint, a0_manifest, device):
    a = a0_manifest.get('arguments') or {}
    ns = SimpleNamespace(gamma=a.get('gamma', 0.99), tau=a.get('tau', 0.005), actor_lr=a.get('actor_lr', 1e-4),
                         critic_lr=a.get('critic_lr', 1e-3), share_actor=bool(a.get('share_actor', False)),
                         checkpoint=str(checkpoint))
    return build_agent(env, ns, device), 'MLPMADDPG.load (original IEEE69 A0 loader, F14 provenance)'


def load_ablation(env, checkpoint, ck, variant, device):
    model = model_class(variant)(**dict(ck['model_kwargs'], device=device))
    model.load(checkpoint, env)          # IEEE69 test-mode loader: sidecar sha + source inventory + provenance
    model.actor.eval()
    return model, 'ieee69.models evaluation loader (sidecar-verified source inventory, provenance)'


# ----------------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, choices=sorted(MODELS))
    p.add_argument('--run', required=True, type=int, choices=[0, 1, 2])
    p.add_argument('--scale', required=True, type=float, choices=ALLOWED_SCALES)
    p.add_argument('--pilot', action='store_true', help=f'optional: {PILOT_TOPOLOGY}, first {PILOT_EPISODES} days')
    p.add_argument('--job-map', required=True, type=Path)
    p.add_argument('--reference-manifest', type=Path,
                   help='A1/A2 1.0x final manifest (default results/ablations/ieee69/<variant>/run_<r>/final_test)')
    p.add_argument('--output-root', type=Path, default=OUTPUT_BASE)
    p.add_argument('--cpu', action='store_true')
    a = p.parse_args(argv)
    a.scale = check_scale(a.scale)
    if a.pilot and a.scale == 1.0:
        p.error('--pilot uses a stressed scale (1.1, 1.2 or 1.3)')
    if a.reference_manifest and a.model == 'a0':
        p.error('A0 references come from the job map')
    return a


def main(argv=None):
    args = parse_args(argv)
    label, variant = MODELS[args.model]
    entry = p69.verify_map(args.job_map, args.run)
    refs, a0_selection, groups = a0_references(entry, args.run)
    plans = {tp: [dict(episode_id=e, start_index=s, seed=z) for e, s, z in ref['plan']] for tp, ref in refs.items()}

    if args.model == 'a0':
        selection = a0_selection['selection_path']
        ref_path = None
    else:
        selection = entry['selections'][variant]
        ref_path = Path(args.reference_manifest or p69.result_root(variant, args.run) / 'final_test' / 'run_manifest.json')
    topologies = [PILOT_TOPOLOGY] if args.pilot else TOPOLOGIES
    protocol = authorize(mode='test', final=True, run_purpose='final_test', learned=True,
                         topologies=topologies, selection=selection)
    checkpoint, checkpoint_sha = protocol['selection']['checkpoint_path'], protocol['selection']['checkpoint_sha256']

    reference_checks, ck = {}, None
    if args.model == 'a0':
        if protocol['selection']['selection_sha256'] != a0_selection.get('selection_sha256'):
            raise StressError('A0 selection file changed since the A0 final evaluation')
        for tp, ref in refs.items():
            m = ref['manifest']           # same identity rule as final_evaluation.references()
            recorded = (m.get('checkpoint_file') or {}).get('sha256') or (m['evaluation_protocol'].get('selection') or {}).get('checkpoint_sha256')
            if recorded != checkpoint_sha:
                raise StressError(f'{tp}: A0 final evaluated a different checkpoint')
        reference_steps = {tp: Path(ref['path']).parent / 'step_metrics.csv' for tp, ref in refs.items()}
    else:
        sel = read_json(selection)
        if sel['model'] != variant or sel['run'] != f'run_{args.run}' or sel['feeder'] != 'ieee69':
            raise StressError('Wrong frozen selection')
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if ck.get('specification') != p69.feature_spec(variant) or ck.get('experiment_metadata', {}).get('run_id') != args.run:
            raise StressError('Wrong ablation/run checkpoint')
        if ck['experiment_metadata'].get('a0_manifest_sha256') != sha(entry['baseline_training_manifest']):
            raise StressError('Wrong A0 training identity')
        reference_checks = verify_ablation_reference(read_json(ref_path), protocol, variant, args.run, plans)
        reference_steps = {tp: ref_path.parent / 'step_metrics.csv' for tp in TOPOLOGIES}

    protected = {Path(g).parent for g in groups}
    if ref_path is not None:
        protected.add(ref_path.resolve().parent)
    output_root = check_output_root_ieee69(args.output_root, protected)
    gates = {}
    if args.scale != 1.0:
        gates['identity'] = identity_status(output_root, args.model, args.run, checkpoint_sha)
        if not args.pilot:
            gates['pilot'] = pilot_status(output_root, args.model, args.run, checkpoint_sha)
    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')

    out = run_dir(output_root, args.model, args.run, args.scale, args.pilot)
    run = RunManifest(out, 'MLP', refs['TP1']['manifest']['seed'], 'test', checkpoint, vars(args))
    if variant is not None:
        namespace['annotate'](run, variant, args.run, entry['baseline_training_manifest'])
    run.record_protocol(protocol, [MLP_POLICY_NAME])
    run.data['load_stress'] = dict(feeder='IEEE69', model=args.model, label=label, run=args.run, policy=MLP_POLICY_NAME,
        load_scaling_factor=args.scale, pv_scaling_factor=PV_SCALE, scenario=scenario(args.scale), pilot=bool(args.pilot),
        topologies=topologies, device=device, gates=gates,
        identity_verified=bool(gates.get('identity', {}).get('status') == 'PASS'),
        a0_topology_groups={g: tps for g, tps in groups.items()},
        a0_references={tp: dict(path=str(Path(ref['path']).resolve()), sha256=sha(ref['path'])) for tp, ref in refs.items()},
        ablation_reference=(dict(path=str(ref_path.resolve()), sha256=sha(ref_path), episode_plan_checks=reference_checks)
                            if ref_path is not None else None),
        reference_step_metrics={tp: str(Path(p).resolve()) for tp, p in reference_steps.items()},
        a0_training_manifest=dict(path=entry['baseline_training_manifest'], sha256=sha(entry['baseline_training_manifest'])),
        job_map=dict(path=str(args.job_map.resolve()), sha256=sha(args.job_map)),
        only_deviation='data_scaling.load_scaling_factor', normalization_note=NORMALIZATION_NOTE, note=POST_HOC,
        environment_comparisons={}, checkpoint_loading={})
    run._write()
    print('LOAD_STRESS_IEEE69', json.dumps(dict(model=args.model, run=args.run, scale=args.scale, pilot=args.pilot,
          out=str(out), device=device)), flush=True)

    frames, energies, observations = [], [], []
    for tp in topologies:
        ref = refs[tp]
        cfg = deepcopy(IEEE69_CONFIG)
        cfg['paths'].update(ref['files'])            # read-only exact archived A0 network bytes
        cfg['topology_case'] = dict(case_name=tp, **ref['case'])
        env = make_stress_env_ieee69(protocol_config(cfg, protocol), 'test', ref['manifest']['seed'], args.scale)
        if [s for _, s, _ in ref['plan']] != list(env.episode_start_indices) or len(ref['plan']) != DAYS:
            raise StressError(f'{tp}: A0 episode indices differ from the current December dataset')
        run.add_environment(env, topology=tp, scenario=scenario(args.scale), intended_episodes=DAYS)
        run.data['load_stress']['environment_comparisons'][tp] = compare_stress_environment_ieee69(
            run.data['environments'][-1], ref['manifest'], args.scale)
        plan = ref['plan'][:PILOT_EPISODES] if args.pilot else ref['plan']
        if args.model == 'a0':
            agent, how = load_a0(env, checkpoint, ref['manifest'], device)
        else:
            agent, how = load_ablation(env, checkpoint, ck, variant, device)
        run.data['load_stress']['checkpoint_loading'][tp] = how
        run.data.setdefault('episode_plans', {})[tp] = [dict(episode_id=e, start_index=s, seed=z) for e, s, z in plan]
        run._write()
        recorder = ObservationRecorder(agent, env, observations)
        rows = []
        for ep, start, seed in plan:
            recorder.context = dict(topology_case=tp, episode_id=ep, start_index=start, step_counter=0)
            rows.extend(evaluate_one_episode(env, recorder, MLP_POLICY_NAME, tp, ep, start, seed))
        frame = pd.DataFrame(rows)
        expected = pd.to_datetime(ref['timestamps'][:len(plan) * STEPS], utc=True)
        if frame.episode_completed.all() and not np.array_equal(pd.to_datetime(frame.date_time, utc=True), expected):
            raise StressError(f'{tp}: timestamps differ from the A0 final reference')
        frame['run_id'], frame['model'], frame['load_scaling_factor'] = args.run, label, args.scale
        if variant is not None:
            frame['ablation'] = variant
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
        df['run_id'], df['model'], df['load_scaling_factor'] = args.run, label, args.scale
    tables = [('step_metrics', steps), ('episode_summary', ep), ('aggregate_summary', agg),
              ('terminal_soc_summary', energy), ('adjusted_cost_summary', energy),
              ('obs_diagnostics', pd.DataFrame(observations).drop(columns='step_counter'))]
    for name, df in tables:
        df.to_csv(out / f'{name}.csv', index=False)

    complete = ep['complete'].astype(bool)
    run.data['load_stress']['episodes'] = dict(planned=int(sum(len(v) for v in run.data['episode_plans'].values())),
                                               evaluated=int(len(ep)), complete=int(complete.sum()))
    run.finish()
    if args.pilot:
        failures = int(steps['episode_failed'].astype(bool).sum())
        status = 'PASS' if len(ep) == PILOT_EPISODES and complete.all() and failures == 0 else 'FAIL'
        write_json(out / 'pilot_check.json', dict(status=status, scale=args.scale, topology=PILOT_TOPOLOGY,
            episodes=int(len(ep)), complete_episodes=int(complete.sum()), failed_steps=failures,
            converged_rate=float(steps['converged'].astype(bool).mean()), checkpoint_sha256=checkpoint_sha,
            evaluation_manifest_sha256=sha(out / 'run_manifest.json'), blocking=False))
        print('PILOT', status, '(informational; does not block full runs)', flush=True)
    print('LOAD_STRESS_IEEE69_DONE', out, flush=True)
    return out


if __name__ == '__main__':
    main()
