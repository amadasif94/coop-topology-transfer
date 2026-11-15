"""IEEE33 load-stress evaluation of the FROZEN A0/A1/A2 checkpoints (and the zero-action baseline).

One model x run x load scale per call, on TP1-TP7 and all 31 December test days, reusing the
existing frozen selections. Post-hoc sensitivity analysis: never used for model choice.

    python -u -m experiments.load_stress.evaluate_load_stress_ieee33 \
        --model {a0,a1,a2,zero} --run {0,1,2} --scale {1.0,1.1,1.2,1.3} [--smoke] --job-map <job_map.json>

Gates (enforced here, per model and run):
  scale 1.0          optional identity run, checked by validate_load_stress_identity.py
  --smoke (1.3 only) optional: TP1, first 2 days; writes smoke_check.json
  1.1 / 1.2 / 1.3    full; records the smoke result (or NOT_PERFORMED) but never requires it
The 1.0x identity check is OPTIONAL: when absent, stress runs proceed and record identity NOT_PERFORMED
(load_stress.identity_verified=false); an existing identity report must be a valid PASS or the run is refused.

Reused unchanged: authorize/protocol_config/episode_plan, build_topology_config, evaluate_one_episode,
summarize, runner.energy_rows/annotate, A0 MLPMADDPG.load and the A1/A2 AblationCheckpoint.load
(through ieee33_eval_compat when the embedded source inventory predates the approved additions).
Outputs: <output-root>/<label>/run_<r>/load_x<pct>[_smoke]/ ; never inside an existing final directory.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from experiments.load_stress.stress_env import (ALLOWED_SCALES, PV_SCALE, check_scale, compare_stress_environment,
    make_stress_env, scale_label, scenario)
from experiments.no_gnn_maddpg.run_tp_eval_mlp import MLP_POLICY_NAME, build_agent, evaluate_one_episode, summarize
from experiments.no_shared_kpi import model_support
from experiments.no_shared_kpi.ieee33_eval_compat import compatibility_loader
from experiments.no_shared_kpi.protocol import (ROOT, VARIANTS, feature_spec, load_baseline, read_json, result_root,
    sha, source_identity, write_json)
from experiments.no_shared_kpi.runner import annotate, energy_rows, model_class
from experiments.topology_generalization.topology_utils import build_topology_config
from utility.evaluation_protocol import authorize, episode_plan, protocol_config
from utility.run_manifest import RunManifest

TOPOLOGIES = [f'TP{i}' for i in range(1, 8)]
MODELS = dict(a0=('a0_original', None), a1=('a1_no_shared_kpi', VARIANTS[0]),
              a2=('a2_no_centralized_critic', VARIANTS[1]), zero=('zero_action', None))
DEFAULT_OUTPUT_ROOT = Path('results/load_stress/ieee33')
SMOKE_SCALE, SMOKE_EPISODES, SMOKE_TOPOLOGY = 1.3, 2, 'TP1'
OBS_FEATURES = ('load_kw', 'feeder_grid_import_kw')
POST_HOC = ('Post-hoc load-stress sensitivity analysis of frozen selections on the December test split; '
            'never used for checkpoint selection, tuning or model choice.')
NORMALIZATION_NOTE = ('Normalizers keep the unscaled bus base load (DESSEnv total_base_load_kw); stressed load and '
                      'feeder-import features therefore exceed the 1.0x training range (out of distribution by design).')


class StressError(RuntimeError):
    pass


def model_dir(output_root, model, run):
    return Path(output_root) / MODELS[model][0] / f'run_{run}'


def run_dir(output_root, model, run, scale, smoke=False):
    return model_dir(output_root, model, run) / (scale_label(scale) + ('_smoke' if smoke else ''))


def check_output_root(output_root, protected):
    out = Path(output_root).resolve()
    for p in [ROOT / 'results/topology_generalization', ROOT / 'results/ablations', *protected]:
        p = Path(p).resolve()
        if out == p or out.is_relative_to(p) or p.is_relative_to(out):
            raise StressError(f'Output root {out} overlaps protected result directory {p}')
    return out


# --------------------------------------------------------------------------------------- gates
def identity_gate(output_root, model, run, checkpoint_sha):
    """Optional: no report -> NOT_PERFORMED (recorded, not blocking); an existing report must be a valid PASS."""
    d = run_dir(output_root, model, run, 1.0)
    path = d / 'identity_report.json'
    if not path.is_file():
        print(f'WARNING: optional 1.0x identity check not performed ({path} absent); recorded as NOT_PERFORMED', flush=True)
        return dict(performed=False, status='NOT_PERFORMED', path=None, sha256=None)
    report = read_json(path)
    if report.get('status') != 'PASS':
        raise StressError(f'Identity gate: {path} status {report.get("status")}')
    if report.get('evaluation_manifest_sha256') != sha(d / 'run_manifest.json'):
        raise StressError('Identity gate: the 1.0x run manifest changed after validation')
    if checkpoint_sha is not None and report.get('checkpoint_sha256') != checkpoint_sha:
        raise StressError('Identity gate: report was produced for a different checkpoint')
    return dict(performed=True, status='PASS', path=str(path.resolve()), sha256=sha(path))


def smoke_gate(output_root, model, run, checkpoint_sha):
    """Informational only: the optional 1.3x smoke result is recorded, never required."""
    path = run_dir(output_root, model, run, SMOKE_SCALE, smoke=True) / 'smoke_check.json'
    if not path.is_file():
        return dict(path=None, sha256=None, status='NOT_PERFORMED')
    report = read_json(path)
    return dict(path=str(path.resolve()), sha256=sha(path), status=report.get('status'),
                same_checkpoint=report.get('checkpoint_sha256') == checkpoint_sha)


# ------------------------------------------------------------------------------- references
def zero_reference(job, a0_path, a0_manifest):
    """Existing 1.0x zero-action final run for the optional zero identity check, or None."""
    args = job.get('analysis_arguments') or []
    if '--zero' in args:
        return Path(args[args.index('--zero') + 1]) / 'run_manifest.json'
    if 'zero' in (a0_manifest.get('evaluation_protocol', {}).get('selection_by_policy') or {}):
        return Path(a0_path)
    return None


def resolve_references(args, job):
    entry = job['runs'][str(args.run)]
    a0_path = Path(entry['baseline_evaluation_manifest'])
    baseline = load_baseline(a0_path, args.run, training=False)
    label, variant = MODELS[args.model]
    info = dict(a0_evaluation_manifest=dict(path=str(a0_path.resolve()), sha256=sha(a0_path)))
    if args.model == 'zero':
        # The episode plan is checked against A0; the zero reference is only for the optional identity check.
        zero = args.reference_manifest or zero_reference(job, a0_path, baseline)
        info['zero_identity_reference'] = (dict(path=str(Path(zero).resolve()), sha256=sha(zero))
                                           if zero and Path(zero).is_file() else None)
        return baseline, entry, None, a0_path, info
    if args.model == 'a0':
        ref_path = Path(args.reference_manifest or a0_path)
        selection = baseline['evaluation_protocol']['selection']['selection_path']
    else:
        ref_path = Path(args.reference_manifest or result_root(variant, args.run) / 'final_test' / 'run_manifest.json')
        selection = entry['selections'][variant]
        info['a0_training_manifest'] = dict(path=entry['baseline_training_manifest'], sha256=sha(entry['baseline_training_manifest']))
    return baseline, entry, selection, ref_path, info


def verify_reference(ref, protocol, model, run, variant):
    p = ref.get('evaluation_protocol') or {}
    if ref.get('status') != 'finished' or p.get('data_split') != 'test' or p.get('final_test') is not True:
        raise StressError('Reference is not a finished authorized December final test')
    if p.get('topologies') != TOPOLOGIES:
        raise StressError('Reference final test does not cover TP1-TP7 in order')
    if model == 'zero':
        return
    sel = protocol['selection']
    if (p.get('selection') or {}).get('selection_sha256') != sel['selection_sha256']:
        raise StressError('Reference final test used a different frozen selection record')
    if (ref.get('checkpoint_file') or {}).get('sha256') != sel['checkpoint_sha256']:
        raise StressError('Reference final test evaluated a different checkpoint')
    if variant is not None and (ref.get('ablation') != variant or int(ref.get('run_id', -1)) != run):
        raise StressError('Reference final test belongs to a different ablation/run')


def reference_plan(ref, tp):
    envs = [e for e in ref['environments'] if e['topology'] == tp and e['mode'] == 'test']
    if not envs:
        raise StressError(f'Reference has no {tp}/test environment')
    return [int(i) for i in envs[0]['available_episode_start_indices']], (ref.get('episode_plans') or {}).get(tp)


# ------------------------------------------------------------------------------------ models
class ObservationRecorder:
    """Pass-through: records two normalized observation features, returns the policy's own action."""

    def __init__(self, agent, env, sink):
        self.agent, self.sink = agent, sink
        self.index = [list(env.agent_obs_features).index(f) for f in OBS_FEATURES]
        self.context = None

    def select_action(self, obs, noise_std=0.0):
        x = obs['agent_obs']
        x = x.detach().cpu().numpy() if hasattr(x, 'detach') else np.asarray(x)
        x = np.asarray(x, dtype=np.float64).reshape(-1, x.shape[-1])
        row = dict(self.context, step=self.context['step_counter'])
        for name, i in zip(OBS_FEATURES, self.index):
            row[f'{name}_feature_max'] = float(x[:, i].max())
            row[f'{name}_feature_mean'] = float(x[:, i].mean())
        self.sink.append(row)
        self.context['step_counter'] += 1
        return self.agent.select_action(obs, noise_std=noise_std)


def load_a0(env, checkpoint, ref, device):
    a = ref.get('arguments') or {}
    ns = SimpleNamespace(gamma=a.get('gamma', 0.99), tau=a.get('tau', 0.005), actor_lr=a.get('actor_lr', 1e-4),
                         critic_lr=a.get('critic_lr', 1e-3), share_actor=bool(a.get('share_actor', False)),
                         checkpoint=str(checkpoint))
    return build_agent(env, ns, device), 'MLPMADDPG.load (original A0 loader, F14 provenance)'


def load_ablation(env, checkpoint, variant, run, training_manifest, device):
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if ck.get('specification') != feature_spec(variant) or ck.get('experiment_metadata', {}).get('run_id') != run:
        raise StressError('Wrong ablation/run checkpoint')
    if ck['experiment_metadata'].get('a0_manifest_sha256') != sha(training_manifest):
        raise StressError('Checkpoint uses a different A0 training manifest')
    model = model_class(variant)(**dict(ck['model_kwargs'], device=device))
    if ck.get('sources') == source_identity():
        model.load(checkpoint, env)
        how = 'AblationCheckpoint.load (embedded source inventory matches)'
    else:
        with compatibility_loader(model_support):
            model.load(checkpoint, env)
        how = 'ieee33_eval_compat.compatibility_loader -> AblationCheckpoint.load (sidecar-verified inventory)'
    model.actor.eval()
    return model, how


# -------------------------------------------------------------------------------------- main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--model', required=True, choices=sorted(MODELS))
    p.add_argument('--run', required=True, type=int, choices=[0, 1, 2])
    p.add_argument('--scale', required=True, type=float, choices=ALLOWED_SCALES)
    p.add_argument('--smoke', action='store_true', help=f'TP1, first {SMOKE_EPISODES} days, scale {SMOKE_SCALE} only')
    p.add_argument('--job-map', required=True, type=Path)
    p.add_argument('--reference-manifest', type=Path,
                   help='Existing 1.0x final manifest of this model/run (default: derived from the job map)')
    p.add_argument('--output-root', type=Path, default=DEFAULT_OUTPUT_ROOT)
    p.add_argument('--cpu', action='store_true')
    a = p.parse_args(argv)
    a.scale = check_scale(a.scale)
    if a.smoke and a.scale != SMOKE_SCALE:
        p.error(f'--smoke uses scale {SMOKE_SCALE}')
    if a.model == 'zero' and a.run != 0:
        p.error('The zero-action baseline does not depend on the training run; use --run 0')
    return a


def main(argv=None):
    args = parse_args(argv)
    label, variant = MODELS[args.model]
    learned = args.model != 'zero'
    job = read_json(args.job_map)
    baseline, entry, selection, ref_path, info = resolve_references(args, job)
    ref = read_json(ref_path)
    protocol = authorize(mode='test', final=True, run_purpose='final_test', learned=learned,
                         topologies=TOPOLOGIES, selection=selection)
    verify_reference(ref, protocol, args.model, args.run, variant)
    checkpoint = protocol['selection']['checkpoint_path'] if learned else None
    checkpoint_sha = protocol['selection']['checkpoint_sha256'] if learned else None
    protected = {Path(info['a0_evaluation_manifest']['path']).parent, ref_path.resolve().parent}
    output_root = check_output_root(args.output_root, protected)

    gates = {}
    if args.scale != 1.0:
        gates['identity'] = (identity_gate(output_root, args.model, args.run, checkpoint_sha) if learned else
                             dict(performed=False, status='NOT_APPLICABLE', path=None, sha256=None))
        if not args.smoke:
            gates['smoke'] = smoke_gate(output_root, args.model, args.run, checkpoint_sha)
    topologies = [SMOKE_TOPOLOGY] if args.smoke else TOPOLOGIES
    seed = int(baseline['seed'])
    if learned and int(ref['seed']) != seed:
        raise StressError('Reference seed differs from the A0 final seed')
    device = 'cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    policy = MLP_POLICY_NAME if learned else 'zero'

    out = run_dir(output_root, args.model, args.run, args.scale, args.smoke)
    run = RunManifest(out, 'MLP' if learned else 'ZERO', seed, 'test', checkpoint, vars(args))
    if variant is not None:
        annotate(run, variant, args.run, entry['baseline_training_manifest'])
    run.record_protocol(protocol, [policy])
    run.data['load_stress'] = dict(info, model=args.model, label=label, run=args.run, policy=policy,
        load_scaling_factor=args.scale, pv_scaling_factor=PV_SCALE, scenario=scenario(args.scale),
        smoke=bool(args.smoke), topologies=topologies, device=device,
        reference_manifest=dict(path=str(ref_path.resolve()), sha256=sha(ref_path)), gates=gates,
        identity_verified=bool(gates.get('identity', {}).get('status') == 'PASS'),
        only_deviation='data_scaling.load_scaling_factor', normalization_note=NORMALIZATION_NOTE, note=POST_HOC,
        environment_comparisons={}, episode_plan_checks={})
    run._write()
    print('LOAD_STRESS', json.dumps(dict(model=args.model, run=args.run, scale=args.scale, smoke=args.smoke,
          out=str(out), device=device)), flush=True)

    frames, energy, observations, loaders = [], [], [], {}
    for tp in topologies:
        cfg = protocol_config(build_topology_config(case_name=tp, output_root=out / 'network_inputs', validate=True), protocol)
        env = make_stress_env(cfg, 'test', seed, args.scale)
        starts, seeds, schedule = episode_plan(env, 'all', seed, protocol)
        if schedule is not None or len(starts) != 31 or env.episode_length != 96:
            raise StressError('Expected 31 December days x 96 steps')
        ref_starts, ref_plan = reference_plan(ref, tp)
        plan = [dict(start_index=int(s), seed=int(z)) for s, z in zip(starts, seeds)]
        if [int(s) for s in starts] != ref_starts or (ref_plan is not None and ref_plan != plan):
            raise StressError(f'{tp}: episode plan differs from the reference final test')
        run.add_environment(env, topology=tp, scenario=scenario(args.scale), intended_episodes=len(starts))
        run.data['load_stress']['environment_comparisons'][tp] = compare_stress_environment(
            run.data['environments'][-1], baseline, args.scale)
        run.data['load_stress']['episode_plan_checks'][tp] = dict(
            matches_reference_start_indices=True, matches_reference_episode_plan=ref_plan is not None)
        if args.smoke:
            plan = plan[:SMOKE_EPISODES]
        run.data.setdefault('episode_plans', {})[tp] = plan
        agent, recorder = None, None
        if args.model == 'a0':
            agent, loaders[tp] = load_a0(env, checkpoint, ref, device)
        elif learned:
            agent, loaders[tp] = load_ablation(env, checkpoint, variant, args.run, entry['baseline_training_manifest'], device)
        if learned:
            recorder = ObservationRecorder(agent, env, observations)
        run.data['load_stress']['checkpoint_loading'] = loaders
        run._write()
        rows = []
        for ep, item in enumerate(plan):
            if recorder is not None:
                recorder.context = dict(topology_case=tp, episode_id=ep, start_index=item['start_index'], step_counter=0)
            rows.extend(evaluate_one_episode(env, recorder, policy, tp, ep, item['start_index'], item['seed']))
        frame = pd.DataFrame(rows)
        frame['run_id'], frame['model'], frame['load_scaling_factor'] = args.run, label, args.scale
        if variant is not None:
            frame['ablation'] = variant
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
        df['run_id'], df['model'], df['load_scaling_factor'] = args.run, label, args.scale
    tables = [('step_metrics', steps), ('episode_summary', ep), ('aggregate_summary', agg),
              ('terminal_soc_summary', energies), ('adjusted_cost_summary', energies)]
    if observations:
        tables.append(('obs_diagnostics', pd.DataFrame(observations).drop(columns='step_counter')))
    for name, df in tables:
        df.to_csv(out / f'{name}.csv', index=False)

    complete = ep['complete'].astype(bool)
    run.data['load_stress']['episodes'] = dict(planned=int(sum(len(v) for v in run.data['episode_plans'].values())),
        evaluated=int(len(ep)), complete=int(complete.sum()))
    run.finish()
    if args.smoke:
        failures = int(steps['episode_failed'].astype(bool).sum()) if 'episode_failed' in steps else -1
        status = 'PASS' if len(ep) == SMOKE_EPISODES and complete.all() and failures == 0 else 'FAIL'
        write_json(out / 'smoke_check.json', dict(status=status, scale=args.scale, topology=SMOKE_TOPOLOGY,
            episodes=int(len(ep)), complete_episodes=int(complete.sum()), failed_steps=failures,
            converged_rate=float(steps['converged'].astype(bool).mean()), checkpoint_sha256=checkpoint_sha,
            evaluation_manifest_sha256=sha(out / 'run_manifest.json')))
        print('SMOKE', status, flush=True)
        if status != 'PASS':
            sys.exit(1)
    print('LOAD_STRESS_DONE', out, flush=True)
    return out


if __name__ == '__main__':
    main()
