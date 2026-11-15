"""Task 4: TP1 requested-kW action replay on an unseen topology versus closed-loop execution. Maintained entry point.

    python -u -m evaluation.final_analyses.action_replay run --feeder ieee33 --controller mlp_a0 --seed 0 --target TP6
                                                         [--scales 1.0 1.1 1.2 1.3] [--max-days N]
    python -u -m evaluation.final_analyses.action_replay compare --feeder ieee33

Experiments: IEEE33 TP1 -> TP6 (primary) and TP1 -> TP7 (secondary); IEEE69 TP1 -> TP10 (primary). A target must be a
maintained topology of the feeder (configs.experiments.TOPOLOGIES) and its environment must build and validate;
otherwise the experiment stops with the reason (no substitute topology).

Protocol (requested-kW replay; primary and only variant):
  source       the controller's authoritative December test run on TP1 (same checkpoint, seed, load factor, dates):
               per step the SOC-feasible requested battery power marl_power_soc_mapped_kw = P after the affine_v2
               mapping, before the physical limiter and the AC network backup. Read from the existing step table.
  closed loop  the controller on the target topology (it observes the target network every 15 minutes and recomputes
               its action). Reused from the authoritative December test run ONLY when that run's episode table contains
               the target topology; otherwise an evaluation-only closed-loop run is made here first with the same frozen
               checkpoint (verified against the source run's recorded checkpoint hash), environment configuration, seed
               42 + k, reset seeds, dates, SoC0, limiter and backup (CPU; evaluation.evaluate.test's procedure for one
               topology), written to .../closed_loop_<target>/load_x<pct>/. Finished outputs are skipped on resubmission;
               unfinished ones are refused (move away by hand).
  replay       NEW: the same dates, seeds (42 + k), initial SoC (0.50), load/PV, prices, load factor and environment
               configuration as the closed-loop target run; the environment is switched to direct_kw dispatch and each
               step requests the recorded source kW. The target's own battery limiter (rating, SoC window) and the AC
               network backup act exactly as in closed loop; nothing is bypassed and no action is re-mapped.
Why requested kW: under the affine mapping a raw actor output a means a FRACTION of the SoC-dependent available power,
so replaying a on the target would re-interpret it against the target's SoC trajectory (not the source's command);
post-backup executed powers already contain the source network's corrections. The mapped kW request is the source
controller's physical command before any network-specific correction, which isolates closed-loop feedback/adaptation.
Raw-action replay is not implemented (it would mix the policy command with the target SoC state).
Outputs (never overwritten): <out-root>/action_replay/<feeder>/<controller>/seed_<k>/TP1_to_<target>/load_x<pct>/
(replay run_manifest.json and the standard tables), and from `compare`: <out-root>/action_replay/<feeder>/compare_<UTC>/
episodes.csv (closed loop vs replay per date), trajectories.csv (per step: source request, closed-loop and replay
requests/limiter/executed kW and SoC), paired.csv (closed - replay, seed mean primary and per seed, date bootstrap).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from configs.experiments import FEEDERS, TOPOLOGIES, write_json
from evaluation.evaluate import POLICY, feeder_tools, run_episodes, write_tables
from evaluation.final_analyses.bootstrap import Resampler, holm, paired_stats
from evaluation.final_analyses.common import (DISPLAY, OUT_ROOT, RESULTS_ROOT, SCALES, SEEDS, STEPS, VARIANT,
                                              guard_output, json_list, rl_test_env, selected_checkpoint, stamp)
from experiments.load_stress.stress_env import check_scale, scale_label, scenario
from utility.evaluation_protocol import episode_plan
from utility.run_manifest import reserve_directory

EXPERIMENTS = dict(ieee33=(('TP6', 'primary'), ('TP7', 'secondary')), ieee69=(('TP10', 'primary'),))
CONTROLLERS = ('mlp_a0', 'mlp_a3')
REQUEST = 'marl_power_soc_mapped_kw'
METRICS = dict(raw_energy_cost='lower', adjusted_energy_cost='lower', total_reward_mean='higher',
               final_network_feasibility_rate='higher', mean_voltage_deviation='lower',
               backup_activation_rate='lower', b3_terminal_shortfall_mean='lower')
TRAJ_COLUMNS = ('marl_power_soc_mapped_kw', 'marl_planned_power_kw', 'physical_limited_power_kw',
                'executed_battery_power_kw', 'soc_after', 'voltage_deviation', 'backup_activated', 'final_network_feasible')


def test_dir(results_root, feeder, key, seed, scale):
    return Path(results_root) / feeder / 'mlp' / VARIANT[key] / f'seed_{seed}' / 'test' / scale_label(scale)


def verify_target(feeder, target):
    if target not in dict(EXPERIMENTS[feeder]):
        raise SystemExit(f'{feeder} {target} is not a configured replay target {EXPERIMENTS[feeder]}')
    if target not in TOPOLOGIES[feeder]:
        raise SystemExit(f'{feeder} {target} is not a maintained topology {TOPOLOGIES[feeder]}; experiment stopped')


def source_requests(results_root, feeder, key, seed, scale, n):
    """{start_index: [96, n] requested kW} from the authoritative TP1 test step table; incomplete days are omitted."""
    d = test_dir(results_root, feeder, key, seed, scale)
    m = json.loads((d / 'run_manifest.json').read_text())
    if m.get('status') != 'finished':
        raise SystemExit(f'{d}: source run not finished')
    steps = pd.read_csv(d / 'step_metrics.csv', usecols=lambda c: c in ('topology_case', 'start_index', 'step',
                                                                         'record_type', REQUEST), low_memory=False)
    if REQUEST not in steps:
        raise SystemExit(f'{d}: step table has no {REQUEST}; requested kW cannot be replayed')
    steps = steps[(steps.topology_case == 'TP1') & (steps.get('record_type', 'dispatch_attempt') == 'dispatch_attempt')]
    out, skipped = {}, []
    for start, g in steps.groupby('start_index'):
        g = g.sort_values('step')
        kw = json_list(g[REQUEST].tolist(), n)
        if len(g) == STEPS and (g.step.to_numpy() == np.arange(STEPS)).all() and np.isfinite(kw).all():
            out[int(start)] = kw
        else:
            skipped.append(int(start))
    return out, skipped, str(d)


class ReplayAgent:
    """Duck-typed agent: returns the recorded source kW request of the current step (direct_kw dispatch)."""

    def __init__(self, env, requests):
        self.env, self.requests = env, requests
        env.set_dispatch_mode('direct_kw')

    def select_action(self, obs, noise_std=0.0):
        start, step = int(self.env.episode_start_index), int(self.env.current_step)
        return np.asarray(self.requests[start][step], dtype=float).copy()


def _status(d):
    m = Path(d) / 'run_manifest.json'
    return json.loads(m.read_text()).get('status') if m.is_file() else None


def topologies_of(d):
    """Topologies present in a finished run folder's episode table (empty if the folder is unusable)."""
    d = Path(d)
    if _status(d) != 'finished' or not (d / 'episode_summary.csv').is_file():
        return set()
    return set(pd.read_csv(d / 'episode_summary.csv', usecols=['topology_case']).topology_case.astype(str))


def closed_loop_dir(results_root, base, feeder, key, seed, scale, target, max_days=None):
    """(folder, origin) of the closed-loop run on `target`: the authoritative test run when it contains the target
    topology, otherwise the evaluation-only run generated by this module (it may not exist yet)."""
    auth = test_dir(results_root, feeder, key, seed, scale)
    if not max_days and target in topologies_of(auth):
        return auth, 'authoritative_test_run'
    gen = base / 'action_replay' / feeder / key / f'seed_{seed}' / f'closed_loop_{target}' / \
        (scale_label(scale) + (f'_first{max_days}' if max_days else ''))
    return gen, 'generated_evaluation_only'


def checkpoint_consistency(results_root, feeder, key, seed, scale, protocol):
    """The frozen selected checkpoint must be the one the authoritative TP1 source run evaluated."""
    d = test_dir(results_root, feeder, key, seed, scale)
    if _status(d) != 'finished':
        raise SystemExit(f'{d}: authoritative source run missing or not finished')
    recorded = (json.loads((d / 'run_manifest.json').read_text()).get('checkpoint_file') or {}).get('sha256')
    selected = protocol['selection']['checkpoint_sha256']
    if recorded and recorded != selected:
        raise SystemExit(f'{d}: evaluated checkpoint {recorded[:12]} differs from the frozen selection {selected[:12]}')
    return dict(checkpoint_sha256=selected, source_run_checkpoint_sha256=recorded,
                checkpoint_match='verified' if recorded else 'source manifest has no checkpoint hash')


def _begin(out):
    """Skip a finished output folder; refuse an unfinished one (move it away by hand); else reserve it."""
    status = _status(out)
    if status == 'finished':
        print('FINAL_REPLAY_SKIPPED (finished)', out, flush=True)
        return False
    if Path(out).exists() and any(Path(out).iterdir()):
        raise SystemExit(f'{out} exists unfinished (status {status}); move it away by hand and resubmit')
    return True


def run_closed_loop(feeder, key, seed, target, scale, out, checkpoint, protocol, record, max_days):
    """Evaluation-only closed-loop run of the frozen checkpoint on `target`, exactly as evaluation.evaluate.test runs
    a topology (same environment configuration, seed 42 + k, reset seeds, SoC0, limiter, backup; CPU)."""
    from evaluation.evaluate import load_model, verified_mapping
    run = feeder_tools(feeder)[3](out, key, 42 + seed, 'test', checkpoint,
                                  dict(feeder=feeder, controller=key, seed=seed, topology=target, scale=scale,
                                       purpose='closed-loop reference for the TP1 action replay', max_days=max_days))
    run.record_protocol(protocol, [POLICY])
    env = rl_test_env(feeder, target, out, protocol, record, seed, scale)
    starts, seeds, _ = episode_plan(env, 'all', 42 + seed, protocol)
    plan = list(zip(starts, seeds))[:max_days or None]
    agent = load_model(key, checkpoint, env, 'cpu')
    run.data['closed_loop'] = dict(controller=key, seed=seed, topology=target, load_scaling_factor=scale, days=len(plan),
                                   device='cpu', **verified_mapping({'run_record': record}, env, agent))
    run.add_environment(env, topology=target, scenario=scenario(scale), intended_episodes=len(plan))
    frame = run_episodes(feeder, env, agent, POLICY, target, plan)
    write_tables(feeder, out, [frame], {target: env}, key, dict(run_id=seed, load_scaling_factor=scale))
    run.finish()
    print('FINAL_CLOSED_LOOP_DONE', feeder, key, seed, target, scale, len(plan), 'days', out, flush=True)


def run_replay(results_root, out_root, feeder, key, seed, target, scales, max_days=None):
    verify_target(feeder, target)
    base = guard_output(out_root, results_root)
    checkpoint, protocol, record = selected_checkpoint(results_root, feeder, key, seed)
    outs = []
    for scale in scales:
        consistency = checkpoint_consistency(results_root, feeder, key, seed, scale, protocol)
        cdir, origin = closed_loop_dir(results_root, base, feeder, key, seed, scale, target, max_days)
        if origin == 'generated_evaluation_only' and _begin(cdir):
            run_closed_loop(feeder, key, seed, target, scale, cdir, checkpoint, protocol, record, max_days)
        out = base / 'action_replay' / feeder / key / f'seed_{seed}' / f'TP1_to_{target}' / \
            (scale_label(scale) + (f'_first{max_days}' if max_days else ''))
        if not _begin(out):
            outs.append(out)
            continue
        run = feeder_tools(feeder)[3](out, f'{key}_tp1_kw_replay', 42 + seed, 'test', None,
                                      dict(feeder=feeder, controller=key, seed=seed, target=target, scale=scale,
                                           max_days=max_days))
        run.record_protocol(protocol, [POLICY])
        env = rl_test_env(feeder, target, out, protocol, record, seed, scale)
        requests, skipped, source = source_requests(results_root, feeder, key, seed, scale, env.num_agents)
        starts, seeds, _ = episode_plan(env, 'all', 42 + seed, protocol)
        plan = [(s, z) for s, z in zip(starts, seeds) if int(s) in requests][:max_days or None]
        run.data['action_replay'] = dict(
            controller=key, display=DISPLAY[key], seed=seed, source_topology='TP1', target_topology=target,
            experiment_role=dict(EXPERIMENTS[feeder])[target], load_scaling_factor=scale, source_run=source,
            closed_loop_run=str(cdir), closed_loop_origin=origin, **consistency,
            replayed_quantity=f'{REQUEST} (SOC-feasible affine_v2 request, before limiter and network backup)',
            dispatch='direct_kw: target battery limiter and AC network backup active', days=len(plan),
            source_days_skipped=skipped, reset_seeds='42 + k + day (same as the closed-loop test run)')
        run.add_environment(env, topology=target, scenario=scenario(scale), intended_episodes=len(plan))
        frame = run_episodes(feeder, env, ReplayAgent(env, requests), POLICY, target, plan)
        write_tables(feeder, out, [frame], {target: env}, f'{key}_tp1_kw_replay',
                     dict(run_id=seed, load_scaling_factor=scale))
        run.finish()
        print('FINAL_REPLAY_DONE', feeder, key, seed, target, scale, len(plan), 'days', out, origin, flush=True)
        outs.append(out)
    return outs


def _episode_frame(path, topology):
    ep = pd.read_csv(path / 'episode_summary.csv')
    ep = ep[ep.topology_case == topology]
    return ep.set_index('start_index')


def _steps(path, topology):
    s = pd.read_csv(path / 'step_metrics.csv', usecols=lambda c: c in ('topology_case', 'start_index', 'step', 'record_type')
                    + TRAJ_COLUMNS, low_memory=False)
    s = s[(s.topology_case == topology) & (s.get('record_type', 'dispatch_attempt') == 'dispatch_attempt')]
    return s


def compare(results_root, out_root, feeder, resamples=10000, allow_incomplete=False):
    base = guard_output(out_root, results_root)
    episodes, traj, missing, origins = [], [], [], {}
    for target, role in EXPERIMENTS[feeder]:
        for key in CONTROLLERS:
            for seed in SEEDS:
                for scale in SCALES:
                    rdir = base / 'action_replay' / feeder / key / f'seed_{seed}' / f'TP1_to_{target}' / scale_label(scale)
                    cdir, origin = closed_loop_dir(results_root, base, feeder, key, seed, scale, target)
                    sdir = test_dir(results_root, feeder, key, seed, scale)          # TP1 source: always authoritative
                    if _status(rdir) != 'finished' or target not in topologies_of(cdir):
                        missing.append(dict(target=target, controller=key, seed=seed, scale=scale,
                                            replay=_status(rdir), closed_loop=str(cdir)))
                        continue
                    origins[f'{target}/{key}/seed_{seed}/{scale_label(scale)}'] = origin
                    rep, clo = _episode_frame(rdir, target), _episode_frame(cdir, target)
                    common = rep.index.intersection(clo.index)
                    for s in common:
                        row = dict(feeder=feeder, target=target, role=role, controller=key, display=DISPLAY[key],
                                   seed=seed, scale=scale, start_index=int(s))
                        for m in METRICS:
                            row[f'closed_{m}'] = clo.at[s, m] if m in clo else np.nan
                            row[f'replay_{m}'] = rep.at[s, m] if m in rep else np.nan
                            row[f'diff_{m}'] = row[f'closed_{m}'] - row[f'replay_{m}']
                        episodes.append(row)
                    rstep, cstep = _steps(rdir, target), _steps(cdir, target)
                    src = pd.read_csv(sdir / 'step_metrics.csv', usecols=lambda c: c in ('topology_case', 'start_index',
                                                                                          'step', REQUEST), low_memory=False)
                    src = src[src.topology_case == 'TP1'][['start_index', 'step', REQUEST]].rename(
                        columns={REQUEST: 'source_requested_kw'})
                    m = cstep.drop(columns=['topology_case']).merge(
                        rstep.drop(columns=['topology_case']), on=['start_index', 'step'], suffixes=('_closed', '_replay'))
                    m = m.merge(src, on=['start_index', 'step'], how='left')
                    traj.append(m.assign(feeder=feeder, target=target, controller=key, seed=seed, scale=scale))
    if missing and not allow_incomplete:
        raise SystemExit(f'{len(missing)} replay/closed-loop runs missing or unfinished (use --allow-incomplete only for '
                         f'a partial look), e.g. {missing[:3]}')
    if not episodes:
        raise SystemExit('no finished replay runs to compare')
    ep = pd.DataFrame(episodes)
    out = reserve_directory(base / 'action_replay' / feeder / f'compare_{stamp()}')
    ep.to_csv(out / 'episodes.csv', index=False)
    pd.concat(traj, ignore_index=True).to_csv(out / 'trajectories.csv', index=False)
    rs, rows = Resampler(resamples, resamples), []
    for (target, key, scale), g in ep.groupby(['target', 'controller', 'scale']):
        for metric, better in METRICS.items():
            for seed in [None, *SEEDS]:
                sub = g if seed is None else g[g.seed == seed]
                per = sub.groupby('start_index')[f'diff_{metric}'].agg(['mean', 'count'])
                if seed is None:
                    per = per[per['count'] == len(SEEDS)]
                per = per[np.isfinite(per['mean'])].sort_index()
                rows.append(dict(feeder=feeder, target=target, controller=key, display=DISPLAY[key], scale=scale,
                                 metric=metric, better_direction=better, difference='closed loop - replay',
                                 seed='mean_of_3' if seed is None else seed,
                                 **paired_stats(per['mean'].to_numpy(), [str(i) for i in per.index], rs, better)))
    paired = pd.DataFrame(rows)
    paired['family'] = paired.apply(lambda r: f'replay|{r.target}|x{r.scale}|{r.metric}|seed{r.seed}', axis=1)
    paired['p_holm'] = np.nan
    for _, idx in paired.groupby('family').groups.items():
        paired.loc[idx, 'p_holm'] = holm(paired.loc[idx, 'p_signflip'].to_numpy())
    paired.to_csv(out / 'paired.csv', index=False)
    write_json(out / 'inclusion.json', dict(feeder=feeder, experiments=EXPERIMENTS[feeder], rows=len(ep),
                                            resamples=resamples, sign_convention='closed loop minus replay',
                                            closed_loop_origin=origins, missing=missing,
                                            complete=not missing))
    print('FINAL_REPLAY_COMPARE_DONE', feeder, out, flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--feeder', required=True, choices=FEEDERS)
    r.add_argument('--controller', required=True, choices=CONTROLLERS)
    r.add_argument('--seed', type=int, required=True, choices=SEEDS)
    r.add_argument('--target', required=True)
    r.add_argument('--scales', type=float, nargs='+', default=list(SCALES))
    r.add_argument('--max-days', type=int, help='smoke: first N December days (separate _first<N> folder)')
    c = sub.add_parser('compare')
    c.add_argument('--feeder', required=True, choices=FEEDERS)
    c.add_argument('--resamples', type=int, default=10000)
    c.add_argument('--allow-incomplete', action='store_true', help='compare whatever is finished (partial look only)')
    for s in (r, c):
        s.add_argument('--results-root', default=RESULTS_ROOT)
        s.add_argument('--out-root', default=OUT_ROOT)
    a = p.parse_args(argv)
    if a.command == 'run':
        return run_replay(a.results_root, a.out_root, a.feeder, a.controller, a.seed, a.target.upper(),
                          [check_scale(s) for s in a.scales], a.max_days)
    return compare(a.results_root, a.out_root, a.feeder, a.resamples, a.allow_incomplete)


if __name__ == '__main__':
    main()
