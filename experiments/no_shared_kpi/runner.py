"""Shared NEW ablation orchestration; imports original training/evaluation implementations."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import FunctionType
import numpy as np
import pandas as pd
import torch
from configs.ieee33_config import IEEE33_CONFIG
from environments.primary_marl import make_primary_env
from experiments.no_gnn_maddpg.trainer_mlp import MLPMADDPGTrainer
from experiments.no_gnn_maddpg.run_train_mlp_ieee33 import set_seed
from experiments.no_gnn_maddpg.run_tp_eval_mlp import evaluate_one_episode, summarize
from experiments.topology_generalization.topology_utils import build_topology_config
from utility.run_manifest import RunManifest, serial
from utility.evaluation_protocol import authorize, protocol_config, episode_plan
from experiments.no_shared_kpi.protocol import (ROOT, TRAIN_SETTINGS, load_baseline, compare_environment,
    result_root, feature_spec, source_identity, sha, write_json, read_json, COST_DEFINITION, inventory_adjusted_cost)


def model_class(variant):
    if variant not in ('no_shared_kpi','no_centralized_critic'):
        raise ValueError('Unknown ablation')
    if variant == 'no_shared_kpi':
        from experiments.no_shared_kpi.model_no_shared_kpi import NoSharedKPI
        return NoSharedKPI
    from experiments.no_centralized_critic.model_independent_critic import IndependentAgents
    return IndependentAgents


def trainer_class(variant):
    # Reuse the exact original constructor with only its MODEL FACTORY binding replaced
    # in a private globals dictionary. Never monkey-patch the original module or copy
    # its learning/replay/reset/periodic-validation loop. Avoid constructing a second
    # disposable model (which would consume the exploration/initialization RNG).
    original = MLPMADDPGTrainer.__init__
    namespace = dict(original.__globals__, MLPMADDPG=model_class(variant))
    constructor = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__, original.__closure__)
    constructor.__kwdefaults__ = original.__kwdefaults__
    def save_checkpoint(self,path):
        self.agent.experiment_metadata.update(training_step=self.global_step,total_steps=self.total_steps,
            exploration=self.exploration_config())
        return MLPMADDPGTrainer.save_checkpoint(self,path)
    return type('AblationMLPTrainer', (MLPMADDPGTrainer,), {'__init__': constructor,'save_checkpoint':save_checkpoint})


def annotate(run, variant, run_id, baseline_path):
    run.data.update(ablation=variant, run_id=run_id, specification=feature_spec(variant),
        ablation_source_files=source_identity(), a0_manifest=dict(path=str(baseline_path), sha256=sha(baseline_path)),
        adjusted_cost_definition=COST_DEFINITION,
        efficiency_convention='Historical comparison: stored-side energy, unit valuation weight; no extra efficiency multiplier')
    run._write()
    print('ABLATION_SPEC',json.dumps(run.data['specification'],sort_keys=True),flush=True)


def train(variant, args):
    baseline = load_baseline(args.baseline_training_manifest, args.run)
    seed = int(baseline['seed'])
    set_seed(seed)
    env = make_primary_env(deepcopy(IEEE33_CONFIG), 'train', seed)
    val = make_primary_env(deepcopy(IEEE33_CONFIG), 'val', seed+1000)
    out = result_root(variant, args.run)/'training'
    run = RunManifest(out, 'MLP', seed, 'train', arguments=dict(baseline['arguments'],
        checkpoint_dir=str(out/'checkpoints'),ablation=variant,run_id=args.run))
    run.data['a0_training_arguments']=baseline['arguments']
    annotate(run, variant, args.run, args.baseline_training_manifest)
    for e in (env, val):
        run.add_environment(e)
        compare_environment(run.data['environments'][-1], baseline)
    config = dict(TRAIN_SETTINGS, checkpoint_dir=str(out/'checkpoints'), share_actor=False,
        store_buffer_on_device=True, action_dim_per_agent=1, max_reset_retries=baseline['arguments'].get('max_reset_retries', 0))
    trainer = trainer_class(variant)(env, val, config, 'cpu' if args.cpu else None)
    trainer.agent.experiment_metadata = dict(run_id=args.run, seed=seed, a0_manifest_sha256=sha(args.baseline_training_manifest))
    run.data['effective_training_config'] = config
    run.record_exploration(dict(trainer.exploration_config(), seed=seed))
    try:
        logs = trainer.train()
    finally:
        run.data['reset_handling'] = trainer.reset_summary()
        run._write()
    write_json(out/'training_logs.json', serial(logs))
    run.data.update(actual_total_steps=trainer.global_step, failed_episode_count=trainer.failed_episode_count,
                    invalid_transition_count=trainer.invalid_transition_count)
    run.finish()


def parse_array(value):
    if isinstance(value, str): return np.asarray(json.loads(value), float)
    return np.asarray(value, float)


def energy_rows(step_df, env):
    """Read executed SOC, never requested power. Reuse historical stored-side formula."""
    from experiments.topology_generalization.compare_tp6_replay_vs_feedback_ieee33 import day_prices
    caps = np.array([b.config.capacity_kwh for b in env.batteries])
    prices = day_prices([int(s) for s in step_df.start_index.unique()])
    rows = []
    for (tp, ep, start), g in step_df.groupby(['topology_case','episode_id','start_index']):
        g = g.sort_values('step')
        valid = len(g) == 96 and g.episode_completed.eq(True).all() and not g.episode_failed.eq(True).any()
        initial = np.full(env.num_agents, np.nan)
        terminal = initial.copy()
        if 'soc_before' in g and isinstance(g.iloc[0]['soc_before'], (str, list, tuple, np.ndarray)):
            initial = parse_array(g.iloc[0]['soc_before'])
        if valid and all(f'soc_agent_{i}' in g for i in range(env.num_agents)):
            terminal = np.array([g.iloc[-1][f'soc_agent_{i}'] for i in range(env.num_agents)], float)
        e0, e1 = float(initial @ caps), float(terminal @ caps)
        cost = float(g.energy_cost.sum(min_count=1)) if valid else np.nan
        r = dict(topology_case=tp, episode_id=int(ep), start_index=int(start), date=str(g.date_time.iloc[0])[:10],
            raw_energy_cost=cost, adjusted_energy_cost=inventory_adjusted_cost(cost,e0,e1,prices[int(start)]),
            initial_stored_energy_kwh=e0, terminal_stored_energy_kwh=e1,
            net_stored_energy_change_kwh=e1-e0, day_mean_price_per_mwh=prices[int(start)],
            adjusted_cost_available=bool(valid and np.isfinite(e0) and np.isfinite(e1)), complete=valid)
        r.update(dispatch_magnitudes(g))
        for i in range(env.num_agents):
            r.update({f'initial_soc_agent_{i}':initial[i], f'terminal_soc_agent_{i}':terminal[i],
                f'initial_energy_kwh_agent_{i}':initial[i]*caps[i], f'terminal_energy_kwh_agent_{i}':terminal[i]*caps[i]})
        rows.append(r)
    return pd.DataFrame(rows)


def dispatch_magnitudes(g):
    """Mean agent-step kW magnitude, kept separate from event frequencies."""
    def delta(left, right):
        if left not in g or right not in g: return np.nan
        vals=[]
        for a,b in zip(g[left],g[right]):
            if isinstance(a,(str,list,tuple,np.ndarray)) and isinstance(b,(str,list,tuple,np.ndarray)):
                x,y=parse_array(a),parse_array(b)
                if np.isfinite(x).all() and np.isfinite(y).all(): vals.extend(np.abs(x-y).tolist())
        return float(np.mean(vals)) if vals else np.nan
    return dict(mean_backup_adjustment_kw=delta('executed_battery_power_kw','doe_power_kw'),
        mean_physical_clipping_kw=delta('policy_requested_power_kw','physical_limited_power_kw'))


def evaluate(variant, args, checkpoint=None, selection_sweep=False):
    training = load_baseline(args.baseline_training_manifest, args.run)
    final = not selection_sweep
    baseline = load_baseline(args.baseline_evaluation_manifest, args.run, training=False) if final else training
    topologies = [f'TP{i}' for i in range(1,8)] if final else ['TP1']
    protocol = authorize(mode='test' if final else 'val', final=final,
        run_purpose='final_test' if final else 'checkpoint_selection', learned=True,
        topologies=topologies, selection=args.selection if final else None,
        checkpoint=checkpoint or args.checkpoint)
    checkpoint = checkpoint or args.checkpoint or protocol['selection']['checkpoint_path']
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if ck.get('specification') != feature_spec(variant) or ck.get('experiment_metadata', {}).get('run_id') != args.run:
        raise ValueError('Wrong ablation/run checkpoint')
    if ck['experiment_metadata'].get('a0_manifest_sha256') != sha(args.baseline_training_manifest):
        raise ValueError('Checkpoint uses a different A0 training manifest')
    if final:
        sel = read_json(args.selection)
        if sel['model'] != variant or sel['run'] != f'run_{args.run}': raise ValueError('Wrong selection identity')
    step = int(Path(checkpoint).stem.rsplit('_',1)[-1]) if selection_sweep else None
    if selection_sweep and ck['experiment_metadata'].get('training_step') != step:
        raise ValueError('Checkpoint filename does not match recorded training step')
    out = result_root(variant, args.run)/('selection_sweep' if selection_sweep else 'final_test')
    if selection_sweep: out /= f'step_{step}'
    seed = int(baseline['seed'])
    run = RunManifest(out, 'MLP', seed, protocol['data_split'], checkpoint, vars(args))
    annotate(run, variant, args.run, args.baseline_training_manifest)
    if final: run.data['a0_evaluation_manifest'] = dict(path=str(args.baseline_evaluation_manifest), sha256=sha(args.baseline_evaluation_manifest))
    run.record_protocol(protocol, ['mlp_maddpg'])
    frames, energy = [], []
    for tp in topologies:
        # Original A0 selection uses run_eval_mlp_ieee33 with the base TP1 files,
        # whereas final topology evaluation uses generated topology CSVs. Preserve
        # this distinction, including byte-level input provenance.
        base_cfg = (build_topology_config(case_name=tp, validate=True, output_root=out/'network_inputs')
                    if final else deepcopy(IEEE33_CONFIG))
        cfg = protocol_config(base_cfg, protocol)
        env = make_primary_env(cfg, protocol['data_split'], seed)
        starts, seeds, schedule = episode_plan(env, 'all', seed, protocol)
        if final and (len(starts) != 31 or env.episode_length != 96): raise ValueError('Expected 31 December days x 96 steps')
        run.add_environment(env, topology=tp, intended_episodes=len(starts))
        compare_environment(run.data['environments'][-1], baseline)
        if schedule: run.record_schedule(schedule)
        model = model_class(variant)(**dict(ck['model_kwargs'], device='cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')))
        model.load(checkpoint, env)
        model.actor.eval()
        run.data.setdefault('episode_plans', {})[tp] = [dict(start_index=int(s), seed=int(z)) for s,z in zip(starts,seeds)]
        run._write()
        rows = []
        for ep, (start, reset_seed) in enumerate(zip(starts,seeds)):
            rows.extend(evaluate_one_episode(env, model, 'mlp_maddpg', tp, ep, start, reset_seed))
        frame = pd.DataFrame(rows)
        frame['run_id'] = args.run
        frame['ablation'] = variant
        frames.append(frame)
        energy.append(energy_rows(frame, env))
    steps = pd.concat(frames, ignore_index=True)
    ep, agg = summarize(steps)
    energies = pd.concat(energy, ignore_index=True)
    ep = ep.merge(energies, on=['topology_case','episode_id','start_index'], validate='one_to_one')
    ep['run_id'], ep['ablation'] = args.run, variant
    extra = ep.groupby('topology_case')[['raw_energy_cost','adjusted_energy_cost','terminal_stored_energy_kwh',
        'mean_backup_adjustment_kw','mean_physical_clipping_kw']].mean().add_prefix('mean_').reset_index()
    agg = agg.merge(extra, on='topology_case', validate='one_to_one')
    agg['run_id'], agg['ablation'] = args.run, variant
    for name, df in [('step_metrics',steps),('episode_summary',ep),('aggregate_summary',agg),
                     ('terminal_soc_summary',energies),('adjusted_cost_summary',energies)]:
        df.to_csv(out/f'{name}.csv', index=False)
    run.finish()
    return out


def main(variant, default_stage=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=['train','selection_eval','sweep','select','evaluate'], default=default_stage, required=default_stage is None)
    p.add_argument('--run', type=int, choices=[0,1,2], required=True)
    p.add_argument('--baseline-training-manifest', type=Path, required=True)
    p.add_argument('--baseline-evaluation-manifest', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--selection', type=Path)
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()
    if args.stage == 'train': return train(variant, args)
    if args.stage == 'evaluate':
        if not args.baseline_evaluation_manifest or not args.selection: p.error('Evaluation requires A0 final manifest and frozen --selection')
        return evaluate(variant, args)
    if args.stage == 'selection_eval':
        if args.checkpoint is None: p.error('--checkpoint required')
        return evaluate(variant, args, selection_sweep=True)
    load_baseline(args.baseline_training_manifest, args.run)
    if args.stage == 'sweep':
        # Original save cadence; include the 200k candidate, not only the selected step.
        directory = result_root(variant,args.run)/'training/checkpoints'
        # Original train() enters its save loop AFTER warmup, so the first
        # scheduled checkpoint is 15000, not 5000 or 10000 (38 candidates).
        checkpoints = [directory/f'maddpg_step_{s}.pt' for s in range(15000,200001,5000)]
        missing = [str(c) for c in checkpoints if not c.is_file()]
        if missing: raise FileNotFoundError('Missing required scheduled checkpoints: '+', '.join(missing))
        for checkpoint in checkpoints: evaluate(variant,args,checkpoint,selection_sweep=True)
    from scripts.select_checkpoint import main as select_original
    return select_original(['--feeder','ieee33','--model',variant,'--run',f'run_{args.run}',
        '--sweep_dir',str(result_root(variant,args.run)/'selection_sweep'),
        '--output_root',str(result_root(variant,args.run)/'selections')])
