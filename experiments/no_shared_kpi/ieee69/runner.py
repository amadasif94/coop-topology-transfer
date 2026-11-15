"""IEEE69 bindings for the existing ablation workflow; no IEEE33 global mutation."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from configs.ieee69_config import IEEE69_CONFIG
from experiments.no_shared_kpi import runner as shared
from experiments.no_gnn_maddpg.run_train_mlp_ieee69 import set_seed
from experiments.no_gnn_maddpg.run_tp_eval_mlp_ieee69 import evaluate_one_episode, summarize
from experiments.topology_generalization.topology_utils_ieee69 import build_topology_config
from environments.primary_marl import make_primary_env as original_env
from . import protocol as p
from .bindings import bind
from .models import model_class
from utility.run_manifest import RunManifest as OriginalManifest


class RunManifest(OriginalManifest):
    def add_environment(self,env,*args,**kwargs):
        result=super().add_environment(env,*args,**kwargs)
        proof=getattr(env,'tp10_extension_equivalence',None)
        if proof:
            self.data['environments'][-1]['tp10_extension_equivalence']=proof
            self._write()
        return result


def make_primary_env(config, mode, seed):
    env = original_env(config, mode, seed)
    if env.num_buses != 69 or env.num_agents != 9 or list(env.dess_buses) != p.DESS_BUSES:
        raise ValueError('Wrong IEEE69 feeder/agent ordering')
    if env.agent_obs_dim != 10 or env.episode_length != 96 or env.delta_t_hours != .25:
        raise ValueError('Wrong IEEE69 observation/time interface')
    from .tp10_provenance import runtime_evidence
    env.tp10_extension_equivalence=runtime_evidence(env)
    return env


def energy_rows(step_df, env):
    """Same stored-energy valuation as IEEE33, but prices from the IEEE69 data.

    Stored-side kWh; no additional efficiency multiplier. SOC already includes
    efficiencies. Credit increased stored energy at mean daily price ($/MWh).
    """
    from utility.data_loader import load_processed_node_time_series
    ts = load_processed_node_time_series(env.config['paths']['processed_time_series'], 69, allow_empty=False)
    prices = ts['df'][ts['price_col']].to_numpy(float) * float(env.config.get('data_scaling', {}).get('price_scaling_factor', 1.0))
    caps = np.asarray([b.config.capacity_kwh for b in env.batteries])
    records = []
    for (tp, ep, start), g in step_df.groupby(['topology_case', 'episode_id', 'start_index']):
        g = g.sort_values('step')
        valid = len(g) == 96 and g.episode_completed.eq(True).all() and not g.episode_failed.eq(True).any()
        initial = np.full(9, np.nan)
        terminal = initial.copy()
        if 'soc_before' in g and isinstance(g.iloc[0]['soc_before'], (str, list, tuple, np.ndarray)):
            initial = shared.parse_array(g.iloc[0]['soc_before'])
        if valid:
            terminal = np.array([g.iloc[-1][f'soc_agent_{i}'] for i in range(9)], float)
        e0, e1 = float(initial @ caps), float(terminal @ caps)
        cost = float(g.energy_cost.sum(min_count=1)) if valid else np.nan
        price = float(prices[int(start):int(start)+96].mean())
        row = dict(topology_case=tp, episode_id=int(ep), start_index=int(start),
                   date=str(g.date_time.iloc[0])[:10], raw_energy_cost=cost,
                   adjusted_energy_cost=p.inventory_adjusted_cost(cost, e0, e1, price),
                   initial_stored_energy_kwh=e0, terminal_stored_energy_kwh=e1,
                   net_stored_energy_change_kwh=e1-e0, day_mean_price_per_mwh=price,
                   adjusted_cost_available=bool(valid and np.isfinite(e0) and np.isfinite(e1)), complete=valid)
        row.update(shared.dispatch_magnitudes(g))
        for i in range(9):
            row.update({f'initial_soc_agent_{i}':initial[i], f'terminal_soc_agent_{i}':terminal[i],
                        f'initial_energy_kwh_agent_{i}':initial[i]*caps[i],
                        f'terminal_energy_kwh_agent_{i}':terminal[i]*caps[i]})
        records.append(row)
    return pd.DataFrame(records)


# Only these private function objects see IEEE69 dependencies. Shared module
# globals and original source files remain unchanged, including during imports.
namespace = dict(vars(shared), IEEE33_CONFIG=IEEE69_CONFIG, set_seed=set_seed,
                 RunManifest=RunManifest,
                 model_class=model_class, make_primary_env=make_primary_env,
                 build_topology_config=build_topology_config,
                 evaluate_one_episode=evaluate_one_episode, summarize=summarize,
                 energy_rows=energy_rows, load_baseline=p.load_baseline,
                 compare_environment=p.compare_environment, feature_spec=p.feature_spec,
                 result_root=p.result_root, source_identity=p.source_identity)
for name in ('trainer_class', 'annotate', 'train', 'evaluate'):
    namespace[name] = bind(getattr(shared, name), namespace)
trainer_class, train, evaluate = (namespace[n] for n in ('trainer_class', 'train', 'evaluate'))


def main(variant, default_stage=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['train','selection_eval','sweep','select','evaluate'],
                        default=default_stage, required=default_stage is None)
    parser.add_argument('--run', type=int, choices=[0,1,2], required=True)
    parser.add_argument('--baseline-training-manifest', type=Path, required=True)
    parser.add_argument('--baseline-evaluation-manifest', type=Path, action='append')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--cpu', action='store_true')
    a = parser.parse_args()
    p.load_baseline(a.baseline_training_manifest, a.run)
    if a.stage == 'train': return train(variant, a)
    if a.stage == 'evaluate':
        if not a.baseline_evaluation_manifest or not a.selection:
            parser.error('Final evaluation requires A0 final manifest and frozen ablation selection')
        selection = p.read_json(a.selection)
        if selection.get('feeder', '').lower() != 'ieee69':
            raise ValueError('Selection is not IEEE69')
        from .final_evaluation import evaluate_final
        return evaluate_final(variant,a)
    if a.stage == 'selection_eval':
        if not a.checkpoint: parser.error('--checkpoint required')
        return evaluate(variant, a, selection_sweep=True)
    if a.stage == 'sweep':
        # Verified same A0 save cadence and warmup convention: 15k through 200k.
        directory = p.result_root(variant, a.run)/'training/checkpoints'
        checkpoints = [directory/f'maddpg_step_{step}.pt' for step in range(15000,200001,5000)]
        missing = [str(c) for c in checkpoints if not c.is_file()]
        if missing: raise FileNotFoundError('Required checkpoints missing: '+', '.join(missing))
        for c in checkpoints: evaluate(variant, a, checkpoint=c, selection_sweep=True)
    from scripts.select_checkpoint import main as select
    return select(['--feeder','ieee69','--model',variant,'--run',f'run_{a.run}',
                   '--sweep_dir',str(p.result_root(variant,a.run)/'selection_sweep'),
                   '--output_root',str(p.result_root(variant,a.run)/'selections')])
