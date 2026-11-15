"""Paired, shared-day-cluster analysis. Read-only inputs; refuses output reuse.

Example: python -m experiments.no_shared_kpi.compare_ablations_ieee33
  --evaluation A0 0 <dir> ... --evaluation A2 2 <dir> --zero <dir> --output <new-dir>
All nine learned evaluations and a matching zero baseline are required. Random is optional.
Intervals condition on these three trained policies, NOT 93 independent training seeds.
"""
import argparse
import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.no_shared_kpi.protocol import ROOT, read_json, sha, write_json, COST_DEFINITION, inventory_adjusted_cost

COSTS = ('raw_energy_cost', 'adjusted_energy_cost')
PHYSICAL = ('mean_voltage_deviation','total_grid_import_mwh','total_curtailment_mwh','total_throughput_mwh',
    'feasible_rate','max_voltage_violation','max_line_current_violation','backup_activation_count',
    'backup_activation_rate','physical_clipping_rate','physical_clipping_count','terminal_stored_energy_kwh',
    'mean_backup_adjustment_kw','mean_physical_clipping_kw','doe_intervention_count',
    'executed_network_violation_count','executed_network_violation_rate',
    'total_reward_mean','total_reward_team','min_voltage_pu','max_voltage_pu',
    'max_line_current_pu','final_network_feasibility_rate')


def bool_values(s):
    return s.map(lambda x: x is True or x == 1 or str(x).lower() == 'true')


def stats(values, rng, resamples=10000):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v): return dict(n=0, mean=np.nan, median=np.nan, std=np.nan, ci_low=np.nan, ci_high=np.nan, direction='unavailable')
    boot = np.mean(v[rng.integers(0,len(v),size=(resamples,len(v)))], axis=1)
    return dict(n=len(v),mean=float(v.mean()),median=float(np.median(v)),std=float(v.std(ddof=1)) if len(v)>1 else np.nan,
        ci_low=float(np.quantile(boot,.025)),ci_high=float(np.quantile(boot,.975)),
        direction='positive' if v.mean()>0 else 'negative' if v.mean()<0 else 'zero')


def pooled_stats(frame, rng, resamples=10000):
    # Intersection across seeds: one shared sampled day index vector applies to ALL three.
    pivot = frame.pivot(index='start_index',columns='run',values='difference').reindex(columns=[0,1,2]).dropna()
    conditional = stats(pivot.mean(axis=1).to_numpy(),rng,resamples)
    run_means = pivot.mean(axis=0)
    conditional.update(matched_days=len(pivot), trained_runs=3,
        between_run_sample_std=float(run_means.std(ddof=1)) if len(pivot) else np.nan,
        individual_run_means=json.dumps(run_means.to_dict()) if len(pivot) else None)
    return conditional


def compatibility(a, b):
    """Check each shared topology's physical/accounting identities before pairing."""
    for tp in [f'TP{i}' for i in range(1,8)]:
        ae = [e for e in a.get('environments',[]) if e.get('topology') == tp]
        be = [e for e in b.get('environments',[]) if e.get('topology') == tp]
        if not ae or not be: raise ValueError(f'Missing {tp} environment provenance')
        x,y = ae[0],be[0]
        for key in ('mode','data_scaling','doe','backup','reward','battery_constraints','network',
                    'normalization_identity','active_current_ratings_a','voltage_tolerance_pu',
                    'current_relative_tolerance','available_episode_start_indices','expected_steps_per_episode'):
            if key not in x or x[key] != y.get(key): raise ValueError(f'Incompatible {tp}: {key}')
        for key,rec in x['input_files'].items():
            if rec.get('sha256') != y['input_files'].get(key,{}).get('sha256'):
                raise ValueError(f'Incompatible input file: {tp}/{key}')
    if a.get('seed') != b.get('seed'):
        raise ValueError('Evaluation reset seeds differ (manifest seed+i must match)')


def read_evaluation(directory, label, run, dataset=None):
    directory = Path(directory)
    man = read_json(directory/'run_manifest.json')
    proto = man.get('evaluation_protocol',{})
    if man.get('status') != 'finished' or proto.get('data_split') != 'test' or proto.get('final_test') is not True:
        raise ValueError(f'{directory}: not a finished authorized final evaluation')
    if label in ('A1','A2'):
        expected = 'no_shared_kpi' if label=='A1' else 'no_centralized_critic'
        if man.get('ablation') != expected or man.get('run_id') != run: raise ValueError('Ablation/run identity mismatch')
    if label == 'A0':
        if man.get('controller','').upper() != 'MLP': raise ValueError('A0 is not MLP')
        sel = (proto.get('selection') or {})
        # Validate run identity from immutable selection id or the actual record.
        sid = sel.get('selection_id','')
        if f'run_{run}_' not in sid: raise ValueError('A0 selection does not identify the requested run')
    step = pd.read_csv(directory/'step_metrics.csv')
    ep = pd.read_csv(directory/'episode_summary.csv')
    policy = label if label in ('zero','random') else 'mlp_maddpg'
    ep = ep[ep.policy == policy].copy()
    step = step[step.policy == policy].copy()
    if ep.duplicated(['topology_case','start_index']).any(): raise ValueError('Duplicate episode/day')
    expected = {f'TP{i}' for i in range(1,8)}
    if set(ep.topology_case) != expected: raise ValueError('Require all TP1-TP7 results, including failed episode rows')
    for tp,g in ep.groupby('topology_case'):
        if len(g) != 31: raise ValueError(f'{directory}/{tp}: intended 31 days, found {len(g)}')
        env=next(e for e in man['environments'] if e['topology']==tp)
        if set(g.start_index) != set(env['available_episode_start_indices']):
            raise ValueError('Recorded days differ from the final-test manifest')
    timestamps=pd.to_datetime(step.loc[step.step>=0,'date_time'],utc=True)
    if not ((timestamps.dt.year==2018)&(timestamps.dt.month==12)).all():
        raise ValueError('Final-test steps must be December 2018 only')
    if step.duplicated(['topology_case','start_index','step']).any(): raise ValueError('Duplicate evaluation timestep')
    valid = bool_values(ep['economic_totals_valid'])
    ep['complete'] = valid
    ep['raw_energy_cost'] = pd.to_numeric(ep['total_energy_cost'], errors='coerce').where(valid)
    if 'adjusted_energy_cost' not in ep:
        ep['adjusted_energy_cost'] = np.nan
        ep['initial_stored_energy_kwh'] = np.nan
        ep['terminal_stored_energy_kwh'] = np.nan
        prices = None
        record = man['environments'][0]['input_files']['processed_time_series']
        path = Path(dataset) if dataset else Path(record['path'])
        if path.is_file() and sha(path) == record.get('sha256'):
            scale = man['environments'][0]['data_scaling'].get('price_scaling_factor',1.0)
            prices = pd.read_csv(path,usecols=['price'])['price'].to_numpy(float)*scale
        for index, row in ep.iterrows():
            g = step[(step.topology_case==row.topology_case)&(step.start_index==row.start_index)].sort_values('step')
            if not row.complete or len(g)!=96 or 'soc_before' not in g: continue
            constraints = next(e for e in man['environments'] if e['topology']==row.topology_case)['battery_constraints']
            caps = np.array([b['capacity_kwh'] for b in constraints])
            cols = [f'soc_agent_{i}' for i in range(len(caps))]
            if not all(c in g for c in cols) or not isinstance(g.iloc[0].soc_before,str): continue
            initial = np.asarray(ast.literal_eval(g.iloc[0].soc_before),float)
            terminal = g.iloc[-1][cols].to_numpy(float)
            e0,e1 = float(initial@caps),float(terminal@caps)
            ep.loc[index,'initial_stored_energy_kwh'] = e0
            ep.loc[index,'terminal_stored_energy_kwh'] = e1
            for i in range(len(caps)):
                for name,val in [('initial_soc',initial[i]),('terminal_soc',terminal[i]),
                    ('initial_energy_kwh',initial[i]*caps[i]),('terminal_energy_kwh',terminal[i]*caps[i])]:
                    ep.loc[index,f'{name}_agent_{i}'] = val
            if prices is not None:
                start = int(row.start_index)
                ep.loc[index,'adjusted_energy_cost'] = inventory_adjusted_cost(row.raw_energy_cost,e0,e1,prices[start:start+96].mean())
    elif man.get('adjusted_cost_definition') != COST_DEFINITION:
        raise ValueError('Unknown adjusted-cost definition')
    ep['net_stored_energy_change_kwh'] = ep.terminal_stored_energy_kwh-ep.initial_stored_energy_kwh
    # Retrospective magnitude accounting from recorded action stages only; missing stays NaN.
    for column,left,right in [('mean_backup_adjustment_kw','executed_battery_power_kw','doe_power_kw'),
                              ('mean_physical_clipping_kw','policy_requested_power_kw','physical_limited_power_kw')]:
        if column in ep: continue
        ep[column]=np.nan
        if left not in step or right not in step: continue
        for index,row in ep.iterrows():
            g=step[(step.topology_case==row.topology_case)&(step.start_index==row.start_index)]
            values=[]
            for a,b in zip(g[left],g[right]):
                if isinstance(a,str) and isinstance(b,str):
                    x,y=np.asarray(json.loads(a),float),np.asarray(json.loads(b),float)
                    if np.isfinite(x).all() and np.isfinite(y).all(): values.extend(np.abs(x-y).tolist())
            if values: ep.loc[index,column]=np.mean(values)
    ep['adjusted_energy_cost'] = ep.adjusted_energy_cost.where(valid)
    ep['configuration'],ep['run'] = label,run
    return ep, man


def analyze(table, output, resamples=10000, seed=20261025):
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    rng = np.random.default_rng(seed)
    gaps, pairs, zero_rows, absolute_pairs = [], [], [], []
    metrics = [m for m in COSTS+PHYSICAL if m in table]
    for metric in metrics:
        for (label,run), group in table.groupby(['configuration','run']):
            base = group[group.topology_case=='TP1'].set_index('start_index')
            for tp, g in group.groupby('topology_case'):
                for _,r in g.iterrows():
                    b = base.loc[r.start_index]
                    val = float(r[metric]) if r.complete else np.nan
                    bv = float(b[metric]) if b.complete else np.nan
                    gaps.append(dict(configuration=label,run=int(run),topology_case=tp,start_index=int(r.start_index),
                        metric=metric,cost_or_metric=val,tp1_value=bv,gap=val-bv))
    gaps = pd.DataFrame(gaps)
    a0 = gaps[gaps.configuration=='A0'].set_index(['run','topology_case','start_index','metric'])
    zero = gaps[gaps.configuration=='zero'].set_index(['topology_case','start_index','metric'])
    for _,r in gaps[gaps.configuration.isin(['A0','A1','A2'])].iterrows():
        key = (r.run,r.topology_case,r.start_index,r.metric)
        b = a0.loc[key]
        common = dict(configuration=r.configuration,run=r.run,topology_case=r.topology_case,start_index=r.start_index,metric=r.metric)
        absolute_pairs.append(dict(common,difference=r.cost_or_metric-b.cost_or_metric))
        if r.configuration != 'A0' and r.topology_case != 'TP1':
            pairs.append(dict(common,difference=r.gap-b.gap,ablation_value=r.cost_or_metric,a0_value=b.cost_or_metric))
        z = zero.loc[(r.topology_case,r.start_index,r.metric)]
        zero_rows.append(dict(common,delta_gap_zero=r.gap-z.gap,delta_cost_zero=r.cost_or_metric-z.cost_or_metric,
            percentage_cost_vs_zero=100*(r.cost_or_metric-z.cost_or_metric)/abs(z.cost_or_metric) if abs(z.cost_or_metric)>1e-12 else np.nan,
            percentage_gap_vs_zero=100*(r.gap-z.gap)/abs(z.gap) if abs(z.gap)>1e-12 else np.nan))
    pairs = pd.DataFrame(pairs)
    paired_stats, pooled = [], []
    comparison_frames = [('delta_generalization_gap',pairs),('absolute_difference_vs_A0',pd.DataFrame(absolute_pairs)),
        ('absolute_value',gaps[gaps.configuration.isin(['A0','A1','A2'])].rename(columns={'cost_or_metric':'difference'}))]
    zdf = pd.DataFrame(zero_rows)
    for field in ('delta_gap_zero','delta_cost_zero'):
        comparison_frames.append((field,zdf.rename(columns={field:'difference'})))
    for comparison,frame in comparison_frames:
        for (label,tp,metric,run),g in frame.groupby(['configuration','topology_case','metric','run']):
            paired_stats.append(dict(comparison=comparison,configuration=label,topology_case=tp,metric=metric,run=run,
                intended_days=31,**stats(g.difference,rng,resamples)))
        for (label,tp,metric),g in frame.groupby(['configuration','topology_case','metric']):
            pooled.append(dict(comparison=comparison,configuration=label,topology_case=tp,metric=metric,
                **pooled_stats(g,rng,resamples)))
    gaps.to_csv(output/'ablation_generalization_gaps.csv',index=False)
    pairs.to_csv(output/'ablation_paired_daily_differences.csv',index=False)
    pd.DataFrame(paired_stats).to_csv(output/'ablation_paired_bootstrap_ci.csv',index=False)
    pd.DataFrame(pooled).to_csv(output/'ablation_three_seed_summary.csv',index=False)
    zdf.to_csv(output/'ablation_vs_zero_baseline.csv',index=False)
    table.to_csv(output/'absolute_performance_and_physical_metrics.csv',index=False)
    table.to_csv(output/'terminal_soc_summary.csv',index=False)
    table[['configuration','run','topology_case','start_index',*COSTS,'complete']].to_csv(output/'adjusted_cost_summary.csv',index=False)
    counts = table.groupby(['configuration','run','topology_case']).agg(intended_days=('complete','size'),
        completed_days=('complete','sum'),adjusted_cost_valid_days=('adjusted_energy_cost','count')).reset_index()
    counts['failed_or_incomplete_days'] = counts.intended_days-counts.completed_days
    counts.to_csv(output/'availability_and_failures.csv',index=False)
    pd.DataFrame(pooled).to_html(output/'publication_summary.html',index=False,float_format=lambda x:f'{x:.6g}',na_rep='N/A')
    write_json(output/'analysis_manifest.json',dict(seed=seed,resamples=resamples,cost_definition=COST_DEFINITION,
        sign='positive delta gap: worse transfer than A0; positive absolute cost difference: more expensive',
        method='Paired days per run; pooled complete shared day clusters across three fixed trained policies',
        uncertainty='Conditional on three trained seeds; between-run sample SD also reported. Not 93 independent seeds.',
        missing='Missing SOC or hash-verified price data yields unavailable adjusted cost, never a fabricated value'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',nargs=3,action='append',required=True,metavar=('LABEL','RUN','DIRECTORY'))
    p.add_argument('--zero',required=True)
    p.add_argument('--random')
    p.add_argument('--dataset',help='Optional relocated processed CSV; must match manifest sha256')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resamples',type=int,default=10000)
    p.add_argument('--seed',type=int,default=20261025)
    a=p.parse_args()
    if a.resamples<10000: p.error('Use at least 10,000 resamples')
    allowed=ROOT/'results/ablations/ieee33'
    if not a.output.resolve().is_relative_to(allowed.resolve()): p.error('Output must be under results/ablations/ieee33')
    expected={(l,r) for l in ('A0','A1','A2') for r in range(3)}
    if len(a.evaluation)!=9 or {(l,int(r)) for l,r,_ in a.evaluation} != expected:
        p.error('Provide exactly A0,A1,A2 for runs 0,1,2')
    frames, identities, reference = [], [], None
    for label,run,path in a.evaluation+ [('zero',-1,a.zero)]+([('random',-1,a.random)] if a.random else []):
        frame,man=read_evaluation(path,label,int(run),a.dataset)
        if reference is None: reference=man
        else: compatibility(reference,man)
        frames.append(frame)
        identities.append(dict(configuration=label,run=int(run),path=path,
            manifest_sha256=sha(Path(path)/'run_manifest.json'),steps_sha256=sha(Path(path)/'step_metrics.csv')))
    analyze(pd.concat(frames,ignore_index=True),a.output,a.resamples,a.seed)
    write_json(a.output/'input_identities.json',identities)


if __name__=='__main__': main()
