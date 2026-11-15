"""Matched original MADDPG versus corrected B3; existing results are read only.

Unverified legacy outputs are listed in exclusions.json; they are never relabeled.
Use the strict frozen evaluator to fill missing metadata/measurements. Do not infer
pre-physical requests from requested_action (which can be DOE-conditioned).
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .metrics import METRICS, summary
from .provenance import paired_identity, verify_finished


def complete_inventory(frame, identity):
    expected=pd.DataFrame([dict(topology_case=tp,**e) for tp,plan in identity['episode_plans'].items() for e in plan])
    keys=['topology_case','start_index']
    if expected.duplicated(keys).any() or expected.duplicated(['topology_case','date']).any():
        raise ValueError('Duplicate day in intended episode plan')
    if frame.duplicated(keys).any(): raise ValueError('Duplicate episode in report')
    if len(frame.merge(expected[keys],on=keys,how='inner')) != len(frame):
        raise ValueError('Unexpected evaluation day')
    recorded=frame.copy()
    if 'date' in recorded:
        check=recorded.merge(expected,on=keys,suffixes=('_recorded','_expected'))
        known=check.date_recorded.notna()
        if not check.loc[known,'date_recorded'].eq(check.loc[known,'date_expected']).all():
            raise ValueError('Date/time-series index mismatch')
        recorded=recorded.drop(columns='date')
    out=expected.merge(recorded,on=keys,how='left',validate='one_to_one',indicator=True)
    missing=out['_merge'].eq('left_only')
    out['missing_episode_record']=missing.astype(int)
    out.loc[missing,'intended_days']=1
    out.loc[missing,'completed_days']=0
    out.loc[missing,'incomplete_days']=1
    out.loc[missing,'failed_days']=0  # unknown, separately counted; not a fabricated PF failure
    out.loc[missing,'failure_status_unknown_days']=1
    for col in ('power_valid_decisions','power_invalid_decisions'):
        out.loc[missing,col]=0
    n=len(next(iter(identity['environments'].values()))['battery_constraints'])
    out.loc[missing,'power_missing_decisions']=96*n
    return out.drop(columns='_merge')


def pair_days(original, corrected):
    keys=['topology_case','date','start_index']
    paired=original.merge(corrected,on=keys,how='outer',suffixes=('_original','_b3'),validate='one_to_one')
    rows=[]
    for tp,g in paired.groupby('topology_case'):
        complete=g.completed_days_original.eq(1)&g.completed_days_b3.eq(1)
        for metric in METRICS:
            a=pd.to_numeric(g[metric+'_original'],errors='coerce')
            b=pd.to_numeric(g[metric+'_b3'],errors='coerce')
            valid=complete & np.isfinite(a)&np.isfinite(b)
            d=b[valid]-a[valid]
            rows.append(dict(topology_case=tp,metric=metric,intended_paired_days=len(g),
                paired_valid_days=int(valid.sum()),unpaired_or_invalid_days=int((~valid).sum()),
                paired_coverage=float(valid.mean()),original_matched_mean=float(a[valid].mean()),
                b3_matched_mean=float(b[valid].mean()),mean_difference_b3_minus_original=float(d.mean()),
                paired_day_sd=float(d.std(ddof=1))))
    return paired,pd.DataFrame(rows)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--original',nargs='+',type=Path,help='strict test/load_x*/ directories')
    p.add_argument('--b3',nargs='+',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--inspect-historical',nargs='+',type=Path,help='Read headers/manifests only; print reconstruction readiness')
    a=p.parse_args(argv)
    if a.inspect_historical:
        for path in a.inspect_historical:
            columns=set(pd.read_csv(path/'step_metrics.csv',nrows=0).columns)
            print(json.dumps(dict(path=str(path),
                direct_power_fields_available={'policy_requested_power_kw','physical_limited_power_kw'}<=columns,
                original_env_power_fields_available={'marl_requested_power_kw','physical_limited_power_kw'}<=columns,
                normalized_proposal_available='policy_proposal_action_normalized' in columns,
                note='Headers alone do not verify values or provenance. Missing original requests require frozen re-evaluation; requested_action/accepted_action are NOT substitutes.'),indent=2))
        return
    if not a.original or not a.b3 or not a.output:
        p.error('comparison requires --original, --b3 and --output')
    if a.output.exists(): raise FileExistsError('Choose a NEW comparison directory')
    records={};excluded=[]
    for side,paths in [('original',a.original),('b3',a.b3)]:
        for path in paths:
            m=json.loads((path/'run_manifest.json').read_text())
            identity=m.get('scientific_identity')
            if not identity:
                excluded.append(dict(path=str(path),reason='unverified identity; frozen re-evaluation required'))
                continue
            wanted='original_maddpg' if side=='original' else 'b3_pre_tanh'
            if identity.get('experiment_class') != wanted:
                excluded.append(dict(path=str(path),reason='not primary experiment class '+wanted));continue
            verify_finished(path,identity)
            if identity['protocol']['stage'] != 'test': raise ValueError('Comparison requires final test outputs')
            key=(identity['feeder'],identity['controller'],identity['run'],identity['protocol']['load_scale'])
            if (side,key) in records: raise ValueError('Duplicate experiment identity')
            records[side,key]=(identity,complete_inventory(pd.read_csv(path/'scientific_episodes.csv'),identity))
    coverage=[];per_seed=[];days=[];missing=[]
    keys=sorted({key for _,key in records})
    for key in keys:
        if ('original',key) not in records or ('b3',key) not in records:
            missing.append(dict(identity=key,reason='missing verified counterpart'));continue
        oi,o=records['original',key];bi,b=records['b3',key]
        paired_identity(oi,bi)
        metadata=dict(feeder=key[0],controller=key[1],run=key[2],seed=42+key[2],load_scale=key[3])
        for side,f in [('original',o),('b3',b)]:
            for tp,g in f.groupby('topology_case'):
                coverage.append(dict(**metadata,side=side,topology_case=tp,
                    missing_episode_records=int(g.missing_episode_record.sum()),**summary(g)))
        paired,stats=pair_days(o,b)
        for f in (paired,stats):
            for k,v in metadata.items(): f[k]=v
        days.append(paired);per_seed.append(stats)
    a.output.mkdir(parents=True)
    (a.output/'exclusions.json').write_text(json.dumps(dict(unverified_or_secondary=excluded,missing_pairs=missing),indent=2))
    if not per_seed:
        raise SystemExit('No verified pairs; see exclusions.json. No scientific comparison produced.')
    by_seed=pd.concat(per_seed,ignore_index=True)
    by_seed.to_csv(a.output/'per_seed.csv',index=False)
    pd.concat(days,ignore_index=True).to_csv(a.output/'paired_days.csv',index=False)
    pd.DataFrame(coverage).to_csv(a.output/'full_failure_and_metric_coverage.csv',index=False)
    rows=[]
    for key,g in by_seed.groupby(['feeder','controller','load_scale','topology_case','metric']):
        # Equal seed weight. Day-level SD is not a substitute for training-seed SD.
        valid=np.isfinite(g.mean_difference_b3_minus_original)
        rows.append(dict(zip(['feeder','controller','load_scale','topology_case','metric'],key),
            mean_difference=g.loc[valid,'mean_difference_b3_minus_original'].mean(),
            between_seed_sd=g.loc[valid,'mean_difference_b3_minus_original'].std(ddof=1),
            original_mean=g.loc[valid,'original_matched_mean'].mean(), b3_mean=g.loc[valid,'b3_matched_mean'].mean(),
            valid_seeds=int(valid.sum()), expected_seeds=3, all_three_seeds_present=set(g.run)=={0,1,2},
            paired_valid_days=int(g.paired_valid_days.sum()),intended_paired_days=int(g.intended_paired_days.sum())))
    pd.DataFrame(rows).to_csv(a.output/'across_seed.csv',index=False)
    print('Comparison written:',a.output,'(coverage columns must accompany every mean)')


if __name__=='__main__': main()
