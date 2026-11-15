"""Final topology coverage from real A0 records, including supplemental TP10 runs.

No synthetic final manifest is created. Each topology keeps its original A0
manifest, snapshot bytes, episode starts and reset seeds as reference evidence.
"""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import pandas as pd
from . import protocol as p


def snapshot(manifest_path, topology, key, record):
    filename={'line_data':'line_data.csv','gridtensor_line_data':'gridtensor_line.csv'}[key]
    candidates=[Path(manifest_path).parent/'network_inputs'/topology/filename,Path(record['path'])]
    for candidate in candidates:
        if candidate.is_file() and p.sha(candidate)==record['sha256']: return candidate
    raise FileNotFoundError(f'Missing hash-matching A0 {topology} {key} snapshot beside {manifest_path}')


def references(paths, run):
    from configs.ieee69_config import IEEE69_CONFIG
    from experiments.topology_generalization.tp_configs_ieee69 import get_topology_case
    from experiments.topology_generalization.topology_utils_ieee69 import validate_topology_files, _apply_status_changes
    refs={}; checkpoint_hashes=set()
    for path in paths:
        path=Path(path); m=deepcopy(p.load_baseline(path,run,training=False))
        # Historical manifests repeated each environment. Deduplicate only
        # identical JSON values within this manifest; never merge across files.
        unique={}; identities={}
        for entry in m['environments']:
            tp=entry['topology']
            identity=json.dumps(entry,sort_keys=True,separators=(',',':'),allow_nan=False)
            if tp in identities and identities[tp]!=identity:
                raise ValueError('Conflicting A0 environment records for '+tp)
            identities[tp]=identity; unique[tp]=entry
        m['environments']=list(unique.values())
        ck=(m.get('checkpoint_file') or {}).get('sha256') or (m['evaluation_protocol'].get('selection') or {}).get('checkpoint_sha256')
        if not ck: raise ValueError('A0 final checkpoint identity missing')
        checkpoint_hashes.add(ck)
        steps=pd.read_csv(path.parent/'step_metrics.csv',usecols=['topology_case','policy','episode_id','start_index','step','date_time'])
        for entry in m['environments']:
            tp=entry['topology']
            if tp in refs: raise ValueError('Ambiguous A0 records for '+tp+'; supply a narrower discovery root')
            g=steps[(steps.topology_case==tp)&(steps.policy=='mlp_maddpg')].copy()
            if len(g)!=31*96: raise ValueError('Require 31 full A0 December days for '+tp)
            plan=[]; dates=set()
            for ep,day in g.groupby('episode_id',sort=True):
                day=day.sort_values('step'); timestamps=pd.to_datetime(day.date_time,utc=True)
                if len(day)!=96 or day.start_index.nunique()!=1 or day.step.tolist()!=list(range(96)):
                    raise ValueError('Incomplete/ambiguous A0 day')
                if not (timestamps.dt.month==12).all() or timestamps.dt.date.nunique()!=1 or not (timestamps.diff().dropna()==pd.Timedelta(minutes=15)).all():
                    raise ValueError('A0 final timestamps do not describe a complete December day')
                if timestamps.iloc[0]!=timestamps.iloc[0].normalize(): raise ValueError('A0 day not at midnight')
                dates.add(timestamps.iloc[0].date())
                plan.append((int(ep),int(day.start_index.iloc[0]),int(m['seed'])+int(ep)))
            if len(plan)!=31 or len({x[1] for x in plan})!=31 or len(dates)!=31 or len({d.year for d in dates})!=1:
                raise ValueError('Duplicate or mixed-year A0 dates')
            if [e for e,_,_ in plan]!=list(range(31)): raise ValueError('Unexpected A0 episode identifiers')
            # Original evaluator uses seed + episode_id. If explicit plans exist,
            # require that they agree rather than silently replacing them.
            explicit=m.get('episode_plans',{}).get(tp)
            if explicit and [(int(x['start_index']),int(x['seed'])) for x in explicit]!=[(s,z) for _,s,z in plan]:
                raise ValueError('A0 recorded reset schedule differs from evaluator convention')
            files={k:snapshot(path,tp,k,entry['input_files'][k]) for k in ('line_data','gridtensor_line_data')}
            line,gt=(pd.read_csv(files[k]) for k in ('line_data','gridtensor_line_data'))
            case=get_topology_case(tp)
            expected_line,expected_gt=_apply_status_changes(
                pd.read_csv(IEEE69_CONFIG['paths']['line_data']),pd.read_csv(IEEE69_CONFIG['paths']['gridtensor_line_data']),
                case['open_edges'],case['close_edges'])
            validate_topology_files(line,gt,69)
            for a,b,flag in ((line,expected_line,'status'),(gt,expected_gt,'STATUS')):
                pd.testing.assert_frame_equal(a[a[flag]==1].reset_index(drop=True),b[b[flag]==1].reset_index(drop=True),check_dtype=False,check_exact=True)
            refs[tp]=dict(path=path,manifest=m,entry=entry,files=files,plan=plan,case=case,
                          timestamps=g.sort_values(['episode_id','step']).date_time.tolist())
    if not set(p.TOPOLOGIES)<=set(refs): raise ValueError('A0 final references must cover TP1-TP7 and TP10; TP8/TP9 are preserved when present')
    if len(checkpoint_hashes)!=1: raise ValueError('A0 supplemental evaluations used different checkpoints')
    return {tp:refs[tp] for tp in sorted(refs,key=lambda x:int(x[2:]))}


def evaluate_final(variant,args):
    import torch
    from configs.ieee69_config import IEEE69_CONFIG
    from utility.evaluation_protocol import authorize, protocol_config
    from .runner import RunManifest, make_primary_env, evaluate_one_episode, summarize, energy_rows, namespace
    from .models import model_class
    refs=references(args.baseline_evaluation_manifest,args.run)
    protocol=authorize(mode='test',final=True,run_purpose='final_test',learned=True,topologies=list(refs),selection=args.selection,checkpoint=args.checkpoint)
    checkpoint=args.checkpoint or protocol['selection']['checkpoint_path']
    ck=torch.load(checkpoint,map_location='cpu',weights_only=False)
    if ck.get('specification')!=p.feature_spec(variant) or ck.get('experiment_metadata',{}).get('run_id')!=args.run:
        raise ValueError('Wrong ablation/run checkpoint')
    if ck['experiment_metadata'].get('a0_manifest_sha256')!=p.sha(args.baseline_training_manifest): raise ValueError('Wrong A0 training identity')
    selection=p.read_json(args.selection)
    if selection['model']!=variant or selection['run']!=f'run_{args.run}' or selection['feeder']!='ieee69': raise ValueError('Wrong frozen selection')
    out=p.result_root(variant,args.run)/'final_test'
    run=RunManifest(out,'MLP',refs['TP1']['manifest']['seed'],'test',checkpoint,vars(args))
    namespace['annotate'](run,variant,args.run,args.baseline_training_manifest)
    run.record_protocol(protocol,['mlp_maddpg'])
    run.data['a0_topology_references']={tp:dict(path=str(ref['path']),sha256=p.sha(ref['path'])) for tp,ref in refs.items()}
    frames=[]; energies=[]
    for tp,ref in refs.items():
        cfg=deepcopy(IEEE69_CONFIG)
        cfg['paths'].update(ref['files'])  # read-only exact archived A0 network bytes
        cfg['topology_case']=dict(case_name=tp,**ref['case'])
        env=make_primary_env(protocol_config(cfg,protocol),'test',ref['manifest']['seed'])
        if [s for _,s,_ in ref['plan']]!=list(env.episode_start_indices): raise ValueError('A0 episode indices differ from current December dataset')
        run.add_environment(env,topology=tp,intended_episodes=31)
        p.compare_environment(run.data['environments'][-1],ref['manifest'])
        model=model_class(variant)(**dict(ck['model_kwargs'],device='cpu' if args.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')))
        model.load(checkpoint,env); model.actor.eval()
        rows=[]
        for ep,start,seed in ref['plan']: rows.extend(evaluate_one_episode(env,model,'mlp_maddpg',tp,ep,start,seed))
        frame=pd.DataFrame(rows)
        if frame.episode_completed.all() and not np.array_equal(pd.to_datetime(frame.date_time,utc=True),pd.to_datetime(ref['timestamps'],utc=True)):
            raise ValueError('Ablation timestamps differ from A0 final reference')
        frame['run_id']=args.run; frame['ablation']=variant
        frames.append(frame); energies.append(energy_rows(frame,env))
        run.data.setdefault('episode_plans',{})[tp]=[dict(episode_id=e,start_index=s,seed=z) for e,s,z in ref['plan']]
        run._write()
    steps=pd.concat(frames,ignore_index=True); ep,agg=summarize(steps); energy=pd.concat(energies,ignore_index=True)
    ep=ep.merge(energy,on=['topology_case','episode_id','start_index'],validate='one_to_one')
    ep['run_id'],ep['ablation']=args.run,variant
    extras=ep.groupby('topology_case')[['raw_energy_cost','adjusted_energy_cost','terminal_stored_energy_kwh','mean_backup_adjustment_kw','mean_physical_clipping_kw']].mean().add_prefix('mean_').reset_index()
    agg=agg.merge(extras,on='topology_case',validate='one_to_one');agg['run_id'],agg['ablation']=args.run,variant
    for name,frame in [('step_metrics',steps),('episode_summary',ep),('aggregate_summary',agg),('terminal_soc_summary',energy),('adjusted_cost_summary',energy)]:
        frame.to_csv(out/f'{name}.csv',index=False)
    run.finish();return out
