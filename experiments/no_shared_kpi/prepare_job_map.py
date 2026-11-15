"""Verify A0 records and create a NEW explicit launch map; no checkpoint guessing."""
import argparse
from pathlib import Path
from experiments.no_shared_kpi.protocol import ROOT, load_baseline, result_root, write_json, VARIANTS, read_json, sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a0-training-root',type=Path,help='Search under this local directory for real finished A0 training manifests')
    p.add_argument('--training-manifest',nargs=2,action='append',default=[],metavar=('RUN','MANIFEST'))
    p.add_argument('--a0-evaluation-root',type=Path,required=True,help='Existing topology_generalization/csv directory')
    p.add_argument('--selection',nargs=3,action='append',default=[],metavar=('VARIANT','RUN','SELECTION_JSON'))
    p.add_argument('--discover-ablation-selections',action='store_true',help='Bind exactly one existing frozen selection per new ablation/run')
    p.add_argument('--zero',type=Path,help='Existing compatible TP1-TP7 zero-action evaluation directory')
    p.add_argument('--analysis-output',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    allowed=[ROOT/'experiments'/v for v in VARIANTS]
    if not any(a.output.resolve().is_relative_to(d.resolve()) for d in allowed):
        p.error('Write job maps only inside the new experiment directories')
    selections={}
    if a.discover_ablation_selections:
        for variant in VARIANTS:
            for run in range(3):
                matches=list((result_root(variant,run)/'selections').rglob('selection.json'))
                if len(matches)!=1:
                    raise ValueError(f'{variant}/run_{run}: found {len(matches)} selection records; supply explicit --selection instead')
                a.selection.append((variant,str(run),str(matches[0])))
    training_paths={int(run):Path(path) for run,path in a.training_manifest}
    if a.a0_training_root:
        for path in a.a0_training_root.rglob('run_manifest.json'):
            m=read_json(path)
            if m.get('controller','').upper()=='MLP' and m.get('mode')=='train' and m.get('seed') in (42,43,44):
                run=int(m['seed'])-42
                if run in training_paths and training_paths[run].resolve()!=path.resolve():
                    raise ValueError(f'Multiple A0 training manifests for run {run}; use explicit --training-manifest')
                training_paths[run]=path
    if set(training_paths)!={0,1,2}:
        p.error('Real A0 training manifests for all three runs are required; discovery found '+str(training_paths))
    for variant,run,path in a.selection:
        if variant not in VARIANTS or int(run) not in (0,1,2): p.error('Bad selection key')
        record=read_json(path)
        if record['model']!=variant or record['run']!=f'run_{run}' or record['data_split']!='val' or record['topology']!='TP1':
            raise ValueError('Selection identity/split mismatch')
        selected=record['selected']
        if sha(selected['checkpoint_path'])!=selected['checkpoint_sha256']: raise ValueError('Selected checkpoint hash mismatch')
        selections[variant,int(run)]=str(Path(path).resolve())
    result=dict(runs={},analysis_arguments=[])
    for run in range(3):
        train=training_paths[run]
        final=a.a0_evaluation_root/f'mlp_ieee33_tp_full_eval_run{run}_best/run_manifest.json'
        # Final directory names were supplied by the user. Training manifests are
        # discovered by CONTENT or supplied explicitly, never invented.
        load_baseline(train,run); load_baseline(final,run,training=False)
        result['runs'][str(run)]=dict(baseline_training_manifest=str(train.resolve()),
            baseline_evaluation_manifest=str(final.resolve()),
            selections={v:selections[v,run] for v in VARIANTS if (v,run) in selections})
        for label,directory in [('A0',final.parent),('A1',result_root(VARIANTS[0],run)/'final_test'),
                                ('A2',result_root(VARIANTS[1],run)/'final_test')]:
            result['analysis_arguments']+=['--evaluation',label,str(run),str(directory.resolve())]
    if a.zero and a.analysis_output:
        result['analysis_arguments']+=['--zero',str(a.zero.resolve()),'--output',str(a.analysis_output.resolve())]
    else:
        result['analysis_arguments']=[]
    write_json(a.output,result)
    print('Verified A0 manifests. Wrote:',a.output)
    print('Bound ablation selections:',len(selections),'of 6; evaluation requires its own bound selection.')


if __name__=='__main__':main()
