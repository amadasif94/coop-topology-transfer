"""Run one explicit job-map entry. No guessed checkpoint/manifest paths."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',required=True,choices=['no_shared_kpi','no_centralized_critic'])
    p.add_argument('--stage',required=True,choices=['validate','train','sweep','select','evaluate','analysis'])
    p.add_argument('--run',required=True,type=int,choices=[0,1,2])
    p.add_argument('--job-map',required=True,type=Path)
    a=p.parse_args()
    data=json.loads(a.job_map.read_text())
    entry=data['runs'][str(a.run)]
    if a.stage=='validate':
        stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        cmd=[sys.executable,'-m',f'experiments.{a.variant}.validate_{a.variant}','--real-smoke',
            '--output',f'results/ablations/ieee33/{a.variant}/run_{a.run}/smoke_{stamp}']
    elif a.stage=='analysis':
        if a.run!=0:raise ValueError('Submit analysis with --array=0 only')
        cmd=[sys.executable,'-m','experiments.no_shared_kpi.compare_ablations_ieee33',*data['analysis_arguments']]
    else:
        cmd=[sys.executable,'-m',f'experiments.{a.variant}.train_{a.variant}_ieee33',
            '--stage',a.stage,'--run',str(a.run),'--baseline-training-manifest',entry['baseline_training_manifest']]
        if a.stage=='evaluate':
            cmd+=['--baseline-evaluation-manifest',entry['baseline_evaluation_manifest'],
                  '--selection',entry['selections'][a.variant]]
    print('Executing:',json.dumps(cmd),flush=True)
    subprocess.run(cmd,check=True)


if __name__=='__main__':main()
