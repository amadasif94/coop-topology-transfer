"""Shared A1/A2 IEEE69 dispatcher. IEEE33 dispatcher remains untouched."""
import argparse
from datetime import datetime, timezone
import subprocess
import sys
from .protocol import verify_map, result_root, VARIANTS


def command(variant, stage, run, job_map):
    entry = verify_map(job_map, run)
    if stage in ('preflight', 'validate'):
        cmd = [sys.executable, '-m', 'experiments.no_shared_kpi.ieee69.validation',
               '--require-runtime', '--variant', variant, '--baseline-training-manifest', entry['baseline_training_manifest'], '--run', str(run)]
        if stage == 'validate':
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            cmd += ['--real-smoke','--output',str(result_root(variant,run)/f'smoke_{stamp}')]
        return cmd
    cmd = [sys.executable, '-m', f'experiments.{variant}.ieee69.train', '--stage', stage,
           '--run', str(run), '--baseline-training-manifest', entry['baseline_training_manifest']]
    if stage == 'evaluate':
        refs=entry.get('baseline_evaluation_manifests') or ([{'path':entry['baseline_evaluation_manifest']}] if entry.get('baseline_evaluation_manifest') else [])
        if not refs or variant not in entry.get('selections', {}):
            raise ValueError('Final evaluation requires a verified A0 final manifest and frozen ablation selection in this map')
        for ref in refs: cmd+=['--baseline-evaluation-manifest',ref['path']]
        cmd += ['--selection',entry['selections'][variant]]
    return cmd


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=VARIANTS, required=True)
    p.add_argument('--stage', choices=['preflight','validate','train','sweep','select','evaluate'], required=True)
    p.add_argument('--run', type=int, choices=[0,1,2], required=True)
    p.add_argument('--job-map', required=True)
    a = p.parse_args()
    cmd = command(a.variant, a.stage, a.run, a.job_map)
    print('Executing:', cmd, flush=True)
    subprocess.run(cmd, check=True)


if __name__ == '__main__': main()
