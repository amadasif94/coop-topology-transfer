"""Read-only GPU/job-map preflight. Never constructs a trainer or writes results."""
import argparse
import hashlib
import json
from pathlib import Path
from experiments.no_shared_kpi.protocol import load_baseline


def check_map(path, run):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    entry = data['runs'][str(run)]
    baseline = load_baseline(entry['baseline_training_manifest'], run)
    return dict(run=run, seed=baseline['seed'], total_steps=baseline['arguments']['total_steps'],
                job_map_sha256=hashlib.sha256(raw).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=int, choices=(0,1,2), required=True)
    p.add_argument('--job-map', type=Path, required=True)
    args = p.parse_args()
    report = check_map(args.job_map, args.run)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('GPU launcher has no usable CUDA device; refusing silent CPU training')
    # One scalar allocation verifies the CUDA context without running a learning step.
    report.update(torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0),
                  cuda_scalar=float(torch.ones(1, device='cuda').item()))
    print('PREFLIGHT_PASS ' + json.dumps(report, sort_keys=True), flush=True)


if __name__ == '__main__': main()
