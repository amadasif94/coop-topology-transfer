"""Evaluation-only compatibility for two approved IEEE33 source additions.

Run from the repository root. --verify-all audits 38 checkpoints in each of
six runs without importing PyTorch. Otherwise use the dispatcher's --variant,
--stage {sweep,select,evaluate}, --run and --job-map arguments.
No output reuse/overwrite policy is changed here.
"""
import argparse
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
MODULE_SOURCE = 'experiments/no_shared_kpi/ieee33_eval_compat.py'
MODULE_PIN = Path(__file__).with_suffix('.sha256')
APPROVED_ADDITIONS = {
    'experiments/no_shared_kpi/launcher_preflight.py':
        '1c71758249983d13065cbae011fc0d1551170abbd2b8ea715983de1f6a27ac85',
    'experiments/no_shared_kpi/test_slurm_launchers.py':
        '0a845eafc6e148189ef8dedfd126d47eeee9c531c4eb002f31bd8d79228f1bbe',
}
VARIANTS = ('no_shared_kpi', 'no_centralized_critic')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def validate_inventory(recorded, current):
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError('Missing/empty recorded source inventory')
    for name, expected in APPROVED_ADDITIONS.items():
        if current.get(name) != expected:
            raise ValueError('Unapproved utility bytes: ' + name)
    # Only this new evaluation module may additionally be absent historically.
    # Its raw bytes are pinned in the separately shipped release checksum file;
    # do not infer approval from whatever bytes happen to be installed.
    pin = MODULE_PIN.read_text(encoding='ascii').strip()
    if len(pin) != 64 or any(c not in '0123456789abcdef' for c in pin):
        raise ValueError('Invalid evaluation-module release checksum')
    if sha(ROOT / MODULE_SOURCE) != pin or current.get(MODULE_SOURCE) != pin:
        raise ValueError('Unapproved evaluation-module bytes or inventory hash')
    missing = set(recorded) - set(current)
    extra = set(current) - set(recorded) - set(APPROVED_ADDITIONS) - {MODULE_SOURCE}
    changed = {name for name in recorded if name in current and recorded[name] != current[name]}
    if missing or extra or changed:
        raise ValueError(f'Source mismatch: missing={sorted(missing)}, extra={sorted(extra)}, changed={sorted(changed)}')
    # The evidenced histories contain either both utilities or neither.
    if len(set(recorded) & set(APPROVED_ADDITIONS)) not in (0, 2):
        raise ValueError('Unverified partial utility inventory')
    return dict(recorded)


def verify_checkpoint(path, identity_function):
    path = Path(path)
    sidecar = json.loads(path.with_suffix('.identity.json').read_text(encoding='utf-8-sig'))
    if sha(path) != sidecar.get('sha256'):
        raise ValueError('Checkpoint SHA256 differs from sidecar: ' + str(path))
    return validate_inventory(sidecar.get('sources'), identity_function())


@contextmanager
def compatibility_loader(module):
    """Temporary, per-load view; original load still checks embedded sources.

    Sidecar approval alone is insufficient: the original loader compares the
    embedded checkpoint inventory with this verified inventory and retains all
    specification, hyperparameter, state-dict and environment provenance checks.
    This wrapper is single-threaded and never used by training or IEEE69.
    """
    original_load = module.AblationCheckpoint.load
    original_identity = module.source_identity

    def checked_load(self, path, env=None):
        recorded = verify_checkpoint(path, original_identity)
        previous = module.source_identity
        module.source_identity = lambda: dict(recorded)
        try:
            return original_load(self, path, env)
        finally:
            module.source_identity = previous

    module.AblationCheckpoint.load = checked_load
    try:
        yield
    finally:
        module.AblationCheckpoint.load = original_load
        module.source_identity = original_identity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-all', action='store_true')
    parser.add_argument('--variant', choices=VARIANTS)
    parser.add_argument('--stage', choices=['sweep', 'select', 'evaluate'])
    parser.add_argument('--run', type=int, choices=[0, 1, 2])
    parser.add_argument('--job-map', type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    from experiments.no_shared_kpi.protocol import source_identity
    if args.verify_all:
        count = 0
        for variant in VARIANTS:
            for run in range(3):
                for step in range(15000, 200001, 5000):
                    path = ROOT / f'results/ablations/ieee33/{variant}/run_{run}/training/checkpoints/maddpg_step_{step}.pt'
                    verify_checkpoint(path, source_identity)
                    count += 1
                print(f'PASS {variant} run {run}: 38 checkpoint hashes and source inventories', flush=True)
        print(f'PASS: {count}/228 verified (sidecars and checkpoint bytes; no model execution)')
        return
    if args.variant is None or args.stage is None or args.run is None or args.job_map is None:
        parser.error('Require --variant, --stage, --run and --job-map, or --verify-all')
    entry = json.loads(args.job_map.read_text(encoding='utf-8-sig'))['runs'][str(args.run)]
    forwarded = ['--stage', args.stage, '--run', str(args.run),
                 '--baseline-training-manifest', entry['baseline_training_manifest']]
    if args.stage == 'evaluate':
        forwarded += ['--baseline-evaluation-manifest', entry['baseline_evaluation_manifest'],
                      '--selection', entry['selections'][args.variant]]
    from experiments.no_shared_kpi import model_support, runner
    previous_argv = sys.argv
    try:
        sys.argv = [str(__file__), *forwarded]
        with compatibility_loader(model_support):
            runner.main(args.variant)
    finally:
        sys.argv = previous_argv


if __name__ == '__main__':
    main()
