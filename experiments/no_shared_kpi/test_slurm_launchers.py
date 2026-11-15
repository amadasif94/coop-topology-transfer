"""Local static/dispatch tests. These do not claim a successful Slurm launch."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from experiments.no_shared_kpi.protocol import ROOT
from experiments.no_shared_kpi.launcher_preflight import check_map

HERE = Path(__file__).parent


class Launchers(unittest.TestCase):
    def test_training_resources_match_a0(self):
        original=(ROOT/'SLURM Script/run_mlp_maddpg_ieee33.slurm').read_text()
        keys=('partition','qos','array','nodes','ntasks','cpus-per-task','mem','gres','time')
        def value(s,k):return next(l for l in s.splitlines() if l.startswith('#SBATCH --'+k+'='))
        for variant in ('no_shared_kpi','no_centralized_critic'):
            s=(ROOT/'experiments'/variant/f'train_{variant}_ieee33.slurm').read_text()
            for k in keys:self.assertEqual(value(s,k),value(original,k))

    def test_argument_transport_and_early_marker(self):
        for variant in ('no_shared_kpi','no_centralized_critic'):
            for kind,default in [('train','train'),('eval','evaluate')]:
                p=ROOT/'experiments'/variant/f'{kind}_{variant}_ieee33.slurm'
                raw=p.read_bytes(); s=raw.decode()
                self.assertTrue(raw.startswith(b'#!/bin/bash\n'))
                self.assertNotIn(b'\r',raw)
                self.assertIn('#SBATCH --export=ALL\n',s)
                self.assertIn('${1:-'+default+'}',s)
                self.assertIn('${2:-experiments/no_shared_kpi/cluster_job_map.json}',s)
                self.assertNotIn('${STAGE',s);self.assertNotIn('${JOB_MAP',s)
                self.assertLess(s.index('ABLATION_BATCH_ENTERED'),s.index('exec /bin/bash'))

    def test_shell_only_check_precedes_conda_python(self):
        s=(HERE/'launch_ablation.sh').read_text()
        self.assertLess(s.index('LAUNCH_CHECK_PASS'),s.index('source /network/'))
        self.assertLess(s.index('exit 0'),s.index('python --version'))
        self.assertIn('exec python -u -m experiments.no_shared_kpi.slurm_dispatch',s)
        self.assertIn('"$STAGE" --run "$RUN_ID" --job-map "$JOB_MAP"',s)

    def test_preflight_selects_exact_map_entry_without_training(self):
        with tempfile.TemporaryDirectory(dir=HERE,prefix='launcher_test_') as tmp:
            p=Path(tmp)/'map.json'
            p.write_text(json.dumps({'runs':{'2':{'baseline_training_manifest':'verified_manifest.json'}}}))
            with patch('experiments.no_shared_kpi.launcher_preflight.load_baseline',return_value={
                'seed':44,'arguments':{'total_steps':200000}}) as load:
                result=check_map(p,2)
                load.assert_called_once_with('verified_manifest.json',2)
            self.assertEqual(result['seed'],44);self.assertEqual(result['total_steps'],200000)
            self.assertEqual(result['job_map_sha256'],hashlib.sha256(p.read_bytes()).hexdigest())
            with self.assertRaises(KeyError):check_map(p,0)

    def test_original_sources_unchanged(self):
        hashes=json.loads((HERE/'launcher_original_hashes.json').read_text())
        for name,h in hashes.items():self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),h,name)


if __name__=='__main__':unittest.main(verbosity=2)
