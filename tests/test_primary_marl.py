"""Protocol wiring tests; actual optimizers are tested by the cluster smoke test."""
import ast
import copy
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from environments.primary_marl import primary_config, validate_primary_env, DOE, BACKUP


class PrimaryProtocol(unittest.TestCase):
    def config(self):
        return dict(data_scaling=dict(load_scaling_factor=1., pv_scaling_factor=1.),
                    doe=dict(doe_enabled=False), backup=dict(backup_mode='legacy_scaling'),
                    network=dict(s_base_kva=1000., v_base_kv=12.66), reward={'x': 7})

    def test_settings_and_no_input_mutation(self):
        old = self.config()
        saved = copy.deepcopy(old)
        cfg = primary_config(old)
        self.assertEqual(old, saved)
        self.assertEqual(cfg['doe'], DOE)
        self.assertEqual(cfg['backup'], BACKUP)
        for key in ('network', 'data_scaling', 'reward'):
            self.assertEqual(cfg[key], old[key])

    def test_scaling_is_rejected_not_repaired(self):
        for key in ('load_scaling_factor', 'pv_scaling_factor'):
            cfg = self.config()
            cfg['data_scaling'][key] = 1.3
            with self.assertRaises(ValueError):
                primary_config(cfg)

    def test_effective_modes_and_csv_ratings(self):
        cfg = primary_config(self.config())
        pu = np.array([255.]) * np.sqrt(3)*12.66/1000
        env = SimpleNamespace(config=cfg, mode='val', doe_enabled=True, doe_w=.015,
            backup_mode='directional_incremental',
            doe_delta_c=.010, doe_pf_failure_mode='logged_fallback', backup_cfg=cfg['backup'],
            grid=SimpleNamespace(in_service_line_df={'max_current_a': [255.]},
                                 get_line_current_limits_pu=lambda: pu))
        before = pu.copy()
        validate_primary_env(env)
        np.testing.assert_array_equal(pu, before)
        env.doe_enabled = False
        with self.assertRaises(ValueError): validate_primary_env(env)
        env.doe_enabled = True
        env.grid.in_service_line_df['max_current_a'] = [200.]
        with self.assertRaises(ValueError): validate_primary_env(env)

    def test_all_primary_entry_points_and_periodic_validation(self):
        files = list((ROOT/'scripts').glob('run_train_ieee*.py'))
        files += list((ROOT/'scripts').glob('run_eval_ieee*.py'))
        files += list((ROOT/'experiments/no_gnn_maddpg').glob('run_train*.py'))
        files += list((ROOT/'experiments/no_gnn_maddpg').glob('run_eval*.py'))
        files += list((ROOT/'experiments/no_gnn_maddpg').glob('run_tp_eval*.py'))
        files += list((ROOT/'experiments/topology_generalization').glob('run_tp_eval*.py'))
        # Exclude archived copies from active entry-point checks.
        files = [p for p in files if not p.stem.endswith('_old')]
        self.assertEqual(len(files), 12)
        for path in files:
            source = path.read_text()
            tree = ast.parse(source)
            calls = [n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
            self.assertIn('make_primary_env', calls, str(path))
            self.assertNotIn('DESSEnv', calls, str(path))
            if 'run_train' in path.name:
                self.assertTrue(any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                    and n.func.id == 'build_env' and any(k.arg == 'train'
                    and isinstance(k.value, ast.Constant) and k.value.value is False
                    for k in n.keywords) for n in ast.walk(tree)))


if __name__ == '__main__':
    unittest.main()
