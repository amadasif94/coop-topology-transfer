"""IEEE69 static/model tests and optional real CPU smoke; no full training."""
import argparse
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from . import protocol as p

HAS_TORCH = importlib.util.find_spec('torch') is not None


class StaticTests(unittest.TestCase):
    def test_private_function_binding(self):
        from .bindings import bind
        namespace = {'value': 33}
        exec('def get(): return value', namespace)
        original = namespace['get']
        other = bind(original, dict(namespace, value=69))
        self.assertEqual(original(), 33)
        self.assertEqual(other(), 69)

    def test_preserved_sources(self):
        for rel, digest in p.read_json(Path(__file__).with_name('preserved_sources.json')).items():
            self.assertEqual(p.sha(p.ROOT/rel), digest, rel)

    def test_python_syntax(self):
        for v in p.VARIANTS:
            for file in (p.ROOT/'experiments'/v/'ieee69').glob('*.py'):
                ast.parse(file.read_text(), filename=str(file))

    def test_nine_agent_specs(self):
        self.assertEqual(p.feature_spec(p.VARIANTS[0])['critic_input_dim'], 99)
        self.assertEqual(p.feature_spec(p.VARIANTS[1])['critic_input_dim'], 11)
        self.assertEqual(p.feature_spec(p.VARIANTS[0])['actor_features'], list(p.FULL_FEATURES[:7]))

    def test_ieee69_validation_schedule(self):
        import pandas as pd
        from types import SimpleNamespace
        from utility.validation_schedule import selection_schedule, periodic_schedule
        dates = pd.date_range('2018-01-01', periods=35040, freq='15min', tz='UTC')
        indices = [i for i,t in enumerate(dates) if t.month in (10,11)]
        env = SimpleNamespace(num_buses=69, mode='val', config={}, episode_length=96,
                              time_series={'df':pd.DataFrame({'time':dates}), 'datetime_col':'time'},
                              splits={'val_indices':indices}, episode_start_indices=indices[::96])
        selection, periodic = selection_schedule(env), periodic_schedule(env)
        self.assertEqual(len(selection['episodes']),61)
        self.assertEqual(len(periodic['episodes']),8)
        actual = p.ROOT/'results/checkpoints/maddpg_ieee69/mlp/mlp_run_0/validation/step_200000/run_manifest.json'
        if actual.exists():
            saved=p.read_json(actual)['validation_schedule']
            self.assertEqual(periodic['sha256'],saved['sha256'])
        env.config={'topology_case':{'case_name':'TP6'}}
        with self.assertRaises(ValueError): selection_schedule(env)

    def test_actual_baselines(self):
        for run in range(3):
            path = p.ROOT/f'results/checkpoints/maddpg_ieee69/mlp/mlp_run_{run}/run_manifest.json'
            if not path.exists(): self.skipTest('A0 training manifest unavailable')
            m = p.load_baseline(path, run)
            self.assertEqual(m['arguments']['total_steps'], 200000)
            self.assertEqual(m['seed'], 42+run)

    def test_reject_wrong_baseline_and_map(self):
        original = p.ROOT/'results/checkpoints/maddpg_ieee69/mlp/mlp_run_0/run_manifest.json'
        if not original.exists(): self.skipTest('A0 manifest unavailable')
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            file = Path(tmp)/'record.json'
            for key, value in [('feeder','IEEE33'), ('backup_mode','legacy_scaling')]:
                m = p.read_json(original); m['environments'][0][key] = value
                file.write_text(json.dumps(m))
                with self.assertRaises(ValueError): p.load_baseline(file, 0)
            file.write_text(json.dumps({'feeder':'IEEE33'}))
            with self.assertRaises(ValueError): p.verify_map(file, 0)

    def test_dispatch_requires_final_artifacts(self):
        from .slurm_dispatch import command
        entry = {'baseline_training_manifest':'manifest.json', 'selections':{}}
        with patch('experiments.no_shared_kpi.ieee69.slurm_dispatch.verify_map', return_value=entry):
            cmd = command(p.VARIANTS[1], 'train', 2, 'map.json')
            self.assertIn('experiments.no_centralized_critic.ieee69.train', cmd)
            self.assertNotIn('ieee33', ' '.join(cmd))
            with self.assertRaises(ValueError): command(p.VARIANTS[0], 'evaluate', 0, 'map.json')

    def test_launcher_transport(self):
        for v in p.VARIANTS:
            for kind in ('train','eval'):
                file = p.ROOT/'experiments'/v/f'{kind}_{v}_ieee69.slurm'
                raw = file.read_bytes(); text = raw.decode()
                self.assertNotIn(b'\r', raw)
                self.assertTrue(raw.startswith(b'#!/bin/bash\n'))
                self.assertIn('#SBATCH --export=ALL\n', text)
                self.assertNotIn('ieee33', text)
                self.assertIn('"$@"', text)
        shell = Path(__file__).with_name('launch.sh').read_text()
        self.assertIn('experiments.no_shared_kpi.ieee69.slurm_dispatch', shell)
        self.assertLess(shell.index('LAUNCH_CHECK_PASS'), shell.index('conda activate'))


@unittest.skipUnless(HAS_TORCH, 'PyTorch is unavailable; run on RIT')
class ModelTests(unittest.TestCase):
    def make(self, variant):
        import torch
        from .models import model_class
        torch.manual_seed(42)
        return model_class(variant)(agent_obs_dim=10, num_agents=9, action_dim_per_agent=1,
                                   device='cpu', actor_lr=5e-5, critic_lr=5e-4)

    def test_dimensions_forward_update_and_checkpoint(self):
        import torch
        for variant, width in zip(p.VARIANTS, (7,10)):
            m = self.make(variant)
            self.assertEqual(len(m.actor.actors), 9)
            self.assertEqual(len(m.critics), 9)
            self.assertEqual(m.actor.actors[0].net[0].in_features, width)
            first = next(x for x in m.critics[0].modules() if isinstance(x, torch.nn.Linear))
            self.assertEqual(first.in_features, 99 if width == 7 else 11)
            batch = dict(agent_obs=torch.randn(4,9,10), next_agent_obs=torch.randn(4,9,10),
                         actions=torch.rand(4,9,1)*2-1, rewards=torch.ones(4,9), dones=torch.zeros(4,1))
            before = [x.detach().clone() for x in m.actor.parameters()]
            logs = m.update(batch)
            self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in logs.values()))
            self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,m.actor.parameters())))
            with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
                file = Path(tmp)/'smoke.pt'; m.save(file)
                other = self.make(variant); other.load(file)
                torch.testing.assert_close(m.actor(batch['agent_obs']), other.actor(batch['agent_obs']))

    def test_actor_mask_and_independent_locality(self):
        import torch
        a1 = self.make(p.VARIANTS[0]); obs = torch.randn(3,9,10)
        changed = obs.clone(); changed[:,:,7:] += 100
        torch.testing.assert_close(a1.actor(obs), a1.actor(changed))
        a2 = self.make(p.VARIANTS[1]); actions = torch.randn(3,9,1)
        x = obs.clone().requires_grad_(); acts = actions.clone().requires_grad_()
        q = a2.critics[4](x[:,4], acts[:,4]); q.sum().backward()
        self.assertEqual(torch.count_nonzero(x.grad[:,[0,1,2,3,5,6,7,8]]).item(), 0)
        self.assertEqual(torch.count_nonzero(acts.grad[:,[0,1,2,3,5,6,7,8]]).item(), 0)

    def test_private_bindings_leave_ieee33_unchanged(self):
        from experiments.no_shared_kpi import runner as original
        old = dict(vars(original))
        from . import runner
        self.assertEqual(vars(original), old)
        self.assertEqual(runner.namespace['IEEE33_CONFIG']['network']['num_buses'],69)
        from experiments.no_shared_kpi.model_no_shared_kpi import NoSharedKPI
        with self.assertRaises(ValueError): NoSharedKPI(agent_obs_dim=10, num_agents=9)


def real_smoke(variant, output, baseline_path, run_id):
    import numpy as np
    import pandas as pd
    import torch
    from configs.ieee69_config import IEEE69_CONFIG
    from utility.provenance import checkpoint_provenance
    from utility.evaluation_reporting import reset_for_evaluation
    from .runner import RunManifest, make_primary_env, trainer_class, evaluate_one_episode, summarize, energy_rows
    baseline = p.load_baseline(baseline_path, run_id)
    seed = baseline['seed']; torch.set_num_threads(1); torch.manual_seed(seed); np.random.seed(seed)
    env = make_primary_env(deepcopy(IEEE69_CONFIG), 'train', seed)
    val = make_primary_env(deepcopy(IEEE69_CONFIG), 'val', seed+1000)
    manifest = RunManifest(output, 'MLP', seed, 'smoke', arguments=dict(ablation=variant, feeder='IEEE69'))
    for e in (env,val):
        manifest.add_environment(e)
        p.compare_environment(manifest.data['environments'][-1], baseline)
    trainer = trainer_class(variant)(env,val,dict(p.TRAIN_SETTINGS,buffer_size=64,batch_size=4,
                    checkpoint_dir=str(Path(output)/'checkpoints')),device='cpu')
    obs,_ = trainer._reset_env(env); updates=0
    for step in range(16):
        real_step = env.step; captured = {}
        def capture(action):
            captured['action']=np.array(action,copy=True); captured['result']=real_step(action)
            return captured['result']
        with patch.object(env,'step',side_effect=capture), patch.object(trainer.replay_buffer,'add',wraps=trainer.replay_buffer.add) as add:
            obs,reward,done,info=trainer.collect_step(obs)
        assert info['replay_transition_stored'] and not info['episode_failed'] and info['step_pf_failed_solves']==0
        saved=add.call_args.kwargs
        np.testing.assert_array_equal(saved['action'],captured['action'])
        np.testing.assert_array_equal(saved['reward'],captured['result'][1])
        for k in saved['next_obs']: np.testing.assert_array_equal(saved['next_obs'][k],captured['result'][0][k])
        if step>=8:
            logs=trainer.agent.update(trainer.replay_buffer.sample(4)); updates+=1
            assert np.isfinite(list(logs.values())).all()
            assert all(torch.isfinite(x).all() for module in (trainer.agent.actor,trainer.agent.critics) for x in module.parameters())
    control=make_primary_env(deepcopy(IEEE69_CONFIG),'train',seed)
    start=env.episode_start_indices[0]
    reset_for_evaluation(env,start,77); reset_for_evaluation(control,start,77)
    for action in (np.zeros(9,dtype=np.float32),np.full(9,-.7,dtype=np.float32)):
        a=env.step(action.copy()); b=control.step(action.copy())
        np.testing.assert_array_equal(a[1],b[1])
        np.testing.assert_array_equal(a[-1]['final_battery_power_kw'],b[-1]['final_battery_power_kw'])
    ck=Path(output)/'checkpoints/smoke.pt'
    trainer.agent.save(ck,checkpoint_provenance(env)); trainer.agent.load(ck,env)
    rows=evaluate_one_episode(val,trainer.agent,'mlp_maddpg','TP1',0,val.episode_start_indices[0],123)
    df=pd.DataFrame(rows); ep,agg=summarize(df)
    assert len(df)==96 and df.episode_completed.all() and not df.episode_failed.any()
    assert df.step_pf_failed_solves.sum()==0
    energy=energy_rows(df,val); assert energy.adjusted_cost_available.all()
    for name,frame in [('step_metrics',df),('episode_summary',ep),('aggregate_summary',agg),('terminal_soc_summary',energy)]:
        frame.to_csv(Path(output)/f'{name}.csv',index=False)
    manifest.data.update(transitions=16,optimizer_updates=updates,evaluation_steps=96)
    manifest.finish(); print('REAL_SMOKE_PASS',variant,16,updates,96)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=p.VARIANTS,default=p.VARIANTS[0])
    parser.add_argument('--baseline-training-manifest',type=Path)
    parser.add_argument('--run',type=int,choices=[0,1,2],default=0)
    parser.add_argument('--real-smoke',action='store_true')
    parser.add_argument('--require-runtime',action='store_true',help='Fail instead of accepting skipped model tests')
    parser.add_argument('--output',type=Path)
    a=parser.parse_args()
    if a.require_runtime and not HAS_TORCH:
        parser.error('PyTorch unavailable: model/runtime validation cannot pass')
    if a.baseline_training_manifest: p.load_baseline(a.baseline_training_manifest,a.run)
    from .test_tp10_provenance import TP10Tests
    suite=unittest.TestSuite(unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (StaticTests,ModelTests,TP10Tests))
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful(): raise SystemExit(1)
    if a.real_smoke:
        if not a.output or not a.baseline_training_manifest or not a.output.resolve().is_relative_to((p.ROOT/'results/ablations/ieee69').resolve()):
            parser.error('Smoke requires baseline manifest and a fresh output under results/ablations/ieee69')
        real_smoke(a.variant,a.output,a.baseline_training_manifest,a.run)


if __name__=='__main__': main()
