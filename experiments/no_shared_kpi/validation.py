"""Deterministic ablation tests; --real-smoke runs actual TP1 AC/optimizer/evaluator checks."""
import argparse
import ast
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from experiments.no_shared_kpi.protocol import ROOT, FULL_FEATURES, feature_spec, read_json, sha, TRAIN_SETTINGS, load_baseline
from experiments.no_shared_kpi.compare_ablations_ieee33 import stats, pooled_stats, analyze, compatibility

HAS_TORCH = importlib.util.find_spec('torch') is not None


class ProtocolTests(unittest.TestCase):
    def test_original_files_unchanged(self):
        records = read_json(Path(__file__).with_name('original_source_hashes.json'))
        normalized = read_json(Path(__file__).with_name('original_source_lf_hashes.json'))
        import hashlib
        self.assertGreater(len(records),100)
        for relative,digest in records.items():
            if sha(ROOT/relative) != digest:
                # WinSCP may normalize line endings; never tolerate source-content drift.
                actual=hashlib.sha256((ROOT/relative).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                self.assertEqual(actual,normalized[relative],relative)

    def test_observation_semantics_and_dimensions(self):
        from utility.obs_interface import AGENT_OBS_FEATURES
        self.assertEqual(tuple(AGENT_OBS_FEATURES),FULL_FEATURES)
        self.assertEqual(feature_spec('no_shared_kpi')['actor_input_dim'],7)
        self.assertEqual(feature_spec('no_shared_kpi')['critic_input_dim'],55)
        self.assertEqual(feature_spec('no_centralized_critic')['actor_input_dim'],10)
        self.assertEqual(feature_spec('no_centralized_critic')['critic_input_dim'],11)

    def test_all_new_python_parses(self):
        for folder in ('no_shared_kpi','no_centralized_critic'):
            for path in (ROOT/'experiments'/folder).glob('*.py'): ast.parse(path.read_text(),filename=str(path))

    def test_baseline_requires_saved_200k_configuration(self):
        valid=dict(status='finished',controller='MLP',mode='train',seed=42,arguments=TRAIN_SETTINGS,
            environments=[dict(feeder='IEEE33',topology='TP1',agent_obs_features=list(FULL_FEATURES),
                doe=dict(enabled=False),backup_mode='directional_incremental')])
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix='test_tmp_') as d:
            path=Path(d)/'run_manifest.json'; path.write_text(json.dumps(valid))
            self.assertEqual(load_baseline(path,0)['arguments']['total_steps'],200000)
            valid['arguments']=dict(TRAIN_SETTINGS,total_steps=100000)
            path.write_text(json.dumps(valid))
            with self.assertRaises(ValueError): load_baseline(path,0)

    def test_paired_bootstrap_exact_constant(self):
        r=stats(np.full(31,2.5),np.random.default_rng(3))
        self.assertEqual(r['mean'],2.5); self.assertEqual(r['ci_low'],2.5); self.assertEqual(r['ci_high'],2.5)

    def test_shared_day_clusters_not_93_independent_samples(self):
        frame=pd.DataFrame([dict(run=r,start_index=d,difference=d+r) for r in range(3) for d in range(31)])
        out=pooled_stats(frame,np.random.default_rng(3))
        self.assertEqual(out['matched_days'],31); self.assertEqual(out['n'],31)
        self.assertEqual(out['between_run_sample_std'],1.0)
        missing=frame.drop(frame[(frame.run==1)&(frame.start_index==0)].index)
        self.assertEqual(pooled_stats(missing,np.random.default_rng(3))['matched_days'],30)

    def test_analysis_sign_pairing_and_no_overwrite(self):
        rows=[]
        for label in ('A0','A1','A2','zero'):
            for run in (range(3) if label!='zero' else [-1]):
                for tp in range(1,8):
                    for day in range(3):
                        cost=10+day+tp*(2 if label=='A1' else 1)
                        rows.append(dict(configuration=label,run=run,topology_case=f'TP{tp}',start_index=day,
                            raw_energy_cost=cost,adjusted_energy_cost=cost,complete=True))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix='test_tmp_') as tmp:
            output=Path(tmp)/'out'
            analyze(pd.DataFrame(rows),output,resamples=100,seed=1)
            gaps=pd.read_csv(output/'ablation_paired_daily_differences.csv')
            self.assertTrue((gaps[(gaps.configuration=='A1')&(gaps.topology_case=='TP2')].difference==1).all())
            self.assertTrue((gaps[gaps.configuration=='A2'].difference==0).all())
            with self.assertRaises(FileExistsError): analyze(pd.DataFrame(rows),output,resamples=100)

    def test_missing_provenance_rejected(self):
        with self.assertRaises(ValueError): compatibility({}, {})

    def test_stored_energy_sign_units_and_missing(self):
        from experiments.no_shared_kpi.protocol import inventory_adjusted_cost
        self.assertEqual(inventory_adjusted_cost(100,50,150,40),96)
        self.assertEqual(inventory_adjusted_cost(100,150,50,40),104)
        self.assertEqual(inventory_adjusted_cost(100,50,50,40),100)
        self.assertTrue(np.isnan(inventory_adjusted_cost(100,np.nan,50,40)))

    def test_manifest_drift_and_same_topology_required(self):
        from experiments.no_shared_kpi.protocol import compare_environment
        entry=dict(topology='TP1',mode='train',agent_obs_features=list(FULL_FEATURES),
            interface_identity={},normalization_identity={},data_scaling={'load_scaling_factor':1.0},
            doe={'enabled':False},backup={},network={},reward={'penalty':-50},battery_constraints=[],
            active_current_ratings_a=[255],voltage_tolerance_pu=1e-6,current_relative_tolerance=1e-6,
            expected_steps_per_episode=96,available_episode_start_indices=[0],input_files={})
        compare_environment(entry,{'environments':[deepcopy(entry)]})
        for key,value in [('doe',{'enabled':True}),('active_current_ratings_a',[250]),
                          ('reward',{'penalty':-20}),('topology','TP2')]:
            changed=deepcopy(entry); changed[key]=value
            with self.assertRaises(ValueError): compare_environment(changed,{'environments':[entry]})

    def test_selection_rejects_december_and_unseen_topologies(self):
        from utility.evaluation_protocol import authorize, ProtocolError
        for mode,topologies in [('test',['TP1']),('val',['TP6'])]:
            with self.assertRaises(ProtocolError):
                authorize(mode=mode,run_purpose='checkpoint_selection',topologies=topologies)
        allowed=authorize(mode='val',run_purpose='checkpoint_selection',topologies=['TP1'])
        self.assertEqual(allowed['data_split'],'val')
        self.assertFalse(allowed['final_test'])
        with self.assertRaises(ProtocolError):authorize(mode='test',final=True,topologies=['TP1','TP7'])


@unittest.skipUnless(HAS_TORCH,'PyTorch is not installed in this local runtime; execute on RIT')
class ModelTests(unittest.TestCase):
    def setUp(self):
        import torch
        torch.set_num_threads(1); torch.manual_seed(123)
        self.kw=dict(agent_obs_dim=10,num_agents=5,device='cpu',share_actor=False)

    def batch(self):
        import torch
        return dict(agent_obs=torch.randn(4,5,10),next_agent_obs=torch.randn(4,5,10),
            actions=torch.rand(4,5,1)*2-1,rewards=torch.randn(4,5),dones=torch.zeros(4,1))

    def test_a1_actor_kpi_invariance_and_critic_full_joint_inputs(self):
        import torch
        from experiments.no_shared_kpi.model_no_shared_kpi import NoSharedKPI
        from experiments.no_gnn_maddpg.maddpg_mlp import MLPMADDPG
        torch.manual_seed(123); base=MLPMADDPG(**self.kw)
        torch.manual_seed(123); model=NoSharedKPI(**self.kw)
        for key,val in base.critics.state_dict().items(): torch.testing.assert_close(val,model.critics.state_dict()[key],rtol=0,atol=0)
        b=self.batch(); changed=b['agent_obs'].clone(); changed[...,7:]+=1000
        torch.testing.assert_close(model.actor(b['agent_obs']),model.actor(changed),rtol=0,atol=0)
        differentiable=b['agent_obs'].clone().requires_grad_()
        model.actor(differentiable).sum().backward()
        self.assertTrue(torch.equal(differentiable.grad[...,7:],torch.zeros_like(differentiable.grad[...,7:])))
        torch.testing.assert_close(model.target_actor(b['next_agent_obs']),
            model.target_actor(torch.cat((b['next_agent_obs'][...,:7],b['next_agent_obs'][...,7:]+999),dim=-1)),rtol=0,atol=0)
        self.assertEqual(model.critics[0].q_network[0].in_features,55)
        self.assertIs(NoSharedKPI.update,MLPMADDPG.update)
        before=[p.detach().clone() for p in model.actor.parameters()]
        seen=[]
        hook=model.critics[0].register_forward_pre_hook(lambda m,a,k:seen.append((k['agent_obs'].shape,k['actions'].shape)),with_kwargs=True)
        log=model.update(b); hook.remove()
        self.assertTrue(all(s[0]==(4,5,10) and s[1]==(4,5,1) for s in seen))
        self.assertTrue(any(not torch.equal(x,p) for x,p in zip(before,model.actor.parameters())))
        self.assertTrue(np.isfinite(list(log.values())).all())

    def test_a2_original_actors_and_strict_local_critics(self):
        import torch
        from experiments.no_centralized_critic.model_independent_critic import IndependentAgents
        from experiments.no_gnn_maddpg.maddpg_mlp import MLPMADDPG
        torch.manual_seed(123); base=MLPMADDPG(**self.kw)
        torch.manual_seed(123); model=IndependentAgents(**self.kw)
        for key,val in base.actor.state_dict().items(): torch.testing.assert_close(val,model.actor.state_dict()[key],rtol=0,atol=0)
        b=self.batch(); o=b['agent_obs'].requires_grad_(); a=b['actions'].requires_grad_()
        model.critics[2](o[:,2],a[:,2]).sum().backward()
        self.assertTrue(torch.equal(o.grad[:,[0,1,3,4]],torch.zeros_like(o.grad[:,[0,1,3,4]])))
        self.assertTrue(torch.equal(a.grad[:,[0,1,3,4]],torch.zeros_like(a.grad[:,[0,1,3,4]])))
        with self.assertRaises(ValueError): model.critics[0](o,a)
        seen=[]; hooks=[]
        for c in list(model.critics)+list(model.target_critics):
            hooks.append(c.register_forward_pre_hook(lambda m,a:seen.append((a[0].shape,a[1].shape))))
        before=[p.detach().clone() for p in model.actor.parameters()]
        log=model.update(self.batch())
        for h in hooks:h.remove()
        self.assertTrue(all(s==((4,10),(4,1)) for s in seen))
        self.assertEqual(len(model.actor_optimizers),5)
        self.assertTrue(any(not torch.equal(x,p) for x,p in zip(before,model.actor.parameters())))
        self.assertTrue(np.isfinite(list(log.values())).all())

    def test_checkpoint_spec_roundtrip_and_wrong_ablation_refused(self):
        import torch
        from experiments.no_shared_kpi.runner import model_class
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent, prefix='test_tmp_') as d:
            for v in ('no_shared_kpi','no_centralized_critic'):
                m=model_class(v)(**self.kw); m.update(self.batch()); path=Path(d)/f'{v}.pt'
                m.save(path); n=model_class(v)(**self.kw); n.load(path)
                obs=torch.randn(5,10)
                torch.testing.assert_close(m.actor(obs),n.actor(obs),rtol=0,atol=0)
                with self.assertRaises(FileExistsError): m.save(path)
                other=model_class('no_centralized_critic' if v=='no_shared_kpi' else 'no_shared_kpi')(**self.kw)
                with self.assertRaises(ValueError):other.load(path)

    def test_trainer_reuses_original_replay_and_learning_loop(self):
        from experiments.no_shared_kpi.runner import trainer_class
        from experiments.no_gnn_maddpg.trainer_mlp import MLPMADDPGTrainer
        for v in ('no_shared_kpi','no_centralized_critic'):
            cls=trainer_class(v)
            self.assertIs(cls.collect_step,MLPMADDPGTrainer.collect_step)
            self.assertIs(cls.train,MLPMADDPGTrainer.train)
            self.assertIs(cls.__init__.__code__,MLPMADDPGTrainer.__init__.__code__)

    def test_independent_bellman_targets_use_own_reward_and_terminal_mask(self):
        import torch
        from unittest.mock import patch
        from experiments.no_centralized_critic.model_independent_critic import IndependentAgents
        from torch.nn import functional as F
        m=IndependentAgents(**self.kw); b=self.batch()
        b['dones'][0]=1
        with torch.no_grad():
            for i,c in enumerate(m.target_critics):
                for p in c.parameters():p.zero_()
                c.q_network[-1].bias.fill_(i+1)
        captured=[]; mse=F.mse_loss
        def capture(pred,target):
            captured.append(target.detach().clone()); return mse(pred,target)
        with patch('experiments.no_centralized_critic.model_independent_critic.F.mse_loss',side_effect=capture):m.update(b)
        self.assertEqual(len(captured),5)
        for i,y in enumerate(captured):
            expected=b['rewards'][:,i:i+1]+m.gamma*(1-b['dones'])*(i+1)
            torch.testing.assert_close(y,expected)


def real_smoke(variant, output):
    """16 actual transitions + 8 updates; separate full 96-step validation day; no December."""
    import torch
    from unittest.mock import patch
    from configs.ieee33_config import IEEE33_CONFIG
    from environments.primary_marl import make_primary_env
    from utility.run_manifest import RunManifest
    from utility.provenance import checkpoint_provenance
    from experiments.no_shared_kpi.runner import trainer_class, evaluate_one_episode, summarize, energy_rows
    torch.set_num_threads(1); torch.manual_seed(42); np.random.seed(42)
    env=make_primary_env(deepcopy(IEEE33_CONFIG),'train',42)
    val=make_primary_env(deepcopy(IEEE33_CONFIG),'val',1042)
    run=RunManifest(output,'MLP',42,'smoke',arguments=dict(ablation=variant))
    for e in (env,val): run.add_environment(e)
    trainer=trainer_class(variant)(env,val,dict(buffer_size=64,batch_size=4,
        checkpoint_dir=str(Path(output)/'checkpoints')),device='cpu')
    obs,_=trainer._reset_env(env); updates=0
    for step in range(16):
        captured={}; real_step=env.step
        def capture(action):
            captured['proposal']=np.array(action,copy=True)
            captured['result']=real_step(action)
            return captured['result']
        with patch.object(env,'step',side_effect=capture), patch.object(trainer.replay_buffer,'add',wraps=trainer.replay_buffer.add) as add:
            obs,reward,done,info=trainer.collect_step(obs)
        assert info['replay_transition_stored'] and not info['episode_failed']
        assert info['step_pf_failed_solves']==0
        stored=add.call_args.kwargs
        np.testing.assert_array_equal(stored['action'],captured['proposal'])
        np.testing.assert_array_equal(stored['reward'],captured['result'][1])
        for k in stored['next_obs']:np.testing.assert_array_equal(stored['next_obs'][k],captured['result'][0][k])
        assert env.doe_enabled is False and env.backup_mode=='directional_incremental'
        if step>=8:
            logs=trainer.agent.update(trainer.replay_buffer.sample(4)); updates+=1
            assert np.isfinite(list(logs.values())).all()
            for module in (trainer.agent.actor,trainer.agent.critics):
                assert all(torch.isfinite(p).all() for p in module.parameters())
    # The SAME proposals reproduce the SAME environment/reward path independent of model class.
    control=make_primary_env(deepcopy(IEEE33_CONFIG),'train',42)
    from utility.evaluation_reporting import reset_for_evaluation
    start=env.episode_start_indices[0]
    reset_for_evaluation(env,start,77); reset_for_evaluation(control,start,77)
    for action in (np.zeros(5,dtype=np.float32),np.full(5,-.7,dtype=np.float32)):
        a=env.step(action.copy()); b=control.step(action.copy())
        np.testing.assert_array_equal(a[1],b[1])
        np.testing.assert_array_equal(a[-1]['final_battery_power_kw'],b[-1]['final_battery_power_kw'])
    ck=Path(output)/'checkpoints/smoke.pt'
    trainer.agent.save(ck,checkpoint_provenance(env)); trainer.agent.load(ck,env)
    rows=evaluate_one_episode(val,trainer.agent,'mlp_maddpg','TP1',0,val.episode_start_indices[0],123)
    df=pd.DataFrame(rows); ep,agg=summarize(df)
    assert len(df)==96 and df.episode_completed.all() and not df.episode_failed.any()
    assert df.step_pf_failed_solves.sum()==0
    df.to_csv(Path(output)/'step_metrics.csv',index=False)
    ep.to_csv(Path(output)/'episode_summary.csv',index=False)
    agg.to_csv(Path(output)/'aggregate_summary.csv',index=False)
    energies=energy_rows(df,val); assert energies.adjusted_cost_available.all()
    energies.to_csv(Path(output)/'terminal_soc_summary.csv',index=False)
    run.data.update(transitions=16,optimizer_updates=updates,ablation=variant,evaluation_steps=96)
    run.finish(); print(dict(ablation=variant,transitions=16,optimizer_updates=updates,evaluation_steps=96))


def main(variant):
    p=argparse.ArgumentParser()
    p.add_argument('--real-smoke',action='store_true')
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(c) for c in (ProtocolTests,ModelTests)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if not result.wasSuccessful():raise SystemExit(1)
    if a.real_smoke:
        if not a.output or not a.output.resolve().is_relative_to((ROOT/'results/ablations/ieee33').resolve()):
            p.error('--real-smoke requires a fresh --output under results/ablations/ieee33')
        real_smoke(variant,a.output)
