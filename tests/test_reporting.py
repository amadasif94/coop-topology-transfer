# Maintained copy of experiments/b3_reporting/test_reporting.py (which stays and now exercises the shim).
"""Small synthetic tests only; no real data, checkpoints or simulations."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import pandas as pd
from evaluation.reporting.metrics import power_metrics, episode_metrics, summary
from evaluation.reporting.provenance import (environment_contract, verify_environment, formulation,
    paired_identity, verify_finished, sha)
from evaluation.reporting.compare import pair_days, complete_inventory, main as compare_main


def entry():
    return dict(feeder='IEEE33',interface_identity={'dess_buses':[1]},normalization_identity={},
        battery_constraints=[dict(capacity_kwh=500,soc_min=.1)],reward={'w_grid':.65},doe={'enabled':False},
        backup={'pf_budget':20},network={'v_min':.95},voltage_tolerance_pu=1e-6,current_relative_tolerance=1e-6,
        expected_steps_per_episode=96,data_scaling={'load_scaling_factor':1.,'pv_scaling_factor':1.},
        input_files={k:dict(exists=True,sha256=k) for k in ['processed_time_series','line_data','gridtensor_line_data']},
        active_current_ratings_a=[255],topology='TP1')


def day(complete=True, price=40.):
    return pd.DataFrame([dict(topology_case='TP1',episode_id=0,start_index=0,step=i,
        date_time=f'2018-12-01 00:00:00',record_type='dispatch_attempt',episode_failed=not complete,
        episode_completed=complete,policy_requested_power_kw='[10]',physical_limited_power_kw='[5]',
        soc_before='[0.5]',soc_after='[0.4]',grid_import_kw=100,delta_t_hours=.25,
        final_network_feasible=True) for i in range(96 if complete else 40)])


def identity(kind='original_maddpg',run=0):
    return dict(schema='matched-evaluation/1',experiment_class=kind,controller='mlp_a0',feeder='ieee33',run=run,
        seed=42+run,architecture={'actor':[10,256]},model_config={'dt_hours':.25},
        environments={'TP1':environment_contract(entry())},
        episode_plans={'TP1':[dict(start_index=0,date='2018-12-01',seed=42+run)]},
        protocol={'stage':'test','load_scale':1.},selection_sha256='frozen')


class Reporting(unittest.TestCase):
    def test_json_null_and_empty_dispatch(self):
        f=pd.DataFrame({'policy_requested_power_kw':['null'],'physical_limited_power_kw':['[0,0]']})
        self.assertEqual(power_metrics(f,2)['power_missing_decisions'],2)
        m=power_metrics(f.iloc[:0],2)
        self.assertEqual(m['power_valid_decisions'],0)
        self.assertTrue(np.isnan(m['material_correction_fraction']))

    def test_nonfinite_terminal_state_is_undefined(self):
        f=day();f.loc[95,'soc_after']='[Infinity]'
        r=episode_metrics(f,entry()['battery_constraints'],{0:[40]*96}).iloc[0]
        self.assertTrue(np.isnan(r.adjusted_energy_cost))
        self.assertTrue(np.isnan(r.terminal_stored_energy_kwh))
        self.assertEqual(r.raw_energy_cost,96.)

    def test_unknown_failure_status_cannot_complete(self):
        f=day();f['episode_failed']=f['episode_failed'].astype(object);f.loc[3,'episode_failed']=None
        r=episode_metrics(f,entry()['battery_constraints'],{0:[40]*96})
        self.assertEqual(r.completed_days.iloc[0],0)
        self.assertEqual(summary(r)['failure_status_unknown_days'],1)
        self.assertTrue(np.isnan(r.raw_energy_cost.iloc[0]))

    def test_partial_and_contradictory_identities_rejected(self):
        a=identity();b=identity('b3_pre_tanh')
        del a['architecture'];del b['architecture']
        with self.assertRaises(ValueError):paired_identity(a,b)
        for ck in [dict(schema='ieee33-mlp-ablation/999',actor={},critics={}),
                   dict(actor={},critics={},controller='b3_mlp_a0'),
                   dict(schema='b3-checkpoint/1',feasibility_formulation='pre_tanh_squared_hinge')]:
            with self.subTest(checkpoint=ck),self.assertRaises(ValueError):formulation(ck)

    def test_duplicate_expected_days_rejected(self):
        f=episode_metrics(day(),entry()['battery_constraints'],{0:[40]*96})
        i=identity();i['episode_plans']['TP1']*=2
        with self.assertRaises(ValueError):complete_inventory(f,i)

    def test_wrong_date_rejected(self):
        f=episode_metrics(day(),entry()['battery_constraints'],{0:[40]*96})
        f.loc[0,'date']='2018-12-02'
        with self.assertRaises(ValueError):complete_inventory(f,identity())

    def test_reset_failure_with_no_dispatch_is_reported(self):
        f=day(False).iloc[:1].copy();f['record_type']='reset_failure';f['step']=-1
        r=episode_metrics(f,entry()['battery_constraints'],{0:[40]*96}).iloc[0]
        self.assertEqual(r.failed_days,1)
        self.assertEqual(r.observed_steps,0)
        self.assertTrue(np.isnan(r.material_correction_fraction))
        self.assertTrue(np.isnan(r.raw_energy_cost))

    def test_valid_power_exact(self):
        m=power_metrics(day(),1)
        self.assertEqual(m['requested_power_removed_fraction'],.5)
        self.assertEqual(m['material_correction_fraction'],1.)
        self.assertEqual(m['removed_abs_kw_sum'],480.)

    def test_missing_excluded(self):
        f=day().iloc[:2].copy();f.loc[1,'physical_limited_power_kw']='[null]'
        m=power_metrics(f,1)
        self.assertEqual(m['material_correction_fraction'],1.)
        self.assertEqual(m['power_missing_decisions'],1)

    def test_entirely_missing(self):
        m=power_metrics(pd.DataFrame(index=range(2)),5)
        self.assertEqual(m['power_missing_decisions'],10)
        self.assertTrue(np.isnan(m['material_correction_fraction']))
        self.assertTrue(np.isnan(m['requested_power_removed_fraction']))

    def test_invalid_distinct(self):
        f=pd.DataFrame({'policy_requested_power_kw':['[Infinity]','bad','[null]'],
                        'physical_limited_power_kw':['[0]']*3})
        m=power_metrics(f,1)
        self.assertEqual(m['power_invalid_decisions'],2)
        self.assertEqual(m['power_missing_decisions'],1)

    def test_zero_denominator(self):
        f=pd.DataFrame({'policy_requested_power_kw':['[0]'],'physical_limited_power_kw':['[0]']})
        m=power_metrics(f,1)
        self.assertTrue(np.isnan(m['requested_power_removed_fraction']))
        self.assertEqual(m['material_correction_fraction'],0.)
        self.assertTrue(m['power_denominator_zero'])

    def test_battery_not_step_denominator_and_strict_threshold(self):
        f=pd.DataFrame({'policy_requested_power_kw':['[2,1,0]'],'physical_limited_power_kw':['[0,0,0]']})
        self.assertEqual(power_metrics(f,3)['material_correction_fraction'],1/3)

    def test_cost_and_terminal_energy(self):
        f=episode_metrics(day(),entry()['battery_constraints'],{0:[40]*96})
        self.assertAlmostEqual(f.raw_energy_cost[0],96.)
        self.assertAlmostEqual(f.adjusted_energy_cost[0],98.)
        self.assertAlmostEqual(f.terminal_soc_shortfall[0],.25)

    def test_incomplete_cost_never_zero(self):
        f=episode_metrics(day(False),entry()['battery_constraints'],{0:[40]*96})
        self.assertTrue(np.isnan(f.raw_energy_cost[0]));self.assertEqual(f.failed_days[0],1)
        s=summary(f);self.assertEqual(s['raw_energy_cost_missing_days'],1)
        self.assertEqual(s['raw_energy_cost_valid_days'],0)

    def test_duplicate_step_rejected(self):
        f=day();f.loc[1,'step']=0
        with self.assertRaises(ValueError):episode_metrics(f,entry()['battery_constraints'],{0:[40]*96})

    def test_post_tanh_identity(self):
        self.assertEqual(formulation({'schema':'b3-checkpoint/1'}),'b3_post_tanh')
        with self.assertRaises(ValueError):formulation({'schema':'b3-checkpoint/2','feasibility_formulation':'post'})
        with self.assertRaises(ValueError):paired_identity(identity(),identity('b3_post_tanh'))

    def test_provenance_changes_rejected(self):
        base=environment_contract(entry())
        for key,value in [('reward',{}),('battery_constraints',[]),('doe',{}),('backup',{}),('interface_identity',{}),('ratings',[270])]:
            changed=copy.deepcopy(base);changed[key]=value
            with self.subTest(key=key), self.assertRaises(ValueError):verify_environment(base,changed)
        changed=copy.deepcopy(base);changed['input_hashes']['processed_time_series']='different'
        with self.assertRaises(ValueError):verify_environment(base,changed)

    def test_authorized_topology_and_load_only(self):
        base=environment_contract(entry());changed=copy.deepcopy(base)
        changed['topology']='TP6';changed['data_scaling']['load_scaling_factor']=1.3
        changed['input_hashes']['line_data']='TP6'
        verify_environment(base,changed,1.3,'TP6')
        with self.assertRaises(ValueError):verify_environment(base,changed,1.2,'TP6')

    def test_missing_provenance_rejected(self):
        e=entry();del e['reward']
        with self.assertRaises(ValueError):environment_contract(e)

    def test_architecture_mismatch_rejected(self):
        b=identity('b3_pre_tanh');b['architecture']={'actor':[7,256]}
        with self.assertRaises(ValueError):paired_identity(identity(),b)

    def test_reuse_requires_exact_identity_and_artifacts(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            p=Path(d);i=identity('b3_pre_tanh')
            names=['step_metrics.csv','episode_summary.csv','aggregate_summary.csv','scientific_episodes.csv']
            for n in names:(p/n).write_text('synthetic')
            m=dict(status='finished',scientific_identity=i,report_sha256={n:sha(p/n) for n in names})
            (p/'run_manifest.json').write_text(json.dumps(m));verify_finished(p,i)
            with self.assertRaises(ValueError):verify_finished(p,identity('b3_post_tanh'))
            (p/names[0]).write_text('changed')
            with self.assertRaises(ValueError):verify_finished(p,i)

    def test_failed_day_pairing_and_full_coverage(self):
        o=episode_metrics(day(),entry()['battery_constraints'],{0:[40]*96})
        b=episode_metrics(day(False),entry()['battery_constraints'],{0:[40]*96})
        _,s=pair_days(o,b)
        self.assertTrue(s.paired_valid_days.eq(0).all())
        self.assertTrue(s.intended_paired_days.eq(1).all())
        i=identity();i['episode_plans']['TP1'].append(dict(start_index=96,date='2018-12-02',seed=43))
        expanded=complete_inventory(o,i)
        self.assertEqual(expanded.incomplete_days.sum(),1)
        self.assertEqual(expanded.missing_episode_record.sum(),1)

    def test_a3_is_corrected_b3_only(self):
        from evaluation.reporting.evaluate import CONTROLLERS, main as evaluate_main
        self.assertIn('mlp_a3',CONTROLLERS)
        for kind in ('original_maddpg','b3_post_tanh'):
            with self.subTest(kind=kind),self.assertRaises(SystemExit):
                evaluate_main(['test','--experiment-class',kind,'--controller','mlp_a3','--feeder','ieee33','--run','0',
                               '--training-manifest','missing.json','--output-root','unused'])

    def test_three_seed_compare_actual_csv_writes(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as d:
            root=Path(d);paths={'original':[],'b3':[]}
            for side,kind in [('original','original_maddpg'),('b3','b3_pre_tanh')]:
                for run in range(3):
                    p=root/f'{side}_{run}';p.mkdir();paths[side].append(str(p))
                    f=episode_metrics(day(),entry()['battery_constraints'],{0:[40]*96})
                    f.to_csv(p/'scientific_episodes.csv',index=False)
                    for n in ['step_metrics.csv','episode_summary.csv','aggregate_summary.csv']:(p/n).write_text('synthetic')
                    m=dict(status='finished',scientific_identity=identity(kind,run),report_sha256={n:sha(p/n) for n in ['step_metrics.csv','episode_summary.csv','aggregate_summary.csv','scientific_episodes.csv']})
                    (p/'run_manifest.json').write_text(json.dumps(m))
            compare_main(['--original',*paths['original'],'--b3',*paths['b3'],'--output',str(root/'out')])
            result=pd.read_csv(root/'out/across_seed.csv')
            self.assertTrue(result.all_three_seeds_present.all())
            self.assertTrue(result.mean_difference.eq(0).all())


if __name__=='__main__':unittest.main()
