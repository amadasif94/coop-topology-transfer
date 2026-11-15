"""Real grid/PF checks and negative provenance tests; no fabricated research results."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from configs.ieee69_config import IEEE69_CONFIG
from . import protocol as p
from .tp10_provenance import APPROVED,recover,electrical_evidence


class TP10Tests(unittest.TestCase):
    def test_exact_hash_recovery_and_reject_unapproved(self):
        for key,(old,new,row) in APPROVED.items():
            raw=Path(IEEE69_CONFIG['paths'][key]).read_bytes()
            self.assertEqual(len(recover(raw,key,old)),len(raw)-len(row))
            for bad in (raw+row,raw.replace(b'1.041',b'1.042'),raw[:-1],raw.replace(b'255.0',b'250.0')):
                if bad == raw: continue  # Solver CSV has no rating column.
                with self.assertRaises(ValueError): recover(bad,key,old)
            with self.assertRaises(ValueError): recover(raw.replace(b'\n',b'\r\n'),key,old)
            with self.assertRaises(ValueError): recover(raw[:-len(row)],key,old)
            with self.assertRaises(ValueError): recover(raw,key,'0'*64)

    def test_real_pf_graph_and_order_equivalence(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            _,report=electrical_evidence(IEEE69_CONFIG,tmp)
            self.assertTrue(report['physical_pf_equal'])
            self.assertTrue(report['graph_feature_values_equal'])
            self.assertEqual(len(report['snapshots']),4)

    def test_without_runtime_proof_comparison_still_fails(self):
        baseline=p.read_json(p.ROOT/'results/checkpoints/maddpg_ieee69/mlp/mlp_run_0/run_manifest.json')
        current=deepcopy(baseline['environments'][0])
        for key,(old,new,row) in APPROVED.items():
            current['input_files'][key].update(path=str(IEEE69_CONFIG['paths'][key]),sha256=new)
        with self.assertRaisesRegex(ValueError,'runtime equivalence'):p.compare_environment(current,baseline)

    def test_approved_comparison_keeps_hashes_and_other_guards(self):
        # Unit fixture for the comparison interface, NOT a runtime proof claim.
        baseline=p.read_json(p.ROOT/'results/checkpoints/maddpg_ieee69/mlp/mlp_run_0/run_manifest.json')
        current=deepcopy(baseline['environments'][0])
        current['tp10_extension_equivalence']=dict(runtime_verified=True,mode='train',files={})
        for key,(old,new,row) in APPROVED.items():
            current['input_files'][key].update(path=str(IEEE69_CONFIG['paths'][key]),sha256=new)
            current['tp10_extension_equivalence']['files'][key]={'current_sha256':new}
        p.compare_environment(current,baseline)
        self.assertEqual(current['input_files']['line_data']['sha256'],APPROVED['line_data'][1])
        self.assertEqual(baseline['environments'][0]['input_files']['line_data']['sha256'],APPROVED['line_data'][0])
        bad=deepcopy(current);bad['input_files']['bus_data']['sha256']='invalid'
        with self.assertRaises(ValueError): p.compare_environment(bad,baseline)
        bad=deepcopy(current);bad['expected_steps_per_episode']=95
        with self.assertRaises(ValueError): p.compare_environment(bad,baseline)
        bad=deepcopy(current);bad['reward']['unapproved_change']=1
        with self.assertRaises(ValueError): p.compare_environment(bad,baseline)

    def test_tp10_network_and_stress_cases_supported(self):
        from experiments.topology_generalization.tp_configs_ieee69 import get_topology_case,list_topology_cases
        from experiments.topology_generalization.topology_utils_ieee69 import build_topology_config
        import pandas as pd
        self.assertTrue({'TP8','TP9','TP10'}<=set(list_topology_cases()))
        self.assertEqual(get_topology_case('TP10')['open_edges'],[(62,63)])
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            cfg=build_topology_config('TP10',output_root=tmp,validate=True)
            line=pd.read_csv(cfg['paths']['line_data']);gt=pd.read_csv(cfg['paths']['gridtensor_line_data'])
            self.assertEqual(int(line.status.sum()),68)
            tie=line[(line.from_bus==26)&(line.to_bus==64)].iloc[0]
            self.assertEqual((tie.r_ohm,tie.x_ohm,tie.status,tie.max_current_a),(1.041,.5302,1.,255.))
            self.assertEqual(int(gt.STATUS.sum()),68)

    def test_final_evaluation_cannot_omit_tp10_reference(self):
        from .final_evaluation import references
        with self.assertRaises(ValueError): references([],0)

    def test_final_reference_all_cases_and_reject_changed_snapshot(self):
        # Synthetic metadata is confined to this unit-test temporary directory.
        # This verifies plumbing, not the existence of original A0 final records.
        from unittest.mock import patch
        import pandas as pd
        from .final_evaluation import references
        from experiments.topology_generalization.topology_utils_ieee69 import build_topology_config
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp); entries=[]; rows=[]
            for n in range(1,11):
                tp=f'TP{n}'
                cfg=build_topology_config(tp,output_root=root/'network_inputs',validate=True)
                files={k:dict(path=str(cfg['paths'][k]),sha256=p.sha(cfg['paths'][k])) for k in APPROVED}
                entries.append(dict(topology=tp,input_files=files))
                for ep in range(31):
                    for step in range(96):
                        rows.append(dict(topology_case=tp,policy='mlp_maddpg',episode_id=ep,start_index=ep*96,
                                         step=step,date_time=pd.Timestamp('2018-12-01',tz='UTC')+pd.Timedelta(days=ep,minutes=15*step)))
            table=pd.DataFrame(rows);table.to_csv(root/'step_metrics.csv',index=False)
            manifest=dict(seed=123,environments=entries,checkpoint_file={'sha256':'unit-test-checkpoint'},evaluation_protocol={})
            with patch.object(p,'load_baseline',return_value=manifest):
                refs=references([root/'run_manifest.json'],0)
                self.assertEqual(list(refs),[f'TP{i}' for i in range(1,11)])
                self.assertEqual(refs['TP10']['plan'][-1],(30,2880,153))
                altered=table.copy()
                altered.loc[altered.episode_id==30,'date_time']=altered.loc[altered.episode_id==30,'date_time']-pd.Timedelta(days=1)
                altered.to_csv(root/'step_metrics.csv',index=False)
                with self.assertRaisesRegex(ValueError,'Duplicate'): references([root/'run_manifest.json'],0)
                table.to_csv(root/'step_metrics.csv',index=False)
                target=Path(entries[-1]['input_files']['line_data']['path'])
                target.write_bytes(target.read_bytes()+b'\n')
                with self.assertRaises(FileNotFoundError): references([root/'run_manifest.json'],0)


if __name__=='__main__':unittest.main(verbosity=2)
