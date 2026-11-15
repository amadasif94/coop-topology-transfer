"""Exact append-only TP10 evidence; never a general line-hash exemption."""
from copy import deepcopy
import hashlib
from io import BytesIO
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd

APPROVED = {
    'line_data': ('be1c8e05bde86391209f16807339161cd53060b674b9b2e8b6c6d66bf6802e32',
                  '94a64294a9c2efa15be52c1fb416be47ef228ad1f100eec978aa038eb059966e',
                  b'26,64,1.041,0.5302,0,tie,255.0\n'),
    'gridtensor_line_data': ('69a654e3268fcc2d32a55ad5564d64b814e76e0baa5499dce0c5d746a0cbd827',
                             '86fee34bd7ec3a680b7860faa1ce0522ff3feeda2ace49e8db90a9b48ca928fd',
                             b'27,65,1.041,0.5302,0.0,0,1.0\n')}


def digest(raw): return hashlib.sha256(raw).hexdigest()


def recover(raw, key, expected):
    old, new, row = APPROVED[key]
    if expected != old or digest(raw) != new or not raw.endswith(row):
        raise ValueError(f'Unapproved TP10 extension: {key}')
    prefix = raw[:-len(row)]
    if digest(prefix) != old:
        raise ValueError(f'Historical fingerprint not recovered: {key}')
    return prefix


def electrical_evidence(config, directory):
    """Actual grid and actual AC solver, with controlled static (not trajectory) snapshots."""
    from utility.grid import DistributionGrid
    from utility.power_flow import PowerFlowSolver
    from utility.graph_builder import build_graph_builder
    cfg = deepcopy(config); report = {'schema':'ieee69-tp10-equivalence/1', 'files':{}}
    frames = {}
    for key,(old,new,row) in APPROVED.items():
        raw = Path(config['paths'][key]).read_bytes()
        prefix = recover(raw,key,old)
        file = Path(directory)/(key+'.csv'); file.write_bytes(prefix)
        cfg['paths'][key] = file
        frames[key] = (pd.read_csv(BytesIO(prefix)),pd.read_csv(BytesIO(raw)))
        report['files'][key] = dict(historical_sha256=old,current_sha256=new,approved_row=row.decode().strip())
    historic,current = frames['line_data']
    pd.testing.assert_frame_equal(historic[historic.status==1].reset_index(drop=True),
                                  current[current.status==1].reset_index(drop=True),check_exact=True)
    gt0,gt1 = frames['gridtensor_line_data']
    pd.testing.assert_frame_equal(gt0[gt0.STATUS==1].reset_index(drop=True),gt1[gt1.STATUS==1].reset_index(drop=True),check_exact=True)
    active = current[current.status==1].reset_index(drop=True)
    solver_active = gt1[gt1.STATUS==1].reset_index(drop=True)
    np.testing.assert_array_equal(active[['from_bus','to_bus']].to_numpy(),solver_active[['FROM','TO']].to_numpy()-1)
    np.testing.assert_array_equal(active[['r_ohm','x_ohm']].to_numpy(),solver_active[['R','X']].to_numpy())
    if len(active)!=68 or not (active.max_current_a==255).all(): raise ValueError('TP1 active ratings/branch count differ')
    bus = pd.read_csv(config['paths']['bus_data']); buses = list(config['dess']['dess_buses'])
    grids=[DistributionGrid(bus,lines,buses,c) for lines,c in ((historic,cfg),(current,config))]
    for attr in ('in_service_line_df','adjacency_matrix','edge_index','edge_attr','dess_indicator','line_current_limits_pu'):
        np.testing.assert_array_equal(getattr(grids[0],attr),getattr(grids[1],attr))
    solvers=[PowerFlowSolver(g,c) for g,c in zip(grids,(cfg,config))]
    builders=[build_graph_builder(g,c) for g,c in zip(grids,(cfg,config))]
    report['snapshots']=[]
    for scale,pv_scale,kw in ((.4,0.,0.),(1.,0.,0.),(.5,1.2,-125.),(.8,.2,125.)):
        load=bus.load_kw.to_numpy()*scale; reactive=bus.load_kvar.to_numpy()*scale
        pv=bus.load_kw.to_numpy()*pv_scale; powers=np.zeros(69);powers[buses]=kw
        results=[s.run_power_flow(load_kw=load,load_kvar=reactive,pv_kw=pv,dess_power_kw=powers) for s in solvers]
        if not all(r['converged'] for r in results): raise ValueError('Nonconverged equivalence snapshot')
        for key in ('voltage_pu','line_current_pu','max_voltage_violation','max_line_current_violation'):
            if not np.isfinite(results[0][key]).all() or not np.isfinite(results[1][key]).all(): raise ValueError('Nonfinite PF')
            np.testing.assert_array_equal(results[0][key],results[1][key])
        graphs=[builder.build_graph(time_index=12,price=40.,load_kw=load,pv_kw=pv,
                 soc_by_bus=builder.build_soc_vector(np.full(9,.5)),voltage_pu=r['voltage_pu'])
                 for builder,r in zip(builders,results)]
        for key in ('x','edge_index','edge_attr'): np.testing.assert_array_equal(graphs[0][key],graphs[1][key])
        report['snapshots'].append(dict(load_scale=scale,pv_to_static_load=pv_scale,battery_kw=kw,
                                      min_voltage=float(min(results[0]['voltage_pu'])),max_voltage=float(max(results[0]['voltage_pu'])),
                                      voltage_difference=0.,current_difference=0.,converged=True))
    report.update(active_branches=68,graph_equal=True,graph_feature_values_equal=True,solver_order_equal=True,physical_pf_equal=True)
    return cfg,report


def runtime_evidence(env):
    """Isolated paired environments: fresh observations, rewards, actions, SOC and backup.

    Called before the extension may pass provenance. No training env state is
    advanced. Failure is fatal; no stale report can authorize a changed file.
    """
    if env.mode not in ('train','val') or env.config.get('topology_case',{}).get('case_name','TP1')!='TP1':
        return None
    raw=Path(env.config['paths']['line_data']).read_bytes()
    if digest(raw)==APPROVED['line_data'][0]: return None
    from environments.primary_marl import make_primary_env
    from utility.evaluation_reporting import reset_for_evaluation
    rng=np.random.get_state()
    try:
        with tempfile.TemporaryDirectory(prefix='ieee69_tp10_equivalence_') as tmp:
            historical,report=electrical_evidence(env.config,tmp)
            left=make_primary_env(historical,env.mode,42)
            right=make_primary_env(deepcopy(env.config),env.mode,42)
            starts=list(env.episode_start_indices)
            selected=[starts[0],starts[len(starts)//2],starts[-1]]
            for start in selected:
                a,ai=reset_for_evaluation(left,start,42);b,bi=reset_for_evaluation(right,start,42)
                if not ai['converged'] or not bi['converged']: raise ValueError('Invalid reset in equivalence proof')
                for key in a:
                    if not np.isfinite(a[key]).all() or not np.isfinite(b[key]).all(): raise ValueError('Nonfinite observation in proof')
                    np.testing.assert_array_equal(a[key],b[key])
                for action in (np.zeros(9,dtype=np.float32),np.full(9,-.8,dtype=np.float32),np.full(9,.8,dtype=np.float32)):
                    x=left.step(action.copy());y=right.step(action.copy())
                    if x[-1].get('episode_failed') or y[-1].get('episode_failed'): raise ValueError('PF failure in equivalence proof')
                    if x[-1].get('step_pf_failed_solves',0) or y[-1].get('step_pf_failed_solves',0): raise ValueError('Failed candidate PF in proof')
                    for key in x[0]:
                        if not np.isfinite(x[0][key]).all() or not np.isfinite(y[0][key]).all(): raise ValueError('Nonfinite next observation in proof')
                        np.testing.assert_array_equal(x[0][key],y[0][key])
                    np.testing.assert_array_equal(x[1],y[1])
                    if x[2:4]!=y[2:4]: raise ValueError('Episode flags differ')
                    for key in ('final_battery_power_kw','accepted_action','feasible','converged','correction_iterations',
                                'major_infeasible_action','max_voltage_violation','max_line_current_violation'):
                        np.testing.assert_array_equal(x[-1][key],y[-1][key])
                    np.testing.assert_array_equal([q.get_soc() for q in left.batteries],[q.get_soc() for q in right.batteries])
            report.update(runtime_verified=True,mode=env.mode,starts=[int(s) for s in selected],
                          paired_dispatches=9,observations_rewards_soc_backup_equal=True)
            return report
    finally:
        np.random.set_state(rng)
