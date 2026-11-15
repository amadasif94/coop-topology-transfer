"""TP1 development smoke pilot, not a trained-policy performance evaluation.

Requires production CSV ratings of 255 A and 40% available PV/load profiles.
Uses deterministic alternating actions to exercise the environment only.
No training, checkpoint access, calibration, or source/config mutation.
"""
import argparse
import copy
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.ieee33_config import IEEE33_CONFIG
from configs.ieee69_config import IEEE69_CONFIG
from environments.dess_env import DESSEnv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--steps', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.steps <= 96:
        parser.error('--steps must be between 1 and 96')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    # Validate both inputs before constructing either environment.
    for label, c in [('IEEE33', IEEE33_CONFIG), ('IEEE69', IEEE69_CONFIG)]:
        lines = pd.read_csv(c['paths']['line_data'])
        active = lines.loc[lines['status'] > 0]
        if 'max_current_a' not in active or not np.allclose(active['max_current_a'], 255):
            raise ValueError(f'{label}: active line CSV must supply max_current_a=255 A')
        ts = pd.read_csv(c['paths']['processed_time_series'])
        load = ts.filter(regex=r'^active_power_node_').to_numpy(dtype=float)
        pv = ts.filter(regex=r'^renewable_active_power_node_').to_numpy(dtype=float)
        if not np.all(np.isfinite(load)) or not np.all(np.isfinite(pv)):
            raise ValueError(f'{label}: nonfinite profiles')
        if not np.isclose(pv.sum()/load.sum(), .4, atol=1e-6, rtol=0):
            raise ValueError(f'{label}: processed available PV/load ratio is not 40%')
        # Find scaling section without relying on its spelling.
        scaling = next(v for v in c.values() if isinstance(v, dict) and 'load_scaling_factor' in v)
        if scaling['load_scaling_factor'] != 1 or scaling['pv_scaling_factor'] != 1:
            raise ValueError(f'{label}: baseline environment factors must both equal 1')

    for label, original in [('IEEE33', IEEE33_CONFIG), ('IEEE69', IEEE69_CONFIG)]:
        c = copy.deepcopy(original)
        c['doe']['doe_enabled'] = True
        env = DESSEnv(c, mode='train', seed=123)
        current_base = c['network']['s_base_kva'] / (np.sqrt(3)*c['network']['v_base_kv'])
        np.testing.assert_allclose(env.grid.get_line_current_limits_pu()*current_base, 255)
        if env.constraint_handling != 'hard_feasibility' or env.action_correction_factor != .8:
            raise ValueError('Pilot requires original hard-feasibility backup with factor 0.8')
        obs, reset_info = env.reset(seed=123)
        records = []
        for t in range(args.steps):
            pre = obs['agent_obs'][:, 3].copy()
            action = np.full(env.num_agents, -.8 if t % 2 else .8)
            obs, reward, terminated, truncated, info = env.step(action)
            np.testing.assert_array_equal(info['pre_action_local_voltage_pu'], pre)
            np.testing.assert_allclose(info['doe_power_kw'], info['requested_battery_actual_power_kw'], atol=1e-8)
            for i, bus in enumerate(env.dess_buses):
                row = dict(episode_id=0, step=t, date_time=str(info['date_time']),
                           bus_number=bus+1, episode_start=str(reset_info['date_time']),
                           terminated=terminated, truncated=truncated)
                for key in ('marl_prepared_action', 'marl_requested_power_kw', 'pre_action_local_voltage_pu',
                            'soc_before', 'soc_after', 'physical_limited_power_kw',
                            'doe_power_kw', 'final_battery_power_kw',
                            'doe_preventive_clipped', 'doe_corrective_active',
                            'doe_corrective_intervened', 'doe_corrective_shortfall_kw'):
                    row[key] = np.asarray(info[key])[i].item()
                for key in ('pre_action_pf_converged', 'backup_activated',
                            'correction_iterations', 'correction_fallback_to_zero',
                            'zero_fallback_feasible', 'final_network_feasible',
                            'converged', 'min_voltage_pu', 'max_voltage_pu',
                            'max_voltage_violation', 'max_line_current_violation'):
                    row[key] = info[key]
                records.append(row)
            if terminated or truncated:
                break
        pd.DataFrame(records).to_csv(output / f'{label}_steps.csv', index=False)
        summary = dict(feeder=label, steps=t+1, current_limit_a=255,
                       final_feasible=bool(info['final_network_feasible']),
                       infeasible_steps=sum(not r['final_network_feasible'] for r in records[::env.num_agents]),
                       nonconverged_steps=sum(not r['converged'] for r in records[::env.num_agents]),
                       action_source='deterministic smoke inputs, not a trained policy')
        (output / f'{label}_summary.json').write_text(json.dumps(summary, indent=2))
        print(summary)


if __name__ == '__main__':
    main()
