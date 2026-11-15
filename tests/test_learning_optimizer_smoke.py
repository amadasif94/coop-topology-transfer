"""Cluster validation with actual replay, target networks and optimizer steps.

Requires torch/PyG and production CSVs. Writes only a NEW output directory.
No checkpoints are loaded or overwritten. These are seeded smoke models.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    import torch
    from training.trainer import GraphMADDPGTrainer
    from experiments.no_gnn_maddpg.trainer_mlp import MLPMADDPGTrainer
    from environments.primary_marl import make_primary_env, validate_primary_env
    from utility.run_manifest import RunManifest
    from ac_intervention_fixture import exercise_interventions
    from environments.directional_backup import evaluation_fields
    from configs.ieee33_config import IEEE33_CONFIG
    from configs.ieee69_config import IEEE69_CONFIG

    torch.set_num_threads(1)
    output = Path(args.output)
    manifest = RunManifest(output, 'GCN+MLP', 123, 'smoke')
    results = []
    for feeder, original in ((33, IEEE33_CONFIG), (69, IEEE69_CONFIG)):
        cfg = copy.deepcopy(original)
        assert cfg['data_scaling']['load_scaling_factor'] == 1.0
        assert cfg['data_scaling']['pv_scaling_factor'] == 1.0
        lines = pd.read_csv(cfg['paths']['line_data'])
        assert np.allclose(lines.loc[lines.status > 0, 'max_current_a'], 255, rtol=0, atol=1e-9)
        ts = pd.read_csv(cfg['paths']['processed_time_series'])
        ratio = ts.filter(regex='^renewable_active_power_node_').to_numpy().sum() / ts.filter(regex='^active_power_node_').to_numpy().sum()
        assert np.isclose(ratio, .4, rtol=0, atol=1e-6)
        if feeder == 33:
            from experiments.topology_generalization.topology_utils import build_topology_config
        else:
            from experiments.topology_generalization.topology_utils_ieee69 import build_topology_config
        unseen_cfg = build_topology_config('TP2', base_config=cfg, output_root=output/f'topology_{feeder}')
        unseen = make_primary_env(unseen_cfg, mode='val', seed=123)
        unseen_obs, unseen_info = unseen.reset(seed=123)
        assert not unseen_info.get('episode_failed', False)
        assert np.isfinite(unseen_obs['agent_obs']).all()
        manifest.add_environment(unseen, topology='TP2')
        for name, trainer_class in (('gcn', GraphMADDPGTrainer), ('mlp', MLPMADDPGTrainer)):
            torch.manual_seed(123)
            np.random.seed(123)
            env = make_primary_env(cfg, mode='train', seed=123)
            manifest.add_environment(env)
            amps = env.grid.get_line_current_limits_pu() * cfg['network']['s_base_kva'] / (np.sqrt(3)*cfg['network']['v_base_kv'])
            assert np.allclose(amps, 255, rtol=0, atol=1e-6)
            trainer = trainer_class(env, device='cpu', config=dict(buffer_size=64,
                batch_size=4, gnn_type='gcn', gnn_hidden_dim=16, gnn_embedding_dim=16,
                gnn_num_layers=1, actor_hidden_dims=(16,), critic_hidden_dims=(16,),
                checkpoint_dir=str(output/f'{feeder}_{name}')))
            obs, _ = trainer._reset_env(env)
            training_rows = []
            for _ in range(16):
                captured = {}
                real_step = env.step
                def capture_step(action):
                    captured['proposal'] = np.array(action, copy=True)
                    captured['result'] = real_step(action)
                    return captured['result']
                with patch.object(env, 'step', side_effect=capture_step):
                    with patch.object(trainer.replay_buffer, 'add', wraps=trainer.replay_buffer.add) as add:
                        obs, reward, _, info = trainer.collect_step(obs)
                stored = add.call_args.kwargs
                np.testing.assert_array_equal(stored['action'], captured['proposal'])
                np.testing.assert_array_equal(stored['reward'], captured['result'][1])
                for key in captured['result'][0]:
                    np.testing.assert_array_equal(stored['next_obs'][key], captured['result'][0][key])
                assert info['doe_enabled'] and info['backup_mode'] == 'directional_incremental'
                for key in ('doe_preventive_clipped', 'doe_corrective_intervened',
                            'doe_power_kw', 'final_battery_power_kw', 'backup_activated',
                            'major_backup_penalty_applied', 'major_backup_penalty_amount'):
                    assert key in info, key
                expected_penalty = (env.reward_cfg['infeasible_action_penalty']
                                    if info['major_backup_penalty_applied'] else 0.0)
                assert info['major_backup_penalty_amount'] == expected_penalty
                training_rows.append(evaluation_fields(info))
                assert info['replay_transition_stored'] and not info['episode_failed']
                assert info['step_pf_failed_solves'] == 0
            assert len(trainer.replay_buffer) == 16
            bad_obs = dict(obs, x=np.full_like(obs['x'], np.nan))
            try:
                trainer.replay_buffer.add(bad_obs, np.zeros(env.num_agents),
                                          np.zeros(env.num_agents), obs, False)
            except ValueError:
                pass
            else:
                raise AssertionError('Replay accepted NaN input')
            assert len(trainer.replay_buffer) == 16

            before_actor = [p.detach().clone() for p in trainer.agent.actor.parameters()]
            before_critic = [p.detach().clone() for p in trainer.agent.critics.parameters()]
            for update in range(8):
                batch = trainer.replay_buffer.sample(4)
                # Exercise actual terminal masking in update() with valid states.
                batch['dones'][:] = torch.tensor([[0.], [1.], [0.], [1.]])
                kw = dict(agent_obs=batch['next_agent_obs'])
                if name == 'gcn':
                    kw.update(x=batch['next_x'], edge_index=batch['edge_index'],
                              edge_attr=batch.get('next_edge_attr'))
                with torch.no_grad():
                    actions = trainer.agent.target_actor(**kw)
                    expected = [float((batch['rewards'][:, i:i+1] + trainer.agent.gamma *
                        (1-batch['dones']) * critic(**kw, actions=actions)).mean())
                        for i, critic in enumerate(trainer.agent.target_critics)]
                log = trainer.agent.update(batch)
                np.testing.assert_allclose(log['mean_target_q'], np.mean(expected), rtol=1e-5, atol=1e-5)
                assert np.isfinite([log['actor_loss'], log['critic_loss']]).all()
                assert all(torch.isfinite(p).all() for module in (trainer.agent.actor, trainer.agent.critics) for p in module.parameters())
            assert any(not torch.equal(a, b) for a, b in zip(before_actor, trainer.agent.actor.parameters()))
            assert any(not torch.equal(a, b) for a, b in zip(before_critic, trainer.agent.critics.parameters()))
            # Real env + real buffer: failed dispatch must NOT append any sample.
            before_size = len(trainer.replay_buffer)
            before_soc = env._snapshot_battery_states()
            actual_reset = trainer._reset_env
            # Avoid resetting SOC before checking rollback; the collector still requests reset.
            with patch.object(env.power_flow, 'run_power_flow', side_effect=RuntimeError('injected PF')):
                with patch.object(trainer, '_reset_env', return_value=(obs, {})) as reset:
                    _, _, done, info = trainer.collect_step(obs)
            assert done and info['truncated'] and not info['terminated']
            assert info['episode_failed'] and not info['replay_transition_stored']
            assert reset.call_count == 1 and len(trainer.replay_buffer) == before_size
            assert env._snapshot_battery_states() == before_soc
            obs, _ = actual_reset(env)
            # Frozen evaluation uses the same permanent DOE-ON protocol.
            validate_primary_env(env)
            evaluation_rows = []
            obs, _ = env.reset(seed=321)
            for _ in range(16):
                action = trainer.agent.select_action(obs, noise_std=0)
                obs, reward, term, trunc, info = env.step(action)
                assert info['doe_enabled'] and info['backup_mode'] == 'directional_incremental'
                assert not info['episode_failed'] and info['step_pf_failed_solves'] == 0
                evaluation_rows.append(evaluation_fields(info))
                if term or trunc:
                    obs, _ = env.reset(seed=321)
            pd.DataFrame(training_rows).to_csv(output/f'{feeder}_{name}_training.csv', index=False)
            pd.DataFrame(evaluation_rows).to_csv(output/f'{feeder}_{name}_evaluation.csv', index=False)
            result = dict(feeder=feeder, controller=name, transitions=16, optimizer_updates=8,
                          doe_enabled=True, backup_mode='directional_incremental',
                          proposal_reward_next_state_verified=True,
                          target_formula_verified=True, failure_excluded=True, evaluation_steps=16,
                          major_penalized_training_steps=sum(bool(r['major_backup_penalty_applied']) for r in training_rows))
            stress_fields = exercise_interventions(env)
            pd.DataFrame([stress_fields]).to_csv(output/f'{feeder}_{name}_controlled_stress.csv', index=False)
            result['controlled_doe_backup_major_penalty_verified'] = True
            results.append(result)
            print(result, flush=True)
    (output/'results.json').write_text(json.dumps(results, indent=2))
    manifest.finish()


if __name__ == '__main__':
    main()
