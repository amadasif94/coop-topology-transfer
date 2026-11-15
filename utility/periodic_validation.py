"""Periodic validation reporting only; no replay writes or optimizer changes."""
from pathlib import Path
import numpy as np
import pandas as pd
from utility.evaluation_reporting import observation_ready, failure_row, step_fields, finish_episode, diagnostic_summary, reset_for_evaluation
from utility.run_manifest import RunManifest
from utility.validation_schedule import periodic_schedule


def evaluate_trainer(trainer):
    env = trainer.eval_env
    if env is None:
        raise ValueError('eval_env must be provided; do not evaluate on training environment')
    controller = trainer.config.get('gnn_type', 'gcn') if 'Graph' in type(trainer).__name__ else 'MLP'
    output = Path(trainer.checkpoint_dir)/'validation'/f'step_{trainer.global_step}'
    # F16: the same fixed 8-day schedule (S-PER-v1) at every evaluation, instead of random days.
    # A smaller explicit eval_episodes evaluates the first N scheduled days (still fixed); a larger
    # one is capped at the schedule. Both are reported and recorded.
    schedule = periodic_schedule(env)
    planned = schedule['episodes']
    days = min(int(trainer.eval_episodes), len(planned)) if trainer.eval_episodes else len(planned)
    if trainer.eval_episodes != len(planned):
        print(f'NOTE: periodic validation uses the fixed {schedule["schedule_id"]} schedule: '
              f'{days} of its {len(planned)} days (eval_episodes={trainer.eval_episodes})', flush=True)
    planned = planned[:days]
    run = RunManifest(output, controller, None, 'validation',
                      arguments={'training_step': trainer.global_step, 'run_purpose': 'periodic_validation',
                                 'data_split': 'val', 'requested_eval_episodes': trainer.eval_episodes,
                                 'schedule_days_evaluated': days})
    run.add_environment(env, intended_episodes=len(planned))
    run.record_schedule(schedule)
    all_rows, episodes = [], []
    for ep, planned_episode in enumerate(planned):
        start = planned_episode['start_index']
        obs, info = reset_for_evaluation(env, start, planned_episode['seed'])
        if not observation_ready(obs, info):
            rows = failure_row(env, info, ep, start, policy='maddpg')
        else:
            rows = []
            for step in range(trainer.max_eval_steps):
                action = trainer.agent.select_action(obs, noise_std=0.0)
                obs, reward, done, term, trunc, info = trainer._step_env(env, action)
                row = dict(episode_id=ep, start_index=start, step=step,
                           reward_mean=float(np.mean(reward)), feasible=info['feasible'],
                           converged=info['converged'], **step_fields(info, env, term, trunc, obs))
                rows.append(row)
                if done or row['episode_failed']: break
            rows = finish_episode(rows, env)
        all_rows.extend(rows)
        frame = pd.DataFrame(rows)
        stats = diagnostic_summary(frame, episode=True)
        # Episode reward only for completed, failure-free episodes (NaN otherwise).
        valid = bool(stats['completed_episodes'] == 1 and stats['failed_episodes'] == 0)
        stats.update(episode_id=ep, schedule_episode_id=planned_episode['episode_id'], start_index=start,
                     seed=planned_episode['seed'], reward=frame['reward_mean'].sum(min_count=1) if valid else np.nan,
                     reward_valid=valid,
                     length=rows[-1]['episode_attempted_steps'],
                     all_feasible=bool(rows[-1]['episode_completed'] and frame['feasible'].all()))
        episodes.append(stats)
    step_df, ep_df = pd.DataFrame(all_rows), pd.DataFrame(episodes)
    step_df.to_csv(output/'step_metrics.csv', index=False)
    ep_df.to_csv(output/'episode_summary.csv', index=False)
    result = dict(mean_reward=ep_df.reward.mean(), std_reward=ep_df.reward.std(ddof=0),
                  mean_length=ep_df.length.mean(), all_feasible_rate=ep_df.all_feasible.mean(),
                  intended_episodes=len(episodes), completed_episodes=int(ep_df.completed_episodes.sum()),
                  failed_episodes=int(ep_df.failed_episodes.sum()), incomplete_episodes=int(ep_df.incomplete_episodes.sum()),
                  reward_valid_episodes=int(ep_df.reward_valid.sum()),
                  schedule_id=schedule['schedule_id'], schedule_sha256=schedule['sha256'],
                  schedule_days_evaluated=days)
    result.update(diagnostic_summary(step_df))
    pd.DataFrame([result]).to_csv(output/'aggregate_summary.csv', index=False)
    run.finish()
    return result
