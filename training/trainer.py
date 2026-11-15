"""
Trainer for graph-aware MADDPG DESS coordination.

Training loop:
    warm-up random rollout
    -> store graph transitions
    -> update MADDPG
    -> collect policy rollout
    -> periodically evaluate
    -> save checkpoint
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from training.transition_semantics import learning_valid, finite_tree
from utility.provenance import checkpoint_provenance
import torch

from models.maddpg import GraphAwareMADDPG
from training.replay_buffer import (
    GraphReplayBuffer,
    build_replay_buffer_from_env_specs,
)


class GraphMADDPGTrainer:
    def __init__(
        self,
        env,
        eval_env=None,
        config: Optional[dict] = None,
        device: Optional[str] = None,
    ):
        self.env = env
        self.eval_env = eval_env
        self.config = {} if config is None else config

        self.device = torch.device(
            device if device is not None else (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        )

        self.num_buses = int(env.num_buses)
        self.num_agents = int(env.num_agents)
        self.agent_obs_dim = int(env.agent_obs_dim)
        self.node_feature_dim = int(env.observation_space["x"].shape[-1])
        self.action_dim_per_agent = int(self.config.get("action_dim_per_agent", 1))

        self.batch_size = int(self.config.get("batch_size", 64))
        self.buffer_size = int(self.config.get("buffer_size", 100000))
        self.warmup_steps = int(self.config.get("warmup_steps", 5000))
        self.total_steps = int(self.config.get("total_steps", 100000))
        self.update_after = int(self.config.get("update_after", self.warmup_steps))
        self.update_every = int(self.config.get("update_every", 1))
        self.eval_every = int(self.config.get("eval_every", 5000))
        self.save_every = int(self.config.get("save_every", 10000))
        self.noise_std = float(self.config.get("noise_std", 0.05))
        # F9: explicit exploration settings, identical for every architecture (GNN and MLP).
        self.warmup_action_low = float(self.config.get("warmup_action_low", -1.0))
        self.warmup_action_high = float(self.config.get("warmup_action_high", 1.0))
        if not -1.0 <= self.warmup_action_low < self.warmup_action_high <= 1.0:
            raise ValueError("warmup action range must satisfy -1 <= low < high <= 1 (action-space limits)")
        if not (np.isfinite(self.noise_std) and self.noise_std >= 0):
            raise ValueError("noise_std must be finite and nonnegative")
        self.eval_episodes = int(self.config.get("eval_episodes", 3))
        self.max_eval_steps = int(
            self.config.get("max_eval_steps", getattr(env, "episode_length", 96))
        )
        self.verbose = bool(self.config.get("verbose", True))

        self.checkpoint_dir = Path(
            self.config.get("checkpoint_dir", "results/checkpoints")
        )
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.replay_buffer: GraphReplayBuffer = build_replay_buffer_from_env_specs(
            env=self.env,
            max_size=self.buffer_size,
            device=str(self.device),
            store_on_device=bool(self.config.get("store_buffer_on_device", True)),
        )

        self.agent = GraphAwareMADDPG(
            node_feature_dim=self.node_feature_dim,
            agent_obs_dim=self.agent_obs_dim,
            dess_buses=self.env.dess_buses,
            num_agents=self.num_agents,
            action_dim_per_agent=self.action_dim_per_agent,
            gamma=float(self.config.get("gamma", 0.99)),
            tau=float(self.config.get("tau", 0.005)),
            actor_lr=float(self.config.get("actor_lr", 1e-4)),
            critic_lr=float(self.config.get("critic_lr", 1e-3)),
            weight_decay=float(self.config.get("weight_decay", 0.0)),
            grad_clip_norm=self.config.get("grad_clip_norm", 1.0),
            device=str(self.device),
            share_actor=bool(self.config.get("share_actor", False)),
            gnn_type=str(self.config.get("gnn_type", "gcn")),
            gnn_hidden_dim=int(self.config.get("gnn_hidden_dim", 64)),
            gnn_embedding_dim=int(self.config.get("gnn_embedding_dim", 64)),
            gnn_num_layers=int(self.config.get("gnn_num_layers", 3)),
            actor_hidden_dims=tuple(
                self.config.get("actor_hidden_dims", (256, 256))
            ),
            critic_hidden_dims=tuple(
                self.config.get("critic_hidden_dims", (256, 256))
            ),
        )

        self.global_step = 0
        self.episode_count = 0
        self.failed_episode_count = 0
        self.invalid_transition_count = 0
        self.consecutive_invalid_transitions = 0
        self.max_consecutive_invalid_transitions = int(self.config.get("max_consecutive_invalid_transitions", 10))
        if self.max_consecutive_invalid_transitions < 1:
            raise ValueError("max_consecutive_invalid_transitions must be positive")
        self.pf_failed_solve_count = 0
        # F6: a failed training reset aborts (fail fast) by default. max_reset_retries > 0 is an
        # explicit, bounded opt-in: up to that many further env.reset() calls (the env RNG picks the
        # next training day deterministically). Every attempt and failure reason is recorded.
        self.max_reset_retries = int(self.config.get("max_reset_retries", 0))
        if not 0 <= self.max_reset_retries <= 10:
            raise ValueError("max_reset_retries must be an integer from 0 to 10")
        self.reset_attempts = 0
        self.reset_failures = []

    def _reset_env(self, env):
        # F6: invalid reset observations never reach the policy or replay. Bounded retries are
        # opt-in (max_reset_retries, default 0 = fail fast); training resets only, never evaluation.
        retries = int(getattr(self, "max_reset_retries", 0))
        if not hasattr(self, "reset_failures"):
            self.reset_failures = []
        reason = "invalid observation"
        for attempt in range(1, retries + 2):
            self.reset_attempts = getattr(self, "reset_attempts", 0) + 1
            output = env.reset()

            if isinstance(output, tuple):
                obs, info = output
            else:
                obs, info = output, {}

            if info.get("episode_failed", False) or not info.get("observation_valid", True) or not finite_tree(obs):
                self.failed_episode_count = getattr(self, "failed_episode_count", 0) + 1
                self.pf_failed_solve_count = getattr(self, "pf_failed_solve_count", 0) + int(info.get("pf_failed_solve_count", 0))
                reason = str(info.get("pf_failure_reason", "invalid observation"))
                self.reset_failures.append(dict(global_step=int(getattr(self, "global_step", 0)), attempt=attempt,
                                                start_index=getattr(env, "episode_start_index", None),
                                                reason=reason))
                continue
            return obs, info
        raise RuntimeError("Reset PF failed; no policy/replay update: " + reason +
                           " (%d attempt(s), max_reset_retries=%d)" % (retries + 1, retries))

    def reset_summary(self):
        """F6: reset attempts and failures, for the run manifest."""
        return dict(max_reset_retries=int(getattr(self, "max_reset_retries", 0)),
                    reset_attempts=int(getattr(self, "reset_attempts", 0)),
                    reset_failures=list(getattr(self, "reset_failures", [])))

    def exploration_config(self):
        """F9: effective exploration settings (recorded in the run manifest)."""
        return dict(controller=str(self.config.get("gnn_type", "gcn")), warmup_distribution="uniform",
                    warmup_action_low=self.warmup_action_low, warmup_action_high=self.warmup_action_high,
                    warmup_steps=self.warmup_steps, noise_type="gaussian, added to the deterministic actor output",
                    noise_std=self.noise_std, action_clipping="[-1, 1] after noise (actor.select_action)",
                    evaluation_noise_std=0.0,
                    rng="numpy global RNG (warmup actions) and torch global RNG (Gaussian noise), "
                        "both seeded by the training script set_seed(seed)")

    def _step_env(self, env, action):
        output = env.step(action)

        if len(output) == 5:
            next_obs, reward, terminated, truncated, info = output
            done = bool(terminated or truncated)
        elif len(output) == 4:
            next_obs, reward, done, info = output
            terminated = bool(done)
            truncated = False
        else:
            raise RuntimeError("env.step(action) must return either 4 or 5 values.")

        return next_obs, reward, done, terminated, truncated, info

    def _random_action(self):
        return np.random.uniform(
            low=getattr(self, "warmup_action_low", -1.0),   # F9: configurable; default = action-space limits
            high=getattr(self, "warmup_action_high", 1.0),  # GNN default was +-0.75
            size=(self.num_agents,),
        ).astype(np.float32)

    def collect_step(self, obs: Dict, random_action: bool = False):
        if not finite_tree(obs):
            raise ValueError("Nonfinite policy observation; refusing action selection")
        if random_action:
            requested_action = self._random_action()
        else:
            requested_action = self.agent.select_action(
                obs,
                noise_std=self.noise_std,
            )

        # Preserve the exact proposal after exploration/pre-step clipping.
        # Keep replay and the actor's output independent of env-side mutation.
        proposed_action = np.array(requested_action, copy=True)
        next_obs, reward, done, terminated, truncated, info = self._step_env(
            self.env,
            proposed_action.copy(),
        )

        # Reset on termination OR truncation, but bootstrap valid time limits.
        # Failed PF transitions never reach the critic, even with finite placeholders.
        valid = learning_valid(obs, proposed_action, reward, next_obs, info)
        info["replay_transition_stored"] = valid
        if valid:
            self.consecutive_invalid_transitions = 0
            self.replay_buffer.add(
                obs=obs,
                action=proposed_action,
                reward=reward,
                next_obs=next_obs,
                done=bool(terminated),
            )
        else:
            self.consecutive_invalid_transitions = getattr(self, "consecutive_invalid_transitions", 0) + 1
            self.invalid_transition_count = getattr(self, "invalid_transition_count", 0) + 1
        if info.get("episode_failed", False):
            self.failed_episode_count = getattr(self, "failed_episode_count", 0) + 1


        self.pf_failed_solve_count = getattr(self, "pf_failed_solve_count", 0) + int(info.get("step_pf_failed_solves", 0))
        self.global_step += 1
        if getattr(self, "consecutive_invalid_transitions", 0) >= getattr(self, "max_consecutive_invalid_transitions", 10):
            raise RuntimeError("Too many consecutive invalid transitions; stopping instead of filling replay with failed PF states")

        if done:
            self.episode_count += 1
            next_obs, _ = self._reset_env(self.env)

        return next_obs, reward, done, info

    def warmup(self):
        obs, _ = self._reset_env(self.env)

        while len(self.replay_buffer) < self.warmup_steps:
            obs, _, _, _ = self.collect_step(
                obs,
                random_action=True,
            )

            if self.verbose and len(self.replay_buffer) % 1000 == 0:
                print(
                    f"warmup buffer={len(self.replay_buffer)} "
                    f"global_step={self.global_step}"
                )

        return obs

    def train(self):
        obs = self.warmup()
        logs = []

        if self.verbose:
            print(
                f"Starting training from global_step={self.global_step}, "
                f"buffer_size={len(self.replay_buffer)}"
            )

        while self.global_step < self.total_steps:
            obs, reward, done, info = self.collect_step(
                obs,
                random_action=False,
            )

            update_log = None

            if (
                self.global_step >= self.update_after
                and self.replay_buffer.can_sample(self.batch_size)
                and self.global_step % self.update_every == 0
            ):
                batch = self.replay_buffer.sample(self.batch_size)
                update_log = self.agent.update(batch)

            if self.eval_env is not None and self.global_step % self.eval_every == 0:
                eval_log = self.evaluate()

                logs.append(
                    {
                        "step": self.global_step,
                        "episode": self.episode_count,
                        "update": update_log,
                        "eval": eval_log,
                    }
                )

                critic_loss_str = (
                    "None"
                    if update_log is None
                    else f"{update_log['critic_loss']:.6f}"
                )
                actor_loss_str = (
                    "None"
                    if update_log is None
                    else f"{update_log['actor_loss']:.6f}"
                )

                print(
                    f"step={self.global_step} "
                    f"episode={self.episode_count} "
                    f"eval_reward={eval_log['mean_reward']:.4f} "
                    f"eval_len={eval_log['mean_length']:.1f} "
                    f"critic_loss={critic_loss_str} "
                    f"actor_loss={actor_loss_str}"
                )

            if self.global_step % self.save_every == 0:
                self.save_checkpoint(
                    self.checkpoint_dir / f"maddpg_step_{self.global_step}.pt"
                )

        self.save_checkpoint(self.checkpoint_dir / "maddpg_final.pt")

        return logs

    @torch.no_grad()
    def evaluate(self):
        from utility.periodic_validation import evaluate_trainer
        return evaluate_trainer(self)

    def save_checkpoint(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.agent.save(str(path), provenance=checkpoint_provenance(self.env))

    def load_checkpoint(self, path):
        self.agent.load(str(path), env=self.env)
