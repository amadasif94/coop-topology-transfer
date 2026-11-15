"""B3 replay: the original GraphReplayBuffer plus two normalised scalars per battery per transition.

Schema 'b3-replay/1' adds feasible_action_lower / feasible_action_upper [max_size, num_agents] (agent order =
env.dess_buses). The bounds are NEVER reconstructed from (normalised, masked) observations: the B3 trainer hands
the exact pre-action bounds recorded by B3DESSEnv for the SAME step to add(), keyed to that step's next_obs
object, and add() refuses a transition without them. Saved buffers carry the schema; load() rejects any payload
(including every pre-B3 buffer) that lacks it or the bounds.
"""
import torch

from environments.battery_feasibility import check_bounds
from training.replay_buffer import GraphReplayBuffer, build_replay_buffer_from_env_specs

SCHEMA = 'b3-replay/1'


class B3ReplayBuffer(GraphReplayBuffer):
    schema = SCHEMA

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        shape = (self.max_size, self.num_agents)
        self.feasible_action_lower = torch.empty(shape, dtype=torch.float32, device=self.store_device)
        self.feasible_action_upper = torch.empty(shape, dtype=torch.float32, device=self.store_device)
        self._pending = None

    def set_pending_bounds(self, lower, upper, next_obs):
        """Bounds of the step that produced next_obs (recorded before that step's action was executed)."""
        self._pending = (*check_bounds(lower, upper, self.num_agents), next_obs)

    def add(self, obs, action, reward, next_obs, done, feasible_action_lower=None, feasible_action_upper=None):
        if feasible_action_lower is None and feasible_action_upper is None:
            if self._pending is None or self._pending[2] is not next_obs:
                raise ValueError('B3 replay transition without the pre-action feasible bounds of the same step')
            feasible_action_lower, feasible_action_upper = self._pending[:2]
        lower, upper = check_bounds(feasible_action_lower, feasible_action_upper, self.num_agents)
        self._pending = None
        idx = self.ptr
        super().add(obs=obs, action=action, reward=reward, next_obs=next_obs, done=done)
        self.feasible_action_lower[idx].copy_(self._to_tensor(lower))
        self.feasible_action_upper[idx].copy_(self._to_tensor(upper))

    def sample(self, batch_size):
        # Same draw as the original sampler (one torch.randint), so the bound rows match the transition rows.
        if self.size <= 0:
            raise ValueError('Cannot sample from an empty replay buffer.')
        indices = torch.randint(low=0, high=self.size, size=(int(batch_size),), device=self.store_device)
        batch = self._gather(indices)
        for name, tensor in batch.items():
            if not torch.isfinite(tensor).all():
                raise ValueError(f'Nonfinite replay sample: {name}; rebuild this buffer')
        return batch

    def _gather(self, indices):
        batch = dict(x=self.x[indices], agent_obs=self.agent_obs[indices], actions=self.actions[indices],
                     rewards=self.rewards[indices], dones=self.dones[indices], next_x=self.next_x[indices],
                     next_agent_obs=self.next_agent_obs[indices],
                     feasible_action_lower=self.feasible_action_lower[indices],
                     feasible_action_upper=self.feasible_action_upper[indices])
        batch = {k: v.to(self.device) for k, v in batch.items()}
        if self.edge_index is not None:
            batch['edge_index'] = self.edge_index.to(self.device)
        if self.edge_attr is not None:
            batch['edge_attr'] = self.edge_attr[indices].to(self.device)
            batch['next_edge_attr'] = self.next_edge_attr[indices].to(self.device)
        return batch

    def save(self, path):
        super().save(path)
        payload = torch.load(path, map_location='cpu')
        payload.update(schema=SCHEMA, feasible_action_lower=self.feasible_action_lower[:self.size].cpu(),
                       feasible_action_upper=self.feasible_action_upper[:self.size].cpu())
        torch.save(payload, path)

    def load(self, path):
        payload = torch.load(path, map_location=self.store_device)
        if payload.get('schema') != SCHEMA or payload.get('feasible_action_lower') is None \
                or payload.get('feasible_action_upper') is None:
            raise ValueError(f'{path}: replay schema {payload.get("schema")!r} has no stored feasible-action bounds; '
                             'B3 never reconstructs them from observations')
        super().load(path)
        n = int(payload['size'])
        self.feasible_action_lower[:n].copy_(payload['feasible_action_lower'].to(self.store_device))
        self.feasible_action_upper[:n].copy_(payload['feasible_action_upper'].to(self.store_device))


def build_b3_replay_buffer(env, max_size, device=None, store_on_device=True):
    """Same construction as build_replay_buffer_from_env_specs, with the B3 buffer class."""
    base = build_replay_buffer_from_env_specs(env, 1, device=device, store_on_device=store_on_device)
    return B3ReplayBuffer(max_size=max_size, num_buses=base.num_buses, node_feature_dim=base.node_feature_dim,
                          num_agents=base.num_agents, agent_obs_dim=base.agent_obs_dim,
                          action_dim_per_agent=base.action_dim_per_agent,
                          edge_index=getattr(env.grid, 'edge_index', None),
                          edge_attr_shape=None if base.edge_attr is None else tuple(base.edge_attr.shape[1:]),
                          device=device, store_on_device=store_on_device)


# Maintained names (the B3-era names above are kept for historical checkpoints/replays and tests; SCHEMA is a data
# format identifier recorded in manifests and is unchanged).
BoundedReplayBuffer = B3ReplayBuffer
build_bounded_replay_buffer = build_b3_replay_buffer
