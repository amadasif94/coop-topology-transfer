"""A1: inherited A0 learning equations and critics; slice ONLY actor inputs."""
from copy import deepcopy
import torch
from experiments.no_gnn_maddpg.maddpg_mlp import MLPMADDPG
from experiments.no_gnn_maddpg.mlp_actor import MultiAgentMLPActors
from experiments.no_shared_kpi.model_support import AblationCheckpoint, checked_kwargs


class LocalActors(MultiAgentMLPActors):
    def forward(self, agent_obs):
        if agent_obs.shape[-1] != 10:
            raise ValueError('Replay/policy interface must retain all ten original features')
        return super().forward(agent_obs[..., :7])


class NoSharedKPI(AblationCheckpoint, MLPMADDPG):
    variant = 'no_shared_kpi'

    def __init__(self, **kwargs):
        self.model_kwargs = checked_kwargs(kwargs)
        super().__init__(**kwargs)
        self.actor = LocalActors(7, self.num_agents, 1, (256,256), share_actor=False).to(self.device)
        self.target_actor = deepcopy(self.actor).to(self.device)
        self.target_actor.eval()
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(),
            lr=kwargs.get('actor_lr', 1e-4), weight_decay=kwargs.get('weight_decay', 0.0))
        # No update override: A0's mean of ALL critics' policy losses and joint gradients
        # are preserved exactly (A0 does not detach other agents' actions).
        # Critics/replay remain full-width; target actors apply the same seven-feature slice.
