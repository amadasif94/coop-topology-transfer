"""A2: independent-critic multi-agent actor-critic with unchanged shared rewards."""
from copy import deepcopy
from time import perf_counter
import torch
from torch import nn
from torch.nn import functional as F
from experiments.no_gnn_maddpg.maddpg_mlp import MLPMADDPG
from experiments.no_gnn_maddpg.mlp_critic import build_mlp
from experiments.no_shared_kpi.model_support import AblationCheckpoint, checked_kwargs
from utility.numerical_guards import require_finite


class IndependentCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_network = build_mlp(11, (256,256), 1)

    def forward(self, own_obs, own_action):
        if own_obs.ndim != 2 or own_obs.shape[-1] != 10 or own_action.shape != (own_obs.shape[0], 1):
            raise ValueError('Independent critic requires ONLY [B,10] own observation and [B,1] own action')
        return self.q_network(torch.cat((own_obs, own_action), dim=-1))


class IndependentAgents(AblationCheckpoint, MLPMADDPG):
    variant = 'no_centralized_critic'

    def __init__(self, **kwargs):
        self.model_kwargs = checked_kwargs(kwargs)
        super().__init__(**kwargs)
        # Original full-observation actors and target actors are retained, agent for agent.
        self.critics = nn.ModuleList([IndependentCritic() for _ in range(self.num_agents)]).to(self.device)
        self.target_critics = deepcopy(self.critics).to(self.device)
        self.target_critics.eval()
        self.critic_optimizers = [torch.optim.AdamW(c.parameters(), lr=kwargs.get('critic_lr', 1e-3),
            weight_decay=kwargs.get('weight_decay', 0.0)) for c in self.critics]
        self.actor_optimizers = [torch.optim.AdamW(a.parameters(), lr=kwargs.get('actor_lr', 1e-4),
            weight_decay=kwargs.get('weight_decay', 0.0)) for a in self.actor.actors]
        del self.actor_optimizer

    def update(self, batch):
        started = perf_counter()
        obs = self._to_device(batch['agent_obs'])
        nxt = self._to_device(batch['next_agent_obs'])
        actions = self._to_device(batch['actions'])
        if actions.ndim == 2: actions = actions.unsqueeze(-1)
        rewards = self._to_device(batch['rewards'])
        if rewards.ndim == 1: rewards = rewards.unsqueeze(-1)
        if rewards.shape[1] == 1: rewards = rewards.expand(-1, self.num_agents)
        dones = self._to_device(batch['dones']).reshape(obs.shape[0], -1).bool().any(1, keepdim=True).float()
        if obs.shape != nxt.shape or obs.shape[1:] != (self.num_agents, 10) or actions.shape != (len(obs), self.num_agents, 1):
            raise ValueError('Unexpected replay dimensions')
        for name, v in dict(obs=obs, nxt=nxt, actions=actions, rewards=rewards, dones=dones).items():
            require_finite(v, name)
        logs = {}
        for i in range(self.num_agents):
            # Slice BEFORE calling any critic: other agents never enter its forward graph.
            o, no, a = obs[:, i], nxt[:, i], actions[:, i]
            with torch.no_grad():
                na = self.target_actor.actors[i](no)
                y = rewards[:, i:i+1] + self.gamma*(1-dones)*self.target_critics[i](no, na)
            c = self.critics[i]
            loss = F.mse_loss(c(o, a), y)
            self._optimize(loss, c, self.critic_optimizers[i])
            logs[f'critic_loss_agent_{i}'] = float(loss.detach())
        # Preserve A0's update ordering, mean-over-agents loss normalization and
        # actor-wide gradient clipping. Separate optimizers own disjoint actors;
        # each actor's gradient comes ONLY from its corresponding local critic.
        # Averaging these separable losses does not expose another agent to a critic.
        self._set_critics_requires_grad(False)
        try:
            actor_losses=[-self.critics[i](obs[:,i],self.actor.actors[i](obs[:,i])).mean()
                          for i in range(self.num_agents)]
            actor_loss=torch.stack(actor_losses).mean()
            require_finite(actor_loss,'independent actor loss')
            for optimizer in self.actor_optimizers: optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_clip_norm is not None:
                require_finite(nn.utils.clip_grad_norm_(self.actor.parameters(),self.grad_clip_norm),'actor gradient')
            for optimizer in self.actor_optimizers: optimizer.step()
            for p in self.actor.parameters():require_finite(p,'actor parameter')
        finally:
            self._set_critics_requires_grad(True)
        for i,loss in enumerate(actor_losses):logs[f'actor_loss_agent_{i}']=float(loss.detach())
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critics, self.critics)
        logs['critic_loss'] = sum(logs[f'critic_loss_agent_{i}'] for i in range(self.num_agents))/self.num_agents
        logs['actor_loss'] = sum(logs[f'actor_loss_agent_{i}'] for i in range(self.num_agents))/self.num_agents
        logs['update_time_ms'] = 1000*(perf_counter()-started)
        return logs

    def _optimize(self, loss, module, optimizer):
        require_finite(loss, 'independent loss')
        optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm is not None:
            require_finite(nn.utils.clip_grad_norm_(module.parameters(), self.grad_clip_norm), 'independent gradient', optimizer)
        optimizer.step()
        for p in module.parameters(): require_finite(p, 'independent parameter')
