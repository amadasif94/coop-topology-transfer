"""A3 learner: fully independent actor-critics on the seven LOCAL features (DTDE).

Actor i (A1 LocalActors):   a_i = tanh(net_i(o_i)),        o_i = agent_obs[i, :7]
Critic i (local):           Q_i(o_i, a_i)                    8 inputs: own seven local features and own action
Each critic is trained on its own reward column r_i only; the shared feeder KPI features 7-9 of the ten-feature
interface, other agents' observations, actions and rewards never enter agent i's critic, target or actor graph.
The update follows A2's IndependentAgents.update (same ordering, targets, mean-over-agents actor loss, actor-wide
gradient clipping, separate per-agent optimisers); the only differences are the seven-feature slice and the local
critic input size. Dones are A2's: any agent done ends the bootstrap for all (environment-level day boundary).
"""
from copy import deepcopy
from time import perf_counter

import torch
from torch import nn
from torch.nn import functional as F

from experiments.no_gnn_maddpg.maddpg_mlp import MLPMADDPG
from experiments.no_gnn_maddpg.mlp_critic import build_mlp
from experiments.no_shared_kpi.model_no_shared_kpi import LocalActors
from utility.numerical_guards import require_finite

LOCAL_FEATURES = 7          # time, price, own SoC, own-bus voltage, own-bus load, own-bus PV, own previous action


class LocalIndependentCritic(nn.Module):
    def __init__(self, hidden_dims=(256, 256)):
        super().__init__()
        self.q_network = build_mlp(LOCAL_FEATURES + 1, tuple(hidden_dims), 1)

    def forward(self, own_local_obs, own_action):
        if own_local_obs.ndim != 2 or own_local_obs.shape[-1] != LOCAL_FEATURES or \
                own_action.shape != (own_local_obs.shape[0], 1):
            raise ValueError('Local critic requires ONLY [B,7] own local observation and [B,1] own action')
        return self.q_network(torch.cat((own_local_obs, own_action), dim=-1))


class IndependentLocalAgents(MLPMADDPG):
    def __init__(self, **kwargs):
        MLPMADDPG.__init__(self, **kwargs)
        if self.agent_obs_dim != 10 or self.action_dim_per_agent != 1:
            raise ValueError('A3 requires the ten-feature interface and scalar actions')
        hidden = tuple(kwargs.get('actor_hidden_dims', (256, 256)))
        self.actor = LocalActors(LOCAL_FEATURES, self.num_agents, 1, hidden, share_actor=False).to(self.device)
        self.target_actor = deepcopy(self.actor).to(self.device)
        self.target_actor.eval()
        self.critics = nn.ModuleList([LocalIndependentCritic(kwargs.get('critic_hidden_dims', (256, 256)))
                                      for _ in range(self.num_agents)]).to(self.device)
        self.target_critics = deepcopy(self.critics).to(self.device)
        self.target_critics.eval()
        wd = kwargs.get('weight_decay', 0.0)
        self.critic_optimizers = [torch.optim.AdamW(c.parameters(), lr=kwargs.get('critic_lr', 1e-3), weight_decay=wd)
                                  for c in self.critics]
        self.actor_optimizers = [torch.optim.AdamW(a.parameters(), lr=kwargs.get('actor_lr', 1e-4), weight_decay=wd)
                                 for a in self.actor.actors]
        del self.actor_optimizer

    def update(self, batch):
        started = perf_counter()
        obs = self._to_device(batch['agent_obs'])
        nxt = self._to_device(batch['next_agent_obs'])
        actions = self._to_device(batch['actions'])
        if actions.ndim == 2: actions = actions.unsqueeze(-1)
        rewards = self._to_device(batch['rewards'])
        if rewards.ndim != 2 or rewards.shape[1] != self.num_agents:
            raise ValueError('A3 requires one individual reward per agent')
        dones = self._to_device(batch['dones']).reshape(obs.shape[0], -1).bool().any(1, keepdim=True).float()
        if obs.shape != nxt.shape or obs.shape[1:] != (self.num_agents, 10) or actions.shape != (len(obs), self.num_agents, 1):
            raise ValueError('Unexpected replay dimensions')
        for name, v in dict(obs=obs, nxt=nxt, actions=actions, rewards=rewards, dones=dones).items():
            require_finite(v, name)
        local, next_local = obs[..., :LOCAL_FEATURES], nxt[..., :LOCAL_FEATURES]
        logs = {}
        for i in range(self.num_agents):
            # Slice BEFORE any network call: only agent i's local features, action and reward enter its graph.
            o, no, a, r = local[:, i], next_local[:, i], actions[:, i], rewards[:, i:i + 1]
            with torch.no_grad():
                y = r + self.gamma * (1 - dones) * self.target_critics[i](no, self.target_actor.actors[i](no))
            loss = F.mse_loss(self.critics[i](o, a), y)
            self._optimize(loss, self.critics[i], self.critic_optimizers[i])
            logs[f'critic_loss_agent_{i}'] = float(loss.detach())
        self._set_critics_requires_grad(False)
        try:
            actor_losses = [-self.critics[i](local[:, i], self.actor.actors[i](local[:, i])).mean()
                            for i in range(self.num_agents)]
            actor_loss = torch.stack(actor_losses).mean()
            require_finite(actor_loss, 'independent actor loss')
            for optimizer in self.actor_optimizers: optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_clip_norm is not None:
                require_finite(nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip_norm), 'actor gradient')
            for optimizer in self.actor_optimizers: optimizer.step()
            for p in self.actor.parameters(): require_finite(p, 'actor parameter')
        finally:
            self._set_critics_requires_grad(True)
        for i, loss in enumerate(actor_losses): logs[f'actor_loss_agent_{i}'] = float(loss.detach())
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critics, self.critics)
        logs['critic_loss'] = sum(logs[f'critic_loss_agent_{i}'] for i in range(self.num_agents)) / self.num_agents
        logs['actor_loss'] = sum(logs[f'actor_loss_agent_{i}'] for i in range(self.num_agents)) / self.num_agents
        logs['update_time_ms'] = 1000 * (perf_counter() - started)
        return logs

    def _optimize(self, loss, module, optimizer):
        require_finite(loss, 'independent loss')
        optimizer.zero_grad()
        loss.backward()
        if self.grad_clip_norm is not None:
            require_finite(nn.utils.clip_grad_norm_(module.parameters(), self.grad_clip_norm), 'independent gradient', optimizer)
        optimizer.step()
        for p in module.parameters(): require_finite(p, 'independent parameter')
