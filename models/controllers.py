"""Maintained controller registry: the ORIGINAL MLP (A0-A3) and graph (GCN/GAT/TAGConv) MADDPG models.

    mlp_a0  experiments.no_gnn_maddpg.maddpg_mlp.MLPMADDPG                (10-feature actors, centralised critics)
    mlp_a1  MLPMADDPG with experiments.no_shared_kpi LocalActors          (7 local features, centralised critics)
    mlp_a2  experiments.no_centralized_critic IndependentAgents.update    (own observation + own action critics)
    mlp_a3  models.independent_local_agents IndependentLocalAgents        (7 local features, local own-action critics,
                                                                           individual rewards; fully independent DTDE)
    gcn / gat / tagconv  models.maddpg.GraphAwareMADDPG                   (graph-aware actors and critics)
Architectures, critics, optimisers, targets, replay semantics and the original update() are unchanged (the original
modules are hash-pinned and imported, never copied).

Options (keyword arguments, removed before the original constructors and NOT part of model_kwargs):
  action_mapping           'soc_feasible' (environments.battery_feasibility.soc_feasible_action) or 'rated' (historical)
  feasibility_supervision  False (soc_feasible): lambda_f must be 0 and training.feasibility_supervision only READS
                           diagnostics; update() is exactly the original one for every controller. True (historical
                           rated protocols): pre-tanh SoC-feasibility supervision with lambda_f.
  legacy_names             False: checkpoints use schema 'graph-marl-dess-checkpoint/1' and the controller key
                           (mlp_a0 ... tagconv) as label. True (historical entry points only): schema 'b3-checkpoint/2'
                           and labels b3_mlp_a0 ... b3_tagconv, exactly as historical runs.
Loading accepts the maintained schema, 'b3-checkpoint/2' and the historical post-tanh 'b3-checkpoint/1', and either
label of the same controller. load(path, env) refuses a checkpoint whose action mapping differs from the environment's
(absent in older checkpoints = 'rated'), so a rated-power checkpoint is never evaluated as SOC-feasible.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from configs.experiments import ACTION_MAPPINGS, CONTROLLER_KEYS, LEGACY_LABELS, action_mapping_version_of
from environments.battery_feasibility import SOC_FEASIBLE_MAPPING_VERSION
from experiments.no_centralized_critic.model_independent_critic import IndependentAgents, IndependentCritic
from experiments.no_gnn_maddpg.maddpg_mlp import MLPMADDPG
from experiments.no_shared_kpi.model_no_shared_kpi import LocalActors
from models.independent_local_agents import LOCAL_FEATURES, IndependentLocalAgents
from training.feasibility_supervision import FORMULATION, HISTORICAL_FORMULATION, FeasibilitySupervision
from utility.provenance import check_checkpoint_provenance

CHECKPOINT_SCHEMA = 'graph-marl-dess-checkpoint/1'
LEGACY_SCHEMA = 'b3-checkpoint/2'                     # same record layout, written by the historical entry points
HISTORICAL_SCHEMAS = {'b3-checkpoint/1': HISTORICAL_FORMULATION}
LOCAL_REWARD_CONTROLLERS = ('mlp_a3',)      # trained/evaluated in environments.local_reward_env.A3DESSEnv
GNN_TYPES = dict(gcn='gcn', gat='gat', tagconv='tagconv')


class ControllerMixin:
    controller_key = None
    legacy_names_default = False
    lambda_feasibility = 0.0
    feasibility_formulation = FORMULATION
    action_mapping = 'rated'
    feasibility_supervision = True

    def _init_controller(self, lambda_feasibility, kwargs):
        if lambda_feasibility is None or not np.isfinite(lambda_feasibility) or lambda_feasibility < 0:
            raise ValueError('lambda_feasibility must be finite and nonnegative')
        if kwargs.get('share_actor', False):
            raise ValueError('Controllers require separate per-agent actors (share_actor=False)')
        self.action_mapping = kwargs.pop('action_mapping', 'rated')
        self.feasibility_supervision = bool(kwargs.pop('feasibility_supervision', True))
        self.legacy_names = bool(kwargs.pop('legacy_names', self.legacy_names_default))
        if self.action_mapping not in ACTION_MAPPINGS:
            raise ValueError(f'Unknown action_mapping {self.action_mapping}')
        # New models are trained with the current soc_feasible mapping version (None for 'rated').
        self.action_mapping_version = SOC_FEASIBLE_MAPPING_VERSION if self.action_mapping == 'soc_feasible' else None
        if not self.feasibility_supervision and float(lambda_feasibility) != 0.0:
            raise ValueError('feasibility_supervision=False requires lambda_feasibility = 0')
        if not self.feasibility_supervision:
            self.feasibility_formulation = 'none'
        self.lambda_feasibility = float(lambda_feasibility)
        self.model_kwargs = {k: list(v) if isinstance(v, tuple) else v for k, v in kwargs.items() if k != 'device'}
        self.experiment_metadata = {}
        self.capture = None             # calibration/verification: per-element tensors of the next update
        self.diagnostics_every = 0      # detailed (component gradient) diagnostics every N updates; 0 = never
        self.diagnostics_callback = None
        self.update_count = 0

    def _set_key(self, key):
        if key not in CONTROLLER_KEYS:
            raise ValueError(f'Unknown controller {key}')
        self.controller_key = key
        self.controller = LEGACY_LABELS[key] if self.legacy_names else key      # label written to checkpoints

    def _actor_forward(self, actor, batch):
        """Actions of `actor` on sampled replay states, built exactly as this controller's update builds them."""
        return actor(agent_obs=self._to_device(batch['agent_obs']))

    def update(self, batch):
        if 'feasible_action_lower' not in batch or 'feasible_action_upper' not in batch:
            raise ValueError('Controller update requires replay-stored feasible_action_lower/upper')
        expected = FORMULATION if self.feasibility_supervision else 'none'
        if self.feasibility_formulation != expected:
            raise ValueError(f'Training is only supported with {expected}; this model was loaded from a historical '
                             f'{self.feasibility_formulation} checkpoint (evaluation only)')
        self.update_count += 1
        detailed = self.capture is not None or (self.diagnostics_every and self.update_count % self.diagnostics_every == 0)
        with FeasibilitySupervision(self, batch, self.lambda_feasibility, detailed=bool(detailed),
                                    attach_loss=self.feasibility_supervision) as supervision:
            logs = super().update(batch)
        diagnostics, capture = supervision.results()
        logs.update(diagnostics, update_count=self.update_count)
        if self.capture is not None:
            self.capture.update(capture)
        if detailed and self.diagnostics_callback is not None:
            self.diagnostics_callback(logs)
        return logs

    # ------------------------------------------------------------------------- checkpoints
    def save(self, path, provenance=None):
        path = Path(path)
        optimizers = self.actor_optimizers if hasattr(self, 'actor_optimizers') else [self.actor_optimizer]
        record = dict(schema=LEGACY_SCHEMA if self.legacy_names else CHECKPOINT_SCHEMA, controller=self.controller,
                      model_kwargs=self.model_kwargs,
                      feasibility_formulation=self.feasibility_formulation, action_mapping=self.action_mapping,
                      action_mapping_version=self.action_mapping_version,
                      feasibility_supervision=self.feasibility_supervision,
                      lambda_feasibility=self.lambda_feasibility, experiment_metadata=self.experiment_metadata,
                      provenance=provenance, actor=self.actor.state_dict(), critics=self.critics.state_dict(),
                      target_actor=self.target_actor.state_dict(), target_critics=self.target_critics.state_dict(),
                      actor_optimizers=[o.state_dict() for o in optimizers],
                      critic_optimizers=[o.state_dict() for o in self.critic_optimizers])
        with path.open('xb') as f:                   # never overwrite a checkpoint
            torch.save(record, f)

    def load(self, path, env=None):
        c = torch.load(path, map_location=self.device, weights_only=False)
        schema = c.get('schema')
        if schema in (CHECKPOINT_SCHEMA, LEGACY_SCHEMA):
            formulation = c.get('feasibility_formulation')
        elif schema in HISTORICAL_SCHEMAS:
            formulation = HISTORICAL_SCHEMAS[schema]
        else:
            raise ValueError(f'{path}: not a controller checkpoint (schema {schema!r})')
        if c.get('controller') not in (self.controller_key, LEGACY_LABELS[self.controller_key]):
            raise ValueError(f'{path}: checkpoint controller {c.get("controller")!r}, model {self.controller_key!r}')
        if c['model_kwargs'] != self.model_kwargs:
            raise ValueError(f'{path}: model hyperparameters differ: {c["model_kwargs"]} vs {self.model_kwargs}')
        mapping = c.get('action_mapping', 'rated')                  # every checkpoint before the option is rated
        env_mapping = getattr(env, 'action_mapping', 'rated') if env is not None else None
        if env_mapping is not None and mapping != env_mapping:
            raise ValueError(f'{path}: checkpoint trained with action_mapping={mapping!r} cannot run in an environment '
                             f'with action_mapping={env_mapping!r}')
        version = action_mapping_version_of(c)                  # soc_feasible without a version: zero_preserving_v1
        env_version = getattr(env, 'action_mapping_version', None) if env is not None else None
        if env is not None and version != env_version:
            raise ValueError(f'{path}: checkpoint trained with action_mapping_version={version!r} cannot run in an '
                             f'environment with action_mapping_version={env_version!r} (action_mapping {mapping!r})')
        self.last_load_provenance_status = (check_checkpoint_provenance(c.get('provenance'), env, path)
                                            if env is not None else 'not_checked')
        for name in ('actor', 'critics', 'target_actor', 'target_critics'):
            getattr(self, name).load_state_dict(c[name])
        optimizers = self.actor_optimizers if hasattr(self, 'actor_optimizers') else [self.actor_optimizer]
        for group, states in ((optimizers, c['actor_optimizers']), (self.critic_optimizers, c['critic_optimizers'])):
            if len(group) != len(states):
                raise ValueError('Optimizer count differs')
            for optimizer, state in zip(group, states):
                optimizer.load_state_dict(state)
        self.lambda_feasibility = float(c['lambda_feasibility'])
        self.feasibility_formulation = formulation
        self.action_mapping = mapping
        self.action_mapping_version = version
        self.feasibility_supervision = bool(c.get('feasibility_supervision', True))
        self.checkpoint_schema = schema
        self.experiment_metadata = c.get('experiment_metadata', {})
        self.target_actor.eval()
        self.target_critics.eval()


class MLPA0Controller(ControllerMixin, MLPMADDPG):
    """A0: original MLP-MADDPG (ten-feature actors incl. shared feeder KPIs, centralised critics)."""

    def __init__(self, lambda_feasibility=None, **kwargs):
        self._init_controller(lambda_feasibility, kwargs)
        self._set_key('mlp_a0')
        MLPMADDPG.__init__(self, **kwargs)


class MLPA1Controller(ControllerMixin, MLPMADDPG):
    """A1: A0 learning equations and centralised critics; actors see only the seven local features (no shared
    feeder KPIs), exactly as experiments/no_shared_kpi (LocalActors slices agent_obs[..., :7])."""

    def __init__(self, lambda_feasibility=None, **kwargs):
        self._init_controller(lambda_feasibility, kwargs)
        self._set_key('mlp_a1')
        MLPMADDPG.__init__(self, **kwargs)
        if self.agent_obs_dim != 10:
            raise ValueError('A1 requires the ten-feature interface')
        self.actor = LocalActors(7, self.num_agents, self.action_dim_per_agent,
                                 tuple(kwargs.get('actor_hidden_dims', (256, 256))), share_actor=False).to(self.device)
        self.target_actor = deepcopy(self.actor).to(self.device)
        self.target_actor.eval()
        self.actor_optimizer = torch.optim.AdamW(self.actor.parameters(), lr=kwargs.get('actor_lr', 1e-4),
                                                 weight_decay=kwargs.get('weight_decay', 0.0))


class MLPA2Controller(ControllerMixin, IndependentAgents):
    """A2: independent (own observation + own action) critics, per-agent actor optimisers, unchanged
    IndependentAgents.update; the original A2 constructor's five-agent restriction is the only part not reused,
    so the same model also runs on the nine-agent IEEE69 feeder."""

    def __init__(self, lambda_feasibility=None, **kwargs):
        self._init_controller(lambda_feasibility, kwargs)
        self._set_key('mlp_a2')
        MLPMADDPG.__init__(self, **kwargs)
        if self.agent_obs_dim != 10 or self.action_dim_per_agent != 1:
            raise ValueError('A2 requires the ten-feature interface and scalar actions')
        self.critics = nn.ModuleList([IndependentCritic() for _ in range(self.num_agents)]).to(self.device)
        self.target_critics = deepcopy(self.critics).to(self.device)
        self.target_critics.eval()
        self.critic_optimizers = [torch.optim.AdamW(c.parameters(), lr=kwargs.get('critic_lr', 1e-3),
                                                    weight_decay=kwargs.get('weight_decay', 0.0)) for c in self.critics]
        self.actor_optimizers = [torch.optim.AdamW(a.parameters(), lr=kwargs.get('actor_lr', 1e-4),
                                                   weight_decay=kwargs.get('weight_decay', 0.0)) for a in self.actor.actors]
        del self.actor_optimizer

    def _actor_forward(self, actor, batch):
        obs = self._to_device(batch['agent_obs'])       # A2's update calls each per-agent actor separately
        return torch.cat([actor.actors[i](obs[:, i]) for i in range(self.num_agents)], dim=1)


class MLPA3Controller(ControllerMixin, IndependentLocalAgents):
    """A3: fully independent DTDE learner (seven local features, local own-action critics, individual rewards from
    environments.local_reward_env)."""

    def __init__(self, lambda_feasibility=None, **kwargs):
        self._init_controller(lambda_feasibility, kwargs)
        self._set_key('mlp_a3')
        IndependentLocalAgents.__init__(self, **kwargs)

    def _actor_forward(self, actor, batch):
        obs = self._to_device(batch['agent_obs'])[..., :LOCAL_FEATURES]   # A3's update calls each local actor separately
        return torch.cat([actor.actors[i](obs[:, i]) for i in range(self.num_agents)], dim=1)


_GRAPH_CLASS = []


def graph_class():
    """GraphController, defined on first use: models.maddpg needs PyTorch Geometric, which the MLP controllers do not."""
    if not _GRAPH_CLASS:
        from models.maddpg import GraphAwareMADDPG

        class GraphController(ControllerMixin, GraphAwareMADDPG):
            """GCN / GAT / TAGConv graph-aware MADDPG: original graph encoders, actors and critics."""

            def __init__(self, lambda_feasibility=None, **kwargs):
                self._init_controller(lambda_feasibility, kwargs)
                self._set_key(str(kwargs.get('gnn_type', 'gcn')))
                GraphAwareMADDPG.__init__(self, **kwargs)

            def _actor_forward(self, actor, batch):
                return actor(x=self._to_device(batch['x']),
                             edge_index=torch.as_tensor(batch['edge_index'], dtype=torch.long, device=self.device),
                             edge_attr=self._to_device(batch.get('edge_attr')),
                             agent_obs=self._to_device(batch['agent_obs']))
        _GRAPH_CLASS.append(GraphController)
    return _GRAPH_CLASS[0]


MLP_CLASSES = dict(mlp_a0=MLPA0Controller, mlp_a1=MLPA1Controller, mlp_a2=MLPA2Controller, mlp_a3=MLPA3Controller)


def model_class(controller):
    if controller in GNN_TYPES:
        return graph_class()
    return MLP_CLASSES[controller]


def build_model(controller, lambda_feasibility, **kwargs):
    if controller in GNN_TYPES:
        kwargs['gnn_type'] = GNN_TYPES[controller]
    return model_class(controller)(lambda_feasibility=lambda_feasibility, **kwargs)
