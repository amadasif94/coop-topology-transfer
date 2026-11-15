"""New checkpoint format: explicit ablation identity, no A0 compatibility guessing."""
from pathlib import Path
import torch
from utility.provenance import check_checkpoint_provenance
from experiments.no_shared_kpi.protocol import feature_spec, source_identity, sha, write_json


class AblationCheckpoint:
    def save(self, path, provenance=None):
        path = Path(path)
        record = dict(schema='ieee33-mlp-ablation/1', specification=feature_spec(self.variant, self.num_agents),
            provenance=provenance, experiment_metadata=getattr(self, 'experiment_metadata', {}),
            model_kwargs=self.model_kwargs, sources=source_identity(),
            actor=self.actor.state_dict(), critics=self.critics.state_dict(),
            target_actor=self.target_actor.state_dict(), target_critics=self.target_critics.state_dict(),
            actor_optimizers=[o.state_dict() for o in self._actor_optimizers()],
            critic_optimizers=[o.state_dict() for o in self.critic_optimizers])
        # Exclusive creation prevents an accidental checkpoint overwrite.
        with path.open('xb') as f:
            torch.save(record, f)
        write_json(path.with_suffix('.identity.json'), dict(sha256=sha(path), specification=record['specification'],
            experiment_metadata=record['experiment_metadata'], sources=record['sources']))

    def _actor_optimizers(self):
        return self.actor_optimizers if hasattr(self, 'actor_optimizers') else [self.actor_optimizer]

    def load(self, path, env=None):
        c = torch.load(path, map_location=self.device, weights_only=False)
        if c.get('schema') != 'ieee33-mlp-ablation/1' or c.get('specification') != feature_spec(self.variant, self.num_agents):
            raise ValueError('Checkpoint ablation/feature specification mismatch')
        if c['model_kwargs'] != self.model_kwargs:
            raise ValueError('Checkpoint model hyperparameters differ')
        if c.get('sources') != source_identity():
            raise ValueError('Ablation/original MLP source changed since checkpoint creation; review before evaluation')
        if env is not None:
            check_checkpoint_provenance(c['provenance'], env, path)
        for name in ('actor', 'critics', 'target_actor', 'target_critics'):
            getattr(self, name).load_state_dict(c[name])
        for group, states in ((self._actor_optimizers(), c['actor_optimizers']), (self.critic_optimizers, c['critic_optimizers'])):
            if len(group) != len(states): raise ValueError('Optimizer count differs')
            for optimizer, state in zip(group, states): optimizer.load_state_dict(state)
        self.experiment_metadata = c['experiment_metadata']
        self.target_actor.eval()
        self.target_critics.eval()


def checked_kwargs(kwargs):
    from utility.obs_interface import AGENT_OBS_FEATURES
    from experiments.no_shared_kpi.protocol import FULL_FEATURES
    if tuple(AGENT_OBS_FEATURES) != FULL_FEATURES or kwargs['agent_obs_dim'] != 10:
        raise ValueError('Expected verified F5 ten-feature environment')
    if kwargs['num_agents'] != 5:
        raise ValueError('These controlled IEEE33 experiments require five agents')
    if kwargs.get('share_actor', False): raise ValueError('Ablations require separate actors')
    if kwargs.get('action_dim_per_agent', 1) != 1: raise ValueError('Expected scalar battery action')
    if tuple(kwargs.get('actor_hidden_dims', (256,256))) != (256,256) or tuple(kwargs.get('critic_hidden_dims', (256,256))) != (256,256):
        raise ValueError('Preserve A0 hidden sizes (256,256)')
    # Device-independent metadata used for exact reload checks.
    return {k: list(v) if isinstance(v, tuple) else v for k,v in kwargs.items() if k != 'device'}
