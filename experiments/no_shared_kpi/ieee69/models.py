"""Nine-agent adapters; original A1/A2 forward/update equations are inherited."""
from .bindings import bind
from .protocol import feature_spec, source_identity
from experiments.no_shared_kpi.model_no_shared_kpi import NoSharedKPI
from experiments.no_centralized_critic.model_independent_critic import IndependentAgents
from experiments.no_shared_kpi.model_support import AblationCheckpoint
import hashlib
import json
from pathlib import Path
import re


_MODEL_PATH = 'experiments/no_shared_kpi/ieee69/models.py'
_OLD_MODEL_SHA = 'c103e8ee1af9a8d996d69330d31f4b0bac5b62f0abbe0f7c3ce790cf514725ce'
_UTILITY_PATH = 'experiments/no_shared_kpi/ieee33_eval_compat.py'
_UTILITY_SHA = '2f79adfa81435aa189d09dfafa7a7605613bda2ce3619c0fbf2347c748442a97'
_PRIOR_MODEL_SHA = 'ca85392225c5ada1ec6cfb07e461f664f70b90415cd41d675e4e29e3b111f90d'
# Exact evaluation-code migration pairs (historical, installed release).
# Accept whole releases only, never a mixture of intermediate file versions.
_EVAL_FILES = {
    'experiments/no_shared_kpi/ieee69/protocol.py': (
        '8eda9ed93454a06249495025a4a1dd26ae7bb86f4562cff050891b1df593228c',
        '937d37a2af5302a4cf8d965758a946a6df49736cd8d048261c9721b3839126ed'),
    'experiments/no_shared_kpi/ieee69/final_evaluation.py': (
        '957ede0411138bb4212ffc23e9e08eec72b24810f0eaa04b2f3a26627f3d3ef4',
        '13badcf31fd953adf61f551c7a138a9ee09caaf4bd21b1da0d997be25b0ff729'),
}
# Release digest of this entire file with only this digest's 64 bytes zeroed.
# Avoids a self-hash fixed point; no code region is excluded from verification.
_RELEASE_SHA = '9b27be2b22d3140b6f06501b8222ded783d407a8413a111a24c101718840b434'


def _digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _evaluation_inventory(recorded, current):
    raw = Path(__file__).read_bytes()
    canonical, count = re.subn(rb"(?m)^_RELEASE_SHA = '[0-9a-f]{64}'$",
                               b"_RELEASE_SHA = '" + b'0' * 64 + b"'", raw)
    if count != 1 or hashlib.sha256(canonical).hexdigest() != _RELEASE_SHA:
        raise ValueError('Unapproved IEEE69 evaluation-loader release')
    installed = hashlib.sha256(raw).hexdigest()
    if current.get(_MODEL_PATH) != installed or current.get(_UTILITY_PATH) != _UTILITY_SHA:
        raise ValueError('Unapproved loader or IEEE33 utility bytes')
    if any(current.get(name)!=new for name,(_,new) in _EVAL_FILES.items()):
        raise ValueError('Unapproved installed evaluation-code release')
    if not isinstance(recorded, dict) or not recorded:
        raise ValueError('Missing checkpoint source inventory')
    expected = dict(current)
    if recorded.get(_MODEL_PATH) in (_OLD_MODEL_SHA, _PRIOR_MODEL_SHA):
        expected[_MODEL_PATH] = recorded[_MODEL_PATH]
        for name,(old,_) in _EVAL_FILES.items():
            expected[name] = old
        if recorded[_MODEL_PATH] == _OLD_MODEL_SHA and _UTILITY_PATH not in recorded:
            del expected[_UTILITY_PATH]
    # Otherwise only the exact current inventory is accepted. Unknown loader
    # revisions and post-patch inventories missing the utility are rejected.
    if recorded != expected:
        raise ValueError('Unapproved IEEE69 checkpoint source difference')
    return dict(recorded)


def _evaluation_load(self, path, env, strict_load):
    sidecar = json.loads(Path(path).with_suffix('.identity.json').read_text(encoding='utf-8-sig'))
    if sidecar.get('sha256') != _digest_file(path):
        raise ValueError('Checkpoint SHA256 differs from identity sidecar')
    recorded = _evaluation_inventory(sidecar.get('sources'), source_identity())
    # Private binding, no module monkeypatch. The original loader also compares
    # the embedded inventory against the verified sidecar inventory, and keeps
    # all model/state-dict/hyperparameter/environment checks unchanged.
    load = bind(strict_load, dict(strict_load.__globals__, source_identity=lambda: dict(recorded)))
    return load(self, path, env)


def checked_kwargs(kwargs):
    from utility.obs_interface import AGENT_OBS_FEATURES
    from .protocol import FULL_FEATURES
    if tuple(AGENT_OBS_FEATURES) != FULL_FEATURES or kwargs['agent_obs_dim'] != 10 or kwargs['num_agents'] != 9:
        raise ValueError('IEEE69 requires nine agents with full ten-feature replay observations')
    if kwargs.get('share_actor', False) or kwargs.get('action_dim_per_agent', 1) != 1:
        raise ValueError('Require separate scalar-action actors')
    for key in ('actor_hidden_dims', 'critic_hidden_dims'):
        if tuple(kwargs.get(key, (256, 256))) != (256, 256):
            raise ValueError('Preserve verified A0 hidden dimensions')
    return {k: list(v) if isinstance(v, tuple) else v for k, v in kwargs.items() if k != 'device'}


def adapt(base):
    constructor = bind(base.__init__, dict(base.__init__.__globals__, checked_kwargs=checked_kwargs))
    # Legacy schema string is retained as the shared serialization format identifier;
    # specification.num_agents=9 plus source identity reject IEEE33 checkpoints.
    methods = {name: bind(getattr(AblationCheckpoint, name),
                         dict(getattr(AblationCheckpoint, name).__globals__,
                              feature_spec=feature_spec, source_identity=source_identity))
               for name in ('save', 'load')}
    strict_load = methods['load']

    def load(self, path, env=None):
        if env is not None and getattr(env, 'mode', None) in ('val', 'test'):
            return _evaluation_load(self, path, env, strict_load)
        return strict_load(self, path, env)

    methods['load'] = load
    return type(base.__name__+'IEEE69', (base,), dict(__init__=constructor, **methods))


NoSharedKPI69 = adapt(NoSharedKPI)
IndependentAgents69 = adapt(IndependentAgents)


def model_class(variant):
    return {'no_shared_kpi': NoSharedKPI69, 'no_centralized_critic': IndependentAgents69}[variant]
