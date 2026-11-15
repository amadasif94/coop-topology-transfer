"""HISTORICAL COMPATIBILITY SHIM: the B3/B4 controller names. Maintained implementation: models.controllers.

Every class and build_model here default to legacy_names=True, so historical entry points and tests keep writing
'b3-checkpoint/2' checkpoints labelled b3_mlp_a0 ... b3_tagconv, exactly as before. Maintained modules never import
this module.
"""
from models.controllers import *                                         # noqa: F401,F403 (historical re-export)
from models.controllers import (GNN_TYPES, LEGACY_SCHEMA as CHECKPOINT_SCHEMA, MLPA0Controller, MLPA1Controller,  # noqa
                                MLPA2Controller, MLPA3Controller, ControllerMixin as FeasibilityAgentMixin,
                                graph_class as _graph_class)
from configs.experiments import LEGACY_LABELS as CONTROLLERS                # noqa: F401  key -> b3_ label


def _legacy(cls, name):
    return type(name, (cls,), dict(legacy_names_default=True, __doc__=f'Historical name of {cls.__name__}.',
                                   __module__=__name__))


B3MLPA0 = _legacy(MLPA0Controller, 'B3MLPA0')
B3MLPA1 = _legacy(MLPA1Controller, 'B3MLPA1')
B3MLPA2 = _legacy(MLPA2Controller, 'B3MLPA2')
B3MLPA3 = _legacy(MLPA3Controller, 'B3MLPA3')
_LEGACY_GRAPH = []


def graph_class():
    if not _LEGACY_GRAPH:
        _LEGACY_GRAPH.append(_legacy(_graph_class(), 'B3Graph'))
    return _LEGACY_GRAPH[0]


def model_class(controller):
    if controller in GNN_TYPES:
        return graph_class()
    return dict(mlp_a0=B3MLPA0, mlp_a1=B3MLPA1, mlp_a2=B3MLPA2, mlp_a3=B3MLPA3)[controller]


def build_model(controller, lambda_feasibility, **kwargs):
    if controller in GNN_TYPES:
        kwargs['gnn_type'] = GNN_TYPES[controller]
    return model_class(controller)(lambda_feasibility=lambda_feasibility, **kwargs)
