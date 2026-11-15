"""HISTORICAL COMPATIBILITY SHIM: the B3/B4 trainer factory. Maintained implementation: training.controller_trainer.

b3_trainer_class keeps the historical defaults (rated mapping, SoC-feasibility supervision on) and historical names
(b3_* labels, 'b3-checkpoint/2'). Maintained modules never import this module.
"""
from training.controller_trainer import *                                 # noqa: F401,F403 (historical re-export)
from training.controller_trainer import BALANCE_FIELDS, DAY_FIELDS, UPDATE_FIELDS, trainer_class  # noqa: F401


def b3_trainer_class(controller, lambda_feasibility, action_mapping='rated', feasibility_supervision=True):
    return trainer_class(controller, lambda_feasibility, action_mapping, feasibility_supervision, legacy_names=True)
