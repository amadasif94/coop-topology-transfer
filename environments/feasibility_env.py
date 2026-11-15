"""HISTORICAL COMPATIBILITY SHIM: the B3/B4 environment names. Maintained implementation: environments.marl_env.

    B3DESSEnv   = environments.marl_env.MarlDESSEnv
    make_b3_env = environments.marl_env.make_marl_env
Maintained modules never import this module.
"""
from environments.marl_env import *                                      # noqa: F401,F403 (historical re-export)
from environments.marl_env import MarlDESSEnv as B3DESSEnv, make_marl_env as make_b3_env  # noqa: F401
