"""Separate episode reset from Bellman terminal masks; reject invalid data."""
import numpy as np


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    a = np.asarray(value)
    return bool(np.isfinite(a).all()) if a.dtype.kind in 'biufc' else True


def learning_valid(obs, action, reward, next_obs, info):
    return (not info.get('episode_failed', False)
            and info.get('transition_valid', True)
            and info.get('observation_valid', True)
            and finite_tree((reward)) and finite_tree(action)
            and finite_tree(obs) and finite_tree(next_obs))
