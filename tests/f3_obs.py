"""F3 test helper: compare actor observations with pre-F3 code, allowing only the documented change.

Pre-F3 code divided the feeder-wide curtailment by the per-bus PV scale (max_pv_kw); F3 divides
it by the feeder curtailment scale (max_curtailment_kw). Every other column must be identical, and
the curtailment column must correspond to the same physical curtailment.
"""
import numpy as np

from utility.obs_interface import AGENT_OBS_FEATURES

CURTAILMENT_COLUMN = AGENT_OBS_FEATURES.index("feeder_curtailment_kw")


def assert_agent_obs_equal_except_f3_curtailment(new_obs, old_obs, new_env, old_env):
    new_obs, old_obs = np.asarray(new_obs), np.asarray(old_obs)
    keep = [c for c in range(new_obs.shape[1]) if c != CURTAILMENT_COLUMN]
    np.testing.assert_array_equal(new_obs[:, keep], old_obs[:, keep])
    np.testing.assert_allclose(new_obs[:, CURTAILMENT_COLUMN] * new_env.normalizer.max_curtailment_kw,
                               old_obs[:, CURTAILMENT_COLUMN] * old_env.normalizer.max_pv_kw, rtol=1e-6, atol=1e-6)
