"""Actor observation interface: feature order and checkpoint compatibility.

Single source of truth for the per-agent observation vector ``agent_obs``.
The global ``grid_stress`` scalar (max voltage violation + max line-current
violation) was removed from ``agent_obs`` (F5); it remains in the KPI reward.
Checkpoints trained with the previous 11-feature observation are incompatible.
"""

AGENT_OBS_FEATURES = (
    "time",                      # local clock: step / (episode_length - 1)
    "price",                     # public energy price
    "soc",                       # own battery
    "voltage_pu",                # own bus, pre-action PF (column read by tests; DOE reads the PF directly)
    "load_kw",                   # own bus
    "pv_kw",                     # own bus
    "previous_action",           # own executed normalized action
    "feeder_grid_import_kw",     # network-wide aggregate
    "feeder_voltage_deviation",  # network-wide aggregate: sum |V - V_ref| over all buses
    "feeder_curtailment_kw",     # network-wide aggregate
)
AGENT_OBS_DIM = len(AGENT_OBS_FEATURES)
VOLTAGE_COLUMN = AGENT_OBS_FEATURES.index("voltage_pu")
LEGACY_AGENT_OBS_DIM = 11  # pre-F5 layout: AGENT_OBS_FEATURES + ("grid_stress",)


def check_checkpoint_obs_dim(checkpoint, expected_dim, inferred_dim=None, path=""):
    """Raise a clear error if a checkpoint's agent_obs_dim differs from the model's.

    ``inferred_dim`` is derived from the first actor layer for checkpoints saved
    before agent_obs_dim was stored. Unknown dimension is refused, not guessed.
    """
    stored = checkpoint.get("agent_obs_dim", None)
    found = int(stored) if stored is not None else inferred_dim
    if found is None:
        raise ValueError(
            f"Checkpoint {path} does not record agent_obs_dim and it could not be inferred; "
            f"refusing to load into a model expecting {expected_dim} agent observation features.")
    if int(found) != int(expected_dim):
        hint = (" It was trained with the pre-F5 11-feature observation that included grid_stress;"
                " retrain with the current 10-feature observation." if int(found) == LEGACY_AGENT_OBS_DIM else "")
        raise ValueError(
            f"Checkpoint {path} has agent_obs_dim={found}, but this model expects "
            f"agent_obs_dim={expected_dim} (features: {', '.join(AGENT_OBS_FEATURES)}).{hint}")
    return int(found)


def first_layer_in_features(state_dict, suffix="net.0.weight"):
    """Input width of the first actor MLP layer, or None if not found."""
    for key, value in state_dict.items():
        if key.endswith(suffix) and getattr(value, "dim", lambda: 0)() == 2:
            return int(value.shape[1])
    return None
