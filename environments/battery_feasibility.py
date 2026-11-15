"""B3 battery feasibility and terminal-SoC quantities from the AUTHORITATIVE battery model.

Feasible powers come from environments.battery.Battery.get_available_charge_kw / get_available_discharge_kw, the
same limiter Battery.step applies (power rating, SoC window, charge/discharge efficiency, timestep). Nothing here
re-derives battery physics. Normalisation follows the original action mapping of Battery.step:
    a > 0: P = a * max_discharge_kw        a < 0: P = a * max_charge_kw
so the feasible action interval is [l, h] = [-P_ch_avail / P_ch_max, P_dis_avail / P_dis_max] with
-1 <= l <= 0 <= h <= 1.
"""
import numpy as np
import torch

SOC_TOLERANCE = 1e-9


def feasible_powers_kw(env):
    """(charge_available_kw, discharge_available_kw) per agent, in env.batteries (= env.dess_buses) order."""
    charge = np.array([max(0.0, float(b.get_available_charge_kw())) for b in env.batteries])
    discharge = np.array([max(0.0, float(b.get_available_discharge_kw())) for b in env.batteries])
    return charge, discharge


def normalized_bounds(env):
    """Pre-action feasible normalised action bounds (lower, upper), float32 arrays of shape [num_agents]."""
    charge, discharge = feasible_powers_kw(env)
    ch_max = np.array([b.config.max_charge_kw for b in env.batteries], dtype=float)
    dis_max = np.array([b.config.max_discharge_kw for b in env.batteries], dtype=float)
    lower = -np.divide(charge, ch_max, out=np.zeros_like(charge), where=ch_max > 0)
    upper = np.divide(discharge, dis_max, out=np.zeros_like(discharge), where=dis_max > 0)
    lower, upper = np.clip(lower, -1.0, 0.0), np.clip(upper, 0.0, 1.0)
    return lower.astype(np.float32), upper.astype(np.float32)


SOC_FEASIBLE_MAPPING_VERSION = 'affine_v2'
"""Version of soc_feasible_action recorded in environments, manifests and checkpoints. Checkpoints written before the
version existed used the zero-preserving mapping ('zero_preserving_v1', configs.experiments)."""


def affine_soc_feasible_power_kw(a, charge_available_kw, discharge_available_kw):
    """Affine SOC-feasible power for a policy action a in [-1, 1] (positive = discharge, kW):

        P = -P_ch_avail + (a + 1) / 2 * (P_ch_avail + P_dis_avail)

    a = -1 -> -P_ch_avail (maximum available charging), a = +1 -> +P_dis_avail (maximum available discharging), linear
    in between; a = 0 is the MIDPOINT of the available interval (0 kW only when P_ch_avail = P_dis_avail)."""
    ch, dis = max(0.0, float(charge_available_kw)), max(0.0, float(discharge_available_kw))
    a = min(1.0, max(-1.0, float(a)))
    return -ch + (a + 1.0) / 2.0 * (ch + dis)


def soc_feasible_action(action, batteries):
    """SOC-feasible execution action a' for the policy action a (soc_feasible mapping, AFFINE, version
    SOC_FEASIBLE_MAPPING_VERSION), from the pre-action SoC.

    P_ch_avail / P_dis_avail come from Battery.get_available_charge_kw / get_available_discharge_kw (the same limiter
    Battery.step applies; no battery physics is re-derived here). The policy action is mapped affinely onto the whole
    available interval [-P_ch_avail, +P_dis_avail] (affine_soc_feasible_power_kw), so every a in [-1, 1] is executable and
    there is no dead zone at a SoC limit (at s_min, a = +1 means 0 kW and a < +1 charges; at s_max, a = -1 means 0 kW
    and a > -1 discharges). Battery.step executes a' as P = a' * P_rated_dis (a' >= 0) or a' * P_rated_ch (a' < 0), so
        P > 0:  a' = P / P_rated_dis        P < 0:  a' = P / P_rated_ch        P = 0:  a' = 0
    (direction-specific ratings). a' is float32 (the env's action dtype) rounded toward zero so that
    |a'| * P_rated <= P_avail of that direction holds exactly; the battery limiter then has nothing SoC-related left to
    remove. Returns a NEW float32 array; the input is never modified.

    The zero-action reference never passes through this mapping (evaluation.zero builds the original DESSEnv, which
    executes 0 kW), and the network backup's candidates are rated-normalised powers, so its zero candidate is 0 kW.
    """
    a = np.asarray(action, dtype=np.float32).reshape(-1)
    if a.shape != (len(batteries),):
        raise ValueError(f'action shape {a.shape} does not match {len(batteries)} batteries')
    if not np.isfinite(a).all():
        raise ValueError('nonfinite policy action')
    out = np.zeros(a.shape, dtype=np.float32)
    for i, (x, battery) in enumerate(zip(a.tolist(), batteries)):
        charge = max(0.0, float(battery.get_available_charge_kw()))
        discharge = max(0.0, float(battery.get_available_discharge_kw()))
        power = affine_soc_feasible_power_kw(x, charge, discharge)
        if power > 0:
            available, rated = discharge, float(battery.config.max_discharge_kw)
        elif power < 0:
            available, rated = charge, float(battery.config.max_charge_kw)
        else:
            continue
        if rated <= 0 or available <= 0:
            continue
        y = np.float32(power / rated)
        while y != 0 and abs(float(y)) * rated > available:          # float32 safety: never exceed P_avail
            y = np.nextafter(y, np.float32(0.0), dtype=np.float32)
        out[i] = np.float32(np.clip(y, -1.0, 1.0))
    return out


def check_bounds(lower, upper, num_agents):
    lower, upper = np.asarray(lower, dtype=np.float32), np.asarray(upper, dtype=np.float32)
    if lower.shape != (num_agents,) or upper.shape != (num_agents,):
        raise ValueError(f'Feasible bounds must have shape ({num_agents},), got {lower.shape} / {upper.shape}')
    if not (np.isfinite(lower).all() and np.isfinite(upper).all()):
        raise ValueError('Nonfinite feasible action bounds')
    if (lower < -1 - 1e-6).any() or (lower > 1e-6).any() or (upper < -1e-6).any() or (upper > 1 + 1e-6).any():
        raise ValueError(f'Feasible bounds violate -1 <= l <= 0 <= h <= 1: {lower} / {upper}')
    return lower, upper


def violation(actions, lower, upper):
    """a - clip(a, l, h): positive = excess discharge request, negative = excess charge request."""
    if torch.is_tensor(actions):
        return actions - torch.maximum(torch.minimum(actions, upper), lower)
    actions = np.asarray(actions, dtype=float)
    return actions - np.clip(actions, lower, upper)


def feasibility_loss(actions, lower, upper):
    """ACTION-space mean of (a - clip(a, l, h))^2: the historical post-tanh B3 loss, kept as a diagnostic only.

    The final B3 trains on the pre-tanh loss in training.feasibility_supervision.
    """
    v = violation(actions, lower, upper)
    return (v ** 2).mean() if torch.is_tensor(v) else float(np.mean(v ** 2))


def terminal_shortfall(soc_start, soc_end, soc_min, tol=SOC_TOLERANCE):
    """d_i = max(0, s_i0 - s_iT) / (s_i0 - s_i_min); 0 when s_i0 is at (or within tol of) the minimum.

    With s_i0 <= s_i_min + tol no valid terminal state lies below the reference (SoC never goes below s_min),
    so the one-sided deficit is defined as 0 instead of dividing by zero.
    """
    s0, sT, smin = (np.asarray(v, dtype=float) for v in (soc_start, soc_end, soc_min))
    room = s0 - smin
    return np.where(room > tol, np.maximum(0.0, s0 - sT) / np.where(room > tol, room, 1.0), 0.0)


def terminal_penalty(lambda_terminal, shortfall):
    """Team-wide normalised penalty -lambda_T * (1/N) * sum_i d_i^2 (one scalar, added once to every agent)."""
    shortfall = np.asarray(shortfall, dtype=float)
    return -float(lambda_terminal) * float(np.mean(shortfall ** 2))
