"""One seeded IEEE69 random-action episode with original-request diagnostics.

This is an environment smoke test, not a trained-policy evaluation. Network
correction causes are read from the original candidate's PF diagnostics; final
voltages and currents refer to the executed candidate in info, not next_obs.
"""

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from configs.ieee69_config import IEEE69_CONFIG
from environments.dess_env import DESSEnv


FEASIBILITY_TOLERANCE = 1e-9


def correction_cause(info, constraint_cfg):
    """Classify one original candidate using the environment's enabled checks."""
    if not info["requested_pf_converged"]:
        return "PF nonconvergence"

    voltage = (
        constraint_cfg.get("enforce_voltage_limits", True)
        and float(info["requested_max_voltage_violation"]) > FEASIBILITY_TOLERANCE
    )
    current = (
        constraint_cfg.get("enforce_line_current_limits", True)
        and float(info["requested_max_line_current_violation"]) > FEASIBILITY_TOLERANCE
    )
    if voltage and current:
        return "voltage and line current"
    if voltage:
        return "voltage only"
    if current:
        return "line current only"
    return "unclassified: inspect diagnostics"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    env = DESSEnv(
        config=IEEE69_CONFIG,
        mode="train",
        seed=args.seed,
    )

    # Environment reset alone does not seed action_space.sample().
    env.action_space.seed(args.seed)
    obs, info = env.reset(seed=args.seed)

    print("===================================")
    print("IEEE69 FULL 96-STEP ROLLOUT TEST")
    print("===================================")
    print("Start date_time:", info["date_time"])
    print("Random-action seed:", args.seed)
    print("Episode length:", env.episode_length)
    print("Delta t hours:", env.delta_t_hours)
    print("Initial feasible:", info["feasible"])
    print("Initial min voltage:", info["min_voltage_pu"])
    print("Initial max voltage:", info["max_voltage_pu"])
    print("Initial max line current:", info["max_line_current_pu"])
    print("Initial PF converged:", info["converged"])
    print("Voltage limits:", env.config["network"]["v_min"], env.config["network"]["v_max"])
    print("Current violations use branch-specific limits, not a 1.0 p.u. threshold.")
    print()

    if env.episode_length != 96:
        raise AssertionError(f"Expected 96 steps, got {env.episode_length}.")
    if not info["converged"] or not info["feasible"]:
        raise AssertionError("Initial reset state is nonconverged or infeasible.")

    causes = Counter()

    total_reward = np.zeros(env.num_agents, dtype=np.float64)

    raw_infeasible_count = 0
    minor_corrected_count = 0
    major_infeasible_count = 0
    fallback_to_zero_count = 0
    accepted_infeasible_count = 0

    correction_iterations = []

    min_voltage_seen = float(info["min_voltage_pu"])
    max_voltage_seen = float(info["max_voltage_pu"])
    max_line_current_seen = float(info["max_line_current_pu"])

    final_info = info

    for step in range(env.episode_length):
        action = env.action_space.sample()

        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward

        # Require these fields: older environments must not silently log zeros.
        raw_infeasible = bool(info["raw_infeasible_requested"])
        minor_corrected = bool(info["minor_corrected_action"])
        major_infeasible = bool(info["major_infeasible_action"])
        fallback_to_zero = bool(info["correction_fallback_to_zero"])
        iters = int(info["correction_iterations"])

        if raw_infeasible:
            raw_infeasible_count += 1
            correction_iterations.append(iters)
            cause = correction_cause(info, env.config["constraints"])
            causes[cause] += 1

            # Print before assertions so failed corrections retain diagnostics.
            print(
                f"\nCORRECTION | step={step:03d} | date={info['date_time']} "
                f"| original_cause={cause}"
            )
            print(
                f"  ORIGINAL: converged={info['requested_pf_converged']} "
                f"| feasible={info['requested_feasible']} "
                f"| minV={info['requested_min_voltage_pu']:.8f} "
                f"| maxV={info['requested_max_voltage_pu']:.8f} "
                f"| voltage_violation={info['requested_max_voltage_violation']:.10g} "
                f"| current_violation={info['requested_max_line_current_violation']:.10g}"
            )
            if not info["requested_pf_converged"]:
                print("  Original PF failed: its voltage/current violations are not reliable diagnoses.")
            print(
                f"  CORRECTED: iterations={iters} | minor={minor_corrected} "
                f"| major={major_infeasible} | fallback_zero={fallback_to_zero} "
                f"| converged={info['converged']} | feasible={info['feasible']} "
                f"| minV={info['min_voltage_pu']:.8f} "
                f"| maxV={info['max_voltage_pu']:.8f} "
                f"| voltage_violation={info['max_voltage_violation']:.10g} "
                f"| current_violation={info['max_line_current_violation']:.10g}"
            )
            print("  Requested normalized actions:", info["requested_action"])
            print("  Accepted normalized actions: ", info["accepted_action"])
            print("  Battery buses (one-based):   ", [int(b) + 1 for b in env.dess_buses])
            print("  Final actual battery kW:     ", np.asarray(info["dess_power_kw"])[env.dess_buses])
            print()

        if minor_corrected:
            minor_corrected_count += 1

        if major_infeasible:
            major_infeasible_count += 1

        if fallback_to_zero:
            fallback_to_zero_count += 1

        min_voltage_seen = min(min_voltage_seen, float(info["min_voltage_pu"]))
        max_voltage_seen = max(max_voltage_seen, float(info["max_voltage_pu"]))
        max_line_current_seen = max(
            max_line_current_seen,
            float(info["max_line_current_pu"]),
        )

        if not info["feasible"]:
            accepted_infeasible_count += 1
            raise AssertionError(
                f"Accepted transition became infeasible at step {step}. "
                f"date_time={info['date_time']}, "
                f"minV={info['min_voltage_pu']}, "
                f"maxV={info['max_voltage_pu']}, "
                f"maxI={info['max_line_current_pu']}"
            )

        if not info["converged"]:
            raise AssertionError(
                f"Power flow did not converge at step {step}. "
                f"date_time={info['date_time']}"
            )

        if not np.all(np.isfinite(reward)):
            raise AssertionError(f"Reward has NaN/Inf at step {step}.")

        if not np.all(np.isfinite(obs["x"])):
            raise AssertionError(f"Observation x has NaN/Inf at step {step}.")

        if not np.all(np.isfinite(obs["agent_obs"])):
            raise AssertionError(f"agent_obs has NaN/Inf at step {step}.")

        final_info = info

        print(
            f"step={step:03d} | "
            f"date={info['date_time']} | "
            f"feasible={info['feasible']} | "
            f"raw={raw_infeasible} | "
            f"minor={minor_corrected} | "
            f"major={major_infeasible} | "
            f"iters={iters} | "
            f"fallback={fallback_to_zero} | "
            f"minV={info['min_voltage_pu']:.4f} | "
            f"maxV={info['max_voltage_pu']:.4f} | "
            f"maxI={info['max_line_current_pu']:.4f} | "
            f"reward_mean={np.mean(reward):.4f}"
        )

        if terminated or truncated:
            print()
            print("Episode ended.")
            print("terminated:", terminated)
            print("truncated:", truncated)
            break

    if step + 1 != 96 or not truncated or terminated:
        raise AssertionError(
            f"Expected 96-step time-limit completion; got steps={step + 1}, "
            f"terminated={terminated}, truncated={truncated}."
        )

    print()
    print("===================================")
    print("IEEE69 ROLLOUT SUMMARY")
    print("===================================")
    print("Final date_time:", final_info["date_time"])
    print("Total reward per agent:", total_reward)
    print("Mean total reward:", np.mean(total_reward))

    print()
    print("Correction diagnostics:")
    print("Raw infeasible requested actions:", raw_infeasible_count)
    print("Minor corrected actions:", minor_corrected_count)
    print("Major infeasible actions:", major_infeasible_count)
    print("Fallback-to-zero corrections:", fallback_to_zero_count)
    print("Accepted infeasible transitions:", accepted_infeasible_count)
    print()
    print("Original-request causes (mutually exclusive; counts sum to raw count):")
    for cause in (
        "voltage only", "line current only", "voltage and line current",
        "PF nonconvergence", "unclassified: inspect diagnostics",
    ):
        print(f"  {cause}: {causes[cause]}")
    print("These classify the original candidate, not each intermediate correction.")
    print("SOC/power limiting is separate and occurs before original-candidate PF.")

    if len(correction_iterations) > 0:
        print("Correction iterations list:", correction_iterations)
        print("Mean correction iterations:", float(np.mean(correction_iterations)))
        print("Max correction iterations:", int(np.max(correction_iterations)))
    else:
        print("Correction iterations list: []")
        print("Mean correction iterations: 0")
        print("Max correction iterations: 0")

    print()
    print("Physical operation:")
    print("Minimum voltage seen:", min_voltage_seen)
    print("Maximum voltage seen:", max_voltage_seen)
    print("Maximum line current seen:", max_line_current_seen)
    print("Final feasible:", final_info["feasible"])
    print("Final converged:", final_info["converged"])
    print("===================================")
    print("IEEE69 FULL ROLLOUT TEST PASSED")
    print("===================================")


if __name__ == "__main__":
    main()
