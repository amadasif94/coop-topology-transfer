"""
Smoke test for IEEE69 topology generalization pipeline.

Checks:
    1. tp_configs_ieee69.py
    2. topology_utils_ieee69.py
    3. run_tp_eval_ieee69.py compatibility through DESSEnv

Run:
    PYTHONPATH=. python tests/test_ieee69_topology_pipeline.py
"""

import importlib
import traceback

import numpy as np

from environments.dess_env import DESSEnv
from experiments.topology_generalization.tp_configs_ieee69 import (
    list_topology_cases,
    get_topology_case,
)
from experiments.topology_generalization.topology_utils_ieee69 import (
    build_topology_config,
)


def check_tp_configs():
    print("\n====================================")
    print("CHECK 1: tp_configs_ieee69.py")
    print("====================================")

    cases = list_topology_cases(include_baseline=True)

    expected = ["TP1", "TP2", "TP3", "TP4", "TP5", "TP6", "TP7", "TP8", "TP9"]

    assert cases == expected, f"Expected {expected}, got {cases}"

    for case_name in cases:
        case = get_topology_case(case_name)

        assert "description" in case
        assert "open_edges" in case
        assert "close_edges" in case

        assert isinstance(case["open_edges"], list)
        assert isinstance(case["close_edges"], list)

        for edge in case["open_edges"] + case["close_edges"]:
            assert len(edge) == 2
            assert edge[0] != edge[1]
            assert 0 <= edge[0] < 69
            assert 0 <= edge[1] < 69

        print(f"{case_name}: OK")

    print("tp_configs_ieee69.py PASSED")


def check_topology_utils():
    print("\n====================================")
    print("CHECK 2: topology_utils_ieee69.py")
    print("====================================")

    cases = list_topology_cases(include_baseline=True)

    for case_name in cases:
        cfg = build_topology_config(
            case_name=case_name,
            validate=True,
        )

        assert cfg["network"]["num_buses"] == 69
        assert cfg["case_name"] == f"ieee69_{case_name}"

        topo = cfg["topology_case"]
        assert topo["case_name"] == case_name
        assert "generated_dir" in topo

        print(
            f"{case_name}: OK | "
            f"open={topo['open_edges']} | "
            f"close={topo['close_edges']}"
        )

    print("topology_utils_ieee69.py PASSED")


def check_env_build_and_rollout():
    print("\n====================================")
    print("CHECK 3: DESSEnv build + one-step rollout")
    print("====================================")

    cases = list_topology_cases(include_baseline=True)

    for case_name in cases:
        cfg = build_topology_config(
            case_name=case_name,
            validate=True,
        )

        env = DESSEnv(
            config=cfg,
            mode="val",  # F2: structural check; the test split needs --final
            seed=123,
        )

        assert env.num_buses == 69
        assert env.num_agents == 9
        assert len(env.dess_buses) == 9
        assert len(env.grid.in_service_line_df) == 68

        obs, info = env.reset(seed=123)

        assert obs["x"].shape[0] == 69
        assert obs["agent_obs"].shape[0] == 9
        assert obs["edge_index"].shape[0] == 2

        zero_action = np.zeros(env.num_agents, dtype=np.float32)

        next_obs, reward, terminated, truncated, step_info = env.step(zero_action)

        assert next_obs["x"].shape[0] == 69
        assert next_obs["agent_obs"].shape[0] == 9
        assert len(reward) == 9

        assert "feasible" in step_info
        assert "converged" in step_info
        assert "kpis" in step_info

        assert bool(step_info["converged"]) is True

        print(
            f"{case_name}: OK | "
            f"edge_index={env.grid.edge_index.shape} | "
            f"feasible={step_info['feasible']} | "
            f"converged={step_info['converged']} | "
            f"min_v={step_info['min_voltage_pu']:.6f} | "
            f"max_i={step_info['max_line_current_pu']:.6f}"
        )

    print("DESSEnv topology build + rollout PASSED")


def check_eval_script_import():
    print("\n====================================")
    print("CHECK 4: run_tp_eval_ieee69.py import")
    print("====================================")

    module = importlib.import_module(
        "experiments.topology_generalization.run_tp_eval_ieee69"
    )

    required_functions = [
        "build_agent",
        "reset_at_episode_start",
        "get_action",
        "evaluate_one_episode",
        "summarize",
        "make_plots",
        "parse_args",
        "main",
    ]

    for name in required_functions:
        assert hasattr(module, name), f"Missing function: {name}"
        print(f"{name}: OK")

    print("run_tp_eval_ieee69.py import PASSED")


def main():
    print("\n====================================")
    print("IEEE69 TOPOLOGY PIPELINE SMOKE TEST")
    print("====================================")

    check_tp_configs()
    check_topology_utils()
    check_env_build_and_rollout()
    check_eval_script_import()

    print("\n====================================")
    print("ALL IEEE69 TOPOLOGY SMOKE TESTS PASSED")
    print("====================================")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n====================================")
        print("SMOKE TEST FAILED")
        print("====================================")
        traceback.print_exc()
        raise