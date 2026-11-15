"""
Open-loop replay of recorded TP1 MLP-MADDPG actions on IEEE 33-bus.

    TP1 identity replay:
        python -u experiments/no_gnn_maddpg/run_tp1_action_replay_mlp_ieee33.py --selection <selection.json> \
            --target_topology TP1 [--max_episodes 1]
    TP1 -> TP6 fixed-action replay (after a PASS full identity validation; full run also needs a PASS TP6 smoke):
        python -u experiments/no_gnn_maddpg/run_tp1_action_replay_mlp_ieee33.py --selection <selection.json> \
            --target_topology TP6 --identity_report <json> [--smoke_report <json>] [--max_episodes 1]

No actor is loaded; the recorded `marl_raw_action` is returned by a ReplayAgent inside the ORIGINAL evaluator
function experiments.no_gnn_maddpg.run_tp_eval_mlp.evaluate_one_episode (unmodified; policy 'mlp_maddpg').
Outputs as in experiments/topology_generalization/run_tp1_action_replay_ieee33.py.
"""

import argparse
import sys

import experiments.no_gnn_maddpg.run_tp_eval_mlp as original
from experiments.topology_generalization.replay_utils_ieee33 import ReplayError, add_replay_arguments, run_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    args = add_replay_arguments(parser).parse_args()
    try:
        run_replay(original, "mlp", args, policy=original.MLP_POLICY_NAME)
    except ReplayError as exc:
        print(f"REPLAY STOPPED: {exc}", file=sys.stderr, flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
