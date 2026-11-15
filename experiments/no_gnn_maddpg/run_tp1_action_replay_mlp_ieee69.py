"""
Open-loop replay of recorded TP1 MLP-MADDPG actions on IEEE 69-bus.

    TP1 identity replay (implementation validation; must reproduce the original TP1 evaluation):
        python -u experiments/no_gnn_maddpg/run_tp1_action_replay_mlp_ieee69.py \
            --selection <selection.json> --target_topology TP1 [--max_episodes 1]
    TP1 -> TP10 action replay (only after a PASS full identity validation and a PASS TP10 smoke validation):
        python -u experiments/no_gnn_maddpg/run_tp1_action_replay_mlp_ieee69.py \
            --selection <selection.json> --target_topology TP10 --feedback_job <job id> \
            --identity_report <identity PASS json> [--smoke_report <TP10 smoke PASS json>] [--max_episodes 1]

No actor is loaded and no new action is generated: the recorded `marl_raw_action` of the original
TP1 final evaluation is returned by a ReplayAgent inside the ORIGINAL evaluator function
experiments.no_gnn_maddpg.run_tp_eval_mlp_ieee69.evaluate_one_episode (unmodified; policy label
'mlp_maddpg'). Outputs as in experiments/topology_generalization/run_tp1_action_replay_ieee69.py.
"""

import argparse
import sys

import experiments.no_gnn_maddpg.run_tp_eval_mlp_ieee69 as original
from experiments.topology_generalization.replay_utils_ieee69 import ReplayError, add_replay_arguments, run_replay


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
