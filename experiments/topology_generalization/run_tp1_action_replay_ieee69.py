"""
Open-loop replay of recorded TP1 GNN-MADDPG actions (GCN / TAGConv / GAT) on IEEE 69-bus.

    TP1 identity replay (implementation validation; must reproduce the original TP1 evaluation):
        python -u experiments/topology_generalization/run_tp1_action_replay_ieee69.py --gnn_type gcn \
            --selection <selection.json> --target_topology TP1 [--max_episodes 1]
    TP1 -> TP10 action replay (only after a PASS full identity validation and a PASS TP10 smoke validation):
        python -u experiments/topology_generalization/run_tp1_action_replay_ieee69.py --gnn_type gcn \
            --selection <selection.json> --target_topology TP10 --feedback_job <job id> \
            --identity_report <identity PASS json> [--smoke_report <TP10 smoke PASS json>] [--max_episodes 1]

No actor is loaded and no new action is generated: the recorded `marl_raw_action` of the original
TP1 final evaluation is returned by a ReplayAgent inside the ORIGINAL evaluator function
experiments.topology_generalization.run_tp_eval_ieee69.evaluate_one_episode (unmodified), so reset,
env.step(), physical limits, DOE, directional backup, power flow, SOC dynamics and all metrics are
exactly the original code. Outputs (never overwritten):
    results/topology_generalization/csv_ieee69/action_replay/<TP>/<model>/<selection_id>/<run_id>/
        step_metrics.csv, episode_summary.csv, aggregate_summary.csv, replay_manifest.json, run_manifest.json
"""

import argparse
import sys

import experiments.topology_generalization.run_tp_eval_ieee69 as original
from experiments.topology_generalization.replay_utils_ieee69 import ReplayError, add_replay_arguments, run_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gnn_type", required=True, choices=["gcn", "tagconv", "gat"])
    args = add_replay_arguments(parser).parse_args()
    try:
        run_replay(original, args.gnn_type, args, policy="maddpg")
    except ReplayError as exc:
        print(f"REPLAY STOPPED: {exc}", file=sys.stderr, flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
