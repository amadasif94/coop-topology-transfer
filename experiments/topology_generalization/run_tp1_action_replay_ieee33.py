"""
Open-loop replay of recorded TP1 GNN-MADDPG actions (GCN / GAT / TAGConv) on IEEE 33-bus.

    TP1 identity replay (must reproduce the original TP1 final evaluation):
        python -u experiments/topology_generalization/run_tp1_action_replay_ieee33.py --gnn_type gcn \
            --selection <selection.json> --target_topology TP1 [--max_episodes 1]
    TP1 -> TP6 fixed-action replay (only after a PASS full identity validation, and for the full run a PASS
    TP6 smoke validation):
        python -u experiments/topology_generalization/run_tp1_action_replay_ieee33.py --gnn_type gcn \
            --selection <selection.json> --target_topology TP6 --identity_report <json> [--smoke_report <json>] [--max_episodes 1]

No actor is loaded and no new action is generated. The recorded `marl_raw_action` is returned by a
ReplayAgent inside the ORIGINAL evaluator function
experiments.topology_generalization.run_tp_eval.evaluate_one_episode (unmodified). Network inputs are the
archived network_inputs/<TP>/ of the original final run (sha256-verified). Outputs (never overwritten):
    results/topology_generalization/csv/action_replay_ieee33/<TP>/<model>/<selection_id>/<run_id>/
        step_metrics.csv, episode_summary.csv, aggregate_summary.csv, replay_manifest.json, run_manifest.json
"""

import argparse
import sys

import experiments.topology_generalization.run_tp_eval as original
from experiments.topology_generalization.replay_utils_ieee33 import ReplayError, add_replay_arguments, run_replay


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gnn_type", required=True, choices=["gcn", "gat", "tagconv"])
    args = add_replay_arguments(parser).parse_args()
    try:
        run_replay(original, args.gnn_type, args, policy="maddpg")
    except ReplayError as exc:
        print(f"REPLAY STOPPED: {exc}", file=sys.stderr, flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
