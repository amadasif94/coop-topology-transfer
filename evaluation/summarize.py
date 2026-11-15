"""Collect every finished test run and the zero-action reference of one feeder into comparison tables. Maintained.

    python -u -m evaluation.summarize --feeder ieee69 [--protocol soc_feasible] [--output-root results/soc_feasible]

Every prespecified controller (MLP A0-A3, GCN, GAT, TAGConv) is reported; nothing is ranked or selected here, and
nothing feeds back into training or checkpoint selection. Per controller x scale x topology: the mean of each seed's
31-day mean, the between-seed standard deviation and the number of seeds; the zero-action reference has one run.
Action mappings never mix: every group must carry one action_mapping, and the learned controllers of one summary must
all share one (the zero-action reference executes 0 kW and is exempt). Writes
<output-root>/<feeder>/summary_<UTC>/{episodes_long.csv, by_controller.csv, inclusion.json}.

The maintained command also includes the deterministic PF-MPC optimization reference (evaluation.oracle.run):
<root>/<feeder>/pf_mpc/load_x<pct>/ and, if present, the optional pf_mpc_replan_on_deviation variant (smoke folders
load_x<pct>_smoke<N> are excluded). Like the zero reference it is ONE deterministic run (runs = 1, no between-run SD):
it is never averaged as if it were training seeds, a second run of one reference group is refused, and it is exempt
from the learned-controller action-mapping check (it dispatches kW directly: action_mapping 'direct_kw').
by_controller.csv then carries result_type: learned_controller_seed_mean, zero_action_reference_single_run,
deterministic_optimization_single_run (PF-MPC), forecast_optimization_single_run (F-MPC, evaluation.forecast_mpc.run:
<root>/<feeder>/forecast_mpc/load_x<pct>/) or rule_based_single_run (PQ-RBC, evaluation.rule_based.pq_rbc:
<root>/<feeder>/price_quantile_rbc/load_x<pct>/). F-MPC and PQ-RBC follow the same single-run rules as PF-MPC; their
validation-day runs (<label>_validation/) and their model/forecast folders are never included.
"""
import argparse
from datetime import datetime, timezone
import json

import pandas as pd

from configs.experiments import (CONTROLLER_KEYS, FEEDERS, HISTORICAL_SOC_FEASIBLE_MAPPING_VERSION, MAINTAINED_PROTOCOL,
                                 PROTOCOLS, RunLayout, maintained_output_root, write_json)
from evaluation.evaluate import ZERO_LABEL
from utility.run_manifest import reserve_directory

METRICS = ['total_reward_mean', 'total_energy_cost', 'adjusted_energy_cost', 'total_grid_import_mwh',
           'total_curtailment_mwh', 'total_throughput_mwh', 'mean_voltage_deviation', 'max_voltage_deviation',
           'min_voltage_pu', 'max_voltage_pu', 'max_line_current_violation', 'max_voltage_violation', 'feasible_rate',
           'converged_rate', 'b3_feasibility_loss', 'b3_outside_bounds_fraction', 'b3_mean_soc_projection_kw',
           'b3_material_clipping_fraction', 'b3_boundary_occupancy', 'b3_max_infeasible_discharge_streak',
           'b3_mean_backup_change_kw', 'b3_backup_activation_rate', 'b3_terminal_shortfall_mean',
           'b3_terminal_shortfall_max', 'b3_terminal_penalty', 'b3_soc_near_min_fraction', 'b3_soc_near_max_fraction',
           'b3_max_soc_boundary_streak', 'b3_charge_throughput_kwh', 'b3_discharge_throughput_kwh']
GROUP = ['controller', 'load_scaling_factor', 'topology_case']
OPTIMIZATION_REFERENCES = ('pf_mpc', 'pf_mpc_replan_on_deviation')     # evaluation.oracle.run PF_LABELS
CONTROL_REFERENCES = dict(forecast_mpc='forecast_optimization_single_run',          # evaluation.forecast_mpc.run
                          price_quantile_rbc='rule_based_single_run')               # evaluation.rule_based.pq_rbc
SINGLE_RUN_REFERENCES = OPTIMIZATION_REFERENCES + tuple(CONTROL_REFERENCES)
REFERENCE_LABELS = (ZERO_LABEL,) + SINGLE_RUN_REFERENCES


def result_type(controller):
    if controller == ZERO_LABEL:
        return 'zero_action_reference_single_run'
    if controller in OPTIMIZATION_REFERENCES:
        return 'deterministic_optimization_single_run'
    if controller in CONTROL_REFERENCES:
        return CONTROL_REFERENCES[controller]
    return 'learned_controller_seed_mean'


def summarize(feeder, layout, marker='SUMMARY', single_mapping=True, optimization_references=False):
    """single_mapping: refuse a summary whose learned controllers use more than one action mapping (maintained).
    optimization_references: also include the PF-MPC single-run reference(s) (maintained command)."""
    root = layout.root / feeder
    dirs = layout.test_dirs(feeder) + sorted(layout.zero_dir(feeder).glob('load_x*'))
    included, excluded, frames = [], [], []
    if optimization_references:
        for label in SINGLE_RUN_REFERENCES:
            for d in sorted((root / label).glob('load_x*')):
                if '_smoke' in d.name:
                    excluded.append(dict(dir=str(d), reason='smoke run (subset of days)'))
                else:
                    dirs.append(d)
    for d in dirs:
        manifest = json.loads((d / 'run_manifest.json').read_text()) if (d / 'run_manifest.json').is_file() else {}
        if manifest.get('status') != 'finished':
            excluded.append(dict(dir=str(d), reason='missing or unfinished'))
            continue
        ep = pd.read_csv(d / 'episode_summary.csv')
        ep['run_id'] = ep['run_id'].fillna(0) if 'run_id' in ep else 0
        frames.append(ep)
        included.append(str(d))
    if not frames:
        raise SystemExit(f'No finished test/zero runs under {root}')
    table = pd.concat(frames, ignore_index=True)
    if 'action_mapping' in table:
        # Episode tables written before mapping versions existed: soc_feasible there is the zero-preserving mapping, so
        # it can never be summarised together with affine runs; rated has no version.
        if 'action_mapping_version' not in table:
            table['action_mapping_version'] = None
        old = table['action_mapping_version'].isna() & table['action_mapping'].notna()
        table.loc[old, 'action_mapping_version'] = table.loc[old, 'action_mapping'].map(
            lambda m: HISTORICAL_SOC_FEASIBLE_MAPPING_VERSION if m == 'soc_feasible' else 'rated')
    found = {}
    for column in ('action_mapping', 'action_mapping_version'):   # labels only (never averaged); never mixed
        if column not in table:
            continue
        labels = table.groupby(GROUP)[column].agg(lambda s: ';'.join(sorted(set(s.dropna().astype(str)))))
        mixed = labels[labels.str.contains(';')]
        if len(mixed):
            raise SystemExit(f'Mixed {column} within controller groups: {mixed.to_dict()}')
        learned = set(table.loc[~table.controller.isin(REFERENCE_LABELS), column].dropna().astype(str))
        if single_mapping and len(learned) > 1:
            raise SystemExit(f'Mixed {column} across controllers of one summary: {sorted(learned)}')
        found[column] = labels
    if optimization_references:
        refs = table[table.controller.isin(SINGLE_RUN_REFERENCES)]
        if len(refs):
            runs = refs.groupby(GROUP)['run_id'].nunique()
            days = refs.groupby(GROUP)['start_index'].agg(lambda s: s.duplicated().any())
            if (runs > 1).any() or days.any():
                raise SystemExit('A PF-MPC / F-MPC / PQ-RBC reference group holds more than one run or repeated days; it '
                                 'is a single deterministic optimization or rule and is never averaged like training seeds: '
                                 f'{runs[runs > 1].to_dict() or days[days].to_dict()}')
    out = reserve_directory(root / f'summary_{datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")}')
    table.to_csv(out / 'episodes_long.csv', index=False)
    metrics = [m for m in METRICS if m in table]
    by_run = table.groupby(GROUP + ['run_id'])[metrics].mean()
    grouped = by_run.groupby(GROUP)
    summary = grouped.mean().add_suffix('_mean').join(grouped.std(ddof=1).add_suffix('_between_run_sd'))
    summary['runs'] = grouped.size()
    for column, labels in found.items():
        summary[column] = labels.replace('', None)
    summary = summary.reset_index()
    if optimization_references:
        summary['result_type'] = summary['controller'].map(result_type)
    summary.to_csv(out / 'by_controller.csv', index=False)
    expected = {layout.label(k) for k in CONTROLLER_KEYS}
    write_json(out / 'inclusion.json', dict(feeder=feeder, included=included, excluded=excluded,
        missing_controllers=sorted(expected - set(table.controller)),
        note='All prespecified controllers are reported; no ranking or selection on test data is performed here.'))
    print(marker, out, flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--feeder', required=True, choices=FEEDERS)
    p.add_argument('--protocol', choices=sorted(PROTOCOLS), default=MAINTAINED_PROTOCOL)
    p.add_argument('--output-root', help='default results/<protocol>')
    a = p.parse_args(argv)
    try:
        root = maintained_output_root(a.protocol, a.output_root)
    except ValueError as exc:
        p.error(str(exc))
    return summarize(a.feeder, RunLayout(root), optimization_references=True)


if __name__ == '__main__':
    main()
