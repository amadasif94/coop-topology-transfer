"""Maintained experiment configuration: feeders, training settings, protocols, coefficients, controllers, run layout.

Protocols (descriptive name -> internal protocol id recorded in every config/manifest as experiment_protocol.name):
  soc_feasible                  -> 'soc_feasible'  MAINTAINED DEFAULT
  rated_feasibility_supervised  -> 'b4'            historical reproduction only (explicitly selected)
  rated_quadratic_terminal      -> 'b3'            historical reproduction only (explicitly selected)
Internal ids are kept so that configurations, manifests and checkpoints stay byte-compatible with existing results.

All protocols share the original feeders, topologies (IEEE33 TP1-TP7, IEEE69 TP1-TP10), datasets and training settings.
'b4' = the original configuration plus (1) batteries for ~40% installed power penetration of the full-profile peak
demand at 0.20C (IEEE33 peak 1774.21 kW: 5 x 700 kWh / 140 kW, 39.45%; IEEE69 peak 3840.30 kW: 9 x 850 kWh / 170 kW,
39.84%; efficiencies 0.95, SoC limits 0.10-0.90), (2) TRAINING episodes start at SoC_i0 ~ U(0.30, 0.70) per battery,
validation/test at 0.50, (3) the two-sided terminal stored-energy valuation instead of the quadratic terminal SoC
penalty (environments.marl_env.terminal_energy_value, the convention of the adjusted net daily cost), (4) gamma = 1.0
with the completed 96th transition of a day as a true terminal (undiscounted finite-horizon return).
'soc_feasible' = 'b4' with SOC-feasible action mapping (environments.battery_feasibility.soc_feasible_action, version
'affine_v2' since 2026-11-07: the policy action a in [-1, 1] is mapped affinely onto [-P_ch_avail(SoC), +P_dis_avail(SoC)];
earlier soc_feasible runs used the zero-preserving a * P_available, 'zero_preserving_v1') and without the pre-tanh
SoC-feasibility supervision.
'b3' = the original configuration (rated mapping, quadratic terminal penalty, gamma 0.99) with the supervision.

Run layout (RunLayout): maintained runs live in <root>/<feeder>/<family>/<variant>/seed_<k>/ (MLP: family 'mlp',
variant A0-A3) or <root>/<feeder>/<family>/seed_<k>/ (GCN/GAT/TAGConv), root default results/<protocol name>. Seed
index k in {0, 1, 2} uses the original RNG seed 42 + k. The historical layout <root>/<feeder>/b3_<controller>/run_<k>/
exists only for the historical entry points (experiments/b3, legacy=True).
"""
import copy
import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch

from environments.battery_feasibility import SOC_FEASIBLE_MAPPING_VERSION
from experiments.no_shared_kpi.protocol import TRAIN_SETTINGS as _ORIGINAL

# soc_feasible mapping versions: 'affine_v2' (current; environments.battery_feasibility.soc_feasible_action) and the
# zero-preserving piecewise mapping a * P_avail used by every soc_feasible run/checkpoint written before 2026-11-07.
HISTORICAL_SOC_FEASIBLE_MAPPING_VERSION = 'zero_preserving_v1'

ROOT = Path(__file__).resolve().parents[1]
FEEDERS = ('ieee33', 'ieee69')
TOPOLOGIES = dict(ieee33=[f'TP{i}' for i in range(1, 8)], ieee69=[f'TP{i}' for i in range(1, 11)])
# Original production settings of every MLP and GNN training launcher (SLURM Script/run_*_maddpg_*.slurm):
# 200000 steps, 10000 warm-up, buffer 100000, batch 128, lr 5e-5 / 5e-4, gamma 0.99, tau 0.005, noise 0.05,
# validation and checkpoints every 5000 steps on 10 validation days. Architecture sizes are the trainers' defaults.
TRAIN_SETTINGS = dict(_ORIGINAL)
ARCHITECTURE = dict(actor_hidden_dims=(256, 256), critic_hidden_dims=(256, 256), gnn_hidden_dim=64,
                    gnn_embedding_dim=64, gnn_num_layers=3)

# ----------------------------------------------------------------------------------------------- protocols
EXPERIMENTS = ('b3', 'b4', 'soc_feasible')                      # internal protocol ids
B4_FAMILY = ('b4', 'soc_feasible')          # protocols built on the 'b4' battery/SoC/terminal/gamma configuration
ACTION_MAPPINGS = ('rated', 'soc_feasible')
PROTOCOLS = dict(soc_feasible='soc_feasible', rated_feasibility_supervised='b4', rated_quadratic_terminal='b3')
MAINTAINED_PROTOCOL = 'soc_feasible'
B4_GAMMA = 1.0
B4_BATTERIES = dict(ieee33=dict(capacity_kwh=700.0, max_charge_kw=140.0, max_discharge_kw=140.0),
                    ieee69=dict(capacity_kwh=850.0, max_charge_kw=170.0, max_discharge_kw=170.0))
B4_PEAK_DEMAND_KW = dict(ieee33=1774.21, ieee69=3840.30)
B4_INITIAL_SOC_TRAIN = (0.30, 0.70)
B4_TERMINAL_VALUATION = 'two_sided_stored_energy_at_day_mean_price'


def protocol_id(name):
    """Internal protocol id of a descriptive protocol name (internal ids are accepted unchanged)."""
    if name in PROTOCOLS:
        return PROTOCOLS[name]
    if name in EXPERIMENTS:
        return name
    raise ValueError(f'Unknown protocol {name}; choose from {sorted(PROTOCOLS)}')


def protocol_name(experiment):
    """Descriptive protocol name of an internal id."""
    return {v: k for k, v in PROTOCOLS.items()}[experiment]


def apply_experiment(config, feeder, experiment):
    """Apply a protocol to a (deep-copied) feeder config in place; 'b3' leaves the original configuration untouched."""
    if experiment == 'b3':
        return config
    if experiment not in B4_FAMILY:
        raise ValueError(f'Unknown experiment {experiment}; choose from {EXPERIMENTS}')
    dess = config['dess']
    if (dess['soc_min'], dess['soc_max'], dess['soc_init']) != (0.10, 0.90, 0.50):
        raise ValueError('b4 expects the configured SoC limits 0.10-0.90 and soc_init 0.50')
    dess.update(B4_BATTERIES[feeder])
    n = len(dess['dess_buses'])
    config['experiment_protocol'] = dict(
        name='b4', feeder=feeder, battery=dict(B4_BATTERIES[feeder], count=n, c_rate=dess['max_discharge_kw'] / dess['capacity_kwh'],
        fleet_capacity_kwh=n * dess['capacity_kwh'], fleet_power_kw=n * dess['max_discharge_kw'],
        peak_demand_kw=B4_PEAK_DEMAND_KW[feeder], power_penetration=n * dess['max_discharge_kw'] / B4_PEAK_DEMAND_KW[feeder]),
        initial_soc=dict(train_uniform=list(B4_INITIAL_SOC_TRAIN), validation_and_test=dess['soc_init']),
        terminal=dict(valuation=B4_TERMINAL_VALUATION, quadratic_shortfall_penalty=False),
        discount=dict(gamma=B4_GAMMA, completed_day_terminal=True))
    if experiment == 'soc_feasible':
        config['experiment_protocol'].update(name='soc_feasible', action_mapping='soc_feasible',
                                             action_mapping_version=SOC_FEASIBLE_MAPPING_VERSION,
                                             feasibility_supervision=False)
    return config


def check_root_mapping_version(root, feeder, experiment):
    """Refuse to add runs to an output root whose existing training runs (any layout) of this feeder were trained with a
    different action-mapping version, e.g. new affine_v2 runs next to zero_preserving_v1 runs. Read-only; returns the
    number of existing runs inspected. Use a fresh --output-root for a new mapping version."""
    current = action_mapping_version_of(feeder_config(feeder, experiment=experiment))
    found = {}
    for manifest in sorted(Path(root, feeder).glob('**/training/run_manifest.json')):
        try:
            record = run_block(json.loads(manifest.read_text()))
        except (OSError, ValueError):
            continue
        if record.get('experiment', 'b3') != experiment:
            continue
        version = action_mapping_version_of(record)
        if version != current:
            found[str(manifest.parent.parent)] = version
    if found:
        sample = dict(list(found.items())[:5])
        raise ValueError(f'output root {root} already holds {len(found)} {feeder} run(s) trained with another '
                         f'action_mapping_version (e.g. {sample}); this code trains {current!r}. '
                         'Use a fresh --output-root; existing results are never modified.')
    return len(found)


def action_mapping_version_of(record):
    """Version of the soc_feasible mapping of a config / run record / checkpoint (None for 'rated'). Records written
    before versions existed used the zero-preserving mapping (HISTORICAL_SOC_FEASIBLE_MAPPING_VERSION)."""
    protocol = record.get('experiment_protocol') or {}
    mapping = record.get('action_mapping', protocol.get('action_mapping', 'rated'))
    if mapping != 'soc_feasible':
        return None
    return record.get('action_mapping_version', protocol.get('action_mapping_version',
                                                             HISTORICAL_SOC_FEASIBLE_MAPPING_VERSION))


def action_mapping_of(config):
    """'soc_feasible' or 'rated' (every b3/b4 configuration, which predates the option)."""
    mapping = (config.get('experiment_protocol') or {}).get('action_mapping', 'rated')
    if mapping not in ACTION_MAPPINGS:
        raise ValueError(f'Unknown action_mapping {mapping}')
    return mapping


def feasibility_supervision_of(experiment):
    """Whether the pre-tanh SoC-feasibility supervision loss is trained (b3, b4) or not attached (soc_feasible)."""
    return experiment != 'soc_feasible'


def feeder_config(feeder, time_series=None, experiment='b3'):
    if feeder == 'ieee33':
        from configs.ieee33_config import IEEE33_CONFIG as config
    elif feeder == 'ieee69':
        from configs.ieee69_config import IEEE69_CONFIG as config
    else:
        raise ValueError(f'Unknown feeder {feeder}')
    config = copy.deepcopy(config)
    if time_series:
        config['paths']['processed_time_series'] = Path(time_series).resolve()
    return apply_experiment(config, feeder, experiment)


def topology_config(feeder, case_name, output_root, time_series=None, experiment='b3'):
    if feeder == 'ieee33':
        from experiments.topology_generalization.topology_utils import build_topology_config
    else:
        from experiments.topology_generalization.topology_utils_ieee69 import build_topology_config
    config = build_topology_config(case_name=case_name, base_config=feeder_config(feeder, time_series, experiment),
                                   output_root=output_root, validate=True)
    return config


# ------------------------------------------------------------------------------------- coefficients
# Selected coefficients (decided 2026-10-29), identical for A0-A3, GCN, GAT and TAGConv of a feeder.
#   lambda_f = 3.0: SoC-feasibility supervision weight of the rated protocols (inactive under soc_feasible).
#   lambda_T: quadratic terminal coefficient of protocol 'b3' only (environments.marl_env.terminal_coefficient):
#     IEEE33: 0.65 * 2.0 * 37.943768 $/MWh * 1.052632 MWh = 51.923   (5 x 500 kWh)
#     IEEE69: 0.65 * 2.0 * 37.943768 $/MWh * 3.789474 MWh = 186.923  (9 x 1000 kWh)
SELECTED_COEFFICIENTS = dict(ieee33=dict(lambda_feasibility=3.0, lambda_terminal=51.923),
                             ieee69=dict(lambda_feasibility=3.0, lambda_terminal=186.923))
COEFFICIENT_NOTE = ('lambda_f = 3.0 is the initial reference value (common, feasibility-oriented; informed by preliminary '
                    'IEEE33 post-tanh gradient evidence; not optimised, not validated on IEEE69). With the final pre-tanh '
                    'feasibility loss it is not numerically equivalent to the historical post-tanh lambda_f = 3.0; loss and '
                    'gradient scales are recorded in training_updates.csv. lambda_T from the training-data rule.')


def load_coefficients(feeder, lambda_feasibility=None, lambda_terminal=None, experiment='b3'):
    """The selected per-feeder coefficients, unless overridden explicitly on the command line.

    b4 and soc_feasible have no quadratic terminal penalty: lambda_T is 0 (an explicit nonzero override is refused).
    soc_feasible attaches no SoC-feasibility supervision: lambda_f is recorded as 0.0 (inactive; a nonzero value is
    refused)."""
    selected = SELECTED_COEFFICIENTS[feeder]
    lf_source = 'command_line_override' if lambda_feasibility is not None else 'selected_default'
    lf_selected = selected['lambda_feasibility']
    if not feasibility_supervision_of(experiment):
        if lambda_feasibility not in (None, 0, 0.0):
            raise ValueError(f'{experiment} has no SoC-feasibility supervision; lambda_f must not be set')
        lambda_feasibility, lf_selected = 0.0, 0.0
        lf_source = f'{experiment}: inactive (no SoC-feasibility supervision attached)'
    lf = selected['lambda_feasibility'] if lambda_feasibility is None else float(lambda_feasibility)
    if experiment in B4_FAMILY:
        if lambda_terminal not in (None, 0, 0.0):
            raise ValueError(f'{experiment} replaces the quadratic terminal penalty; lambda_T must be 0')
        return float(lf), 0.0, dict(
            selected=dict(lambda_feasibility=lf_selected, lambda_terminal=0.0), note=COEFFICIENT_NOTE,
            lambda_feasibility_source=lf_source,
            lambda_terminal_source=f'{experiment}: quadratic terminal penalty removed (two-sided terminal energy valuation)')
    lt = selected['lambda_terminal'] if lambda_terminal is None else float(lambda_terminal)
    return float(lf), float(lt), dict(
        selected=dict(selected), note=COEFFICIENT_NOTE,
        lambda_feasibility_source='command_line_override' if lambda_feasibility is not None else 'selected_default',
        lambda_terminal_source='command_line_override' if lambda_terminal is not None else 'selected_default')


# --------------------------------------------------------------------------------------- controllers
FAMILIES = ('mlp', 'gcn', 'gat', 'tagconv')
VARIANTS = ('A0', 'A1', 'A2', 'A3')                   # MLP ablations; GNN controllers have none
CONTROLLER_KEYS = ('mlp_a0', 'mlp_a1', 'mlp_a2', 'mlp_a3', 'gcn', 'gat', 'tagconv')
LEGACY_LABELS = {key: 'b3_' + key for key in CONTROLLER_KEYS}   # historical run/checkpoint labels (b3_mlp_a0 ...)


def controller_key(family, variant=None):
    """('mlp', 'A0') -> 'mlp_a0'; ('gcn', None) -> 'gcn'."""
    if family not in FAMILIES:
        raise ValueError(f'Unknown controller family {family}; choose from {FAMILIES}')
    if family == 'mlp':
        if variant not in VARIANTS:
            raise ValueError(f'MLP needs a variant in {VARIANTS}')
        return f'mlp_{variant.lower()}'
    if variant is not None:
        raise ValueError(f'{family} has no A0-A3 variant')
    return family


def family_variant(key):
    return ('mlp', key.split('_')[1].upper()) if key.startswith('mlp_') else (key, None)


class RunLayout:
    """Where runs, selections, zero-action references and summaries live (maintained or historical layout)."""

    def __init__(self, root, legacy=False):
        self.root, self.legacy = Path(root), bool(legacy)

    def label(self, key):
        return LEGACY_LABELS[key] if self.legacy else key

    def run_key(self, seed):
        """Run identifier used in selection records: run_<k> (historical) or seed_<k> (maintained)."""
        return f'run_{seed}' if self.legacy else f'seed_{seed}'

    @property
    def block(self):
        """Manifest block holding the run record: 'b3' (historical) or 'run_record' (maintained)."""
        return 'b3' if self.legacy else 'run_record'

    def run_dir(self, feeder, key, seed):
        if self.legacy:
            return self.root / feeder / LEGACY_LABELS[key] / f'run_{seed}'
        family, variant = family_variant(key)
        base = self.root / feeder / family
        return (base / variant if variant else base) / f'seed_{seed}'

    def selections_dir(self, feeder):
        return self.root / feeder / 'selections'

    def zero_dir(self, feeder):
        return self.root / feeder / 'zero_action_no_backup'

    def test_dirs(self, feeder):
        base = self.root / feeder
        if self.legacy:
            return [d for key in CONTROLLER_KEYS for d in sorted(base.glob(f'{LEGACY_LABELS[key]}/run_*/test/load_x*'))]
        return sorted(base.glob('mlp/A?/seed_*/test/load_x*')) + sorted(
            d for f in FAMILIES[1:] for d in base.glob(f'{f}/seed_*/test/load_x*'))


def maintained_output_root(protocol, output_root=None):
    """Default results/<protocol name>; never inside a historical results/b3 or results/b4 tree."""
    root = Path(output_root or f'results/{protocol_name(protocol_id(protocol))}')
    parts = [p.lower() for p in root.resolve().parts]
    if 'b3' in parts or 'b4' in parts:
        raise ValueError(f'maintained runs must not write into a historical B3/B4 results tree (path part b3 or b4): {root}')
    return root


def run_block(manifest):
    """Run record of a training manifest: 'run_record' (maintained) or 'b3' (historical)."""
    return manifest.get('run_record') or manifest.get('b3') or {}


def experiment_of(training_manifest):
    """Internal protocol id of a finished training run (manifests written before 'b4' existed are 'b3')."""
    return run_block(training_manifest).get('experiment', 'b3')


# ------------------------------------------------------------------------------------------ records
# Maintained source modules (hashed into every maintained manifest: source_inventory).
MAINTAINED_SOURCES = ('configs/experiments.py', 'environments/battery_feasibility.py', 'environments/marl_env.py',
                      'environments/local_reward_env.py', 'models/controllers.py', 'models/independent_local_agents.py',
                      'training/bounded_replay.py', 'training/feasibility_supervision.py',
                      'training/controller_trainer.py', 'training/train.py', 'utility/feasibility_metrics.py',
                      'evaluation/evaluate.py', 'evaluation/zero.py', 'evaluation/summarize.py')


def source_inventory():
    return {p: sha(ROOT / p) for p in MAINTAINED_SOURCES}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def write_json(path, data):
    """Exclusive creation: an existing record is never overwritten."""
    from utility.run_manifest import serial
    with Path(path).open('x', encoding='utf-8') as f:
        json.dump(serial(data), f, indent=2, allow_nan=False, default=str)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
