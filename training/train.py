"""Train one controller (one feeder, one seed). Maintained entry point.

    python -u -m training.train --feeder ieee69 --controller mlp --variant A0 --seed 0 [--cpu]
    python -u -m training.train --feeder ieee33 --controller gcn --seed 1

--controller mlp|gcn|gat|tagconv, --variant A0|A1|A2|A3 (MLP only), --seed 0|1|2 (RNG seed 42 + seed index; TP1
training split, TP1 validation split for periodic validation). --protocol defaults to soc_feasible (SOC-feasible action
mapping, no SoC-feasibility supervision; configs.experiments); the historical rated protocols
rated_feasibility_supervised / rated_quadratic_terminal must be selected explicitly. Original training settings
(configs.experiments.TRAIN_SETTINGS / ARCHITECTURE; gamma 1.0 for the soc_feasible and rated_feasibility_supervised
protocols) unless overridden explicitly (overrides recorded).
Output (refused if not empty): <output-root>/<feeder>/<controller>[/<variant>]/seed_<k>/training/, output root default
results/<protocol>:
    run_manifest.json   seeds, dataset/time-series and network file hashes, TP1 environments, architecture,
                        hyperparameters, exploration, protocol and coefficients (run_record), maintained source hashes,
                        git revision, PyTorch version; reset handling and step counts at the end
    training_updates.csv / training_days.csv / training_gradient_balance.csv   training diagnostics
    checkpoints/maddpg_step_<N>.pt (every save_every steps) and maddpg_final.pt  (schema graph-marl-dess-checkpoint/1)
    checkpoints/validation/step_<N>/  periodic TP1 validation (original S-PER-v1 schedule)
    training_logs.json
"""
import argparse
import platform
import sys

import torch

from configs.experiments import (ARCHITECTURE, B4_FAMILY, B4_GAMMA, FAMILIES, FEEDERS, MAINTAINED_PROTOCOL, PROTOCOLS,
                                 TRAIN_SETTINGS, VARIANTS, RunLayout, action_mapping_of, action_mapping_version_of,
                                 check_root_mapping_version, controller_key,
                                 feasibility_supervision_of, feeder_config, load_coefficients, maintained_output_root,
                                 protocol_id, protocol_name, set_seed, source_inventory, write_json)
from environments.local_reward_env import REWARD_DEFINITION, make_a3_env
from environments.marl_env import make_marl_env, terminal_coefficient
from models.controllers import CHECKPOINT_SCHEMA, GNN_TYPES, LEGACY_SCHEMA, LOCAL_REWARD_CONTROLLERS
from training.bounded_replay import SCHEMA as REPLAY_SCHEMA
from training.controller_trainer import trainer_class
from training.feasibility_supervision import FORMULATION
from utility.run_manifest import RunManifest, serial


def train_run(feeder, controller, seed, experiment, out, settings, *, lambda_feasibility=None, lambda_terminal=None,
              max_reset_retries=0, diagnostics_every=100, gradient_balance_log=False, time_series=None, cpu=False,
              legacy=False, sources=None, arguments=None, selection_note=None):
    """Train controller key `controller` (mlp_a0 ... tagconv) with seed index `seed` under internal protocol
    `experiment` into the empty directory `out`. legacy=True writes the historical names (b3_* label, 'b3' manifest
    block, 'b3-checkpoint/2'); sources: source-hash inventory recorded in the manifest (default: maintained)."""
    b4_family = experiment in B4_FAMILY
    settings = dict(settings)
    if b4_family:
        if settings['gamma'] not in (TRAIN_SETTINGS['gamma'], B4_GAMMA):
            raise SystemExit(f'{experiment} uses gamma = {B4_GAMMA} (undiscounted finite horizon); '
                             f'--gamma {settings["gamma"]} refused')
        settings['gamma'] = B4_GAMMA
    lambda_f, lambda_t, coefficient_source = load_coefficients(feeder, lambda_feasibility, lambda_terminal, experiment)
    label = RunLayout('.', legacy=legacy).label(controller)
    rng_seed = 42 + seed
    set_seed(rng_seed)
    device = 'cpu' if cpu else ('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = feeder_config(feeder, time_series, experiment)
    action_mapping, supervision = action_mapping_of(cfg), feasibility_supervision_of(experiment)
    local_reward = controller in LOCAL_REWARD_CONTROLLERS
    make_env = make_a3_env if local_reward else make_marl_env
    env = make_env(cfg, 'train', rng_seed, lambda_terminal=0.0)
    rule_lt, derivation = terminal_coefficient(env)            # recomputed from THIS training data, recorded only
    derivation['difference_from_used_lambda_terminal'] = rule_lt - lambda_t
    if b4_family:
        derivation['note'] = f'{experiment}: recorded only; the quadratic terminal penalty is not used (lambda_T = 0)'
    elif abs(rule_lt - lambda_t) > 0.01:
        print(f'WARNING: training-data rule gives lambda_T={rule_lt:.4f}, but {lambda_t} is used '
              f'({coefficient_source["lambda_terminal_source"]}); check the training data', flush=True)
    env.lambda_terminal = lambda_t
    val = make_env(cfg, 'val', rng_seed + 1000, lambda_terminal=lambda_t)

    config = dict(settings, **ARCHITECTURE, checkpoint_dir=str(out / 'checkpoints'), share_actor=False,
                  store_buffer_on_device=True, action_dim_per_agent=1, max_reset_retries=max_reset_retries,
                  gnn_type=GNN_TYPES.get(controller, 'gcn'), diagnostics_every=diagnostics_every,
                  gradient_balance_log=local_reward or b4_family or gradient_balance_log)
    run = RunManifest(out, label, rng_seed, 'train', arguments=dict(arguments or {}, device=device))
    run.add_environment(env)
    run.add_environment(val)
    if b4_family:
        terminal = ('per-battery own valuation w_grid*phi_cost*p_day*(E_iT-E_i0)/1000 on completed days' if local_reward
                    else 'team valuation w_grid*phi_cost*p_day*sum_j(E_jT-E_j0)/1000 to every agent on completed days')
    else:
        terminal = ('per-battery -kappa_i * d_i^2 on completed days, kappa_i = lambda_T * R_i / sum_j R_j'
                    if local_reward else 'team-wide -lambda_T * mean_i d_i^2 on completed days')
    record = dict(controller=label, controller_key=controller, feeder=feeder, run=seed, seed_index=seed,
        experiment=experiment, protocol=protocol_name(experiment), experiment_protocol=cfg.get('experiment_protocol'),
        lambda_feasibility=lambda_f, lambda_terminal=lambda_t, lambda_terminal_training_data_rule=derivation,
        coefficient_source=coefficient_source, replay_schema=REPLAY_SCHEMA, terminal_penalty=terminal,
        reward=(dict(definition=REWARD_DEFINITION, module='environments.local_reward_env',
                     terminal_coefficients=([float(k) for k in val.terminal_coefficients] if experiment == 'b3'
                                            else None))
                if local_reward else dict(definition='original_cooperative_team_reward')),
        feasibility_formulation=FORMULATION if supervision else 'none', feasibility_supervision=supervision,
        action_mapping=action_mapping, action_mapping_version=action_mapping_version_of(cfg),
        checkpoint_schema=LEGACY_SCHEMA if legacy else CHECKPOINT_SCHEMA,
        settings_overridden={k: v for k, v in settings.items() if v != TRAIN_SETTINGS[k]},
        architecture=ARCHITECTURE, torch_version=torch.__version__, python_version=platform.python_version(),
        checkpoint_selection=selection_note or 'evaluation.evaluate select: TP1 validation, S-SEL-v1 schedule, mean '
                                               'episode reward, original eligibility rule')
    record['b3_sources' if legacy else 'source_inventory'] = sources if sources is not None else source_inventory()
    run.data['b3' if legacy else 'run_record'] = record
    run._write()

    trainer = trainer_class(controller, lambda_f, action_mapping, supervision, legacy_names=legacy)(env, val, config,
                                                                                                  device)
    trainer.agent.experiment_metadata = dict(feeder=feeder, run=seed, seed=rng_seed, lambda_feasibility=lambda_f,
                                             lambda_terminal=lambda_t, coefficient_source=coefficient_source,
                                             experiment=experiment, action_mapping=action_mapping,
                                             action_mapping_version=action_mapping_version_of(cfg),
                                             feasibility_supervision=supervision)
    run.data['effective_training_config'] = config
    run.record_exploration(dict(trainer.exploration_config(), seed=rng_seed))
    try:
        logs = trainer.train()
    finally:
        run.data['reset_handling'] = trainer.reset_summary()
        run.data.update(actual_total_steps=trainer.global_step, failed_episode_count=trainer.failed_episode_count,
                        invalid_transition_count=trainer.invalid_transition_count)
        run._write()
    write_json(out / 'training_logs.json', serial(logs))
    run.finish()
    return out


def add_controller_arguments(p):
    p.add_argument('--feeder', required=True, choices=FEEDERS)
    p.add_argument('--controller', required=True, choices=FAMILIES)
    p.add_argument('--variant', choices=VARIANTS, help='A0-A3, required for --controller mlp')
    p.add_argument('--seed', required=True, type=int, choices=[0, 1, 2], help='seed index (RNG seed 42 + index)')
    p.add_argument('--protocol', choices=sorted(PROTOCOLS), default=MAINTAINED_PROTOCOL)
    p.add_argument('--output-root', help='default results/<protocol>')


def resolve(args, parser):
    try:
        args.key = controller_key(args.controller, args.variant)
        args.root = maintained_output_root(args.protocol, args.output_root)
    except ValueError as exc:
        parser.error(str(exc))
    args.experiment = protocol_id(args.protocol)
    return args


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_controller_arguments(p)
    p.add_argument('--lambda-feasibility', type=float, help='rated protocols only')
    p.add_argument('--lambda-terminal', type=float, help='rated_quadratic_terminal only')
    for key, value in TRAIN_SETTINGS.items():
        p.add_argument('--' + key.replace('_', '-'), type=type(value), default=value)
    p.add_argument('--max-reset-retries', type=int, default=0)
    p.add_argument('--diagnostics-every', type=int, default=100, help='updates between detailed diagnostic rows')
    p.add_argument('--gradient-balance-log', action='store_true')
    p.add_argument('--time-series', help='override the processed time-series CSV (local copies only; recorded)')
    p.add_argument('--cpu', action='store_true')
    return resolve(p.parse_args(argv), p)


def main(argv=None):
    args = parse_args(argv)
    try:                                      # never mix mapping versions in one output root (read-only check)
        check_root_mapping_version(args.root, args.feeder, args.experiment)
    except ValueError as exc:
        raise SystemExit(str(exc))
    out = RunLayout(args.root).run_dir(args.feeder, args.key, args.seed) / 'training'
    train_run(args.feeder, args.key, args.seed, args.experiment, out, {k: getattr(args, k) for k in TRAIN_SETTINGS},
              lambda_feasibility=args.lambda_feasibility, lambda_terminal=args.lambda_terminal,
              max_reset_retries=args.max_reset_retries, diagnostics_every=args.diagnostics_every,
              gradient_balance_log=args.gradient_balance_log, time_series=args.time_series, cpu=args.cpu,
              arguments=dict(vars(args), root=str(args.root)))
    print('TRAINING_DONE', args.protocol, args.feeder, args.key, f'seed_{args.seed}', out, flush=True)
    return out


if __name__ == '__main__':
    sys.exit(0 if main() else 1)
