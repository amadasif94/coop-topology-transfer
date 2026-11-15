"""Checkpoint selection, frozen zero-shot evaluation, load stress and the pure zero-action reference. Maintained entry point.

    python -u -m evaluation.evaluate select --feeder ieee69 --controller mlp --variant A0 --seed 0
    python -u -m evaluation.evaluate test   --feeder ieee69 --controller gcn --seed 0 [--scales 1.0 1.1 1.2 1.3]
    python -u -m evaluation.evaluate zero   --feeder ieee69 [--scales 1.0 1.1 1.2 1.3]

--protocol defaults to soc_feasible (configs.experiments); the training manifest's protocol must match, and every
selection/test environment is built with that protocol's configuration (validation and test episodes start at SoC 0.50).
select  every saved checkpoint of the finished training run is evaluated on TP1 VALIDATION days (fixed S-SEL-v1
        schedule) and the ORIGINAL frozen selection rule picks one (scripts/select_checkpoint: all scheduled days
        completed and failure-free, original aggregate safety rule, highest mean episode reward, earliest step on
        ties). The selection record is written under <output-root>/<feeder>/selections/. No test data and no held-out
        topology is used.
test    the frozen selected checkpoint, unchanged, on every topology of the feeder (IEEE33 TP1-TP7, IEEE69 TP1-TP10;
        TP1 = training topology) and all 31 December test days, at each load scale (1.0 nominal; 1.1-1.3 stress, PV
        1.0). Learned controllers keep the full network backup.
zero    the pure zero-action reference (evaluation.zero: 0 kW requested, SoC-feasible and executed; backup can never
        move a battery; violations and PF failures reported as measured) on the same topologies, days and scales.
Selection and test verify that training manifest, environment and checkpoint use the same action_mapping
(verified_mapping) and record it: a rated-power checkpoint is refused under SOC-feasible mapping and vice versa.
MLP A3 (trained on individual local rewards) is selected and tested with the same system-level reward as every other
controller (selection_reward_note).
Outputs (never overwritten): <run dir>/{selection_sweep/step_N, test/load_x<pct>} with run dir
<output-root>/<feeder>/<controller>[/<variant>]/seed_<k>, and <output-root>/<feeder>/zero_action_no_backup/load_x<pct>;
each with run_manifest.json, step_metrics.csv, episode_summary.csv, aggregate_summary.csv, terminal_soc_summary.csv.
"""
import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys

import pandas as pd
import torch

from configs.experiments import (TOPOLOGIES, RunLayout, action_mapping_version_of, experiment_of, feeder_config, run_block, set_seed,
                                 source_inventory, topology_config)
from environments.marl_env import make_marl_env
from evaluation.zero import assert_zero_dispatch, make_zero_action_env
from experiments.load_stress.stress_env import ALLOWED_SCALES, check_scale, scale_label, scenario
from models.controllers import LOCAL_REWARD_CONTROLLERS, build_model
from utility.evaluation_protocol import authorize, episode_plan, existing_selections, protocol_config, selection_key_dir
from utility.feasibility_metrics import episode_diagnostics
from utility.run_manifest import RunManifest

POLICY = 'maddpg'          # learned-policy name accepted by the original evaluators and selection script
ZERO_LABEL = 'zero_action_no_backup'
SELECTION_REWARD_NOTE = ('Trained on individual local rewards (a3_local_e3_own_bus_tariff); selected and evaluated with '
                         'the same system-level reward as A0/A1/A2 (make_marl_env). No test data used for selection.')


@dataclass
class EvaluationSpec:
    """One controller run to select or test. layout decides names (maintained or historical)."""
    feeder: str
    key: str                      # controller key mlp_a0 ... tagconv
    seed: int                     # seed index k (RNG seed 42 + k)
    experiment: str               # internal protocol id
    layout: RunLayout
    cpu: bool = False
    sources: dict = None          # source-hash inventory recorded in every manifest (default: maintained)
    arguments: dict = field(default_factory=dict)
    done_prefix: str = ''         # '' (maintained) or 'B3_' (historical command line)

    @property
    def label(self):
        return self.layout.label(self.key)

    @property
    def run_dir(self):
        return self.layout.run_dir(self.feeder, self.key, self.seed)

    @property
    def device(self):
        return 'cpu' if self.cpu else ('cuda' if torch.cuda.is_available() else 'cpu')


def _sources(layout, sources):
    return {('b3_sources' if layout.legacy else 'source_inventory'): sources if sources is not None
            else source_inventory()}


def feeder_tools(feeder):
    if feeder == 'ieee69':
        from experiments.no_gnn_maddpg.run_tp_eval_mlp_ieee69 import evaluate_one_episode, summarize
        from experiments.no_shared_kpi.ieee69.runner import RunManifest as Manifest, energy_rows
    else:
        from experiments.no_gnn_maddpg.run_tp_eval_mlp import evaluate_one_episode, summarize
        from experiments.no_shared_kpi.runner import energy_rows
        Manifest = RunManifest
    return evaluate_one_episode, summarize, energy_rows, Manifest


def training_record(spec):
    path = spec.run_dir / 'training' / 'run_manifest.json'
    record = json.loads(path.read_text()) if path.is_file() else {}
    if record.get('status') != 'finished':
        raise SystemExit(f'{path}: training run missing or not finished')
    if experiment_of(record) != spec.experiment:
        raise SystemExit(f'{path}: trained as experiment {experiment_of(record)}, not {spec.experiment}')
    # Refuse BEFORE anything is written: a run trained with another soc_feasible mapping version (e.g. the
    # zero-preserving runs made before affine_v2) is never selected or tested with this code, and no new folder is
    # created inside its result tree.
    trained = action_mapping_version_of(run_block(record))
    current = action_mapping_version_of(feeder_config(spec.feeder, experiment=spec.experiment))
    if trained != current:
        raise SystemExit(f'{path}: trained with action_mapping_version {trained!r}; this code evaluates {current!r}. '
                         'Historical runs are not re-evaluated or modified; use the matching code version.')
    return record


def verified_mapping(training, env, agent):
    """Training manifest, evaluation environment and checkpoint must agree on the action mapping (rated before the
    option existed); a rated checkpoint is never evaluated under SOC-feasible mapping or vice versa."""
    found = dict(training=run_block(training).get('action_mapping', 'rated'),
                 environment=getattr(env, 'action_mapping', 'rated'), checkpoint=getattr(agent, 'action_mapping', 'rated'))
    if len(set(found.values())) != 1:
        raise SystemExit(f'action_mapping mismatch: {found}')
    # soc_feasible mapping version (training manifests without one used the zero-preserving mapping): a checkpoint is
    # never evaluated under a different version of the mapping it was trained with.
    versions = dict(training=action_mapping_version_of(run_block(training)),
                    environment=getattr(env, 'action_mapping_version', None),
                    checkpoint=getattr(agent, 'action_mapping_version', None))
    if len(set(versions.values())) != 1:
        raise SystemExit(f'action_mapping_version mismatch: {versions}')
    return dict(action_mapping=found['environment'], action_mapping_version=versions['environment'],
                feasibility_supervision=bool(getattr(agent, 'feasibility_supervision', True)))


def load_model(controller, checkpoint, env, device, legacy_names=False):
    """controller: key (mlp_a0 ...). The checkpoint's schema and either label (key or historical b3_*) are accepted."""
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    model = build_model(controller, ck['lambda_feasibility'],
                        **dict(ck['model_kwargs'], device=device, legacy_names=legacy_names))
    model.load(checkpoint, env)          # schema, controller, mapping, hyperparameters and F14 provenance checks
    model.actor.eval()
    return model


def run_episodes(feeder, env, agent, policy, tp, plan):
    evaluate_one_episode = feeder_tools(feeder)[0]
    rows = []
    for ep, (start, seed) in enumerate(plan):
        rows.extend(evaluate_one_episode(env, agent, policy, tp, ep, int(start), int(seed)))
    return pd.DataFrame(rows)


def write_tables(feeder, out, frames, envs, label, extra):
    _, summarize, energy_rows, _ = feeder_tools(feeder)
    steps = pd.concat(frames, ignore_index=True)
    ep, agg = summarize(steps)
    energy = pd.concat([energy_rows(f, envs[f.topology_case.iloc[0]]) for f in frames], ignore_index=True)
    diag = pd.concat([episode_diagnostics(f, envs[f.topology_case.iloc[0]]) for f in frames], ignore_index=True)
    keys = ['topology_case', 'episode_id', 'start_index']
    ep = ep.merge(energy, on=keys, validate='one_to_one').merge(diag, on=keys, validate='one_to_one')
    means = ep.groupby('topology_case')[[c for c in ep if c.startswith('b3_') or c in (
        'raw_energy_cost', 'adjusted_energy_cost', 'terminal_stored_energy_kwh')]].mean().add_prefix('mean_').reset_index()
    agg = agg.merge(means, on='topology_case', validate='one_to_one')
    for df in (steps, ep, agg, energy):
        for k, v in dict(controller=label, **extra).items():
            df[k] = v
    for name, df in [('step_metrics', steps), ('episode_summary', ep), ('aggregate_summary', agg),
                     ('terminal_soc_summary', energy)]:
        df.to_csv(out / f'{name}.csv', index=False)
    return ep


def select(spec):
    layout, label, block = spec.layout, spec.label, spec.layout.block
    training = training_record(spec)
    lambda_t, seed = run_block(training)['lambda_terminal'], 42 + spec.seed
    checkpoints = sorted(Path(training['effective_training_config']['checkpoint_dir']).glob('maddpg_step_*.pt'),
                         key=lambda p: int(p.stem.rsplit('_', 1)[-1]))
    if not checkpoints:
        raise SystemExit('No saved checkpoints to select from')
    sweep = spec.run_dir / 'selection_sweep'
    protocol = authorize(mode='val', final=False, run_purpose='checkpoint_selection', learned=True, topologies=['TP1'])
    for checkpoint in checkpoints:
        step = int(checkpoint.stem.rsplit('_', 1)[-1])
        out = sweep / f'step_{step}'
        if (out / 'run_manifest.json').is_file() and json.loads((out / 'run_manifest.json').read_text()).get('status') == 'finished':
            continue                                    # resumable: finished candidates are kept, never recomputed
        env = make_marl_env(protocol_config(feeder_config(spec.feeder, experiment=spec.experiment), protocol), 'val',
                            seed, lambda_t)
        starts, seeds, schedule = episode_plan(env, 'all', seed, protocol)
        run = feeder_tools(spec.feeder)[3](out, label, seed, 'val', checkpoint, spec.arguments)
        run.record_protocol(protocol, [POLICY])
        run.add_environment(env, intended_episodes=len(starts))
        run.record_schedule(schedule)
        run.data[block] = dict(controller=label, controller_key=spec.key, training_step=step,
                               **_sources(layout, spec.sources))
        if spec.key in LOCAL_REWARD_CONTROLLERS:
            run.data[block]['selection_reward_note'] = SELECTION_REWARD_NOTE
        agent = load_model(spec.key, checkpoint, env, spec.device, layout.legacy)
        run.data[block].update(feasibility_formulation=agent.feasibility_formulation,
                               checkpoint_schema=agent.checkpoint_schema, **verified_mapping(training, env, agent))
        frame = run_episodes(spec.feeder, env, agent, POLICY, 'TP1', zip(starts, seeds))
        write_tables(spec.feeder, out, [frame], {'TP1': env}, label, dict(run_id=spec.seed, training_step=step))
        run.finish()
    from scripts.select_checkpoint import main as select_original
    return select_original(['--feeder', spec.feeder, '--model', label, '--run', layout.run_key(spec.seed),
                            '--sweep_dir', str(sweep), '--output_root', str(layout.selections_dir(spec.feeder))])


def test(spec, scales):
    layout, label, block = spec.layout, spec.label, spec.layout.block
    training = training_record(spec)
    record = run_block(training)
    records = existing_selections(selection_key_dir(layout.selections_dir(spec.feeder), spec.feeder, label,
                                                    layout.run_key(spec.seed)))
    if not records:
        raise SystemExit('No frozen selection; run the select stage first')
    topologies = TOPOLOGIES[spec.feeder]
    protocol = authorize(mode='test', final=True, run_purpose='final_test', learned=True, topologies=topologies,
                         selection=str(records[-1]))
    checkpoint, seed = protocol['selection']['checkpoint_path'], 42 + spec.seed
    outs = []
    for scale in scales:
        out = spec.run_dir / 'test' / scale_label(scale)
        run = feeder_tools(spec.feeder)[3](out, label, seed, 'test', checkpoint, spec.arguments)
        run.record_protocol(protocol, [POLICY])
        run.data[block] = dict(controller=label, controller_key=spec.key, load_scaling_factor=scale,
                               pv_scaling_factor=1.0, scenario=scenario(scale), lambda_terminal=record['lambda_terminal'],
                               lambda_feasibility=record['lambda_feasibility'], **_sources(layout, spec.sources),
                               note='Frozen checkpoint, no retraining or fine-tuning on any evaluation topology.')
        if spec.key in LOCAL_REWARD_CONTROLLERS:
            run.data[block]['selection_reward_note'] = SELECTION_REWARD_NOTE
        frames, envs = [], {}
        for tp in topologies:
            cfg = protocol_config(topology_config(spec.feeder, tp, out / 'network_inputs', experiment=spec.experiment),
                                  protocol)
            env = make_marl_env(cfg, 'test', seed, record['lambda_terminal'], scale)
            starts, seeds, schedule = episode_plan(env, 'all', seed, protocol)
            if schedule is not None or len(starts) != 31:
                raise SystemExit('Expected 31 December test days')
            run.add_environment(env, topology=tp, scenario=scenario(scale), intended_episodes=len(starts))
            run.data.setdefault('episode_plans', {})[tp] = [dict(start_index=int(s), seed=int(z)) for s, z in zip(starts, seeds)]
            agent = load_model(spec.key, checkpoint, env, spec.device, layout.legacy)
            run.data[block].update(feasibility_formulation=agent.feasibility_formulation,
                                   checkpoint_schema=agent.checkpoint_schema, **verified_mapping(training, env, agent))
            frames.append(run_episodes(spec.feeder, env, agent, POLICY, tp, zip(starts, seeds)))
            envs[tp] = env
        write_tables(spec.feeder, out, frames, envs, label, dict(run_id=spec.seed, load_scaling_factor=scale))
        run.finish()
        print(f'{spec.done_prefix}TEST_DONE', out, flush=True)
        outs.append(out)
    return outs


def zero(feeder, experiment, layout, scales, *, sources=None, arguments=None, done_prefix=''):
    topologies = TOPOLOGIES[feeder]
    protocol = authorize(mode='test', final=True, run_purpose='final_test', learned=False, topologies=topologies)
    seed, outs = 42, []
    for scale in scales:
        out = layout.zero_dir(feeder) / scale_label(scale)
        run = feeder_tools(feeder)[3](out, 'ZERO', seed, 'test', None, dict(arguments or {}))
        run.record_protocol(protocol, ['zero'])
        run.data[layout.block] = dict(controller=ZERO_LABEL, load_scaling_factor=scale, pv_scaling_factor=1.0,
                                      backup='disabled for the zero-action reference only (evaluation.zero)',
                                      zero_dispatch_checks={}, **_sources(layout, sources))
        frames, envs = [], {}
        for tp in topologies:
            cfg = protocol_config(topology_config(feeder, tp, out / 'network_inputs', experiment=experiment), protocol)
            env = make_zero_action_env(feeder, cfg, 'test', seed, scale)
            starts, seeds, schedule = episode_plan(env, 'all', seed, protocol)
            run.add_environment(env, topology=tp, primary=False, scenario=scenario(scale), intended_episodes=len(starts))
            frame = run_episodes(feeder, env, None, 'zero', tp, zip(starts, seeds))
            run.data[layout.block]['zero_dispatch_checks'][tp] = assert_zero_dispatch(frame, env.num_agents)
            frames.append(frame)
            envs[tp] = env
        write_tables(feeder, out, frames, envs, ZERO_LABEL, dict(run_id=None, load_scaling_factor=scale))
        run.finish()
        print(f'{done_prefix}ZERO_DONE', out, flush=True)
        outs.append(out)
    return outs


def main(argv=None):
    from training.train import add_controller_arguments, resolve
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('stage', choices=['select', 'test', 'zero'])
    add_controller_arguments(p)
    for action in p._actions:                  # zero needs only --feeder
        if action.dest in ('controller', 'seed'):
            action.required = False
    p.add_argument('--scales', type=float, nargs='+', default=list(ALLOWED_SCALES))
    p.add_argument('--cpu', action='store_true')
    a = p.parse_args(argv)
    a.scales = [check_scale(s) for s in a.scales]
    if a.stage == 'zero':
        from configs.experiments import maintained_output_root, protocol_id
        try:
            root = maintained_output_root(a.protocol, a.output_root)
        except ValueError as exc:
            p.error(str(exc))
        set_seed(42)
        return zero(a.feeder, protocol_id(a.protocol), RunLayout(root), a.scales,
                    arguments=dict(vars(a), root=str(root)))
    if a.controller is None or a.seed is None:
        p.error(f'{a.stage} needs --controller and --seed')
    resolve(a, p)
    set_seed(42 + a.seed)
    spec = EvaluationSpec(a.feeder, a.key, a.seed, a.experiment, RunLayout(a.root), cpu=a.cpu,
                          arguments=dict(vars(a), root=str(a.root)))
    if a.stage == 'select':
        result = select(spec)
        print('SELECT_DONE', a.protocol, a.feeder, a.key, f'seed_{a.seed}', flush=True)
        return result
    return test(spec, a.scales)


if __name__ == '__main__':
    main()
