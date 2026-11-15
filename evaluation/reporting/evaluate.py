"""Strict frozen-policy evaluator (maintained: evaluation.reporting.evaluate). No update(), optimizer step, or training
imports. Accepts maintained (run_record, controller key) and historical (b3 block, b3_* label) training manifests.

Use --help. New result roots only; selection uses the existing TP1/S-SEL-v1 rule.
Original A1/A2 use their existing integrity/source-compatibility loaders unchanged.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path

from configs.experiments import LEGACY_LABELS, run_block
from .provenance import (CLASSES, environment_contract, formulation, model_shapes,
    require_equal, result_identity, sha, verify_environment, verify_finished)

CONTROLLERS = ('mlp_a0','mlp_a1','mlp_a2','mlp_a3','gcn','gat','tagconv')
# A3 exists only as corrected B3 (no original/post-tanh checkpoints). It is trained on individual local rewards but
# selected and evaluated here exactly like the others (make_marl_env: the same system-level reward and metrics).
B3_ONLY = ('mlp_a3',)
LOCAL_FEATURE_CONTROLLERS = ('mlp_a1','mlp_a3')
LOCAL_CRITIC_CONTROLLERS = ('mlp_a2','mlp_a3')


def require_in(value, allowed, what):
    if value not in allowed:
        raise ValueError(f'{what} mismatch: {value!r} not in {allowed!r}')


def load_agent(args, checkpoint, env, training):
    import torch
    device = 'cpu' if args.cpu or not torch.cuda.is_available() else 'cuda'
    ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
    require_equal(formulation(ck), args.experiment_class, 'checkpoint experiment class')
    historical = next(e for e in training['environments'] if e.get('mode') == 'train' and e.get('topology') == 'TP1')
    stored = ck.get('provenance') or {}
    for key in ('interface_identity','normalization_identity'):
        checkpoint_key = 'interface' if key == 'interface_identity' else 'normalization'
        require_equal(stored.get(checkpoint_key), historical.get(key), 'checkpoint/training '+key)
    require_equal(stored.get('input_files'), {k:v.get('sha256') for k,v in historical['input_files'].items()},
                  'checkpoint/training input files')
    require_equal(stored.get('configuration_sha256'), historical.get('configuration_sha256'), 'checkpoint/training config')
    config = training.get('effective_training_config', training.get('arguments', {}))
    if args.experiment_class != 'original_maddpg':
        from models.controllers import build_model
        record = run_block(training)
        labels = (args.controller, LEGACY_LABELS[args.controller])      # maintained key or historical b3_* label
        require_in(record.get('controller'), labels, 'training controller')
        require_in(ck.get('controller'), labels, 'checkpoint controller')
        require_equal(ck['lambda_feasibility'], record['lambda_feasibility'], 'feasibility coefficient')
        require_equal(ck.get('experiment_metadata', {}).get('lambda_terminal'), record['lambda_terminal'], 'terminal coefficient')
        model = build_model(args.controller, ck['lambda_feasibility'], **dict(ck['model_kwargs'], device=device))
        model.load(checkpoint, env)
    else:
        if ck.get('model_kwargs'):
            kwargs = dict(ck['model_kwargs'], device=device)
        else:
            # Original A0/GNN serialization predates model_kwargs. Constructor values
            # are reconstructed from training arguments; strict state_dict loading
            # below verifies layer dimensions, and manifests retain this evidence.
            kwargs = dict(agent_obs_dim=env.agent_obs_dim, num_agents=env.num_agents,
                          action_dim_per_agent=1, device=device, share_actor=False)
            for key in ('gamma','tau','actor_lr','critic_lr'):
                if key not in config:
                    raise ValueError('Missing original training parameter '+key)
                kwargs[key] = config[key]
            for key in ('actor_hidden_dims','critic_hidden_dims'):
                kwargs[key] = config.get(key, (256,256))
            if args.controller in ('gcn','gat','tagconv'):
                require_equal(config.get('gnn_type'), args.controller, 'original graph variant')
                kwargs.update(node_feature_dim=env.observation_space['x'].shape[-1], dess_buses=env.dess_buses,
                    gnn_type=args.controller, gnn_hidden_dim=config.get('gnn_hidden_dim',64),
                    gnn_embedding_dim=config.get('gnn_embedding_dim',64), gnn_num_layers=config.get('gnn_num_layers',3))
        context = nullcontext()
        if args.controller == 'mlp_a0':
            from experiments.no_gnn_maddpg.maddpg_mlp import MLPMADDPG as Model
        elif args.controller in ('mlp_a1','mlp_a2'):
            variant = 'no_shared_kpi' if args.controller == 'mlp_a1' else 'no_centralized_critic'
            if args.feeder == 'ieee69':
                from experiments.no_shared_kpi.ieee69.models import model_class
                Model = model_class(variant)
            else:
                from experiments.no_shared_kpi import model_support
                from experiments.no_shared_kpi.ieee33_eval_compat import compatibility_loader
                context = compatibility_loader(model_support)
                if args.controller == 'mlp_a1':
                    from experiments.no_shared_kpi.model_no_shared_kpi import NoSharedKPI as Model
                else:
                    from experiments.no_centralized_critic.model_independent_critic import IndependentAgents as Model
        else:
            from models.maddpg import GraphAwareMADDPG as Model
        with context:
            model = Model(**kwargs)
            model.load(checkpoint, env=env)
    model.actor.eval(); model.critics.eval()
    # Preserve full architecture, including non-parametric encoder variant/mask.
    architecture = model_shapes(ck)
    architecture['variant'] = args.controller
    architecture['actor_features'] = list(range(7 if args.controller in LOCAL_FEATURE_CONTROLLERS else 10))
    architecture['critic_scope'] = 'local' if args.controller in LOCAL_CRITIC_CONTROLLERS else 'joint'
    return model, architecture


def tables(out, frames, envs):
    import numpy as np
    import pandas as pd
    from .metrics import episode_metrics, summary
    from evaluation.evaluate import feeder_tools
    frame = pd.concat(frames, ignore_index=True)
    old_summary = feeder_tools('ieee'+str(next(iter(envs.values())).num_buses))[1]
    ep, agg = old_summary(frame)
    scientific = []
    for tp, env in envs.items():
        g = frame[frame.topology_case == tp]
        prices = {int(s): [env._get_profiles(int(s)+i)[3] for i in range(env.episode_length)] for s in g.start_index.unique()}
        scientific.append(episode_metrics(g, [vars(b.config) for b in env.batteries], prices,
                                          env.episode_length, env.delta_t_hours))
    sci = pd.concat(scientific, ignore_index=True)
    keys = ['topology_case','episode_id','start_index']
    ep = ep.merge(sci[keys+[c for c in sci if c not in ep]], on=keys, validate='one_to_one')
    summaries = pd.DataFrame([dict(topology_case=tp, **summary(g)) for tp,g in sci.groupby('topology_case')])
    agg = agg.merge(summaries, on='topology_case', validate='one_to_one', suffixes=('', '_scientific'))
    for name, df in [('step_metrics',frame),('episode_summary',ep),('aggregate_summary',agg),('scientific_episodes',sci)]:
        df.to_csv(out/(name+'.csv'), index=False)


def evaluate_checkpoint(args, training, checkpoint, out, stage, scale=1., selection=None):
    import pandas as pd
    from environments.marl_env import make_marl_env
    from configs.experiments import experiment_of, feeder_config, topology_config, TOPOLOGIES
    # Same experiment protocol as training (b3 for every original/historical manifest); the environment contract
    # below then fails closed on any battery/reward difference between the training and evaluation configuration.
    experiment = experiment_of(training)
    from utility.evaluation_protocol import authorize, protocol_config, episode_plan
    from utility.run_manifest import RunManifest
    from utility.provenance import input_files, interface_identity, normalization_identity
    from environments.primary_marl import effective_current_ratings_a
    from utility.feasibility import VOLTAGE_TOLERANCE_PU, CURRENT_RELATIVE_TOLERANCE
    from evaluation.evaluate import run_episodes
    from utility.run_manifest import serial
    from .metrics import summary
    historical = next(e for e in training['environments'] if e.get('mode') == 'train' and e.get('topology') == 'TP1')
    base = environment_contract(historical)
    mode = 'val' if stage == 'select' else 'test'
    topologies = ['TP1'] if mode == 'val' else TOPOLOGIES[args.feeder]
    protocol = authorize(mode=mode, final=mode=='test', run_purpose='checkpoint_selection' if mode=='val' else 'final_test',
                         learned=True, topologies=topologies, selection=selection, checkpoint=str(checkpoint))
    terminal = 0. if args.experiment_class == 'original_maddpg' else run_block(training)['lambda_terminal']

    def capture(env):
        # Same fields as RunManifest, without reserving a result during preflight.
        e = dict(feeder='IEEE'+str(env.num_buses), topology=(env.config.get('topology_case') or {}).get('case_name','TP1'),
            interface_identity=interface_identity(env), normalization_identity=normalization_identity(env),
            battery_constraints=[vars(b.config) for b in env.batteries], reward=env.reward_cfg,
            doe=dict(enabled=env.doe_enabled,w=env.doe_w,delta_c=env.doe_delta_c,pf_failure_mode=env.doe_pf_failure_mode,penalty_coef=env.doe_penalty_coef),
            backup=env.backup_cfg, network=env.config['network'], voltage_tolerance_pu=VOLTAGE_TOLERANCE_PU,
            current_relative_tolerance=CURRENT_RELATIVE_TOLERANCE, expected_steps_per_episode=env.episode_length,
            data_scaling=env.config['data_scaling'], input_files=input_files(env), active_current_ratings_a=effective_current_ratings_a(env).tolist())
        if env.delta_t_hours != .25 or env.episode_length != 96:
            raise ValueError('Expected 96 fifteen-minute decisions')
        return environment_contract(serial(e))

    baseline_env = make_marl_env(feeder_config(args.feeder, args.time_series, experiment), 'val',42+args.run,terminal)
    # Older manifests lack explicit timestep metadata. Recover it only from the
    # exact recorded configuration source, not from an assumed 96-step convention.
    config_path = Path(__file__).resolve().parents[2]/'configs'/(args.feeder+'_config.py')
    recorded_config = historical.get('source_files',{}).get('files',{}).get('configs/'+args.feeder+'_config.py')
    import hashlib
    raw = config_path.read_bytes()
    if historical.get('delta_t_hours') is not None:
        require_equal(historical['delta_t_hours'], baseline_env.delta_t_hours, 'training timestep')
    elif recorded_config not in (hashlib.sha256(raw).hexdigest(),hashlib.sha256(raw.replace(b'\r\n',b'\n')).hexdigest()):
        raise ValueError('Cannot verify historical timestep: configuration source hash changed and no timestep recorded')
    verify_environment(base, capture(baseline_env))  # no blanket exemption for base-line changes
    envs, contracts, plans, models = {}, {}, {}, {}
    schedule = None
    # Topology builder writes only NEW snapshots, not existing result/input files.
    # Existing finished output uses an isolated temporary snapshot directory.
    import tempfile
    with tempfile.TemporaryDirectory(prefix='strict-eval-topology-') as snapshots:
        for tp in topologies:
            cfg = (feeder_config(args.feeder,args.time_series,experiment) if tp=='TP1' else
                   topology_config(args.feeder,tp,Path(snapshots),args.time_series,experiment))
            env = make_marl_env(protocol_config(cfg,protocol),mode,42+args.run,terminal,scale)
            contract = capture(env); verify_environment(base,contract,scale,tp)
            starts,seeds,schedule = episode_plan(env,'all',42+args.run,protocol)
            if len(starts) != (61 if mode=='val' else 31):
                raise ValueError('Incomplete evaluation schedule')
            ts=env.time_series; dates=[]
            for s in starts:
                stamps=pd.to_datetime(ts['df'].loc[s:s+95,ts['datetime_col']])
                if len(stamps)!=96 or stamps.dt.date.nunique()!=1 or not stamps.diff().iloc[1:].eq(pd.Timedelta(minutes=15)).all():
                    raise ValueError('Not a complete fifteen-minute evaluation day')
                if mode=='test' and stamps.iloc[0].month != 12:
                    raise ValueError('Final test outside December')
                dates.append(str(stamps.iloc[0].date()))
            plans[tp]=[dict(start_index=int(s),seed=int(z),date=d) for s,z,d in zip(starts,seeds,dates)]
            contracts[tp]=contract; envs[tp]=env
            models[tp], shapes = load_agent(args,checkpoint,env,training)
        identity=result_identity(args.experiment_class,args.controller,args.feeder,args.run,checkpoint,args.training_manifest,
            shapes,dict(controller=args.controller,dt_hours=.25),
            dict(lambda_terminal=terminal,lambda_feasibility=run_block(training).get('lambda_feasibility',0.)),
            dict(stage=stage,schedule_id=schedule['schedule_id'] if schedule else 'December-31-days',load_scale=scale),
            contracts,plans,selection)
        if out.exists():
            verify_finished(out,identity); return
        historical_label = args.experiment_class != 'original_maddpg' and 'run_record' not in training
        run=RunManifest(out, LEGACY_LABELS[args.controller] if historical_label else args.controller,
                        42+args.run,mode,checkpoint,vars(args))
        run.record_protocol(protocol,['maddpg'])
        if schedule: run.record_schedule(schedule)
        run.data['scientific_identity']=identity
        run._write()
        frames=[]
        for tp,env in envs.items():
            run.add_environment(env,tp,scenario='production' if scale==1 else 'stress',intended_episodes=len(plans[tp]))
            # Retain actual topology bytes before the temporary builder directory disappears.
            import shutil
            dest=out/'network_inputs'/tp; dest.mkdir(parents=True)
            for key in ('line_data','gridtensor_line_data'):
                source=Path(env.config['paths'][key]); shutil.copyfile(source,dest/(key+'.csv'))
                # Point the persisted manifest at the retained snapshot, not a
                # TemporaryDirectory which disappears after this evaluation.
                run.data['environments'][-1]['input_files'][key]['path'] = str((dest/(key+'.csv')).resolve())
            run._write()
            f=run_episodes(args.feeder,env,models[tp],'maddpg',tp,[(p['start_index'],p['seed']) for p in plans[tp]])
            frames.append(f)
        tables(out,frames,envs)
        run.data['report_sha256']={name:sha(out/name) for name in ('step_metrics.csv','episode_summary.csv','aggregate_summary.csv','scientific_episodes.csv')}
        run.finish()


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=['select','test'])
    p.add_argument('--experiment-class',required=True,choices=CLASSES)
    p.add_argument('--controller',required=True,choices=CONTROLLERS)
    p.add_argument('--feeder',required=True,choices=['ieee33','ieee69'])
    p.add_argument('--run',type=int,required=True,choices=[0,1,2])
    p.add_argument('--training-manifest',type=Path,required=True)
    p.add_argument('--checkpoint-dir',type=Path)
    p.add_argument('--selection',type=Path)
    p.add_argument('--output-root',type=Path,required=True)
    p.add_argument('--time-series',help='Relocated identical bytes only; fingerprint is still checked')
    p.add_argument('--scales',nargs='+',type=float,default=[1.,1.1,1.2,1.3])
    p.add_argument('--cpu',action='store_true')
    a=p.parse_args(argv)
    if a.controller in B3_ONLY and a.experiment_class not in ('b3_pre_tanh', 'soc_feasible'):
        p.error(a.controller+' exists only as experiment class b3_pre_tanh or soc_feasible')
    training=json.loads(a.training_manifest.read_text(encoding='utf-8-sig'))
    if training.get('status') != 'finished': raise ValueError('Training is not finished')
    require_equal(training.get('seed'),42+a.run,'training seed/run')
    from configs.experiments import set_seed
    set_seed(42+a.run)
    root=a.output_root/a.feeder/a.experiment_class/a.controller/f'run_{a.run}'
    if a.stage=='select':
        if not a.checkpoint_dir: p.error('select requires --checkpoint-dir')
        checkpoints=sorted(a.checkpoint_dir.glob('maddpg_step_*.pt'),key=lambda x:int(x.stem.split('_')[-1]))
        if not checkpoints: raise ValueError('No checkpoints')
        for ck in checkpoints:
            evaluate_checkpoint(a,training,ck,root/'selection_sweep'/('step_'+ck.stem.split('_')[-1]),'select')
        from scripts.select_checkpoint import main as select
        select(['--feeder',a.feeder,'--model',a.experiment_class+'_'+a.controller,'--run',f'run_{a.run}',
                '--sweep_dir',str(root/'selection_sweep'),'--output_root',str(a.output_root/'selections')])
    else:
        if not a.selection: p.error('test requires --selection')
        from utility.evaluation_protocol import verify_selection
        selected=verify_selection(a.selection)
        record=json.loads(a.selection.read_text())
        require_equal(record.get('feeder'),a.feeder,'selection feeder')
        require_equal(record.get('run'),f'run_{a.run}','selection run')
        # Existing historical original selections may use their original labels;
        # checkpoint variant/state shapes and training provenance are still verified.
        for scale in a.scales:
            if scale not in (1.,1.1,1.2,1.3): raise ValueError('Unsupported stress scale')
            evaluate_checkpoint(a,training,Path(selected['checkpoint_path']),root/'test'/f'load_x{round(scale*100)}',
                                'test',scale,a.selection)


if __name__=='__main__': main()
