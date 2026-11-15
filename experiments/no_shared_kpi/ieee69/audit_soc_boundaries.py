"""Read-only IEEE33 A0 log audit. No environment/model imports or simulations.

python -m experiments.no_gnn_maddpg.audit_soc_boundaries --root results/topology_generalization/csv --output analysis/a0_soc_audit
Uses one CSV row at a time; retains counters, event histograms, step identities
and unique timestamp prices. Only policy == 'mlp_maddpg' is analyzed. Existing
output directories are refused. --self-test uses arithmetic and temporary
synthetic mixed-policy CSVs, not research data or power flow.
"""
import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

POLICY = 'mlp_maddpg'


def truth(x):
    return str(x).lower() in ('true', '1', '1.0')


def array(row, *names):
    for name in names:
        if row.get(name, '') not in ('', None):
            a = json.loads(row[name])
            if len(a) != 5 or not all(math.isfinite(float(v)) for v in a):
                raise ValueError(f'Invalid five-battery array: {name}')
            return list(map(float, a))
    raise ValueError(f'Missing required array: {names}')


def available(s, b, dt):
    charge = min(b['max_charge_kw'], max(0., b['soc_max']-s)*b['capacity_kwh']/b['charge_efficiency']/dt)
    discharge = min(b['max_discharge_kw'], max(0., s-b['soc_min'])*b['capacity_kwh']*b['discharge_efficiency']/dt)
    return charge, discharge


def quantile(x, q):
    x = sorted(x)
    z = (len(x)-1)*q
    i = int(z)
    return x[i] + (x[min(i+1, len(x)-1)]-x[i])*(z-i)


class Stats:
    def __init__(self):
        self.c = Counter()
    def add(self, **values):
        self.c['n'] += 1
        for k, v in values.items():
            self.c[k] += v
    def result(self):
        c = dict(self.c)
        n = c['n']
        out = dict(c)
        for k, v in c.items():
            if k != 'n':
                out['mean_'+k] = v/n
        out['retained_fraction'] = c.get('physical_abs_kw', 0)/c['requested_abs_kw'] if c.get('requested_abs_kw') else None
        for k in ('lower_outward', 'upper_outward', 'material_clip', 'unavailable_large_request'):
            out[k+'_percent_active'] = 100*c.get(k, 0)/c['active'] if c.get('active') else None
        return out


def records(path, topologies):
    """Filter before prices, duplicate checks, counters, or battery parsing."""
    with path.open(newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        if 'policy' not in (reader.fieldnames or []):
            raise ValueError(f'Missing required policy column: {path}')
        for row in reader:
            if row['policy'] == POLICY and row['topology_case'] in topologies:
                yield row


def step_identity(run, row):
    # The directory supplies the independent run ID, even when CSVs lack it.
    return (run, row['policy'], row['topology_case'], str(row['episode_id']), int(row['step']))


def price(row):
    # Only explicitly exported physical price, never normalized observations
    # or energy_cost/grid_import (undefined at zero import).
    for k in ('price', 'electricity_price', 'price_per_mwh'):
        if row.get(k, '') not in ('', None):
            v = float(row[k])
            if math.isfinite(v):
                return v
    return None


def write_table(path, rows):
    rows = list(rows)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    cols = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('x', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main(args):
    paths = [args.root/f'mlp_ieee33_tp_full_eval_run{r}_best' for r in range(3)]
    # All inputs validated before output creation.
    for p in paths:
        for name in ('step_metrics.csv', 'run_manifest.json'):
            if not (p/name).is_file():
                raise FileNotFoundError(p/name)
    if args.output.exists():
        raise FileExistsError(args.output)
    prices = {}
    seen = set()
    learned_counts = Counter()
    learned_episodes = defaultdict(set)
    episode_starts = {}
    for run, p in enumerate(paths):
        for row in records(p/'step_metrics.csv', args.topologies):
            ident = step_identity(run, row)
            if ident in seen:
                raise ValueError(f'Duplicate learned-policy step {ident}')
            seen.add(ident)
            episode = ident[:-1]
            start = str(row['start_index'])
            if episode in episode_starts and episode_starts[episode] != start:
                raise ValueError(f'Conflicting start_index within episode {episode}')
            episode_starts[episode] = start
            if ident[-1] >= 0:
                if ident[-1] >= 96:
                    raise ValueError(f'Unexpected step outside 0..95: {ident}')
                learned_counts[(run,row['topology_case'])] += 1
                learned_episodes[episode].add(ident[-1])
            v = price(row)
            if v is not None:
                key = row['date_time']
                if key in prices and abs(prices[key]-v) > 1e-6:
                    raise ValueError('Different physical prices for a matched timestamp')
                prices[key] = v
    cuts = [quantile(list(prices.values()), q) for q in (.25,.5,.75)] if prices else []
    stats = defaultdict(Stats)
    steps = defaultdict(Stats)
    events = Counter()
    episode_counts = Counter()
    excluded = Counter()
    max_error = 0.
    excessive_error = 0
    fingerprints = []
    for run, p in enumerate(paths):
        manifest = json.loads((p/'run_manifest.json').read_text(encoding='utf-8-sig'))
        fingerprints.append(dict(run=run, manifest_sha256=sha(p/'run_manifest.json'), csv_sha256=sha(p/'step_metrics.csv')))
        specs = {}
        for tp in args.topologies:
            envs = [e for e in manifest['environments'] if e['topology']==tp and e['mode']=='test']
            if not envs or any(e['battery_constraints'] != envs[0]['battery_constraints'] for e in envs):
                raise ValueError(f'Missing/conflicting battery specifications: {tp}')
            specs[tp] = envs[0]['battery_constraints']
            if len(specs[tp]) != 5 or any(b['soc_min'] != .1 or b['soc_max'] != .9 for b in specs[tp]):
                raise ValueError('Not the requested five-battery IEEE33 SOC configuration')
        streaks = {}
        previous = {}
        def close(key):
            n = streaks.pop(key, 0)
            if n:
                tp, ep, start, battery, kind = key
                for r in (str(run), 'all'):
                    for b in (str(battery), 'all'):
                        events[(tp,r,b,kind,n)] += 1
        for row in records(p/'step_metrics.csv', args.topologies):
            tp = row['topology_case']
            ep = (tp, row['episode_id'], row['start_index'])
            step = int(row['step'])
            # Duplicate validation already ran over filtered rows, with run and
            # policy explicitly included. Streak state is fresh for each run.
            if truth(row.get('episode_failed',False)) or row.get('dispatch_executed') and not truth(row['dispatch_executed']) or step < 0:
                excluded[(run,tp)] += 1
                for key in list(streaks):
                    if key[:3] == ep: close(key)
                previous.pop(ep,None)
                continue
            time = datetime.fromisoformat(row['date_time'].replace('Z','+00:00'))
            if time.month != 12:
                raise ValueError('Non-December evaluation row')
            if ep in previous:
                oldstep, oldtime = previous[ep]
                if step <= oldstep:
                    raise ValueError('Rows not chronological within episode')
                if step == oldstep+1 and (time-oldtime).total_seconds() != 900:
                    raise ValueError('Expected 15-minute timestep')
                if step != oldstep+1:
                    for key in list(streaks):
                        if key[:3] == ep: close(key)
            previous[ep] = (step,time)
            episode_counts[(run,*ep)] += 1
            s = array(row,'soc_before')
            a = array(row,'marl_raw_action','policy_proposal_action_normalized')
            req = array(row,'marl_requested_power_kw','policy_requested_power_kw')
            physical = array(row,'physical_limited_power_kw')
            final = array(row,'final_battery_power_kw','executed_battery_power_kw')
            total = sum(abs(x-y) for x,y in zip(req,physical))
            for r in (str(run),'all'):
                steps[(tp,r)].add(material_step=total>1, aggregate_clip_kw=total,
                    backup=truth(row.get('backup_activated',False)),
                    all_physical_zero=all(abs(x)<=1e-6 for x in physical))
            pv = price(row)
            pb = 'missing' if pv is None else 'Q'+str(1+sum(pv>c for c in cuts))
            for i,b in enumerate(specs[tp]):
                if not b['soc_min']-1e-6 <= s[i] <= b['soc_max']+1e-6:
                    raise ValueError('SOC outside manifest limits')
                ch,dis = available(s[i],b,.25)
                av = dis if req[i]>0 else ch
                expected = min(abs(req[i]),av) * (1 if req[i]>=0 else -1)
                err = abs(expected-physical[i])
                # Float32 exported SOC and action-power multiplication require
                # a separate reconstruction tolerance, not the event threshold.
                tol = max(.001, b['capacity_kwh']/.25/min(b['charge_efficiency'],b['discharge_efficiency'])*1e-6)
                max_error = max(max_error,err)
                excessive_error += err>tol
                low = s[i] <= b['soc_min']+args.soc_tolerance
                high = s[i] >= b['soc_max']-args.soc_tolerance
                active = abs(req[i])>1
                clip = abs(req[i]-physical[i])
                lower = low and req[i]>1
                upper = high and req[i]<-1
                unavailable = active and av<1
                region = 'lower' if low else 'upper' if high else 'interior'
                direction = 'discharge' if req[i]>1 else 'charge' if req[i]<-1 else 'idle'
                availbin = 'lt1' if av<1 else '1to10' if av<10 else '10plus'
                values = dict(active=active, lower=low, upper=high, interior=not(low or high),
                    lower_outward=lower, upper_outward=upper, material_clip=clip>1,
                    material_away_boundary=clip>1 and not(low or high),
                    unavailable_large_request=abs(req[i])>10 and av<1,
                    requested_abs_kw=abs(req[i]), physical_abs_kw=abs(physical[i]),
                    final_abs_kw=abs(final[i]), adjustment_kw=clip, raw_action=a[i],
                    raw_abs_action=abs(a[i]), soc=s[i], available_kw=av,
                    discharge_available_kw=dis, charge_available_kw=ch,
                    reconstruction_error_kw=err, discharge=req[i]>1, charge=req[i]<-1)
                conditions = [('overall','all'),('soc',region),('direction',direction),
                    ('soc_direction',region+'_'+direction),('availability_direction',availbin+'_'+direction),
                    ('price',pb),('hour',str(time.hour))]
                if pv is not None:
                    conditions.append(('negative_price',str(pv<0)))
                if lower: conditions.append(('saturation','lower_outward'))
                if upper: conditions.append(('saturation','upper_outward'))
                if clip>1 and not(low or high): conditions.append(('saturation','partial_interior'))
                for r in (str(run),'all'):
                    for bi in (str(i),'all'):
                        for typ,label in conditions:
                            stats[(tp,r,bi,typ,label)].add(**values)
                for kind,on in (('lower',low),('upper',high),('unavailable_discharge',unavailable and req[i]>1),('unavailable_charge',unavailable and req[i]<-1)):
                    key = (*ep,i,kind)
                    if on: streaks[key] = streaks.get(key,0)+1
                    else: close(key)
        for key in list(streaks): close(key)
    coverage = []
    for tp in args.topologies:
        for run in range(3):
            n = learned_counts[(run,tp)]
            episodes = [v for k,v in learned_episodes.items() if k[0]==run and k[2]==tp]
            valid = steps.get((tp,str(run)), Stats()).c['n']
            coverage.append(dict(topology=tp,run=str(run),policy=POLICY,
                learned_rows=n,expected_steps=2976,valid_steps=valid,
                episode_count=len(episodes),complete_episode_count=sum(v==set(range(96)) for v in episodes),
                counts_match=n==2976 and len(episodes)==31 and all(v==set(range(96)) for v in episodes),
                all_steps_valid=valid==2976))
        coverage.append(dict(topology=tp,run='all',policy=POLICY,
            learned_rows=sum(learned_counts[(r,tp)] for r in range(3)),expected_steps=8928,
            valid_steps=steps.get((tp,'all'), Stats()).c['n'],
            counts_match=all(x['counts_match'] for x in coverage if x['topology']==tp)))
    total_rows = sum(learned_counts.values())
    counts_match = all(x['counts_match'] for x in coverage)
    args.output.mkdir(parents=True, exist_ok=False)
    write_table(args.output/'policy_step_counts.csv', coverage)
    write_table(args.output/'conditional_battery_statistics.csv',
        (dict(zip(('topology','run','battery_zero_based','condition','bin'),k),**v.result()) for k,v in sorted(stats.items())))
    write_table(args.output/'step_summary.csv',
        (dict(topology=k[0],run=k[1],**v.result()) for k,v in sorted(steps.items())))
    write_table(args.output/'streak_histogram.csv',
        (dict(topology=k[0],run=k[1],battery_zero_based=k[2],kind=k[3],duration_steps=k[4],duration_hours=k[4]*.25,count=v)
         for k,v in sorted(events.items())))
    write_table(args.output/'episode_coverage.csv',
        (dict(run=k[0],topology=k[1],episode=k[2],start_index=k[3],valid_steps=v,complete_96=v==96) for k,v in sorted(episode_counts.items())))
    report = dict(inputs=fingerprints, policy=POLICY, soc_tolerance=args.soc_tolerance,
        learned_policy_rows=total_rows, expected_learned_policy_rows=8928*len(args.topologies),
        expected_counts_match=counts_match,
        active_threshold_kw=1, material_adjustment_threshold_kw=1, near_zero_available_kw=1,
        price_quartiles=cuts, price_units='physical exported price; verify manifest currency/MWh',
        price_quantiles_scope='descriptive December unique timestamps, not training thresholds',
        max_reconstruction_error_kw=max_error, reconstruction_failures=excessive_error,
        excluded_rows={str(k):v for k,v in excluded.items()},
        notes=['Means in battery table use battery-decisions in that condition; n is the denominator.',
               'Percent_active uses active decisions in the same condition; overall rows provide unconditional active denominator.',
               'Streaks never cross episode/run/battery boundaries; episode-edge streaks may be censored.',
               'Price ties use <= boundaries; quartiles may have unequal counts. Hour retains CSV timezone.',
               'soc_before is pre-action, physical is pre-backup, final is executed; these are not interchangeable.',
               'No actor observations are inferred from SOC logs. Actual actor-input runtime verification is separate.',
               'No causal or optimality inference follows from conditional means.'])
    (args.output/'audit_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    if not counts_match:
        print('WARNING: incomplete learned-policy coverage; inspect policy_step_counts.csv before pooling results')
    if excessive_error: raise SystemExit('RECONSTRUCTION FAILED: inspect errors before interpreting causes')


def self_test():
    b=dict(capacity_kwh=500,max_charge_kw=75,max_discharge_kw=75,soc_min=.1,soc_max=.9,charge_efficiency=.95,discharge_efficiency=.95)
    assert available(.1,b,.25)==(75,0)
    assert available(.9,b,.25)==(0,75)
    assert available(.5,b,.25)==(75,75)
    assert abs(available(.11,b,.25)[1]-19)<1e-10
    assert abs(available(.89,b,.25)[0]-500*.01/.95/.25)<1e-10
    assert quantile([0,1,2,3,4],.25)==1
    s=Stats(); s.add(active=True,requested_abs_kw=50,physical_abs_kw=10,material_clip=True)
    assert s.result()['retained_fraction']==.2
    assert s.result()['material_clip_percent_active']==100
    print('PASS: 8 synthetic arithmetic/statistics assertions; no simulations')
    policy_tests(b)


def policy_tests(b):
    """Full-size synthetic mixed-policy CSVs; never read research results."""
    import contextlib
    import io
    import tempfile
    from datetime import timedelta
    from types import SimpleNamespace
    from unittest.mock import patch

    # Fail closed if older exports do not identify their policies.
    with patch.object(Path, 'open', return_value=io.StringIO('topology_case,step\nTP1,0\n')):
        try:
            list(records(Path('synthetic.csv'), ['TP1']))
        except ValueError as exc:
            assert 'policy column' in str(exc)
        else:
            raise AssertionError('Missing policy column accepted')
    print('PASS: missing policy column rejected')

    # Temporary fixtures are explicitly contained within the working directory.
    workspace = Path.cwd().resolve()
    with tempfile.TemporaryDirectory(dir=workspace, prefix='synthetic_soc_policy_test_') as temp:
        root = Path(temp)
        assert root.resolve().is_relative_to(workspace)
        topologies = ['TP1','TP5','TP6']
        first = None
        fields = ['policy','topology_case','episode_id','start_index','step','date_time','price',
                  'soc_before','marl_raw_action','marl_requested_power_kw',
                  'physical_limited_power_kw','final_battery_power_kw','backup_activated']
        for run in range(3):
            folder = root/f'mlp_ieee33_tp_full_eval_run{run}_best'
            folder.mkdir()
            (folder/'run_manifest.json').write_text(json.dumps(dict(environments=[
                dict(topology=t,mode='test',battery_constraints=[b]*5) for t in topologies])), encoding='utf-8')
            with (folder/'step_metrics.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for tp in topologies:
                    for ep in range(31):
                        for step in range(96):
                            row = dict(policy=POLICY,topology_case=tp,episode_id=str(ep),
                                start_index=str(32064+96*ep),step=step,
                                date_time=str(datetime(2018,12,1)+timedelta(days=ep,minutes=15*step)),price=50,
                                soc_before=json.dumps([.1]*5),marl_raw_action=json.dumps([1]*5),
                                marl_requested_power_kw=json.dumps([75]*5),physical_limited_power_kw=json.dumps([0]*5),
                                final_battery_power_kw=json.dumps([0]*5),backup_activated=False)
                            if first is None: first = row.copy()
                            writer.writerow(row)
                            # Identical identities in other policies, with intentionally
                            # invalid arrays and different prices: none may be parsed.
                            for policy in ('random','zero'):
                                other = dict(row,policy=policy,price=-999,soc_before='not JSON')
                                writer.writerow(other)
        args = SimpleNamespace(root=root,output=root/'audit',topologies=topologies,soc_tolerance=1e-6)
        with contextlib.redirect_stdout(io.StringIO()): main(args)
        report = json.loads((args.output/'audit_manifest.json').read_text())
        assert report['expected_counts_match'] and report['learned_policy_rows']==26784
        assert report['price_quartiles']==[50,50,50] and report['reconstruction_failures']==0
        with (args.output/'policy_step_counts.csv').open() as f: counts = list(csv.DictReader(f))
        assert len(counts)==12
        assert all(int(x['learned_rows'])==(8928 if x['run']=='all' else 2976) for x in counts)
        with (args.output/'conditional_battery_statistics.csv').open() as f:
            overall = [r for r in csv.DictReader(f) if r['condition']=='overall' and r['run']=='all' and r['battery_zero_based']=='all']
        assert len(overall)==3 and all(int(r['n'])==44640 for r in overall)
        with (args.output/'streak_histogram.csv').open() as f: streaks = list(csv.DictReader(f))
        assert streaks and all(int(r['duration_steps'])==96 for r in streaks)
        assert all(int(r['count'])==93 for r in streaks if r['run']=='all' and r['battery_zero_based']!='all')
        print('PASS: mixed policies filtered before prices, arrays and counters')
        print('PASS: repeated episode/step IDs across runs accepted; 2976/run/topology, 8928/topology, 26784 total')
        print('PASS: battery denominators and streak isolation across episodes, batteries and runs')
        with (root/'mlp_ieee33_tp_full_eval_run0_best'/'step_metrics.csv').open('a',newline='') as f:
            csv.DictWriter(f,fieldnames=fields).writerow(first)
        args.output=root/'duplicate_must_fail'
        try:
            with contextlib.redirect_stdout(io.StringIO()): main(args)
        except ValueError as exc:
            assert 'Duplicate learned-policy step' in str(exc)
            assert "(0, 'mlp_maddpg', 'TP1', '0', 0)" in str(exc)
        else:
            raise AssertionError('Genuine same-run duplicate accepted')
        assert not args.output.exists()
        print('PASS: genuine same-run learned-policy duplicate rejected before output creation')


def runtime_soc_check():
    """Real MLP input hooks; synthetic battery states, no PF or checkpoint load.

    The irrelevant graph payload is stubbed explicitly. This checks the actual
    battery -> observation -> tensor -> individual actor path, not historical
    checkpoint behavior or whole-episode observation timing.
    """
    import importlib.util
    missing = [x for x in ('torch','gymnasium','scipy') if importlib.util.find_spec(x) is None]
    if missing:
        raise SystemExit('SKIPPED runtime SOC check: missing '+', '.join(missing))
    import numpy as np
    from types import SimpleNamespace
    from environments.battery import Battery, BatteryConfig
    from environments.dess_env import DESSEnv
    from utility.normalization import ObservationNormalizer
    from experiments.no_gnn_maddpg.maddpg_mlp import MLPMADDPG
    from configs.ieee33_config import IEEE33_CONFIG
    from configs.ieee69_config import IEEE69_CONFIG
    for feeder,cfg,n in ((33,IEEE33_CONFIG,5),(69,IEEE69_CONFIG,9)):
        for normalized in (False,True):
            env = DESSEnv.__new__(DESSEnv)
            env.num_agents=n
            env.agent_obs_dim=10
            env.dess_buses=list(range(n))  # synthetic positions, no network claim
            env.current_step=0
            env.normalize_observations=normalized
            env.normalizer=ObservationNormalizer()
            env.previous_actions=np.zeros(n)
            env.current_kpis=dict(grid_import_kw=100,voltage_deviation=.01,curtailment_kw=0)
            bcfg=BatteryConfig(**{k:v for k,v in cfg['dess'].items() if k not in ('dess_buses','num_dess')})
            env.batteries=[Battery(bcfg,.25) for _ in range(n)]
            env.graph_builder=SimpleNamespace(build_soc_vector=lambda s:s,
                build_graph=lambda **kw:dict(x=np.zeros((n,7)),edge_index=np.zeros((2,0)),edge_attr=np.zeros((0,2))))
            agent=MLPMADDPG(agent_obs_dim=10,num_agents=n,device='cpu',share_actor=False)
            captured={}
            hooks=[actor.register_forward_pre_hook(lambda module,inputs,i=i:captured.update({i:inputs[0].detach().cpu().numpy().copy()}))
                   for i,actor in enumerate(agent.actor.actors)]
            try:
                for values in (np.linspace(.1,.9,n),np.linspace(.9,.1,n)):
                    for b,v in zip(env.batteries,values): b.soc=float(v)
                    obs=env._build_observation(np.ones(n),np.zeros(n),50,np.ones(n))
                    agent.select_action(obs,noise_std=0)
                    for i in range(n):
                        assert abs(float(captured[i][0,2])-env.batteries[i].soc)<1e-6
            finally:
                for hook in hooks: hook.remove()
            print(f'PASS synthetic IEEE{feeder} dimensions, normalize={normalized}: actual MLP input hooks preserve current SOC')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('results/topology_generalization/csv'))
    p.add_argument('--output',type=Path)
    p.add_argument('--topologies',nargs='+',default=['TP1','TP5','TP6'])
    p.add_argument('--soc-tolerance',type=float,default=1e-6)
    p.add_argument('--self-test',action='store_true')
    p.add_argument('--runtime-soc-check',action='store_true')
    a=p.parse_args()
    if a.self_test: self_test()
    elif a.runtime_soc_check: runtime_soc_check()
    else:
        if a.output is None: p.error('--output required')
        if not 0<a.soc_tolerance<.01: p.error('SOC tolerance must be positive and less than .01')
        main(a)
