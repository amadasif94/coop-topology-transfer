from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

ROOT = Path('results/b4/ieee33')
OUT = ROOT / 'analysis'

TOPS = [f'TP{i}' for i in range(1, 8)]
LOADS = [100, 110, 120, 130]

MODELS = [
    'zero', 'random',
    'mlp_a0', 'mlp_a1', 'mlp_a2', 'mlp_a3'
]

METRICS = [
    'adjusted_cost_usd_day',
    'raw_cost_usd_day',
    'savings_vs_zero_usd_day',
    'gap_vs_tp1_usd_day',
    'savings_retention_vs_tp1_usd_day',
    'voltage_deviation',
    'worst_min_voltage_pu',
    'network_violation_pct',
    'zero_network_violation_pct',
    'backup_activation_pct',
    'power_removed_pct',
    'soc_shortfall_pp'
]

WARNED = set()


def warn(message):
    if message not in WARNED:
        WARNED.add(message)
        print('WARNING:', message)


def path_for(model, run, load, filename):
    if model == 'zero':
        folder = (
            ROOT / 'zero_action_no_backup'
            / f'load_x{load}'
        )
    elif model == 'random':
        folder = (
            ROOT / 'random_action'
            / f'run_{run}' / 'test'
            / f'load_x{load}'
        )
    else:
        folder = (
            ROOT / f'b3_{model}'
            / f'run_{run}' / 'test'
            / f'load_x{load}'
        )

    return folder / filename


def number(df, field, required=False):
    if field not in df:
        if required:
            raise ValueError(
                f'Required column missing: {field}'
            )
        return pd.Series(
            np.nan, index=df.index, dtype=float
        )

    return pd.to_numeric(
        df[field], errors='coerce'
    )


def episodes(model, run, load):
    file = path_for(
        model, run, load, 'episode_summary.csv'
    )

    if not file.is_file():
        raise FileNotFoundError(
            f'Incomplete evaluation: {file}'
        )

    df = pd.read_csv(file)

    need = [
        'topology_case',
        'start_index',
        'adjusted_energy_cost',
        'raw_energy_cost',
        'mean_voltage_deviation'
    ]

    for field in need:
        if field not in df:
            raise ValueError(
                f'{file}: missing {field}'
            )

    counts = (
        df.groupby('topology_case')
        .size()
        .to_dict()
    )

    expected = {t: 31 for t in TOPS}

    if counts != expected:
        raise ValueError(
            f'{file}: invalid 31-day coverage: {counts}'
        )

    if df.duplicated([
        'topology_case', 'start_index'
    ]).any():
        raise ValueError(
            f'{file}: repeated topology/day keys'
        )

    if 'economic_totals_valid' in df:
        valid = (
            df['economic_totals_valid']
            .astype(str)
            .str.lower()
            .isin(['true', '1'])
        )

        if not valid.all():
            raise ValueError(
                f'{file}: invalid economic totals'
            )

    for field in [
        'adjusted_energy_cost',
        'raw_energy_cost',
        'mean_voltage_deviation'
    ]:
        df[field] = number(
            df, field, required=True
        )

        if not np.isfinite(
            df[field].to_numpy()
        ).all():
            raise ValueError(
                f'{file}: nonfinite {field}'
            )

    # Check the stored-energy cost formula
    # whenever its input fields are available.
    terms = [
        'initial_stored_energy_kwh',
        'terminal_stored_energy_kwh',
        'day_mean_price_per_mwh'
    ]

    if all(f in df for f in terms):
        initial, terminal, price = (
            number(df, f) for f in terms
        )

        expected_cost = (
            df['raw_energy_cost']
            - price * (terminal - initial) / 1000.0
        )

        if (
            not np.isfinite(expected_cost).all()
            or not np.allclose(
                df['adjusted_energy_cost'],
                expected_cost,
                rtol=1e-8,
                atol=1e-4
            )
        ):
            raise ValueError(
                f'{file}: inventory cost does not '
                'reproduce source formula'
            )
    else:
        warn(
            'Cost formula cannot be independently '
            'rechecked in every file: some '
            'inventory fields missing'
        )

    return df


def weighted_rate(
    frame, numerator, denominator, label
):
    if (
        numerator not in frame
        or denominator not in frame
    ):
        warn(
            f'{label}: missing {numerator} or '
            f'{denominator}; reporting NA'
        )
        return np.nan

    n = number(frame, numerator)
    d = number(frame, denominator)

    if (
        not np.isfinite(n).all()
        or not np.isfinite(d).all()
        or (d < 0).any()
        or (n < 0).any()
        or (n > d).any()
        or d.sum() <= 0
    ):
        warn(
            f'{label}: invalid step-count '
            'denominators; reporting NA'
        )
        return np.nan

    return 100.0 * n.sum() / d.sum()


def soc_shortfalls(model, run, load, ep):
    # SOC shortfall is calculated in percentage
    # points from explicit 0-to-1 SOC records.
    source = ep

    if not any(
        c.startswith('initial_soc_agent_')
        for c in source
    ):
        file = path_for(
            model, run, load,
            'terminal_soc_summary.csv'
        )

        if not file.is_file():
            warn(
                f'{model}: no per-agent initial/'
                'terminal SOC records; '
                'SOC shortfall in pp unavailable'
            )
            return pd.Series(
                np.nan, index=ep.index
            )

        terminal = pd.read_csv(file)

        keys = [
            'topology_case',
            'start_index'
        ]

        if not all(
            k in terminal for k in keys
        ):
            warn(
                f'{model}: terminal_soc_summary '
                'lacks topology/start_index '
                'for verified alignment'
            )
            return pd.Series(
                np.nan, index=ep.index
            )

        if terminal.duplicated(keys).any():
            raise ValueError(
                f'{file}: duplicated terminal-SOC '
                'alignment keys'
            )

        source = ep[keys].merge(
            terminal,
            on=keys,
            how='left',
            validate='one_to_one',
            sort=False
        )

    ids = sorted(
        int(m.group(1))
        for col in source.columns
        if (
            m := re.fullmatch(
                r'initial_soc_agent_(\d+)', col
            )
        )
        and (
            f'terminal_soc_agent_{m.group(1)}'
            in source
        )
    )

    if not ids:
        warn(
            f'{model}: no matched per-agent '
            'initial/terminal SOC columns; '
            'SOC shortfall pp unavailable'
        )
        return pd.Series(
            np.nan, index=ep.index
        )

    a = source[
        [f'initial_soc_agent_{i}' for i in ids]
    ].apply(
        pd.to_numeric, errors='coerce'
    ).to_numpy()

    b = source[
        [f'terminal_soc_agent_{i}' for i in ids]
    ].apply(
        pd.to_numeric, errors='coerce'
    ).to_numpy()

    if (
        not np.isfinite(a).all()
        or not np.isfinite(b).all()
        or (a < -1e-6).any()
        or (a > 1 + 1e-6).any()
        or (b < -1e-6).any()
        or (b > 1 + 1e-6).any()
    ):
        warn(
            f'{model}: SOC values not verified '
            'on 0-to-1 scale; reporting NA'
        )
        return pd.Series(
            np.nan, index=ep.index
        )

    values = (
        100.0
        * np.maximum(a - b, 0)
        .mean(axis=1)
    )

    return pd.Series(
        values, index=ep.index
    )


def parse_vector(value):
    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        return None

    try:
        vector = np.asarray(
            json.loads(value),
            dtype=float
        ).reshape(-1)

    except (ValueError, TypeError):
        return None

    if (
        vector.size == 0
        or not np.isfinite(vector).all()
    ):
        return None

    return vector


def clipping_by_topology(model, run, load):
    file = path_for(
        model, run, load, 'step_metrics.csv'
    )

    if not file.is_file():
        warn(
            f'{model}: missing step_metrics.csv; '
            'power removed reported NA'
        )
        return {}

    headers = pd.read_csv(
        file, nrows=0
    ).columns

    fields = [
        'topology_case',
        'policy_requested_power_kw',
        'physical_limited_power_kw'
    ]

    if not all(
        c in headers for c in fields
    ):
        warn(
            f'{model}: step_metrics missing '
            'requested or limited battery power; '
            'reporting NA'
        )
        return {}

    total_req = {
        t: 0.0 for t in TOPS
    }

    total_removed = {
        t: 0.0 for t in TOPS
    }

    counts = {
        t: 0 for t in TOPS
    }

    failed = {
        t: 0 for t in TOPS
    }

    # Read only the required columns using Python's CSV reader.
    # Avoids the Pandas C-parser memory error on wide step CSVs.
    def stream_chunks():
        import csv

        with file.open(newline='', encoding='utf-8') as fh:
            reader = csv.reader(fh)
            columns = next(reader)
            indices = [columns.index(name) for name in fields]

            batch = []

            for row in reader:
                batch.append(tuple(row[i] for i in indices))

                if len(batch) >= 500:
                    yield pd.DataFrame(batch, columns=fields)
                    batch = []

            if batch:
                yield pd.DataFrame(batch, columns=fields)

    chunks = stream_chunks()

    for chunk in chunks:
        for tp, requested, limited in (
            chunk[fields].itertuples(
                index=False, name=None
            )
        ):
            if tp not in total_req:
                raise ValueError(
                    f'{file}: unknown topology {tp}'
                )

            counts[tp] += 1

            req = parse_vector(requested)
            lim = parse_vector(limited)

            if (
                req is None
                or lim is None
                or req.shape != lim.shape
            ):
                failed[tp] += 1
                continue

            total_req[tp] += float(
                np.abs(req).sum()
            )

            total_removed[tp] += float(
                np.maximum(
                    np.abs(req) - np.abs(lim),
                    0
                ).sum()
            )

    result = {}

    for tp in TOPS:
        if (
            failed[tp]
            or counts[tp] != 31 * 96
            or total_req[tp] <= 0
        ):
            warn(
                f'{model} load {load/100:.1f}x '
                f'{tp}: incomplete/zero requested '
                'power; power removed NA'
            )

            result[tp] = (
                np.nan, np.nan
            )
        else:
            result[tp] = (
                total_removed[tp],
                total_req[tp]
            )

    return result


def summarize_run(
    model, run, load, ep, zero, clipping
):
    zlookup = zero.set_index([
        'topology_case',
        'start_index'
    ])['adjusted_energy_cost']

    zkeys = pd.MultiIndex.from_frame(
        ep[['topology_case', 'start_index']]
    )

    if not zkeys.isin(
        zlookup.index
    ).all():
        raise ValueError(
            f'{model} run {run}: '
            'Zero Action dates do not align'
        )

    ep = ep.copy()

    ep['_zero_cost'] = (
        zlookup.reindex(zkeys).to_numpy()
    )

    ep['_saving'] = (
        ep['_zero_cost']
        - ep['adjusted_energy_cost']
    )

    ep['_soc_shortfall_pp'] = (
        soc_shortfalls(
            model, run, load, ep
        )
    )

    baseline = (
        ep[ep.topology_case == 'TP1']
        .set_index('start_index')
    )

    if not baseline.index.is_unique:
        raise ValueError(
            'TP1 daily keys are duplicated'
        )

    rows = []

    for tp in TOPS + ['ALL']:
        if tp == 'ALL':
            g = ep
        else:
            g = ep[
                ep.topology_case == tp
            ]

        if tp in TOPS:
            aligned = g.set_index(
                'start_index'
            )

            if not aligned.index.isin(
                baseline.index
            ).all():
                raise ValueError(
                    f'Unpaired dates in {model}/{tp}'
                )

            tp1 = baseline.reindex(
                aligned.index
            )

            gap = float((
                aligned.adjusted_energy_cost
                - tp1.adjusted_energy_cost
            ).mean())

            retention = float((
                aligned._saving
                - tp1._saving
            ).mean())
        else:
            # ALL includes TP1 and therefore
            # has no single-topology gap.
            gap = np.nan
            retention = np.nan

        if tp == 'ALL':
            pair = [
                clipping.get(
                    t, (np.nan, np.nan)
                )
                for t in TOPS
            ]
        else:
            pair = [
                clipping.get(
                    tp, (np.nan, np.nan)
                )
            ]

        if (
            pair
            and all(
                np.isfinite(x)
                and np.isfinite(y)
                for x, y in pair
            )
        ):
            num, den = map(
                sum, zip(*pair)
            )

            clip = (
                100.0 * num / den
                if den > 0
                else np.nan
            )
        else:
            clip = np.nan

        vmin = number(
            g, 'min_voltage_pu'
        )

        if not np.isfinite(
            vmin
        ).all():
            warn(
                f'{model}: min_voltage_pu '
                'unavailable or nonfinite; '
                'reporting NA'
            )
            worst = np.nan
        else:
            worst = float(
                vmin.min()
            )

        rows.append(dict(
            load=load / 100,
            controller=model.upper(),
            run=run,
            topology=tp,
            n_episodes=len(g),

            adjusted_cost_usd_day=float(
                g.adjusted_energy_cost.mean()
            ),

            raw_cost_usd_day=float(
                g.raw_energy_cost.mean()
            ),

            savings_vs_zero_usd_day=float(
                g._saving.mean()
            ),

            gap_vs_tp1_usd_day=gap,

            savings_retention_vs_tp1_usd_day=retention,

            voltage_deviation=float(
                g.mean_voltage_deviation.mean()
            ),

            worst_min_voltage_pu=worst,

            network_violation_pct=weighted_rate(
                g,
                'network_violation_steps',
                'network_violation_measured_steps',
                model
            ),

            backup_activation_pct=weighted_rate(
                g,
                'backup_activation_count',
                'backup_activation_measured_steps',
                model
            ),

            power_removed_pct=clip,

            soc_shortfall_pp=(
                float(
                    g._soc_shortfall_pp.mean()
                )
                if g._soc_shortfall_pp.notna().all()
                else np.nan
            )
        ))

    return rows


def main():
    records = []

    for load in LOADS:
        zero = episodes(
            'zero', 0, load
        )

        for model in MODELS:
            runs = (
                [0]
                if model == 'zero'
                else range(3)
            )

            for run in runs:
                if model == 'zero':
                    ep = zero
                else:
                    ep = episodes(
                        model, run, load
                    )

                if model == 'zero':
                    clipping = {}
                else:
                    clipping = clipping_by_topology(
                        model, run, load
                    )

                records.extend(
                    summarize_run(
                        model,
                        run,
                        load,
                        ep,
                        zero,
                        clipping
                    )
                )

    runs = pd.DataFrame(records)

    zero_violation = (
        runs[
            runs.controller == 'ZERO'
        ]
        .set_index([
            'load', 'topology'
        ])['network_violation_pct']
    )

    for idx, row in runs.iterrows():
        key = (
            row['load'],
            row['topology']
        )

        runs.loc[
            idx,
            'zero_network_violation_pct'
        ] = zero_violation.loc[key]

    summary = []

    grouped = runs.groupby(
        ['load', 'controller', 'topology'],
        sort=False
    )

    for (load, model, tp), group in grouped:
        row = dict(
            load=load,
            controller=model,
            topology=tp,
            n_runs=group.run.nunique()
        )

        for metric in METRICS:
            x = group[
                metric
            ].to_numpy(dtype=float)

            valid = np.isfinite(
                x
            ).all()

            row[
                metric + '_mean'
            ] = (
                float(np.mean(x))
                if valid
                else np.nan
            )

            row[
                metric + '_sd'
            ] = (
                float(
                    np.std(x, ddof=1)
                )
                if len(x) > 1 and valid
                else np.nan
            )

            row[
                metric + '_ci95_halfwidth'
            ] = (
                4.302652729
                * row[metric + '_sd']
                / np.sqrt(3)
                if len(x) == 3 and valid
                else np.nan
            )

        summary.append(row)

    result = pd.DataFrame(summary)

    OUT.mkdir(
        parents=True, exist_ok=True
    )

    runs.to_csv(
        OUT / 'ieee33_b4_all_metrics_by_run.csv',
        index=False
    )

    result.to_csv(
        OUT / 'ieee33_b4_all_metrics_three_seed_summary.csv',
        index=False
    )

    for load in LOADS:
        subset = result[
            np.isclose(
                result.load, load / 100
            )
        ]

        subset.to_csv(
            OUT / f'ieee33_b4_all_metrics_load_x{load}.csv',
            index=False
        )

        print(
            f'\n====== IEEE33 load '
            f'{load/100:.1f}x '
            '— per-topology results ======'
        )

        cols = [
            'topology',
            'controller',
            'adjusted_cost_usd_day_mean',
            'adjusted_cost_usd_day_sd',
            'savings_vs_zero_usd_day_mean',
            'gap_vs_tp1_usd_day_mean',
            'voltage_deviation_mean',
            'worst_min_voltage_pu_mean',
            'network_violation_pct_mean',
            'zero_network_violation_pct_mean',
            'backup_activation_pct_mean',
            'power_removed_pct_mean',
            'soc_shortfall_pp_mean'
        ]

        print(
            subset[cols]
            .round(3)
            .to_string(
                index=False,
                na_rep='NA'
            )
        )

    print(
        '\nSaved 6 CSV files under:',
        OUT
    )

    if WARNED:
        print(
            f'NOTICE: {len(WARNED)} distinct '
            'data-quality warnings '
            'were printed above.'
        )
    else:
        print(
            'All requested source fields '
            'passed validation.'
        )


if __name__ == '__main__':
    main()
