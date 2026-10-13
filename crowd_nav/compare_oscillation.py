#!/usr/bin/env python3
"""Compare oscillation metrics on successful and paired-success episodes."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = [
    ('action_switch_hz', 'Action switches', '1/s'),
    ('turn_reversal_hz', 'Turn-direction reversals', '1/s'),
    ('heading_flip_hz', 'Heading flips >= 90 deg', '1/s'),
    ('curvature_rad_per_m', 'Integrated curvature', 'rad/m'),
    ('lateral_sign_change_hz', 'Lateral-velocity sign changes', '1/s'),
]

FAILURE_AWARE_METRICS = [
    ('stalled_window_ratio', 'Stalled 2-s windows', 'ratio', 'lower'),
    (
        'oscillatory_stall_window_ratio',
        'Oscillatory stalled 2-s windows',
        'ratio',
        'lower',
    ),
    ('backtracking_ratio', 'Backward-motion steps', 'ratio', 'lower'),
    ('goal_progress_efficiency', 'Goal-progress efficiency', 'ratio', 'higher'),
]

ALL_METRIC_NAMES = [
    metric for metric, _, _ in METRICS
] + [
    metric for metric, _, _, _ in FAILURE_AWARE_METRICS
]


def load_rows(path):
    rows = []
    with open(path, newline='', encoding='utf-8') as handle:
        for row in csv.DictReader(handle):
            parsed = dict(row)
            parsed['scenario_index'] = int(row['scenario_index'])
            parsed['episode'] = int(row['episode'])
            parsed['seed'] = int(row['seed'])
            parsed['steps'] = int(row['steps'])
            for metric in ALL_METRIC_NAMES:
                parsed[metric] = float(row[metric])
            rows.append(parsed)
    if not rows:
        raise RuntimeError(f'No rows found in {path}')
    return rows


def row_key(row):
    return row['scenario'], row['episode'], row['seed']


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float('nan'), float('nan')
    return float(values.mean()), float(values.std(ddof=1)) if values.size > 1 else 0.0


def paired_bootstrap_ci(full_values, direct_values, samples=5000, seed=42):
    full_values = np.asarray(full_values, dtype=np.float64)
    direct_values = np.asarray(direct_values, dtype=np.float64)
    if full_values.size == 0:
        return float('nan'), float('nan')
    deltas = direct_values - full_values
    if deltas.size == 1:
        return float(deltas[0]), float(deltas[0])
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, deltas.size, size=(samples, deltas.size))
    means = deltas[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def scenario_order(rows):
    return [
        scenario
        for _, scenario in sorted({
            (row['scenario_index'], row['scenario']) for row in rows
        })
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', required=True, help='Full Mamba-VL oscillation CSV')
    parser.add_argument('--direct', required=True, help='Direct-action baseline oscillation CSV')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--full_label', default='Mamba-VL')
    parser.add_argument('--direct_label', default='Discrete-Mamba')
    args = parser.parse_args()

    full_rows = load_rows(args.full)
    direct_rows = load_rows(args.direct)
    scenarios = scenario_order(full_rows + direct_rows)

    full_by_key = {row_key(row): row for row in full_rows}
    direct_by_key = {row_key(row): row for row in direct_rows}
    common_keys = sorted(set(full_by_key) & set(direct_by_key))
    if not common_keys:
        raise RuntimeError('The two CSV files contain no paired evaluation cases')

    mismatched = [
        key for key in common_keys
        if full_by_key[key]['seed'] != direct_by_key[key]['seed']
    ]
    if mismatched:
        raise RuntimeError(f'Seed mismatch in {len(mismatched)} paired cases')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    success_rows = []
    paired_rows = []
    all_episode_rows = []
    report_lines = [
        '# Oscillation Comparison',
        '',
        'Primary analysis uses only cases where both policies succeed.',
        'Lower values indicate less oscillatory motion.',
        '',
        '## Evaluation Outcomes',
        '',
        '| Scenario | Full success | Direct success | Paired successes |',
        '|---|---:|---:|---:|',
    ]

    for scenario in scenarios + ['ALL']:
        if scenario == 'ALL':
            scope_keys = common_keys
        else:
            scope_keys = [key for key in common_keys if key[0] == scenario]

        full_scope = [full_by_key[key] for key in scope_keys]
        direct_scope = [direct_by_key[key] for key in scope_keys]
        full_success = [row for row in full_scope if row['outcome'] == 'success']
        direct_success = [row for row in direct_scope if row['outcome'] == 'success']
        paired_keys = [
            key for key in scope_keys
            if full_by_key[key]['outcome'] == 'success'
            and direct_by_key[key]['outcome'] == 'success'
        ]

        report_lines.append(
            f"| {scenario} | {len(full_success)}/{len(full_scope)} | "
            f"{len(direct_success)}/{len(direct_scope)} | {len(paired_keys)} |"
        )

        for model_label, rows in (
            (args.full_label, full_success),
            (args.direct_label, direct_success),
        ):
            for metric, display, unit in METRICS:
                mean, std = mean_std([row[metric] for row in rows])
                success_rows.append({
                    'scenario': scenario,
                    'cohort': 'all_successes',
                    'model': model_label,
                    'metric': metric,
                    'display': display,
                    'unit': unit,
                    'n': len(rows),
                    'mean': mean,
                    'std': std,
                })

        for metric, display, unit in METRICS:
            full_values = [full_by_key[key][metric] for key in paired_keys]
            direct_values = [direct_by_key[key][metric] for key in paired_keys]
            full_mean, full_std = mean_std(full_values)
            direct_mean, direct_std = mean_std(direct_values)
            delta_mean = (
                float(np.mean(np.asarray(direct_values) - np.asarray(full_values)))
                if paired_keys else float('nan')
            )
            ci_low, ci_high = paired_bootstrap_ci(full_values, direct_values)
            paired_rows.append({
                'scenario': scenario,
                'cohort': 'paired_successes',
                'metric': metric,
                'display': display,
                'unit': unit,
                'n': len(paired_keys),
                'full_mean': full_mean,
                'full_std': full_std,
                'direct_mean': direct_mean,
                'direct_std': direct_std,
                'direct_minus_full': delta_mean,
                'ci95_low': ci_low,
                'ci95_high': ci_high,
            })

        for metric, display, unit, preferred in FAILURE_AWARE_METRICS:
            full_values = [row[metric] for row in full_scope]
            direct_values = [row[metric] for row in direct_scope]
            full_mean, full_std = mean_std(full_values)
            direct_mean, direct_std = mean_std(direct_values)
            delta_mean = (
                float(np.mean(np.asarray(direct_values) - np.asarray(full_values)))
                if scope_keys else float('nan')
            )
            ci_low, ci_high = paired_bootstrap_ci(full_values, direct_values)
            all_episode_rows.append({
                'scenario': scenario,
                'cohort': 'all_paired_episodes',
                'metric': metric,
                'display': display,
                'unit': unit,
                'preferred': preferred,
                'n': len(scope_keys),
                'full_mean': full_mean,
                'full_std': full_std,
                'direct_mean': direct_mean,
                'direct_std': direct_std,
                'direct_minus_full': delta_mean,
                'ci95_low': ci_low,
                'ci95_high': ci_high,
            })

    with open(output_dir / 'oscillation_all_successes.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=success_rows[0].keys())
        writer.writeheader()
        writer.writerows(success_rows)

    with open(output_dir / 'oscillation_paired_summary.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=paired_rows[0].keys())
        writer.writeheader()
        writer.writerows(paired_rows)

    with open(output_dir / 'failure_aware_paired_summary.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=all_episode_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_episode_rows)

    report_lines.extend([
        '',
        '## Paired-Success Oscillation Metrics',
        '',
        f'Difference is {args.direct_label} minus {args.full_label}; positive values favor {args.full_label}.',
        '',
        '| Metric | N | Full mean | Direct mean | Difference | 95% paired bootstrap CI |',
        '|---|---:|---:|---:|---:|---:|',
    ])
    for row in paired_rows:
        if row['scenario'] != 'ALL':
            continue
        report_lines.append(
            f"| {row['display']} ({row['unit']}) | {row['n']} | "
            f"{row['full_mean']:.4f} | {row['direct_mean']:.4f} | "
            f"{row['direct_minus_full']:.4f} | "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] |"
        )

    report_lines.extend([
        '',
        '## Failure-Aware Motion Metrics',
        '',
        'This analysis includes every common evaluation case, including failures.',
        'A stalled window has less than 0.2 m forward progress over 2 s. '
        'It is oscillatory when it also contains at least two lateral or turn-direction reversals.',
        '',
        '| Metric | Preferred | N | Full mean | Direct mean | Direct - Full | 95% paired bootstrap CI |',
        '|---|---:|---:|---:|---:|---:|---:|',
    ])
    for row in all_episode_rows:
        if row['scenario'] != 'ALL':
            continue
        report_lines.append(
            f"| {row['display']} ({row['unit']}) | {row['preferred']} | {row['n']} | "
            f"{row['full_mean']:.4f} | {row['direct_mean']:.4f} | "
            f"{row['direct_minus_full']:.4f} | "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] |"
        )

    with open(output_dir / 'oscillation_report.md', 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(report_lines) + '\n')

    print(f"Paired cases: {len(common_keys)}")
    print(f"Report: {output_dir / 'oscillation_report.md'}")
    print(f"Summary: {output_dir / 'oscillation_paired_summary.csv'}")
    print(f"Failure-aware: {output_dir / 'failure_aware_paired_summary.csv'}")


if __name__ == '__main__':
    main()
