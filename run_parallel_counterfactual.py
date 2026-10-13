#!/usr/bin/env python3
"""Parallel Runner for Phase P1 Counterfactual Study on Single-Stream GRU (B0)

Divides 200 cases (90000 - 90199) across 4 parallel subprocesses on GPU/CPU,
monitors execution, merges results into the unified dataset, and performs
the full empirical counterfactual audit across archetypes.
"""

import sys
import os
import json
import time
import subprocess
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / 'benchmark_results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FINAL_DATASET = RESULTS_DIR / 'b0_counterfactual_study_200cases.json'

def main():
    start_case = 90000
    total_cases = 200
    num_workers = 4
    cases_per_worker = total_cases // num_workers

    chunks = []
    for w in range(num_workers):
        w_start = start_case + w * cases_per_worker
        w_num = cases_per_worker
        w_out = RESULTS_DIR / f'b0_counterfactual_part{w}.json'
        chunks.append({
            'worker_id': w,
            'start': w_start,
            'num': w_num,
            'out_file': w_out
        })

    print("=" * 85)
    print(f"LAUNCHING PHASE P1: 200-CASE B0 COUNTERFACTUAL STUDY ({num_workers} PARALLEL WORKERS)")
    print("=" * 85)
    for c in chunks:
        print(f"Worker {c['worker_id']}: Cases {c['start']} to {c['start'] + c['num'] - 1} -> {c['out_file'].name}")
    print("-" * 85)

    processes = []
    t_start = time.time()
    for c in chunks:
        cmd = [
            sys.executable,
            str(REPO_ROOT / 'run_b0_counterfactual_study.py'),
            '--start', str(c['start']),
            '--num', str(c['num']),
            '--device', 'cuda',
            '--out_file', str(c['out_file'])
        ]
        log_f = open(RESULTS_DIR / f'worker_{c["worker_id"]}.log', 'w')
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT)
        processes.append((c, proc, log_f))

    print(f"All {num_workers} workers spawned. Monitoring progress...")

    # Wait for all processes to finish
    for c, proc, log_f in processes:
        proc.wait()
        log_f.close()
        ret = proc.returncode
        wid = c['worker_id']
        print(f"Worker {wid} (Cases {c['start']}-{c['start']+c['num']-1}) finished with exit code {ret}.")
        if ret != 0:
            log_path = RESULTS_DIR / f"worker_{wid}.log"
            print(f"ERROR: Worker {wid} failed! Check {log_path}")

    # Merge results
    merged_data = []
    for c in chunks:
        if c['out_file'].exists():
            with open(c['out_file'], 'r') as f:
                data = json.load(f)
                merged_data.extend(data)
        else:
            print(f"WARNING: Output file {c['out_file']} missing!")

    # Sort strictly by case_id
    merged_data.sort(key=lambda x: x['case_id'])

    with open(FINAL_DATASET, 'w') as f:
        json.dump(merged_data, f, indent=2)

    t_total = time.time() - t_start
    print("\n" + "=" * 95)
    print(f"PHASE P1 COMPLETE: 200 CASES HARVESTED & MERGED IN {t_total:.1f}s ({t_total/60:.2f} mins)")
    print(f"Saved master dataset to: {FINAL_DATASET}")
    print("=" * 95)

    # Auditing results
    all_snaps = [s for c in merged_data for s in c['snapshots']]
    parity_failures = [
        s['snapshot_id'] for s in all_snaps
        if not s['branch_A_base_k0'].get('parity_pass', False)
    ]

    print(f"Total Cases: {len(merged_data)}")
    print(f"Total Snapshots: {len(all_snaps)}")
    print(f"Base Trajectory Parity Check: {len(all_snaps) - len(parity_failures)}/{len(all_snaps)} passed")
    if parity_failures:
        print(f"WARNING: Parity failed on {len(parity_failures)} snapshots: {parity_failures[:5]}")

    print("\n" + "-" * 95)
    print(f"{'Archetype':<20} | {'N':<5} | {'Base SR':<8} | {'K=2 SR':<8} | {'K=4 SR':<8} | {'K=8 SR':<8} | {'Gain(K=2)':<10} | {'Gain(K=4)':<10} | {'Gain(K=8)':<10}")
    print("-" * 95)

    archetypes = sorted(list(set(s['archetype'] for s in all_snaps)))
    for arch in archetypes:
        arch_snaps = [s for s in all_snaps if s['archetype'] == arch]
        n = len(arch_snaps)
        sr_a = np.mean([1 if s['branch_A_base_k0']['outcome'] == 'SUCCESS' else 0 for s in arch_snaps]) * 100
        sr_b = np.mean([1 if s['branch_B_dwell_k2']['outcome'] == 'SUCCESS' else 0 for s in arch_snaps]) * 100
        sr_c = np.mean([1 if s['branch_C_dwell_k4']['outcome'] == 'SUCCESS' else 0 for s in arch_snaps]) * 100
        sr_d = np.mean([1 if s['branch_D_dwell_k8']['outcome'] == 'SUCCESS' else 0 for s in arch_snaps]) * 100

        avg_g2 = np.mean([s['branch_B_dwell_k2']['gain_vs_base'] for s in arch_snaps])
        avg_g4 = np.mean([s['branch_C_dwell_k4']['gain_vs_base'] for s in arch_snaps])
        avg_g8 = np.mean([s['branch_D_dwell_k8']['gain_vs_base'] for s in arch_snaps])

        print(f"{arch:<20} | {n:<5} | {sr_a:6.1f}% | {sr_b:6.1f}% | {sr_c:6.1f}% | {sr_d:6.1f}% | {avg_g2:+8.2f}   | {avg_g4:+8.2f}   | {avg_g8:+8.2f}")
    print("-" * 95)

if __name__ == '__main__':
    main()
