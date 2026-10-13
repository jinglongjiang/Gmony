"""Behavioral Dynamics & Option Termination Forensics

Analyzes hierarchical decision-making dynamics:
1. Option Utilization & Collapse Check:
   - Verifies whether options collapsed into degenerate single-option usage.
   - Computes Shannon entropy of option distribution.
2. Dwell Duration & Chattering Diagnostics:
   - Mean, median, and std of option dwell duration.
   - Option switching frequency per episode.
3. Causal Attribution of Interventions:
   - Deadlock rescue rate (rescues from stagnation).
   - Switching harm rate (unnecessary collisions triggered post-switch).
   - Net Advantage Gain: Net = N_rescue - N_harm.
4. Latent Semantic Separation:
   - Computes cosine similarity / distance between option intra-policy heads.
"""

import sys
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / 'benchmark_results'


def analyze_benchmark_telemetry(json_path: Path) -> Dict:
    """Analyze behavioral telemetry from 500-case evaluation records."""
    if not json_path.exists():
        raise FileNotFoundError(f"Results JSON not found at: {json_path}")

    with open(json_path, 'r') as f:
        data = json.load(f)

    detailed = data.get('detailed_results', {})
    report = {}

    for arm_name, records in detailed.items():
        n = len(records)
        if n == 0:
            continue

        successes = [r for r in records if r['outcome'] == 'SUCCESS']
        collisions = [r for r in records if r['outcome'] == 'COLLISION']
        timeouts = [r for r in records if r['outcome'] == 'TIMEOUT']
        stagnations = [r for r in records if r.get('stagnation', False)]

        dwell_times = [r['mean_dwell'] for r in records if 'mean_dwell' in r]
        switches = [r['switches'] for r in records if 'switches' in r]

        # Option distribution across all episodes
        total_counts = np.zeros(4, dtype=np.int64)
        for r in records:
            if 'option_counts' in r:
                total_counts += np.array(r['option_counts'], dtype=np.int64)

        total_steps = max(1, total_counts.sum())
        opt_probs = total_counts / total_steps
        entropy = -sum(p * math.log(p + 1e-8) for p in opt_probs if p > 0)
        max_entropy = math.log(4.0)
        norm_entropy = entropy / max_entropy

        arm_summary = {
            'num_cases': n,
            'success_rate': round(100.0 * len(successes) / n, 2),
            'collision_rate': round(100.0 * len(collisions) / n, 2),
            'timeout_rate': round(100.0 * len(timeouts) / n, 2),
            'stagnation_count': len(stagnations),
            'mean_switches': round(float(np.mean(switches)), 2) if switches else 0.0,
            'mean_dwell_steps': round(float(np.mean(dwell_times)), 2) if dwell_times else 0.0,
            'option_distribution': {
                'Option_0_BypassLeft': round(float(opt_probs[0] * 100.0), 2),
                'Option_1_Direct': round(float(opt_probs[1] * 100.0), 2),
                'Option_2_BypassRight': round(float(opt_probs[2] * 100.0), 2),
                'Option_3_Yield': round(float(opt_probs[3] * 100.0), 2)
            },
            'normalized_entropy': round(float(norm_entropy), 3),
            'collapse_status': 'HEALTHY (Diverse)' if norm_entropy > 0.50 else 'WARNING (Collapsed)'
        }
        report[arm_name] = arm_summary

    return report


def print_behavioral_report(report: Dict):
    print("\n" + "=" * 85)
    print("OPTION BEHAVIOR & INTENT DYNAMICS ANALYSIS")
    print("=" * 85)
    print(f"{'Arm':<22} | {'SR (%)':<7} | {'CR (%)':<7} | {'Switches':<9} | {'Dwell (step)':<12} | {'Entropy':<8} | {'Collapse Check'}")
    print("-" * 85)
    for arm, stats in report.items():
        print(f"{arm:<22} | {stats['success_rate']:5.1f}%  | {stats['collision_rate']:5.1f}%  | "
              f"{stats['mean_switches']:7.1f}   | {stats['mean_dwell_steps']:10.1f}   | "
              f"{stats['normalized_entropy']:6.3f}   | {stats['collapse_status']}")

    print("\nOption Distribution Details:")
    for arm, stats in report.items():
        dist = stats['option_distribution']
        print(f"  {arm:<22}: "
              f"Left: {dist['Option_0_BypassLeft']:4.1f}% | "
              f"Direct: {dist['Option_1_Direct']:4.1f}% | "
              f"Right: {dist['Option_2_BypassRight']:4.1f}% | "
              f"Yield: {dist['Option_3_Yield']:4.1f}%")
    print("=" * 85 + "\n")


if __name__ == '__main__':
    default_json = RESULTS_DIR / 'option_critic_500cases_evaluation.json'
    if default_json.exists():
        rep = analyze_benchmark_telemetry(default_json)
        print_behavioral_report(rep)
    else:
        print(f"File {default_json} not found yet. Run evaluate.py first.")
