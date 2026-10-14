import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
import sys
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)

REPO_ROOT = Path('/home/abc/workspace/autonomous_research_20261012')
sys.path.insert(0, str(REPO_ROOT))

from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import JointState
from run_b0_counterfactual_study import create_environment
from run_stage4_500cases_evaluation import load_stage4_policy
from replication.stage2_policy import load_stage2_policy, STAGE2_FEATURE_KEYS

REPLICATION_DIR = REPO_ROOT / 'replication'
OUT_DATASET = REPLICATION_DIR / 'stage2_branching_dataset.json'
OUT_STATS = REPLICATION_DIR / 'stage2_dataset_stats.json'

FORBIDDEN_RANGES = [
    (92000, 92499),
    (90000, 90299),
    (93000, 93199),
    (95000, 96499)
]

def assert_case_guard(start_case, num_cases):
    end_case = start_case + num_cases - 1
    for f_start, f_end in FORBIDDEN_RANGES:
        if not (end_case < f_start or start_case > f_end):
            raise ValueError(
                f"SAFETY VIOLATION: Case range [{start_case}, {end_case}] overlaps "
                f"with strictly prohibited range [{f_start}, {f_end}]!"
            )

def collect_single_case_counterfactuals(case_id, env, policy, gamma=0.99):
    assert_case_guard(case_id, 1)

    # 1. Run baseline trajectory with standard LR Gated (K=4)
    obs, _ = env.reset(options={'test_case': case_id})
    policy.reset_episode_stats()

    history_actions = []
    step_records = []
    step = 0
    done = False

    while not done and step < 201:
        step += 1
        state = JointState(env.robot.get_full_state(), [h.get_observable_state() for h in env.humans])
        if len(env.humans):
            curr_phys_dists = [
                math.hypot(env.robot.px - h.px, env.robot.py - h.py) - env.robot.radius - h.radius
                for h in env.humans
            ]
            phys_clearance = float(min(curr_phys_dists))
        else:
            phys_clearance = 999.0

        # Capture complete internal policy state BEFORE prediction
        saved_pol_state = policy.capture_internal_state()
        was_committed = (policy.committed_intent is not None)
        dwell_held = policy.dwell_steps_held
        orig_intent = policy.committed_intent

        # Step is eligible for termination sampling if commitment is currently active
        is_sampled = (was_committed and dwell_held >= 1)

        # Predict
        act = policy.predict(state)
        history_actions.append(ActionXY(act.vx, act.vy))

        feat = None
        if was_committed:
            if policy.last_term_features is not None:
                feat = dict(policy.last_term_features)
            elif hasattr(policy, 'last_base_res') and policy.last_base_res is not None:
                feat = policy.extract_termination_features(state, policy.last_base_res, phys_clearance)

        obs, rew, term, trunc, info = env.step(act)
        done = bool(term or trunc)

        step_records.append({
            'step': step,
            'was_committed': was_committed,
            'dwell_held': dwell_held,
            'orig_intent': orig_intent,
            'is_sampled': is_sampled,
            'feat': feat,
            'reward': float(rew),
            'done': done,
            'event': info.get('event', 'nothing'),
            'saved_pol_state': saved_pol_state
        })

    total_steps = len(step_records)
    # Compute actual returns from each step t (G_continue)
    returns_continue = [0.0] * total_steps
    cur_g = 0.0
    for t in reversed(range(total_steps)):
        cur_g = step_records[t]['reward'] + gamma * cur_g
        returns_continue[t] = cur_g

    # 2. For each step where commitment was active, simulate counterfactual abandon branch
    branch_samples = []

    for t_idx, rec in enumerate(step_records):
        # We sample termination decisions for steps where an intent is being held (dwell_held >= 1)
        if not rec['is_sampled'] or rec['feat'] is None:
            continue

        target_step = rec['step']
        # Replay prefix 0..t_idx-1
        obs, _ = env.reset(options={'test_case': case_id})
        policy.reset_episode_stats()

        for p_idx in range(t_idx):
            env.step(history_actions[p_idx])

        # Restore authentic policy state using identical function as null branch gate
        policy.restore_internal_state(rec['saved_pol_state'])

        # At target step: force abandon commitment!
        held_intent = policy.committed_intent
        policy.committed_intent = None
        policy.dwell_steps_held = 0
        policy.dwell_steps_remaining = 0
        policy.cusum_accumulator = 0.0

        # Run counterfactual continuation from state s_t
        cf_step = target_step
        cf_done = False
        cf_rewards = []
        recommitted = False
        steps_to_recommit = -1
        recommitted_intent = None
        intent_matched = False

        while not cf_done and cf_step < 201:
            state = JointState(env.robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            act = policy.predict(state)

            # Track re-commitment after abandon
            if not recommitted and policy.committed_intent is not None:
                recommitted = True
                steps_to_recommit = (cf_step - target_step) + 1
                recommitted_intent = int(policy.committed_intent)
                intent_matched = (held_intent is not None and recommitted_intent == int(held_intent))

            obs, rew, term, trunc, info = env.step(act)
            cf_done = bool(term or trunc)
            cf_rewards.append(float(rew))
            cf_step += 1

        # Compute G_abandon
        cur_cf_g = 0.0
        for r in reversed(cf_rewards):
            cur_cf_g = r + gamma * cur_cf_g
        g_abandon = cur_cf_g
        g_continue = returns_continue[t_idx]
        delta_g = g_abandon - g_continue

        # Label: 1 if abandon is better (> 0.0), 0 otherwise
        label = 1 if delta_g > 0.0 else 0

        branch_samples.append({
            'case_id': case_id,
            'step': target_step,
            'dwell_held': rec['dwell_held'],
            'orig_intent': int(held_intent) if held_intent is not None else None,
            'recommitted': recommitted,
            'steps_to_recommit': steps_to_recommit,
            'recommitted_intent': recommitted_intent,
            'intent_matched': intent_matched,
            'features': rec['feat'],
            'g_continue': round(g_continue, 4),
            'g_abandon': round(g_abandon, 4),
            'delta_g': round(delta_g, 4),
            'label': label,
            'split': 'train' if case_id < 99280 else 'val'
        })

    return branch_samples

def collect_chunk(start_case, num_cases, device='cpu', out_file=None):
    assert_case_guard(start_case, num_cases)
    env, env_config = create_environment()
    env.test_sim = 'square_crossing'
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

    # Load Stage 2 policy with forced_step4 (100% identical to learned_lr, with full termination feature extraction)
    policy = load_stage2_policy(termination_mode='forced_step4', device=device)
    robot.set_policy(policy)
    robot.env = env
    env.set_robot(robot)

    all_samples = []
    t_start = time.time()

    for idx in range(num_cases):
        case_id = start_case + idx
        t_case_start = time.time()
        samples = collect_single_case_counterfactuals(case_id, env, policy)
        all_samples.extend(samples)
        case_elapsed = time.time() - t_case_start
        total_elapsed = time.time() - t_start

        # Per-case incremental save
        if out_file:
            with open(out_file, 'w') as f:
                json.dump(all_samples, f, indent=2)

        # Per-case progress logging with explicit flush
        prefix = Path(out_file).name if out_file else f'Worker-{start_case}'
        print(
            f"[{prefix}] Case {case_id} ({idx+1}/{num_cases}) done in {case_elapsed:.1f}s | "
            f"Case samples: {len(samples)} | Cumulative samples: {len(all_samples)} | "
            f"Elapsed: {total_elapsed:.1f}s",
            flush=True
        )

    return all_samples

def main():
    import argparse
    import subprocess
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_case', type=int, default=99000)
    parser.add_argument('--total_cases', type=int, default=400)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--chunk_mode', action='store_true')
    parser.add_argument('--out_file', type=str, default=None)
    args = parser.parse_args()

    if args.chunk_mode:
        collect_chunk(args.start_case, args.total_cases, device=args.device, out_file=args.out_file)
        return

    assert_case_guard(args.start_case, args.total_cases)
    num_workers = min(args.workers, 6)

    print("=" * 95)
    print(f"STAGE 2 DATA COLLECTION: CASES {args.start_case}..{args.start_case + args.total_cases - 1} | WORKERS={num_workers}")
    print("=" * 95)

    cases_per_worker = args.total_cases // num_workers
    remainder = args.total_cases % num_workers

    chunks = []
    processes = []
    curr_start = args.start_case
    t_start = time.time()

    for w in range(num_workers):
        w_cases = cases_per_worker + (1 if w < remainder else 0)
        w_out = REPLICATION_DIR / f'stage2_data_part{w}.json'
        chunks.append({'worker_id': w, 'start': curr_start, 'num': w_cases, 'out_file': w_out})

        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            '--chunk_mode',
            '--start_case', str(curr_start),
            '--total_cases', str(w_cases),
            '--device', args.device,
            '--out_file', str(w_out)
        ]
        log_f = open(REPLICATION_DIR / f'stage2_data_worker_{w}.log', 'w')
        sub_env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=sub_env)
        processes.append((proc, log_f, w))
        curr_start += w_cases

    print(f"Launched {num_workers} parallel workers. Waiting for data collection...")
    for proc, log_f, wid in processes:
        proc.wait()
        log_f.close()
        print(f"Worker {wid} completed with exit code {proc.returncode}.")
        if proc.returncode != 0:
            print(f"ERROR: Worker {wid} exited with error!")
            sys.exit(1)

    merged_samples = []
    for c in chunks:
        with open(c['out_file'], 'r') as f:
            merged_samples.extend(json.load(f))

    with open(OUT_DATASET, 'w') as f:
        json.dump(merged_samples, f, indent=2)

    train_samples = [s for s in merged_samples if s['split'] == 'train']
    val_samples = [s for s in merged_samples if s['split'] == 'val']

    train_pos = sum(1 for s in train_samples if s['label'] == 1)
    val_pos = sum(1 for s in val_samples if s['label'] == 1)

    stats = {
        'total_samples': len(merged_samples),
        'train_samples': len(train_samples),
        'train_cases': 280,
        'train_pos_label': train_pos,
        'train_pos_ratio': round(train_pos / max(len(train_samples), 1) * 100.0, 2),
        'val_samples': len(val_samples),
        'val_cases': 120,
        'val_pos_label': val_pos,
        'val_pos_ratio': round(val_pos / max(len(val_samples), 1) * 100.0, 2),
        'feature_keys': STAGE2_FEATURE_KEYS
    }

    with open(OUT_STATS, 'w') as f:
        json.dump(stats, f, indent=2)

    t_total = time.time() - t_start
    print("=" * 95)
    print(f"COLLECTION COMPLETE: Total Samples: {len(merged_samples)} | Train: {len(train_samples)} (Pos: {stats['train_pos_ratio']}%) | Val: {len(val_samples)} (Pos: {stats['val_pos_ratio']}%)")
    print(f"Saved dataset: {OUT_DATASET}")
    print(f"Total time: {t_total:.1f}s")
    print("=" * 95)

if __name__ == '__main__':
    main()
