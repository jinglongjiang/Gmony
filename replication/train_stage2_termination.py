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
import subprocess
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

REPO_ROOT = Path('/home/abc/workspace/autonomous_research_20261012')
sys.path.insert(0, str(REPO_ROOT))

from crowd_sim.envs.utils.robot import Robot
from run_b0_counterfactual_study import create_environment
from run_stage4_500cases_evaluation import run_single_episode
from replication.stage2_policy import (
    Stage2TerminationPolicy,
    load_stage2_policy,
    STAGE2_FEATURE_KEYS
)

REPLICATION_DIR = REPO_ROOT / 'replication'
DATASET_FILE = REPLICATION_DIR / 'stage2_branching_dataset.json'
OUT_CHAMPION_META = REPLICATION_DIR / 'stage2_champion_metadata.json'
OUT_CHAMPION_MODEL = REPLICATION_DIR / 'stage2_champion_model.joblib'
OUT_SHUFFLED_MODEL = REPLICATION_DIR / 'stage2_shuffled_model.joblib'

FORBIDDEN_RANGES = [
    (92000, 92499),
    (90000, 90299),
    (93000, 93199),
    (95000, 96499),
    (98000, 98999)
]

def assert_case_guard(start_case, num_cases):
    end_case = start_case + num_cases - 1
    for f_start, f_end in FORBIDDEN_RANGES:
        if not (end_case < f_start or start_case > f_end):
            raise ValueError(
                f"SAFETY VIOLATION: Case range [{start_case}, {end_case}] overlaps "
                f"with strictly prohibited range [{f_start}, {f_end}]!"
            )

def build_candidates():
    return {
        'cand1_lr_c1.0': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
        'cand2_lr_c0.1': LogisticRegression(C=0.1, max_iter=1000, random_state=42),
        'cand3_rf_d4_n50': RandomForestClassifier(n_estimators=50, max_depth=4, min_samples_leaf=5, random_state=42),
        'cand4_mlp_h32_16': MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=300, random_state=42, early_stopping=True)
    }

def evaluate_closed_loop(model, scaler, val_cases, model_name="cand", threshold=0.5, device='cpu', print_progress=True):
    env, env_config = create_environment()
    env.test_sim = 'square_crossing'
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

    policy = load_stage2_policy(
        termination_mode='learned',
        device=device,
        learned_model=model,
        learned_scaler=scaler,
        learned_threshold=threshold
    )
    robot.set_policy(policy)
    robot.env = env
    env.set_robot(robot)

    outcomes = []
    times = []
    dwells = []
    t_start = time.time()

    for idx, case_id in enumerate(val_cases):
        assert_case_guard(case_id, 1)
        t_case = time.time()
        res = run_single_episode(policy, env, case_id)
        dt = time.time() - t_case
        outcomes.append(res['outcome'])
        if res['outcome'] == 'SUCCESS':
            times.append(res['final_time'])
        dwells.extend(policy.dwell_history)

        if print_progress:
            cum_sr = float(np.mean([1.0 if o == 'SUCCESS' else 0.0 for o in outcomes]) * 100.0)
            print(
                f"[{model_name}] Case {case_id} ({idx+1}/{len(val_cases)}) in {dt:.1f}s | "
                f"outcome={res['outcome']} | cum_SR={cum_sr:.1f}% | "
                f"last_dwell={policy.dwell_history[-1] if policy.dwell_history else 0} | "
                f"elapsed={time.time() - t_start:.1f}s",
                flush=True
            )

    sr = float(np.mean([1.0 if o == 'SUCCESS' else 0.0 for o in outcomes]) * 100.0)
    cr = float(np.mean([1.0 if o == 'COLLISION' else 0.0 for o in outcomes]) * 100.0)
    tr = float(np.mean([1.0 if o == 'TIMEOUT' else 0.0 for o in outcomes]) * 100.0)
    avg_t = float(np.mean(times)) if times else 0.0
    mean_dwell = float(np.mean(dwells)) if dwells else 0.0

    return {
        'sr': round(sr, 2),
        'cr': round(cr, 2),
        'tr': round(tr, 2),
        'avg_time': round(avg_t, 2),
        'mean_dwell': round(mean_dwell, 2)
    }

def run_single_candidate_train(cand_name, out_file, device='cpu'):
    with open(DATASET_FILE, 'r') as f:
        data = json.load(f)

    train_data = [d for d in data if d['split'] == 'train']
    val_data = [d for d in data if d['split'] == 'val']

    X_train = np.array([[d['features'][k] for k in STAGE2_FEATURE_KEYS] for d in train_data], dtype=np.float32)
    y_train = np.array([d['label'] for d in train_data], dtype=np.int32)
    X_val = np.array([[d['features'][k] for k in STAGE2_FEATURE_KEYS] for d in val_data], dtype=np.float32)
    y_val = np.array([d['label'] for d in val_data], dtype=np.int32)

    mean = np.mean(X_train, axis=0)
    scale = np.std(X_train, axis=0)
    scale[scale < 1e-6] = 1.0
    scaler = {'mean': mean.tolist(), 'scale': scale.tolist()}

    X_train_norm = (X_train - mean) / scale
    X_val_norm = (X_val - mean) / scale

    val_cases = list(range(99280, 99400))
    assert_case_guard(val_cases[0], len(val_cases))

    cands = build_candidates()
    if cand_name not in cands:
        raise ValueError(f"Unknown candidate {cand_name}")
    model = cands[cand_name]

    t0 = time.time()
    print(f"[{cand_name}] Fitting model on {len(X_train)} train samples...", flush=True)
    model.fit(X_train_norm, y_train)

    probs_val = model.predict_proba(X_val_norm)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_val_norm)
    preds_val = (probs_val >= 0.5).astype(int)
    acc = float(accuracy_score(y_val, preds_val))
    auc = float(roc_auc_score(y_val, probs_val)) if len(np.unique(y_val)) > 1 else 0.5
    f1 = float(f1_score(y_val, preds_val, zero_division=0))

    print(f"[{cand_name}] Offline Val: Acc={acc*100:.2f}%, AUC={auc:.4f}, F1={f1:.4f}. Starting Closed-Loop Eval...", flush=True)
    sim_metrics = evaluate_closed_loop(model, {'mean': mean, 'scale': scale}, val_cases, model_name=cand_name, threshold=0.5, device=device, print_progress=True)
    elapsed = time.time() - t0

    result = {
        'name': cand_name,
        'offline_acc': round(acc * 100.0, 2),
        'offline_auc': round(auc, 4),
        'offline_f1': round(f1, 4),
        'sim_sr': sim_metrics['sr'],
        'sim_cr': sim_metrics['cr'],
        'sim_tr': sim_metrics['tr'],
        'sim_time': sim_metrics['avg_time'],
        'sim_dwell': sim_metrics['mean_dwell'],
        'train_time_s': round(elapsed, 1)
    }

    if out_file:
        with open(out_file, 'w') as f:
            json.dump(result, f, indent=2)

    print(f"[{cand_name}] COMPLETE | SR={result['sim_sr']}% | CR={result['sim_cr']}% | Time={result['sim_time']}s | Total: {elapsed:.1f}s", flush=True)
    return result

def train_and_select(device='cpu', workers=4):
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Missing dataset: {DATASET_FILE}")

    with open(DATASET_FILE, 'r') as f:
        data = json.load(f)

    train_data = [d for d in data if d['split'] == 'train']
    val_data = [d for d in data if d['split'] == 'val']

    X_train = np.array([[d['features'][k] for k in STAGE2_FEATURE_KEYS] for d in train_data], dtype=np.float32)
    y_train = np.array([d['label'] for d in train_data], dtype=np.int32)
    X_val = np.array([[d['features'][k] for k in STAGE2_FEATURE_KEYS] for d in val_data], dtype=np.float32)
    y_val = np.array([d['label'] for d in val_data], dtype=np.int32)

    mean = np.mean(X_train, axis=0)
    scale = np.std(X_train, axis=0)
    scale[scale < 1e-6] = 1.0
    scaler = {'mean': mean.tolist(), 'scale': scale.tolist()}

    X_train_norm = (X_train - mean) / scale
    X_val_norm = (X_val - mean) / scale

    val_cases = list(range(99280, 99400))
    assert_case_guard(val_cases[0], len(val_cases))

    print("=" * 95)
    print(f"TRAINING STAGE 2 CANDIDATE TERMINATION HEADS")
    print(f"Train samples: {len(X_train)} (Pos: {np.mean(y_train)*100:.1f}%) | Val samples: {len(X_val)} (Pos: {np.mean(y_val)*100:.1f}%)")
    print(f"Workers: {workers}")
    print("=" * 95, flush=True)

    cands = build_candidates()
    cand_names = list(cands.keys())
    cand_results = {}

    if workers <= 1:
        for name in cand_names:
            res = run_single_candidate_train(name, None, device=device)
            cand_results[name] = res
    else:
        num_workers = min(workers, len(cand_names))
        processes = []
        part_files = []
        for w, name in enumerate(cand_names):
            part_out = REPLICATION_DIR / f'stage2_cand_{name}.json'
            part_files.append((name, part_out))
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                '--cand_mode',
                '--cand_name', name,
                '--out_file', str(part_out),
                '--device', device
            ]
            log_f = open(REPLICATION_DIR / f'stage2_cand_worker_{name}.log', 'w')
            sub_env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, env=sub_env)
            processes.append((proc, log_f, name))
            print(f"Launched Worker for candidate [{name}] -> log: stage2_cand_worker_{name}.log", flush=True)

        for proc, log_f, name in processes:
            proc.wait()
            log_f.close()
            print(f"Worker for [{name}] finished with exit code {proc.returncode}.", flush=True)
            if proc.returncode != 0:
                raise RuntimeError(f"Worker for candidate [{name}] failed! Check stage2_cand_worker_{name}.log")

        for name, pf in part_files:
            with open(pf, 'r') as f:
                cand_results[name] = json.load(f)

    # Select champion based on highest ClosedLoop Validation SR (tie-breaker: lower CR, higher offline AUC)
    sorted_cands = sorted(cand_results.items(), key=lambda kv: (-kv[1]['sim_sr'], kv[1]['sim_cr'], -kv[1]['offline_auc']))
    champion_name, champ_info = sorted_cands[0]

    print("=" * 95)
    print(f"CHAMPION SELECTED: {champion_name}")
    print(f"  Closed-Loop Validation SR: {champ_info['sim_sr']}%, CR: {champ_info['sim_cr']}%, Time: {champ_info['sim_time']}s, Dwell: {champ_info['sim_dwell']}")
    print(f"  Offline Val: Acc={champ_info['offline_acc']}%, AUC={champ_info['offline_auc']:.4f}, F1={champ_info['offline_f1']:.4f}")
    print("=" * 95, flush=True)

    # Retrain champion model cleanly to save
    champion_model = build_candidates()[champion_name]
    champion_model.fit(X_train_norm, y_train)

    # Train control arm: identical champion architecture trained on SHUFFLED labels
    print(f"Training Shuffled Control using architecture of [{champion_name}]...", flush=True)
    shuffled_model = build_candidates()[champion_name]
    np.random.seed(42)
    y_train_shuffled = np.random.permutation(y_train)
    shuffled_model.fit(X_train_norm, y_train_shuffled)

    shuf_probs = shuffled_model.predict_proba(X_val_norm)[:, 1] if hasattr(shuffled_model, 'predict_proba') else shuffled_model.predict(X_val_norm)
    shuf_preds = (shuf_probs >= 0.5).astype(int)
    shuf_acc = float(accuracy_score(y_val, shuf_preds))
    shuf_auc = float(roc_auc_score(y_val, shuf_probs)) if len(np.unique(y_val)) > 1 else 0.5
    shuf_f1 = float(f1_score(y_val, shuf_preds, zero_division=0))

    print(f"[Shuffled Control] Offline Val: Acc={shuf_acc*100:.2f}%, AUC={shuf_auc:.4f}, F1={shuf_f1:.4f}. Starting Closed-Loop Eval...", flush=True)
    shuf_sim = evaluate_closed_loop(shuffled_model, {'mean': mean, 'scale': scale}, val_cases, model_name=f"{champion_name}_shuffled", threshold=0.5, device=device, print_progress=True)
    print(f"[Shuffled Control] ClosedLoop Val: SR={shuf_sim['sr']:5.1f}%, CR={shuf_sim['cr']:4.1f}%, Time={shuf_sim['avg_time']:5.2f}s, Dwell={shuf_sim['mean_dwell']:4.1f}", flush=True)

    # Calculate differences between Real Label Champion and Shuffled Label Control
    diff_sr = round(champ_info['sim_sr'] - shuf_sim['sr'], 2)
    diff_cr = round(champ_info['sim_cr'] - shuf_sim['cr'], 2)
    diff_time = round(champ_info['sim_time'] - shuf_sim['avg_time'], 2)
    diff_dwell = round(champ_info['sim_dwell'] - shuf_sim['mean_dwell'], 2)
    diff_auc = round(champ_info['offline_auc'] - shuf_auc, 4)

    print("=" * 95)
    print("COMPARISON: REAL LABELS vs SHUFFLED LABELS CONTROL")
    print(f"  Real Label Champion [{champion_name}]: SR={champ_info['sim_sr']}%, CR={champ_info['sim_cr']}%, Time={champ_info['sim_time']}s, Dwell={champ_info['sim_dwell']}, AUC={champ_info['offline_auc']:.4f}")
    print(f"  Shuffled Label Control:               SR={shuf_sim['sr']}%, CR={shuf_sim['cr']}%, Time={shuf_sim['avg_time']}s, Dwell={shuf_sim['mean_dwell']}, AUC={shuf_auc:.4f}")
    print(f"  DIFFERENCE (Real - Shuffled):         ΔSR={diff_sr:+.2f}%, ΔCR={diff_cr:+.2f}%, ΔTime={diff_time:+.2f}s, ΔDwell={diff_dwell:+.2f}, ΔAUC={diff_auc:+.4f}")
    print("=" * 95, flush=True)

    # Save artifacts
    joblib.dump(champion_model, OUT_CHAMPION_MODEL)
    joblib.dump(shuffled_model, OUT_SHUFFLED_MODEL)

    meta = {
        'champion_name': champion_name,
        'feature_keys': STAGE2_FEATURE_KEYS,
        'scaler_mean': scaler['mean'],
        'scaler_scale': scaler['scale'],
        'threshold': 0.5,
        'champion_metrics': {
            'offline_acc': champ_info['offline_acc'],
            'offline_auc': champ_info['offline_auc'],
            'offline_f1': champ_info['offline_f1'],
            'sim_sr': champ_info['sim_sr'],
            'sim_cr': champ_info['sim_cr'],
            'sim_tr': champ_info['sim_tr'],
            'sim_time': champ_info['sim_time'],
            'sim_dwell': champ_info['sim_dwell']
        },
        'shuffled_metrics': {
            'offline_acc': round(shuf_acc * 100.0, 2),
            'offline_auc': round(shuf_auc, 4),
            'offline_f1': round(shuf_f1, 4),
            'sim_sr': shuf_sim['sr'],
            'sim_cr': shuf_sim['cr'],
            'sim_tr': shuf_sim['tr'],
            'sim_time': shuf_sim['avg_time'],
            'sim_dwell': shuf_sim['mean_dwell']
        },
        'delta_real_minus_shuffled': {
            'delta_sr': diff_sr,
            'delta_cr': diff_cr,
            'delta_time': diff_time,
            'delta_dwell': diff_dwell,
            'delta_auc': diff_auc
        },
        'all_candidates_summary': {
            k: {
                'offline_acc': v['offline_acc'],
                'offline_auc': v['offline_auc'],
                'offline_f1': v['offline_f1'],
                'sim_sr': v['sim_sr'],
                'sim_cr': v['sim_cr'],
                'sim_tr': v['sim_tr'],
                'sim_time': v['sim_time'],
                'sim_dwell': v['sim_dwell'],
                'train_time_s': v['train_time_s']
            }
            for k, v in cand_results.items()
        }
    }

    with open(OUT_CHAMPION_META, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"Saved champion metadata: {OUT_CHAMPION_META}")
    print(f"Saved champion model: {OUT_CHAMPION_MODEL}")
    print(f"Saved shuffled model: {OUT_SHUFFLED_MODEL}")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cand_mode', action='store_true')
    parser.add_argument('--cand_name', type=str, default=None)
    parser.add_argument('--out_file', type=str, default=None)
    args = parser.parse_args()

    if args.cand_mode:
        run_single_candidate_train(args.cand_name, args.out_file, device=args.device)
    else:
        train_and_select(device=args.device, workers=args.workers)
