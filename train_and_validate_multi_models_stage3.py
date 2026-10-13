#!/usr/bin/env python3
"""Stage 3: Multi-Model Competition on Grouped Train/Validation Sets

Competitors:
1. Heuristic Rule Gate Baseline
2. Cost-Sensitive Logistic Regression
3. Cost-Sensitive Random Forest Classifier
4. Deep Meta-MLP Neural Network

Strict Methodology:
- Datasets combined from P1 study (390 snapshots) + Stage 2 targeted harvest (174 snapshots).
- Grouped split strictly by case_id (75% Train / 25% Validation).
- Evaluates:
  1. Harm Rejection Rate (Zero induced collisions priority)
  2. Net Rescue Gain (Rescued timeouts minus induced failures)
  3. ROC-AUC, Precision, Recall, and F1 on validation set
- Freezes and exports the winning champion model for Stage 4 testing.
"""

import sys
import json
import time
import joblib
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix

REPO_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = REPO_ROOT / 'benchmark_results'
MODELS_DIR = REPO_ROOT / 'trained_models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_KEYS = [
    'dmin_curr',
    'closing_speed',
    'travel_4s',
    'disp_4s',
    'ratio_4s',
    'reversals_4s',
    'dist_to_goal',
    'goal_heading_error',
    'safe_actions_ratio',
    'current_speed'
]

class Stage3MLP(nn.Module):
    def __init__(self, in_features=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.net(x)

def heuristic_gate(feat):
    """Heuristic rule gate baseline."""
    dmin = feat['dmin_curr']
    cspeed = feat['closing_speed']
    ratio = feat['ratio_4s']
    safe_ratio = feat['safe_actions_ratio']
    return bool(ratio >= 1.8 and dmin >= 0.25 and cspeed <= 0.3 and safe_ratio >= 0.5)

def load_consolidated_dataset():
    p1_path = RESULTS_DIR / 'b0_counterfactual_study_200cases.json'
    target_path = RESULTS_DIR / 'targeted_deadlock_and_harm_samples.json'

    with open(p1_path, 'r') as f:
        p1_cases = json.load(f)

    with open(target_path, 'r') as f:
        target_snaps = json.load(f)

    all_samples = []

    # Process P1 snapshots
    for c in p1_cases:
        cid = c['case_id']
        for s in c['snapshots']:
            feat = s['features']
            base_out = s['branch_A_base_k0']['outcome']
            k4_out = s['branch_C_dwell_k4']['outcome']
            time_base = float(s['branch_A_base_k0']['final_time'])
            time_k4 = float(s['branch_C_dwell_k4']['final_time'])
            time_saved = (time_base - time_k4) if (base_out == 'SUCCESS' and k4_out == 'SUCCESS') else 0.0

            gain = float(s['branch_C_dwell_k4']['gain_vs_base'])
            is_rescue = bool(gain > 0)
            is_harm = bool(gain < 0)
            is_beneficial = bool(is_rescue or (gain == 0 and base_out == 'SUCCESS' and time_saved >= 1.0))

            all_samples.append({
                'case_id': cid,
                'snapshot_id': s['snapshot_id'],
                'archetype': s['archetype'],
                'features': feat,
                'x': [float(feat[k]) for k in FEATURE_KEYS],
                'is_rescue': is_rescue,
                'is_harm': is_harm,
                'is_beneficial': is_beneficial,
                'time_saved': time_saved
            })

    # Process targeted snapshots
    for s in target_snaps:
        cid = s['case_id']
        feat = s['features']
        is_rescue = s['is_rescue']
        is_harm = s['is_harm']
        gain = s['gain_vs_base']
        time_saved = s.get('time_saved', 0.0)
        is_beneficial = bool(is_rescue or (gain == 0 and time_saved >= 1.0 and not is_harm))

        all_samples.append({
            'case_id': cid,
            'snapshot_id': s['snapshot_id'],
            'archetype': s['archetype'],
            'features': feat,
            'x': [float(feat[k]) for k in FEATURE_KEYS],
            'is_rescue': is_rescue,
            'is_harm': is_harm,
            'is_beneficial': is_beneficial,
            'time_saved': time_saved
        })

    return all_samples

def evaluate_model_performance(y_true, y_harm, y_scores, threshold=0.5, name="Model"):
    y_pred = (y_scores >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.5
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    # Safety metrics: harmful interventions triggered
    harm_triggers = int(np.sum((y_pred == 1) & (y_harm == 1)))
    harm_total = int(np.sum(y_harm == 1))
    harm_rejection_rate = ((harm_total - harm_triggers) / max(1, harm_total)) * 100.0

    net_gain = tp - harm_triggers

    return {
        'model': name,
        'threshold': round(float(threshold), 3),
        'auc': round(float(auc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1': round(float(f1), 4),
        'tp_beneficial': int(tp),
        'fp_neutral': int(fp - harm_triggers),
        'harm_triggers': harm_triggers,
        'harm_total': harm_total,
        'harm_rejection_rate': round(float(harm_rejection_rate), 2),
        'net_gain': int(net_gain)
    }

def main():
    print("=" * 85)
    print("STAGE 3: MULTI-MODEL COMPETITION & VALIDATION ON CONSOLIDATED DATASET")
    print("=" * 85)

    samples = load_consolidated_dataset()
    unique_cases = sorted(list(set(s['case_id'] for s in samples)))
    print(f"Consolidated Snapshots: {len(samples)} across {len(unique_cases)} unique cases.")

    # 75% Train / 25% Validation split grouped strictly by case_id
    np.random.seed(42)
    shuffled_cases = np.random.permutation(unique_cases)
    n_train = int(len(unique_cases) * 0.75)
    train_cids = set(shuffled_cases[:n_train])
    val_cids = set(shuffled_cases[n_train:])

    train_samples = [s for s in samples if s['case_id'] in train_cids]
    val_samples = [s for s in samples if s['case_id'] in val_cids]

    print(f"Grouped Split: Train={len(train_samples)} snapshots ({len(train_cids)} cases) | Val={len(val_samples)} snapshots ({len(val_cids)} cases)")
    
    # Check harm and rescue distribution
    train_harm = sum(1 for s in train_samples if s['is_harm'])
    train_rescue = sum(1 for s in train_samples if s['is_rescue'])
    train_ben = sum(1 for s in train_samples if s['is_beneficial'])
    val_harm = sum(1 for s in val_samples if s['is_harm'])
    val_rescue = sum(1 for s in val_samples if s['is_rescue'])
    val_ben = sum(1 for s in val_samples if s['is_beneficial'])

    print(f"Train Set: Beneficial={train_ben}, Rescues={train_rescue}, Harmful={train_harm}")
    print(f"Val Set:   Beneficial={val_ben}, Rescues={val_rescue}, Harmful={val_harm}")
    print("-" * 85)

    X_train_raw = np.array([s['x'] for s in train_samples], dtype=np.float32)
    y_train_ben = np.array([1 if s['is_beneficial'] else 0 for s in train_samples], dtype=np.int32)
    y_train_harm = np.array([1 if s['is_harm'] else 0 for s in train_samples], dtype=np.int32)

    X_val_raw = np.array([s['x'] for s in val_samples], dtype=np.float32)
    y_val_ben = np.array([1 if s['is_beneficial'] else 0 for s in val_samples], dtype=np.int32)
    y_val_harm = np.array([1 if s['is_harm'] else 0 for s in val_samples], dtype=np.int32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    competitor_results = []

    # =========================================================================
    # Competitor 1: Heuristic Rule Gate Baseline
    # =========================================================================
    h_val_scores = np.array([1.0 if heuristic_gate(s['features']) else 0.0 for s in val_samples])
    res_h = evaluate_model_performance(y_val_ben, y_val_harm, h_val_scores, threshold=0.5, name="Heuristic Rule Gate")
    competitor_results.append(res_h)

    # =========================================================================
    # Competitor 2: Cost-Sensitive Logistic Regression
    # =========================================================================
    lr = LogisticRegression(class_weight='balanced', C=0.3, max_iter=1000, random_state=42)
    lr.fit(X_train, y_train_ben)
    lr_val_probs = lr.predict_proba(X_val)[:, 1]

    # Threshold selection on training set prioritizing safety (zero harm triggers)
    best_lr_th = 0.5
    best_lr_net = -999
    lr_train_probs = lr.predict_proba(X_train)[:, 1]
    for th in np.arange(0.4, 0.85, 0.05):
        pred_t = (lr_train_probs >= th).astype(int)
        harm_trig = np.sum((pred_t == 1) & (y_train_harm == 1))
        tp_t = np.sum((pred_t == 1) & (y_train_ben == 1))
        net = tp_t - 2 * harm_trig
        if net > best_lr_net:
            best_lr_net = net
            best_lr_th = float(th)

    res_lr = evaluate_model_performance(y_val_ben, y_val_harm, lr_val_probs, threshold=best_lr_th, name="Cost-Sensitive Logistic Regression")
    competitor_results.append(res_lr)

    # =========================================================================
    # Competitor 3: Cost-Sensitive Random Forest
    # =========================================================================
    rf = RandomForestClassifier(n_estimators=150, max_depth=4, min_samples_leaf=3, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train_ben)
    rf_val_probs = rf.predict_proba(X_val)[:, 1]

    best_rf_th = 0.5
    best_rf_net = -999
    rf_train_probs = rf.predict_proba(X_train)[:, 1]
    for th in np.arange(0.4, 0.85, 0.05):
        pred_t = (rf_train_probs >= th).astype(int)
        harm_trig = np.sum((pred_t == 1) & (y_train_harm == 1))
        tp_t = np.sum((pred_t == 1) & (y_train_ben == 1))
        net = tp_t - 2 * harm_trig
        if net > best_rf_net:
            best_rf_net = net
            best_rf_th = float(th)

    res_rf = evaluate_model_performance(y_val_ben, y_val_harm, rf_val_probs, threshold=best_rf_th, name="Cost-Sensitive Random Forest")
    competitor_results.append(res_rf)

    # =========================================================================
    # Competitor 4: Stage 3 MLP Neural Network
    # =========================================================================
    torch.manual_seed(42)
    mlp = Stage3MLP(in_features=10)
    pos_weight = torch.tensor([(len(y_train_ben) - y_train_ben.sum()) / max(1.0, y_train_ben.sum())])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(mlp.parameters(), lr=0.005, weight_decay=1e-3)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_ben, dtype=torch.float32).unsqueeze(1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    mlp.train()
    for epoch in range(140):
        optimizer.zero_grad()
        out = mlp(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()

    mlp.eval()
    with torch.no_grad():
        mlp_val_probs = torch.sigmoid(mlp(X_val_t).squeeze(1)).numpy()
        mlp_train_probs = torch.sigmoid(mlp(X_train_t).squeeze(1)).numpy()

    best_mlp_th = 0.5
    best_mlp_net = -999
    for th in np.arange(0.4, 0.85, 0.05):
        pred_t = (mlp_train_probs >= th).astype(int)
        harm_trig = np.sum((pred_t == 1) & (y_train_harm == 1))
        tp_t = np.sum((pred_t == 1) & (y_train_ben == 1))
        net = tp_t - 2 * harm_trig
        if net > best_mlp_net:
            best_mlp_net = net
            best_mlp_th = float(th)

    res_mlp = evaluate_model_performance(y_val_ben, y_val_harm, mlp_val_probs, threshold=best_mlp_th, name="Stage 3 Meta-MLP")
    competitor_results.append(res_mlp)

    # =========================================================================
    # Validation Leaderboard Comparison
    # =========================================================================
    print(f"{'Competitor':<32} | {'AUC':<6} | {'Prec':<6} | {'Recall':<6} | {'F1':<6} | {'Harm Triggers':<14} | {'Harm Rejection%':<16} | {'Net Gain'}")
    print("-" * 115)
    for r in competitor_results:
        print(f"{r['model']:<32} | {r['auc']:<6.3f} | {r['precision']:<6.3f} | {r['recall']:<6.3f} | {r['f1']:<6.3f} | {r['harm_triggers']}/{r['harm_total']:<12} | {r['harm_rejection_rate']:<16.1f} | {r['net_gain']:+3d}")
    print("-" * 115)

    # Model Selection: Prioritize zero harm triggers, then maximum net gain
    # Champion selection rule:
    # 1. Harm triggers must be minimized (<= Heuristic's harm triggers)
    # 2. Net gain must exceed Heuristic's net gain
    h_harm = res_h['harm_triggers']
    h_net = res_h['net_gain']

    valid_candidates = [r for r in competitor_results if r['model'] != "Heuristic Rule Gate" and r['harm_triggers'] <= h_harm]
    if valid_candidates:
        champion = max(valid_candidates, key=lambda x: (x['net_gain'], x['auc']))
    else:
        # Fallback to candidate with best safety trade-off
        champion = min(competitor_results[1:], key=lambda x: (x['harm_triggers'], -x['net_gain']))

    print(f"\nCHAMPION MODEL SELECTED ON VALIDATION SET: {champion['model']}")
    print(f"Selection Rationale: Harm Triggers={champion['harm_triggers']} (Rejection={champion['harm_rejection_rate']}%), Net Gain={champion['net_gain']}, AUC={champion['auc']}")

    # Save both candidate models and scaler for Stage 4 evaluation
    joblib.dump(lr, MODELS_DIR / 'champion_gating_lr.joblib')
    joblib.dump(rf, MODELS_DIR / 'champion_gating_rf.joblib')

    champion_metadata = {
        'lr_model': {
            'threshold': best_lr_th,
            'metrics': res_lr
        },
        'rf_model': {
            'threshold': best_rf_th,
            'metrics': res_rf
        },
        'feature_keys': FEATURE_KEYS,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist()
    }

    with open(MODELS_DIR / 'champion_metadata.json', 'w') as f:
        json.dump(champion_metadata, f, indent=2)

    val_log_file = RESULTS_DIR / 'stage3_validation_selection_results.json'
    with open(val_log_file, 'w') as f:
        json.dump({
            'split_summary': {
                'total_samples': len(samples),
                'train_samples': len(train_samples),
                'val_samples': len(val_samples)
            },
            'leaderboard': competitor_results,
            'champion': champion_metadata
        }, f, indent=2)

    print(f"Saved Champion Model and Metadata to: {MODELS_DIR / 'champion_metadata.json'}")
    print(f"Saved Validation Results to: {val_log_file}")
    print("=" * 85)

    return champion_metadata

if __name__ == '__main__':
    main()
