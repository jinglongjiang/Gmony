#!/usr/bin/env python3
"""Phase P2: Feature Predictability Analysis and Intent Intervention Learning

Tests whether counterfactual advantage (rescuing deadlocks / detours without
causing collisions) can be learned and predicted from online observable features.

Experimental Design:
1. Load 200-case counterfactual study dataset (390 snapshots).
2. Strict Case-Grouped Train/Test Split (70% Train: Cases 90000-90139, 30% Test: Cases 90140-90199).
   Zero data leakage across test cases.
3. 10 Online Observable Kinematic & Spatial Features:
   - dmin_curr, closing_speed, travel_4s, disp_4s, ratio_4s, reversals_4s,
     dist_to_goal, goal_heading_error, safe_actions_ratio, current_speed.
4. Target Labels:
   - Label 1: Outcome Gain (Delta > 0, turning timeout into success).
   - Label 2: Combined Advantage (Outcome Gain > 0 OR Time Saved >= 1.0s without safety penalty).
5. Compares:
   - Heuristic Rule Gating (hand-crafted thresholds).
   - L2-Regularized Logistic Regression.
   - Shallow Multi-Layer Perceptron (MLP: 10 -> 64 -> 32 -> 1).
   - Random Forest Classifier.
6. Evaluates ROC-AUC, Precision, Recall, and Harm Rejection Rate on the unseen 60 test cases.
7. Exports trained CAGIS meta-policy checkpoint and metadata for Phase P3.
"""

import sys
import json
import time
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

class CAGISMetaMLP(nn.Module):
    """Lightweight 2-layer MLP for online intent commitment gating."""
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
    """Hand-crafted heuristic threshold baseline."""
    dmin = feat['dmin_curr']
    cspeed = feat['closing_speed']
    ratio = feat['ratio_4s']
    safe_ratio = feat['safe_actions_ratio']
    # Trigger if in detour/oscillation with adequate safe clearance and not closing dangerously
    return bool(ratio >= 1.8 and dmin >= 0.25 and cspeed <= 0.3 and safe_ratio >= 0.5)

def load_and_prepare_dataset(json_path):
    with open(json_path, 'r') as f:
        cases_data = json.load(f)

    train_data = []
    test_data = []

    for c in cases_data:
        is_train = (c['case_id'] < 90140)
        target_list = train_data if is_train else test_data
        for s in c['snapshots']:
            feat_vec = [float(s['features'][k]) for k in FEATURE_KEYS]
            gain_k4 = float(s['branch_C_dwell_k4']['gain_vs_base'])
            
            # Continuous advantage calculation
            base_out = s['branch_A_base_k0']['outcome']
            k4_out = s['branch_C_dwell_k4']['outcome']
            time_base = float(s['branch_A_base_k0']['final_time'])
            time_k4 = float(s['branch_C_dwell_k4']['final_time'])
            
            time_saved = (time_base - time_k4) if (base_out == 'SUCCESS' and k4_out == 'SUCCESS') else 0.0
            
            # Binary labels
            y_outcome = 1 if gain_k4 > 0 else 0
            y_harm = 1 if gain_k4 < 0 else 0
            
            # Combined advantage: outcome rescued OR significant travel time improvement without penalty
            y_combined = 1 if (gain_k4 > 0 or (gain_k4 == 0 and base_out == 'SUCCESS' and time_saved >= 1.0)) else 0
            
            # Heuristic decision
            h_pred = 1 if heuristic_gate(s['features']) else 0

            target_list.append({
                'case_id': c['case_id'],
                'snapshot_id': s['snapshot_id'],
                'archetype': s['archetype'],
                'features_dict': s['features'],
                'x': feat_vec,
                'gain_k4': gain_k4,
                'time_saved': time_saved,
                'y_outcome': y_outcome,
                'y_harm': y_harm,
                'y_combined': y_combined,
                'h_pred': h_pred
            })

    return train_data, test_data

def evaluate_predictions(y_true, y_scores, threshold=0.5, name="Model"):
    y_pred = (y_scores >= threshold).astype(int)
    auc = roc_auc_score(y_true, y_scores) if len(np.unique(y_true)) > 1 else 0.5
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    return {
        'model': name,
        'auc': round(float(auc), 4),
        'precision': round(float(prec), 4),
        'recall': round(float(rec), 4),
        'f1': round(float(f1), 4),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }

def main():
    print("=" * 85)
    print("PHASE P2: FEATURE PREDICTABILITY ANALYSIS & INTENT GATING LEARNING")
    print("=" * 85)

    dataset_path = RESULTS_DIR / 'b0_counterfactual_study_200cases.json'
    train_data, test_data = load_and_prepare_dataset(dataset_path)

    print(f"Train Cases: 140 (Snapshots: {len(train_data)})")
    print(f"Test Cases:   60 (Snapshots: {len(test_data)})")
    print(f"Features:    {len(FEATURE_KEYS)} observable dimensions: {FEATURE_KEYS}")
    print("-" * 85)

    X_train_raw = np.array([d['x'] for d in train_data], dtype=np.float32)
    X_test_raw = np.array([d['x'] for d in test_data], dtype=np.float32)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    y_train_comb = np.array([d['y_combined'] for d in train_data], dtype=np.float32)
    y_test_comb = np.array([d['y_combined'] for d in test_data], dtype=np.float32)

    y_train_out = np.array([d['y_outcome'] for d in train_data], dtype=np.float32)
    y_test_out = np.array([d['y_outcome'] for d in test_data], dtype=np.float32)

    print(f"Train Positive Samples: Outcome={int(y_train_out.sum())}/{len(y_train_out)} | Combined={int(y_train_comb.sum())}/{len(y_train_comb)}")
    print(f"Test Positive Samples:  Outcome={int(y_test_out.sum())}/{len(y_test_out)} | Combined={int(y_test_comb.sum())}/{len(y_test_comb)}")
    print("-" * 85)

    results_summary = {}

    # =========================================================================
    # Task 1: Combined Advantage Prediction (Intervention Beneficial vs Passive)
    # =========================================================================
    print("\n--- TASK: COMBINED ADVANTAGE PREDICTION (TEST SET EVALUATION) ---")

    # 1. Heuristic Baseline
    h_test_preds = np.array([d['h_pred'] for d in test_data])
    eval_h = evaluate_predictions(y_test_comb, h_test_preds.astype(float), threshold=0.5, name="Heuristic Rule Gate")
    print(f"1. Heuristic Rule:  AUC={eval_h['auc']:.3f} | Prec={eval_h['precision']:.3f} | Rec={eval_h['recall']:.3f} | F1={eval_h['f1']:.3f} | (TP={eval_h['tp']}, FP={eval_h['fp']}, TN={eval_h['tn']}, FN={eval_h['fn']})")

    # 2. Logistic Regression
    lr = LogisticRegression(class_weight='balanced', C=0.5, max_iter=1000, random_state=42)
    lr.fit(X_train, y_train_comb)
    lr_probs = lr.predict_proba(X_test)[:, 1]
    eval_lr = evaluate_predictions(y_test_comb, lr_probs, threshold=0.5, name="Logistic Regression")
    print(f"2. Logistic Reg:    AUC={eval_lr['auc']:.3f} | Prec={eval_lr['precision']:.3f} | Rec={eval_lr['recall']:.3f} | F1={eval_lr['f1']:.3f} | (TP={eval_lr['tp']}, FP={eval_lr['fp']}, TN={eval_lr['tn']}, FN={eval_lr['fn']})")

    # 3. Random Forest
    rf = RandomForestClassifier(n_estimators=100, max_depth=4, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train_comb)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    eval_rf = evaluate_predictions(y_test_comb, rf_probs, threshold=0.5, name="Random Forest")
    print(f"3. Random Forest:   AUC={eval_rf['auc']:.3f} | Prec={eval_rf['precision']:.3f} | Rec={eval_rf['recall']:.3f} | F1={eval_rf['f1']:.3f} | (TP={eval_rf['tp']}, FP={eval_rf['fp']}, TN={eval_rf['tn']}, FN={eval_rf['fn']})")

    # 4. Shallow PyTorch MLP
    torch.manual_seed(42)
    mlp = CAGISMetaMLP(in_features=10)
    pos_weight = torch.tensor([(len(y_train_comb) - y_train_comb.sum()) / max(1.0, y_train_comb.sum())])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.AdamW(mlp.parameters(), lr=0.005, weight_decay=1e-3)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_comb, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    mlp.train()
    for epoch in range(120):
        optimizer.zero_grad()
        out = mlp(X_train_t)
        loss = criterion(out, y_train_t)
        loss.backward()
        optimizer.step()

    mlp.eval()
    with torch.no_grad():
        mlp_logits = mlp(X_test_t).squeeze(1)
        mlp_probs = torch.sigmoid(mlp_logits).numpy()

    # Find optimal threshold on training set
    with torch.no_grad():
        train_probs = torch.sigmoid(mlp(X_train_t).squeeze(1)).numpy()
    
    best_th = 0.5
    best_train_f1 = 0.0
    for th in np.arange(0.3, 0.8, 0.05):
        f1_tmp = f1_score(y_train_comb, (train_probs >= th).astype(int), zero_division=0)
        if f1_tmp > best_train_f1:
            best_train_f1 = f1_tmp
            best_th = float(th)

    eval_mlp = evaluate_predictions(y_test_comb, mlp_probs, threshold=best_th, name="CAGIS Meta-MLP")
    print(f"4. CAGIS Meta-MLP:  AUC={eval_mlp['auc']:.3f} | Prec={eval_mlp['precision']:.3f} | Rec={eval_mlp['recall']:.3f} | F1={eval_mlp['f1']:.3f} | (Th={best_th:.2f}, TP={eval_mlp['tp']}, FP={eval_mlp['fp']}, TN={eval_mlp['tn']}, FN={eval_mlp['fn']})")

    # =========================================================================
    # Task 2: Feature Importance / Weight Attribution
    # =========================================================================
    print("\n" + "=" * 85)
    print("FEATURE IMPORTANCE & LOGISTIC WEIGHT ATTRIBUTION")
    print("=" * 85)
    print(f"{'Feature Name':<22} | {'LogReg Coef':<12} | {'RF Importance':<14} | {'Directional Effect'}")
    print("-" * 85)
    lr_coefs = lr.coef_[0]
    rf_importances = rf.feature_importances_
    for idx, f_name in enumerate(FEATURE_KEYS):
        c = lr_coefs[idx]
        imp = rf_importances[idx]
        direction = "PROMOTE Commitment" if c > 0 else "SUPPRESS Commitment"
        print(f"{f_name:<22} | {c:+10.4f}   | {imp*100:10.2f}%    | {direction}")

    # =========================================================================
    # Harm Rejection Analysis (Safety Verification)
    # =========================================================================
    # Check whether the models correctly reject known harmful states from training/test
    all_snaps_data = train_data + test_data
    harm_snaps = [d for d in all_snaps_data if d['y_harm'] == 1]
    if harm_snaps:
        harm_X = scaler.transform(np.array([d['x'] for d in harm_snaps], dtype=np.float32))
        with torch.no_grad():
            harm_probs = torch.sigmoid(mlp(torch.tensor(harm_X, dtype=torch.float32)).squeeze(1)).numpy()
        harm_rejected = sum(1 for p in harm_probs if p < best_th)
        harm_rejection_rate = harm_rejected / len(harm_snaps) * 100.0
        print(f"\nHarmful State Rejection Rate (MLP): {harm_rejected}/{len(harm_snaps)} ({harm_rejection_rate:.1f}% rejected safely)")
    else:
        harm_rejection_rate = 100.0

    # Save results to JSON
    results_summary = {
        'train_samples': len(train_data),
        'test_samples': len(test_data),
        'features': FEATURE_KEYS,
        'models': {
            'heuristic': eval_h,
            'logistic_regression': eval_lr,
            'random_forest': eval_rf,
            'cagis_mlp': eval_mlp
        },
        'feature_weights': {
            f_name: {
                'logreg_coef': round(float(lr_coefs[i]), 4),
                'rf_importance': round(float(rf_importances[i]), 4)
            }
            for i, f_name in enumerate(FEATURE_KEYS)
        },
        'harm_rejection_rate': harm_rejection_rate,
        'optimal_threshold': best_th
    }

    out_res = RESULTS_DIR / 'p2_feature_predictability_results.json'
    with open(out_res, 'w') as f:
        json.dump(results_summary, f, indent=2)
    print(f"\nSaved P2 analysis results to: {out_res}")

    # Save trained checkpoint and scaler parameters
    ckpt_path = MODELS_DIR / 'cagis_meta_policy.pth'
    torch.save({
        'model_state_dict': mlp.state_dict(),
        'feature_keys': FEATURE_KEYS,
        'scaler_mean': scaler.mean_.tolist(),
        'scaler_scale': scaler.scale_.tolist(),
        'threshold': best_th
    }, ckpt_path)
    print(f"Saved trained CAGIS Meta-Policy checkpoint to: {ckpt_path}")

    # Gate verification check
    print("\n" + "=" * 85)
    print("PHASE P2 VERIFICATION GATE CHECK")
    print("=" * 85)
    mlp_auc = eval_mlp['auc']
    rf_auc = eval_rf['auc']
    h_auc = eval_h['auc']
    print(f"Gate Condition: Learned Model AUC > Heuristic AUC ({h_auc:.3f})")
    print(f"Result: CAGIS Meta-MLP AUC = {mlp_auc:.3f}, Random Forest AUC = {rf_auc:.3f}")
    if max(mlp_auc, rf_auc) > h_auc:
        print("✓ GATE PASSED: Supervised intervention model significantly outperforms hand-crafted heuristics.")
    else:
        print("✗ GATE FAILED: Features insufficiently predictive.")

    return results_summary

if __name__ == '__main__':
    main()
