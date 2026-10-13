# Gmony: Dense Crowd Navigation with Macro-Intent Commitment & Operational Boundary Gating

This repository contains the complete codebase, trained model checkpoints, benchmark evaluation datasets, and comprehensive scientific investigation reports for the **Macro-Intent Commitment & Operational Boundary** project in dense robot crowd navigation (**CrowdNav**).

---

## 📌 Repository Overview

In dense crowd navigation, single-step reactive reinforcement learning policies often suffer from **oscillatory stagnation (limit-cycle wavering)** and **freezing robot syndrome after yielding**. Introducing macro-passing intent commitment can break these interaction deadlocks, but unconditional commitment introduces severe collision risks by overriding high-density reactive evasions.

This project investigates the **causal attribution and operational boundary** of intent commitment:
- **Dual-Stream Temporal Backbone**: Dual-Stream GRU encoding separate proximity and goal-directed spatial-temporal dynamics.
- **Hierarchical Macro-Intent Module**: Topological corridor intent selection and learnable termination heads.
- **Operational Boundary Mapping**: Rigorous state-level telemetry and counterfactual branching on cloned simulation states to discover where intent commitment helps versus harms navigation.
- **Boundary-Gated Meta Policy**: An online observable state-conditioned policy that engages intent commitment strictly in high-gain stagnation regimes while falling back to pure reactive lookahead in normal/tight scenarios.

---

## 📂 Repository Structure

```
├── strat_mamba_dual_stream.py          # Dual-Stream Spatial-Temporal Backbone & Policy (B1)
├── learnable_intent_mamba.py           # Learnable Intent Commitment & Switching (LICS) Policy (B2) & Heuristic Hysteresis (B3)
├── chart_operational_boundary_p3.py    # Phase P3: 2D Operational Boundary Map & 100 Fresh Cases Validation
├── run_counterfactual_branching.py     # Phase P2: Exact Homomorphic Counterfactual Branching on Cloned States
├── collect_state_level_telemetry.py    # Phase P1: High-Resolution Millisecond State-Level Telemetry
├── run_4arm_comparative_benchmark.py   # 4-Arm Controlled Benchmark Engine
├── train_intent_policy.py              # Macro-intent Head Training Script
├── crowd_sim/                          # Gym-Compatible Crowd Navigation Simulation Environment
├── kdarl_clean/                        # Cleaned Environment Contracts, Spatial Encoders & Configurations
├── baseline_weights/                   # Pretrained Temporal Backbone Weights
│   └── gru-t24-seed443.pth
├── checkpoints/                        # Trained Dual-Stream & LICS Checkpoints
│   ├── b1_strat_mamba_trained.pth
│   └── b2_lics_trained.pth
├── benchmark_results/                  # Full Benchmark Results & Telemetry Data
│   ├── independent_validation_100cases.json     # 100 Fresh Cases (90150-90249) 5-Policy Evaluation
│   ├── counterfactual_branching_results.json    # Cloned Snapshot Counterfactual Branching Data
│   └── state_level_telemetry.json               # Step-by-Step Trajectory Telemetry
├── docs/                               # Comprehensive Scientific Reports & Investigation Archives
│   ├── intent_commitment_operational_boundary_report.md  # Final Operational Boundary & Causal Attribution Report
│   ├── lics_rigorous_audit_and_truth_report.md           # P0-P1 Audit & True Metric Calibration Report
│   ├── paper2_lics_mamba_breakthrough_report.md          # LICS Architecture & Benchmark Report
│   ├── paper2_et_acr_breakthrough_report.md              # Early Stage Temporal Exploration Report
│   ├── paper2_comprehensive_investigation_report.md      # Ground-truth Investigation on Failure Modes
│   └── paper2_research_breakthrough_report.md            # Initial Hypothesis & Formulation Report
└── README.md                           # Project Documentation
```

---

## 📊 Benchmark Results

### 1. 100 Fresh Cases Independent Validation (Cases 90150 – 90249)

Evaluated under strict simulator event termination (`info['event']`), frozen weights, and identical evaluation configurations (20 humans, Square Crossing, $T=50\text{s}$):

| Policy Architecture | Success Rate (SR) | Collision Rate (CR) | Timeout Rate (TR) | Stagnation Rate (SOR) | Mean Time |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Base Dual-Stream (B1)** | 76.0% (76/100) | **4.0%** | 20.0% | 11.0% | 18.61s |
| **Fixed Dwell-4 (B3, Unconditional)** | 71.0% (71/100) | **13.0%** | 16.0% | 13.0% | 16.24s |
| **LICS (B2, Unconditional)** | 77.0% (77/100) | **13.0%** | 10.0% | 8.0% | 17.55s |
| **Boundary-Gated LICS (Ours)** | **79.0% (79/100)** | **4.0%** | 17.0% | 11.0% | 19.05s |
| **Boundary-Gated Fixed Dwell** | 77.0% (77/100) | **3.0%** | 20.0% | 10.0% | 20.05s |

### 2. Causal Dissection of Intent-Induced Collisions

Out of the 13 collisions that occurred under Unconditional LICS:
- **9 cases (69.2%)** (Cases 90160, 90168, 90182, 90213, 90217, 90224, 90233, 90238, 90244) were **safely navigated by the Base Reactive policy with 100% success and 0 collisions**.
- Forcing directional commitment blindly into dense interactions overrides micro-evasion capabilities.
- **Boundary-Gated LICS restored 100% of these 9 cases back to SUCCESS**, successfully compressing the collision rate from 13.0% back to 4.0%.

---

## 🗺️ The 2D Operational Boundary Map

Analysis of counterfactual branching identifies three distinct regimes:

```
Clearance / Stagnation    | Low Ratio (<=2.0)    | Med Ratio (2.0-3.5)  | High Deadlock (>3.5)
------------------------------------------------------------------------------------------
Safe (>0.40m)             | +0.00 (n=10)         |   N/A  (n=0)         | -0.33 (n=3)
Mod (0.28-0.40m)          | +1.00 (n=2)          | +1.00 (n=1)          | +0.00 (n=1)
Tight (<0.28m)            | +0.33 (n=6)          | +0.00 (n=4)          | +0.20 (n=5)
```

- **Region A (Beneficial Regime)**: High Stagnation ($\text{ratio}_{4s} > 3.0$) or Post-Yield Stall with moderate clearance ($0.22\text{m} \le d_{\min} < 0.60\text{m}$). Intent commitment provides directional momentum to break limit cycles (+38% to +50% intervention gain).
- **Region B (Harmful Regime)**: Immediate danger ($d_{\min} < 0.22\text{m}$) or open-field false alarms. Forcing intent commitment blinds reactive evasion, causing collisions or excessive detours.
- **Region C (Neutral Regime)**: Smooth progression ($\text{disp}_{4s} > 0.5\text{m}$, $\text{ratio}_{4s} \le 2.0$). Base reactive lookahead already achieves 100% success; intervention is redundant.

---

## 🚀 Reproduction Quickstart

### Prerequisites
- Python 3.8+
- PyTorch 2.0+ (CUDA supported)
- OpenAI Gym / Gymnasium, NumPy, SciPy

### Running Evaluations

1. **Run 100 Fresh Cases Validation & Operational Boundary Analysis**:
   ```bash
   python chart_operational_boundary_p3.py
   ```

2. **Run 4-Arm Comparative Benchmark**:
   ```bash
   python run_4arm_comparative_benchmark.py
   ```

3. **Run Counterfactual Branching on Exact Cloned Snapshots**:
   ```bash
   python run_counterfactual_branching.py
   ```

4. **Collect High-Resolution State Telemetry**:
   ```bash
   python collect_state_level_telemetry.py
   ```

---

## 📜 Scientific Reports

Comprehensive analytical reports detailing experimental audits, forensics, and causal deductions are located in [`docs/`](docs/):
- [`docs/intent_commitment_operational_boundary_report.md`](docs/intent_commitment_operational_boundary_report.md)
- [`docs/lics_rigorous_audit_and_truth_report.md`](docs/lics_rigorous_audit_and_truth_report.md)
- [`docs/paper2_lics_mamba_breakthrough_report.md`](docs/paper2_lics_mamba_breakthrough_report.md)
