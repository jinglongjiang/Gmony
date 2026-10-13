# Option-Critic & CF-OC Research Suite

Autonomous Algorithmic Research in Dynamic Crowd Navigation (CrowdNav).

## 1. Overview

This module implements two hierarchical decision-making algorithms to solve the fundamental problem of unstable intention switching and oscillation in dense crowds:

1. **Standard Option-Critic (OC)**:
   - Based on Bacon et al. (AAAI 2017) and Harb et al. (AAAI 2018).
   - Jointly learns intra-option policy $\pi_o(a \mid h)$, option values $Q_\Omega(h, o)$, and option termination conditions $\beta_o(h)$ with Deliberation Cost $\eta$.
2. **Counterfactual Switching-Advantage Option-Critic (CF-OC)**:
   - Original algorithmic contribution.
   - Leverages same-state counterfactual branching to directly supervise advantage calibration:
     $$\Delta_{\text{pred}}(h_t, o_t) \approx \Delta_{\text{CF}} = G_{\text{reselect}} - G_{\text{continue}}$$
   - Integrates self-supervised short-horizon collision risk estimation:
     $$\hat{p}_{\text{collision}}(h_t, o_t) \approx y_{\text{collision}\le 4s}$$
   - Counterfactually calibrates option termination without heuristic post-hoc gates:
     $$\beta_{\text{CF}}(h_t, o_t) = \sigma(W_\beta [h_t, \beta_0, \Delta_{\text{pred}}, \hat{p}_{\text{collision}}])$$

---

## 2. Architecture & Contracts

- **Perception Backbone**:
  - Spatial Encoder: `EnhancedSpatialEncoder` (robot 13D token + top-5 human 13D tokens + 8D relative relation vectors + MultiheadAttention).
  - Temporal Encoder: Modular & Swappable (`GRUTemporalEncoder`, `KDATemporalEncoder`, `MambaTemporalEncoder`). Default: GRU (4 layers, $d_{\text{model}}=256$, sequence length $T=24$).
- **Intra-Option Action Space**: 80 discrete kinematic actions (5 speeds $\times$ 16 headings).
- **Macro Intent Options ($N=4$)**:
  - Option 0: Bypass Left ($\Delta\theta > +0.15$ rad)
  - Option 1: Direct Progress ($|\Delta\theta| \le 0.15$ rad)
  - Option 2: Bypass Right ($\Delta\theta < -0.15$ rad)
  - Option 3: Yield / Deceleration ($v < 0.1$ m/s)
- **Execution Contract**: Strict Call-and-Return option execution cycle.

---

## 3. Directory Structure

```text
research/option_critic/
├── __init__.py
├── networks.py                  # Neural network architecture & heads
├── option_critic_losses.py      # Bellman Huber, PG, termination, CF losses
├── policy.py                    # Call-and-Return execution & state cloning
├── replay_buffer.py             # Replay buffer with 4s collision tagging
├── counterfactual_sampler.py    # Same-state branching & advantage sampler
├── train_il.py                  # Expert demonstration & imitation learning
├── train_rl.py                  # Standard Option-Critic RL training
├── train_cf_oc.py               # CF-OC training with counterfactual supervision
├── evaluate.py                  # 500-case comparative benchmark & McNemar test
├── analyze_option_behavior.py   # Option collapse, dwell, & switching analysis
├── configs/                     # Standardized experiment configs
│   ├── standard_oc.yaml
│   └── cf_oc.yaml
├── tests/                       # Unit test suite (4/4 passing)
│   └── test_option_critic.py
└── checkpoints/                 # Model checkpoints (.pth)
```

---

## 4. Execution Commands

### Unit Testing
```bash
python3 -m unittest research/option_critic/tests/test_option_critic.py
```

### Phase 2: Imitation Learning Pre-Training
```bash
python3 research/option_critic/train_il.py --episodes 80 --epochs 10 --device cuda
```

### Phase 3: Standard Option-Critic RL
```bash
python3 research/option_critic/train_rl.py --episodes 200 --delib_cost 0.05 --device cuda
```

### Phase 4: CF-OC Training
```bash
python3 research/option_critic/train_cf_oc.py --episodes 200 --cf_prob 0.20 --device cuda
```

### Phase 5: 500-Case Evaluation & McNemar Tests
```bash
python3 research/option_critic/evaluate.py --start_case 90800 --num_cases 500 --device cuda
```
