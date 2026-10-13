# CF-ICVL v0 — backbone-agnostic integrated navigation-value research prototype

**Status: UNIT-LEVEL PROTOTYPE ONLY. No CrowdNav training or evaluation has been run for this model.**
**Base:** Gmony p0-foundation-audit at `24ba2ac`. The old CAGIS results remain historical baselines, NOT evaluation of CF-ICVL.

## Hypothesis

In dense crowd navigation, greedy re-evaluation of 80 primitive actions may alternate
between competing passing tendencies even with recurrent history. Forcing commitment
everywhere introduces collision risk. A unified recurrent value learner might instead
learn the **conditional long-term return of (previous intent, next intent, primitive
action)**, using matched simulation branches to supervise difficult switch decisions.

The interesting gap is NOT "long-term intention" itself (Option-Critic covers learned
options/termination; COMPASS covers passing-side consistency with safety rules). The
**hypothesis to test**, not a novelty guarantee, is that state-matched counterfactual
return supervision improves the *relative ordering* of switch/continue/reactive
choices when jointly trained with IL and online RL.

## Algorithm at the research stage

1. Let `h_t = Encoder_theta(o_{t-H:t})`, with Encoder = GRU / KDA / Mamba.
   The temporal encoder and intention-conditioned critic share gradients.
2. Keep `z_(t-1)` as explicit policy state. Prototype modes are REACTIVE, LEFT,
   RIGHT, YIELD. These are **coarse directional proxies**: they are not yet
   track-ID-bound topological homotopy classes. A topology claim requires explicit
   per-person / per-group association across time.
3. Learn `Q_theta(h_t, z_(t-1), z_t, a_t)` for 4 intents x 80 primitive actions,
   plus an action-conditioned collision-risk prediction. The prior intent enters
   the network, not a fixed dwell-time counter or a post-hoc direction bonus.
4. Jointly select an intent and a safe primitive action from *learned* Q-values
   under shared geometric candidate feasibility; do not add
   `base_score + 0.10 * directional_bonus`.
   Apply common safety-filter/fallback logic and log all-unsafe cases.
5. Imitation: maximize marginal likelihood of expert primitive actions
   `-log p(a_teacher | history)` over latent intents. Do NOT invent expert
   passing-side labels from momentary steering.
6. Counterfactual preference: in an EXACT same-state snapshot, branch the
   controller with different modes, use matched continuation and comparable
   simulation random state, derive **sample** return differences
   `Delta_G=G(z_a)-G(z_b)`, train the option-value ordering with a calibrated
   margin loss. A single realization is not the expected advantage.
7. RL: update the selected (mode, action) joint value with a Bellman target whose
   next-state previous-mode is the selected mode; continue collecting trajectories
   under the evolving joint policy to reduce off-policy distribution shift.
8. Risk supervision: train collision outcome prediction from actual explored
   (mode, action) pairs, without promising collision-free behavior.

**NOTE**: This v0 module contains the differentiable joint critic and core losses.
It does not yet contain the CrowdNav production policy adapter, full replay-buffer
plumbing, mode/action semantic constraints, SMDP duration learning, official
KDA-on-CUDA integration, or on-policy RL trainer.

## Executable local unit test

```sh
python -m research.cf_icvl.test_core
```

Expected: PASS for output dimensions, action feasibility, IL/TD/CF/risk gradient
flow to the temporal encoder, and a synthetic paired-branch preference learning
test. These are *mechanics tests*, not navigation effectiveness evidence.

## Immediate implementation tasks

- **M0, wiring and parity:** Implement `IntegratedCFICVLPolicy` inside the
  decision/training graph, not as inference-time classifier. Start from the
  existing `kdarl_clean/crowd_nav/policy/mamba_rl.py` GRU/KDA/Mamba interface.
  Preserve the same 80 candidate commands, TTC5 feature contract, risk mask,
  0.25 s step, and environment event labels. Verify the chosen action is exactly
  the action actually scored; on unsupported KDA CPU, fail clearly (CUDA required).
- **M1, semantics:** Ensure passing options are behaviorally distinct. Start
  with a documented directional-proxy ablation. Upgrade to track-ID-bound
  passing relations only after implementing association and reset/lifecycle
  tests. Avoid calling a goal-relative heading a topological passing class.
- **M2, training:** Build IL initialization with *shared* encoder and joint
  Q; train counterfactual preference with grouped episodes and no test leakage;
  run RL updates which actually reach both the backbone and the joint critic.
  Keep the original scalar-V SARL baseline as an unchanged comparator.
- **M3, four decisive falsification arms:** (A) GRU+ordinary IL/RL joint
  value (no CF); (B) GRU+CF+IL/RL; (C) KDA+ordinary IL/RL joint value (no CF);
  (D) KDA+CF+IL/RL. Mamba optional under a matched budget.
  Also retain B0, CAGIS 83.2%-style frozen plugin, fixed dwell, simple
  switch penalty, standard Option-Critic/deliberation-cost reference.
  If adding CF gives no incremental gain under both backbones, reject the
  proposed core innovation. If KDA offers no gain, do not force KDA.
- **M4, evaluation:** Use only newly locked testing cases; Cases 90000–90799
  have already been used in prior research. Compare paired SR/CR/TR/SOR,
  per-case wins/losses, induced collisions, oscillatory switching, travel
  efficiency, and inference time. Report CIs and method training budget.

## Practical restrictions

- There is **no claim of publication readiness** or verified novelty.
- Cross-method candidate/reward/observation fairness is mandatory.
- Humans are configured `robot.visible=false`; no claim of reciprocal human
  navigation or human-robot negotiation should be made from these tests.
- Existing Gmony baseline KDA adapter uses the official FLA KDA layer and
  requires CUDA; this v0 CPU unit test exercises a GRU encoder only.
- Do not overwrite `p0-foundation-audit`, `main`, or the 500-case results.
- Any generated model improvement must be re-verified on held-out cases.

## Nearest comparison papers (to reproduce or implement matched ideas)

- Bacon et al., *The Option-Critic Architecture*, AAAI 2017.
- Harutyunyan et al., *Learning With Options That Terminate Off-Policy*, AAAI 2018.
- Mavrogiannis et al., *Social Momentum*, ACM THRI 2022.
- MODSRL, *Multi-objective crowd-aware robot navigation system using deep
  reinforcement learning*, Applied Soft Computing 2024.
- COMPASS, *Temporally Consistent Topological Passing Decisions*, Oct 2026
  **preprint**, not a peer-reviewed performance benchmark.
