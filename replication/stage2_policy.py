import os
import sys
import json
import math
from pathlib import Path
import numpy as np
import torch
import joblib

REPO_ROOT = Path('/home/abc/workspace/autonomous_research_20261012')
sys.path.insert(0, str(REPO_ROOT))

from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY
from run_stage4_500cases_evaluation import (
    Stage4ControllerPolicy,
    load_stage4_policy,
    derive_macro_intent_from_action,
    compute_intent_compatibility,
    FEATURE_KEYS
)

STAGE2_FEATURE_KEYS = FEATURE_KEYS + [
    'dwell_step',
    'score_gap',
    'intent_compat_curr',
    'clearance_margin'
]

class Stage2TerminationPolicy(Stage4ControllerPolicy):
    def __init__(
        self,
        termination_mode='fixed_k4',  # 'fixed_k4', 'forced_step4', 'cusum', 'learned', 'shuffled'
        config=None,
        device='cpu',
        cusum_params=None,  # dict(decay=1.0, threshold=0.2, floor=0.0)
        learned_model=None,
        learned_scaler=None,
        learned_threshold=0.5,
        max_dwell_cap=12  # Safety cap to avoid infinite commitment
    ):
        super().__init__(mode='learned_lr', config=config, device=device)
        self.termination_mode = termination_mode
        self.cusum_params = cusum_params or {'decay': 1.0, 'threshold': 0.2, 'floor': 0.0}
        self.learned_model = learned_model
        self.learned_scaler = learned_scaler
        self.learned_threshold = learned_threshold
        self.max_dwell_cap = max_dwell_cap

        # Stage 2 tracking buffers
        self.dwell_steps_held = 0
        self.cusum_accumulator = 0.0
        self.dwell_history = []  # records length of each commitment
        self.last_term_features = None

    def reset_episode_stats(self):
        super().reset_episode_stats()
        self.dwell_steps_held = 0
        self.cusum_accumulator = 0.0
        self.dwell_history.clear()
        self.last_term_features = None

    def extract_termination_features(self, state: JointState, base_res, current_physical_clearance):
        base_feat = self.extract_observable_features(state, base_res)
        rob = state.self_state

        # Unconstrained best action
        best_cmd_unconstrained = base_res['commands'][base_res['scores'].argmax().item()]
        best_q_unconstrained = base_res['scores'].max().item()

        # Score gap & compatibility
        if self.committed_intent is not None:
            compat_all = compute_intent_compatibility(base_res['commands'], rob, self.committed_intent).to(self.device)
            gate = max(0.0, min(1.0, (current_physical_clearance - 0.22) / 0.15))
            alpha = 0.10 * gate
            cand_scores = base_res['scores'] + alpha * compat_all
            best_cmd_constrained = base_res['commands'][cand_scores.argmax().item()]
            q_constrained = base_res['scores'][cand_scores.argmax().item()].item()
            score_gap = max(0.0, best_q_unconstrained - q_constrained)

            # Compatibility of unconstrained best with current intent
            comp_single = compute_intent_compatibility([best_cmd_unconstrained], rob, self.committed_intent).item()
        else:
            score_gap = 0.0
            comp_single = 1.0

        feat = dict(base_feat)
        feat['dwell_step'] = float(self.dwell_steps_held + 1)
        feat['score_gap'] = float(score_gap)
        feat['intent_compat_curr'] = float(comp_single)
        feat['clearance_margin'] = float(current_physical_clearance - 0.22)
        return feat

    def check_termination_decision(self, state: JointState, base_res, current_physical_clearance):
        """
        Evaluates whether current commitment should terminate for the upcoming step.
        Returns True if commitment should terminate, False if it should continue.
        """
        if self.dwell_steps_held == 0:
            # First step of commitment: always execute at least step 1
            return False

        if self.dwell_steps_held >= self.max_dwell_cap:
            # Safety hard cap
            return True

        if self.termination_mode in ('fixed_k4', 'forced_step4'):
            # Terminate exactly when 4 steps have been held
            return (self.dwell_steps_held >= 4)

        feat = self.extract_termination_features(state, base_res, current_physical_clearance)
        self.last_term_features = feat

        if self.termination_mode == 'cusum':
            dt = 0.25
            D_t = feat['score_gap']
            decay = self.cusum_params.get('decay', 1.0)
            floor_val = self.cusum_params.get('floor', 0.0)
            threshold = self.cusum_params.get('threshold', 0.2)

            # COMPASS leaky CUSUM accumulator:
            # e_k = max(0, decay * e_{k-1} + (D_t - floor) * dt)
            self.cusum_accumulator = max(0.0, decay * self.cusum_accumulator + (D_t - floor_val) * dt)
            should_term = (self.cusum_accumulator >= threshold)
            if should_term:
                self.cusum_accumulator = 0.0
            return should_term

        elif self.termination_mode in ('learned', 'shuffled'):
            if self.learned_model is None:
                raise ValueError("Learned model not set for learned termination mode!")
            x = np.array([feat[k] for k in STAGE2_FEATURE_KEYS], dtype=np.float32).reshape(1, -1)
            if self.learned_scaler is not None:
                x = (x - self.learned_scaler['mean']) / (self.learned_scaler['scale'] + 1e-8)

            if hasattr(self.learned_model, 'predict_proba'):
                prob = float(self.learned_model.predict_proba(x)[0, 1])
                return (prob >= self.learned_threshold)
            elif hasattr(self.learned_model, 'predict'):
                pred = float(self.learned_model.predict(x)[0])
                return (pred > 0.0)
            else:
                # PyTorch model
                with torch.no_grad():
                    xt = torch.from_numpy(x).float().to(self.device)
                    out = self.learned_model(xt)
                    prob = float(torch.sigmoid(out).item())
                    return (prob >= self.learned_threshold)

        return False

    def score_sarl_candidates(self, state: JointState):
        base_res = super(Stage4ControllerPolicy, self).score_sarl_candidates(state)
        rob = state.self_state

        if len(state.human_states):
            curr_phys_dists = [
                math.hypot(rob.px - h.px, rob.py - h.py) - rob.radius - h.radius
                for h in state.human_states
            ]
            current_physical_clearance = float(min(curr_phys_dists))
        else:
            current_physical_clearance = 999.0

        # Emergency clearance abort (<0.22m)
        if current_physical_clearance < 0.22:
            if self.dwell_steps_held > 0:
                self.dwell_history.append(self.dwell_steps_held)
            self.dwell_steps_remaining = 0
            self.dwell_steps_held = 0
            self.committed_intent = None
            self.cusum_accumulator = 0.0

        features = self.extract_observable_features(state, base_res)
        self.last_decision_features = features

        # Check termination if currently committed
        if self.committed_intent is not None:
            should_terminate = self.check_termination_decision(state, base_res, current_physical_clearance)
            if should_terminate:
                self.dwell_history.append(self.dwell_steps_held)
                self.committed_intent = None
                self.dwell_steps_remaining = 0
                self.dwell_steps_held = 0
                self.cusum_accumulator = 0.0

        # Intervention activation check (when not committed)
        if self.committed_intent is None:
            best_cmd = base_res['commands'][base_res['scores'].argmax().item()]
            cand_intent = derive_macro_intent_from_action(rob, best_cmd)

            is_symptom = (features['ratio_4s'] >= 1.8 or features['reversals_4s'] >= 2)
            should_intervene = False
            if is_symptom:
                feat_vec = np.array([features[k] for k in FEATURE_KEYS], dtype=np.float32).reshape(1, -1)
                feat_norm = (feat_vec - self.scaler_mean) / (self.scaler_scale + 1e-8)
                prob = float(self.lr_model.predict_proba(feat_norm)[0, 1])
                should_intervene = (prob >= self.lr_threshold)

            if should_intervene:
                self.committed_intent = cand_intent
                self.dwell_steps_held = 0
                self.dwell_steps_remaining = 4  # maintained for backward compatibility
                self.intervention_count += 1
                self.cusum_accumulator = 0.0

        # Modulated intent commitment (if currently active)
        if self.committed_intent is not None:
            compat = compute_intent_compatibility(base_res['commands'], rob, self.committed_intent).to(self.device)
            gate = max(0.0, min(1.0, (current_physical_clearance - 0.22) / 0.15))
            alpha = 0.10 * gate
            modulated_scores = base_res['scores'] + alpha * compat

            if self._phase in ('test', 'val', 'eval') and self.test_min_clearance > 0.0:
                safe_mask = base_res['clearance'] >= self.test_min_clearance
                if safe_mask.any():
                    modulated_scores = modulated_scores.masked_fill(~safe_mask, -1e9)

            base_res['scores'] = modulated_scores
            self.dwell_steps_held += 1
            if self.dwell_steps_remaining > 0:
                self.dwell_steps_remaining -= 1

        return base_res

def load_stage2_policy(
    termination_mode='forced_step4',
    device='cpu',
    cusum_params=None,
    learned_model=None,
    learned_scaler=None,
    learned_threshold=0.5,
    max_dwell_cap=12
):
    from run_b0_counterfactual_study import get_base_config
    from run_selection_study_50cases import install_selection

    cfg = get_base_config()
    policy = Stage2TerminationPolicy(
        termination_mode=termination_mode,
        config=cfg,
        device=device,
        cusum_params=cusum_params,
        learned_model=learned_model,
        learned_scaler=learned_scaler,
        learned_threshold=learned_threshold,
        max_dwell_cap=max_dwell_cap
    )

    ckpt_path = REPO_ROOT / 'baseline_weights/gru-t24-seed443.pth'
    ckpt = torch.load(ckpt_path, map_location=device)
    raw_state = ckpt.get('policy_state', ckpt.get('model', ckpt))
    clean_dict = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in raw_state.items()}
    policy.load_state_dict(clean_dict, strict=True)
    policy.eval()
    policy.build_action_space(1.0)
    policy.use_sarl_predict = True
    policy.set_phase('test')
    policy.test_min_clearance = 0.2
    policy.test_risk_lambda = 0.8
    policy.test_action_smoothing = 0.3
    install_selection(policy, 'ttc5')
    policy.init_learned_models()

    return policy

