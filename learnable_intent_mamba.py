#!/usr/bin/env python3
"""Learnable Intent Commitment & Switching (LICS) Architecture on Dual-Stream Mamba

Implements:
1. LICSIntentHead: SMDP / Option-Critic Intent Selection and Termination Module
   - Discrete Passing Intent Space I in {0: Bypass Left, 1: Direct Advance, 2: Bypass Right, 3: Yield}
   - Intent Policy pi_phi(I | h_t)
   - Learnable Termination Function beta_psi(h_t, I_{t-1})
   - Hysteresis Regularization for commitment
   - Reactive safety bailout fallback
2. LICSStratMambaPolicy: Full hierarchical policy combining StratMamba dual-stream with LICS
3. HeuristicHysteresisPolicy: Fixed dwell-time baseline (B3) for fair ablation
"""

import sys
from pathlib import Path
import math
from typing import Optional, List, Dict, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

WORKSPACE = Path('/home/abc/workspace/autonomous_research_20261012')
KDARL_CLEAN = WORKSPACE / 'kdarl_clean'
sys.path = [str(KDARL_CLEAN), str(WORKSPACE)] + [p for p in sys.path if p not in (str(WORKSPACE), str(KDARL_CLEAN), '')]

from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import JointState
from strat_mamba_dual_stream import StratMambaDualStreamPolicy, DualStreamTemporalEncoder

INTENT_BYPASS_LEFT = 0
INTENT_DIRECT = 1
INTENT_BYPASS_RIGHT = 2
INTENT_YIELD = 3
INTENT_NAMES = {0: 'Bypass_Left', 1: 'Direct', 2: 'Bypass_Right', 3: 'Yield'}

class LICSIntentHead(nn.Module):
    """Semi-Markov Intent Head with Learnable Commitment & Switching.
    
    Parameters:
    - d_model: feature dimension of the fused dual-stream backbone (256)
    - d_intent: intent embedding dimension (64)
    - num_intents: 4 discrete macro-passing intents
    - hysteresis_margin: cost penalty eta to discourage frivolous switching
    """
    def __init__(self, d_model: int = 256, d_intent: int = 64, num_intents: int = 4, hysteresis_margin: float = 0.15):
        super().__init__()
        self.d_model = d_model
        self.d_intent = d_intent
        self.num_intents = num_intents
        self.hysteresis_margin = hysteresis_margin

        # Intent Embeddings
        self.intent_embeddings = nn.Embedding(num_intents, d_intent)

        # Macro Intent Q-network: Q_phi(h_t, I)
        self.intent_q_net = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, num_intents)
        )

        # Learnable Termination Network: beta_psi(h_t, I_{t-1}) in [0, 1]
        self.termination_net = nn.Sequential(
            nn.Linear(d_model + d_intent, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        # Learnable Intent Guidance Weight alpha
        self.intent_guidance_weight = nn.Parameter(torch.tensor([0.10], dtype=torch.float32))

    def get_termination_prob(self, h_fused: torch.Tensor, prev_intent: int) -> torch.Tensor:
        """Compute beta_psi(h_t, I_{t-1}) in [0, 1]"""
        # h_fused: [B, d_model]
        device = h_fused.device
        intent_tensor = torch.tensor([prev_intent], dtype=torch.long, device=device).expand(h_fused.shape[0])
        emb = self.intent_embeddings(intent_tensor)  # [B, d_intent]
        cat_feat = torch.cat([h_fused, emb], dim=-1)  # [B, d_model + d_intent]
        beta = self.termination_net(cat_feat).squeeze(-1)  # [B]
        return beta

    def get_intent_q_values(self, h_fused: torch.Tensor) -> torch.Tensor:
        """Compute Q_phi(h_t, I) for all I in {0, 1, 2, 3}"""
        return self.intent_q_net(h_fused)  # [B, num_intents]

    def compute_action_intent_compatibility(self, commands: List[ActionXY], robot_state, intent: int) -> torch.Tensor:
        """Calculates compatibility score mu(a | I) between candidate actions and intent."""
        # Goal angle relative to world frame
        gx, gy = robot_state.gx, robot_state.gy
        px, py = robot_state.px, robot_state.py
        theta_goal = math.atan2(gy - py, gx - px)

        A = len(commands)
        scores = np.zeros(A, dtype=np.float32)

        for i, cmd in enumerate(commands):
            vx, vy = cmd.vx, cmd.vy
            speed = math.hypot(vx, vy)
            v_pref = max(robot_state.v_pref, 0.01)
            norm_speed = min(speed / v_pref, 1.0)

            if speed < 0.05:
                # Stopped / idle
                if intent == INTENT_YIELD:
                    scores[i] = 1.0
                else:
                    scores[i] = -0.5
                continue

            theta_a = math.atan2(vy, vx)
            # Angular error wrapped to [-pi, pi]
            diff = (theta_a - theta_goal + math.pi) % (2 * math.pi) - math.pi

            if intent == INTENT_BYPASS_LEFT:
                # Prefer moving to the left (+45 deg offset)
                target_offset = math.pi / 4.0
                scores[i] = math.cos(diff - target_offset) * norm_speed
            elif intent == INTENT_DIRECT:
                # Direct advance towards goal
                scores[i] = math.cos(diff) * norm_speed
            elif intent == INTENT_BYPASS_RIGHT:
                # Prefer moving to the right (-45 deg offset)
                target_offset = -math.pi / 4.0
                scores[i] = math.cos(diff - target_offset) * norm_speed
            elif intent == INTENT_YIELD:
                # Prefer low speed or stopping
                scores[i] = (1.0 - norm_speed) * 0.8
            else:
                scores[i] = 0.0

        return torch.tensor(scores, dtype=torch.float32)

class LICSStratMambaPolicy(StratMambaDualStreamPolicy):
    """Arm B2: Learnable Intent Commitment & Switching on Dual-Stream Mamba (LICS).
    Combines StratMamba dual-stream spatial-temporal representation with
    Option-Critic / SMDP macro-intent commitment and switching.
    """
    def __init__(self, config=None, device='cpu', backbone='gru', hysteresis_margin: float = 0.15):
        super().__init__(config=config, device=device, backbone=backbone)
        d_model = config.getint('mamba', 'd_model')

        # Add LICS Intent Head
        self.intent_head = LICSIntentHead(
            d_model=d_model,
            d_intent=64,
            num_intents=4,
            hysteresis_margin=hysteresis_margin
        )
        self.to(self.device)

        # Episode state tracking
        self.active_intent: Optional[int] = None
        self.intent_dwell_steps: int = 0
        self.intent_switch_history: List[int] = []
        self.termination_prob_history: List[float] = []

    def reset_episode_stats(self):
        super().reset_episode_stats()
        self.active_intent = None
        self.intent_dwell_steps = 0
        self.intent_switch_history.clear()
        self.termination_prob_history.clear()

    def get_fused_hidden_state(self, windows_prox: torch.Tensor, windows_goal: torch.Tensor) -> torch.Tensor:
        """Extract last fused temporal representation h_t from dual streams"""
        spatial_features = self.spatial_encoder(windows_prox)  # [1, T, d_model]
        h_fused = self.dual_temporal(spatial_features, windows_goal)  # [1, T, d_model]
        return h_fused[:, -1, :]  # [1, d_model]

    def select_or_update_intent(self, h_fused: torch.Tensor, robot_state, min_clearance: float) -> Tuple[int, float, bool]:
        """Option-Critic intent commitment / switching decision:
        Returns: (intent, beta_prob, switched_flag)
        """
        # Reactive safety bailout: if clearance drops below 0.22m, force immediate termination
        if min_clearance < 0.22:
            beta = 1.0
        elif self.active_intent is None:
            beta = 1.0
        else:
            with torch.no_grad():
                beta_t = self.intent_head.get_termination_prob(h_fused, self.active_intent).item()
            beta = beta_t

        switched = False
        if beta >= 0.5 or self.active_intent is None:
            # Switch to new intent using Intent Q-network
            with torch.no_grad():
                q_vals = self.intent_head.get_intent_q_values(h_fused).squeeze(0)  # [4]
                # If currently holding an intent, apply hysteresis penalty to discourage frivolous switching
                if self.active_intent is not None:
                    penalty = torch.zeros_like(q_vals)
                    for i in range(4):
                        if i != self.active_intent:
                            penalty[i] = self.intent_head.hysteresis_margin
                    q_vals = q_vals - penalty
                new_intent = q_vals.argmax().item()

            if new_intent != self.active_intent:
                switched = True
                self.active_intent = new_intent
                self.intent_dwell_steps = 1
            else:
                self.intent_dwell_steps += 1
        else:
            # Commit to active intent
            self.intent_dwell_steps += 1

        self.intent_switch_history.append(self.active_intent)
        self.termination_prob_history.append(beta)
        return self.active_intent, beta, switched

    def score_sarl_candidates(self, state: JointState):
        """Hierarchical scoring: Evaluates base dual-stream SARL scores and modulates by committed intent."""
        # 1. Base StratMamba scoring
        base_res = super().score_sarl_candidates(state)
        rob = state.self_state
        # Distinguish current actual physical clearance from candidate endpoint clearances
        if len(state.human_states):
            curr_phys_dists = [
                math.hypot(rob.px - h.px, rob.py - h.py) - rob.radius - h.radius
                for h in state.human_states
            ]
            current_physical_clearance = float(min(curr_phys_dists))
        else:
            current_physical_clearance = 999.0

        # 2. Extract pure fused state from actual observation history up to current step
        base_hist = list(self._history) + [base_res['current_token']]
        base_goal_hist = list(self._goal_history) + [base_res['current_goal']]
        h_fused = self.get_current_hidden_state(base_hist, base_goal_hist)  # [1, d_model]

        # 3. Macro Intent Commitment / Switching (evaluated using current physical clearance)
        intent, beta, switched = self.select_or_update_intent(h_fused, rob, current_physical_clearance)

        # 4. Modulate candidate actions by committed intent with Clearance-Gated Safety Adaptation
        compat = self.intent_head.compute_action_intent_compatibility(base_res['commands'], rob, intent)
        compat = compat.to(self.device)

        # Clearance gate based on current physical clearance:
        # If robot is already close (<0.22m), gate=0 (pure reactive lookahead).
        # When robot has clearance >0.35m, full intent guidance operates.
        gate = max(0.0, min(1.0, (current_physical_clearance - 0.22) / 0.15))
        base_alpha = float(torch.clamp(self.intent_head.intent_guidance_weight, 0.02, 0.20).item())
        alpha = base_alpha * gate
        modulated_scores = base_res['scores'] + alpha * compat

        # Re-apply safety mask to guarantee safety filter integrity
        if self._phase in ('test', 'val', 'eval') and self.test_min_clearance > 0.0:
            safe_mask = base_res['clearance'] >= self.test_min_clearance
            if safe_mask.any():
                modulated_scores = modulated_scores.masked_fill(~safe_mask, -1e9)

        base_res['scores'] = modulated_scores
        base_res['active_intent'] = intent
        base_res['termination_prob'] = beta
        base_res['switched'] = switched
        return base_res

class HeuristicHysteresisPolicy(StratMambaDualStreamPolicy):
    """Arm B3: Heuristic Hysteresis Baseline.
    Holds committed macro-direction for fixed dwell-time K steps, with emergency safety abort.
    """
    def __init__(self, config=None, device='cpu', backbone='gru', dwell_k: int = 4):
        super().__init__(config=config, device=device, backbone=backbone)
        self.dwell_k = dwell_k
        self.committed_intent: Optional[int] = None
        self.steps_remaining: int = 0
        self.intent_history: List[int] = []

    def reset_episode_stats(self):
        super().reset_episode_stats()
        self.committed_intent = None
        self.steps_remaining = 0
        self.intent_history.clear()

    def score_sarl_candidates(self, state: JointState):
        base_res = super().score_sarl_candidates(state)
        rob = state.self_state
        if len(state.human_states):
            curr_phys_dists = [
                math.hypot(rob.px - h.px, rob.py - h.py) - rob.radius - h.radius
                for h in state.human_states
            ]
            current_physical_clearance = float(min(curr_phys_dists))
        else:
            current_physical_clearance = 999.0

        # Safety abort based on current physical clearance
        if current_physical_clearance < 0.22:
            self.steps_remaining = 0
            self.committed_intent = None

        if self.steps_remaining <= 0 or self.committed_intent is None:
            # Choose intent from best action heading
            best_raw_idx = base_res['scores'].argmax().item()
            best_cmd = base_res['commands'][best_raw_idx]
            gx, gy = rob.gx, rob.gy
            px, py = rob.px, rob.py
            theta_goal = math.atan2(gy - py, gx - px)
            theta_a = math.atan2(best_cmd.vy, best_cmd.vx)
            diff = (theta_a - theta_goal + math.pi) % (2 * math.pi) - math.pi

            if math.hypot(best_cmd.vx, best_cmd.vy) < 0.1:
                intent = INTENT_YIELD
            elif diff > 0.15:
                intent = INTENT_BYPASS_LEFT
            elif diff < -0.15:
                intent = INTENT_BYPASS_RIGHT
            else:
                intent = INTENT_DIRECT

            self.committed_intent = intent
            self.steps_remaining = self.dwell_k
        else:
            self.steps_remaining -= 1

        self.intent_history.append(self.committed_intent)

        # Modulate scores with heuristic compatibility
        from learnable_intent_mamba import LICSIntentHead
        dummy_head = LICSIntentHead()
        compat = dummy_head.compute_action_intent_compatibility(base_res['commands'], rob, self.committed_intent).to(self.device)
        # Controlled ablation: exactly identical clearance gate and alpha=0.10 as B2
        gate = max(0.0, min(1.0, (current_physical_clearance - 0.22) / 0.15))
        alpha = 0.10 * gate
        modulated_scores = base_res['scores'] + alpha * compat
        if self._phase in ('test', 'val', 'eval') and self.test_min_clearance > 0.0:
            safe_mask = base_res['clearance'] >= self.test_min_clearance
            if safe_mask.any():
                modulated_scores = modulated_scores.masked_fill(~safe_mask, -1e9)

        base_res['scores'] = modulated_scores
        base_res['active_intent'] = self.committed_intent
        return base_res
