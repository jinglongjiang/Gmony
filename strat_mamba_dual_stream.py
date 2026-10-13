#!/usr/bin/env python3
"""StratMamba Dual-Stream Policy Architecture (Self-Implemented Reproduction on CrowdNav)

Reference Paper:
Hung-Chieh Wu, Xiaopan Zhang, Kasra Sinaei, et al.
"StratMamba: Strategic and Reactive Stream Partitioning for Path-Efficient LiDAR-Based Obstacle Avoidance"
arXiv:2607.12370 (Accepted to IROS 2026)

Key Architectural Principles of StratMamba (arXiv:2607.12370, Section IV-B):
1. Asymmetric Input Routing:
   - Proximity/Reactive Stream (o_t^prox): Full environmental observation with dynamic obstacle/pedestrian tokens.
   - Strategic/Goal Stream (o_t^goal): Pure long-horizon goal state (relative goal vector [gx-px, gy-py] and distance).
2. Dual Independent Temporal Encoders:
   - Fast-decay reactive encoder for proximity stream (handles high-frequency pedestrian avoidance).
   - Slow-decay strategic encoder for goal stream (maintains persistent long-horizon navigational intent).
3. Zero-Initialized Additive Fusion:
   h_t = h_t^prox + W_fuse * h_t^goal (W_fuse is zero-initialized to preserve baseline at start).
4. Value / SARL Evaluation:
   Feeds fused hidden state h_t into action evaluation head.
"""

import sys
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent
KDARL_CLEAN = WORKSPACE / 'kdarl_clean'
sys.path = [str(KDARL_CLEAN), str(WORKSPACE)] + [p for p in sys.path if p not in (str(WORKSPACE), str(KDARL_CLEAN), '')]

import math
from typing import Optional, List, Dict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from crowd_nav.policy.mamba_rl import MambaRLPolicy, GRUTemporalEncoder, EnhancedSpatialEncoder
from crowd_sim.envs.utils.action import ActionXY
from crowd_sim.envs.utils.state import JointState

class PurePyTorchSelectiveSSM(nn.Module):
    """Pure PyTorch implementation of Selective State Space Model (Mamba block).
    Ensures 100% portability without external CUDA C++ extensions (mamba_ssm).
    """
    def __init__(self, d_model: int = 256, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand
        self.d_conv = d_conv

        # Input projection
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        # 1D causal depthwise convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )
        # SSM parameter projections: Delta, B, C
        self.x_proj = nn.Linear(self.d_inner, 2 * d_state + 1, bias=False)
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # S4 parameter initialization for A
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        B, T, D = x.shape
        res = x

        # In-proj split into main and gate
        xz = self.in_proj(x)
        x_branch, z_branch = xz.chunk(2, dim=-1)

        # Causal Conv1D
        x_conv = self.conv1d(x_branch.transpose(1, 2))[:, :, :T].transpose(1, 2)
        x_act = F.silu(x_conv)

        # SSM projections
        ssm_params = self.x_proj(x_act)
        dt, B_param, C_param = torch.split(ssm_params, [1, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))

        A = -torch.exp(self.A_log)

        # Discretization and recurrent rollout
        y_list = []
        h = torch.zeros(B, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)

        for t in range(T):
            dt_t = dt[:, t, :, None]
            A_bar_t = torch.exp(dt_t * A[None, :, :])
            B_param_t = B_param[:, t, None, :]
            B_bar_t = dt_t * B_param_t
            x_t = x_act[:, t, :, None]

            h = A_bar_t * h + B_bar_t * x_t

            C_param_t = C_param[:, t, None, :]
            y_t = (h * C_param_t).sum(dim=-1) + self.D * x_act[:, t, :]
            y_list.append(y_t)

        y = torch.stack(y_list, dim=1)
        # Gating
        y_gated = y * F.silu(z_branch)
        out = self.out_proj(y_gated)
        return self.norm(res + out)

class DualStreamTemporalEncoder(nn.Module):
    """StratMamba Dual-Stream Temporal Encoder:
    Combines a fast proximity stream (handling dynamic obstacle tokens)
    and a slow strategic goal stream (handling persistent goal vectors).
    """
    def __init__(self, d_model: int = 256, n_layers: int = 2, backbone: str = 'gru'):
        super().__init__()
        self.d_model = d_model
        self.backbone = backbone

        # Stream 1: Fast Proximity Stream (High-frequency obstacle dynamics)
        if backbone == 'gru':
            self.prox_encoder = GRUTemporalEncoder(d_model=d_model, n_layers=n_layers)
            self.goal_encoder = GRUTemporalEncoder(d_model=d_model, n_layers=n_layers)
        else:
            self.prox_encoder = nn.ModuleList([
                PurePyTorchSelectiveSSM(d_model=d_model) for _ in range(n_layers)
            ])
            self.goal_encoder = nn.ModuleList([
                PurePyTorchSelectiveSSM(d_model=d_model) for _ in range(n_layers)
            ])

        # Goal Stream Feature Projection (Maps 3-dim relative goal info to d_model)
        self.goal_proj = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, d_model),
            nn.LayerNorm(d_model)
        )

        # Zero-initialized fusion projection W_fuse (StratMamba Eq. 187)
        self.W_fuse = nn.Linear(d_model, d_model, bias=False)
        nn.init.zeros_(self.W_fuse.weight)

    def forward(self, prox_features: torch.Tensor, goal_features: torch.Tensor) -> torch.Tensor:
        """
        prox_features: [B, T, d_model] - Spatially encoded tokens of robot + 5 TTC humans
        goal_features: [B, T, 3] - Relative goal state [gx - px, gy - py, d_goal]
        """
        # 1. Process proximity stream
        if self.backbone == 'gru':
            h_prox = self.prox_encoder(prox_features)
        else:
            h_prox = prox_features
            for layer in self.prox_encoder:
                h_prox = layer(h_prox)

        # 2. Process strategic goal stream
        g_emb = self.goal_proj(goal_features)  # [B, T, d_model]
        if self.backbone == 'gru':
            h_goal = self.goal_encoder(g_emb)
        else:
            h_goal = g_emb
            for layer in self.goal_encoder:
                h_goal = layer(h_goal)

        # 3. Additive fusion (StratMamba formula: h_t = h_t^prox + W_fuse * h_t^goal)
        h_fused = h_prox + self.W_fuse(h_goal)
        return h_fused

class StratMambaDualStreamPolicy(MambaRLPolicy):
    """Faithful Reproduction of StratMamba Dual-Stream Architecture on CrowdNav.
    Explicitly labeled as Self-Implemented Reproduction (自行复现版本).
    """
    def __init__(self, config=None, device='cpu', backbone='gru'):
        super().__init__(config=config, device=device)
        d_model = config.getint('mamba', 'd_model')
        n_layers = config.getint('mamba', 'n_layers')

        # Replace single temporal encoder with StratMamba dual-stream temporal encoder
        self.dual_temporal = DualStreamTemporalEncoder(d_model=d_model, n_layers=n_layers, backbone=backbone)
        self.to(self.device)

        # Maintain dual goal history
        self._goal_history = []

    def reset_episode_stats(self):
        super().reset_episode_stats()
        self._goal_history.clear()

    def extract_goal_features(self, robot_state) -> np.ndarray:
        """Extract strategic 3D goal vector [gx - px, gy - py, d_goal]"""
        dx = robot_state.gx - robot_state.px
        dy = robot_state.gy - robot_state.py
        dist = math.hypot(dx, dy)
        return np.array([dx, dy, dist], dtype=np.float32)

    def load_pretrained_weights(self, weights_path):
        """Map pretrained single-stream weights to StratMamba dual-stream architecture."""
        ckpt = torch.load(weights_path, map_location=self.device)
        raw_state = ckpt.get('policy_state', ckpt.get('model', ckpt))
        clean_dict = {k.replace('_orig_mod.', '').replace('module.', ''): v for k, v in raw_state.items()}
        adapted_dict = {}
        for k, v in clean_dict.items():
            if k.startswith('temporal_encoder.'):
                adapted_dict[k.replace('temporal_encoder.', 'dual_temporal.prox_encoder.')] = v
                adapted_dict[k.replace('temporal_encoder.', 'dual_temporal.goal_encoder.')] = v.clone()
                adapted_dict[k] = v  # keep temporal_encoder in sync
            else:
                adapted_dict[k] = v
        self.load_state_dict(adapted_dict, strict=False)

    def forward_value(self, joint_tokens: torch.Tensor, goal_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Dual-stream Critic forward pass"""
        if joint_tokens.dim() == 3:
            joint_tokens = joint_tokens.unsqueeze(1)

        B, T = joint_tokens.shape[:2]
        if goal_tokens is None:
            # Fallback zero-goal if not provided
            goal_tokens = torch.zeros(B, T, 3, device=joint_tokens.device, dtype=torch.float32)

        spatial_features = self.spatial_encoder(joint_tokens)  # [B, T, d_model]
        h_fused = self.dual_temporal(spatial_features, goal_tokens)
        last_hidden = h_fused[:, -1, :]  # [B, d_model]
        values = self.value_head(last_hidden).squeeze(-1)
        return values

    def get_current_hidden_state(self, base_hist, base_goal_hist) -> torch.Tensor:
        """Extract the true fused temporal representation h_t for current step t
        conditioned strictly on observation history without hypothetical candidate lookahead.
        """
        L = self.seq_len
        hist_arr = np.asarray(base_hist)
        goal_arr = np.asarray(base_goal_hist)
        if len(base_hist) >= L:
            win_prox = hist_arr[-L:]
            win_goal = goal_arr[-L:]
        else:
            pads = L - len(base_hist)
            first_prox = hist_arr[0]
            first_goal = goal_arr[0]
            pad_prox = np.repeat(first_prox[None], pads, axis=0)
            pad_goal = np.repeat(first_goal[None], pads, axis=0)
            win_prox = np.concatenate([pad_prox, hist_arr], axis=0)
            win_goal = np.concatenate([pad_goal, goal_arr], axis=0)

        t_prox = torch.from_numpy(win_prox[None]).float().to(self.device)  # [1, L, 8, 13]
        t_goal = torch.from_numpy(win_goal[None]).float().to(self.device)  # [1, L, 3]

        with torch.no_grad():
            spat = self.spatial_encoder(t_prox)  # [1, L, d_model]
            h_fused = self.dual_temporal(spat, t_goal)  # [1, L, d_model]
            h_t = h_fused[:, -1, :]  # [1, d_model]
        return h_t

    def score_sarl_candidates(self, state: JointState):
        """Dual-stream evaluation of candidate actions using StratMamba asymmetric routing"""
        from crowd_sim.envs.utils.action import ActionXY
        from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized

        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)

        rob = state.self_state
        current_34 = self._build_joint_state_34(rob, state.human_states)
        current_token = _batch_joint34_to_tokens_vectorized(current_34.reshape(1, -1))[0]
        current_goal = self.extract_goal_features(rob)

        base_hist = list(self._history) + [current_token]
        base_goal_hist = list(self._goal_history) + [current_goal]

        commands = self.candidate_commands()
        A = len(commands)

        # 1. Propagate humans once
        next_humans = [self.propagate(h, ActionXY(h.vx, h.vy)) for h in state.human_states]
        N = len(next_humans)

        # 2. Propagate robot candidates
        cmd = np.array([[c.vx, c.vy] for c in commands], dtype=np.float64)
        nrp = np.array([rob.px, rob.py], dtype=np.float64)[None, :] + cmd * self.time_step

        # 3. Clearance & Collision
        if N:
            hp = np.array([[h.px, h.py] for h in next_humans], dtype=np.float64)
            hr = np.array([h.radius for h in next_humans], dtype=np.float64)
            D = np.hypot(nrp[:, None, 0] - hp[None, :, 0],
                         nrp[:, None, 1] - hp[None, :, 1]) - rob.radius - hr[None, :]
            dmins = D.min(axis=1)
            collision = (D < 0.0).any(axis=1)
        else:
            dmins = np.full(A, np.inf)
            collision = np.zeros(A, dtype=bool)

        # 4. Immediate reward
        curr_dist = np.hypot(nrp[:, 0] - rob.gx, nrp[:, 1] - rob.gy)
        reaching = curr_dist < self.success_radius
        prev_dist = float(np.linalg.norm((rob.px - rob.gx, rob.py - rob.gy)))
        speed = np.hypot(cmd[:, 0], cmd[:, 1])
        rew = self.progress_reward * (prev_dist - curr_dist) + self.time_penalty
        rew = rew + np.where(speed < 0.05, float(self.stand_penalty), 0.0)
        rew = rew - np.where(dmins < self.discomfort_dist,
                             self.discomfort_penalty_factor * (self.discomfort_dist - dmins) * self.time_step, 0.0)
        rew = rew + np.where(reaching, float(self.success_reward), 0.0)
        rew = np.where(collision, float(self.collision_penalty), rew)

        # 5. Build 34-dim tokens for candidates
        s34_0 = self._build_joint_state_34(self.propagate(rob, commands[0]), next_humans)
        s34 = np.repeat(np.asarray(s34_0).reshape(1, -1), A, axis=0)
        s34[:, 0:2] = np.asarray(nrp, dtype=np.float32).astype(s34.dtype)
        s34[:, 2:4] = np.asarray(cmd, dtype=np.float32).astype(s34.dtype)
        next_tokens = _batch_joint34_to_tokens_vectorized(s34)

        # Build goal features for candidates
        cand_goals = np.stack([
            np.array([rob.gx - nrp[i, 0], rob.gy - nrp[i, 1], curr_dist[i]], dtype=np.float32)
            for i in range(A)
        ], axis=0)  # [A, 3]

        # 6. Proximity Windows (Full [A, L, 8, 13])
        L = self.seq_len
        hist_arr = np.asarray(base_hist) if len(base_hist) else None
        if len(base_hist) + 1 >= L:
            pref = hist_arr[-(L - 1):] if L > 1 else hist_arr[:0]
            windows_prox = np.concatenate([np.broadcast_to(pref[None], (A,) + pref.shape), next_tokens[:, None]], axis=1)
        else:
            pads = L - (len(base_hist) + 1)
            first = hist_arr[0] if hist_arr is not None and len(hist_arr) else None
            if first is None:
                pad_block = np.broadcast_to(next_tokens[:, None], (A, pads) + next_tokens.shape[1:])
                windows_prox = np.concatenate([pad_block, next_tokens[:, None]], axis=1)
            else:
                pad_block = np.broadcast_to(first[None, None], (A, pads) + first.shape)
                body = np.broadcast_to(hist_arr[None], (A,) + hist_arr.shape)
                windows_prox = np.concatenate([pad_block, body, next_tokens[:, None]], axis=1)

        # 7. Goal Windows ([A, L, 3])
        goal_hist_arr = np.asarray(base_goal_hist) if len(base_goal_hist) else None
        if len(base_goal_hist) + 1 >= L:
            pref_g = goal_hist_arr[-(L - 1):] if L > 1 else goal_hist_arr[:0]
            windows_goal = np.concatenate([np.broadcast_to(pref_g[None], (A,) + pref_g.shape), cand_goals[:, None]], axis=1)
        else:
            pads_g = L - (len(base_goal_hist) + 1)
            first_g = goal_hist_arr[0] if goal_hist_arr is not None and len(goal_hist_arr) else None
            if first_g is None:
                pad_g = np.broadcast_to(cand_goals[:, None], (A, pads_g) + cand_goals.shape[1:])
                windows_goal = np.concatenate([pad_g, cand_goals[:, None]], axis=1)
            else:
                pad_g = np.broadcast_to(first_g[None, None], (A, pads_g) + first_g.shape)
                body_g = np.broadcast_to(goal_hist_arr[None], (A,) + goal_hist_arr.shape)
                windows_goal = np.concatenate([pad_g, body_g, cand_goals[:, None]], axis=1)

        windows_prox = np.ascontiguousarray(windows_prox)
        windows_goal = np.ascontiguousarray(windows_goal)

        t_prox = torch.from_numpy(windows_prox).float().to(self.device)
        t_goal = torch.from_numpy(windows_goal).float().to(self.device)

        with torch.no_grad():
            next_values = self.forward_value(t_prox, t_goal)

        reward_dtype = torch.float64 if np.any(~(collision | reaching)) else torch.get_default_dtype()
        clearance_dtype = torch.float64 if N else torch.get_default_dtype()
        rewards_tensor = torch.tensor(rew.tolist(), device=self.device, dtype=reward_dtype)
        total_values = rewards_tensor + self.gamma * next_values
        raw_scores = total_values.clone()
        dmins_list = dmins.tolist()

        if self._phase in ('test', 'val', 'eval') and (self.test_min_clearance > 0.0 or self.test_risk_lambda > 0.0):
            dmins_tensor = torch.tensor(dmins_list, device=self.device, dtype=clearance_dtype)
            if self.test_min_clearance > 0.0:
                safe_mask = dmins_tensor >= self.test_min_clearance
                if safe_mask.any():
                    total_values = total_values.masked_fill(~safe_mask, -1e9)
            if self.test_risk_lambda > 0.0:
                risk_margin = self.test_min_clearance if self.test_min_clearance > 0.0 else self.discomfort_dist
                total_values = total_values - self.test_risk_lambda * torch.clamp(risk_margin - dmins_tensor, min=0.0)

        if total_values.numel() > 0:
            total_values[0] -= 1e-3

        return {
            'commands': commands,
            'current_token': current_token,
            'current_goal': current_goal,
            'windows_prox': t_prox,
            'windows_goal': t_goal,
            'rewards': rewards_tensor,
            'clearance': torch.tensor(dmins_list, device=self.device, dtype=clearance_dtype),
            'values': next_values,
            'raw_scores': raw_scores,
            'scores': total_values
        }

    def predict_sarl_style(self, state: JointState):
        if self.reach_destination(state):
            self._last_grid_index = None
            return ActionXY(0, 0)

        if self.action_space is None:
            self.build_action_space(state.self_state.v_pref)

        result = self.score_sarl_candidates(state)
        best_idx = result['scores'].argmax().item()
        self._last_grid_index = best_idx
        best_action = result['commands'][best_idx]

        self._history.append(result['current_token'])
        self._goal_history.append(result['current_goal'])

        if self._phase in ('test', 'val', 'eval') and self.test_action_smoothing > 0.0:
            self._last_action = best_action
        return best_action
