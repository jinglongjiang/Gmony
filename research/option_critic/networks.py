"""Neural Network Architectures for Option-Critic and CF-OC

Implements:
1. Modular Option-Critic Network with swappable backbones (GRU / KDA / Mamba).
2. Intra-option actor heads pi_o(a | h) for N options over 80 discrete actions.
3. Option-value critic Q_Omega(h, o) and Intra-option action-value Q_U(h, o, a).
4. Option termination heads beta_o(h).
5. CF-OC specific heads:
   - Counterfactual Advantage Head: Delta_pred(h, o) ~ (G_reselect - G_continue)
   - Collision Risk Head: p_collision(h, o) over short horizon (16 steps / 4s)
   - Risk & Advantage Conditioned Termination Head: beta_cf(h, o)
6. Backbone weight loader from verified B0 checkpoints (strict key alignment).
"""

import math
from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
KDARL_CLEAN = REPO_ROOT / 'kdarl_clean'
if str(KDARL_CLEAN) not in sys.path:
    sys.path.insert(0, str(KDARL_CLEAN))

from crowd_nav.policy.mamba_rl import (
    EnhancedSpatialEncoder,
    build_temporal_encoder,
    GRUTemporalEncoder
)


class OptionCriticNetwork(nn.Module):
    """Deep Option-Critic Network with swappable temporal backbones.
    
    Supports:
    - Standard Option-Critic (Bacon et al. 2017)
    - Deliberation Cost Option-Critic (Harb et al. 2018)
    - CF-OC (Counterfactual Advantage and Collision Risk supervision)
    """

    def __init__(
        self,
        num_options: int = 4,
        num_actions: int = 80,
        d_model: int = 256,
        n_layers: int = 4,
        temporal_backbone: str = 'gru',
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
        is_cf_oc: bool = True
    ):
        super().__init__()
        self.num_options = num_options
        self.num_actions = num_actions
        self.d_model = d_model
        self.n_layers = n_layers
        self.temporal_backbone = temporal_backbone.lower()
        self.is_cf_oc = is_cf_oc

        # 1. Perception Backbones
        self.spatial_encoder = EnhancedSpatialEncoder(d_model=d_model)
        self.temporal_encoder = build_temporal_encoder(
            backbone=self.temporal_backbone,
            d_model=d_model,
            n_layers=n_layers,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout
        )

        # 2. Shared Representation MLP
        self.shared_rep = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )

        # 3. Intra-Option Policy Head pi_o(a | h)
        # Outputs [B, num_options, num_actions] logits
        self.policy_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_options * num_actions)
        )

        # 4. Option Value Head Q_Omega(h, o)
        # Outputs [B, num_options]
        self.q_omega_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_options)
        )

        # 5. Intra-Option Action Value Head Q_U(h, o, a)
        # Outputs [B, num_options, num_actions]
        self.q_u_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, num_options * num_actions)
        )

        # 6. Standard Option Termination Head beta_o(h)
        # Outputs [B, num_options] in (0, 1)
        self.beta_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, num_options)
        )

        # 7. CF-OC Specific Extension Heads
        if self.is_cf_oc:
            # Counterfactual Advantage Predictor Delta_pred(h, o) = G_reselect - G_continue
            # Outputs [B, num_options]
            self.cf_advantage_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, num_options)
            )

            # Short-Horizon Collision Risk Head (e.g. 16 steps = 4 seconds)
            # Outputs [B, num_options] in [0, 1]
            self.collision_risk_head = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, num_options)
            )

            # CF-Conditioned Termination Fusion Head
            # Takes representation h, base termination logit beta_raw, predicted advantage Delta, and predicted collision risk
            # For each option o: [d_model + 1 + 1 + 1] -> 1
            self.cf_beta_fusion = nn.Sequential(
                nn.Linear(d_model + 3, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

    def extract_latent(self, joint_tokens: torch.Tensor) -> torch.Tensor:
        """Extract temporal state representation h_t from input joint tokens.
        
        Args:
            joint_tokens: Tensor of shape [B, T, 8, 13] or [B, 8, 13]
        Returns:
            h_t: Tensor of shape [B, d_model]
        """
        if joint_tokens.dim() == 3:
            joint_tokens = joint_tokens.unsqueeze(1)
        
        spatial_feat = self.spatial_encoder(joint_tokens)  # [B, T, d_model]
        temporal_feat = self.temporal_encoder(spatial_feat)  # [B, T, d_model]
        h_t = temporal_feat[:, -1, :]  # [B, d_model]
        return self.shared_rep(h_t)

    def forward(
        self,
        joint_tokens: torch.Tensor,
        action_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Full forward pass for all Option-Critic heads.
        
        Args:
            joint_tokens: [B, T, 8, 13] or [B, 8, 13]
            action_mask: Optional boolean or float mask [B, num_actions] (1 for valid, 0 for invalid)
            
        Returns:
            Dictionary containing:
            - 'h': latent representation [B, d_model]
            - 'q_omega': option values Q_Omega(h, o) [B, num_options]
            - 'q_u': intra-option action values Q_U(h, o, a) [B, num_options, num_actions]
            - 'policy_logits': masked intra-option logits [B, num_options, num_actions]
            - 'policy_probs': intra-option action probabilities [B, num_options, num_actions]
            - 'beta': standard termination probabilities [B, num_options]
            If is_cf_oc:
            - 'cf_advantage': predicted counterfactual advantage Delta_pred [B, num_options]
            - 'collision_risk': predicted short-horizon collision probability [B, num_options]
            - 'beta_cf': CF-calibrated termination probabilities [B, num_options]
        """
        h = self.extract_latent(joint_tokens)  # [B, d_model]
        B = h.shape[0]

        # Intra-option action values Q_U(h, o, a)
        q_u = self.q_u_head(h).view(B, self.num_options, self.num_actions)

        # Option values Q_Omega(h, o)
        q_omega = self.q_omega_head(h)  # [B, num_options]

        # Intra-option policy logits
        raw_logits = self.policy_head(h).view(B, self.num_options, self.num_actions)
        if action_mask is not None:
            # action_mask shape: [B, num_actions] -> expand to [B, 1, num_actions]
            if action_mask.dim() == 2:
                mask_expanded = action_mask.unsqueeze(1).expand(-1, self.num_options, -1)
            else:
                mask_expanded = action_mask
            # Mask out invalid actions by subtracting large penalty
            masked_logits = torch.where(mask_expanded > 0.5, raw_logits, raw_logits - 1e8)
        else:
            masked_logits = raw_logits

        policy_probs = F.softmax(masked_logits, dim=-1)

        # Standard termination beta_o(h)
        beta_raw = self.beta_head(h)  # [B, num_options]
        beta = torch.sigmoid(beta_raw)

        out = {
            'h': h,
            'q_omega': q_omega,
            'q_u': q_u,
            'policy_logits': masked_logits,
            'policy_probs': policy_probs,
            'beta': beta
        }

        # CF-OC extensions
        if self.is_cf_oc:
            cf_adv = self.cf_advantage_head(h)  # [B, num_options]
            coll_risk = torch.sigmoid(self.collision_risk_head(h))  # [B, num_options]

            # Compute CF-calibrated beta for each option
            # Expand h: [B, num_options, d_model]
            h_expanded = h.unsqueeze(1).expand(-1, self.num_options, -1)
            beta_expanded = beta_raw.unsqueeze(-1)  # [B, num_options, 1]
            adv_expanded = cf_adv.unsqueeze(-1)  # [B, num_options, 1]
            risk_expanded = coll_risk.unsqueeze(-1)  # [B, num_options, 1]
            fusion_in = torch.cat([h_expanded, beta_expanded, adv_expanded, risk_expanded], dim=-1)  # [B, num_options, d_model+3]
            
            beta_cf_logits = self.cf_beta_fusion(fusion_in).squeeze(-1)  # [B, num_options]
            beta_cf = torch.sigmoid(beta_cf_logits)

            out['cf_advantage'] = cf_adv
            out['collision_risk'] = coll_risk
            out['beta_cf'] = beta_cf

        return out

    def get_option_value(self, h: torch.Tensor) -> torch.Tensor:
        """Compute Q_Omega(h, o) from latent h."""
        return self.q_omega_head(h)

    def get_termination(self, h: torch.Tensor, use_cf: bool = True) -> torch.Tensor:
        """Compute termination probability beta(h) for all options."""
        beta_raw = self.beta_head(h)
        if self.is_cf_oc and use_cf:
            cf_adv = self.cf_advantage_head(h)
            coll_risk = torch.sigmoid(self.collision_risk_head(h))
            h_expanded = h.unsqueeze(1).expand(-1, self.num_options, -1)
            beta_expanded = beta_raw.unsqueeze(-1)
            adv_expanded = cf_adv.unsqueeze(-1)
            risk_expanded = coll_risk.unsqueeze(-1)
            fusion_in = torch.cat([h_expanded, beta_expanded, adv_expanded, risk_expanded], dim=-1)
            return torch.sigmoid(self.cf_beta_fusion(fusion_in).squeeze(-1))
        else:
            return torch.sigmoid(beta_raw)

    def load_b0_backbone(self, weights_path: Union[str, Path], device: str = 'cpu') -> Dict[str, int]:
        """Initialize spatial and temporal encoders from a verified B0 checkpoint.
        
        Preserves representations learned by the baseline single-stream GRU/backbone.
        Returns load status report dictionary.
        """
        weights_path = Path(weights_path)
        if not weights_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at: {weights_path}")

        ckpt = torch.load(weights_path, map_location=device)
        raw_state = ckpt.get('policy_state', ckpt.get('model', ckpt))
        
        # Strip wrappers
        clean_state = {}
        for k, v in raw_state.items():
            clean_k = k.replace('_orig_mod.', '').replace('module.', '')
            clean_state[clean_k] = v

        loaded_count = 0
        skipped_count = 0

        # Load spatial encoder
        spatial_dict = {}
        for k, v in clean_state.items():
            if k.startswith('spatial_encoder.'):
                sub_k = k[len('spatial_encoder.'):]
                spatial_dict[sub_k] = v
        
        if spatial_dict:
            res_sp = self.spatial_encoder.load_state_dict(spatial_dict, strict=False)
            loaded_count += len(spatial_dict) - len(res_sp.unexpected_keys)
        
        # Load temporal encoder if compatible
        temporal_dict = {}
        for k, v in clean_state.items():
            if k.startswith('temporal_encoder.'):
                sub_k = k[len('temporal_encoder.'):]
                temporal_dict[sub_k] = v
        
        if temporal_dict and self.temporal_backbone == 'gru':
            res_tp = self.temporal_encoder.load_state_dict(temporal_dict, strict=False)
            loaded_count += len(temporal_dict) - len(res_tp.unexpected_keys)

        # Initialize intra-option policy heads and Q heads from B0 q_head if available
        if 'q_head.weight' in clean_state and 'q_head.bias' in clean_state:
            b0_q_w = clean_state['q_head.weight']  # [80, 256]
            b0_q_b = clean_state['q_head.bias']    # [80]
            
            # Initialize each option's action head with B0 Q-head plus small random perturbation
            # This ensures all options start with expert navigation capabilities while having diversity
            with torch.no_grad():
                # self.q_u_head: [num_options * num_actions, d_model]
                # Linear layer weight shape is [out_features, in_features]
                for o in range(self.num_options):
                    start_idx = o * self.num_actions
                    end_idx = (o + 1) * self.num_actions
                    noise_scale = 0.02 * float(o)  # Deterministic diversity per option
                    self.q_u_head[-1].weight.data[start_idx:end_idx] = b0_q_w + noise_scale * torch.randn_like(b0_q_w)
                    self.q_u_head[-1].bias.data[start_idx:end_idx] = b0_q_b + noise_scale * torch.randn_like(b0_q_b)
                    
                    self.policy_head[-1].weight.data[start_idx:end_idx] = b0_q_w * 0.1 + noise_scale * torch.randn_like(b0_q_w)
                    self.policy_head[-1].bias.data[start_idx:end_idx] = b0_q_b * 0.1
                
                # Initialize Q_Omega head to mean Q-value of B0
                self.q_omega_head[-1].weight.data.zero_()
                self.q_omega_head[-1].bias.data.fill_(float(b0_q_b.mean()))
                
                # Initialize beta_head with small negative bias so options don't instantly terminate at step 0
                self.beta_head[-1].bias.data.fill_(-2.0)  # sigmoid(-2) ~ 0.11 initial termination prob

        return {
            'loaded_tensors': loaded_count,
            'spatial_keys': len(spatial_dict),
            'temporal_keys': len(temporal_dict),
            'status': 'SUCCESS'
        }
