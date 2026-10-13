"""CF-ICVL experimental core (NOT a trained CrowdNav policy).

Representations are supplied by a trainable GRU/Mamba/KDA encoder. The
network values joint (intent, primitive-action) decisions conditioned on the
previous intent. All losses can backpropagate into the encoder; nothing is
added as a test-time bonus to a frozen base critic.

This first implementation does NOT claim track-ID-consistent left/right
topological classes, formal safety guarantees, or a navigation RL result.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F

REACTIVE, LEFT, RIGHT, YIELD = range(4)
MODE_NAMES = ("reactive", "left", "right", "yield")


@dataclass
class JointValues:
    q: Tensor                # [batch, mode, primitive_action]
    collision_logit: Tensor  # [batch, mode, primitive_action]


class IntentConditionedJointCritic(nn.Module):
    """Learn Q(h_t, z_{t-1}, z_t, a_t), instead of post-hoc action bias.

    Features should retain gradients back into a shared temporal encoder.
    Previous_mode is part of the learned policy state, not a switch threshold.
    """

    def __init__(self, feature_dim: int = 256, n_actions: int = 80,
                 n_modes: int = 4, mode_embed_dim: int = 24, hidden: int = 128):
        super().__init__()
        if n_modes < 2 or n_actions < 1:
            raise ValueError("Need at least two modes and one primitive action")
        self.feature_dim, self.n_actions, self.n_modes = feature_dim, n_actions, n_modes
        self.mode_embed = nn.Embedding(n_modes, mode_embed_dim)
        self.previous_embed = nn.Embedding(n_modes, mode_embed_dim)
        in_dim = feature_dim + 2 * mode_embed_dim
        self.fusion = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(),
                                    nn.Linear(hidden, hidden), nn.SiLU())
        self.q_head = nn.Linear(hidden, n_actions)
        self.collision_head = nn.Linear(hidden, n_actions)

    def forward(self, features: Tensor, previous_mode: Tensor) -> JointValues:
        if features.ndim != 2 or features.size(-1) != self.feature_dim:
            raise ValueError("features must have shape [B, feature_dim]")
        if previous_mode.shape != (features.size(0),):
            raise ValueError("previous_mode must have shape [B]")
        b = features.size(0)
        candidates = torch.arange(self.n_modes, device=features.device)
        intent = self.mode_embed(candidates)[None].expand(b, -1, -1)
        previous = self.previous_embed(previous_mode.long())[:, None].expand(-1, self.n_modes, -1)
        shared = features[:, None].expand(-1, self.n_modes, -1)
        joint = self.fusion(torch.cat((shared, previous, intent), dim=-1))
        return JointValues(q=self.q_head(joint), collision_logit=self.collision_head(joint))


def select_joint_decision(values: JointValues, safe_actions: Tensor,
                          mode_action_mask: Tensor) -> tuple[Tensor, Tensor]:
    """Greedy decision under declared feasibility masks.

    The geometric safety mask must use the SAME benchmark candidate contract
    across methods. If no action is feasible, the caller should follow the
    benchmark's documented fallback (this function refuses to silently guess).
    Collision logits are predictions, NOT a physical safety guarantee.
    """
    q = values.q
    b, modes, actions = q.shape
    if safe_actions.shape != (b, actions):
        raise ValueError("safe_actions must have shape [B, A]")
    if mode_action_mask.shape != (b, modes, actions):
        raise ValueError("mode_action_mask must have shape [B, M, A]")
    valid = safe_actions.bool()[:, None, :] & mode_action_mask.bool()
    if not bool(valid.flatten(1).any(dim=1).all()):
        raise ValueError("No feasible joint decision; use controlled fallback")
    masked = q.masked_fill(~valid, -torch.inf)
    flat_index = masked.flatten(1).argmax(dim=1)
    return torch.div(flat_index, actions, rounding_mode="floor"), flat_index % actions


def cf_regret_loss(values: JointValues, better_mode: Tensor,
                   worse_mode: Tensor, advantage: Tensor,
                   valid_actions: Tensor, margin: float = 0.05) -> Tensor:
    """Order intent-values using paired SAME-STATE counterfactual returns.

    Advantage is an observed G(better)-G(worse), NOT an oracle expectation.
    Only positive pairs enter; swap mode labels for negative return differences.
    Both modes share the SAME previous-mode and state representation.
    """
    if torch.any(advantage <= 0):
        raise ValueError("advantage must be positive; swap mode labels otherwise")
    if valid_actions.shape != values.q.shape:
        raise ValueError("valid_actions must match [B,M,A]")
    if not bool(valid_actions.any(dim=2).all()):
        raise ValueError("Every example/mode needs at least one feasible action")
    mode_q = values.q.masked_fill(~valid_actions.bool(), -torch.inf).max(dim=-1).values
    top = mode_q.gather(1, better_mode.long()[:, None]).squeeze(1)
    bottom = mode_q.gather(1, worse_mode.long()[:, None]).squeeze(1)
    return (F.softplus(margin - (top - bottom)) * advantage.detach().clamp(max=2.0)).mean()


def behavior_clone_loss(values: JointValues, teacher_action: Tensor) -> Tensor:
    """IL marginalizes unobserved intents; do not invent expert intent labels."""
    action_logits = torch.logsumexp(values.q, dim=1)
    return F.cross_entropy(action_logits, teacher_action.long())


def td_joint_loss(values: JointValues, mode: Tensor, action: Tensor,
                  reward: Tensor, next_values: JointValues,
                  terminal: Tensor, gamma: float = 0.99) -> Tensor:
    """One-step joint Bellman update for the selected mode and action.

    next_values must be evaluated at h_(t+1) with previous_mode=z_t.
    Fresh policy rollouts are required for OOD mitigation after intervention.
    """
    b = values.q.size(0)
    chosen = values.q[torch.arange(b, device=values.q.device), mode.long(), action.long()]
    with torch.no_grad():
        bootstrap = next_values.q.flatten(1).max(dim=1).values
        target = reward + gamma * (1 - terminal.float()) * bootstrap
    return F.smooth_l1_loss(chosen, target)


def collision_supervision_loss(values: JointValues, mode: Tensor,
                               action: Tensor, collision: Tensor) -> Tensor:
    """Calibrate risk only for the observed state/mode/action combination."""
    b = values.q.size(0)
    pred = values.collision_logit[torch.arange(b, device=values.q.device),
                                  mode.long(), action.long()]
    return F.binary_cross_entropy_with_logits(pred, collision.float())
