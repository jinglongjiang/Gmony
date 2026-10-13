"""Option-Critic and CF-OC Objective Functions and Losses

Implements:
1. Intra-Option Policy Gradient Loss (with entropy regularization).
2. Option Termination Loss (Bacon et al. 2017) with Deliberation Cost (Harb et al. 2018).
3. Option-Critic Bellman TD Error (Huber loss on expected return targets).
4. CF-OC Counterfactual Calibration Loss (Huber regression on Delta_CF = G_reselect - G_continue).
5. CF-OC Collision Risk Loss (Binary Cross-Entropy on 4s collision occurrence).
6. Regularizers: Option switching penalty, entropy bonus, termination bounding.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class OptionCriticLosses(nn.Module):
    """Computes all optimization objectives for Option-Critic and CF-OC."""

    def __init__(
        self,
        gamma: float = 0.99,
        deliberation_cost: float = 0.05,
        entropy_coef: float = 0.01,
        termination_reg_coef: float = 0.01,
        cf_loss_coef: float = 0.5,
        risk_loss_coef: float = 0.5,
        huber_delta: float = 1.0
    ):
        super().__init__()
        self.gamma = gamma
        self.deliberation_cost = deliberation_cost
        self.entropy_coef = entropy_coef
        self.termination_reg_coef = termination_reg_coef
        self.cf_loss_coef = cf_loss_coef
        self.risk_loss_coef = risk_loss_coef
        self.huber_delta = huber_delta

    def compute_q_target(
        self,
        reward: torch.Tensor,
        done: torch.Tensor,
        next_q_omega: torch.Tensor,
        next_beta: torch.Tensor,
        curr_option: torch.Tensor
    ) -> torch.Tensor:
        """Compute expected option Bellman target:
        y_t = r_t + gamma * (1 - d_t) * [(1 - beta_{o_t}(h_{t+1})) * Q_Omega(h_{t+1}, o_t) + beta_{o_t}(h_{t+1}) * V_Omega(h_{t+1})]
        
        Args:
            reward: [B]
            done: [B] (1.0 if episode ended, else 0.0)
            next_q_omega: [B, num_options]
            next_beta: [B, num_options]
            curr_option: [B] LongTensor of option indices
        Returns:
            target: [B]
        """
        B = reward.shape[0]
        batch_idx = torch.arange(B, device=reward.device)

        # Expected value over options at next state V_Omega(h')
        v_next = torch.max(next_q_omega, dim=-1)[0]  # [B]

        # Q_Omega(h', o_t) for current option
        q_next_curr = next_q_omega[batch_idx, curr_option]  # [B]

        # beta_{o_t}(h') for current option
        beta_next_curr = next_beta[batch_idx, curr_option]  # [B]

        # Expected next value under call-and-return termination
        next_value = (1.0 - beta_next_curr) * q_next_curr + beta_next_curr * v_next
        target = reward + self.gamma * (1.0 - done.float()) * next_value
        return target.detach()

    def compute_q_loss(
        self,
        q_omega: torch.Tensor,
        curr_option: torch.Tensor,
        target_q: torch.Tensor
    ) -> torch.Tensor:
        """Huber loss between predicted Q_Omega(h, o_t) and Bellman target."""
        B = q_omega.shape[0]
        batch_idx = torch.arange(B, device=q_omega.device)
        pred_q = q_omega[batch_idx, curr_option]
        return F.huber_loss(pred_q, target_q, delta=self.huber_delta)

    def compute_q_u_loss(
        self,
        q_u: torch.Tensor,
        curr_option: torch.Tensor,
        action: torch.Tensor,
        target_q: torch.Tensor
    ) -> torch.Tensor:
        """Huber loss between predicted Q_U(h, o_t, a_t) and Bellman target."""
        B = q_u.shape[0]
        batch_idx = torch.arange(B, device=q_u.device)
        pred_qu = q_u[batch_idx, curr_option, action]
        return F.huber_loss(pred_qu, target_q, delta=self.huber_delta)

    def compute_policy_loss(
        self,
        policy_probs: torch.Tensor,
        curr_option: torch.Tensor,
        action: torch.Tensor,
        advantage: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Intra-option policy gradient loss with entropy bonus:
        L_pi = - E [ log pi_o(a | h) * A ] - entropy_coef * H(pi_o)
        
        Args:
            policy_probs: [B, num_options, num_actions]
            curr_option: [B] LongTensor
            action: [B] LongTensor
            advantage: [B]
        Returns:
            policy_loss: scalar tensor
            mean_entropy: scalar tensor
        """
        B = policy_probs.shape[0]
        batch_idx = torch.arange(B, device=policy_probs.device)

        # Extract action distribution for selected option: [B, num_actions]
        curr_probs = policy_probs[batch_idx, curr_option]
        log_probs = torch.log(curr_probs + 1e-8)

        # Selected action log probability: [B]
        selected_log_prob = log_probs[batch_idx, action]

        # Policy gradient loss
        pg_loss = -(selected_log_prob * advantage.detach()).mean()

        # Entropy of current option policy: [B]
        entropy = -(curr_probs * log_probs).sum(dim=-1).mean()

        total_policy_loss = pg_loss - self.entropy_coef * entropy
        return total_policy_loss, entropy

    def compute_termination_loss(
        self,
        beta_next: torch.Tensor,
        next_q_omega: torch.Tensor,
        curr_option: torch.Tensor,
        deliberation_cost: Optional[float] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Option-Critic termination gradient loss (Harb et al. 2018):
        L_beta = E [ beta_{o_t}(h_{t+1}) * (Q_Omega(h_{t+1}, o_t) - V_Omega(h_{t+1}) + eta) ]
        
        Args:
            beta_next: [B, num_options]
            next_q_omega: [B, num_options]
            curr_option: [B]
            deliberation_cost: Optional override for eta
        Returns:
            term_loss: scalar tensor
            continuation_advantage: scalar tensor (mean of Q - V)
        """
        if deliberation_cost is None:
            deliberation_cost = self.deliberation_cost

        B = beta_next.shape[0]
        batch_idx = torch.arange(B, device=beta_next.device)

        # Detach next Q-values so termination gradient does not alter critic weights
        q_next_detached = next_q_omega.detach()
        v_next_detached = torch.max(q_next_detached, dim=-1)[0]
        q_curr_detached = q_next_detached[batch_idx, curr_option]

        # Continuation advantage: positive when continuing is better than switching
        cont_adv = q_curr_detached - v_next_detached  # [B]

        # Termination objective
        beta_curr = beta_next[batch_idx, curr_option]  # [B]
        term_loss = (beta_curr * (cont_adv + deliberation_cost)).mean()

        # Regularization to prevent beta from collapsing to 0 or 1
        reg_loss = self.termination_reg_coef * (beta_curr * (1.0 - beta_curr)).mean()
        total_term_loss = term_loss - reg_loss

        return total_term_loss, cont_adv.mean()

    def compute_cf_oc_losses(
        self,
        pred_cf_adv: torch.Tensor,
        target_cf_adv: torch.Tensor,
        pred_collision_risk: torch.Tensor,
        target_collision: torch.Tensor,
        curr_option: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute CF-OC specific supervisory losses.
        
        Args:
            pred_cf_adv: [B, num_options]
            target_cf_adv: [B] (from counterfactual snapshots: G_reselect - G_continue)
            pred_collision_risk: [B, num_options]
            target_collision: [B] (1.0 if collision occurred within 4s, else 0.0)
            curr_option: [B] LongTensor
        """
        B = pred_cf_adv.shape[0]
        batch_idx = torch.arange(B, device=pred_cf_adv.device)

        # Predicted counterfactual advantage for current option
        pred_adv_curr = pred_cf_adv[batch_idx, curr_option]
        cf_loss = F.huber_loss(pred_adv_curr, target_cf_adv, delta=self.huber_delta)

        # Predicted short-horizon collision risk for current option
        pred_risk_curr = pred_collision_risk[batch_idx, curr_option]
        risk_loss = F.binary_cross_entropy(pred_risk_curr, target_collision.float())

        return {
            'loss_cf': self.cf_loss_coef * cf_loss,
            'loss_risk': self.risk_loss_coef * risk_loss,
            'raw_cf_error': (pred_adv_curr - target_cf_adv).abs().mean(),
            'raw_risk_error': (pred_risk_curr - target_collision.float()).abs().mean()
        }
