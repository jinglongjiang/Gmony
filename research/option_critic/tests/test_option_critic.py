"""Unit Tests for Option-Critic & CF-OC Implementations

Tests:
1. Forward pass shapes and head integrity.
2. Gradient flow across intra-option policy, critic, termination, and CF heads.
3. Call-and-Return execution cycle and deterministic option switching.
4. Clone and restore parity for counterfactual branching.
5. Replay buffer storage, short-horizon collision tagging, and batch sampling.
"""

import sys
import unittest
from pathlib import Path
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
KDARL_CLEAN = REPO_ROOT / 'kdarl_clean'
if str(KDARL_CLEAN) not in sys.path:
    sys.path.insert(0, str(KDARL_CLEAN))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.option_critic.networks import OptionCriticNetwork
from research.option_critic.policy import OptionCriticPolicy
from research.option_critic.option_critic_losses import OptionCriticLosses
from research.option_critic.replay_buffer import OptionReplayBuffer


class TestOptionCriticSuite(unittest.TestCase):

    def setUp(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.num_options = 4
        self.num_actions = 80
        self.seq_len = 24
        self.d_model = 256

        self.net = OptionCriticNetwork(
            num_options=self.num_options,
            num_actions=self.num_actions,
            d_model=self.d_model,
            n_layers=4,
            temporal_backbone='gru',
            is_cf_oc=True
        ).to(self.device)

        self.losses = OptionCriticLosses(
            gamma=0.99,
            deliberation_cost=0.05,
            entropy_coef=0.01,
            termination_reg_coef=0.01,
            cf_loss_coef=0.5,
            risk_loss_coef=0.5
        )

    def test_forward_shapes(self):
        """Verify network output tensor dimensions."""
        B = 4
        dummy_tokens = torch.randn(B, self.seq_len, 8, 13, device=self.device)
        dummy_mask = torch.ones(B, self.num_actions, device=self.device)

        out = self.net(dummy_tokens, action_mask=dummy_mask)
        self.assertEqual(out['h'].shape, (B, self.d_model))
        self.assertEqual(out['q_omega'].shape, (B, self.num_options))
        self.assertEqual(out['q_u'].shape, (B, self.num_options, self.num_actions))
        self.assertEqual(out['policy_probs'].shape, (B, self.num_options, self.num_actions))
        self.assertEqual(out['beta'].shape, (B, self.num_options))
        self.assertEqual(out['cf_advantage'].shape, (B, self.num_options))
        self.assertEqual(out['collision_risk'].shape, (B, self.num_options))
        self.assertEqual(out['beta_cf'].shape, (B, self.num_options))

    def test_gradient_flow(self):
        """Verify that gradients propagate to all network heads without NaN."""
        B = 4
        obs = torch.randn(B, self.seq_len, 8, 13, device=self.device)
        next_obs = torch.randn(B, self.seq_len, 8, 13, device=self.device)
        rewards = torch.tensor([1.0, 0.0, -10.0, 0.5], device=self.device)
        dones = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device)
        options = torch.tensor([0, 1, 2, 3], device=self.device)
        actions = torch.tensor([10, 20, 30, 40], device=self.device)
        advantages = torch.tensor([0.5, -0.2, 1.2, -0.8], device=self.device)
        cf_targets = torch.tensor([0.4, 0.0, 1.1, -0.1], device=self.device)
        risk_targets = torch.tensor([0.0, 0.0, 1.0, 0.0], device=self.device)

        out_curr = self.net(obs)
        out_next = self.net(next_obs)

        # 1. Q loss
        target_q = self.losses.compute_q_target(
            rewards, dones, out_next['q_omega'], out_next['beta_cf'], options
        )
        loss_q = self.losses.compute_q_loss(out_curr['q_omega'], options, target_q)

        # 2. Q_U loss
        loss_qu = self.losses.compute_q_u_loss(out_curr['q_u'], options, actions, target_q)

        # 3. Policy loss
        loss_pi, entropy = self.losses.compute_policy_loss(
            out_curr['policy_probs'], options, actions, advantages
        )

        # 4. Termination loss
        loss_beta, cont_adv = self.losses.compute_termination_loss(
            out_next['beta_cf'], out_next['q_omega'], options
        )

        # 5. CF-OC supervisory loss
        cf_losses = self.losses.compute_cf_oc_losses(
            out_curr['cf_advantage'], cf_targets,
            out_curr['collision_risk'], risk_targets,
            options
        )

        total_loss = loss_q + loss_qu + loss_pi + loss_beta + cf_losses['loss_cf'] + cf_losses['loss_risk']
        self.net.zero_grad()
        total_loss.backward()

        # Check gradients exist
        for name, param in self.net.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Missing gradient for {name}")
                self.assertFalse(torch.isnan(param.grad).any(), f"NaN gradient in {name}")

    def test_policy_call_and_return_and_cloning(self):
        """Verify policy call-and-return logic and state cloning."""
        policy = OptionCriticPolicy(
            network=self.net,
            device=self.device,
            num_options=self.num_options,
            num_actions=self.num_actions,
            seq_len=self.seq_len,
            is_cf_oc=True,
            termination_threshold=0.5
        )

        dummy_state = np.random.randn(34)
        policy.reset_episode_stats()
        self.assertIsNone(policy.current_option)

        # Step 0
        act_xy, act_idx, opt_0 = policy.act(dummy_state, deterministic=True)
        self.assertIsNotNone(policy.current_option)
        self.assertEqual(policy.current_option, opt_0)
        self.assertEqual(policy.option_duration, 0)

        # Clone state
        snapshot = policy.clone_state()

        # Step 1
        act_xy1, act_idx1, opt_1 = policy.act(dummy_state, deterministic=True)
        
        # Restore state
        policy.restore_state(snapshot)
        self.assertEqual(policy.current_option, snapshot['current_option'])
        self.assertEqual(policy.option_duration, snapshot['option_duration'])

    def test_replay_buffer(self):
        """Verify buffer transitions, collision tagging, and sampling."""
        buffer = OptionReplayBuffer(
            capacity=100,
            seq_len=self.seq_len,
            num_actions=self.num_actions,
            collision_horizon=3,
            device=self.device
        )

        dummy_obs = np.random.randn(self.seq_len, 8, 13).astype(np.float32)
        dummy_next = np.random.randn(self.seq_len, 8, 13).astype(np.float32)

        # Add 5 steps
        for i in range(5):
            buffer.add(
                obs=dummy_obs,
                option=i % 4,
                action=i * 10,
                reward=-0.1,
                next_obs=dummy_next,
                done=(i == 4),
                cf_adv=0.5 if i == 2 else None
            )

        # Finish episode with collision
        buffer.finish_episode('collision')

        # Last 3 steps should be marked as collision risk = 1.0
        self.assertEqual(buffer.coll_risk_buf[4], 1.0)
        self.assertEqual(buffer.coll_risk_buf[3], 1.0)
        self.assertEqual(buffer.coll_risk_buf[2], 1.0)
        self.assertEqual(buffer.coll_risk_buf[1], 0.0)

        # Sample batch
        batch = buffer.sample(4)
        self.assertEqual(batch['obs'].shape, (4, self.seq_len, 8, 13))
        self.assertEqual(batch['option'].shape, (4,))
        self.assertEqual(batch['has_cf'].shape, (4,))


if __name__ == '__main__':
    unittest.main()
