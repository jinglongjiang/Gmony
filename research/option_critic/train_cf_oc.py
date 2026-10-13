"""CF-OC: Counterfactual Switching-Advantage Option-Critic Training

Implements the original algorithmic contribution:
1. Online Call-and-Return hierarchical RL in CrowdNav.
2. State-level Counterfactual Branching at contested navigation states.
3. Supervision of Counterfactual Advantage Delta_pred(h, o) ~ (G_reselect - G_continue).
4. Self-supervised Short-Horizon Collision Risk Head p_collision(h, o) (16 steps = 4s).
5. Joint learning of CF-Conditioned Option Termination beta_cf(h, o).
6. Checkpoint persistence to checkpoints/cf_oc_trained.pth.
"""

import sys
import os
import math
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
KDARL_CLEAN = REPO_ROOT / 'kdarl_clean'
sys.path = [str(KDARL_CLEAN), str(REPO_ROOT)] + [p for p in sys.path if p not in (str(REPO_ROOT), str(KDARL_CLEAN), '')]

from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY
from test_p0_regression_and_parity import create_environment, get_base_config
from research.option_critic.networks import OptionCriticNetwork
from research.option_critic.policy import OptionCriticPolicy
from research.option_critic.option_critic_losses import OptionCriticLosses
from research.option_critic.replay_buffer import OptionReplayBuffer
from research.option_critic.counterfactual_sampler import CounterfactualSampler


def train_cf_oc(
    num_episodes: int = 250,
    start_case: int = 1000,
    deliberation_cost: float = 0.05,
    gamma: float = 0.99,
    lr: float = 1e-4,
    cf_sample_prob: float = 0.20,
    batch_size: int = 64,
    buffer_capacity: int = 40000,
    update_freq: int = 2,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    pretrained_path: Optional[str] = 'research/option_critic/checkpoints/oc_il_pretrained.pth',
    save_path: str = 'research/option_critic/checkpoints/cf_oc_trained.pth'
):
    """Execute end-to-end CF-OC training with counterfactual supervision."""
    print(f"\n{'='*80}")
    print(f"[CF-OC TRAIN] Starting Counterfactual Switching-Advantage Option-Critic on {device}")
    print(f"[CF-OC TRAIN] Episodes: {num_episodes} | Deliberation Cost: {deliberation_cost} | CF Sampling Prob: {cf_sample_prob}")
    print(f"{'='*80}\n")

    out_dir = Path(save_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Environment & Policy Setup
    env, env_config = create_environment()
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

    network = OptionCriticNetwork(
        num_options=4,
        num_actions=80,
        d_model=256,
        n_layers=4,
        temporal_backbone='gru',
        is_cf_oc=True
    ).to(device)

    # Warm-start from IL checkpoint or B0 baseline
    if pretrained_path and Path(pretrained_path).exists():
        print(f"[CF-OC TRAIN] Loading pre-trained weights from: {pretrained_path}")
        ckpt = torch.load(pretrained_path, map_location=device)
        net_sd = ckpt.get('model_state_dict', ckpt.get('policy_state', ckpt))
        clean_sd = {k.replace('module.', '').replace('_orig_mod.', ''): v for k, v in net_sd.items()}
        incomp = network.load_state_dict(clean_sd, strict=False)
        print(f"[CF-OC TRAIN] Warm-start loaded (Missing: {len(incomp.missing_keys)}, Unexpected: {len(incomp.unexpected_keys)})")
    else:
        print("[CF-OC TRAIN] Initializing spatial/temporal backbone from baseline B0 weights...")
        network.load_b0_backbone(REPO_ROOT / 'baseline_weights/gru-t24-seed443.pth', device=device)

    policy = OptionCriticPolicy(
        network=network,
        device=device,
        num_options=4,
        num_actions=80,
        seq_len=24,
        is_cf_oc=True,
        deliberation_cost=deliberation_cost,
        use_sampling_termination=True
    )
    robot.set_policy(policy)

    # 2. Optimization, Buffer and Sampler Setup
    losses = OptionCriticLosses(
        gamma=gamma,
        deliberation_cost=deliberation_cost,
        entropy_coef=0.01,
        termination_reg_coef=0.01,
        cf_loss_coef=0.5,
        risk_loss_coef=0.5
    )

    optimizer = torch.optim.Adam(network.parameters(), lr=lr, weight_decay=1e-5)
    replay_buffer = OptionReplayBuffer(
        capacity=buffer_capacity,
        seq_len=24,
        num_actions=80,
        collision_horizon=16,
        device=device
    )
    cf_sampler = CounterfactualSampler(k_steps=16, gamma=gamma)

    # Metrics history
    ep_rewards = []
    ep_outcomes = []
    ep_switches = []
    ep_durations = []
    cf_sample_count = 0
    option_usage = np.zeros(4, dtype=np.int64)

    total_steps = 0
    start_time = time.time()

    # 3. Main CF-OC Training Loop
    for ep in range(1, num_episodes + 1):
        case_id = start_case + (ep % 8000)
        obs, _ = env.reset(options={'train_case': case_id, 'phase': 'train'})
        policy.reset_episode_stats()
        policy.epsilon = max(0.05, 0.40 * (1.0 - ep / (0.7 * num_episodes)))

        done = False
        step = 0
        ep_reward = 0.0

        # Rollout episode
        while not done and step < 200:
            rob = env.robot
            state = JointState(rob.get_full_state(), [h.get_observable_state() for h in env.humans])
            
            # Physical clearance check
            if len(env.humans):
                min_clearance = min(
                    math.hypot(rob.px - h.px, rob.py - h.py) - rob.radius - h.radius
                    for h in env.humans
                )
            else:
                min_clearance = 999.0

            # Decide whether to execute counterfactual branching at this step
            cf_adv_label = None
            should_cf_sample = (
                (min_clearance < 0.80 or np.random.random() < cf_sample_prob)
                and step > 5
                and policy.current_option is not None
            )

            if should_cf_sample:
                cf_res = cf_sampler.evaluate_branches(env, policy, policy.current_option)
                cf_adv_label = cf_res['delta_cf']
                cf_sample_count += 1

            # Step policy
            action_xy, action_idx, option_idx = policy.act(state, deterministic=False)
            option_usage[option_idx] += 1

            # Extract history tokens before step
            hist_before = policy.get_current_history_array()

            # Step env
            obs, reward, term, trunc, info = env.step(action_xy)
            done = term or trunc
            ep_reward += reward
            total_steps += 1
            step += 1

            # Extract next state history tokens
            next_state = JointState(rob.get_full_state(), [h.get_observable_state() for h in env.humans])
            next_token = policy._state_to_tokens(next_state)
            hist_after = np.concatenate([hist_before[1:], next_token.reshape(1, 8, 13)], axis=0)

            # Store in replay buffer
            replay_buffer.add(
                obs=hist_before,
                option=option_idx,
                action=action_idx,
                reward=reward,
                next_obs=hist_after,
                done=done,
                cf_adv=cf_adv_label
            )

            # Gradient update
            if len(replay_buffer) >= batch_size and total_steps % update_freq == 0:
                batch = replay_buffer.sample(batch_size)
                network.train()
                optimizer.zero_grad()

                out_curr = network(batch['obs'], action_mask=batch['action_mask'])
                out_next = network(batch['next_obs'])

                # Critic Q target
                next_beta = out_next['beta_cf']
                target_q = losses.compute_q_target(
                    batch['reward'], batch['done'], out_next['q_omega'], next_beta, batch['option']
                )

                loss_q = losses.compute_q_loss(out_curr['q_omega'], batch['option'], target_q)
                loss_qu = losses.compute_q_u_loss(out_curr['q_u'], batch['option'], batch['action'], target_q)

                # True intra-option action advantage: A(h, o, a) = Q_U(h, o, a) - Q_Omega(h, o)
                B = batch_size
                b_idx = torch.arange(B, device=device)
                q_u_val = out_curr['q_u'][b_idx, batch['option'], batch['action']]
                q_omega_val = out_curr['q_omega'][b_idx, batch['option']]
                advantage = (q_u_val - q_omega_val).detach()
                adv_std = advantage.std()
                if adv_std > 1e-4:
                    advantage = (advantage - advantage.mean()) / (adv_std + 1e-8)
                else:
                    advantage = advantage - advantage.mean()

                loss_pi, entropy = losses.compute_policy_loss(
                    out_curr['policy_probs'], batch['option'], batch['action'], advantage
                )

                # CF-calibrated termination loss
                loss_beta, cont_adv = losses.compute_termination_loss(
                    next_beta, out_next['q_omega'], batch['option']
                )

                # Base loss
                total_loss = loss_q + 0.5 * loss_qu + loss_pi + loss_beta

                # CF-OC supervisory losses (Advantage Huber + Collision Risk BCE)
                cf_losses = losses.compute_cf_oc_losses(
                    out_curr['cf_advantage'], batch['cf_adv'],
                    out_curr['collision_risk'], batch['coll_risk'],
                    batch['option']
                )
                
                # Weight CF loss only on samples with valid counterfactual supervision
                has_cf_mask = batch['has_cf']
                if has_cf_mask.sum() > 0:
                    cf_adv_loss = (cf_losses['loss_cf'] * has_cf_mask).sum() / has_cf_mask.sum()
                else:
                    cf_adv_loss = torch.tensor(0.0, device=device)

                total_loss = total_loss + cf_adv_loss + cf_losses['loss_risk']

                total_loss.backward()
                nn.utils.clip_grad_norm_(network.parameters(), max_norm=1.0)
                optimizer.step()

        # Finalize episode outcome
        last_event = info.get('event', 'nothing') if info else 'nothing'
        if last_event == 'collision':
            outcome = 'COLLISION'
        elif last_event in ('reach_goal', 'reachgoal') or env.robot.reached_destination():
            outcome = 'SUCCESS'
        else:
            outcome = 'TIMEOUT'

        replay_buffer.finish_episode(last_event)
        ep_rewards.append(ep_reward)
        ep_outcomes.append(outcome)
        ep_switches.append(policy.total_switches)
        if len(policy.telemetry['durations']) > 0:
            ep_durations.append(float(np.mean(policy.telemetry['durations'])))

        # Periodic logging every 25 episodes
        if ep % 25 == 0 or ep == num_episodes:
            recent_outcomes = ep_outcomes[-25:]
            sr = recent_outcomes.count('SUCCESS') / len(recent_outcomes)
            cr = recent_outcomes.count('COLLISION') / len(recent_outcomes)
            tr = recent_outcomes.count('TIMEOUT') / len(recent_outcomes)
            avg_rew = float(np.mean(ep_rewards[-25:]))
            avg_sw = float(np.mean(ep_switches[-25:]))
            avg_dur = float(np.mean(ep_durations[-25:])) if ep_durations else 0.0
            
            usage_sum = max(1, option_usage.sum())
            usage_pct = [f"O{i}: {100.0 * option_usage[i] / usage_sum:.1f}%" for i in range(4)]
            
            print(f"[CF-OC EP {ep:03d}/{num_episodes}] SR: {sr:.1%} | CR: {cr:.1%} | TO: {tr:.1%} | "
                  f"Rew: {avg_rew:.2f} | Switches: {avg_sw:.1f} | Dwell: {avg_dur:.1f} | CF Samples: {cf_sample_count} | "
                  f"[{', '.join(usage_pct)}]")

    elapsed = time.time() - start_time
    print(f"\n[CF-OC TRAIN] Training completed in {elapsed / 60.0:.1f} minutes. Total CF samples: {cf_sample_count}")

    # 4. Save Final Trained CF-OC Policy
    torch.save({
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'episodes': num_episodes,
        'deliberation_cost': deliberation_cost,
        'is_cf_oc': True,
        'cf_sample_count': cf_sample_count,
        'final_stats': {
            'success_rate': float(ep_outcomes[-50:].count('SUCCESS') / 50.0),
            'collision_rate': float(ep_outcomes[-50:].count('COLLISION') / 50.0),
            'mean_switches': float(np.mean(ep_switches[-50:])),
            'option_usage': option_usage.tolist()
        }
    }, save_path)
    print(f"[CF-OC TRAIN] Checkpoint saved to: {save_path}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=200)
    parser.add_argument('--delib_cost', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--cf_prob', type=float, default=0.20)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--pretrained', type=str, default='research/option_critic/checkpoints/oc_il_pretrained.pth')
    parser.add_argument('--save_path', type=str, default='research/option_critic/checkpoints/cf_oc_trained.pth')
    args = parser.parse_args()

    train_cf_oc(
        num_episodes=args.episodes,
        deliberation_cost=args.delib_cost,
        lr=args.lr,
        cf_sample_prob=args.cf_prob,
        device=args.device,
        pretrained_path=args.pretrained,
        save_path=args.save_path
    )
