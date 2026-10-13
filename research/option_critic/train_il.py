"""Option-Critic Imitation Learning & Option Specialization Pre-Training

Pre-trains OptionCriticNetwork using expert demonstrations:
1. Rolls out expert trajectories from B0 baseline across separate training cases.
2. Derives macro-intent options (Bypass_Left, Direct, Bypass_Right, Yield) for each step.
3. Warm-starts spatial/temporal encoders from verified B0 weights.
4. Performs multi-option behavioral cloning & marginal action probability maximization.
5. Saves pre-trained checkpoint to checkpoints/oc_il_pretrained.pth.
"""

import sys
import os
import math
import time
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
KDARL_CLEAN = REPO_ROOT / 'kdarl_clean'
sys.path = [str(KDARL_CLEAN), str(REPO_ROOT)] + [p for p in sys.path if p not in (str(REPO_ROOT), str(KDARL_CLEAN), '')]

from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from crowd_sim.envs.utils.action import ActionXY
from crowd_nav.policy.mamba_rl import MambaRLPolicy
from run_selection_study_50cases import install_selection
from run_b0_counterfactual_study import (
    get_base_config, create_environment,
    INTENT_BYPASS_LEFT, INTENT_DIRECT, INTENT_BYPASS_RIGHT, INTENT_YIELD,
    derive_macro_intent_from_action
)
from research.option_critic.networks import OptionCriticNetwork


class ExpertDemonstrationDataset(Dataset):
    """Stores expert (obs_tokens, option_label, action_index) tuples."""

    def __init__(self, obs_tokens: np.ndarray, option_labels: np.ndarray, action_indices: np.ndarray):
        self.obs = torch.from_numpy(obs_tokens).float()
        self.options = torch.from_numpy(option_labels).long()
        self.actions = torch.from_numpy(action_indices).long()

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return self.obs[idx], self.options[idx], self.actions[idx]


def collect_expert_demonstrations(
    num_episodes: int = 80,
    start_case: int = 80000,
    device: str = 'cpu'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect expert trajectories using B0 policy on isolated training cases."""
    from test_p0_regression_and_parity import load_b0_policy, create_environment
    print(f"\n[IL-COLLECT] Collecting {num_episodes} expert episodes (Cases {start_case} - {start_case + num_episodes - 1})...")
    
    env, env_config = create_environment()
    robot = Robot(env_config, 'robot')
    env.set_robot(robot)

    # Load B0 expert policy
    b0_policy, _ = load_b0_policy(device=device)
    robot.set_policy(b0_policy)

    all_tokens = []
    all_options = []
    all_actions = []

    success_count = 0
    start_time = time.time()

    for ep_idx in range(num_episodes):
        case_id = start_case + ep_idx
        obs, _ = env.reset(options={'test_case': case_id})
        b0_policy.reset_episode_stats()

        done = False
        step = 0

        while not done and step < 200:
            rob = env.robot
            state = JointState(rob.get_full_state(), [h.get_observable_state() for h in env.humans])
            action, action_idx = b0_policy.predict(state, deterministic=True, return_idx=True)

            # Extract history tensor from policy
            hist_list = list(b0_policy._history)
            if len(hist_list) < 24:
                pad_frame = hist_list[0]
                pad_needed = 24 - len(hist_list)
                hist_list = [pad_frame.copy() for _ in range(pad_needed)] + hist_list
            token_seq = np.array(hist_list[-24:], dtype=np.float32)

            # Derive option label
            opt_label = derive_macro_intent_from_action(rob, action)

            all_tokens.append(token_seq)
            all_options.append(opt_label)
            all_actions.append(action_idx)

            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc
            step += 1

        last_event = info.get('event', 'nothing') if info else 'nothing'
        if last_event in ('reach_goal', 'reachgoal') or env.robot.reached_destination():
            success_count += 1

        if (ep_idx + 1) % 20 == 0:
            print(f"  Collected {ep_idx + 1}/{num_episodes} episodes ({len(all_tokens)} steps, SR: {success_count / (ep_idx + 1):.1%})")

    elapsed = time.time() - start_time
    print(f"[IL-COLLECT] Finished in {elapsed:.1f}s. Total transitions: {len(all_tokens)}. Expert SR: {success_count / num_episodes:.1%}\n")

    return np.array(all_tokens, dtype=np.float32), np.array(all_options, dtype=np.int64), np.array(all_actions, dtype=np.int64)


def train_il(
    num_episodes: int = 80,
    start_case: int = 80000,
    epochs: int = 12,
    batch_size: int = 64,
    lr: float = 1e-4,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    save_path: str = 'research/option_critic/checkpoints/oc_il_pretrained.pth'
):
    """Run end-to-end Imitation Learning pre-training for Option-Critic."""
    out_dir = Path(save_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Collect demonstrations
    tokens, options, actions = collect_expert_demonstrations(
        num_episodes=num_episodes, start_case=start_case, device='cpu'
    )

    dataset = ExpertDemonstrationDataset(tokens, options, actions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 2. Build OptionCriticNetwork and warm-start backbone
    print(f"[IL-TRAIN] Initializing OptionCriticNetwork on {device}...")
    net = OptionCriticNetwork(
        num_options=4,
        num_actions=80,
        d_model=256,
        n_layers=4,
        temporal_backbone='gru',
        is_cf_oc=True
    ).to(device)

    load_rep = net.load_b0_backbone(REPO_ROOT / 'baseline_weights/gru-t24-seed443.pth', device=device)
    print(f"[IL-TRAIN] Backbone warm-start status: {load_rep}")

    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 3. Training Loop
    print(f"[IL-TRAIN] Starting multi-option behavioral cloning for {epochs} epochs...")
    net.train()

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_opt_loss = 0.0
        total_act_loss = 0.0
        total_correct = 0
        total_samples = 0

        for b_obs, b_opt, b_act in dataloader:
            b_obs = b_obs.to(device)
            b_opt = b_opt.to(device)
            b_act = b_act.to(device)
            B = b_obs.shape[0]

            optimizer.zero_grad()
            out = net(b_obs)

            # Option classification loss (teaching Q_Omega / option selection to predict matching option)
            opt_logits = out['q_omega']  # [B, 4]
            loss_opt = F.cross_entropy(opt_logits, b_opt)

            # Intra-option action loss for the target option: [B, 4, 80]
            policy_logits = out['policy_logits']  # [B, 4, 80]
            batch_idx = torch.arange(B, device=device)
            target_policy_logits = policy_logits[batch_idx, b_opt]  # [B, 80]
            loss_act = F.cross_entropy(target_policy_logits, b_act)

            # Marginal action likelihood regularization: sum_o pi(o) * pi_o(a)
            pi_omega = F.softmax(opt_logits, dim=-1).unsqueeze(-1)  # [B, 4, 1]
            pi_intra = F.softmax(policy_logits, dim=-1)             # [B, 4, 80]
            marginal_probs = (pi_omega * pi_intra).sum(dim=1)       # [B, 80]
            loss_marginal = -torch.log(marginal_probs[batch_idx, b_act] + 1e-8).mean()

            loss = loss_act + 0.5 * loss_opt + 0.5 * loss_marginal
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * B
            total_opt_loss += loss_opt.item() * B
            total_act_loss += loss_act.item() * B
            preds = target_policy_logits.argmax(dim=-1)
            total_correct += (preds == b_act).sum().item()
            total_samples += B

        scheduler.step()

        avg_loss = total_loss / total_samples
        avg_act_loss = total_act_loss / total_samples
        avg_opt_loss = total_opt_loss / total_samples
        acc = total_correct / total_samples
        print(f"  Epoch {epoch:02d}/{epochs:02d} | Loss: {avg_loss:.4f} (Act: {avg_act_loss:.4f}, Opt: {avg_opt_loss:.4f}) | Action Acc: {acc:.1%}")

    # 4. Save checkpoint
    torch.save({
        'epoch': epochs,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'config': {
            'num_options': 4,
            'num_actions': 80,
            'd_model': 256,
            'n_layers': 4,
            'temporal_backbone': 'gru',
            'is_cf_oc': True
        }
    }, save_path)
    print(f"\n[IL-TRAIN] Pre-trained Option-Critic checkpoint saved to: {save_path}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=80)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    train_il(num_episodes=args.episodes, epochs=args.epochs, device=args.device)
