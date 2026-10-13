#!/usr/bin/env python3
"""Training Pipeline for StratMamba Dual-Stream Goal Stream and LICS Intent Head

Executes:
Stage 1: Collect transition rollouts in CrowdSim (20 humans, square crossing) on training cases.
Stage 2: Train StratMamba Dual-Stream Goal Branch (goal_proj, goal_encoder, W_fuse) via Bellman TD loss.
Stage 3: Train LICS Intent Head (Intent Q-Net Q_phi, Termination Net beta_psi) via Option-Critic loss with Hysteresis Regularization.
Stage 4: Save checkpoints to checkpoints/ directory.
"""

import sys
import os
import time
import math
from pathlib import Path
import configparser
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

WORKSPACE = Path('/home/abc/workspace/autonomous_research_20261012')
KDARL_CLEAN = WORKSPACE / 'kdarl_clean'
sys.path = [str(KDARL_CLEAN), str(WORKSPACE)] + [p for p in sys.path if p not in (str(WORKSPACE), str(KDARL_CLEAN), '')]

from crowd_sim.envs.crowd_sim import CrowdSim
from crowd_sim.envs.utils.robot import Robot
from crowd_sim.envs.utils.state import JointState
from run_selection_study_50cases import install_selection
from strat_mamba_dual_stream import StratMambaDualStreamPolicy
from learnable_intent_mamba import (
    LICSStratMambaPolicy,
    INTENT_BYPASS_LEFT, INTENT_DIRECT, INTENT_BYPASS_RIGHT, INTENT_YIELD
)

CHECKPOINT_DIR = WORKSPACE / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

class TrajectoryDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

def collect_training_rollouts(num_episodes=40, device='cuda'):
    print(f"\n--- COLLECTING {num_episodes} EPISODES OF TRAINING ROLLOUTS ON GPU ---")
    env_config = configparser.RawConfigParser()
    env_config.read(KDARL_CLEAN / 'crowd_nav/configs/env.config')
    policy_config = configparser.RawConfigParser()
    policy_config.read(KDARL_CLEAN / 'crowd_nav/configs/policy.config')
    train_config = configparser.RawConfigParser()
    train_config.read(KDARL_CLEAN / 'crowd_nav/configs/train.config')

    cfg = configparser.RawConfigParser()
    for c in [env_config, policy_config, train_config]:
        for sec in c.sections():
            if not cfg.has_section(sec): cfg.add_section(sec)
            for k, v in c.items(sec): cfg.set(sec, k, v)

    cfg.set('env', 'time_limit', '50')
    cfg.set('mamba', 'temporal_backbone', 'gru')
    cfg.set('buffer', 'seq_len', '24')
    cfg.set('sim', 'human_num', '20')

    weights_path = WORKSPACE / 'baseline_weights/gru-t24-seed443.pth'

    policy = StratMambaDualStreamPolicy(cfg, device='cpu', backbone='gru')
    policy.load_pretrained_weights(weights_path)
    policy.eval()
    policy.build_action_space(1.0)
    policy.use_sarl_predict = True
    policy.set_phase('test')
    policy.test_min_clearance = 0.2
    policy.test_risk_lambda = 0.8
    policy.test_action_smoothing = 0.3
    install_selection(policy, 'ttc5')

    env = CrowdSim()
    env.configure(env_config)
    env.time_limit = 50.0
    env.human_num = 20
    env.square_width = 10.0
    env.circle_radius = 4.0
    env.phase = 'train'
    env.train_sim = 'square_crossing'

    robot = Robot(env_config, 'robot')
    robot.set_policy(policy)
    robot.env = env
    env.set_robot(robot)

    samples = []
    t0 = time.time()

    for ep in range(num_episodes):
        # Use training cases (seeds 1000 - 1999) to strictly isolate from test cases (90000+)
        train_case_id = 1000 + ep
        env.reset(options={'train_case': train_case_id})
        policy.reset_episode_stats()

        done = False
        step = 0
        ep_history = []

        while not done and step < 200:
            step += 1
            state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
            res = policy.score_sarl_candidates(state)
            best_idx = res['scores'].argmax().item()
            action = res['commands'][best_idx]

            policy._history.append(res['current_token'])
            policy._goal_history.append(res['current_goal'])
            if policy.test_action_smoothing > 0.0:
                policy._last_action = action

            prev_pos = (robot.px, robot.py)
            obs, reward, term, trunc, info = env.step(action)
            done = term or trunc

            # Build current observation window strictly matching get_current_hidden_state
            L = policy.seq_len
            base_hist = list(policy._history) + [res['current_token']]
            base_goal_hist = list(policy._goal_history) + [res['current_goal']]
            hist_arr = np.asarray(base_hist)
            goal_arr = np.asarray(base_goal_hist)
            if len(base_hist) >= L:
                win_prox = hist_arr[-L:]
                win_goal = goal_arr[-L:]
            else:
                pads = L - len(base_hist)
                pad_prox = np.repeat(hist_arr[0:1], pads, axis=0)
                pad_goal = np.repeat(goal_arr[0:1], pads, axis=0)
                win_prox = np.concatenate([pad_prox, hist_arr], axis=0)
                win_goal = np.concatenate([pad_goal, goal_arr], axis=0)

            gx, gy = robot.gx, robot.gy
            px, py = robot.px, robot.py
            theta_goal = math.atan2(gy - py, gx - px)
            theta_a = math.atan2(action.vy, action.vx)
            diff = (theta_a - theta_goal + math.pi) % (2 * math.pi) - math.pi
            speed = math.hypot(action.vx, action.vy)

            # Heuristic ground-truth macro-intent label
            if speed < 0.1:
                intent_lbl = INTENT_YIELD
            elif diff > 0.15:
                intent_lbl = INTENT_BYPASS_LEFT
            elif diff < -0.15:
                intent_lbl = INTENT_BYPASS_RIGHT
            else:
                intent_lbl = INTENT_DIRECT

            dmins = res['clearance'].cpu().numpy()
            dmin = float(dmins.min()) if len(dmins) else 999.0

            ep_history.append({
                'win_prox': win_prox,
                'win_goal': win_goal,
                'reward': reward,
                'done': done,
                'intent_lbl': intent_lbl,
                'dmin': dmin,
                'prev_pos': prev_pos,
                'curr_pos': (px, py),
                'goal': (gx, gy)
            })

        # Compute return-to-go G_t
        G = 0.0
        for item in reversed(ep_history):
            G = item['reward'] + 0.99 * G
            item['return_to_go'] = G
            samples.append(item)

    t1 = time.time()
    print(f"Collected {len(samples)} transitions across {num_episodes} episodes in {t1 - t0:.2f}s")
    return samples

def train_strat_mamba_dual_stream(samples, device='cuda', epochs=15):
    print("\n--- STAGE 1: TRAINING STRATMAMBA DUAL-STREAM GOAL FUSION ---")
    env_config = configparser.RawConfigParser()
    env_config.read(KDARL_CLEAN / 'crowd_nav/configs/env.config')
    policy_config = configparser.RawConfigParser()
    policy_config.read(KDARL_CLEAN / 'crowd_nav/configs/policy.config')
    train_config = configparser.RawConfigParser()
    train_config.read(KDARL_CLEAN / 'crowd_nav/configs/train.config')

    cfg = configparser.RawConfigParser()
    for c in [env_config, policy_config, train_config]:
        for sec in c.sections():
            if not cfg.has_section(sec): cfg.add_section(sec)
            for k, v in c.items(sec): cfg.set(sec, k, v)

    cfg.set('mamba', 'temporal_backbone', 'gru')
    cfg.set('buffer', 'seq_len', '24')

    weights_path = WORKSPACE / 'baseline_weights/gru-t24-seed443.pth'
    model = StratMambaDualStreamPolicy(cfg, device=device, backbone='gru')
    model.load_pretrained_weights(weights_path)
    model.to(device)

    # Freeze base spatial and proximal temporal encoders; train only goal stream and W_fuse
    for p in model.spatial_encoder.parameters():
        p.requires_grad = False
    for p in model.dual_temporal.prox_encoder.parameters():
        p.requires_grad = False
    for p in model.value_head.parameters():
        p.requires_grad = False

    trainable_params = list(model.dual_temporal.goal_proj.parameters()) + \
                       list(model.dual_temporal.goal_encoder.parameters()) + \
                       list(model.dual_temporal.W_fuse.parameters())

    optimizer = optim.Adam(trainable_params, lr=1e-3, weight_decay=1e-5)
    criterion = nn.MSELoss()

    # Create tensors
    prox_all = torch.tensor(np.stack([s['win_prox'] for s in samples]), dtype=torch.float32)
    goal_all = torch.tensor(np.stack([s['win_goal'] for s in samples]), dtype=torch.float32)
    G_all = torch.tensor(np.array([s['return_to_go'] for s in samples]), dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(prox_all, goal_all, G_all)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for batch_prox, batch_goal, batch_G in loader:
            batch_prox = batch_prox.to(device)
            batch_goal = batch_goal.to(device)
            batch_G = batch_G.to(device)

            optimizer.zero_grad()
            vals = model.forward_value(batch_prox, batch_goal)
            loss = criterion(vals, batch_G)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_G)

        avg_loss = total_loss / len(samples)
        if epoch % 3 == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs:2d}: Dual-Stream Value MSE Loss = {avg_loss:.4f}")

    ckpt_path = CHECKPOINT_DIR / 'b1_strat_mamba_trained.pth'
    torch.save(model.state_dict(), ckpt_path)
    print(f"Saved trained StratMamba checkpoint to {ckpt_path}")
    return model

def train_lics_intent_head(samples, dual_model, device='cuda', epochs=20):
    print("\n--- STAGE 2: TRAINING LICS INTENT COMMITMENT & SWITCHING HEAD ---")
    env_config = configparser.RawConfigParser()
    env_config.read(KDARL_CLEAN / 'crowd_nav/configs/env.config')
    policy_config = configparser.RawConfigParser()
    policy_config.read(KDARL_CLEAN / 'crowd_nav/configs/policy.config')
    train_config = configparser.RawConfigParser()
    train_config.read(KDARL_CLEAN / 'crowd_nav/configs/train.config')

    cfg = configparser.RawConfigParser()
    for c in [env_config, policy_config, train_config]:
        for sec in c.sections():
            if not cfg.has_section(sec): cfg.add_section(sec)
            for k, v in c.items(sec): cfg.set(sec, k, v)

    cfg.set('mamba', 'temporal_backbone', 'gru')
    cfg.set('buffer', 'seq_len', '24')

    lics_policy = LICSStratMambaPolicy(cfg, device=device, backbone='gru', hysteresis_margin=0.15)
    # Load trained dual backbone
    lics_policy.load_state_dict(dual_model.state_dict(), strict=False)
    lics_policy.to(device)

    # Freeze dual backbone; train intent head
    for p in lics_policy.spatial_encoder.parameters(): p.requires_grad = False
    for p in lics_policy.dual_temporal.parameters(): p.requires_grad = False
    for p in lics_policy.value_head.parameters(): p.requires_grad = False

    intent_params = list(lics_policy.intent_head.parameters())
    optimizer = optim.Adam(intent_params, lr=1e-3, weight_decay=1e-5)

    prox_all = torch.tensor(np.stack([s['win_prox'] for s in samples]), dtype=torch.float32)
    goal_all = torch.tensor(np.stack([s['win_goal'] for s in samples]), dtype=torch.float32)
    lbl_all = torch.tensor(np.array([s['intent_lbl'] for s in samples]), dtype=torch.long)
    rew_all = torch.tensor(np.array([s['reward'] for s in samples]), dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(prox_all, goal_all, lbl_all, rew_all)
    loader = DataLoader(dataset, batch_size=128, shuffle=True)

    lics_policy.train()
    criterion_q = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        total_q_loss = 0.0
        total_beta_loss = 0.0

        for batch_prox, batch_goal, batch_lbl, batch_rew in loader:
            batch_prox = batch_prox.to(device)
            batch_goal = batch_goal.to(device)
            batch_lbl = batch_lbl.to(device)
            batch_rew = batch_rew.to(device)

            optimizer.zero_grad()

            with torch.no_grad():
                spat = lics_policy.spatial_encoder(batch_prox)
                h_fused = lics_policy.dual_temporal(spat, batch_goal)[:, -1, :]  # [B, d_model]

            # 1. Intent Q / Policy Loss (align with progressive passing corridor)
            q_vals = lics_policy.intent_head.get_intent_q_values(h_fused)  # [B, 4]
            loss_q = criterion_q(q_vals, batch_lbl)

            # 2. Option-Critic Termination Loss with Hysteresis Regularization
            # beta_psi(h_t, I_prev)
            # If current Q is high, beta should be LOW (commit!). If alternative is much better, beta can rise.
            prev_intents = batch_lbl  # active intent
            emb = lics_policy.intent_head.intent_embeddings(prev_intents)
            cat_feat = torch.cat([h_fused, emb], dim=-1)
            beta = lics_policy.intent_head.termination_net(cat_feat).squeeze(-1)  # [B]

            # Option-Critic termination gradient objective: beta * (Q(s, I) - max_I' Q(s, I') + eta)
            cur_q = q_vals.gather(1, prev_intents.unsqueeze(1)).squeeze(1)
            max_q = q_vals.max(dim=1)[0]
            advantage_to_switch = max_q - cur_q  # >= 0
            # Regulate with hysteresis margin eta = 0.15
            loss_beta = torch.mean(beta * (lics_policy.intent_head.hysteresis_margin - advantage_to_switch))

            total_loss = loss_q + 0.5 * loss_beta
            total_loss.backward()
            optimizer.step()

            total_q_loss += loss_q.item() * len(batch_lbl)
            total_beta_loss += loss_beta.item() * len(batch_lbl)

        if epoch % 4 == 0 or epoch == epochs:
            avg_q = total_q_loss / len(samples)
            avg_b = total_beta_loss / len(samples)
            print(f"Epoch {epoch:2d}/{epochs:2d}: Intent Q Loss = {avg_q:.4f}, Termination Hysteresis Loss = {avg_b:.4f}")

    ckpt_path = CHECKPOINT_DIR / 'b2_lics_trained.pth'
    torch.save(lics_policy.state_dict(), ckpt_path)
    print(f"Saved trained LICS checkpoint to {ckpt_path}")
    return lics_policy

def main():
    print("=================================================================")
    print("STARTING COMPLETE TRAINING PIPELINE (DUAL-STREAM & LICS)")
    print("=================================================================")
    t_start = time.time()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Training Device: {device}")

    # 1. Collect rollouts
    samples = collect_training_rollouts(num_episodes=40, device=device)

    # 2. Train StratMamba Dual-Stream Goal Fusion
    dual_model = train_strat_mamba_dual_stream(samples, device=device, epochs=15)

    # 3. Train LICS Intent Head
    lics_model = train_lics_intent_head(samples, dual_model, device=device, epochs=20)

    t_end = time.time()
    print("=================================================================")
    print(f"TRAINING COMPLETE IN {t_end - t_start:.2f} SECONDS!")
    print("=================================================================")

if __name__ == '__main__':
    main()
