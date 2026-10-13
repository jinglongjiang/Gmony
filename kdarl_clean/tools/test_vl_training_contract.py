"""Regression: value-lookahead actions must not be dropped by replay."""

import configparser
from pathlib import Path
import sys
import tempfile
import unittest
import subprocess
import os

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from crowd_nav.contracts import init_grid_from_cfg
from crowd_nav.policy.mamba_rl import MambaRLPolicy
from crowd_nav.utils.ppo_buffer import ReplayBufferIQL
from crowd_sim.envs.utils.state import FullState, ObservableState, JointState


class TrainingContract(unittest.TestCase):
    def test_resume_skips_il_and_restores_optimizer(self):
        from crowd_nav.train import load_config
        torch.set_num_threads(2)
        cfg = load_config(str(ROOT / 'crowd_nav/configs/env.config'))
        for section, name, value in (
                ('mamba', 'temporal_backbone', 'gru'), ('mamba', 'd_model', '32'),
                ('mamba', 'n_layers', '1'), ('train', 'train_episodes', '2'),
                ('train', 'save_every', '2'), ('train', 'replay_warmup', '1'),
                ('train', 'updates_per_ep', '4'), ('sarl', 'batch_size', '4'),
                ('sarl', 'eval_every', '999999'), ('buffer', 'capacity', '32'),
                ('imitation_learning', 'max_il_prefill', '1')):
            cfg.set(section, name, value)
        net = MambaRLPolicy(cfg, device='cpu')
        optimizer = torch.optim.AdamW(net.parameters(), lr=cfg.getfloat('sarl', 'value_lr'),
                                       weight_decay=.01, betas=(.9, .999))
        net.forward_value(torch.randn(2, 24, 8, 13)).square().mean().backward()
        optimizer.step()
        state = JointState(FullState(0, -4, 0, 0, .3, 0, 4, 1, 0),
                           [ObservableState(3 + i, i, 0, 0, .3) for i in range(5)])
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            dataset = directory / 'teacher.pth'
            torch.save({'trajectories': [{'states': [state.to_array()] * 4,
                                         'actions': [(0., 1.)] * 4,
                                         'rewards': [0., 0., 0., 1.],
                                         'meta': {'flag_success': True}}]}, dataset)
            cfg.set('train', 'offline_il_dataset', str(dataset))
            config = directory / 'resume.ini'
            with open(config, 'w') as stream:
                cfg.write(stream)
            with open(directory / 'env.config', 'w') as stream:
                cfg.write(stream)
            checkpoint = directory / 'initial.pth'
            torch.save({'episode': 1, 'stage': 'rl_training_sarl', 'algo': 'sarl',
                        'policy_state': net.state_dict(), 'target_value_net_state': net.state_dict(),
                        'optim_value_state': optimizer.state_dict()}, checkpoint)
            command = [sys.executable, '-m', 'crowd_nav.train', '--config', str(config),
                       '--outdir', str(directory / 'run'), '--device', 'cpu', '--seed', '419',
                       '--resume', str(checkpoint), '--resume-rebuild-il-replay']
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=180,
                                    env={**os.environ, 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'})
            self.assertEqual(result.returncode, 0, result.stdout[-3000:] + result.stderr[-3000:])
            console = result.stdout + result.stderr
            self.assertNotIn('[IL-BC] Starting BC training phase', console)
            self.assertIn('[RESUME-OPTIMIZER] restored_steps=[1]', console)
            self.assertIn('Successfully resumed from episode 2', console)
            final = torch.load(directory / 'run/rl_model_ep2.pth', weights_only=False)
            self.assertEqual(final['episode'], 2)
            steps = {int(s['step'].item()) for s in final['optim_value_state']['state'].values()}
            self.assertEqual(steps, {5})

    def test_fixed_budget_config_wrapper(self):
        from crowd_nav.train import TrainConfig, _run_policy_il_pretrain
        torch.set_num_threads(2)
        cfg = configparser.RawConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
        cfg.read([str(ROOT / 'crowd_nav/configs' / f'{name}.config')
                  for name in ('env', 'policy', 'train')])
        cfg.set('mamba', 'temporal_backbone', 'gru')
        cfg.set('train', 'il_fixed_budget', 'true')
        net = MambaRLPolicy(cfg, device='cpu')
        state = JointState(FullState(0, -4, 0, 0, .3, 0, 4, 1, 0),
                           [ObservableState(3 + i, i, 0, 0, .3) for i in range(5)])
        with tempfile.TemporaryDirectory() as directory:
            wrapper = TrainConfig(cfg, directory)
            self.assertTrue(wrapper.il_fixed_budget)
            path = Path(directory) / 'il.pth'
            trajectory = ([state.to_array()] * 9, [(0., 1.)] * 9, [0.] * 8 + [1.])
            _run_policy_il_pretrain(net, torch.device('cpu'), 24, 2, 8, str(path), None,
                                    gamma=.99, success_trajs=[(trajectory, {})], config=wrapper)
            checkpoint = torch.load(path, weights_only=False)
            self.assertEqual(checkpoint['meta']['epochs'], 2)
            self.assertEqual(checkpoint['meta']['objective'], 'value_regression')
            self.assertTrue(np.isfinite(checkpoint['meta']['value_loss']))

    def test_token_conversion_commutes_with_windowing(self):
        from crowd_nav.train import make_il_token_windows
        from crowd_nav.contracts import joint34_to_tokens
        rng = np.random.default_rng(419)
        for length in (1, 3, 24, 31):
            raw = rng.normal(size=(length, 34)).astype(np.float32)
            for seq_len in (1, 3, 24):
                windows = []
                for t in range(length):
                    window = raw[max(0, t - seq_len + 1):t + 1]
                    window = np.concatenate([np.repeat(window[:1], seq_len - len(window), axis=0), window])
                    windows.append(window)
                expected = joint34_to_tokens(np.asarray(windows).reshape(-1, 34))
                expected = expected.reshape(length, seq_len, 8, 13).numpy()
                np.testing.assert_array_equal(make_il_token_windows(raw, seq_len), expected)

    def test_value_policy_records_real_grid_index_and_mc_targets(self):
        torch.set_num_threads(2)
        cfg = configparser.RawConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
        cfg.read([str(ROOT / 'crowd_nav/configs' / f'{name}.config')
                  for name in ('env', 'policy', 'train')])
        env_cfg = configparser.RawConfigParser(inline_comment_prefixes=(';', '#'))
        env_cfg.read(str(ROOT / 'crowd_nav/configs/env.config'))
        init_grid_from_cfg(env_cfg)
        cfg.set('mamba', 'temporal_backbone', 'gru')
        net = MambaRLPolicy(cfg, device='cpu')
        net.use_sarl_predict = True
        net.set_phase('train')
        state = JointState(FullState(0, -4, 0, 0, .3, 0, 4, 1, 0),
                           [ObservableState(3 + i, i, 0, 0, .3) for i in range(5)])
        actions, indices = [], []
        with torch.no_grad():
            for epsilon in (1., 0., 0.):
                net.epsilon = epsilon
                action, index = net.act(state)
                self.assertIsInstance(index, int)
                self.assertEqual(tuple(action), tuple(net.action_space[index]))
                actions.append(tuple(action))
                indices.append(index)
            action, index = net.predict(state, return_idx=True)
            self.assertEqual(tuple(action), tuple(net.action_space[index]))
        replay = ReplayBufferIQL(capacity=16, seq_len=24, gamma=.99)
        replay.push_episode({'states': [state.to_array()] * 3, 'actions': actions,
                             'action_indices': indices, 'rewards': [0., 0., 1.],
                             'dones': [False, False, True]})
        self.assertEqual(len(replay), 3)
        np.testing.assert_allclose(replay.returns[:3], [.99 ** 2, .99, 1.])
        np.testing.assert_array_equal(replay.action_indices[:3], indices)


if __name__ == '__main__':
    unittest.main()
