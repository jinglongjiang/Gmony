"""Merged scorer/replay parity with frozen IL weights, including train dropout."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from crowd_nav.utils.ppo_buffer import ReplayBufferIQL, ReplayBufferMC


def replay_parity():
    rng = np.random.default_rng(419)
    checked = 0
    for per in (False, True):
        for mc_only in (False, True):
            for n_step in (1, 3):
                kw = dict(capacity=127, seq_len=24, gamma=.99, use_per=per, n_step=n_step)
                old, new = ReplayBufferIQL(**kw), ReplayBufferMC(**kw, mc_only=mc_only)
                for length in (2, 3, 24, 61, 43, 38):
                    episode = {'states': rng.normal(size=(length, 34)).astype('float32'),
                               'actions': rng.normal(size=(length, 2)).astype('float32'),
                               'rewards': rng.normal(size=length).astype('float32'),
                               'dones': np.arange(length) == length - 1,
                               'action_indices': rng.integers(0, 80, length).tolist()}
                    old.store_episode(episode)
                    new.store_episode(episode)
                assert (old.ptr, old.size) == (new.ptr, new.size)
                for field in ('states', 'actions', 'action_indices', 'returns', 'rewards', 'dones', 'priorities'):
                    np.testing.assert_array_equal(getattr(old, field), getattr(new, field))
                if mc_only:
                    assert new.next_states is None
                else:
                    np.testing.assert_array_equal(old.next_states, new.next_states)
                np.random.seed(17)
                a = old.sample(64)
                np.random.seed(17)
                b = new.sample(64)
                for field in b:
                    if b[field] is None:
                        continue
                    np.testing.assert_array_equal(a[field], b[field])
                checked += 1
    return {'passed': True, 'combinations': checked, 'ring_wrap_tested': True,
            'mc_targets_exact': True, 'sample_indices_exact': True}


def scorer_parity(previous):
    from crowd_nav.train import load_config, build_env_and_robot, bind_policy
    from crowd_nav.policy.mamba_rl import MambaRLPolicy
    from crowd_nav.contracts import init_grid_from_cfg
    from crowd_sim.envs.utils.state import JointState
    torch.set_num_threads(2)
    rows = {}
    for arm in ('kda', 'mamba', 'gru'):
        cfg = load_config(str(previous / f'{arm}.ini'))
        init_grid_from_cfg(cfg)
        model = MambaRLPolicy(cfg, device='cuda').cuda().eval()
        ckpt = torch.load(previous / arm / 'il_policy.pth', map_location='cpu', weights_only=False)
        assert ckpt['meta']['epochs'] == 50 and ckpt['meta']['objective'] == 'value_regression'
        weights = ckpt['value']
        assert all(torch.isfinite(x).all() for x in weights.values())
        model.load_state_dict(weights, strict=True)
        model.use_sarl_predict = True
        model.multiagent_training = True
        model.epsilon = 0.
        model.set_training_mode('rl')
        worst, count = {}, 0
        for n in (5, 10, 20):
            for geometry in ('circle_crossing', 'square_crossing'):
                cfg.set('sim', 'human_num', str(n))
                cfg.set('sim', 'test_sim', geometry)
                env, robot = build_env_and_robot(cfg)
                bind_policy(robot, model, env, epsilon=0.)
                env.phase = 'test'
                for case in (88000, 88001):
                    env.reset(seed=case, options={'test_case': case})
                    model.reset_episode_stats()
                    model.eval()
                    model.set_phase('test')
                    for step in range(26):
                        state = JointState(robot.get_full_state(), [h.get_observable_state() for h in env.humans])
                        if step in (0, 2, 23, 25):
                            for training in (False, True):
                                model.train(training)
                                model.set_phase('train' if training else 'test')
                                cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
                                history = [x.copy() for x in model._history]
                                a = model._score_sarl_candidates_legacy(state)
                                torch.set_rng_state(cpu_rng)
                                torch.cuda.set_rng_state(gpu_rng)
                                b = model.score_sarl_candidates(state)
                                assert len(a['commands']) == len(b['commands']) == 80
                                assert a['commands'] == b['commands']
                                for field in ('windows', 'current_token', 'rewards', 'clearance', 'values', 'raw_scores', 'scores'):
                                    x, y = a[field], b[field]
                                    x = torch.as_tensor(x, device='cuda')
                                    y = torch.as_tensor(y, device='cuda')
                                    error = float((x - y).abs().max())
                                    worst[field] = max(worst.get(field, 0.), error)
                                    if field in ('windows', 'current_token'):
                                        assert torch.equal(x, y), (arm, field, error)
                                    else:
                                        torch.testing.assert_close(x, y, rtol=0., atol=1e-6)
                                assert a['scores'].argmax() == b['scores'].argmax(), (arm, n, geometry, case, step)
                                assert len(history) == len(model._history)
                                for x, y in zip(history, model._history):
                                    np.testing.assert_array_equal(x, y)
                                count += 1
                        model.eval()
                        model.set_phase('test')
                        out = env.step(model.predict(state))
                        done = (out[2] or out[3]) if len(out) == 5 else out[2]
                        if done:
                            break
        rows[arm] = {'passed': True, 'comparisons': count, 'max_abs_error': worst,
                     'argmax_mismatches': 0, 'history_unchanged': True,
                     'strict_il_load': True, 'train_dropout_rng_matched': True}
        print(json.dumps({arm: rows[arm]}), flush=True)
        del model, ckpt
        torch.cuda.empty_cache()
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    results = {'replay': replay_parity()}
    if args.previous:
        results['scorer'] = scorer_parity(Path(args.previous))
    results['passed'] = True
    Path(args.output).write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
