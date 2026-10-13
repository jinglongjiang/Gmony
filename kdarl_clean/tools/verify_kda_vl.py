"""Correctness-only VL-v2 freeze. No environment rollout or optimizer steps."""

import argparse
import configparser
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tarfile
import tempfile
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
ORIGINAL = Path('/home/abc/old/CrowdNav(20270731_backup2)/CrowdNav')
DEFAULT_CHECKPOINT = ORIGINAL / 'crowd_nav/runs/mamba_vl/rl_model_ep10000_T24.pth'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def config(root):
    cfg = configparser.ConfigParser()
    cfg.read([str(root / 'crowd_nav/configs' / f'{name}.config')
              for name in ('env', 'policy', 'train')])
    return cfg


def fixture(n=5):
    from crowd_sim.envs.utils.state import FullState, ObservableState, JointState
    robot = FullState(1.25, -2.0, .2, -.1, .3, 3.75, 4.5, 1.2, .4)
    humans = [ObservableState(4.5 + i, -.5 + .1 * i, -.2, .15, .3)
              for i in range(n)]
    if n > 5:
        humans[-1] = ObservableState(1.7, -2., -.5, 0., .3)
    return JointState(robot, humans)


def load_model(root, checkpoint):
    from crowd_nav.policy.mamba_rl import MambaRLPolicy
    model = MambaRLPolicy(config(root), device=torch.device('cuda')).eval()
    data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    weights = {k.removeprefix('_orig_mod.'): v for k, v in data['policy_state'].items()}
    model.load_state_dict(weights, strict=True)
    model.set_phase('test')
    model.epsilon = 0.
    return model


def legacy_worker(args):
    root = Path(args.source_root)
    sys.path.insert(0, str(root))
    from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized
    from crowd_sim.envs.utils.action import ActionXY
    torch.set_num_threads(2)
    torch.manual_seed(419)
    model = load_model(root, args.checkpoint)
    state = fixture()
    model._last_action = ActionXY(state.self_state.vx, state.self_state.vy)
    arrays = {}
    captures = {}
    def save(name):
        def hook(module, inputs, output):
            captures[name] = output.detach().float().cpu().numpy()
        return hook
    model.spatial_encoder.register_forward_hook(save('scene'))
    model.temporal_encoder.register_forward_hook(save('temporal'))
    def record_actor_input(module, inputs):
        captures['actor-input'] = inputs[0].detach().cpu().numpy()
    model.spatial_encoder.human_encoder.register_forward_pre_hook(record_actor_input)
    original_forward = model.forward_value
    def forward(tokens, *a, **kw):
        captures['windows'] = tokens.detach().cpu().numpy()
        result = original_forward(tokens, *a, **kw)
        captures['values'] = result.detach().cpu().numpy()
        return result
    model.forward_value = forward
    prediction_code = model.predict_sarl_style.__func__.__code__
    def record_prediction(frame, event, arg):
        if frame.f_code is prediction_code and event == 'return':
            local = frame.f_locals
            captures['scores'] = local['total_values'].detach().cpu().numpy()
            captures['rewards'] = local['rewards_tensor'].detach().cpu().numpy()
            captures['raw-scores'] = (local['rewards_tensor'] + model.gamma * local['next_values']).detach().cpu().numpy()
            captures['clearance'] = np.asarray(local['dmins_batch'])
        return record_prediction
    for step in range(3):
        arrays[f'{step}-previous-executed'] = np.asarray(model._last_action)
        packed = model._build_joint_state_34(state.self_state, state.human_states)
        arrays[f'{step}-token'] = _batch_joint34_to_tokens_vectorized(packed[None])
        sys.settrace(record_prediction)
        try:
            with torch.no_grad():
                action = model.predict_sarl_style(state)
        finally:
            sys.settrace(None)
        for key, value in captures.items():
            arrays[f'{step}-{key}'] = value
        arrays[f'{step}-grid'] = np.asarray(model.action_space)
        arrays[f'{step}-executed'] = np.array(action)
        state.self_state = model.propagate(state.self_state, action)
        state.human_states = [model.propagate(h, ActionXY(h.vx, h.vy)) for h in state.human_states]
    np.savez_compressed(args.output, **arrays)


def legacy_parity(out, checkpoint):
    report = {'checkpoint': str(checkpoint), 'checkpoint_sha256': sha(checkpoint),
              'paper_checkpoint_provenance': 'UNVERIFIED',
              'paper_benchmark': 'NOT_RUN', 'historical_runtime_parity': 'UNVERIFIED',
              'scope': 'original backup versus tagged restored source, same installed runtime'}
    with tempfile.TemporaryDirectory(prefix='vl-legacy-') as directory:
        folder = Path(directory)
        files = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only',
                                         'mamba-vl-legacy-20261006', 'CrowdNav'],
                                        cwd=PROJECT.parent, text=True).splitlines()
        files = [p for p in files if p.endswith(('.py', '.config'))]
        archive = subprocess.check_output(['git', 'archive', 'mamba-vl-legacy-20261006', *files],
                                          cwd=PROJECT.parent)
        with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
            stream.extractall(folder, filter='data')
        snapshot = folder / 'CrowdNav'
        paths = [ORIGINAL, snapshot, snapshot]
        arrays = []
        for i, root in enumerate(paths):
            target = out / f'legacy-trace-{i}.npz'
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '--legacy-worker',
                            '--source-root', str(root), '--checkpoint', str(checkpoint),
                            '--output', str(target)], cwd=root, check=True,
                           env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'})
            arrays.append(dict(np.load(target)))
        report['comparisons'] = {}
        for label, left, right in [('source_vs_tag', arrays[0], arrays[1]),
                                   ('repeatability', arrays[1], arrays[2])]:
            errors = {k: float(np.max(np.abs(left[k] - right[k]))) for k in left}
            assert all(error == 0 for error in errors.values()), errors
            report['comparisons'][label] = {'bitwise_equal': True, 'max_abs_errors': errors}
    report['config_hashes'] = {p.name: sha(p) for p in ORIGINAL.joinpath('crowd_nav/configs').glob('*.config')}
    write_json(out / 'legacy-parity.json', report)


def state_contract(out):
    from crowd_nav.contracts import _batch_joint34_to_tokens_vectorized
    from crowd_nav.policy.mamba_rl import MambaRLPolicy
    model = MambaRLPolicy(config(PROJECT), device=torch.device('cuda')).eval()
    state = fixture()
    packed = model._build_joint_state_34(state.self_state, state.human_states)
    token = _batch_joint34_to_tokens_vectorized(packed[None])[0]
    expected = np.hypot(3.75 - 1.25, 4.5 + 2.0)
    assert np.isclose(token[0, 9], expected)
    assert np.isclose(token[0, 11], 1.2)
    selection = []
    for n in (5, 10, 20):
        state = fixture(n)
        packed = model._build_joint_state_34(state.self_state, state.human_states)
        selected = _batch_joint34_to_tokens_vectorized(packed[None])[0, 3:8, :2]
        supplied = [[h.px, h.py] for h in state.human_states]
        entering = (selected + np.array([state.self_state.px, state.self_state.py])).tolist()
        assert all(any(np.allclose(point, s) for s in supplied[:5]) for point in entering)
        selection.append({'n': n, 'supplied_humans': supplied, 'entering_tokens': entering,
                          'first_five_only': True})
    legacy = dict(np.load(out / 'legacy-trace-0.npz'))
    actor = legacy['0-actor-input'][0]
    robot_xy = legacy['0-windows'][0, 0, 0, :2]
    assert np.allclose(actor[13:15], actor[:2] - robot_xy)
    action_errors = []
    for step in range(3):
        best = int(legacy[f'{step}-scores'].argmax())
        scored_grid = legacy[f'{step}-grid'][best]
        execution = legacy[f'{step}-executed']
        action_errors.append({'step': step, 'scored_grid': scored_grid.tolist(),
                              'executed': execution.tolist(),
                              'mismatch_norm': float(np.linalg.norm(scored_grid - execution))})
    assert any(row['mismatch_norm'] > 1e-6 for row in action_errors)
    return {'packed_robot_fields': ['px', 'py', 'vx', 'vy', 'radius', 'gx', 'gy', 'v_pref', 'theta'],
            'sentinel_packed': packed[:9].tolist(), 'correct_goal_distance': float(token[0, 9]),
            'correct_v_pref': float(token[0, 11]), 'selection': selection,
            'legacy_goal_distance': float(legacy['0-token'][0, 0, 9]),
            'legacy_v_pref': float(legacy['0-token'][0, 0, 11]),
            'legacy_relation_input': {'token_relative_xy': actor[:2].tolist(),
                                      'robot_absolute_xy': robot_xy.tolist(),
                                      'derived_relation_xy': actor[13:15].tolist(),
                                      'subtracts_robot_twice': True},
            'legacy_scoring_execution_mismatch': action_errors}


def tensor_hash(state):
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode())
        h.update(str((value.dtype, value.shape)).encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def native_kernel_parity():
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.kda.naive import naive_recurrent_kda
    from fla.utils import assert_close
    torch.manual_seed(7)
    shape = (1, 24, 4, 64)
    q, k = [torch.nn.functional.normalize(torch.randn(shape, device='cuda'), dim=-1).to(torch.bfloat16)
            for _ in range(2)]
    v = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
    g = torch.nn.functional.logsigmoid(torch.randn(shape, device='cuda')) / 10
    beta = torch.randn(1, 24, 4, device='cuda').sigmoid().to(torch.bfloat16)
    tensors = [x.detach().requires_grad_() for x in (q, k, v, g, beta)]
    ref, _ = naive_recurrent_kda(*tensors)
    result = {}
    for name, kernel in [('chunk', chunk_kda), ('recurrent', fused_recurrent_kda)]:
        inputs = [x.detach().requires_grad_() for x in tensors]
        got, _ = kernel(q=inputs[0], k=inputs[1], v=inputs[2], g=inputs[3], beta=inputs[4])
        assert_close(name, ref, got, 0.005)
        result[name] = {'max_abs_error': float((got.float() - ref.float()).abs().max()),
                        'official_rms_tolerance': .005}
        if name == 'chunk':
            ref_grads = torch.autograd.grad(ref.float().square().mean(), tensors, retain_graph=True)
            got_grads = torch.autograd.grad(got.float().square().mean(), inputs)
            for label, a, b in zip(('q', 'k', 'v', 'g', 'beta'), ref_grads, got_grads):
                assert_close(label, a, b, 0.02)
            result[name]['gradient_parity'] = True
    from unittest.mock import patch
    from fla.layers.kda import KimiDeltaAttention
    from fla.ops.kda.gate import naive_kda_gate
    layer = KimiDeltaAttention(hidden_size=256, head_dim=64, num_heads=4).cuda().train()
    x = torch.randn(1, 24, 256, device='cuda')
    def reference_layer_kernel(**kw):
        q = torch.nn.functional.normalize(kw['q'].float(), dim=-1).to(kw['q'].dtype)
        k = torch.nn.functional.normalize(kw['k'].float(), dim=-1).to(kw['k'].dtype)
        g = naive_kda_gate(kw['g'], kw['A_log'], kw['dt_bias'])
        beta = kw['beta'].float().sigmoid().to(kw['beta'].dtype)
        return naive_recurrent_kda(q, k, kw['v'], g, beta)
    with torch.autocast('cuda', dtype=torch.bfloat16):
        optimized = layer(x)[0]
        with patch('fla.layers.kda.chunk_kda', reference_layer_kernel):
            reference = layer(x)[0]
    assert_close('full_layer', reference, optimized, .01)
    result['full_layer'] = {'max_abs_error': float((reference.float() - optimized.float()).abs().max()),
                            'rms_tolerance': .01, 'reference': 'official naive recurrence and gate'}
    return result


def backbones(out):
    from crowd_nav.policy.mamba_rl import MambaRLPolicy
    from crowd_nav.policy.shared_initialization import initialize_shared
    from crowd_sim.envs.utils.action import ActionXY
    results, hashes = {}, {}
    common_reward, common_commands = None, None
    for seed in (419, 443, 467, 491):
        hashes[str(seed)] = {}
        for kind in ('mamba', 'gru', 'kda'):
            print(f'Checking seed={seed} backbone={kind}', flush=True)
            cfg = config(PROJECT)
            cfg.set('mamba', 'temporal_backbone', kind)
            torch.manual_seed(seed)
            model = MambaRLPolicy(cfg, device=torch.device('cuda'))
            initialize_shared(model, seed)
            shared = {k: v for k, v in model.state_dict().items() if not k.startswith('temporal_encoder.')}
            hashes[str(seed)][kind] = tensor_hash(shared)
            if seed != 419:
                del model
                continue
            model.eval()
            model.set_phase('test')
            state = fixture()
            model._last_action = ActionXY(state.self_state.vx, state.self_state.vy)
            history_before = list(model._history)
            previous_action = model._last_action
            actor_inputs = []
            def capture_actor(module, inputs):
                actor_inputs.append(inputs[0].detach().cpu().numpy())
            handle = model.spatial_encoder.human_encoder.register_forward_pre_hook(capture_actor)
            scored = model.score_sarl_candidates(state)
            handle.remove()
            actor = actor_inputs[0]
            assert np.array_equal(actor[:, :2], actor[:, 13:15])
            assert list(model._history) == history_before and model._last_action == previous_action
            reward = scored['rewards'].cpu().numpy()
            commands = np.asarray(scored['commands'])
            if common_reward is None:
                common_reward, common_commands = reward.copy(), commands.copy()
            else:
                assert np.array_equal(reward, common_reward)
                assert np.array_equal(commands, common_commands)
            expected = .3 * np.array(previous_action) + .7 * np.array(model.action_space)
            assert np.allclose(commands, expected)
            chosen = int(scored['scores'].argmax())
            executed = model.predict_sarl_style(state)
            assert executed == scored['commands'][chosen] and len(model._history) == 1
            np.savez_compressed(out / f'candidate-contract-{kind}.npz',
                                grid=np.asarray(model.action_space), commands=commands,
                                windows=scored['windows'].cpu().numpy(), rewards=reward,
                                clearance=scored['clearance'].cpu().numpy(),
                                values=scored['values'].cpu().numpy(),
                                raw_scores=scored['raw_scores'].cpu().numpy(),
                                scores=scored['scores'].cpu().numpy(), executed=np.asarray(executed))
            postprocess_checks = []
            for distance in (.65, .01):
                model._last_action = previous_action
                mixed = fixture()
                mixed.human_states[0].px = mixed.self_state.px + distance
                mixed.human_states[0].py = mixed.self_state.py
                mixed.human_states[0].vx = 0.
                mixed.human_states[0].vy = 0.
                check = model.score_sarl_candidates(mixed)
                clearance = check['clearance']
                safe = clearance >= model.test_min_clearance
                expected_scores = check['raw_scores'].clone()
                if safe.any():
                    expected_scores = expected_scores.masked_fill(~safe, -1e9)
                expected_scores = expected_scores - model.test_risk_lambda * torch.clamp(
                    model.test_min_clearance - clearance, min=0.)
                expected_scores[0] -= 1e-3
                torch.testing.assert_close(check['scores'], expected_scores, atol=0, rtol=0)
                postprocess_checks.append({'safe_count': int(safe.sum()), 'all_unsafe': not bool(safe.any())})
            assert 0 < postprocess_checks[0]['safe_count'] < 80
            assert postprocess_checks[1]['all_unsafe']
            with torch.no_grad():
                scalar = torch.cat([model.forward_value(w[None]) for w in scored['windows']])
                torch.testing.assert_close(scalar, scored['values'], atol=1e-4, rtol=1e-4)
                torch.manual_seed(8)
                x = torch.randn(2, 24, 256, device='cuda')
                changed = x.clone()
                changed[:, 12:] += 4.
                first = model.temporal_encoder(x)
                second = model.temporal_encoder(changed)
                torch.testing.assert_close(first[:, :12], second[:, :12], atol=1e-4, rtol=1e-4)
                again = model.temporal_encoder(x)
                torch.testing.assert_close(first, again, atol=0, rtol=0)
                assert first.shape == (2, 24, 256)
            with tempfile.TemporaryDirectory() as tmp:
                weights = Path(tmp) / 'weights.pth'
                torch.save(model.state_dict(), weights)
                restored = MambaRLPolicy(cfg, device=torch.device('cuda')).eval()
                restored.load_state_dict(torch.load(weights, weights_only=True), strict=True)
                with torch.no_grad():
                    torch.testing.assert_close(model.forward_value(scored['windows'][:2]),
                                               restored.forward_value(scored['windows'][:2]), atol=0, rtol=0)
                del restored
            model.train()
            model.zero_grad(set_to_none=True)
            model.forward_value(scored['windows'][:2]).square().mean().backward()
            gradients = {}
            for name in ('spatial_encoder', 'temporal_encoder', 'value_head'):
                grads = [p.grad for p in getattr(model, name).parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
                norm = float(torch.stack([g.float().square().sum() for g in grads]).sum().sqrt())
                assert norm > 0
                gradients[name] = norm
            model.eval()
            with torch.no_grad():
                for _ in range(3):
                    model.forward_value(scored['windows'])
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                baseline_memory = torch.cuda.memory_allocated()
                start = time.perf_counter()
                for _ in range(10):
                    model.forward_value(scored['windows'])
                torch.cuda.synchronize()
                latency = (time.perf_counter() - start) * 1000 / 10
            results[kind] = {'shape': list(first.shape), 'causal': True, 'save_load': True,
                             'window_isolation': True, 'history_isolation': True,
                             'batch_scalar_max_error': float((scalar - scored['values']).abs().max()),
                             'execution_contract': True, 'n_actions': len(commands),
                             'filter_risk_fallback': postprocess_checks,
                             'rewards_identical': True, 'commands_identical': True,
                             'parameters_total': sum(p.numel() for p in model.parameters()),
                             'parameters_temporal': sum(p.numel() for p in model.temporal_encoder.parameters()),
                             'parameters_active_value_path': sum(p.numel() for name in
                                 ('spatial_encoder', 'temporal_encoder', 'value_head')
                                 for p in getattr(model, name).parameters()),
                             'gradient_norms': gradients, 'batch80_forward_ms': latency,
                             'peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                             'extra_forward_bytes': torch.cuda.max_memory_allocated() - baseline_memory,
                             'precision': 'FP32 all backbones; no autocast'}
            if kind == 'mamba':
                from mamba_ssm.modules import mamba_simple
                results[kind]['native_fast_path_requested'] = model.temporal_encoder.use_fast_path
                results[kind]['optional_causal_conv1d_available'] = mamba_simple.causal_conv1d_fn is not None
            print(f'{kind} correctness PASS; batch80={latency:.2f} ms', flush=True)
            model.reset_episode_stats()
            assert len(model._history) == 0 and model._last_action is None
            del model
        assert len(set(hashes[str(seed)].values())) == 1, hashes
    results['official_kernel_reference'] = native_kernel_parity()
    results['state_contract'] = state_contract(out)
    results['training'] = 'NOT_RUN'
    results['freeze_status'] = 'CORRECTNESS_PASS'
    write_json(out / 'backbone-correctness.json', results)
    write_json(out / 'shared-init-hashes.json', hashes)


def manifest(out):
    names = ('torch', 'triton', 'numpy', 'mamba-ssm', 'flash-linear-attention', 'transformers', 'pyrvo2')
    data = {'python': sys.version, 'python_executable': sys.executable,
            'platform': platform.platform(), 'cuda_runtime': torch.version.cuda,
            'gpu': torch.cuda.get_device_name(),
            'packages': {name: importlib.metadata.version(name) for name in names},
            'legacy_tag': 'mamba-vl-legacy-20261006',
            'legacy_commit': subprocess.check_output(['git', 'rev-parse', 'mamba-vl-legacy-20261006'],
                                                     cwd=PROJECT, text=True).strip(),
            'configs': {str(p.relative_to(PROJECT)): sha(p) for p in PROJECT.joinpath('crowd_nav/configs').glob('*.config')},
            'source_files': {str(p.relative_to(PROJECT)): sha(p) for folder in ('crowd_nav', 'tools')
                             for p in PROJECT.joinpath(folder).rglob('*.py')},
            'container_glibc': platform.libc_ver(),
            'container_image': 'nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04'}
    try:
        data['optional_causal_conv1d'] = importlib.metadata.version('causal-conv1d')
    except importlib.metadata.PackageNotFoundError:
        data['optional_causal_conv1d'] = 'NOT_INSTALLED'
    # Absolute assets are recorded as dependencies, never imported from shixu.
    repos = Path('/home/abc/workspace/il_x_rl_candidates_20261003/repos')
    data['official_repositories'] = {}
    for name in ('flash-linear-attention', 'Kimi-Linear'):
        repo = repos / name
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
        status = subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True)
        data['official_repositories'][name] = {'path': str(repo), 'commit': commit, 'status': status}
    write_json(out / 'environment-manifest.json', data)
    freeze = subprocess.check_output(['uv', 'pip', 'freeze', '--python', sys.executable], text=True)
    (out / 'environment-requirements.txt').write_text(
        '--extra-index-url https://download.pytorch.org/whl/cu126\n' + freeze)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output', type=Path, default=PROJECT / 'artifacts/kda-vl-freeze')
    parser.add_argument('--source-root')
    parser.add_argument('--legacy-worker', action='store_true')
    args = parser.parse_args()
    if args.legacy_worker:
        legacy_worker(args)
        return
    sys.path.insert(0, str(PROJECT))
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    legacy_parity(args.output, args.checkpoint)
    print('Legacy source/checkpoint numerical parity PASS', flush=True)
    backbones(args.output)
    manifest(args.output)
    print('VL-v2 correctness PASS; no training or performance evaluation performed', flush=True)


if __name__ == '__main__':
    main()
