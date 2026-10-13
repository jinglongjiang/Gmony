"""One paired-seed, final-budget VL comparison using the parent train entry."""

import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
ARMS = ('kda', 'mamba', 'gru')


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def base_config():
    from crowd_nav.train import load_config
    return load_config(str(PROJECT / 'crowd_nav/configs/env.config'))


def model(cfg, arm, seed):
    from crowd_nav.policy.mamba_rl import MambaRLPolicy
    from crowd_nav.policy.shared_initialization import initialize_shared
    torch.manual_seed(seed)
    cfg.set('mamba', 'temporal_backbone', arm)
    net = MambaRLPolicy(cfg, device='cuda')
    initialize_shared(net, seed)
    return net


def shared_hash(net):
    digest = hashlib.sha256()
    for name, value in sorted(net.state_dict().items()):
        if name.startswith('temporal_encoder.'):
            continue
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def prepare(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(PROJECT / 'crowd_nav/configs/env.config', out / 'env.config')
    manifest = out / 'protocol.json'
    if manifest.exists():
        old = json.loads(manifest.read_text())
        assert old['seed'] == args.seed and old['dataset_sha256'] == sha(args.dataset)
        return old
    dataset = torch.load(args.dataset, map_location='cpu', weights_only=False)
    assert dataset['config']['human_num'] == 5
    assert all(np.asarray(t['states']).shape[1] == 34 for t in dataset['trajectories'])
    cfg = base_config()
    cfg.set('train', 'train_episodes', '3000')
    cfg.set('train', 'il_fixed_budget', 'true')
    cfg.set('train', 'offline_il_dataset', str(Path(args.dataset).resolve()))
    cfg.set('train', 'save_every', '500')
    hashes = {}
    measurements = {}
    for arm in ARMS:
        net = model(cfg, arm, args.seed)
        hashes[arm] = shared_hash(net)
        net.train()
        x = torch.randn(2, 24, 8, 13, device='cuda')
        y = net.forward_value(x)
        y.square().mean().backward()
        assert torch.isfinite(y).all()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        measurements[arm] = {'params': sum(p.numel() for p in net.parameters())}
        del net
        torch.cuda.empty_cache()
    assert len(set(hashes.values())) == 1, hashes
    for arm in ARMS:
        cfg.set('mamba', 'temporal_backbone', arm)
        with open(out / f'{arm}.ini', 'w') as stream:
            cfg.write(stream)
    cfg.set('mamba', 'temporal_backbone', 'kda')
    cfg.set('train', 'train_episodes', '100')
    cfg.set('train', 'save_every', '100')
    with open(out / 'smoke.ini', 'w') as stream:
        cfg.write(stream)
    protocol = {
        'seed': args.seed, 'backbones': ARMS, 'rl_episodes': 3000,
        'il_epochs': cfg.getint('train', 'il_epochs'), 'fixed_il_budget': True,
        'dataset': str(Path(args.dataset).resolve()), 'dataset_sha256': sha(args.dataset),
        'dataset_config': dataset['config'], 'demonstrations': len(dataset['trajectories']),
        'teacher_action_grid_is_not_student_grid': True,
        'il_objective': 'scalar MC value MSE; teacher action indices unused',
        'shared_initialization_hashes': hashes, 'untrained_checks': measurements,
        'eval_cases': list(range(args.case_start, args.case_start + 32)),
        'eval_cells': [[n, g] for n in (5, 10, 20) for g in ('circle_crossing', 'square_crossing')],
        'eval_time_limit': 50, 'circle_radius': 4, 'square_width': 10,
        'selection': 'final episode 3000 only; no best or intermediate checkpoint selection',
        'smoke_isolation': 'random-init online stability only; no IL; independent output; formal starts fresh IL',
        'parent_freeze_commit': '19e6031',
        'experiment_source_commit': args.source_commit,
        'code_sha256': {str(p.relative_to(PROJECT)): sha(p)
                        for base in ('crowd_nav', 'crowd_sim', 'tools')
                        for p in (PROJECT / base).rglob('*.py')},
        'runtime': {'python': platform.python_version(), 'torch': torch.__version__,
                    'gpu': torch.cuda.get_device_name(),
                    'driver': subprocess.check_output(['nvidia-smi', '--query-gpu=driver_version',
                                                       '--format=csv,noheader'], text=True).strip()},
        'formal_status': 'NOT_RUN',
    }
    import fla
    import mamba_ssm
    from mamba_ssm.modules import mamba_simple
    protocol['runtime'].update({'fla': fla.__version__, 'mamba_ssm': mamba_ssm.__version__,
                                'mamba_causal_conv1d_available': mamba_simple.causal_conv1d_fn is not None,
                                'fla_source_commit': '9f38d24980c46d46bd38614e743cdacd21906578'})
    save(manifest, protocol)
    return protocol


def train(args, name, config_name):
    out = Path(args.output)
    run = out / name
    if (run / 'completed.json').exists():
        return
    if shutil.disk_usage(out).free < 3 * 1024 ** 3:
        raise RuntimeError('Less than 3 GiB available; stop before training')
    started = time.time()
    command = [sys.executable, '-m', 'crowd_nav.train', '--config', str(out / config_name),
               '--outdir', str(run), '--device', 'cuda', '--seed', str(args.seed),
               '--temporal-backbone', 'kda' if name == 'smoke' else name]
    if name == 'smoke':
        command.append('--skip-il-pretrain')
    # Running an interrupted output without a complete-state resume is not fair.
    if run.exists():
        raise RuntimeError(f'Incomplete run exists: {run}; inspect before retry')
    run.mkdir()
    peak_memory_mib = 0
    with open(run / 'console.log', 'w') as log:
        process = subprocess.Popen(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            memory = subprocess.check_output(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
            peak_memory_mib = max(peak_memory_mib, int(memory.strip().splitlines()[0]))
            save(run / 'running.json', {'elapsed_seconds': time.time() - started,
                                       'gpu_memory_peak_sampled_mib': peak_memory_mib,
                                       'pid': process.pid, 'stage': 'training'})
            time.sleep(10)
    if process.returncode:
        raise RuntimeError(f'{name} failed: inspect {run / "console.log"}')
    budget = 100 if name == 'smoke' else 3000
    checkpoint = run / f'rl_model_ep{budget}.pth'
    data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert data['episode'] == budget and data['algo'] == 'sarl'
    assert all(torch.isfinite(t).all() for t in data['policy_state'].values())
    optimizer_states = data['optim_value_state']['state']
    assert optimizer_states, 'No optimizer updates: episode count is not training evidence'
    assert all(torch.isfinite(v).all() for state in optimizer_states.values()
               for v in state.values() if torch.is_tensor(v))
    save(run / 'completed.json', {'wall_seconds': time.time() - started,
                                  'checkpoint_sha256': sha(checkpoint), 'command': command,
                                  'gpu_memory_peak_sampled_mib': peak_memory_mib,
                                  'final_episode': budget})


def evaluate(args, arm, protocol, budget=3000):
    from crowd_nav.train import build_env_and_robot, bind_policy, load_config
    from crowd_sim.envs.utils.state import JointState
    out = Path(args.output)
    target = out / arm / 'evaluation.json'
    if target.exists():
        return
    checkpoint = out / arm / f'rl_model_ep{budget}.pth'
    data = torch.load(checkpoint, map_location='cpu', weights_only=False)
    assert data['episode'] == budget
    cfg = load_config(str(out / f'{arm}.ini'))
    net = model(cfg, arm, args.seed).eval()
    # Parent wraps the RL policy in torch.compile; names, not tensors, change.
    weights = {name.removeprefix('_orig_mod.'): value
               for name, value in data['policy_state'].items()}
    net.load_state_dict(weights, strict=True)
    net.multiagent_training = True
    net.use_sarl_predict = True
    net.set_phase('test')
    net.set_training_mode('rl')
    net.epsilon = 0.
    rows = []
    started = time.time()
    latencies = []
    with torch.no_grad():
        for n, geometry in protocol['eval_cells']:
            cfg.set('sim', 'human_num', str(n))
            cfg.set('sim', 'test_sim', geometry)
            env, robot = build_env_and_robot(cfg)
            bind_policy(robot, net, env, epsilon=0.)
            env.phase = 'test'
            for case in protocol['eval_cases']:
                net.reset_episode_stats()
                env.reset(seed=case, options={'test_case': case})
                assert len(env.humans) == n
                result, total_return, minimum, steps = 'timeout', 0., float('inf'), 0
                positions = []
                while steps < 201:
                    state = JointState(robot.get_full_state(),
                                       [h.get_observable_state() for h in env.humans])
                    torch.cuda.synchronize()
                    tick = time.perf_counter()
                    action = net.predict(state)
                    torch.cuda.synchronize()
                    latencies.append((time.perf_counter() - tick) * 1000)
                    step = env.step(action)
                    if len(step) == 5:
                        _, reward, terminal, truncated, info = step
                        done = terminal or truncated
                    else:
                        _, reward, done, info = step
                    total_return += .99 ** steps * float(reward)
                    minimum = min(minimum, min(float(np.hypot(h.px - robot.px, h.py - robot.py)
                                                   - h.radius - robot.radius) for h in env.humans))
                    positions.append([robot.px, robot.py])
                    steps += 1
                    if done:
                        event = str(info.get('event', '')) if isinstance(info, dict) else type(info).__name__
                        event = event.lower().replace('_', '')
                        if 'success' in event or 'reachgoal' in event:
                            result = 'success'
                        elif 'collision' in event:
                            result = 'collision'
                        break
                rows.append({'case': case, 'human_num': n, 'geometry': geometry,
                             'outcome': result, 'return': total_return,
                             'minimum_clearance': minimum, 'time': steps * .25})
                save(out / arm / 'evaluation-progress.json', {'episodes': rows})
    save(target, {'checkpoint_sha256': sha(checkpoint), 'episodes': rows,
                  'wall_seconds': time.time() - started,
                  'policy_latency_ms_median': float(np.median(latencies)),
                  'policy_latency_ms_p95': float(np.percentile(latencies, 95)),
                  'latency_includes_full_80_action_scoring': True})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--seed', type=int, default=419)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--case-start', type=int, default=88000)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--evaluate-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    protocol = prepare(args)
    if args.prepare_only:
        return
    if not args.evaluate_only:
        train(args, 'smoke', 'smoke.ini')
        for arm in ARMS:
            train(args, arm, f'{arm}.ini')
    else:
        assert all((Path(args.output) / arm / 'completed.json').exists() for arm in ARMS)
    protocol['evaluation_source_commit'] = args.source_commit
    for arm in ARMS:
        evaluate(args, arm, protocol)
    protocol['formal_status'] = 'COMPLETED'
    save(Path(args.output) / 'protocol.json', protocol)


if __name__ == '__main__':
    main()
