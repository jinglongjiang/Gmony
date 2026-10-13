"""Wait for the frozen 3k comparison, then resume all three arms to 10k."""

import argparse
import configparser
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

import torch

from run_vl_minimal import ARMS, PROJECT, evaluate, save, sha


def validate_checkpoint(path, arm, episode, steps):
    data = torch.load(path, map_location='cpu', weights_only=False)
    assert data['episode'] == episode and data['stage'] == 'rl_training_sarl'
    assert data['config']['temporal_backbone'] == arm
    assert all(torch.isfinite(value).all() for value in data['policy_state'].values())
    states = data['optim_value_state']['state']
    assert states and {int(s['step'].item()) for s in states.values()} == {steps}
    assert all(torch.isfinite(v).all() for s in states.values()
               for v in s.values() if torch.is_tensor(v))
    return sha(path)


def summary(output):
    rows = {}
    for arm in ARMS:
        result = json.loads((output / arm / 'evaluation.json').read_text())
        episodes = result['episodes']
        cells = {}
        for n in (5, 10, 20):
            for geometry in ('circle_crossing', 'square_crossing'):
                subset = [e for e in episodes if e['human_num'] == n and e['geometry'] == geometry]
                assert len(subset) == 32
                cells[f'{n}-{geometry}'] = {
                    outcome: sum(e['outcome'] == outcome for e in subset)
                    for outcome in ('success', 'collision', 'timeout')}
        assert len(episodes) == 192
        rows[arm] = {
            'episodes': len(episodes), 'cells': cells,
            'totals': {outcome: sum(e['outcome'] == outcome for e in episodes)
                       for outcome in ('success', 'collision', 'timeout')},
            'checkpoint_sha256': result['checkpoint_sha256'],
            'policy_latency_ms_median': result['policy_latency_ms_median'],
            'policy_latency_ms_p95': result['policy_latency_ms_p95'],
        }
    save(output / 'comparison.json', rows)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--previous-pid', type=int, default=0)
    args = parser.parse_args()
    previous, output = Path(args.previous), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    status = {'stage': 'WAITING_FOR_3000_EVALUATION', 'pid': os.getpid(), 'arms': {}}
    save(output / 'status.json', status)
    try:
        # Completed weights alone are insufficient: the entire primary evaluation must finish first.
        waiting_started = time.time()
        while True:
            try:
                ready = json.loads((previous / 'protocol.json').read_text())['formal_status'] == 'COMPLETED'
            except (OSError, json.JSONDecodeError):
                ready = False
            if ready and all((previous / arm / 'evaluation.json').exists() for arm in ARMS):
                break
            if args.previous_pid:
                os.kill(args.previous_pid, 0)
            if time.time() - waiting_started > 6 * 3600:
                raise RuntimeError('Previous primary evaluation did not finish within six hours')
            time.sleep(30)
        old_protocol = json.loads((previous / 'protocol.json').read_text())
        initial_hashes = {
            arm: validate_checkpoint(previous / arm / 'rl_model_ep3000.pth', arm, 3000, 12000)
            for arm in ARMS}
        for arm in ARMS:
            result = json.loads((previous / arm / 'evaluation.json').read_text())
            assert result['checkpoint_sha256'] == initial_hashes[arm]
            expected = {(n, g, case) for n, g in old_protocol['eval_cells']
                        for case in old_protocol['eval_cases']}
            assert {(e['human_num'], e['geometry'], e['case']) for e in result['episodes']} == expected
        summary(previous)
        assert shutil.disk_usage(output).free >= 3 * 1024 ** 3
        shutil.copyfile(previous / 'env.config', output / 'env.config')
        configs = {}
        for arm in ARMS:
            cfg = configparser.RawConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
            cfg.read(previous / f'{arm}.ini')
            cfg.set('train', 'train_episodes', '10000')
            with open(output / f'{arm}.ini', 'w') as stream:
                cfg.write(stream)
            configs[arm] = sha(output / f'{arm}.ini')
        protocol = {
            **old_protocol,
            'rl_episodes': 10000, 'additional_episodes': 7000,
            'experiment_source_commit': args.source_commit,
            'evaluation_source_commit': args.source_commit,
            'initial_checkpoint_sha256': initial_hashes,
            'config_sha256': configs,
            'selection': 'final episode 10000 only; same 88000 block is repeated, not fresh confirmation',
            'resume_contract': {
                'restored': ['policy', 'target_value_network', 'AdamW optimizer', 'episode', 'stats_history'],
                'not_restored': ['online replay', 'Python/NumPy/Torch RNG states', 'environment counters'],
                'replay': 'Same original teacher pool prefill; no repeat IL; online replay restarts',
                'not_equivalent_to_uninterrupted_10000': True,
                'epsilon_schedule': 'Unchanged config; cumulative episode 3001 onward',
            },
            'formal_status': 'RESUMING_PARALLEL',
            'continuation_code_sha256': {str(p.relative_to(PROJECT)): sha(p)
                                         for base in ('crowd_nav', 'crowd_sim', 'tools')
                                         for p in (PROJECT / base).rglob('*.py')},
        }
        save(output / 'protocol.json', protocol)
        processes, logs = {}, {}
        status['stage'] = 'PARALLEL_CONTINUATION'
        for arm in ARMS:
            run = output / arm
            run.mkdir()
            command = [sys.executable, '-m', 'crowd_nav.train', '--config', str(output / f'{arm}.ini'),
                       '--outdir', str(run), '--device', 'cuda', '--seed', str(old_protocol['seed']),
                       '--temporal-backbone', arm, '--resume', str(previous / arm / 'rl_model_ep3000.pth'),
                       '--resume-rebuild-il-replay']
            logs[arm] = open(run / 'console.log', 'w')
            processes[arm] = subprocess.Popen(command, cwd=PROJECT, stdout=logs[arm], stderr=subprocess.STDOUT)
            status['arms'][arm] = {'pid': processes[arm].pid, 'command': command,
                                   'started_at': time.time(), 'state': 'running', 'latest_episode': 3000}
        save(output / 'status.json', status)
        while any(p.poll() is None for p in processes.values()):
            for arm, process in processes.items():
                text = (output / arm / 'console.log').read_text(errors='replace')
                episodes = re.findall(r'\[RL-EP-(\d+)\]', text)
                if episodes:
                    status['arms'][arm]['latest_episode'] = int(episodes[-1])
                if process.poll() is not None and status['arms'][arm]['state'] == 'running':
                    status['arms'][arm]['state'] = 'completed' if process.returncode == 0 else 'failed'
                    status['arms'][arm]['exit_code'] = process.returncode
                    status['arms'][arm]['finished_at'] = time.time()
                # Never silently fall back to new IL or episode 1.
                if '[IL-BC] Starting BC training phase' in text or '[RESUME] Starting fresh training' in text:
                    raise RuntimeError(f'{arm}: unexpected new IL or failed restore')
                if process.poll() not in (None, 0):
                    raise RuntimeError(f'{arm} failed: inspect console.log')
            status['disk_free_gib'] = shutil.disk_usage(output).free / 1024 ** 3
            save(output / 'status.json', status)
            if status['disk_free_gib'] < 2:
                raise RuntimeError('Less than 2 GiB free; stop only owned continuation jobs')
            time.sleep(15)
        for arm in ARMS:
            process = processes[arm]
            assert process.returncode == 0
            status['arms'][arm].update(state='completed', latest_episode=10000, exit_code=0)
            status['arms'][arm].setdefault('finished_at', time.time())
            checkpoint_hash = validate_checkpoint(output / arm / 'rl_model_ep10000.pth', arm, 10000, 40000)
            save(output / arm / 'completed.json', {
                'final_episode': 10000, 'checkpoint_sha256': checkpoint_hash,
                'initial_checkpoint_sha256': initial_hashes[arm],
                'wall_seconds': status['arms'][arm]['finished_at'] - status['arms'][arm]['started_at'],
                'command': status['arms'][arm]['command'], 'optimizer_step': 40000,
            })
        status['stage'] = 'EVALUATING_10000'
        save(output / 'status.json', status)
        eval_args = argparse.Namespace(output=str(output), seed=old_protocol['seed'])
        for arm in ARMS:
            evaluate(eval_args, arm, protocol, budget=10000)
        summary(output)
        protocol['formal_status'] = 'COMPLETED'
        save(output / 'protocol.json', protocol)
        status['stage'] = 'COMPLETED'
        save(output / 'status.json', status)
    except Exception as error:
        for process in locals().get('processes', {}).values():
            if process.poll() is None:
                process.terminate()
        for process in locals().get('processes', {}).values():
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        status.update(stage='FAILED', error=str(error))
        save(output / 'status.json', status)
        raise
    finally:
        for log in locals().get('logs', {}).values():
            log.close()


if __name__ == '__main__':
    main()
