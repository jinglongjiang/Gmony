"""Gate a fresh 10k RL run on merged parity and isolated 200-episode smoke."""
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
from resume_vl_parallel import summary, validate_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    previous, out = Path(args.previous), Path(args.output)
    torch.set_num_threads(2)
    parity = json.loads((out / 'merged-parity.json').read_text())
    assert parity['passed'] and all(parity['scorer'][arm]['passed'] for arm in ARMS)
    assert shutil.disk_usage(out).free >= 3 * 1024 ** 3
    old_protocol = json.loads((previous / 'protocol.json').read_text())
    shutil.copyfile(previous / 'env.config', out / 'env.config')
    status = {'stage': 'PREPARING', 'controller_pid': os.getpid(), 'arms': {}}
    save(out / 'status.json', status)
    processes, logs = {}, {}
    try:
        checkpoints = {}
        for arm in ARMS:
            ckpt = previous / arm / 'il_policy.pth'
            data = torch.load(ckpt, map_location='cpu', weights_only=False)
            assert data['meta']['epochs'] == 50 and data['meta']['seq_len'] == 24
            assert data['meta']['objective'] == 'value_regression'
            assert all(torch.isfinite(x).all() for x in data['value'].values())
            checkpoints[arm] = sha(ckpt)
        protocol = {**old_protocol, 'rl_episodes': 10000, 'formal_status': 'SMOKE_PENDING',
                    'experiment_source_commit': args.source_commit,
                    'evaluation_source_commit': args.source_commit,
                    'initial_il_checkpoint_sha256': checkpoints,
                    'il': 'Reuse existing per-arm IL50 weights; no IL optimization',
                    'rl': 'Fresh episode 1 to 10000; fresh AdamW; smoke updates excluded',
                    'eval_cases': list(range(89000, 89032)),
                    'selection': 'final episode 10000 only; no intermediate selection',
                    'merged_parity': parity,
                    'code_sha256': {str(p.relative_to(PROJECT)): sha(p)
                                    for base in ('crowd_nav', 'crowd_sim', 'tools')
                                    for p in (PROJECT / base).rglob('*.py')}}
        save(out / 'protocol.json', protocol)

        def configure(name, arm, episodes):
            run = out / name
            run.mkdir(exist_ok=False)
            shutil.copyfile(previous / arm / 'il_policy.pth', run / 'il_policy.pth')
            cfg = configparser.RawConfigParser(inline_comment_prefixes=(';', '#'), strict=False)
            cfg.read(previous / f'{arm}.ini')
            cfg.set('train', 'train_episodes', str(episodes))
            cfg.set('train', 'il_ckpt', str(run / 'il_policy.pth'))
            cfg.set('train', 'il_force_retrain', 'false')
            cfg.set('train', 'save_every', '200' if name == 'smoke' else '500')
            path = out / f'{name}.ini'
            with open(path, 'w') as stream:
                cfg.write(stream)
            return [sys.executable, '-m', 'crowd_nav.train', '--config', str(path),
                    '--outdir', str(run), '--device', 'cuda', '--seed', str(old_protocol['seed']),
                    '--temporal-backbone', arm]

        def start(name, arm, episodes):
            command = configure(name, arm, episodes)
            logs[name] = open(out / name / 'console.log', 'w')
            process = subprocess.Popen(command, cwd=PROJECT, stdout=logs[name], stderr=subprocess.STDOUT)
            processes[name] = process
            status['arms'][name] = {'pid': process.pid, 'command': command, 'state': 'running',
                                    'started_at': time.time(), 'latest_episode': 0}
            save(out / 'status.json', status)

        def monitor(names):
            while True:
                for name in names:
                    process = processes[name]
                    console = (out / name / 'console.log').read_text(errors='replace')
                    episodes = re.findall(r'\[RL-EP-(\d+)\]', console)
                    if episodes:
                        status['arms'][name]['latest_episode'] = int(episodes[-1])
                    if re.search(r'\[IL-BC\] Starting \d+ epochs', console):
                        raise RuntimeError(f'{name}: unexpected IL retraining')
                    if ('Partial load completed' in console or 'Failed to load checkpoint' in console
                            or 'Will train from scratch' in console):
                        raise RuntimeError(f'{name}: IL restore failure')
                    if process.poll() not in (None, 0):
                        raise RuntimeError(f'{name}: training exited {process.returncode}; inspect console.log')
                    if process.poll() == 0:
                        status['arms'][name]['state'] = 'completed'
                        status['arms'][name].setdefault('finished_at', time.time())
                status['disk_free_gib'] = shutil.disk_usage(out).free / 1024 ** 3
                if status['disk_free_gib'] < 2:
                    raise RuntimeError('Disk guard: less than 2 GiB free')
                save(out / 'status.json', status)
                if all(processes[name].poll() == 0 for name in names):
                    break
                time.sleep(10)

        status['stage'] = 'SMOKE_200'
        start('smoke', 'kda', 200)
        monitor(['smoke'])
        smoke = out / 'smoke' / 'rl_model_ep200.pth'
        smoke_hash = validate_checkpoint(smoke, 'kda', 200, 800)
        console = (out / 'smoke' / 'console.log').read_text(errors='replace')
        assert '[IL-BC] Loaded all parameters (strict mode)' in console or 'Loaded all parameters (strict mode)' in console
        assert '[MC-REPLAY]' in console and 'epoch=50/50' not in console
        smoke_result = {'passed': True, 'episodes': 200, 'optimizer_steps': 800,
                        'checkpoint_sha256': smoke_hash, 'il_retrained': False,
                        'wall_seconds': status['arms']['smoke']['finished_at'] - status['arms']['smoke']['started_at']}
        save(out / 'smoke' / 'completed.json', smoke_result)
        protocol['smoke'] = smoke_result
        protocol['formal_status'] = 'PARALLEL_RL_10000'
        save(out / 'protocol.json', protocol)
        status['stage'] = 'PARALLEL_RL_10000'
        for arm in ARMS:
            start(arm, arm, 10000)
        monitor(ARMS)
        for arm in ARMS:
            digest = validate_checkpoint(out / arm / 'rl_model_ep10000.pth', arm, 10000, 40000)
            save(out / arm / 'completed.json', {
                'final_episode': 10000, 'optimizer_steps': 40000, 'checkpoint_sha256': digest,
                'initial_il_checkpoint_sha256': checkpoints[arm],
                'wall_seconds': status['arms'][arm]['finished_at'] - status['arms'][arm]['started_at'],
                'command': status['arms'][arm]['command']})
        status['stage'] = 'EVALUATING_10000'
        save(out / 'status.json', status)
        eval_args = argparse.Namespace(output=str(out), seed=old_protocol['seed'])
        for arm in ARMS:
            evaluate(eval_args, arm, protocol, budget=10000)
        summary(out)
        status['stage'] = protocol['formal_status'] = 'COMPLETED'
        save(out / 'status.json', status)
        save(out / 'protocol.json', protocol)
    except Exception as error:
        status.update(stage='FAILED', error=repr(error))
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
        save(out / 'status.json', status)
        raise
    finally:
        for stream in logs.values():
            stream.close()


if __name__ == '__main__':
    main()
