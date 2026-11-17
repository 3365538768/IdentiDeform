"""V3e worker pool (nohup), used instead of scripts/r_queue.py slots because the shared queue counts all
phases' running jobs per GPU and V3d already fills max_per_gpu=4; this pool adds a fixed, modest number of
workers per GPU (default 2) without changing other phases' scheduling.

  nohup .venv/bin/python scripts/v3e/pool.py docs/v3e/jobs.json --per-gpu 2 > docs/v3e/logs/pool.log 2>&1 &
jobs.json: [{"id": "v3e_...", "cmd": "..."}]; a job is done when runs/v3e_done/<id> exists (resumable).
Re-reads jobs.json every cycle, so jobs can be appended while it runs. Stops when docs/v3e/POOL_STOP exists.
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    a = argparse.ArgumentParser(); a.add_argument('jobs'); a.add_argument('--per-gpu', type=int, default=2)
    a.add_argument('--gpus', default='0,1,2,3,4,5,6,7'); a.add_argument('--retries', type=int, default=2); a.add_argument('--tag', default='v3e'); args = a.parse_args()
    gpus = [int(g) for g in args.gpus.split(',')]; done_dir = ROOT / f'runs/{args.tag}_done'; done_dir.mkdir(parents=True, exist_ok=True)
    logs = ROOT / f'docs/{args.tag}/logs'; logs.mkdir(parents=True, exist_ok=True)
    running, attempts = {}, {}
    while True:
        if (ROOT / f'docs/{args.tag}/POOL_STOP').exists():
            print('stop file', flush=True); break
        jobs = json.loads((ROOT / args.jobs).read_text())
        for jid, (p, g) in list(running.items()):
            if p.poll() is not None:
                print(time.strftime('%H:%M:%S'), 'exit', jid, p.returncode, 'gpu', g, flush=True); running.pop(jid)
        pending = [j for j in jobs if j['id'] not in running and not (done_dir / j['id']).exists()
                   and attempts.get(j['id'], 0) <= args.retries]
        if not pending and not running:
            print('all done', flush=True); break
        for j in pending:
            load = {g: sum(1 for _, gg in running.values() if gg == g) for g in gpus}
            free = [g for g in gpus if load[g] < args.per_gpu]
            if not free:
                break
            g = min(free, key=lambda x: load[x])
            attempts[j['id']] = attempts.get(j['id'], 0) + 1
            cmd = f"cd {ROOT} && source scripts/r1_env.sh && export CUDA_VISIBLE_DEVICES={g} && ( {j['cmd']} ) && touch runs/{args.tag}_done/{j['id']}"
            p = subprocess.Popen(['bash', '-c', cmd], stdout=(logs / f"{j['id']}.log").open('a'), stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
            running[j['id']] = (p, g); print(time.strftime('%H:%M:%S'), 'start', j['id'], 'gpu', g, flush=True)
        time.sleep(15)


if __name__ == '__main__':
    main()
