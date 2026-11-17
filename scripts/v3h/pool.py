"""V3h worker pool (nohup) on a800-4 GPUs 4-7 (the shared r_queue does not schedule there); adapted from
scripts/v3e/pool.py. A job is done when runs/v3h_done/<id> exists (resumable). Re-reads the jobs file every cycle.
Stops launching when docs/v3h/POOL_STOP exists (running jobs finish).

  nohup .venv/bin/python scripts/v3h/pool.py docs/v3h/jobs.json --gpus 4,5,6,7 --per-gpu 5 > docs/v3h/logs/pool.log 2>&1 &
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    a = argparse.ArgumentParser(); a.add_argument('jobs'); a.add_argument('--per-gpu', type=int, default=5)
    a.add_argument('--gpus', default='4,5,6,7'); a.add_argument('--retries', type=int, default=3); args = a.parse_args()
    gpus = [int(g) for g in args.gpus.split(',')]; done_dir = ROOT / 'runs/v3h_done'; done_dir.mkdir(parents=True, exist_ok=True)
    logs = ROOT / 'docs/v3h/logs'; logs.mkdir(parents=True, exist_ok=True)
    running, attempts = {}, {}
    while True:
        jobs = json.loads((ROOT / args.jobs).read_text())
        for jid, (p, g) in list(running.items()):
            if p.poll() is not None:
                print(time.strftime('%H:%M:%S'), 'exit', jid, p.returncode, 'gpu', g, flush=True); running.pop(jid)
        stop = (ROOT / 'docs/v3h/POOL_STOP').exists()
        pending = [] if stop else [j for j in jobs if j['id'] not in running and not (done_dir / j['id']).exists()
                                   and attempts.get(j['id'], 0) <= args.retries]
        if not pending and not running:
            print('all done' if not stop else 'stopped', flush=True); break
        for j in pending:
            load = {g: sum(1 for _, gg in running.values() if gg == g) for g in gpus}
            free = [g for g in gpus if load[g] < args.per_gpu]
            if not free:
                break
            g = min(free, key=lambda x: load[x])
            attempts[j['id']] = attempts.get(j['id'], 0) + 1
            cmd = f"cd {ROOT} && source scripts/r1_env.sh && export CUDA_VISIBLE_DEVICES={g} && ( {j['cmd']} ) && touch runs/v3h_done/{j['id']}"
            p = subprocess.Popen(['bash', '-c', cmd], stdout=(logs / f"{j['id']}.log").open('a'), stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
            running[j['id']] = (p, g); print(time.strftime('%H:%M:%S'), 'start', j['id'], 'gpu', g, flush=True)
            time.sleep(2)
        time.sleep(15)


if __name__ == '__main__':
    main()
