"""V3q worker: claim the first runnable task of docs/v3q/tasks.json (priority order), run it, exit.

Tasks carry a 'host' (a2 | a4): flock is not coherent across hosts on this filesystem, so a task is only claimed on its
own host (claim locks are then host-local and safe). Used by r_queue tickets on a800-2 (docs/v3q/queue.json, ids v3q_t_*) and by scripts/v3q/pool.py on a800-4, so both
hosts share one ordered task list. Claim = non-blocking flock on runs/v3q_claims/<task>.lock held for the task's
lifetime (released if the process dies). Done = runs/v3q_done/<task>. A task is retried up to 3 times.
Skips tasks whose 'after' tasks are not done and tasks marked "hold": true. Stops claiming if docs/v3q/STOP exists.
  .venv/bin/python scripts/v3q/worker.py --host a2
"""
import argparse, fcntl, json, os, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
CLAIMS, DONE, ATT = ROOT / 'runs/v3q_claims', ROOT / 'runs/v3q_done', ROOT / 'runs/v3q_attempts'
MAX_ATTEMPTS = 4


def main():
    a = argparse.ArgumentParser(); a.add_argument('--host', default='?'); args = a.parse_args()
    for d in (CLAIMS, DONE, ATT, ROOT / 'docs/v3q/logs'):
        d.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'docs/v3q/STOP').exists():
        print('STOP present'); return
    tasks = json.loads((ROOT / 'docs/v3q/tasks.json').read_text())['tasks']
    for t in tasks:
        if t.get('host') != args.host or t.get('hold') or (DONE / t['id']).exists():
            continue
        if any(not (DONE / d).exists() for d in t.get('after', [])):
            continue
        att = ATT / t['id']
        n = int(att.read_text()) if att.exists() else 0
        if n >= MAX_ATTEMPTS:
            continue
        lock = open(CLAIMS / f"{t['id']}.lock", 'w')
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            continue
        if (DONE / t['id']).exists():
            continue
        att.write_text(str(n + 1))
        gpu = os.environ.get('CUDA_VISIBLE_DEVICES', '?')
        print(time.strftime('%H:%M:%S'), 'claim', t['id'], 'host', args.host, 'gpu', gpu, 'attempt', n + 1, flush=True)
        with (ROOT / 'docs/v3q/logs' / f"{t['id']}.log").open('a') as log:
            log.write(f"\n== {time.strftime('%Y-%m-%d %H:%M:%S')} host {args.host} gpu {gpu} attempt {n + 1}\n"); log.flush()
            code = subprocess.call(['bash', '-c', f"cd {ROOT} && {t['cmd']}"], stdout=log, stderr=subprocess.STDOUT)
        if code == 0:
            (DONE / t['id']).write_text(f"{args.host} {time.time()}\n")
        print(time.strftime('%H:%M:%S'), 'exit', t['id'], code, flush=True)
        return
    print('no runnable task', flush=True)


if __name__ == '__main__':
    main()
