"""V3j worker pool (nohup, a800-4): adapted from scripts/v3h/pool.py with job slot weights (a CMA-ES job runs a
generation of 4 simulations in parallel -> 4 slots), dependencies ('after') and a global slot cap. A job is done
when runs/v3j_done/<id> exists (resumable). Re-reads the jobs file every cycle; stops launching when
docs/v3j/POOL_STOP exists. Jobs are launched in file order (a blocked job does not block later ones).

  nohup .venv/bin/python scripts/v3j/pool.py docs/v3j/jobs.json --gpus 0,1,2,3,4,5,6,7 --slots 56 --per-gpu 9 > docs/v3j/logs/pool.log 2>&1 &
"""
import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    a = argparse.ArgumentParser(); a.add_argument('jobs'); a.add_argument('--slots', type=int, default=56)
    a.add_argument('--per-gpu', type=int, default=9); a.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    a.add_argument('--retries', type=int, default=2)
    a.add_argument('--stop-file', default='docs/v3j/POOL_STOP'); args = a.parse_args()
    gpus = [int(g) for g in args.gpus.split(',')]; done_dir = ROOT / 'runs/v3j_done'; done_dir.mkdir(parents=True, exist_ok=True)
    logs = ROOT / 'docs/v3j/logs'; logs.mkdir(parents=True, exist_ok=True)
    running, attempts = {}, {}
    while True:
        jobs = json.loads((ROOT / args.jobs).read_text())
        for jid, (p, g, s) in list(running.items()):
            if p.poll() is not None:
                print(time.strftime('%H:%M:%S'), 'exit', jid, p.returncode, 'gpu', g, flush=True); running.pop(jid)
        stop = (ROOT / args.stop_file).exists()
        pending = [] if stop else [j for j in jobs if j['id'] not in running and not (done_dir / j['id']).exists()
                                   and attempts.get(j['id'], 0) <= args.retries]
        if not pending and not running:
            print('all done' if not stop else 'stopped', flush=True); break
        for j in pending:
            if any(not (done_dir / d).exists() for d in j.get('after', [])):
                continue
            s = j.get('slots', 1)
            if sum(v[2] for v in running.values()) + s > args.slots:
                break
            load = {g: sum(v[2] for v in running.values() if v[1] == g) for g in gpus}
            free = [g for g in gpus if load[g] + s <= args.per_gpu]
            if not free:
                continue
            g = min(free, key=lambda x: load[x])
            attempts[j['id']] = attempts.get(j['id'], 0) + 1
            cmd = f"cd {ROOT} && source scripts/r1_env.sh && export CUDA_VISIBLE_DEVICES={g} && ( {j['cmd']} ) && touch runs/v3j_done/{j['id']}"
            p = subprocess.Popen(['bash', '-c', cmd], stdout=(logs / f"{j['id']}.log").open('a'), stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
            running[j['id']] = (p, g, s); print(time.strftime('%H:%M:%S'), 'start', j['id'], 'gpu', g, 'slots', s, flush=True)
            time.sleep(2)
        time.sleep(15)


if __name__ == '__main__':
    main()
