"""V3p job pool for a800-4 (nohup allowed there; pattern of scripts/v3n/pool.py).
Job file: JSON list of {"id", "cmd", "after": [ids], "gpu": bool}. A job is done when runs/v3p_done/<id> exists
(written only on exit 0). Re-reads the job file every cycle; stop launching with docs/v3p/POOL_STOP.
  nohup .venv/bin/python scripts/v3p/pool.py docs/v3p/jobs_a4.json --gpus 0,1,2,3,4,5,6,7 --per-gpu 2 --max-total 9 > docs/v3p/logs/pool_a4.log 2>&1 &
"""
import argparse, json, subprocess, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
DONE = ROOT / 'runs/v3p_done'


def busy(pattern):
    """True if some other process (e.g. a run launched by an earlier pool) matches the job's 'busy' regex."""
    return bool(pattern) and subprocess.run(['pgrep', '-f', pattern], capture_output=True).returncode == 0


def main():
    a = argparse.ArgumentParser(); a.add_argument('jobs'); a.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    a.add_argument('--per-gpu', type=int, default=2); a.add_argument('--max-total', type=int, default=9)
    a.add_argument('--retries', type=int, default=1); a.add_argument('--done-dir', default='runs/v3p_done'); args = a.parse_args()
    global DONE; DONE = ROOT / args.done_dir
    gpus = [int(g) for g in args.gpus.split(',')]; running, attempts = {}, {}
    DONE.mkdir(parents=True, exist_ok=True); (ROOT / 'docs/v3p/logs').mkdir(parents=True, exist_ok=True)
    while True:
        jobs = json.loads((ROOT / args.jobs).read_text())
        for k, (p, g) in list(running.items()):
            if p.poll() is not None:
                print(time.strftime('%H:%M:%S'), 'exit', k, p.returncode, 'gpu', g, flush=True); running.pop(k)
        stop = (ROOT / 'docs/v3p/POOL_STOP').exists()
        isdone = lambda i: (DONE / i).exists()
        pend = [] if stop else [j for j in jobs if j['id'] not in running and not isdone(j['id'])
                                and attempts.get(j['id'], 0) <= args.retries]
        ready = [j for j in pend if all(isdone(d) for d in j.get('after', [])) and not busy(j.get('busy'))]
        if not pend and not running:
            print('all done' if not stop else 'stopped', flush=True); break
        for j in ready:
            if len(running) >= args.max_total:
                break
            g = ''
            if j.get('gpu', True):
                load = {g: sum(1 for v in running.values() if v[1] == g) for g in gpus}
                free = [g for g in gpus if load[g] < args.per_gpu]
                if not free:
                    break
                g = min(free, key=lambda x: load[x])
            attempts[j['id']] = attempts.get(j['id'], 0) + 1
            cmd = (f"cd {ROOT} && source scripts/r1_env.sh && export CUDA_VISIBLE_DEVICES={g} && "
                   f"( {j['cmd']} ) && touch {DONE}/{j['id']}")
            log = open(ROOT / f"docs/v3p/logs/{j['id']}.log", 'a')
            p = subprocess.Popen(['bash', '-c', cmd], stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
            running[j['id']] = (p, g); print(time.strftime('%H:%M:%S'), 'start', j['id'], 'gpu', g, flush=True); time.sleep(1)
        time.sleep(15)


if __name__ == '__main__':
    main()
