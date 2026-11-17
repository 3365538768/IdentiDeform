"""V3 data generation pool (nohup on a800-4). Runs scripts/v3n/gen_v3.py for (split, index) pairs; a sample is done
when <root>/<split>/<idx>/result.json has exit 0. Re-reads the job file every cycle; stops launching when
docs/v4/POOL_STOP exists.
  nohup .venv/bin/python scripts/v3n/pool.py docs/v3n/jobs_a4.json --gpus 0,1,2,3,4,5,6,7 --per-gpu 5 > docs/v4/logs/pool_a4.log 2>&1 &
"""
import argparse, json, subprocess, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]


def done(root, split, i):
    p = ROOT / root / split / f'{i:06d}/result.json'
    try:
        return json.loads(p.read_text()).get('exit') == 0
    except Exception:
        return False


def main():
    a = argparse.ArgumentParser(); a.add_argument('jobs'); a.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    a.add_argument('--per-gpu', type=int, default=5); a.add_argument('--retries', type=int, default=1); args = a.parse_args()
    gpus = [int(g) for g in args.gpus.split(',')]; running, attempts = {}, {}
    (ROOT / 'docs/v4/logs').mkdir(parents=True, exist_ok=True)
    while True:
        jobs = json.loads((ROOT / args.jobs).read_text())
        for k, (p, g) in list(running.items()):
            if p.poll() is not None:
                print(time.strftime('%H:%M:%S'), 'exit', k, 0 if done(*json.loads(k)) else 1, 'gpu', g, flush=True); running.pop(k)
        stop = (ROOT / 'docs/v4/POOL_STOP').exists()
        pend = [] if stop else [j for j in jobs if json.dumps(j) not in running and not done(*j) and attempts.get(json.dumps(j), 0) <= args.retries]
        if not pend and not running:
            print('all done' if not stop else 'stopped', flush=True); break
        for j in pend:
            load = {g: sum(1 for v in running.values() if v[1] == g) for g in gpus}
            free = [g for g in gpus if load[g] < args.per_gpu]
            if not free:
                break
            g = min(free, key=lambda x: load[x]); k = json.dumps(j); attempts[k] = attempts.get(k, 0) + 1
            if attempts[k] > 1:  # a failed sample is regenerated from scratch
                subprocess.run(['rm', '-rf', str(ROOT / j[0] / j[1] / f'{j[2]:06d}')])
            cmd = (f"cd {ROOT} && source scripts/r1_env.sh && export CUDA_VISIBLE_DEVICES={g} && "
                   f".venv/bin/python scripts/v3n/gen_v3.py --split {j[1]} --indices {j[2]} --root {j[0]} --g-all -2")
            p = subprocess.Popen(['bash', '-c', cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                                 start_new_session=True)
            running[k] = (p, g); print(time.strftime('%H:%M:%S'), 'start', k, 'gpu', g, flush=True); time.sleep(1)
        time.sleep(15)


if __name__ == '__main__':
    main()
