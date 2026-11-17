"""V3q pool (nohup on a800-4): keeps up to per_gpu scripts/v3q/worker.py processes per GPU. Settings are re-read every
cycle from docs/v3q/pool_a4.json ({"gpus": [...], "per_gpu": n}); stops when docs/v3q/STOP exists and nothing runs.
  nohup .venv/bin/python scripts/v3q/pool.py > docs/v3q/logs/pool_a4.log 2>&1 &
"""
import json, subprocess, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]


def main():
    running = []
    idle_since = None
    while True:
        cfg = json.loads((ROOT / 'docs/v3q/pool_a4.json').read_text())
        running = [(p, g) for p, g in running if p.poll() is None]
        stop = (ROOT / 'docs/v3q/STOP').exists()
        if stop and not running:
            print('stopped', flush=True); break
        if not stop:
            for g in cfg['gpus']:
                while sum(1 for _, gg in running if gg == g) < cfg['per_gpu']:
                    cmd = (f"cd {ROOT} && source scripts/r1_env.sh && export CUDA_VISIBLE_DEVICES={g} && "
                           f".venv/bin/python scripts/v3q/worker.py --host a4")
                    p = subprocess.Popen(['bash', '-c', cmd], stdout=(ROOT / 'docs/v3q/logs/workers_a4.log').open('a'),
                                         stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
                    running.append((p, g)); time.sleep(3)
                    p.poll()
                    if p.returncode is not None:  # nothing to claim right now
                        break
        time.sleep(30)


if __name__ == '__main__':
    main()
