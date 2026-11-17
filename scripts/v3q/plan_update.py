"""Update only the v3q entry of docs/v3_pipeline/PLAN.json (re-read right before writing; atomic replace).
  .venv/bin/python scripts/v3q/plan_update.py --status running --done 3 --total 360 --detail '...'
  .venv/bin/python scripts/v3q/plan_update.py --auto [--detail '...']   (done = complete searches from state.json)
"""
import argparse, json, os, fcntl
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
F = ROOT / 'docs/v3_pipeline/PLAN.json'


def count():
    tasks = json.loads((ROOT / 'docs/v3q/tasks.json').read_text())['tasks']
    planned = [t for t in tasks if 'gtobs' not in t['id'] and '_b3_12_' not in t['id'] and not t.get('cut')]
    done = sum((ROOT / 'runs/v3q_done' / t['id']).exists() for t in planned)
    return done, len(planned)


def main():
    a = argparse.ArgumentParser(); a.add_argument('--status'); a.add_argument('--done', type=int); a.add_argument('--total', type=int)
    a.add_argument('--detail'); a.add_argument('--auto', action='store_true'); args = a.parse_args()
    if args.auto:
        args.done, args.total = count()
    with open(str(F) + '.v3q.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        P = json.loads(F.read_text())
        e = next(s for s in P['stages'] if s['id'] == 'v3q')
        for k in ('status', 'done', 'total', 'detail'):
            v = getattr(args, k)
            if v is not None:
                e[k] = v
        tmp = F.with_suffix('.v3q.tmp'); tmp.write_text(json.dumps(P, indent=1, ensure_ascii=False)); os.replace(tmp, F)
    print(e)


if __name__ == '__main__':
    main()
