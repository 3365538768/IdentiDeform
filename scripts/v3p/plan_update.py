"""Update only the v3p entry of docs/v3_pipeline/PLAN.json (re-read right before writing).
  .venv/bin/python scripts/v3p/plan_update.py --status running --done 1 --detail "..."
"""
import argparse, json, os, fcntl
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'docs/v3_pipeline/PLAN.json'
a = argparse.ArgumentParser(); a.add_argument('--status'); a.add_argument('--done', type=int); a.add_argument('--total', type=int)
a.add_argument("--detail"); a.add_argument("--id", default="v3p"); args = a.parse_args()
with open(ROOT / 'docs/v3_pipeline/.plan.lock', 'w') as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    plan = json.loads(P.read_text())
    e = next(s for s in plan['stages'] if s["id"] == args.id)
    for k in ('status', 'done', 'total', 'detail'):
        if getattr(args, k) is not None:
            e[k] = getattr(args, k)
    tmp = P.with_suffix('.json.v3p.tmp'); tmp.write_text(json.dumps(plan, indent=1, ensure_ascii=False)); os.replace(tmp, P)
print(json.dumps(e, ensure_ascii=False))
