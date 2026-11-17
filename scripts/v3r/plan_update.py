"""Update one stage entry (v3r / v3s only) in docs/v3_pipeline/PLAN.json: re-read, edit that entry, atomic write.
  python3 scripts/v3r/plan_update.py v3r --status running --done 0 --total 55 --detail "..."
"""
import argparse
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'docs/v3_pipeline/PLAN.json'
a = argparse.ArgumentParser(); a.add_argument("stage", choices=["v3r", "v3s", "v5r"])
a.add_argument('--status'); a.add_argument('--done', type=int); a.add_argument('--total', type=int); a.add_argument('--detail')
args = a.parse_args()
d = json.loads(P.read_text())
e = next(s for s in d['stages'] if s['id'] == args.stage)
for k in ('status', 'done', 'total', 'detail'):
    v = getattr(args, k)
    if v is not None:
        e[k] = v
e['updated_utc'] = time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())
tmp = P.with_suffix('.tmp.v3rs'); tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False)); os.replace(tmp, P)
print(json.dumps(e, ensure_ascii=False))
