"""Create/update the v3w entry in docs/v3_pipeline/PLAN.json (re-read, edit own entry, atomic write).
  python3 scripts/v3w/plan_update.py --status running --done 1 --total 6 --eta_lo 1794... --eta_hi ... --detail "..."
"""
import argparse, json, os, time, fcntl
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
P = ROOT / 'docs/v3_pipeline/PLAN.json'
a = argparse.ArgumentParser()
for k in ('status', 'detail', 'unit', 'label', 'host', 'deliverable'): a.add_argument('--' + k)
for k in ('done', 'total', 'eta_lo', 'eta_hi'): a.add_argument('--' + k, type=int)
args = a.parse_args()
with open(P.parent / 'PLAN.json.v3w.lock', 'w') as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    d = json.loads(P.read_text())
    e = next((s for s in d['stages'] if s['id'] == 'v3w'), None)
    if e is None:
        e = {'id': 'v3w', 'label': '公开真实视频验证（数据集调研 + Physics 101 跌落零样本刚度排序）',
             'host': 'a800-2 (GPU 7) + 4090-1 下载', 'status': 'pending', 'unit': '步', 'total': 6, 'done': 0,
             'deliverable': 'docs/v3w/RESULTS.json', 'eta_lo': 0, 'eta_hi': 0, 'eta_note': ''}
        d['stages'].append(e)
    for k in ('status', 'detail', 'unit', 'label', 'host', 'deliverable', 'done', 'total', 'eta_lo', 'eta_hi'):
        v = getattr(args, k)
        if v is not None:
            e[k] = v
    e['updated_utc'] = time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())
    tmp = P.with_suffix('.tmp.v3w'); tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False)); os.replace(tmp, P)
print(json.dumps(e, ensure_ascii=False))
