"""Create/update only the v3j entry of docs/v3_pipeline/PLAN.json (re-read right before writing; atomic replace).
  .venv/bin/python scripts/v3j/plan_update.py --status running --eta-lo '2026-11-16 07:30' --eta-hi '2026-11-16 08:15' --detail '...'
done/total = finished v3j jobs of docs/v3j/jobs_full.json (runs/v3j_done markers).
"""
import argparse, calendar, fcntl, json, os, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
F = ROOT / 'docs/v3_pipeline/PLAN.json'


def main():
    a = argparse.ArgumentParser(); a.add_argument('--status'); a.add_argument('--detail'); a.add_argument('--eta-lo'); a.add_argument('--eta-hi')
    a.add_argument('--eta-note'); args = a.parse_args()
    jobs = json.loads((ROOT / 'docs/v3j/jobs_full.json').read_text())
    done = sum((ROOT / 'runs/v3j_done' / j['id']).exists() for j in jobs)
    ep = lambda s: calendar.timegm(time.strptime(s, '%Y-%m-%d %H:%M'))
    with open(str(F) + '.v3j.lock', 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        P = json.loads(F.read_text())
        e = next((s for s in P['stages'] if s['id'] == 'v3j'), None)
        if e is None:
            e = {'id': 'v3j', 'label': 'v3j 审稿人对照：CMA-ES 无先验基线 + 解冻 hardening/ν 的 4-D 精修（V2 val 20 样本）', 'host': 'a800-3',
                 'unit': '作业', 'deliverable': 'docs/v3j/DONE'}
            pos = next((i for i, s in enumerate(P['stages']) if s['id'] == 'v3t'), len(P['stages']))
            P['stages'].insert(pos, e)
        e.update(done=done, total=len(jobs), updated_utc=time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime()))
        for k, v in (('status', args.status), ('detail', args.detail), ('eta_note', args.eta_note)):
            if v is not None:
                e[k] = v
        if args.eta_lo:
            e['eta_lo'] = ep(args.eta_lo)
        if args.eta_hi:
            e['eta_hi'] = ep(args.eta_hi)
        tmp = F.with_suffix('.v3j.tmp'); tmp.write_text(json.dumps(P, indent=1, ensure_ascii=False)); os.replace(tmp, F)
    print(e)


if __name__ == '__main__':
    main()
