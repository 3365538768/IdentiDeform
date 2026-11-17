"""V5r: squeeze jobs (scripts/v3e/squeeze_run.py; V5b data / V5 twins / particle clamp / depth grid) split by SAMPLE over hosts.

  python3 scripts/v5r/make_jobs.py --methods gt --depths 0.06,...,0.64 [--tag x]
Every job of a sample goes to the same host (flock is not coherent across hosts): host = HOSTS[k % len] by position in
docs/v5r/SAMPLES.json. Lists: docs/v5r/jobs_a3.json, jobs_a5.json (nohup pools, scripts/v3e/pool.py --tag v5r) and the a800-2
r_queue docs/v5r/queue.json (phase v5r). Appends only; existing ids are kept.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HOSTS = ['a3', 'a5', 'a3', 'a5', 'a2']  # 2/5 a800-3, 2/5 a800-5, 1/5 a800-2


def host_of(k):
    return HOSTS[k % len(HOSTS)]


def main():
    a = argparse.ArgumentParser(); a.add_argument('--methods', required=True); a.add_argument('--depths', required=True)
    a.add_argument('--tag', default=''); a.add_argument('--indices', default=''); args = a.parse_args()
    idx = json.loads((ROOT / 'docs/v5r/SAMPLES.json').read_text())['indices']
    sel = {int(v) for v in args.indices.split(',')} if args.indices else None
    env = f'export FB_GRIP_DATA=data/v5b_main/val FB_GRIP_TWINS=v5 FB_SQUEEZE_CLAMP=1 FB_GRIP_DEPTHS={args.depths}'
    lists = {h: (ROOT / f'docs/v5r/jobs_{h}.json') for h in ('a3', 'a5')}
    L = {h: (json.loads(p.read_text()) if p.exists() else []) for h, p in lists.items()}
    qp = ROOT / 'docs/v5r/queue.json'
    q = json.loads(qp.read_text()) if qp.exists() else {'phase': 'v5r', 'max_per_gpu': 8, 'gpus': [0, 1, 2, 3, 4, 5, 6, 7], 'jobs': [],
                                                        'note': 'V5r probe-then-grasp squeeze sims on a800-2 (other samples: a800-3/5 pools)'}
    have = {j['id'] for v in L.values() for j in v} | {j['id'] for j in q['jobs']}
    n = {'a2': 0, 'a3': 0, 'a5': 0}
    for m in args.methods.split(','):
        for k, i in enumerate(idx):
            if sel is not None and i not in sel:
                continue
            jid = f'v5r_{m}{args.tag}_{i:06d}'
            if jid in have:
                continue
            cmd = f'{env} && .venv/bin/python scripts/v3e/squeeze_run.py --index {i} --method {m} --out data/v5r/runs'
            h = host_of(k); n[h] += 1
            if h == 'a2':
                q['jobs'].append({'id': jid, 'kind': 'cmd', 'group': f'v5r-{m}', 'mem_mib': 5000, 'retries': 2, 'cmd': cmd})
            else:
                L[h].append({'id': jid, 'cmd': cmd})
    for h, p in lists.items():
        p.write_text(json.dumps(L[h], indent=1))
    qp.write_text(json.dumps(q, indent=1)); print('new', n)


if __name__ == '__main__':
    main()
