"""V3r: squeeze jobs for the V3 gripper task (scripts/v3e/squeeze_run.py, V3 data / V3 twins / particle clamp).

  python3 scripts/v3r/make_jobs.py --methods gt [--a4-share 0.67]
Appends (never removes) jobs: a800-4 nohup pool list docs/v3r/jobs_a4.json (scripts/v3e/pool.py --tag v3r) and the
a800-2 shared r_queue docs/v3r/queue.json. Samples are split disjointly between the hosts (by position in
docs/v3r/SAMPLES.json indices) so no job can run twice; a job already present in either list keeps its host.
"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV = 'export FB_GRIP_DATA=data/v3_main/val FB_GRIP_TWINS=v3 FB_SQUEEZE_CLAMP=1'


def cmd(i, m):
    return f'{ENV} && .venv/bin/python scripts/v3e/squeeze_run.py --index {i} --method {m} --out data/v3r/runs'


def main():
    a = argparse.ArgumentParser(); a.add_argument('--methods', default='gt'); a.add_argument('--a4-share', type=float, default=0.67)
    a.add_argument('--max-per-gpu', type=int, default=6); a.add_argument('--indices', default='')
    args = a.parse_args()
    idx = json.loads((ROOT / 'docs/v3r/SAMPLES.json').read_text())['indices']
    if args.indices:
        keep = {int(v) for v in args.indices.split(',')}; idx = [i for i in idx if i in keep]
    a4p = ROOT / 'docs/v3r/jobs_a4.json'; qp = ROOT / 'docs/v3r/queue.json'
    a4 = json.loads(a4p.read_text()) if a4p.exists() else []
    q = json.loads(qp.read_text()) if qp.exists() else {'phase': 'v3r', 'gpus': [0, 1, 2, 3, 4, 5, 6, 7], 'jobs': [],
                                                        'note': 'V3r gripper task squeeze sims on a800-2 (a800-4 share: docs/v3r/jobs_a4.json)'}
    q['max_per_gpu'] = args.max_per_gpu
    have = {j['id'] for j in a4} | {j['id'] for j in q['jobs']}
    n_new = [0, 0]
    for m in args.methods.split(','):
        for k, i in enumerate(idx):
            jid = f'v3r_{m}_{i:06d}'
            if jid in have:
                continue
            if (k % 3) < round(3 * args.a4_share):
                a4.append({'id': jid, 'cmd': cmd(i, m)}); n_new[0] += 1
            else:
                q['jobs'].append({'id': jid, 'kind': 'cmd', 'group': f'v3r-{m}', 'mem_mib': 6000, 'retries': 2, 'cmd': cmd(i, m)}); n_new[1] += 1
    a4p.write_text(json.dumps(a4, indent=1)); qp.write_text(json.dumps(q, indent=1))
    print('new jobs a800-4', n_new[0], 'a800-2', n_new[1], 'totals', len(a4), len(q['jobs']))


if __name__ == '__main__':
    main()
