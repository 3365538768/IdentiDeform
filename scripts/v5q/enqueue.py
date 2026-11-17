"""V3q: build docs/v5q/tasks.json (pre-registered schedule P1-P4 of docs/v5q/PROTOCOL.json) and docs/v5q/SAMPLES_RESOLVED.json
(failed generations replaced by the registered rule), and append r_queue tickets to docs/v5q/queue.json.

  .venv/bin/python scripts/v5q/enqueue.py [--tickets N] [--mem 6000] [--max-per-gpu 7]
Prior tasks (P2, P4) are written with "hold": true until docs/v5q/PRIOR_ENSEMBLE_{val,test}.json exist. Re-running keeps
task ids and only re-derives hold flags / order; existing tickets are kept and new ones appended.
"""
import argparse, json, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
PROTO = json.loads((ROOT / 'docs/v5q/PROTOCOL.json').read_text())
PY = '.venv/bin/python scripts/v5q/c1_refine.py'
PENDING = set()


def ok(split, i):
    try:
        return json.loads((ROOT / f'data/v5b_main/{split}/{i:06d}/result.json').read_text()).get('exit') == 0
    except (OSError, ValueError):
        return None if not (ROOT / f'data/v5b_main/{split}/{i:06d}/result.json').exists() else False


def resolve(split):
    P = PROTO['samples']; plan = P[split]['samples']; pools = P['replacement_pools'][split]
    chosen = [p['index'] for p in plan]; out = []
    for p in plan:
        i = p['index']; st = ok(split, i)
        if st is None:  # not generated yet: keep, tasks held until it exists (re-run enqueue)
            PENDING.add((split, i)); out.append(i); continue
        if not st:
            pool = pools[f"{p['law']}/{p['action']}"]
            i = next(j for j in pool if j not in chosen and j not in out and ok(split, j))
            print('replace', split, p['index'], '->', i)
        out.append(i)
    return out


def main():
    a = argparse.ArgumentParser(); a.add_argument('--tickets', type=int, default=0); a.add_argument('--mem', type=int, default=6000)
    a.add_argument('--max-per-gpu', type=int, default=7); args = a.parse_args()
    R = {s: resolve(s) for s in ('val', 'test')}
    (ROOT / 'docs/v5q/SAMPLES_RESOLVED.json').write_text(json.dumps(R, indent=1) + '\n')
    gto = {s: [p['index'] for p in PROTO['samples'][s]['samples'] if p['gt_observed']] for s in R}
    have_prior = {s: (ROOT / f'docs/v5q/PRIOR_ENSEMBLE_{s}.json').exists() for s in R}
    fo = ROOT / 'docs/v5q/tasks.json'
    old_host = {x['id']: x['host'] for x in json.loads(fo.read_text())['tasks'] if 'host' in x} if fo.exists() else {}
    T = []
    t = lambda tid, cmd, phase, **kw: T.append({'id': tid, 'cmd': cmd, 'phase': phase, **kw})
    # P1
    for s in ('val', 'test'):
        for i in gto[s]:
            if i in R[s]:
                t(f'v5q_{s}_gtobs_{i:06d}', f'{PY} --split {s} --index {i} --arm gt_observed', 'P1')
    for s in ('val', 'test'):
        for i in R[s]:
            t(f'v5q_{s}_gt24_{i:06d}', f'{PY} --split {s} --index {i} --arm lawmean_gt --budget 24', 'P1')
        for i in R[s]:
            t(f'v5q_{s}_b3_12_{i:06d}', f'{PY} --split {s} --index {i} --arm lawmean_best3 --budget 12', 'P1')
    # P2
    for s, arm, tag in (('val', 'prior_top1', 't1'), ('test', 'prior_top1', 't1'), ('val', 'prior_top2', 't2')):
        for i in R[s]:
            t(f'v5q_{s}_{tag}24_{i:06d}', f'{PY} --split {s} --index {i} --arm {arm} --budget 24', 'P2', hold=not have_prior[s])
    # P3
    for s in ('val', 'test'):
        for i in R[s]:
            t(f'v5q_{s}_b3_24_{i:06d}', f'{PY} --split {s} --index {i} --arm lawmean_best3 --budget 24', 'P3', after=[f'v5q_{s}_b3_12_{i:06d}'])
    # P4
    for i in R['test']:
        t(f'v5q_test_t224_{i:06d}', f'{PY} --split test --index {i} --arm prior_top2 --budget 24', 'P4', hold=not have_prior['test'])
    # Host partition by sample (flock is NOT coherent across a800-2 / a800-4 on this filesystem): every task of a sample
    # runs on one host, so the per-host claim lock and the cross-arm simulation cache are safe. Alternating in sample order.
    host_of = {(s, i): ('a4' if n % 2 == 0 else 'a2') for s in R for n, i in enumerate(R[s])}
    for x in T:
        sp, i = x['id'].split('_')[1], int(x['id'].rsplit('_', 1)[1])
        x['host'] = old_host.get(x['id'], host_of[(sp, i)])
        if (sp, i) in PENDING:
            x['hold'] = True
    old = {}
    f = ROOT / 'docs/v5q/tasks.json'
    if f.exists():
        old = {x['id']: x for x in json.loads(f.read_text())['tasks']}
    # Lead instruction 15:20 UTC: P2 is launched the moment the hand-off appears -> once a split's prior exists, its P2
    # tasks are placed before every pending P1/P3 task (running tasks are not affected). Timing only.
    rank = {'P2': 0, 'P1': 1, 'P3': 2, 'P4': 3}
    T.sort(key=lambda x: (0 if 'gtobs' in x['id'] else 1, rank[x['phase']] if not x.get('hold') else 9))
    for x in T:  # manual cuts (cut=true) survive re-runs
        if old.get(x['id'], {}).get('cut'):
            x['cut'] = True; x['hold'] = True
    tmp = f.with_suffix('.tmp'); tmp.write_text(json.dumps({'note': 'priority order = list order; see PROTOCOL schedule', 'tasks': T}, indent=1)); os.replace(tmp, f)
    qf = ROOT / 'docs/v5q/queue.json'
    q = json.loads(qf.read_text()) if qf.exists() else {'phase': 'v5q', 'jobs': []}
    q['max_per_gpu'] = args.max_per_gpu
    n0 = len(q['jobs'])
    for n in range(n0, n0 + args.tickets):
        q['jobs'].append({'id': f'v5q_u_{n:05d}', 'kind': 'cmd', 'group': 'v5q-c1', 'mem_mib': args.mem, 'retries': 0,
                          'cmd': '.venv/bin/python scripts/v5q/worker.py --host a2'})
    tmp = qf.with_suffix('.tmp'); tmp.write_text(json.dumps(q, indent=1)); os.replace(tmp, qf)
    print(len(T), 'tasks;', sum(not x.get('hold') for x in T), 'runnable;', len(q['jobs']), 'tickets')


if __name__ == '__main__':
    main()
