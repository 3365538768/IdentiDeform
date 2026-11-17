"""V3e: run the squeeze at all candidate depths (plus the d=0 reference) for one (sample, method) twin.

  .venv/bin/python scripts/v3e/squeeze_run.py --index 134 --method gt [--depths 0,0.04,...] [--keep]
State: data/v3e/runs/<index>/<method>/state.json (resumable; identical parameter sets of different methods
are simulated once and shared via data/v3e/runs/<index>/cache/<hash>/). Full particle dumps are deleted after
measuring unless --keep; a 4096-particle subset trajectory is kept for figures.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import squeeze_sim as S  # noqa: E402

ROOT = S.ROOT


def main():
    a = argparse.ArgumentParser(); a.add_argument('--index', type=int, required=True); a.add_argument('--method', required=True)
    a.add_argument('--depths', default=','.join(str(d) for d in [0.0] + S.DEPTHS)); a.add_argument('--keep', action='store_true')
    a.add_argument('--out', default='data/v3e/runs'); args = a.parse_args()
    gt = json.loads((ROOT / f'data/v2_main/val/{args.index:06d}/gt.json').read_text()); obj = gt['object']
    params, meta = S.twin_params(args.index, args.method)
    key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
    base = ROOT / args.out / f'{args.index:06d}'; cache = base / 'cache' / key; cache.mkdir(parents=True, exist_ok=True)
    mdir = base / args.method; mdir.mkdir(parents=True, exist_ok=True)
    (mdir / 'twin.json').write_text(json.dumps({'params': params, 'meta': meta, 'cache': key, 'object': obj}, indent=1))
    import fcntl
    lock = open(cache / 'lock', 'w'); fcntl.flock(lock, fcntl.LOCK_EX)  # one writer per parameter set
    sp = cache / 'state.json'
    state = json.loads(sp.read_text()) if sp.exists() else {'params': params, 'object': obj, 'runs': {}}
    def save():
        tmp = sp.with_suffix('.tmp'); tmp.write_text(json.dumps(state, indent=1)); os.replace(tmp, sp)
    depths = [float(v) for v in args.depths.split(',')]
    if 0.0 not in depths:
        depths = [0.0] + depths
    state.pop('ref_failed', None); state.pop('complete', None)
    def settled(k):
        x = state['runs'].get(k)
        return x is not None and ('metrics' in x or (x.get('failed') and x.get('retried')))
    need_ref = not all(settled(f'{d:.3f}') for d in depths if d != 0.0)
    for d in depths:
        k = f'{d:.3f}'; r = state['runs'].get(k)
        if r is not None and not r.get('failed') and ((cache / k / 'traj.npz').exists() or (d != 0.0 and 'metrics' in r)):
            continue
        if r is not None and r.get('failed') and r.get('retried'):
            continue  # failed again after the retry pass: recorded as a failed (unstable) simulation
        retried = bool(r is not None and r.get('failed'))
        if d == 0.0 and r is not None and not r.get('failed') and not need_ref:
            continue  # reference dump cleaned up, but every depth is already measured
        ok, sec = S.simulate(cache / k, params, obj, d, args.index)
        state['runs'][k] = {'d': d, 'seconds': sec, 'failed': not ok, 'retried': retried}; save()
        print('sim', args.index, args.method, d, ok, round(sec), flush=True)
    ref_path = cache / '0.000' / 'traj.npz'
    if state['runs'].get('0.000', {}).get('failed') or not ref_path.exists():
        state['ref_failed'] = True; save()
    else:
        ref = dict(np.load(ref_path))
        for d in depths:
            k = f'{d:.3f}'; r = state['runs'][k]
            if d == 0.0 or r.get('failed') or 'metrics' in r:
                continue
            bc = json.loads((cache / k / 'squeeze_bc.json').read_text())
            r['metrics'] = S.measure(dict(np.load(cache / k / 'traj.npz')), ref, bc); save()
            print('measure', args.index, args.method, d, r['metrics'], flush=True)
    # cleanup: keep subset trajectories (+bc) only
    for k in list(state['runs']):
        dd = cache / k
        if not dd.exists():
            continue
        if (dd / 'traj.npz').exists() and not (dd / 'subset.npz').exists():
            t = np.load(dd / 'traj.npz')
            np.savez_compressed(dd / 'subset.npz', **{n: t[n] for n in ('idx', 'subset', 'subset_t', 'width', 'zmax')},
                                x0=t['x0'][t['idx']], x_tail=t['x_tail'][t['idx']])
        if not args.keep and all('metrics' in state['runs'][j] or state['runs'][j].get('failed') or j == '0.000'
                                 for j in state['runs']) and not state.get('ref_failed'):
            for f in dd.iterdir():
                if f.name not in ('subset.npz', 'squeeze_bc.json', 'run.log', 'config.json'):
                    shutil.rmtree(f) if f.is_dir() else f.unlink()
    state['complete'] = True; save()
    (mdir / 'state.json').write_text(json.dumps({**state, 'method': args.method, 'meta': meta, 'cache': key}, indent=1))
    print('done', args.index, args.method, flush=True)


if __name__ == '__main__':
    main()
