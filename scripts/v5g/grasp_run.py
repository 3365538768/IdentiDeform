"""V5g: simulate every candidate grasp (+ the no-jaw reference) for one (sample, twin) and measure the outcomes.

  .venv/bin/python scripts/v5g/grasp_run.py --index 2 --method gt [--split val] [--cands end:0.04,...] [--out data/v5g/runs]
Twins: scripts/v5r/twins.py (gt / prior_vision / prior_vt / lawmean_pred; density 1000 + specific moduli, hardening/softening 0.25).
Identical parameter sets are simulated once (data/v5g/runs/<split>/<idx>/cache/<hash>/, file lock; keep each sample on one host).
Outcome metrics (measure()):
  sag      = (lift - mean dz of all particles over the hold window) / height      (droop / slip / tear of the lifted object)
  left     = fraction of particles that rose less than half of the jaw lift over the hold window (pieces left behind / dropped)
  crush    = permanent jaw-segment crush: span (1-99 %) along the jaw axis of the gripped segment's particles after release
             (mean over the last 0.2 s), whole object Procrustes-aligned to the no-jaw reference run of the same parameters, minus the
             same span in the reference, over the segment thickness (> 0 = squeezed)
  dent     = P95 |aligned final - reference final| over the segment particles / thickness (strain proxy, sensitivity)
  grip     = mean jaw-axis force of the two jaws on the object over the hold window (grasp_force.json; simulation units, config density)
"""
import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/v3b'), str(ROOT / 'scripts/v2'), str(ROOT / 'scripts'), str(ROOT / 'upstream/DeformFieldBench'),
                str(ROOT / 'scripts/v5r'), str(ROOT / 'scripts/v3e')]
from gen_v2 import ASSETS, TEMPLATE, V1_RUN_PARAMETERS  # noqa: E402
from upstream_cli import build  # noqa: E402
from squeeze_sim import procrustes  # noqa: E402
import twins  # noqa: E402

GRAVITY = 2.0  # V5b: gravity 2 on every action
CANDS = os.environ.get('FB_GRASP_CANDS', 'end:0.04,end:0.10,end:0.20,end:0.30,end:0.40,centre:0.04,centre:0.10,centre:0.20,centre:0.30,centre:0.40')
QUICK_FAIL_SECONDS, QUICK_RETRIES = 60, 3
HOLD = (130, 160)  # frames of the hold window (1.3-1.6 s)


def config_for(params):
    cfg = json.loads(TEMPLATE.read_text())
    cfg.update(params, rpic_damping=0.0, grid_v_damping_scale=0.99999, substep_dt=1e-4, frame_dt=0.01, frame_num=260, g=[0.0, 0.0, -GRAVITY])
    return cfg


def simulate(dst, params, obj, pos, level, index):
    shutil.rmtree(dst, ignore_errors=True); dst.mkdir(parents=True)
    (dst / 'config.json').write_text(json.dumps(config_for(params), indent=1))
    rp = json.loads(V1_RUN_PARAMETERS.read_text())
    rp['frame_num_simulated'] = 260; rp['time_params']['frame_num'] = 260  # 2.6 s clip (grasp_main.FRAMES)
    rp['render_frame_indices'] = [f for f in rp.get('render_frame_indices', []) if f < 260]
    over = {'sim_type': 'stretch', 'render_outputs_per_sim_second': 4.0, 'compile_video': False, 'delete_png_sequences_after_compile_video': False,
            'random_render_views': True, 'random_render_views_seed': index, 'camera_distance_scale': 0.65, 'pack_sample_pack': False,
            'output_view_stress_gaussian': False, 'output_view_flow_gaussian': False, 'output_view_force_mask': False, 'output_view_object_mask': False,
            'compress_render_pngs': False, 'pack_arch4_lmdb': False, 'render_export_max_side': 0, 'render_export_scale': 1.0, 'render_img': False,
            'output_h5': True}
    cli = build(rp, ASSETS / f'{obj}.ply', dst / 'config.json', dst, over) + ['--num_views', '3', '--num_render_views', '1', '--no_compress_render_pngs']
    env = dict(os.environ); [env.pop(k, None) for k in ('FB_ACTION', 'FB_BC_FIX', 'FB_PRESS_FORCE', 'FB_WEAK')]
    env['FB_GRASP_POS'] = pos; env['FB_GRASP_LEVEL'] = repr(float(level)); env['FB_FORCES'] = '0'
    t0 = time.time()
    for attempt in range(QUICK_RETRIES + 1):
        t1 = time.time()
        with (dst / 'run.log').open('w') as log:
            code = subprocess.call([str(ROOT / 'scripts/v5g/run_grasp.sh'), *cli], stdout=log, stderr=subprocess.STDOUT, env=env)
        if code == 0 or time.time() - t1 > QUICK_FAIL_SECONDS:
            break
        time.sleep(15)
    return code == 0 and (dst / 'traj.npz').exists(), time.time() - t0


def measure(sq, ref, bc, forces):
    x0 = sq['x0'].astype(np.float64); L, F = bc['L'], bc['F']; H = bc['ext'][2]
    a0, a1 = bc['segment']; seg = (x0[:, L] >= a0) & (x0[:, L] <= a1)
    lift = bc['lift']
    dz = sq['zmean'][HOLD[0]:HOLD[1]].mean(); sag = float((lift - dz) / H); left = float(sq['left'][HOLD[0]:HOLD[1]].mean())
    xs, xr = sq['x_tail'].astype(np.float64), ref['x_tail'].astype(np.float64)
    al = procrustes(xs, xr)
    span = lambda x: float(np.subtract(*np.quantile(x[seg, F], [0.99, 0.01])))
    th = bc.get('thickness') or span(x0)
    crush = float((span(xr) - span(al)) / th)
    dent = float(np.quantile(np.linalg.norm(al[seg] - xr[seg], axis=1), 0.95) / th)
    grip = None
    if forces is not None:
        f = np.array(forces['mean_force_per_jaw_hold']); grip = float(np.abs(f[:, F]).mean())
    return {'sag': sag, 'left': left, 'crush': crush, 'dent_p95': dent, 'grip': grip, 'lift': lift, 'height': H,
            'n_particles': int(len(x0)), 'seg_particles': int(seg.sum())}


def main():
    a = argparse.ArgumentParser(); a.add_argument('--index', type=int, required=True); a.add_argument('--method', required=True)
    a.add_argument('--split', default='val'); a.add_argument('--out', default='data/v5g/runs'); a.add_argument('--cands', default=CANDS)
    args = a.parse_args()
    if args.split != 'val':
        twins.DATA = ROOT / f'data/v5b_main/{args.split}'
    gt = json.loads((twins.DATA / f'{args.index:06d}/gt.json').read_text()); obj = gt['object']
    params, meta = twins.twin_params(args.index, args.method)
    key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
    base = ROOT / args.out / args.split / f'{args.index:06d}'; cache = base / 'cache' / key; cache.mkdir(parents=True, exist_ok=True)
    mdir = base / args.method; mdir.mkdir(parents=True, exist_ok=True)
    (mdir / 'twin.json').write_text(json.dumps({'params': params, 'meta': meta, 'cache': key, 'object': obj}, indent=1))
    sp = cache / 'state.json'
    def update(fn):
        with open(cache / 'state.lock', 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            st = json.loads(sp.read_text()) if sp.exists() else {'params': params, 'object': obj, 'runs': {}}
            fn(st); tmp = sp.with_suffix(f'.tmp{os.getpid()}'); tmp.write_text(json.dumps(st, indent=1)); os.replace(tmp, sp)
            return st
    def done(c, st):
        r = st['runs'].get(c)
        return r is not None and ('metrics' in r or r.get('failed') or (c == 'none:0' and (cache / 'ref' / 'traj.npz').exists()))
    def run_one(c):
        with open(cache / f"{c.replace(':', '_')}.lock", 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)  # one writer per candidate; a second job waits and then finds it done
            if done(c, update(lambda s: None)):
                return
            pos, lev = c.split(':'); d = cache / ('ref' if c == 'none:0' else c.replace(':', '_'))
            ok, sec = simulate(d, params, obj, pos, float(lev), args.index)
            rec = {'pos': pos, 'level': float(lev), 'seconds': sec, 'failed': not ok}
            print('sim', args.index, args.method, c, ok, round(sec), flush=True)
            if c != 'none:0' and ok:
                run_one('none:0')
                ref_p = cache / 'ref' / 'traj.npz'
                if ref_p.exists():
                    bc = json.loads((d / 'grasp_bc.json').read_text()); fz = d / 'grasp_force.json'
                    rec['metrics'] = measure(dict(np.load(d / 'traj.npz')), dict(np.load(ref_p)), bc, json.loads(fz.read_text()) if fz.exists() else None)
                    print('measure', c, rec['metrics'], flush=True)
                    np.savez_compressed(d / 'subset.npz', **{k: v for k, v in np.load(d / 'traj.npz').items() if k in ('idx', 'subset', 'subset_t', 'zmean', 'left')})
                    (d / 'traj.npz').unlink()
                else:
                    rec['ref_failed'] = True
            update(lambda s: s['runs'].__setitem__(c, rec))
    allc = ['none:0'] + CANDS.split(',')
    for c in ['none:0'] + [c for c in args.cands.split(',') if c != 'none:0']:
        run_one(c)
    st = update(lambda s: s.__setitem__('complete', all(done(c, s) for c in allc)))
    if st['complete']:
        (mdir / 'state.json').write_text(json.dumps({**st, 'method': args.method, 'meta': meta, 'cache': key}, indent=1))
    print('done', args.index, args.method, args.cands, st['complete'], flush=True)


if __name__ == '__main__':
    main()
