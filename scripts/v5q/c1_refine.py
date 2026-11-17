"""V5q (V5b data, data/v5b_main; gen_v3.py --g-all -2 --weak hold --ranges v5): copy of scripts/v3q/c1_refine.py with
the V5b setup (FB_WEAK=hold in the env, gravity -2 on every action, V5 BOUNDS, V5b law means). Original docstring:

V3q: the ORIGINAL C1 prior-guided few-simulation refinement (scripts/v3b/c1_refine.py, as generalised in
scripts/v3d/c1_refine.py) re-run on the V3 data (data/v5b_main) with the V3 contact simulation setup.
Pre-registration: docs/v5q/PROTOCOL.json (written by scripts/v5q/make_protocol.py before any search).

Unchanged from V2 C1: objective L = (1 - mean object-mask IoU) + RGB-MAE(union mask)/255 vs the observed 3-view
video (sample_pack.npz); deterministic opportunistic compass search over log10(E/rho) [+ log10(sigma_y/rho) unless
the law is jelly], initial step 0.5 decade halved after an unsuccessful poll, stop below 0.02; every simulation
counts (incl. start point / law probes); nested budgets; quick crash (<60 s) retried once after 75 s, a failed
candidate gets loss 9 (counted, never selected); nu fixed at the start point's nu; rho irrelevant (density 1000);
hardening/softening 0.25; arms prior_top1 / prior_top2 / lawmean_gt (privileged) / lawmean_best3.
Helpers (read_view, load_frames, load_pack, compare, candidate) are imported unchanged from scripts/v3b/c1_refine.py.

Changed for V3 (only):
  * simulate(): config, gravity, CLI and environment are built by the SAME rule as scripts/v3n/gen_v3.py for the
    observed sample (TEMPLATE + damping/dt/frames + g=[0,0,gz] with gz = -2 for drop/press/tap30hold else 0;
    FB_BC_FIX=1, FB_PRESS_MODE=gsmall, FB_G_ON_AT=1.0 (tap30hold) / 3.0, FB_DROP_GAP=0.25, FB_ACTION=tap30hold for
    that action). Only the material keys differ. Every candidate is checked against the observed sample: the
    non-material config keys must equal data/v5b_main/<split>/<idx>/config.json and the contact boundary conditions
    written by scripts/v2/contact_bc.py (contact_bc.json: bcs + press plate) must equal the observed ones; a mismatch
    aborts the job (never scored). As in V2 the extra render passes (stress/flow/force) and packing are off; they do
    not touch the MPM state or RGB/mask rasterisation (checked by the GT re-simulation noise floor, --gt-observed).
  * BOUNDS per law = V5 generator ranges (V5q): log10 E/rho jelly 3.0-5.5, plasticine 3.0-4.5, metal 4.5-6.0; sigma_y/rho plasticine 2.5-4.0, metal 3.0-4.5.
  * Law means = V5b train 0-1999 means (generated gt.json) (docs/v5q/LAW_MEANS.json, scripts/v5q/law_means.py).
  * Parameters are stored in log10 specific units (x); z-space errors are computed in the report (scripts/v5q/report.py)
    so that the z statistics (V3 prior's stats, and V2 stats for the side-by-side) can be applied afterwards.
  * Simulation cache: a candidate (law, E, sigma_y, nu, action, sample) simulated by one arm is reused by another arm
    of the same sample (data/v5q_refine/<split>/<idx>/_cache, file lock); identical config -> identical setup.
  * No unseen-action replay (pre-registered: accuracy only, as v3h).

  .venv/bin/python scripts/v5q/c1_refine.py --split val --index 12 --arm lawmean_gt --budget 24
"""
import argparse
import fcntl
import glob
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location('v3b_c1_refine', ROOT / 'scripts/v3b/c1_refine.py')
C = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(C)  # V2 helpers; also puts scripts/v2 etc. on sys.path
from gen_v2 import ASSETS, LAWS, TEMPLATE, V1_RUN_PARAMETERS  # noqa: E402
from upstream_cli import build  # noqa: E402

ACTIONS = ['press', 'drop', 'shear', 'stretch', 'bend', 'tap30hold']  # scripts/v3n/gen_v3.py
DATA = 'data/v5b_main'  # --data-root (V4: data/v4_main)
# V5 generator ranges (scripts/v3n/gen_v3.py --ranges v5), log10 specific
BOUNDS_E = {'jelly': (3.0, 5.5), 'plasticine': (3.0, 4.5), 'metal': (4.5, 6.0)}
BOUNDS_S_LAW = {'plasticine': (2.5, 4.0), 'metal': (3.0, 4.5), 'jelly': (2.5, 4.5)}  # jelly: placeholder, never searched
S_PLACEHOLDER = 2.5
STEP0 = C.STEP0
QUICK_FAIL_SECONDS, QUICK_RETRIES = C.QUICK_FAIL_SECONDS, C.QUICK_RETRIES
FAILED_LOSS = C.FAILED_LOSS
MATERIAL_KEYS = ('material', 'E', 'nu', 'density', 'yield_stress', 'hardening', 'softening')
ARMS = ('lawmean_gt', 'lawmean_best3', 'prior_top1', 'prior_top2')


def bounds(law, d):
    return BOUNDS_E[law] if d == 0 else BOUNDS_S_LAW[law]


LAW_MEANS_PATH = 'docs/v5q/LAW_MEANS.json'
WEAK, RELEASE, RANGES = 'hold', None, 'v5'  # gen_v3.py --weak hold (V5b), --release, --ranges
G_ALL = -2.0  # gen_v3.py --g-all (V4: -2.0 on every action); None = V3 rule


def gravity(action, press_mode='gsmall', g_small=-2.0, g_drop=-2.0):
    """gz exactly as scripts/v3n/gen_v3.py."""
    if G_ALL is not None:
        return G_ALL
    return g_drop if action == 'drop' else (g_small if press_mode == 'gsmall' and action in ('press', 'tap30hold') else 0.0)


def v3_env(action, press_force=None):
    """Environment exactly as scripts/v3n/gen_v3.py (FB_DROP_GAP pinned to its default 0.25)."""
    env = dict(os.environ); env.pop('FB_ACTION', None); env.pop('FB_PRESS_FORCE', None); env['FB_BC_FIX'] = '1'; env['FB_PRESS_MODE'] = 'gsmall'
    env['FB_G_ON_AT'] = '1.0' if action == 'tap30hold' else '3.0'; env['FB_DROP_GAP'] = '0.25'
    if press_force and action == 'press':
        env['FB_PRESS_FORCE'] = repr(press_force)
    env.pop('FB_WEAK', None); env.pop('FB_RELEASE', None)
    if WEAK:  # gen_v3.py --weak: set for every action
        env['FB_WEAK'] = WEAK
    if RELEASE:
        env['FB_RELEASE'] = RELEASE
    if action == 'tap30hold':
        env['FB_ACTION'] = 'tap30hold'
    return env


def strip(cfg):
    return {k: v for k, v in cfg.items() if k not in MATERIAL_KEYS}


def bc_signature(path):
    info = json.loads(Path(path).read_text())
    return json.dumps({'bcs': info['bcs'], 'press_plate': info.get('press_plate')}, sort_keys=True)


class Sample:
    def __init__(self, split, index):
        self.split, self.index = split, index
        self.dir = ROOT / DATA / split / f'{index:06d}'
        res = json.loads((self.dir / 'result.json').read_text())
        assert res.get('exit') == 0, f'observed sample {self.dir} not generated'
        self.gt = json.loads((self.dir / 'gt.json').read_text())
        self.cfg = json.loads((self.dir / 'config.json').read_text())
        self.obj, self.action = self.gt['object'], self.gt['action']
        assert self.gt.get('contact_fix') and self.gt.get('ranges') == RANGES, self.gt
        if self.action in ('shear', 'stretch', 'bend'):
            assert self.gt.get('weak_design') == WEAK, self.gt
        if self.action in ('press', 'tap30hold'):
            assert self.gt.get('release') == RELEASE, self.gt
        assert self.cfg['g'] == [0.0, 0.0, gravity(self.action)], (self.cfg['g'], self.action)
        assert self.gt['gravity'] == gravity(self.action)
        self.bc_sig = bc_signature(self.dir / 'contact_bc.json')
        self.rp = json.loads(V1_RUN_PARAMETERS.read_text())


def simulate(work, tag, params, s, keep=False):
    """One candidate simulation of the observed sample s under the V3 setup (scripts/v3n/gen_v3.py)."""
    dst = work / tag
    shutil.rmtree(dst, ignore_errors=True); dst.mkdir(parents=True)
    cfg = json.loads(TEMPLATE.read_text())
    cfg.update(params, rpic_damping=0.0, grid_v_damping_scale=0.99999, substep_dt=1e-4, frame_dt=0.01, frame_num=400,
               g=[0.0, 0.0, gravity(s.action)])
    if strip(cfg) != strip(s.cfg):
        raise SystemExit(f'non-material config differs from the observed sample: {set(strip(cfg).items()) ^ set(strip(s.cfg).items())}')
    (dst / 'config.json').write_text(json.dumps(cfg, indent=1))
    # gen_v3.py CLI; as in V2 C1 only the extra render passes (stress/flow/force) and packing are switched off.
    cli = build(s.rp, ASSETS / f'{s.obj}.ply', dst / 'config.json', dst,
                {'sim_type': 'press' if s.action == 'tap30hold' else s.action, 'render_outputs_per_sim_second': 32.0,
                 'compile_video': False, 'delete_png_sequences_after_compile_video': False, 'random_render_views': True,
                 'random_render_views_seed': s.index, 'camera_distance_scale': 0.65, 'pack_sample_pack': False,
                 'output_view_stress_gaussian': False, 'output_view_flow_gaussian': False,
                 'output_view_force_mask': False, 'output_view_object_mask': True, 'compress_render_pngs': False,
                 'pack_arch4_lmdb': False, 'render_export_max_side': 0, 'render_export_scale': 1.0})
    cli += ['--num_views', '3', '--num_render_views', '3', '--no_compress_render_pngs']
    env = v3_env(s.action, s.gt.get('press_force'))
    started = time.time()
    for attempt in range(QUICK_RETRIES + 1):
        launched = time.time()
        with (dst / 'run.log').open('w') as log:
            code = subprocess.call([str(ROOT / 'scripts/v2/run_upstream_v2.sh'), *cli], stdout=log, stderr=subprocess.STDOUT, env=env)
        if code == 0 or time.time() - launched > QUICK_FAIL_SECONDS:
            break
        print('quick failure', tag, attempt, flush=True)
        if attempt < QUICK_RETRIES:
            time.sleep(75)
    bc_file = dst / 'contact_bc.json'
    if bc_file.exists() and bc_signature(bc_file) != s.bc_sig:
        (work / f'{tag}.bc_mismatch.json').write_text(bc_file.read_text())
        raise SystemExit(f'contact boundary conditions differ from the observed sample ({tag})')
    if code != 0:
        tail = (dst / 'run.log').read_text(errors='replace')[-600:]
        (work / f'{tag}.failed.log').write_text(tail)
        shutil.rmtree(dst, ignore_errors=True)
        print('simulation failed', tag, code, flush=True)
        return None, time.time() - started, bool(bc_file.exists())
    if not bc_file.exists():
        raise SystemExit(f'contact_bc.json missing ({tag}): FB_BC_FIX path not taken')
    frames = C.load_frames(dst)
    if keep:
        np.savez_compressed(work / f'{tag}.npz', mask=np.packbits(frames[0], -1), shape=np.array(frames[0].shape),
                            rgb=frames[1].astype(np.uint8))
    shutil.rmtree(dst, ignore_errors=True)
    return frames, time.time() - started, True


def law_means():
    """V3 train means per law: {law: {'logE': mean ln1p(E/rho spec), 'logS': ..., 'nu': ...}} (docs/v5q/LAW_MEANS.json)."""
    return json.loads((ROOT / LAW_MEANS_PATH).read_text())['means']


def from_ln1p(v):
    return math.log10(max(math.expm1(v), 1e-3))


def main():
    a = argparse.ArgumentParser(); a.add_argument('--split', choices=('val', 'test'), required=True)
    a.add_argument('--index', type=int, required=True); a.add_argument('--arm', choices=ARMS + ('gt_observed',), required=True)
    a.add_argument('--budget', type=int, default=24); a.add_argument('--out', default='data/v5q_refine')
    a.add_argument('--prior', default='docs/v5q/PRIOR_ENSEMBLE_{split}.json')
    a.add_argument('--data-root', default='data/v5b_main'); a.add_argument('--g-all', type=float, default=-2.0)
    a.add_argument('--law-means', default='docs/v5q/LAW_MEANS.json'); args = a.parse_args()
    global DATA, G_ALL, LAW_MEANS_PATH
    DATA, G_ALL, LAW_MEANS_PATH = args.data_root, args.g_all, args.law_means
    s = Sample(args.split, args.index)
    root = ROOT / args.out / args.split / f'{args.index:06d}'; root.mkdir(parents=True, exist_ok=True)
    cache = root / '_cache'; cache.mkdir(exist_ok=True)
    work = root / args.arm; work.mkdir(parents=True, exist_ok=True)
    observed = C.load_pack(glob.glob(str(s.dir / '*/sample_pack.npz'))[0])
    gt_log = [math.log10(s.gt['labels']['E_over_rho_specific']),
              math.log10(s.gt['labels']['sigma_over_rho_specific']) if 'sigma_over_rho_specific' in s.gt['labels'] else None]

    if args.arm == 'gt_observed':  # noise floor: GT parameters re-simulated with the candidate code path
        gp = {k: s.cfg[k] for k in MATERIAL_KEYS if k in s.cfg}
        frames, sec, _ = simulate(work, 'gt_observed', gp, s)
        out = {'sample_id': f'v5bmain/{args.split}/{args.index:06d}', 'object': s.obj, 'observed_action': s.action,
               'gt_law': s.gt['law'], 'seconds': sec, **(C.compare(observed, frames) if frames is not None else {'failed': True}),
               'complete': True}
        (work / 'state.json').write_text(json.dumps(out, indent=1)); print('gt_observed', out, flush=True); return

    lock = open(work / 'run.lock', 'w')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print('arm already running elsewhere', flush=True); sys.exit(3)
    state_path = work / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'evals': []}
    bad = next((i for i, e in enumerate(state['evals']) if e.get('failed') and e['seconds'] < QUICK_FAIL_SECONDS), None)
    if bad is not None:
        state['evals'] = state['evals'][:bad]
    state.pop('complete', None)

    def save():
        tmp = state_path.with_suffix('.tmp'); tmp.write_text(json.dumps(state, indent=1)); os.replace(tmp, state_path)

    LM = law_means()
    clip = lambda law, x: [round(float(np.clip(x[d], *bounds(law, d))), 6) for d in range(2)]

    def start_mean(law):
        m = LM[law]
        return clip(law, [from_ln1p(m['logE']), from_ln1p(m['logS']) if m.get('logS') is not None else S_PLACEHOLDER]), m['nu']

    extra = {}
    if args.arm == 'lawmean_gt':  # privileged: GT law, V3 train mean of that law
        law = s.gt['law']; x0, nu = start_mean(law); candidates_law = [law]
    elif args.arm in ('prior_top1', 'prior_top2'):
        P = json.loads((ROOT / args.prior.format(split=args.split)).read_text())
        ens = P['samples'][str(args.index)]
        votes = ens['law_posterior_votes']; xp = ens['x_log10']  # decoded ensemble mean (log10 E/rho, log10 sigma_y/rho)
        # Tie rule as V2: equal votes -> law whose train-mean E/rho is closest to the prediction.
        order = sorted(range(3), key=lambda k: (-votes[k], abs(LM[LAWS[k]]['logE'] - math.log1p(10 ** xp[0]))))
        candidates_law = [LAWS[k] for k in order[:2 if args.arm == 'prior_top2' else 1]]
        law = candidates_law[0]
        xp = [xp[0], xp[1] if xp[1] is not None else S_PLACEHOLDER]
        x0 = clip(law, xp); nu = float(ens['nu'])
        extra = {'ensemble_law_votes': votes, 'prior_x_log10_unclipped': xp, 'prior_source': P.get('source')}
    else:  # lawmean_best3: the 3 law means are simulated first, search continues from the best
        candidates_law = list(LAWS); law = None; x0 = nu = None
    state.update(sample_id=f'v5bmain/{args.split}/{args.index:06d}', split=args.split, index=args.index, arm=args.arm,
                 object=s.obj, observed_action=s.action, gt_law=s.gt['law'], gt_x_log10=gt_log,
                 censored_sigma=bool(s.gt['labels'].get('censored_sigma')), initial_steps=[STEP0, STEP0], **extra)

    def evaluate(law_name, x, nu_):
        key = [round(v, 6) for v in x]
        for e in state['evals']:
            if e['x'] == key and e.get('law', law_name) == law_name:
                return e
        if len(state['evals']) >= args.budget:
            return None
        params = C.candidate(law_name, key[0], key[1], nu_)
        h = hashlib.sha1(json.dumps([params, s.action, args.split, args.index], sort_keys=True).encode()).hexdigest()[:16]
        with open(cache / f'{h}.lock', 'w') as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            cf = cache / f'{h}.json'
            if cf.exists():
                fit = json.loads(cf.read_text()); src = fit.pop('arm')
            else:
                frames, seconds, _ = simulate(work, f'obs_{len(state["evals"]):02d}', params, s)
                fit = C.compare(observed, frames) if frames is not None else {'iou': None, 'iou_last': None, 'rgb_mae_union': None,
                                                                                  'loss': FAILED_LOSS, 'failed': True}
                fit['seconds'] = seconds; src = args.arm
                if not (fit.get('failed') and seconds < QUICK_FAIL_SECONDS):  # infrastructure crash: not cached
                    tmp = cf.with_suffix('.tmp'); tmp.write_text(json.dumps({**fit, 'arm': src, 'params': params})); os.replace(tmp, cf)
            fit.pop('params', None)
        e = {'n': len(state['evals']) + 1, 'law': law_name, 'x': key, 'nu': nu_, 'cache': h, 'simulated_by': src, **fit}
        state['evals'].append(e); save(); print('eval', e, flush=True)
        return e

    if args.arm == 'lawmean_best3':
        probes = []
        for l in LAWS:
            xl, nul = start_mean(l)
            probes.append((evaluate(l, xl, nul), l, xl, nul))
        probes = [p for p in probes if p[0] is not None]
        _, law, x0, nu = min(probes, key=lambda p: p[0]['loss'])
    law0 = law
    if args.arm == 'prior_top2':
        # each of the top-2 laws is simulated once at the prediction (clipped to that law's bounds)
        probes = [(evaluate(l, clip(l, xp), nu), l) for l in candidates_law]
        probes = [p for p in probes if p[0] is not None]
        if probes:
            law = min(probes, key=lambda p: p[0]['loss'])[1]
        x0_law = clip(law, xp)
    else:
        x0_law = x0
    state.update(law=law, law_start=law0, x0=x0, x0_search=x0_law, nu=nu, dims=1 if law == 'jelly' else 2); save()
    dims = state['dims']

    # Deterministic compass search, verbatim from scripts/v3b/c1_refine.py (bounds per law).
    names = [0, 1]
    best = evaluate(law, x0_law, nu); step = [STEP0, STEP0]; last = None
    directions = [(d, sg) for d in range(dims) for sg in (+1, -1)]
    while best is not None and len(state['evals']) < args.budget and max(step[:dims]) >= 0.02:
        improved = False
        for d, sg in ([last] if last else []) + [v for v in directions if v != last]:
            y = list(best['x']); y[d] = float(np.clip(y[d] + sg * step[d], *bounds(law, names[d])))
            if [round(v, 6) for v in y] == best['x']:
                continue
            e = evaluate(law, y, nu)
            if e is None:
                break
            if e['loss'] < best['loss']:
                best, last, improved = e, (d, sg), True
                break
        if not improved:
            step = [v / 2 for v in step]; last = None

    # Nested budgets: best observed loss among the first k simulations (k = 0: start point, no simulation).
    budgets = {}
    for k in range(args.budget + 1):
        n = min(k, len(state['evals']))
        if n == 0:
            if args.arm == 'lawmean_best3':
                continue
            budgets[str(k)] = {'law': law0, 'x': clip(law0, x0), 'sims': 0}
            continue
        e = min(state['evals'][:n], key=lambda e: e['loss'])
        budgets[str(k)] = {'law': e['law'], 'x': e['x'], 'sims': n, 'loss': e['loss'], 'iou': e.get('iou'),
                           'wall_seconds': float(sum(v['seconds'] for v in state['evals'][:n]))}
    state['budgets'] = budgets
    state['search_exhausted'] = len(state['evals']) < args.budget
    state['complete_budget'] = args.budget; state['complete'] = True; save()
    print('done', args.split, args.index, args.arm, len(state['evals']), {k: (b['law'], b['x']) for k, b in budgets.items() if k in ('0', '4', '12', '24')}, flush=True)


if __name__ == '__main__':
    main()
