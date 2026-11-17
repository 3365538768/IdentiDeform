"""V3b C1: simulation-in-the-loop test-time refinement (validation split only).

For one V2-main validation sample:
  1. Start from a trained model's prediction (best validation epoch of --run): MAP law, decoded specific
     E/rho, nu, sigma_y/rho (rho is irrelevant to MPM motion: density fixed to 1000, as replay_eval.py).
  2. Re-simulate the OBSERVED action (same object, same cameras/seed, same physics as gen_v2_main.py) and
     minimise  L = (1 - mean object-mask IoU) + RGB-MAE(union mask)/255  against the observed video,
     over log10(E/rho) and, for metal/plasticine, log10(sigma_y/rho). Optimiser: deterministic compass
     (pattern) search, step 0.5 decade halved after an unsuccessful poll, opportunistic acceptance.
     Every simulation counts towards the budget, including the start point; budgets 0/4/12 are the
     best-so-far after 0/4/12 simulations of the same search (nested).
  3. Unseen-action replay: simulate the swapped action (ACTIONS[(i+3)%6]) with GT parameters and with
     the budget-0/4/12 estimates; report object-mask IoU / RGB MAE vs the GT replay.
hardening/softening are not predicted: 0.25 for every candidate (GT replay uses the true values).
Simulation directories are deleted once their metrics are stored (state.json, resumable).

  .venv/bin/python scripts/v3b/c1_refine.py --run v2main_C_seed29 --index 12 [--budget 12] [--gt-observed]
"""
import argparse
import glob
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
sys.path[:0] = [str(ROOT / 'scripts/v2'), str(ROOT / 'scripts'), str(ROOT / 'upstream/DeformFieldBench')]
from gen_v2 import ASSETS, LAWS, TEMPLATE, V1_RUN_PARAMETERS  # noqa: E402
from gen_v2_main import ACTIONS  # noqa: E402
from upstream_cli import build  # noqa: E402
from r1_report import best_dir  # noqa: E402
from my_utils.pack_tensors import _sorted_pngs  # noqa: E402
import cv2  # noqa: E402

STATS = json.loads((ROOT / 'data/manifests/v2main.specific.stats.json').read_text())
BOUNDS = {'E': (1.5, 6.5), 'sigma_y': (-0.5, 5.0)}  # log10 specific; generator ranges are E 2-6, sigma 0-4.5
STEP0 = 0.5
QUICK_FAIL_SECONDS, QUICK_RETRIES = 60, 1
FAILED_LOSS = 9.0  # failed (unstable) simulation: worse than any real candidate (loss <= 1 + 1)


def decode(z):
    out = []
    for j, v in enumerate(z):
        x = v * STATS['std'][j] + STATS['mean'][j]
        out.append(x if j == 1 else math.expm1(min(max(x, 0.), 40.)))
    return out  # E/rho, nu, rho, sigma_y/rho (specific units)


def encode(j, value):
    return (math.log1p(value) - STATS['mean'][j]) / STATS['std'][j]


def candidate(law, log_e, log_s, nu):
    p = {'material': law, 'E': 10 ** log_e, 'nu': float(np.clip(nu, 0.05, 0.48)), 'density': 1000.0}
    if law != 'jelly':
        p.update(yield_stress=10 ** log_s, hardening=0.25)
        if law == 'plasticine':
            p['softening'] = 0.25
    return p


def read_view(paths, export=512, size=224):
    """Bit-identical in-memory version of the dataset path: upstream downscale_png_directory (native -> 512,
    INTER_AREA, lossless PNG) followed by sample_pack's _read_view_pngs_to_thwc_uint8 (-> 224, INTER_AREA).
    Skipping the on-disk 512 rewrite avoids ~770 extra PNG writes per simulation on the shared filesystem."""
    frames = []
    for p in paths:
        raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise RuntimeError(f'cannot read {p}')
        h, w = raw.shape[:2]
        if max(h, w) > export:
            s = export / max(h, w)
            tw, th = max(2, int(round(w * s)) // 2 * 2), max(2, int(round(h * s)) // 2 * 2)
            raw = cv2.resize(raw, (tw, th), interpolation=cv2.INTER_AREA)
        bgr = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR) if raw.ndim == 2 or raw.shape[2] == 1 else raw[:, :, :3]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.shape[:2] != (size, size):
            rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
        frames.append(rgb)
    return np.stack(frames)  # T,H,W,3 uint8


def load_frames(sample_dir):
    """images/ + object_mask/ PNGs -> (mask[V,T,H,W] bool, rgb[V,T,H,W,3] float32), as the sample_pack."""
    roots = list(sample_dir.glob('*/images'))
    if len(roots) != 1:
        raise RuntimeError(f'missing renders in {sample_dir}')
    base = roots[0].parent; masks, rgbs = [], []
    for view in sorted(p.name for p in roots[0].iterdir() if p.is_dir()):
        rgbs.append(read_view(_sorted_pngs(base / 'images' / view)))
        masks.append(read_view(_sorted_pngs(base / 'object_mask' / view))[..., 0] > 127)
    return np.stack(masks), np.stack(rgbs).astype(np.float32)


def load_pack(path):
    n = np.load(path)
    return n['object_mask'][:, :, 0] > 127, n['rgb'].transpose(0, 1, 3, 4, 2).astype(np.float32)


def compare(ref, sim):
    (ma, ra), (mb, rb) = ref, sim
    t = min(ma.shape[1], mb.shape[1]); ma, ra, mb, rb = ma[:, :t], ra[:, :t], mb[:, :t], rb[:, :t]
    inter = (ma & mb).sum((-1, -2)); union = (ma | mb).sum((-1, -2))
    iou = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    u = ma | mb
    rgb = float((np.abs(ra - rb).mean(-1) * u).sum() / max(u.sum(), 1))
    return {'iou': float(iou.mean()), 'iou_last': float(iou[:, -1].mean()), 'rgb_mae_union': rgb,
            'loss': float(1 - iou.mean() + rgb / 255.)}


def simulate(work, tag, params, action, obj, index, rp, keep=False):
    dst = work / tag
    shutil.rmtree(dst, ignore_errors=True); dst.mkdir(parents=True)
    cfg = json.loads(TEMPLATE.read_text())
    cfg.update(params, rpic_damping=0.0, grid_v_damping_scale=0.99999, substep_dt=1e-4, frame_dt=0.01, frame_num=400,
               g=[0.0, 0.0, -9.8])
    (dst / 'config.json').write_text(json.dumps(cfg, indent=1))
    # Same CLI as gen_v2_main.py; only extra render passes (stress/flow/force) and packing are switched off,
    # which do not touch the MPM state or the RGB/object-mask rasterisation.
    cli = build(rp, ASSETS / f'{obj}.ply', dst / 'config.json', dst,
                {'sim_type': 'press' if action == 'tap30hold' else action, 'render_outputs_per_sim_second': 32.0,
                 'compile_video': False, 'delete_png_sequences_after_compile_video': False, 'random_render_views': True,
                 'random_render_views_seed': index, 'camera_distance_scale': 0.65, 'pack_sample_pack': False,
                 'output_view_stress_gaussian': False, 'output_view_flow_gaussian': False,
                 'output_view_force_mask': False, 'output_view_object_mask': True, 'compress_render_pngs': False,
                 'pack_arch4_lmdb': False, 'render_export_max_side': 0, 'render_export_scale': 1.0})
    cli += ['--num_views', '3', '--num_render_views', '3', '--no_compress_render_pngs']
    env = dict(os.environ); env.pop('FB_ACTION', None)
    if action == 'tap30hold':
        env['FB_ACTION'] = 'tap30hold'
    started = time.time()
    for attempt in range(QUICK_RETRIES + 1):
        launched = time.time()
        with (dst / 'run.log').open('w') as log:
            code = subprocess.call([str(ROOT / 'scripts/v2/run_upstream_v2.sh'), *cli], stdout=log, stderr=subprocess.STDOUT, env=env)
        if code == 0 or time.time() - launched > QUICK_FAIL_SECONDS:
            break
        # A crash within seconds (CUDA error 700 before the first frames) may be transient: retry once.
        # A repeated crash is deterministic for the parameters (explosion within the first frame) and is
        # recorded as a failed candidate (its recorded time includes the wait, so it is not re-run).
        print('quick failure', tag, attempt, flush=True)
        if attempt < QUICK_RETRIES:
            time.sleep(75)
    if code != 0:
        # Mid-simulation MPM blow-ups are recorded as failed candidates.
        tail = (dst / 'run.log').read_text(errors='replace')[-600:]
        (work / f'{tag}.failed.log').write_text(tail)
        shutil.rmtree(dst, ignore_errors=True)
        print('simulation failed', tag, code, flush=True)
        return None, time.time() - started
    frames = load_frames(dst)
    if keep:
        np.savez_compressed(work / f'{tag}.npz', mask=np.packbits(frames[0], -1), shape=np.array(frames[0].shape),
                            rgb=frames[1].astype(np.uint8))
    shutil.rmtree(dst, ignore_errors=True)
    return frames, time.time() - started


def load_kept(work, tag):
    n = np.load(work / f'{tag}.npz'); shape = tuple(n['shape'])
    return np.unpackbits(n['mask'], -1)[..., :shape[-1]].astype(bool), n['rgb'].astype(np.float32)


ARMS = ('prior', 'prior_sigma', 'lawmean_gt', 'lawmean_best3', 'prior_top2')


def law_means():
    """Train-split means per GT law in z space (E/rho, nu, rho, sigma_y/rho), as replay_eval.law_means."""
    sums = {}
    for r in map(json.loads, (ROOT / 'data/manifests/v2main.jsonl').read_text().splitlines()):
        if r['split'] != 'train':
            continue
        p = r['parameters']; scale = p['rho'] / 1000.
        vals = [p['E'] / scale, p['nu'], p['rho'], None if p.get('sigma_y') is None else p['sigma_y'] / scale]
        acc = sums.setdefault(r['law_id'], [np.zeros(4), np.zeros(4)])
        for j, v in enumerate(vals):
            if v is None:
                continue
            acc[0][j] += ((v if j == 1 else math.log1p(v)) - STATS['mean'][j]) / STATS['std'][j]
            acc[1][j] += 1
    means = {LAWS[k]: (v[0] / np.maximum(v[1], 1)).tolist() for k, v in sums.items()}
    # Jelly has no sigma_y: report the all-law train mean of the observed values (z = 0).
    return means


def gt_replay(root, replay_action, obj, index, rp, gt_params):
    """Shared GT replay of the unseen action, one per sample, created under a file lock."""
    import fcntl
    tag = 'replay_gt_shared'
    with open(root / f'{tag}.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (root / f'{tag}.failed').exists():
            return None
        if not (root / f'{tag}.npz').exists():
            frames, _ = simulate(root, tag, gt_params, replay_action, obj, index, rp, keep=True)
            if frames is None:
                (root / f'{tag}.failed').write_text('failed'); return None
            return frames
    return load_kept(root, tag)


def main():
    a = argparse.ArgumentParser(); a.add_argument('--run', required=True); a.add_argument('--index', type=int, required=True)
    a.add_argument('--budget', type=int, default=12); a.add_argument('--gt-observed', action='store_true')
    a.add_argument('--arm', choices=ARMS, default='prior')
    a.add_argument('--replay-budgets', default='0,4,12,24')
    a.add_argument('--report-budgets', default='0,2,3,4,8,12,16,24,32,48')
    a.add_argument('--out', default='data/v3b_refine'); args = a.parse_args()
    sid = f'v2main/val/{args.index:06d}'
    pred_dir, _ = best_dir(ROOT / f'runs/{args.run}')
    pr = next(r for r in map(json.loads, (pred_dir / 'predictions.jsonl').read_text().splitlines()) if r['sample_id'] == sid)
    gt = json.loads((ROOT / f'data/v2_main/val/{args.index:06d}/gt.json').read_text())
    gcfg = json.loads((ROOT / f'data/v2_main/val/{args.index:06d}/config.json').read_text())
    obj, action = gt['object'], gt['action']
    replay_action = ACTIONS[(ACTIONS.index(action) + 3) % len(ACTIONS)]
    rp = json.loads(V1_RUN_PARAMETERS.read_text())
    root = ROOT / args.out / args.run / f'{args.index:06d}'; root.mkdir(parents=True, exist_ok=True)
    work = root if args.arm == 'prior' else root / args.arm  # 'prior' keeps the original layout
    work.mkdir(parents=True, exist_ok=True)
    state_path = work / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'evals': [], 'replay': {}}
    old_budgets = state.get('budgets', {})
    # Repair states written before quick (infrastructure) failures were retried: drop the first such
    # evaluation and everything after it (later search steps depended on it), and any replay of a budget.
    bad = next((i for i, e in enumerate(state['evals']) if e.get('failed') and e['seconds'] < QUICK_FAIL_SECONDS), None)
    if bad is not None:
        state['evals'] = state['evals'][:bad]
        state['replay'] = {k: v for k, v in state['replay'].items() if k.startswith('gt')}
    quick = lambda v: isinstance(v, dict) and v.get('failed') and v.get('seconds', 1e9) < QUICK_FAIL_SECONDS
    for k in [k for k, v in state['replay'].items() if quick(v)]:
        state['replay'].pop(k)
    if quick(state.get('gt_observed')):
        state.pop('gt_observed')
    state.pop('complete', None)
    def save():
        tmp = state_path.with_suffix('.tmp'); tmp.write_text(json.dumps(state, indent=1)); os.replace(tmp, state_path)

    observed = load_pack(glob.glob(str(ROOT / f'data/v2_main/val/{args.index:06d}/*/sample_pack.npz'))[0])
    names = ['E', 'sigma_y']
    means = law_means()
    def start(law_name, z):
        e, nu_, _, sg = decode(z)
        x = [math.log10(max(e, 1e-3)), math.log10(max(sg, 1e-3))]
        return [round(float(np.clip(x[d], *BOUNDS[names[d]])), 6) for d in range(2)], nu_
    steps = [STEP0, STEP0]
    if args.arm in ('prior', 'prior_sigma'):
        law = LAWS[pr['predicted_law_id']]
        x0, nu = start(law, pr['prediction_z'])
        base_z = list(pr['prediction_z'])
        if args.arm == 'prior_sigma':
            # Initial compass step per dimension = the model's own predictive std (NLL-trained heteroscedastic
            # head, ~95 % empirical coverage on val), converted from z to log10 units; clipped to [0.1, 1.5].
            sz = [math.exp(.5 * pr['logvar'][j]) * STATS['std'][j] / math.log(10) for j in (0, 3)]
            steps = [float(np.clip(v, .1, 1.5)) for v in sz]
        candidates_law = [law]
    elif args.arm == 'lawmean_gt':  # privileged: GT law, train mean of that law
        law = gt['law']; x0, nu = start(law, means[law]); base_z = list(means[law]); candidates_law = [law]
    elif args.arm == 'prior_top2':
        # Fair counterpart of lawmean_best3: the 9-run B/C/D ensemble's continuous prediction is simulated under
        # the top-2 laws of its law posterior (vote fractions over the 9 runs; if unanimous, the second law is the
        # remaining law whose train-mean E/rho is closest to the prediction). Budget 0 = top-1 law, no sims
        # (the 'prior_ens' point).
        ens = json.loads((ROOT / 'docs/v3b/ENSEMBLE_BCD9_VAL.json').read_text())['samples'][sid]
        votes = ens['law_posterior_votes']; base_z = list(ens['prediction_z'])
        order = sorted(range(3), key=lambda k: (-votes[k], abs(means[LAWS[k]][0] - base_z[0])))
        candidates_law = [LAWS[k] for k in order[:2]]
        law = candidates_law[0]; x0, nu = start(law, base_z)
        state.update(ensemble_law_votes=votes)
    else:  # lawmean_best3: simulate each law's train mean first (3 sims), continue from the best law
        candidates_law = list(LAWS); law = None; x0 = nu = base_z = None
    state.update(sample_id=sid, run=args.run, arm=args.arm, prediction_dir=str(pred_dir), object=obj, observed_action=action,
                 replay_action=replay_action, gt_law=gt['law'], target_z=pr['target_z'], target_mask=pr['target_mask'],
                 prediction_z=pr['prediction_z'], initial_steps=steps)

    def evaluate(law_name, x, nu_):
        key = [round(v, 6) for v in x]
        for e in state['evals']:
            if e['x'] == key and e.get('law', law_name) == law_name:
                return e
        if len(state['evals']) >= args.budget:
            return None
        frames, seconds = simulate(work, f'obs_{len(state["evals"]):02d}', candidate(law_name, key[0], key[1], nu_), action, obj, args.index, rp)
        fit = compare(observed, frames) if frames is not None else {'iou': None, 'iou_last': None, 'rgb_mae_union': None,
                                                                        'loss': FAILED_LOSS, 'failed': True}
        e = {'n': len(state['evals']) + 1, 'law': law_name, 'x': key, 'seconds': seconds, **fit}
        state['evals'].append(e); save(); print('eval', e, flush=True)
        return e

    if args.arm == 'lawmean_best3':
        probes = []
        for l in LAWS:
            xl, nul = start(l, means[l])
            probes.append((evaluate(l, xl, nul), l, xl, nul))
        if any(p[0] is None for p in probes):
            probes = [p for p in probes if p[0] is not None]
        e_best, law, x0, nu = min(probes, key=lambda p: p[0]['loss'])
        base_z = list(means[law])
    law0 = law
    if args.arm == 'prior_top2':
        probes = [(evaluate(l, x0, nu), l) for l in candidates_law]
        probes = [p for p in probes if p[0] is not None]
        if probes:
            law = min(probes, key=lambda p: p[0]['loss'])[1]
    state.update(law=law, law_start=law0, x0=x0, nu=nu, dims=1 if law == 'jelly' else 2, base_z=base_z); save()
    dims = state['dims']

    # Deterministic compass search (replays exactly from state.json after a restart).
    best = evaluate(law, x0, nu); step = list(steps); last = None
    directions = [(d, s) for d in range(dims) for s in (+1, -1)]
    while best is not None and len(state['evals']) < args.budget and max(step[:dims]) >= 0.02:
        improved = False
        for d, s in ([last] if last else []) + [v for v in directions if v != last]:
            y = list(best['x']); y[d] = float(np.clip(y[d] + s * step[d], *BOUNDS[names[d]]))
            if [round(v, 6) for v in y] == best['x']:
                continue
            e = evaluate(law, y, nu)
            if e is None:
                break
            if e['loss'] < best['loss']:
                best, last, improved = e, (d, s), True
                break
        if not improved:
            step = [v / 2 for v in step]; last = None

    def best_after(k):
        """Best candidate among the first k simulations (any law for best3); k=0 is the start point."""
        if k == 0 or not state['evals'][:k]:
            return (law0 if args.arm != 'lawmean_best3' else None), x0
        e = min(state['evals'][:k], key=lambda e: e['loss'])
        return e.get('law', law), e['x']
    budgets = {}
    report = sorted({int(v) for v in args.report_budgets.split(',')} | {int(v) for v in args.replay_budgets.split(',')})
    for k in [k for k in report if k <= args.budget]:
        if args.arm == 'lawmean_best3' and k == 0:
            continue
        lk, x = best_after(min(k, len(state['evals'])))
        z = list(base_z) if (lk == law or args.arm == 'prior_top2') else list(means[lk])
        z[0] = encode(0, 10 ** x[0])
        if lk != 'jelly':
            z[3] = encode(3, 10 ** x[1])
        elif args.arm not in ('prior', 'prior_sigma', 'prior_top2'):
            z[3] = 0.0  # jelly has no yield stress: all-law train mean
        m = np.array(pr['target_mask'], bool); err = np.abs(np.array(z) - np.array(pr['target_z']))
        n = min(k, len(state['evals']))
        budgets[str(k)] = {'law': lk, 'x': x, 'sims': n, 'wall_seconds': float(sum(e['seconds'] for e in state['evals'][:n])),
                           'prediction_z': z, 'abs_err_z': [float(v) if mm else None for v, mm in zip(err, m)],
                           'mae_z': float(err[m].mean()),
                           'observed_fit': next((dict(iou=e['iou'], rgb_mae_union=e['rgb_mae_union'], loss=e['loss'], failed=e.get('failed', False))
                                                 for e in state['evals'][:max(n, 1)] if e['x'] == x and e.get('law', lk) == lk), None) if n else None}
    state['budgets'] = budgets; save()

    # Unseen-action replay vs the shared GT replay.
    gt_params = {k: gcfg[k] for k in ('material', 'E', 'nu', 'density', 'yield_stress', 'hardening', 'softening') if k in gcfg}
    wanted = [str(k) for k in sorted({int(v) for v in args.replay_budgets.split(',')}) if str(k) in budgets]
    for k in [k for k in state['replay'] if k.isdigit() and 'tag' not in state['replay'][k]]:
        # Replays written by the first (budget-12) version carry no tag: keep only if the estimate is unchanged.
        if k in budgets and [round(v, 6) for v in old_budgets.get(k, {}).get('x', [])] == budgets[k]['x']:
            state['replay'][k]['tag'] = json.dumps([budgets[k]['law'], budgets[k]['x']])
        else:
            state['replay'].pop(k)
    if any(k not in state['replay'] for k in wanted):
        ref = gt_replay(root, replay_action, obj, args.index, rp, gt_params)
        state['replay']['gt_failed'] = ref is None
        if ref is not None:
            done = {}
            for k in wanted:
                b = budgets[k]; tag = json.dumps([b['law'], b['x']])
                if k in state['replay'] and state['replay'][k].get('tag', tag) == tag:
                    done[tag] = state['replay'][k]; continue
                if tag not in done:
                    frames, sec = simulate(work, f'replay_b{k}', candidate(b['law'], b['x'][0], b['x'][1], nu if (b['law'] == law or args.arm == 'prior_top2') else
                                                                          decode(means[b['law']])[1]), replay_action, obj, args.index, rp)
                    done[tag] = {**(compare(ref, frames) if frames is not None else {'failed': True}), 'seconds': sec, 'tag': tag}
                state['replay'][k] = done[tag]; save(); print('replay', k, done[tag], flush=True)
    if args.gt_observed and 'gt_observed' not in state:
        frames, sec = simulate(work, 'gt_observed', gt_params, action, obj, args.index, rp)
        state['gt_observed'] = {**(compare(observed, frames) if frames is not None else {'failed': True}), 'seconds': sec}; save()
    state['complete'] = True; save()
    print('done', args.index, args.arm, {k: v['mae_z'] for k, v in budgets.items()}, flush=True)


if __name__ == '__main__':
    main()
