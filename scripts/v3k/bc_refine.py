"""V3k-B / V3k-C: prior_top1 simulation-in-the-loop refinement (V3d/V3h arm, verbatim objective/optimiser/budgets/
failure rule from scripts/v3d/c1_refine.py, imported) under two stress conditions, validation split only.

  .venv/bin/python scripts/v3k/bc_refine.py --index 21 --variant hull --asset data/v3k/hull_assets/000021/<obj>.ply
  .venv/bin/python scripts/v3k/bc_refine.py --index 21 --variant mismatch

variant hull (B): every candidate simulation uses the RECONSTRUCTED geometry (3-view frame-0 visual hull written as a
  3DGS PLY by scripts/v3k/b_hull.py) through the identical asset routine (x90, robust normalisation, particle filling);
  the target is the ORIGINAL observed video.  Unseen-action replay (ACTIONS[(i+3)%6]) of the budget 0/4/12 estimates
  with the hull geometry vs a fresh GT replay (GT asset, GT parameters).  Diagnostic last: GT parameters + hull
  geometry on the observed action (geometry-only floor).
variant mismatch (C, "inverse crime"): the observed video is REGENERATED with GT parameters (incl. true hardening/
  softening), GT asset, same cameras/action, but a perturbed simulator: MPM grid n_grid 150 -> 112 (x0.75) and grid
  velocity damping 1 - grid_v_damping_scale 1e-5 -> 1e-4 (x10).  Refinement uses the nominal simulator. No replay.
Outputs data/v3k/bc/<variant>/<idx>/state.json (resumable; simulation dirs deleted after metrics are stored).
"""
import argparse, glob, json, math, os, shutil, subprocess, sys, time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/v3d')]
import c1_refine as C  # noqa: E402  (V3d copy of V3b C1: build/compare/candidate/decode/encode/law_means/...)

MISMATCH = {'n_grid': 112, 'grid_v_damping_scale': 0.9999}


def simulate(work, tag, params, action, obj, index, rp, asset, cfg_over=None, keep=False):
    """scripts/v3d/c1_refine.simulate with the asset path and optional simulator-config overrides exposed."""
    dst = work / tag
    shutil.rmtree(dst, ignore_errors=True); dst.mkdir(parents=True)
    cfg = json.loads(C.TEMPLATE.read_text())
    cfg.update(params, rpic_damping=0.0, grid_v_damping_scale=0.99999, substep_dt=1e-4, frame_dt=0.01, frame_num=400,
               g=[0.0, 0.0, -9.8])
    cfg.update(cfg_over or {})
    (dst / 'config.json').write_text(json.dumps(cfg, indent=1))
    cli = C.build(rp, Path(asset), dst / 'config.json', dst,
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
    for attempt in range(C.QUICK_RETRIES + 1):
        launched = time.time()
        with (dst / 'run.log').open('w') as log:
            code = subprocess.call([str(C.ROOT / 'scripts/v2/run_upstream_v2.sh'), *cli], stdout=log, stderr=subprocess.STDOUT, env=env)
        if code == 0 or time.time() - launched > C.QUICK_FAIL_SECONDS:
            break
        print('quick failure', tag, attempt, flush=True)
        if attempt < C.QUICK_RETRIES:
            time.sleep(75)
    if code != 0:
        tail = (dst / 'run.log').read_text(errors='replace')[-600:]
        (work / f'{tag}.failed.log').write_text(tail)
        shutil.rmtree(dst, ignore_errors=True)
        print('simulation failed', tag, code, flush=True)
        return None, time.time() - started
    frames = C.load_frames(dst)
    if keep:
        np.savez_compressed(work / f'{tag}.npz', mask=np.packbits(frames[0], -1), shape=np.array(frames[0].shape),
                            rgb=frames[1].astype(np.uint8))
    shutil.rmtree(dst, ignore_errors=True)
    return frames, time.time() - started


def main():
    a = argparse.ArgumentParser(); a.add_argument('--index', type=int, required=True)
    a.add_argument('--variant', choices=('hull', 'mismatch'), required=True); a.add_argument('--asset')
    a.add_argument('--budget', type=int, default=12); a.add_argument('--replay-budgets', default='0,4,12')
    a.add_argument('--run', default='v2main_C_seed29'); a.add_argument('--ensemble', default='docs/v3b/ENSEMBLE_BCD9_VAL.json')
    a.add_argument('--out', default='data/v3k/bc'); args = a.parse_args()
    sid = f'v2main/val/{args.index:06d}'
    pred_dir = C.best_dir(ROOT / f'runs/{args.run}')[0]
    pr = next(r for r in map(json.loads, (pred_dir / 'predictions.jsonl').read_text().splitlines()) if r['sample_id'] == sid)
    gt = json.loads((ROOT / f'data/v2_main/val/{args.index:06d}/gt.json').read_text())
    gcfg = json.loads((ROOT / f'data/v2_main/val/{args.index:06d}/config.json').read_text())
    obj, action = gt['object'], gt['action']
    replay_action = C.ACTIONS[(C.ACTIONS.index(action) + 3) % len(C.ACTIONS)]
    rp = json.loads(C.V1_RUN_PARAMETERS.read_text())
    gt_asset = C.ASSETS / f'{obj}.ply'
    asset = Path(args.asset).resolve() if args.variant == 'hull' else gt_asset
    assert asset.exists(), asset
    work = ROOT / args.out / args.variant / f'{args.index:06d}'; work.mkdir(parents=True, exist_ok=True)
    state_path = work / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'evals': [], 'replay': {}}
    bad = next((i for i, e in enumerate(state['evals']) if e.get('failed') and e['seconds'] < C.QUICK_FAIL_SECONDS), None)
    if bad is not None:
        state['evals'] = state['evals'][:bad]
    state.pop('complete', None)
    def save():
        tmp = state_path.with_suffix('.tmp'); tmp.write_text(json.dumps(state, indent=1)); os.replace(tmp, state_path)
    gt_params = {k: gcfg[k] for k in ('material', 'E', 'nu', 'density', 'yield_stress', 'hardening', 'softening') if k in gcfg}

    if args.variant == 'mismatch':
        if not (work / 'observed_mismatch.npz').exists():
            frames, sec = simulate(work, 'observed_mismatch', gt_params, action, obj, args.index, rp, gt_asset, MISMATCH, keep=True)
            state['observed_regen'] = {'failed': frames is None, 'seconds': sec, 'perturbation': MISMATCH}; save()
            if frames is None:
                state['complete'] = True; save(); print('observed regeneration failed', flush=True); return
        observed = C.load_kept(work, 'observed_mismatch')
        orig = C.load_pack(glob.glob(str(ROOT / f'data/v2_main/val/{args.index:06d}/*/sample_pack.npz'))[0])
        state['observed_regen_vs_original'] = C.compare(orig, observed); save()
    else:
        observed = C.load_pack(glob.glob(str(ROOT / f'data/v2_main/val/{args.index:06d}/*/sample_pack.npz'))[0])

    means = C.law_means()
    ens = json.loads((ROOT / args.ensemble).read_text())['samples'][sid]
    votes = ens['law_posterior_votes']; base_z = list(ens['prediction_z'])
    order = sorted(range(3), key=lambda k: (-votes[k], abs(means[C.LAWS[k]][0] - base_z[0])))
    law = C.LAWS[order[0]]
    e_, nu, _, sg = C.decode(base_z)
    names = ['E', 'sigma_y']
    x0 = [round(float(np.clip(v, *C.BOUNDS[names[d]])), 6) for d, v in enumerate([math.log10(max(e_, 1e-3)), math.log10(max(sg, 1e-3))])]
    dims = 1 if law == 'jelly' else 2
    state.update(sample_id=sid, variant=args.variant, asset=str(asset.relative_to(ROOT)), object=obj, observed_action=action,
                 replay_action=replay_action, gt_law=gt['law'], law=law, ensemble_law_votes=votes, x0=x0, nu=nu, dims=dims,
                 base_z=base_z, target_z=pr['target_z'], target_mask=pr['target_mask'],
                 simulator_for_refinement='nominal (V3b C1)', observed_simulator=MISMATCH if args.variant == 'mismatch' else 'original V2 data')
    save()

    def evaluate(x):
        key = [round(v, 6) for v in x]
        for e in state['evals']:
            if e['x'] == key:
                return e
        if len(state['evals']) >= args.budget:
            return None
        frames, seconds = simulate(work, f'obs_{len(state["evals"]):02d}', C.candidate(law, key[0], key[1], nu), action, obj, args.index, rp, asset)
        fit = C.compare(observed, frames) if frames is not None else {'iou': None, 'iou_last': None, 'rgb_mae_union': None,
                                                                          'loss': C.FAILED_LOSS, 'failed': True}
        e = {'n': len(state['evals']) + 1, 'law': law, 'x': key, 'seconds': seconds, **fit}
        state['evals'].append(e); save(); print('eval', e, flush=True)
        return e

    best = evaluate(x0); step = [C.STEP0, C.STEP0]; last = None
    directions = [(d, s) for d in range(dims) for s in (+1, -1)]
    while best is not None and len(state['evals']) < args.budget and max(step[:dims]) >= 0.02:
        improved = False
        for d, s in ([last] if last else []) + [v for v in directions if v != last]:
            y = list(best['x']); y[d] = float(np.clip(y[d] + s * step[d], *C.BOUNDS[names[d]]))
            if [round(v, 6) for v in y] == best['x']:
                continue
            e = evaluate(y)
            if e is None:
                break
            if e['loss'] < best['loss']:
                best, last, improved = e, (d, s), True
                break
        if not improved:
            step = [v / 2 for v in step]; last = None

    budgets = {}
    m = np.array(pr['target_mask'], bool)
    for k in range(0, args.budget + 1):
        n = min(k, len(state['evals']))
        x = x0 if n == 0 else min(state['evals'][:n], key=lambda e: e['loss'])['x']
        z = list(base_z); z[0] = C.encode(0, 10 ** x[0])
        if law != 'jelly':
            z[3] = C.encode(3, 10 ** x[1])
        err = np.abs(np.array(z) - np.array(pr['target_z']))
        fit = next((e for e in state['evals'][:max(n, 1)] if e['x'] == x), None) if n else None
        budgets[str(k)] = {'x': x, 'sims': n, 'prediction_z': z, 'abs_err_z': [float(v) if mm else None for v, mm in zip(err, m)],
                           'mae_z': float(err[m].mean()), 'observed_fit': {kk: fit.get(kk) for kk in ('iou', 'rgb_mae_union', 'loss', 'failed')} if fit else None}
    state['budgets'] = budgets; save()

    if args.variant == 'hull':
        wanted = [k for k in args.replay_budgets.split(',') if k in budgets]
        if any(k not in state['replay'] for k in wanted):
            if (work / 'replay_gt.npz').exists():
                ref = C.load_kept(work, 'replay_gt')
            elif state['replay'].get('gt_failed'):
                ref = None
            else:
                ref, sec = simulate(work, 'replay_gt', gt_params, replay_action, obj, args.index, rp, gt_asset, keep=True)
                state['replay']['gt_failed'] = ref is None; state['replay']['gt_seconds'] = sec; save()
            if ref is not None:
                done = {}
                for k in wanted:
                    b = budgets[k]; tag = json.dumps(b['x'])
                    if k in state['replay'] and state['replay'][k].get('tag') == tag:
                        done[tag] = state['replay'][k]; continue
                    if tag not in done:
                        frames, sec = simulate(work, f'replay_b{k}', C.candidate(law, b['x'][0], b['x'][1], nu), replay_action, obj, args.index, rp, asset)
                        done[tag] = {**(C.compare(ref, frames) if frames is not None else {'failed': True}), 'seconds': sec, 'tag': tag}
                    state['replay'][k] = done[tag]; save(); print('replay', k, done[tag], flush=True)
        if 'gt_params_hull_observed' not in state:
            frames, sec = simulate(work, 'gt_hull_observed', gt_params, action, obj, args.index, rp, asset)
            state['gt_params_hull_observed'] = {**(C.compare(observed, frames) if frames is not None else {'failed': True}), 'seconds': sec}; save()
    state['complete'] = True; save()
    print('done', args.index, args.variant, {k: round(budgets[k]['mae_z'], 3) for k in ('0', '4', '12') if k in budgets}, flush=True)


if __name__ == '__main__':
    main()
