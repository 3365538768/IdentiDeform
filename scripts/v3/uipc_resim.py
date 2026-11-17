"""V3a sanity check: re-simulate uipc val cases with PREDICTED (E, nu) and compare to the observed motion.

For N val cases (first N/2 of each val geometry, by sample id) and each arm, the predicted E and nu from
docs/v3a/UIPC_FEWSHOT.json (one source seed) replace the ground truth in the case; everything else (mesh,
load schedule, solver) is unchanged. Error vs the observed uipc surface trajectory (traj.npz):
  disp_err_m      mean over frames and surface nodes of |u_pred - u_obs|
  disp_err_rel    disp_err_m / mean |u_obs|
  tip_rmse_px     RMSE of the mean tip y displacement, in side-view pixels
Each simulation runs in its own subprocess with a timeout; failures are recorded, never dropped.
'oracle' re-simulates with the ground-truth (E, nu) (solver repeatability floor).

  .engine-envs/uipc/bin/python scripts/v3/uipc_resim.py --seed 17 --out docs/v3a/resim/seed17.json
  .engine-envs/uipc/bin/python scripts/v3/uipc_resim.py --merge docs/v3a/resim --into docs/v3a/UIPC_FEWSHOT.json
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ARMS = 'oracle,zero_shot,headft_8,engcond_8,scratch_8,headft_32,engcond_32,scratch_32'
E_CLAMP = (1.0e3, 1.0e7)


def one(spec):
    sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'scripts/v3'))
    from gen_uipc_video import IMAGE, simulate
    from physical_renderer import exterior_triangles
    d = ROOT / 'data/v3_uipc/val' / spec['case_dir']
    case = json.loads((d / 'gt.json').read_text())['case']
    case['young_pa'], case['poisson'] = spec['E_pa'], spec['nu']
    ws = ROOT / '.cache/v3a_resim_ws' / f"{spec['case_dir']}_{os.getpid()}"
    try:
        rest, tets, disp, _, health = simulate(case, ws)
    finally:
        shutil.rmtree(ws, ignore_errors=True)
    surface = np.unique(exterior_triangles(tets))
    obs = np.load(d / 'traj.npz')
    u_obs = obs['surface_displacement_m'].astype(np.float64); u = disp[:, surface]
    if u.shape != u_obs.shape:
        raise ValueError('surface mismatch')
    tip = np.isclose(rest[:, 0], case['dimensions_m'][0])
    px = IMAGE / (1.45 * case['dimensions_m'][0])
    err = np.linalg.norm(u - u_obs, axis=-1).mean()
    return dict(disp_err_m=float(err), disp_err_rel=float(err / np.linalg.norm(u_obs, axis=-1).mean()),
                tip_rmse_px=float(np.sqrt(((disp[:, tip, 1].mean(1) - obs['tip_y_m']) ** 2).mean()) * px),
                min_jacobian=health['min_jacobian'])


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--seed', default='17'); a.add_argument('--fewshot', default='docs/v3a/UIPC_FEWSHOT.json')
    a.add_argument('--arms', default=ARMS); a.add_argument('--n', type=int, default=10); a.add_argument('--out')
    a.add_argument('--timeout', type=int, default=900); a.add_argument('--one'); a.add_argument('--merge'); a.add_argument('--into')
    args = a.parse_args(); os.chdir(ROOT)
    if args.one:
        print('RESULT ' + json.dumps(one(json.loads(args.one)))); return
    if args.merge:
        parts = {p.stem: json.loads(p.read_text()) for p in sorted(Path(args.merge).glob('seed*.json'))}
        summary = {}
        for arm in args.arms.split(','):
            vals = {m: [c[m] for p in parts.values() for c in p['cases'] if c['arm'] == arm and c.get('ok')] for m in ('disp_err_m', 'disp_err_rel', 'tip_rmse_px')}
            fails = sum(1 for p in parts.values() for c in p['cases'] if c['arm'] == arm and not c.get('ok'))
            summary[arm] = dict(n_ok=len(vals['disp_err_m']), n_failed=fails,
                                **{m: dict(mean=float(np.mean(v)) if v else None, median=float(np.median(v)) if v else None) for m, v in vals.items()})
        target = Path(args.into); res = json.loads(target.read_text())
        res['resimulation_check'] = dict(protocol=__doc__.split('\n\n')[0] + ' ' + __doc__.split('\n\n')[1], e_clamp_pa=E_CLAMP,
                                         per_seed=parts, summary=summary)
        tmp = target.with_suffix('.tmp'); tmp.write_text(json.dumps(res, indent=1)); tmp.replace(target)
        for arm, s in summary.items():
            print(arm, s)
        return
    res = json.loads((ROOT / args.fewshot).read_text()); run = res['runs'][args.seed]
    ids = run['val_sample_ids']
    by_geo = {}
    for i, sid in enumerate(ids):
        g = json.loads((ROOT / 'data/v3_uipc/val' / sid.split('/')[-1] / 'gt.json').read_text())['geometry']
        by_geo.setdefault(g, []).append(i)
    picks = [i for g in sorted(by_geo) for i in sorted(by_geo[g], key=lambda j: ids[j])[:args.n // len(by_geo)]]
    cases = []
    for i in picks:
        gt = json.loads((ROOT / 'data/v3_uipc/val' / ids[i].split('/')[-1] / 'gt.json').read_text())
        for arm in args.arms.split(','):
            if arm == 'oracle':
                if args.seed != str(res['protocol']['seeds']).split(',')[0]:
                    continue  # seed independent; run once
                E, nu = gt['regression']['E'], gt['regression']['nu']
            else:
                E, nu = run['arms'][arm]['val_pred']['E_pa'][i], run['arms'][arm]['val_pred']['nu'][i]
            spec = dict(case_dir=ids[i].split('/')[-1], E_pa=float(np.clip(E, *E_CLAMP)), nu=float(nu))
            row = dict(arm=arm, sample_id=ids[i], E_true=gt['regression']['E'], nu_true=gt['regression']['nu'], E_pred=E, nu_pred=nu,
                       E_clamped=bool(spec['E_pa'] != E))
            t0 = time.time()
            try:
                p = subprocess.run([sys.executable, __file__, '--one', json.dumps(spec)], capture_output=True, text=True, timeout=args.timeout)
                line = [l for l in p.stdout.splitlines() if l.startswith('RESULT ')]
                if p.returncode or not line:
                    row.update(ok=False, error=(p.stderr or p.stdout)[-600:])
                else:
                    row.update(ok=True, **json.loads(line[-1][7:]))
            except subprocess.TimeoutExpired:
                row.update(ok=False, error=f'timeout {args.timeout}s')
            row['wall_s'] = time.time() - t0
            cases.append(row); print(json.dumps({k: row[k] for k in row if k != 'error'}), flush=True)
    out = ROOT / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(seed=args.seed, source=args.fewshot, cases=cases), indent=1))


if __name__ == '__main__':
    main()
