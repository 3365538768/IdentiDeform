"""V3h: C1 refinement (scripts/v3b/c1_refine.py, identical objective / compass search / failure rule) from
alternative initializations, MAP-law start, no law probes, no replay (validation only). See docs/v3h/PROTOCOL.json.

Arms: dino_c (DINOv3 C seed ensemble), gtnoise_<sigma> (PRIVILEGED GT law + GT params + fixed Gaussian noise in
log10 decades), ens_map (9-run V-JEPA B/C/D ensemble, top-1 law, no probes).

  .venv/bin/python scripts/v3h/c1h_refine.py --index 12 --arm dino_c --budget 12
State: data/v3h_refine/<arm>/<index>/state.json (resumable; simulation dirs deleted after scoring).
"""
import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v3b'))
import c1_refine as C  # noqa: E402  (sets up sys.path for gen_v2 / upstream / r1_report)
from c1_refine import LAWS, ACTIONS, BOUNDS, STEP0, FAILED_LOSS, QUICK_FAIL_SECONDS, STATS  # noqa: E402,F401
from r1_report import best_dir  # noqa: E402

DINO_RUNS = [f'v3c_dino_C_seed{s}' for s in (17, 29, 43)]
PROTO = json.loads((ROOT / 'docs/v3h/PROTOCOL.json').read_text())


def preds(run, sid):
    d, _ = best_dir(ROOT / 'runs' / run)
    for line in (d / 'predictions.jsonl').read_text().splitlines():
        r = json.loads(line)
        if r['sample_id'] == sid:
            return r, str(d)
    raise KeyError(sid)


def vote_law(votes, z0, means):
    order = sorted(range(3), key=lambda k: (-votes[k], abs(means[LAWS[k]][0] - z0)))
    return LAWS[order[0]]


def main():
    a = argparse.ArgumentParser(); a.add_argument('--index', type=int, required=True)
    a.add_argument('--arm', required=True); a.add_argument('--budget', type=int, default=12)
    a.add_argument('--out', default='data/v3h_refine'); args = a.parse_args()
    sid = f'v2main/val/{args.index:06d}'
    ref, ref_dir = preds('v2main_C_seed29', sid)  # targets (identical across runs)
    gt = json.loads((ROOT / f'data/v2_main/val/{args.index:06d}/gt.json').read_text())
    obj, action = gt['object'], gt['action']
    rp = json.loads(C.V1_RUN_PARAMETERS.read_text())
    work = ROOT / args.out / args.arm / f'{args.index:06d}'; work.mkdir(parents=True, exist_ok=True)
    state_path = work / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'evals': []}
    bad = next((i for i, e in enumerate(state['evals']) if e.get('failed') and e['seconds'] < QUICK_FAIL_SECONDS), None)
    if bad is not None:
        state['evals'] = state['evals'][:bad]
    state.pop('complete', None)

    def save():
        tmp = state_path.with_suffix('.tmp'); tmp.write_text(json.dumps(state, indent=1)); os.replace(tmp, state_path)

    observed = C.load_pack(glob.glob(str(ROOT / f'data/v2_main/val/{args.index:06d}/*/sample_pack.npz'))[0])
    names = ['E', 'sigma_y']
    means = C.law_means()
    clip = lambda x: [round(float(np.clip(x[d], *BOUNDS[names[d]])), 6) for d in range(2)]

    def start(z):
        e, nu_, _, sg = C.decode(z)
        return clip([math.log10(max(e, 1e-3)), math.log10(max(sg, 1e-3))]), nu_

    extra = {}
    if args.arm == 'dino_c':
        rows = [preds(r, sid) for r in DINO_RUNS]
        base_z = np.mean([r['prediction_z'] for r, _ in rows], 0).tolist()
        votes = [0, 0, 0]
        for r, _ in rows:
            votes[r['predicted_law_id']] += 1
        votes = [v / len(rows) for v in votes]
        law = vote_law(votes, base_z[0], means); x0, nu = start(base_z)
        extra = {'law_votes': votes, 'member_laws': [LAWS[r['predicted_law_id']] for r, _ in rows],
                 'prediction_dirs': [d for _, d in rows]}
    elif args.arm == 'ens_map':
        ens = json.loads((ROOT / 'docs/v3b/ENSEMBLE_BCD9_VAL.json').read_text())['samples'][sid]
        base_z = list(ens['prediction_z']); votes = ens['law_posterior_votes']
        law = vote_law(votes, base_z[0], means); x0, nu = start(base_z); extra = {'law_votes': votes}
    elif args.arm.startswith('gtnoise_'):
        sigma = float(args.arm.split('_', 1)[1])
        eps = PROTO['noise_eps'][str(args.index)]
        law = gt['law']; base_z = list(ref['target_z'])
        e, _, _, sg = C.decode(base_z)
        xg = [math.log10(e), math.log10(sg) if ref['target_mask'][3] else 0.0]
        x0 = clip([xg[d] + sigma * eps[d] for d in range(2)])
        nu = C.decode(means[law])[1]
        extra = {'sigma': sigma, 'eps': eps, 'x_gt': xg}
    else:
        raise SystemExit(f'unknown arm {args.arm}')
    state.update(sample_id=sid, arm=args.arm, object=obj, observed_action=action, gt_law=gt['law'], law=law, x0=x0, nu=nu,
                 base_z=base_z, dims=1 if law == 'jelly' else 2, target_z=ref['target_z'], target_mask=ref['target_mask'],
                 initial_steps=[STEP0, STEP0], **extra)
    save()
    dims = state['dims']

    def evaluate(x):
        key = [round(v, 6) for v in x]
        for e in state['evals']:
            if e['x'] == key:
                return e
        if len(state['evals']) >= args.budget:
            return None
        frames, seconds = C.simulate(work, f'obs_{len(state["evals"]):02d}', C.candidate(law, key[0], key[1], nu), action, obj, args.index, rp)
        fit = C.compare(observed, frames) if frames is not None else {'iou': None, 'iou_last': None, 'rgb_mae_union': None,
                                                                          'loss': FAILED_LOSS, 'failed': True}
        e = {'n': len(state['evals']) + 1, 'law': law, 'x': key, 'seconds': seconds, **fit}
        state['evals'].append(e); save(); print('eval', e, flush=True)
        return e

    # Deterministic compass search, verbatim from c1_refine.py.
    best = evaluate(x0); step = [STEP0, STEP0]; last = None
    directions = [(d, s) for d in range(dims) for s in (+1, -1)]
    while best is not None and len(state['evals']) < args.budget and max(step[:dims]) >= 0.02:
        improved = False
        for d, s in ([last] if last else []) + [v for v in directions if v != last]:
            y = list(best['x']); y[d] = float(np.clip(y[d] + s * step[d], *BOUNDS[names[d]]))
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
    m = np.array(ref['target_mask'], bool)
    for k in range(args.budget + 1):
        n = min(k, len(state['evals']))
        x = x0 if n == 0 else min(state['evals'][:n], key=lambda e: e['loss'])['x']
        z = list(base_z); z[0] = C.encode(0, 10 ** x[0])
        if law != 'jelly':
            z[3] = C.encode(3, 10 ** x[1])
        err = np.abs(np.array(z) - np.array(ref['target_z']))
        budgets[str(k)] = {'law': law, 'x': x, 'sims': n, 'wall_seconds': float(sum(e['seconds'] for e in state['evals'][:n])),
                           'prediction_z': z, 'abs_err_z': [float(v) if mm else None for v, mm in zip(err, m)],
                           'mae_z': float(err[m].mean())}
    state['budgets'] = budgets; state['complete'] = True; save()
    print('done', args.index, args.arm, {k: round(v['mae_z'], 3) for k, v in budgets.items() if int(k) in (0, 4, 12)}, flush=True)


if __name__ == '__main__':
    main()
