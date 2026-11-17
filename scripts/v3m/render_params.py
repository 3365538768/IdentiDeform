"""V3m (project website): render ONE parameter set for a V2-main validation sample on its observed action,
using exactly the simulation/render path of scripts/v3d/c1_refine.py (simulate(..., keep=True)).
Tags:
  lawmean_gt  GT-class train mean (budget 0)
  prior       prior_top1 budget-0 start (our video prior, no simulation)
  prior4      best (lowest loss) of the first 4 simulations of prior_top1 (evals n <= 4; n=1 is the prior itself)
  prior12     same with n <= 12
  refined     best of all prior_top1 evaluations
  best3_24    best of the first 24 evaluations of lawmean_best3 (uninformed: 3 law-mean probes + compass search)
  pixie       PIXIE feed-forward prediction (runs/v3f_pixie_filled_val), converted exactly like the prior arm's start point
Output: data/v3m_site/<idx>/<tag>.npz (mask packbits + rgb uint8 at 224, V,T,H,W[,3]) and <tag>.json (params + cost).
If an existing tag of this sample has identical parameters, its render is copied instead of re-simulated.
"""
import argparse, json, math, shutil, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v3d'))
import c1_refine as C  # noqa: E402

TAGS = ('lawmean_gt', 'prior', 'prior4', 'prior12', 'refined', 'best3_24', 'pixie')
PIXIE = ROOT / 'runs/v3f_pixie_filled_val/predictions.jsonl'
a = argparse.ArgumentParser(); a.add_argument('--index', type=int, required=True)
a.add_argument('--tag', choices=TAGS, required=True)
a.add_argument('--run', default='v2main_C_seed29'); a.add_argument('--dry', action='store_true', help='print parameters only')
args = a.parse_args()
idx = f'{args.index:06d}'
gt = json.loads((ROOT / f'data/v2_main/val/{idx}/gt.json').read_text())
base = ROOT / f'data/v3b_refine/{args.run}/{idx}'


def start(z):
    """Identical to c1_refine.main().start: z (E/rho, nu, rho, sigma_y/rho; ln1p + train z-score) -> x, nu."""
    e, nu_, _, sg = C.decode(z)
    x = [math.log10(max(e, 1e-3)), math.log10(max(sg, 1e-3))]
    return [round(float(np.clip(x[d], *C.BOUNDS[n])), 6) for d, n in enumerate(('E', 'sigma_y'))], nu_


def best(evals, nmax=None):
    ev = [e for e in evals if (nmax is None or e['n'] <= nmax)]
    ok = [e for e in ev if not e.get('failed') and e.get('iou') is not None]
    return min(ok, key=lambda e: e['loss']), ev


cost = {}
if args.tag == 'lawmean_gt':
    s = json.loads((base / 'lawmean_gt/state.json').read_text()); law, x, nu = s['law'], s['x0'], s['nu']
elif args.tag == 'pixie':
    sid = f'v2main/val/{idx}'
    pr = next(r for r in map(json.loads, PIXIE.read_text().splitlines()) if r['sample_id'] == sid)
    law = C.LAWS[pr['predicted_law_id']]; x, nu = start(pr['prediction_z'])
    cost = {'source': str(PIXIE.relative_to(ROOT)), 'prediction_z': pr['prediction_z'], 'sims': 0}
elif args.tag == 'best3_24':
    s = json.loads((base / 'lawmean_best3/state.json').read_text())
    e, ev = best(s['evals'], 24); law = e['law']
    # probes of the other laws used that law's own train-mean nu; the search under the chosen law uses state nu
    nu = s['nu'] if law == s['law'] else C.decode(C.law_means()[law])[1]
    x = e['x']
    cost = {'best_n': e['n'], 'sims': len(ev), 'sim_seconds': sum(v['seconds'] for v in ev), 'state_iou3': e['iou']}
else:
    s = json.loads((base / 'prior_top1/state.json').read_text())
    if args.tag == 'prior':
        law, x, nu = s['law_start'], s['x0'], s['nu']
        e0 = next((e for e in s['evals'] if e['x'] == x and e.get('law', law) == law), None)
        cost = {'sims': 0, 'sim_seconds': 0.0, 'state_iou3': e0 and e0.get('iou')}
    else:
        e, ev = best(s['evals'], {'prior4': 4, 'prior12': 12, 'refined': None}[args.tag])
        law, x, nu = e.get('law', s['law']), e['x'], s['nu']
        cost = {'best_n': e['n'], 'sims': len(ev), 'sim_seconds': sum(v['seconds'] for v in ev), 'state_iou3': e['iou']}
params = C.candidate(law, x[0], x[1], nu)
if args.dry:
    print(idx, args.tag, law, x, round(nu, 4), cost); sys.exit(0)
out = ROOT / 'data/v3m_site' / idx; out.mkdir(parents=True, exist_ok=True)
(out / f'{args.tag}.json').write_text(json.dumps({'law': law, 'x': x, 'nu': nu, 'params': params, 'action': gt['action'],
                                                  'object': gt['object'], 'cost': cost}, indent=1))
for t in TAGS:
    if t != args.tag and (out / f'{t}.npz').exists() and (out / f'{t}.json').exists() \
            and json.loads((out / f'{t}.json').read_text())['params'] == params:
        shutil.copyfile(out / f'{t}.npz', out / f'{args.tag}.npz')
        print('done', idx, args.tag, 'copied from', t, flush=True); sys.exit(0)
rp = json.loads(C.V1_RUN_PARAMETERS.read_text())
frames, sec = C.simulate(out, args.tag, params, gt['action'], gt['object'], args.index, rp, keep=True)
print('done', idx, args.tag, frames is not None, round(sec, 1), flush=True)
if frames is None:
    sys.exit(1)
