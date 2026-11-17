"""V3r: material twins for the gripper task on V3 val (used by scripts/v3e/squeeze_sim.py when FB_GRIP_TWINS=v3).

Same definitions as V3e (scripts/v3e/squeeze_sim.twin_params), with the V3 prior and the V3 refinement (V3q):
  gt              GT config of data/v3_main/val/<i> (true hardening/softening)
  ensemble_prior  = V3q prior_top2 budget 0: V3 prior ensemble (docs/v3q/PRIOR_ENSEMBLE_val.json: decoded mean
                    log10 E/rho, log10 sigma_y/rho, nu; law votes), top-1 law (tie -> law whose train-mean E/rho is
                    closest), x clipped to that law's V3 bounds, nu = ensemble nu. No simulations.
  lawmean_pred    V3 train mean (docs/v3q/LAW_MEANS.json) of the ensemble's top-1 law
  refined_top2    V3q prior_top2 at budget 12 (data/v3q_refine/val/<i>/prior_top2/state.json; nested budgets of the
                    24-sim search; nu = the arm's fixed nu)   [fallback: same script run by v3r, same out layout]
  refined_best3   V3q lawmean_best3 at budget 12
Twins: density 1000 + specific moduli, hardening/softening 0.25 (not predicted), as V3e (V3q candidate()).
"""
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LAWS = ['jelly', 'metal', 'plasticine']
BOUNDS_E = {'jelly': (3.5, 6.0), 'plasticine': (3.5, 5.5), 'metal': (4.5, 6.0)}  # = scripts/v3q/c1_refine.py
BOUNDS_S = (3.2, 4.5)
DATA = ROOT / 'data/v3_main/val'
ENS = ROOT / os.environ.get('FB_GRIP_PRIOR', 'docs/v3q/PRIOR_ENSEMBLE_val.json')
REFINE = ROOT / os.environ.get('FB_GRIP_REFINE', 'data/v3q_refine')
ARM = {'refined_top2': 'prior_top2', 'refined_best3': 'lawmean_best3'}
BUDGET = 12


def candidate(law, log_e, log_s, nu):  # = scripts/v3b/c1_refine.candidate
    p = {'material': law, 'E': 10 ** log_e, 'nu': float(np.clip(nu, 0.05, 0.48)), 'density': 1000.0}
    if law != 'jelly':
        p.update(yield_stress=10 ** log_s, hardening=0.25)
        if law == 'plasticine':
            p['softening'] = 0.25
    return p


def law_means():
    return json.loads((ROOT / 'docs/v3q/LAW_MEANS.json').read_text())['means']


def from_ln1p(v):
    return math.log10(max(math.expm1(v), 1e-3))


def clip(law, x):
    return [round(float(np.clip(x[0], *BOUNDS_E[law])), 6), round(float(np.clip(x[1], *BOUNDS_S)), 6)]


def gt_params(index):
    g = json.loads((DATA / f'{index:06d}/config.json').read_text())
    return {k: g[k] for k in ('material', 'E', 'nu', 'density', 'yield_stress', 'hardening', 'softening') if k in g}


def prior(index):
    ens = json.loads(ENS.read_text())['samples'][str(index)]; LM = law_means(); votes = ens['law_posterior_votes']
    xp = ens['x_log10']
    order = sorted(range(3), key=lambda k: (-votes[k], abs(LM[LAWS[k]]['logE'] - math.log1p(10 ** xp[0]))))
    xp = [xp[0], xp[1] if xp[1] is not None else BOUNDS_S[0]]
    return LAWS[order[0]], xp, float(ens['nu']), votes


def best_at(st, k):
    """Budget-k estimate. Budgets are nested (best observed loss among the first k sims), so before the 24-sim search
    has finished the budget-12 point is already final once 12 evaluations exist."""
    if 'budgets' in st and str(k) in st['budgets']:
        return st['budgets'][str(k)]
    ev = st['evals']
    if len(ev) < k:
        raise RuntimeError(f'refinement has only {len(ev)} < {k} evaluations')
    e = min(ev[:k], key=lambda e: e['loss'])
    return {'law': e['law'], 'x': e['x'], 'sims': k, 'wall_seconds': float(sum(v['seconds'] for v in ev[:k]))}


def twin_params(index, method):
    if method == 'gt':
        p = gt_params(index); return p, {'law': p['material']}
    if method == 'ensemble_prior':
        law, xp, nu, votes = prior(index); x = clip(law, xp)
        return candidate(law, x[0], x[1], nu), {'law': law, 'x': x, 'nu': nu, 'votes': votes, 'sims_refine': 0, 'refine_wall_seconds': 0.0,
                                                'source': str(ENS.relative_to(ROOT))}
    if method == 'lawmean_pred':
        law = prior(index)[0]; m = law_means()[law]
        x = [from_ln1p(m['logE']), from_ln1p(m['logS']) if m.get('logS') is not None else BOUNDS_S[0]]
        return candidate(law, x[0], x[1], m['nu']), {'law': law, 'x': x, 'nu': m['nu']}
    if method in ARM:
        sp = REFINE / f'val/{index:06d}/{ARM[method]}/state.json'; st = json.loads(sp.read_text())
        b = best_at(st, BUDGET)
        return candidate(b['law'], b['x'][0], b['x'][1], st['nu']), {
            'law': b['law'], 'x': b['x'], 'nu': st['nu'], 'sims_refine': b['sims'], 'refine_wall_seconds': b.get('wall_seconds', 0.0),
            'search_complete_budget': st.get('complete_budget'), 'state': str(sp.relative_to(ROOT))}
    raise ValueError(method)
