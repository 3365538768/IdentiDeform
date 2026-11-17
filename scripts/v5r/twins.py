"""V5r: material twins for the probe-then-grasp task on V5b val (scripts/v3e/squeeze_sim.py with FB_GRIP_TWINS=v5).

  gt            GT config of data/v5b_main/val/<i> (true hardening/softening)
  lawmean_pred  (a) V5b train mean (docs/v5q/LAW_MEANS.json) of the vision prior's top-1 law
  prior_vision  (b) vision prior: V5b 9-run B/C/D ensemble (docs/v5q/PRIOR_ENSEMBLE_val.json, built from docs/v5p/HANDOFF.json and
                    checked against docs/v5p/ENSEMBLE_BCD9_VAL.json): top-1 law by vote fraction (tie -> law whose train-mean E/rho
                    is closest), decoded mean log10 E/rho, log10 sigma_y/rho, nu; x clipped to the V5 law bounds
  prior_vt      (c) vision + touch (+action) prior: docs/v5m/HANDOFF.json arm vision_touch_action, val json; same decoding rule
                    (law = top-1 of the mean softmax, physical = decoded mean ln1p), same clipping
  prior_vis3    supplementary: v5m vision-only 3-seed arm (same model family / training as prior_vt; isolates touch+action)
  refined_vt    (e, optional) docs/v5q refinement arm started from the vision+touch prior, budget 12, if it exists
Twins: density 1000 + specific moduli, hardening/softening 0.25, nu from the respective prior (class mean: train mean nu).
"""
import json
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
LAWS = ['jelly', 'metal', 'plasticine']
BOUNDS_E = {'jelly': (3.0, 5.5), 'plasticine': (3.0, 4.5), 'metal': (4.5, 6.0)}  # = scripts/v5q/c1_refine.py (V5 ranges)
BOUNDS_S = {'plasticine': (2.5, 4.0), 'metal': (3.0, 4.5), 'jelly': (2.5, 4.5)}
DATA = ROOT / 'data/v5b_main/val'
VIS = ROOT / 'docs/v5q/PRIOR_ENSEMBLE_val.json'
VT_ARM = os.environ.get('FB_GRIP_VT_ARM', 'vision_touch_action')
REFINE_VT = os.environ.get('FB_GRIP_REFINE_VT', '')  # e.g. data/v5q_refine/val/{index:06d}/<arm>/state.json


def candidate(law, log_e, log_s, nu):  # = scripts/v3b/c1_refine.candidate
    p = {'material': law, 'E': 10 ** log_e, 'nu': float(np.clip(nu, 0.05, 0.48)), 'density': 1000.0}
    if law != 'jelly':
        p.update(yield_stress=10 ** log_s, hardening=0.25)
        if law == 'plasticine':
            p['softening'] = 0.25
    return p


def law_means():
    return json.loads((ROOT / 'docs/v5q/LAW_MEANS.json').read_text())['means']


def from_ln1p(v):
    return math.log10(max(math.expm1(v), 1e-3))


def clip(law, x):
    return [round(float(np.clip(x[0], *BOUNDS_E[law])), 6), round(float(np.clip(x[1], *BOUNDS_S[law])), 6)]


def top1(votes, log10E):
    LM = law_means()
    return LAWS[sorted(range(3), key=lambda k: (-votes[k], abs(LM[LAWS[k]]['logE'] - math.log1p(10 ** log10E))))[0]]


def gt_params(index):
    g = json.loads((DATA / f'{index:06d}/config.json').read_text())
    return {k: g[k] for k in ('material', 'E', 'nu', 'density', 'yield_stress', 'hardening', 'softening') if k in g}


def vision(index):
    e = json.loads(VIS.read_text())['samples'][str(index)]; xp = e['x_log10']
    return top1(e['law_posterior_votes'], xp[0]), [xp[0], xp[1] if xp[1] is not None else 3.0], float(e['nu']), e['law_posterior_votes']


def vision_touch(index, arm=None):
    h = json.loads((ROOT / 'docs/v5m/HANDOFF.json').read_text())
    e = json.loads((ROOT / h['arms'][arm or VT_ARM]['val']['json']).read_text())['samples'][f'v5bmain/val/{index:06d}']
    ph = e['physical']; xp = [math.log10(max(ph['E_over_rho'], 1e-3)), math.log10(max(ph['sigma_y_over_rho'], 1e-3))]
    return top1(e['law_posterior_votes'], xp[0]), xp, float(ph['nu']), e['law_posterior_votes']


def twin_params(index, method):
    if method == 'gt':
        p = gt_params(index); return p, {'law': p['material']}
    if method in ('prior_vision', 'prior_vt', 'prior_vis3'):
        law, xp, nu, votes = {'prior_vision': vision, 'prior_vt': vision_touch,
                              'prior_vis3': lambda i: vision_touch(i, 'vision')}[method](index); x = clip(law, xp)
        return candidate(law, x[0], x[1], nu), {'law': law, 'x': x, 'x_unclipped': xp, 'nu': nu, 'votes': votes, 'sims_refine': 0,
                                                'refine_wall_seconds': 0.0}
    if method == 'lawmean_pred':
        law = vision(index)[0]; m = law_means()[law]
        x = [from_ln1p(m['logE']), from_ln1p(m['logS']) if m.get('logS') is not None else 3.0]
        return candidate(law, x[0], x[1], m['nu']), {'law': law, 'x': x, 'nu': m['nu']}
    if method == 'refined_vt':
        sp = ROOT / REFINE_VT.format(index=index); st = json.loads(sp.read_text())
        b = st['budgets']['12'] if '12' in st.get('budgets', {}) else min(st['evals'][:12], key=lambda e: e['loss'])
        return candidate(b['law'], b['x'][0], b['x'][1], st['nu']), {'law': b['law'], 'x': b['x'], 'nu': st['nu'], 'sims_refine': 12,
                                                                     'refine_wall_seconds': b.get('wall_seconds', 0.0), 'state': str(sp.relative_to(ROOT))}
    raise ValueError(method)
