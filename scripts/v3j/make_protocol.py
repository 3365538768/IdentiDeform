"""V3j: pre-register the reviewer-control experiment (validation only) -> docs/v3j/PROTOCOL.json + docs/v3j/jobs.json.
Refuses to change an existing protocol. Subset = first 20 of the v3h 30-sample subset in its stratified selection
order (scripts/v3h/make_protocol.py select() before sorting: 18 law x action cells, then the per-law extras).

  .venv/bin/python scripts/v3j/make_protocol.py
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v3h'))
import make_protocol as H  # noqa: E402


def stratified_order():
    plan = json.loads((ROOT / 'docs/v3b/C1_SAMPLES.json').read_text())['samples']
    rng = np.random.default_rng(H.SEED); cells = {}
    for p in plan:
        cells.setdefault((p['law'], p['action']), []).append(p['index'])
    keys = sorted(cells); pick = {k: list(rng.permutation(cells[k])) for k in keys}
    chosen = [int(pick[k][0]) for k in keys]
    for law in sorted({k[0] for k in keys}):
        acts = [k for k in keys if k[0] == law and len(pick[k]) > 1]
        for j in rng.permutation(len(acts))[:4]:
            chosen.append(int(pick[acts[j]][1]))
    v3h = json.loads((ROOT / 'docs/v3h/PROTOCOL.json').read_text())['subset']['indices']
    assert sorted(chosen) == v3h
    return chosen


def main():
    order = stratified_order(); subset = order[:20]
    proto = {
        'created_utc': '2026-11-15', 'split': 'validation only (v2main/val); test untouched',
        'questions': {'exp1': 'Does the learned-prior advantage survive stronger uninformed (CMA-ES) search baselines?',
                      'exp2': 'Is the ~0.2-0.3 identifiable z-MAE floor caused by the frozen nuisance parameters (hardening, nu)? Does the prior advantage persist when they are searched?'},
        'subset': {'n': 20, 'indices': subset, 'v3h_stratified_order': order,
                   'rule': 'first 20 of the v3h 30-sample subset in its stratified selection order (18 law x action cells, then 2 jelly extras); '
                           'composition 8 jelly / 6 metal / 6 plasticine'},
        'common': {'objective': '(1 - mean object-mask IoU) + RGB-MAE(union)/255 vs observed 3-view video (scripts/v3d/c1_refine.py compare, simulate)',
                   'budgets_reported': [0, 4, 12, 24], 'nested': 'budget k = best observed loss among the first k simulations; k = 0 = start point (cma_best3: from k = 3)',
                   'failure_rule': 'as C1: quick crash (<60 s) retried once; failed sims loss 9, count toward the budget, never selected; no sample dropped',
                   'reuse': 'a simulation with exactly the same law and candidate parameters as one stored in data/v3b_refine/v2main_C_seed29 (prior_top1 / lawmean_gt / lawmean_best3) is reused and counted in the budget',
                   'code': 'scripts/v3j/c1j_refine.py', 'outputs': 'data/v3j_refine/<arm>/<index>/state.json'},
        'exp1_arms': {
            'cma_gtmean': 'PRIVILEGED GT law; start = GT-law train mean (as lawmean_gt); CMA-ES sigma0 = 1 decade',
            'cma_best3': '3 law-mean probes (sims 1-3), CMA-ES from the best probe (its law and law mean); sigma0 = 1 decade',
            'cma_prior': '9-run V-JEPA ensemble prediction, top-1 vote law (= prior_top1 start); CMA-ES sigma0 = 0.5 decade',
            'cma_details': 'own numpy (mu/mu_w, lambda)-CMA-ES (pycma not installed), Hansen tutorial default parameters, lambda = 4 (mu = 2), '
                           'search space log10(E/rho) [+ log10(sigma_y/rho) unless law = jelly], box = train range E [2, 6], sigma_y [0, 4.5] '
                           '(clip-repair, clipped points used in the update); start point evaluated as sim 1; generations of 4 evaluated in parallel; '
                           'rng seed [20261115, index, arm_id]; nu = arm nu, hardening/softening 0.25 (as C1)',
            'compared_with': 'existing compass arms on the same 20 samples: prior_top1, lawmean_gt, lawmean_best3 (data/v3b_refine)'},
        'exp2_arms': {
            'u4_prior': 'prior_top1 start (ensemble prediction, top-1 law, ensemble nu clipped to [0.10, 0.46], hardening 0.25)',
            'u4_gtmean': 'PRIVILEGED GT law, GT-law train mean (law-mean nu, hardening 0.25)',
            'optimizer': '4-D deterministic opportunistic compass search (C1 code path generalised): dims log10 E/rho, log10 sigma_y/rho '
                         '(bounds as C1 [1.5, 6.5], [-0.5, 5.0]), hardening [0.05, 0.45], nu [0.10, 0.46] (train ranges); jelly: E, nu. '
                         'initial steps 0.5, 0.5 decade, 0.1, 0.08; all halved after an unsuccessful poll; stop when the E step < 0.02. '
                         'Plasticine softening stays 0.25.',
            'metrics': 'identifiable z-MAE (E/rho, sigma_y/rho, as C1); |hardening error| (GT metal/plasticine samples with a non-jelly estimate), '
                       '|nu error| (raw); observed-fit IoU of the selected candidate; unseen-action replay IoU at budgets 0/4/12/24 '
                       '(swapped action ACTIONS[(i+3)%6], vs GT-parameter replay; skipped where the C1 GT replay failed); '
                       'frozen counterparts = prior_top1 / lawmean_gt (stored replays at their budgets)'},
        'metric': 'identifiable z-MAE: |z_hat - z_gt| on E/rho and sigma_y/rho (train-standardized ln1p specific; sigma_y only where defined), '
                  'pooled over (sample, target) elements; paired object-clustered bootstrap CIs (2000 draws, seed 17) vs prior_top1',
        'sims_to_target': 'target T = lawmean_gt (compass) pooled MAE at 24 sims on these 20 samples; smallest k (linear interpolation) at which '
                          'the arm pooled MAE <= T; censored if never',
        'scheduling': 'a800-4, 8 GPUs; budget 12 first for every arm, then extension to 24 (nested, resumed deterministically). If the projection '
                      'exceeds the 08:00 UTC deadline: drop budget 24 of cma_prior first, then report on the samples complete in all arms at 07:30 UTC.',
    }
    out = ROOT / 'docs/v3j/PROTOCOL.json'
    if out.exists():
        if json.loads(out.read_text())['subset'] != proto['subset']:
            raise SystemExit('protocol already registered with a different subset')
        print('protocol exists')
    else:
        out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(proto, indent=1) + '\n')
    base = '.venv/bin/python scripts/v3j/c1j_refine.py --index {i} --arm {a} --budget {b}'
    jobs = []
    for i in subset:  # Exp 2 searches: long sequential chains, start first
        for a in ('u4_prior', 'u4_gtmean'):
            jobs.append({'id': f'v3j_{a}_{i:06d}', 'cmd': base.format(i=i, a=a, b=24), 'slots': 1})
    for i in subset:
        for a in ('u4_prior', 'u4_gtmean'):
            jobs.append({'id': f'v3j_{a}_{i:06d}_replay', 'cmd': base.format(i=i, a=a, b=24) + ' --replay', 'slots': 4,
                         'after': [f'v3j_{a}_{i:06d}']})
    for i in subset:
        for a in ('cma_prior', 'cma_gtmean', 'cma_best3'):
            jobs.append({'id': f'v3j_{a}_{i:06d}_b12', 'cmd': base.format(i=i, a=a, b=12), 'slots': 4})
    for i in subset:
        for a in ('cma_gtmean', 'cma_best3'):
            jobs.append({'id': f'v3j_{a}_{i:06d}_b24', 'cmd': base.format(i=i, a=a, b=24), 'slots': 4, 'after': [f'v3j_{a}_{i:06d}_b12']})
    for i in subset:
        jobs.append({'id': f'v3j_cma_prior_{i:06d}_b24', 'cmd': base.format(i=i, a='cma_prior', b=24), 'slots': 4,
                     'after': [f'v3j_cma_prior_{i:06d}_b12']})
    (ROOT / 'docs/v3j/jobs.json').write_text(json.dumps(jobs, indent=1) + '\n')
    print(subset, len(jobs))


if __name__ == '__main__':
    main()
