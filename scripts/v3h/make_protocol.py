"""V3h: pre-register the prior-quality experiment (validation only) -> docs/v3h/PROTOCOL.json + docs/v3h/jobs.json.

Question: is the C1 speed-up specific to the learned V-JEPA prior, or would any reasonable initialization do, and
how does the benefit scale with prior (initialization) quality?  Same refinement as scripts/v3b/c1_refine.py
(objective, compass search, step 0.5 decade halved after a failed poll, failure rule, nested budgets), MAP-law start,
no law probes, budget 12, on a fixed stratified 30-sample subset of the 60 C1 val samples.

  .venv/bin/python scripts/v3h/make_protocol.py        (idempotent; refuses to change an existing protocol)
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SEED = 20261114
SIGMAS = [0.25, 0.5, 1.0]


def select(plan):
    rng = np.random.default_rng(SEED)
    cells = {}
    for p in plan:
        cells.setdefault((p['law'], p['action']), []).append(p['index'])
    keys = sorted(cells)
    pick = {k: list(rng.permutation(cells[k])) for k in keys}
    chosen = [int(pick[k][0]) for k in keys]                       # 1 per law x action cell (18)
    for law in sorted({k[0] for k in keys}):                         # + 4 per law from 4 random actions (12)
        acts = [k for k in keys if k[0] == law and len(pick[k]) > 1]
        for j in rng.permutation(len(acts))[:4]:
            chosen.append(int(pick[acts[j]][1]))
    return sorted(chosen)


def main():
    out = ROOT / 'docs/v3h/PROTOCOL.json'
    plan = json.loads((ROOT / 'docs/v3b/C1_SAMPLES.json').read_text())['samples']
    subset = select(plan)
    rng = np.random.default_rng(SEED + 1)
    eps = {str(i): [round(float(v), 6) for v in rng.standard_normal(2)] for i in subset}
    proto = {
        'created_utc': '2026-11-14', 'split': 'validation only (v2main/val); test untouched',
        'question': 'Is the C1 simulation saving specific to the learned V-JEPA prior, and how does it scale with initialization quality?',
        'subset': {'n': len(subset), 'indices': subset, 'source': 'docs/v3b/C1_SAMPLES.json (60)', 'seed': SEED,
                   'rule': 'per (GT law, observed action) cell one random sample (18) + per law a second sample from 4 random actions (12)'},
        'refinement': {'code': 'scripts/v3h/c1h_refine.py (imports simulate/compare/candidate/encode/decode/law_means from scripts/v3b/c1_refine.py)',
                       'objective': '(1 - mean object-mask IoU) + RGB-MAE(union)/255 vs observed 3-view 128-frame video',
                       'optimizer': 'deterministic opportunistic compass search over log10(E/rho) [+ log10(sigma_y/rho) unless law=jelly], '
                                    'initial step 0.5 decade, halved after an unsuccessful poll; bounds E [1.5,6.5], sigma_y [-0.5,5.0]',
                       'law': 'fixed at the arm\'s start law (MAP / GT); no law probes', 'budget': 12,
                       'budgets_reported': list(range(13)), 'nested': 'budget k = best observed loss among the first k simulations; k=0 = start point',
                       'failure_rule': 'quick crash (<60 s) retried once; failed sims get loss 9, count toward the budget, never selected; no sample dropped',
                       'fixed': 'hardening/softening 0.25; rho irrelevant (density 1000)', 'replay': 'not run (accuracy only)'},
        'metric': 'identifiable z-MAE: |z_hat - z_gt| on E/rho and sigma_y/rho (train-standardized ln1p specific; sigma_y only where defined), '
                  'pooled over (sample, target) elements; CIs = object-clustered bootstrap (2000 draws, seed 17)',
        'arms_new_primary': {
            'dino_c': 'seed ensemble of runs/v3c_dino_C_seed{17,29,43} (best val epoch): mean prediction_z, majority-vote law '
                      '(1/1/1 tie -> law whose train-mean E/rho z is closest to the mean prediction); nu = decoded ensemble nu',
            **{f'gtnoise_{s:g}': f'PRIVILEGED: GT law; x0 = clip(log10 GT (E/rho, sigma_y/rho) + {s:g} * eps), eps ~ N(0, I2) one fixed '
                                 'draw per sample (listed below; the same eps is scaled for every level); nu = train mean nu of the GT law '
                                 '(as lawmean_gt)' for s in SIGMAS}},
        'arms_new_secondary': {'ens_map': 'run only if time permits, after all primary jobs: 9-run B/C/D V-JEPA ensemble (docs/v3b/ENSEMBLE_BCD9_VAL.json) '
                                          'mean prediction, top-1 vote law, NO probes (same protocol as dino_c)'},
        'arms_reused': {'prior': 'data/v3b_refine/v2main_C_seed29/<idx>/state.json (single V-JEPA C seed29, MAP law, no probes)',
                        'prior_top2': '.../prior_top2 (9-run ensemble, 2 law probes counted in the budget)',
                        'lawmean_gt': '.../lawmean_gt (PRIVILEGED GT law, train mean of that law)',
                        'lawmean_best3': '.../lawmean_best3 (3 law-mean probes counted; first point at k=3)'},
        'noise_eps': eps,
        'analyses': ['per arm: identifiable MAE at budgets 0/4/12 (and full 0..12 curve) on the 30-sample subset; final error vs initial (budget-0) error',
                     'sims to reach target T = lawmean_gt pooled MAE at 24 sims on the same subset (also T12 = lawmean_gt at 12), '
                     'linear interpolation in k on the 0..12 (reused arms: 0..24) curve; censored if not reached; object bootstrap CI',
                     'paired differences vs lawmean_gt at the same budget (object bootstrap)',
                     'breakdown of the existing 60-sample C1 results by observed action and by GT law (prior, prior_top2, lawmean_gt, lawmean_best3)'],
    }
    if out.exists():
        old = json.loads(out.read_text())
        if old['subset'] != proto['subset'] or old['noise_eps'] != proto['noise_eps']:
            raise SystemExit('protocol already registered with different subset/noise; refusing to overwrite')
        print('protocol unchanged'); return
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(proto, indent=1) + '\n')
    base = '.venv/bin/python scripts/v3h/c1h_refine.py --index {i} --arm {a} --budget 12'
    jobs = []
    primary = ['dino_c'] + [f'gtnoise_{s:g}' for s in SIGMAS]
    for i in subset:  # interleave by sample so a balanced common set accumulates early
        for a in primary:
            jobs.append({'id': f'v3h_{a}_{i:06d}', 'cmd': base.format(i=i, a=a)})
    for i in subset:
        jobs.append({'id': f'v3h_ens_map_{i:06d}', 'cmd': base.format(i=i, a='ens_map')})
    (ROOT / 'docs/v3h/jobs.json').write_text(json.dumps(jobs, indent=1) + '\n')
    print(len(subset), subset, len(jobs))


if __name__ == '__main__':
    main()
