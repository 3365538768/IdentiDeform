"""V3j: render the RESULTS.json tables as Markdown (pasted into docs/v3j/RESULTS_ZH.md).

  .venv/bin/python scripts/v3j/tables_md.py > docs/v3j/tables.md
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v3j'))
from report import LABEL  # noqa: E402

R = json.loads((ROOT / 'docs/v3j/RESULTS.json').read_text())
f = lambda v, d=3: '-' if v is None else f'{v:.{d}f}'
ci = lambda c: '-' if not c else f'[{c[0]:+.3f}, {c[1]:+.3f}]'


def mae_table(key):
    T = R[key]; arms = list(T['24']['arms']) if T['24']['arms'] else list(T['12']['arms'])
    arms = sorted({a for row in T.values() for a in row['arms']}, key=lambda a: list(LABEL).index(a))
    print('| arm | ' + ' | '.join(f"k={k} (n={T[k]['n_samples']})" for k in T) + ' |')
    print('|---|' + '---|' * len(T))
    for a in arms:
        cells = []
        for k, row in T.items():
            d = row['arms'].get(a)
            cells.append('-' if d is None or d['mae'] is None else f"{d['mae']:.3f}" + (f" (Δ {d['diff_vs_prior_top1']:+.3f} {ci(d['diff_ci'])})" if 'diff_vs_prior_top1' in d else ''))
        print(f'| {LABEL[a]} | ' + ' | '.join(cells) + ' |')
    print()


def stt(key):
    T = R.get(key)
    if not T or 'arms' not in T:
        return
    print(f"target T = {T['target']:.3f} (n = {T['n_samples']})\n")
    print('| arm | sims to T | 95% CI (censored at 25) | P(reached) |\n|---|---|---|---|')
    for a, d in T['arms'].items():
        print(f"| {LABEL[a]} | {f(d['k_star'], 1) if d['k_star'] is not None else '>24'} | [{d['ci_censored_at_25'][0]:.1f}, {d['ci_censored_at_25'][1]:.1f}] | {d['frac_draws_reached']:.2f} |")
    print()


def nuis():
    N = R['exp2_nuisance']
    print('| arm | k | n | ident z-MAE | abs hardening err | abs nu err | observed-fit IoU | replay IoU (n) |\n|---|---|---|---|---|---|---|---|')
    for k, row in N.items():
        for a, d in row['arms'].items():
            m = R['exp2_table'][k]['arms'][a]['mae']
            print(f"| {LABEL[a]} | {k} | {row['n_samples']} | {f(m)} | {f(d['hardening_abs_err'])} (n={d['n_h']}) | {f(d['nu_abs_err'])} | {f(d['observed_fit_iou'])} | {f(d['replay_iou'])} ({d['n_replay']}) |")
    print()


print('## Exp1 identifiable z-MAE\n'); mae_table('exp1_table')
if 'exp1_table_24_available_arms' in R:
    print('## Exp1 (arms with budget 24)\n'); mae_table('exp1_table_24_available_arms')
print('## Exp1 sims-to-target\n'); stt('sims_to_target_exp1'); stt('sims_to_target_exp1_24')
print('## Exp2 identifiable z-MAE\n'); mae_table('exp2_table')
print('## Exp2 sims-to-target\n'); stt('sims_to_target_exp2')
print('## Exp2 nuisance / fit / replay\n'); nuis()
print('## Coverage / failures\n')
print('| arm | samples | with k=24 | sims (reused) | failed sims | samples with failures |\n|---|---|---|---|---|---|')
for a, c in R['coverage'].items():
    print(f"| {LABEL[a]} | {c['samples']} | {c['budget24']} | {c['sims_total']} ({c['sims_reused']}) | {c['failed_sims']} | {c['samples_with_failure']} |")
