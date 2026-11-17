"""F8: vision + touch + action prior on the gripper-held variant (V5b). Reads docs/v5m/RESULTS.json (test split,
ensembles; falls back to the per-seed entry if no ensemble exists). Writes docs/paper_figs/F8_touch.{pdf,png}.
  .engine-envs/plot/bin/python scripts/paper_figs/make_fig_touch.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/paper_figs'
plt.rcParams.update({
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
    'ytick.labelsize': 7, 'legend.fontsize': 6.5, 'axes.linewidth': 0.6, 'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 2.5, 'ytick.major.size': 2.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'legend.frameon': False, 'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300, 'axes.grid': True,
    'grid.color': '#e3e3e3', 'grid.linewidth': 0.5, 'axes.axisbelow': True, 'errorbar.capsize': 1.2})
OI = dict(blue='#0072B2', sky='#56B4E9', green='#009E73', verm='#D55E00', orange='#E69F00', black='#000000')
ARMS = [('vision', 'Vision', OI['blue']), ('vision_action', 'Vision + action', OI['sky']),
        ('touch_action', 'Touch + action', OI['green']), ('vision_touch_action', 'Vision + touch + action', OI['verm'])]
TG = [('E_over_rho', r'$E/\rho$'), ('sigma_y_over_rho', r'$\sigma_y/\rho$'), ('rho', r'$\rho$'), ('E', r'$E$')]

R = json.loads((ROOT / 'docs/v5m/RESULTS.json').read_text())['splits']['test']['arms']


def entry(arm):
    for k in (f'{arm} ensemble',) + tuple(k for k in R if k.startswith(arm + ' (')):
        if k in R:
            return R[k]
    raise KeyError(arm)


fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.25, 3.05), gridspec_kw=dict(height_ratios=[1.15, 1], hspace=0.62))
w = 0.19; x = np.arange(len(TG))
for j, (arm, lab, c) in enumerate(ARMS):
    e = entry(arm); p = e['point']; ci = e.get('ci95', {})
    g = [p[f'gain_{t}'] for t, _ in TG]
    lo = [g[i] - ci.get(f'gain_{t}', [g[i], g[i]])[0] for i, (t, _) in enumerate(TG)]
    hi = [ci.get(f'gain_{t}', [g[i], g[i]])[1] - g[i] for i, (t, _) in enumerate(TG)]
    a1.bar(x + (j - 1.5) * w, g, w, color=c, label=lab, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], error_kw=dict(lw=0.6))
p0 = entry('vision')['point']
for i, (t, _) in enumerate(TG):
    og = 1 - p0[f'oracle_{t}'] / p0[f'const_{t}']
    a1.plot([i - 2.1 * w, i + 2.1 * w], [og, og], color=OI['black'], lw=0.9, ls=(0, (3, 2)))
a1.plot([], [], color=OI['black'], lw=0.9, ls=(0, (3, 2)), label='True-class mean (oracle)')
a1.set_xticks(x); a1.set_xticklabels([l for _, l in TG]); a1.set_ylabel('Error reduction\nvs. constant')
a1.set_ylim(0, 0.62); a1.legend(ncol=2, loc='lower center', bbox_to_anchor=(0.5, 1.07), handlelength=1.2, columnspacing=0.8)
a1.text(-0.02, 1.02, '(a)', transform=a1.transAxes, ha='right', va='bottom', fontsize=8, fontweight='bold')
RS = [('rho', r'$\rho$'), ('E', r'$E$'), ('E_over_rho', r'$E/\rho$')]
x2 = np.arange(len(RS))
for j, (arm, lab, c) in enumerate(ARMS):
    e = entry(arm); p = e['point']; ci = e.get('ci95', {})
    v = [p[f'rs_{t}'] for t, _ in RS]
    lo = [v[i] - ci.get(f'rs_{t}', [v[i], v[i]])[0] for i, (t, _) in enumerate(RS)]
    hi = [ci.get(f'rs_{t}', [v[i], v[i]])[1] - v[i] for i, (t, _) in enumerate(RS)]
    a2.bar(x2 + (j - 1.5) * w, v, w, color=c, yerr=[np.maximum(lo, 0), np.maximum(hi, 0)], error_kw=dict(lw=0.6))
a2.set_xticks(x2); a2.set_xticklabels([l for _, l in RS]); a2.set_ylabel('Within-class\nSpearman'); a2.set_ylim(0, 0.9)
a2.text(-0.02, 1.02, '(b)', transform=a2.transAxes, ha='right', va='bottom', fontsize=8, fontweight='bold')
for ext in ('pdf', 'png'):
    fig.savefig(OUT / f'F8_touch.{ext}', bbox_inches='tight', pad_inches=0.02)
print('wrote F8_touch', {a: entry(a)['members'] if 'members' in entry(a) else 'seed' for a, _, _ in ARMS})
