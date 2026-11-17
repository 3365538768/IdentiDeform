"""V3v aggregation: data/v3v/eval/<obj>.json -> docs/v3v/RESULTS.json + docs/v3v/table.tex (+ markdown tables on stdout).
  .venv/bin/python -I scripts/v3v/aggregate.py
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / 'data/v3v'; OUT = ROOT / 'docs/v3v'
OBJS = ['bun', 'burger', 'dog', 'pig', 'potato']; CAMS = ['C0733', 'C0787', 'C0801']
ARMS = ['prior', 'prior_refine', 'class_mean', 'global_constant_loo', 'oracle', 'rigid_track_2cam', 'persist_last', 'prior_refine_damp', 'prior_refine_damp_loo', 'oracle_damp']
BLOCKS = {'heldout_camera': ['iou', 'psnr', 'ssim'], 'heldout_camera_post_hit': ['iou', 'psnr', 'ssim'],
          'future_frames': ['iou', 'psnr', 'ssim'], 'future_3d': ['centroid_err_cm', 'lowest_err_cm', 'chamfer_cm'],
          'all_3d': ['centroid_err_cm', 'lowest_err_cm', 'chamfer_cm']}
RNG = np.random.default_rng(0)
IDX = RNG.integers(0, 5, size=(10000, 5))


def stats(v):
    v = np.asarray(v, float); bs = v[IDX].mean(1)
    loo = [np.delete(v, i).mean() for i in range(len(v))]
    return {'mean': float(v.mean()), 'ci95_bootstrap_objects': [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
            'loo_mean_range': [float(min(loo)), float(max(loo))], 'n_objects': len(v), 'per_object': dict(zip(OBJS, v.tolist()))}


E = {o: json.loads((D / 'eval' / f'{o}.json').read_text()) for o in OBJS}
REF = {o: json.loads((D / 'sims' / o / 'refine.json').read_text()) for o in OBJS}
per = {}
for o in OBJS:
    per[o] = {}
    for arm in ARMS:
        per[o][arm] = {}
        for b, ms in BLOCKS.items():
            vals = [E[o]['folds'][h]['arms'].get(arm, {}).get(b) for h in CAMS]
            if any(v is None for v in vals):
                continue
            per[o][arm][b] = {m: float(np.mean([v[m] for v in vals])) for m in ms}
        lgs = [E[o]['folds'][h]['arms'].get(arm, {}).get('lg') for h in CAMS]
        rp_ = [E[o]['folds'][h]['arms'].get(arm, {}).get('rpic') for h in CAMS]
        if rp_[0] is not None:
            per[o][arm]['rpic_damping_per_fold'] = dict(zip(CAMS, rp_))
        if lgs[0] is not None:
            per[o][arm]['log10_E_over_rho_x1000_per_fold'] = dict(zip(CAMS, lgs))
summary, diffs = {}, {}
for arm in ARMS:
    summary[arm] = {}
    for b, ms in BLOCKS.items():
        if all(b in per[o][arm] for o in OBJS):
            summary[arm][b] = {m: stats([per[o][arm][b][m] for o in OBJS]) for m in ms}
for arm in ARMS:
    if arm == 'prior':
        continue
    diffs[f'{arm} - prior'] = {}
    for b, ms in BLOCKS.items():
        if all(b in per[o][arm] and b in per[o]['prior'] for o in OBJS):
            diffs[f'{arm} - prior'][b] = {m: stats([per[o][arm][b][m] - per[o]['prior'][b][m] for o in OBJS]) for m in ms}
for a1, a2 in [('prior_refine', 'class_mean'), ('prior_refine', 'global_constant_loo'), ('prior', 'class_mean'),
               ('prior', 'global_constant_loo'), ('prior_refine', 'rigid_track_2cam'), ('prior_refine', 'persist_last'), ('oracle', 'prior_refine'),
               ('prior_refine_damp', 'prior_refine'), ('prior', 'persist_last'), ('prior', 'rigid_track_2cam'), ('prior_refine_damp', 'persist_last'),
               ('oracle', 'prior'), ('prior_refine_damp_loo', 'prior'), ('prior_refine_damp_loo', 'persist_last'),
               ('prior_refine_damp_loo', 'prior_refine'), ('prior_refine_damp_loo', 'class_mean'), ('prior_refine_damp_loo', 'global_constant_loo'),
               ('prior_refine_damp_loo', 'prior_refine_damp'), ('prior_refine_damp_loo', 'rigid_track_2cam')]:
    diffs[f'{a1} - {a2}'] = {}
    for b, ms in BLOCKS.items():
        if all(b in per[o][a1] and b in per[o][a2] for o in OBJS):
            diffs[f'{a1} - {a2}'][b] = {m: stats([per[o][a1][b][m] - per[o][a2][b][m] for o in OBJS]) for m in ms}
sims = {o: {'prior_log10E': REF[o]['prior_lg'], 'refined_log10E_per_fold': {h: REF[o]['folds'][h]['lg'] for h in CAMS},
            'refine_sims_per_fold': {h: REF[o]['folds'][h]['n_sims'] for h in CAMS},
            'refine_unique_sims_union': REF[o]['refine_unique_sims_union_over_folds'], 'all_sims_total': REF[o]['unique_sims_total'],
            'oracle_log10E': E[o]['oracle_lg'], 'frame0_iou_ceiling': E[o]['frame0_iou_ceiling'],
            'color_visible_frac': {h: E[o]['folds'][h]['color_visible_frac'] for h in CAMS},
            'global_constant_loo_log10E_per_fold': {h: E[o]['folds'][h]['arms']['global_constant_loo']['lg'] for h in CAMS}} for o in OBJS}
prot = json.loads((OUT / 'PROTOCOL.json').read_text())
res = {'stage': 'v3v', 'protocol': 'docs/v3v/PROTOCOL.json', 'n_objects': 5,
       'note': 'n = 5 objects (one drop each); CIs are percentile bootstrap over objects (10000 resamples) and are wide; '
               'per-object values are averages over the 3 held-out-camera folds.',
       'arms': list(prot['arms'].keys()), 'per_object': per, 'summary': summary, 'paired_differences': diffs, 'twins': sims,
       'deviations': json.loads((OUT / 'deviations.json').read_text()) if (OUT / 'deviations.json').exists() else []}
(OUT / 'RESULTS.json').write_text(json.dumps(res, indent=1))

# ---- markdown dump + LaTeX table
NAMES = {'prior': 'Prior only (0 sims)', 'prior_refine': 'Prior + refinement ($\\le$12 sims)', 'class_mean': 'Class-mean $E$',
         'global_constant_loo': 'Global constant $E$ (LOO)', 'oracle': 'Oracle $E$ (all frames/cams)',
         'rigid_track_2cam': 'Rigid hull on 2-view track$^\\dagger$', 'persist_last': 'Hold last fit frame$^\\dagger$',
         'prior_refine_damp': 'Prior + refine $E$, damping$^\\ast$', 'oracle_damp': 'Oracle $E$, damping$^\\ast$',
         'prior_refine_damp_loo': 'Prior + refine $E$, LOO damping$^\\ast$'}


def cell(arm, b, m, fmt):
    s = summary[arm].get(b, {}).get(m)
    return '--' if s is None else fmt.format(s['mean'])


for b, ms in BLOCKS.items():
    print(f'\n### {b}')
    print('| arm | ' + ' | '.join(ms) + ' |'); print('|---' * (len(ms) + 1) + '|')
    for arm in ARMS:
        if b in summary[arm]:
            print(f'| {arm} | ' + ' | '.join(f"{summary[arm][b][m]['mean']:.3f} [{summary[arm][b][m]['ci95_bootstrap_objects'][0]:.3f}, {summary[arm][b][m]['ci95_bootstrap_objects'][1]:.3f}]" for m in ms) + ' |')
print('\n### per object (heldout IoU / future IoU / future PSNR / future centroid cm)')
for o in OBJS:
    print(o, {arm: [round(per[o][arm].get(b, {}).get(m, float('nan')), 3) for b, m in
                    [('heldout_camera', 'iou'), ('future_frames', 'iou'), ('future_frames', 'psnr'), ('future_3d', 'centroid_err_cm')]] for arm in ARMS})

rows = []
for arm in ARMS:
    rows.append(f"{NAMES[arm]} & {cell(arm, 'heldout_camera', 'iou', '{:.3f}')} & {cell(arm, 'heldout_camera', 'psnr', '{:.2f}')} & "
                f"{cell(arm, 'heldout_camera', 'ssim', '{:.3f}')} & {cell(arm, 'future_frames', 'iou', '{:.3f}')} & "
                f"{cell(arm, 'future_frames', 'psnr', '{:.2f}')} & {cell(arm, 'future_frames', 'ssim', '{:.3f}')} & "
                f"{cell(arm, 'future_3d', 'centroid_err_cm', '{:.2f}')} & {cell(arm, 'future_3d', 'chamfer_cm', '{:.2f}')} \\\\")
rows.insert(5, '\\midrule'); rows.insert(8, '\\midrule')
tex = r"""% auto-generated by scripts/v3v/aggregate.py from docs/v3v/RESULTS.json
\begin{table*}[t]
\centering
\caption{\textbf{Real-video replay on the Spring-Gaus real captures} (5 objects, one drop each, 3 calibrated cameras, 119.88\,fps, 20 frames).
Twins are re-simulated in our MPM engine from a 3-view visual hull and fitted only on 2 cameras $\times$ the first 10 frames.
\emph{Held-out camera}: the third camera, frames 1--19; \emph{future}: frames 10--19, all cameras; both averaged over the 3 camera folds.
PSNR/SSIM are computed inside the union of observed and simulated masks (object on black); 3D errors are against the triangulated mask
centroid and the per-frame 3-view visual hull (a 3D proxy). Values are means over $n{=}5$ objects; with $n{=}5$ the bootstrap 95\% CIs in
the supplement are wide. $^\dagger$Non-physical references that use observations of the evaluated frames (rigid track) or no dynamics (hold).
$^\ast$Post hoc (added after the pre-registered results): one extra strain-rate damping parameter (RPIC) fitted with the same 12-simulation budget and fit set.}
\label{tab:real_replay}
\small
\setlength{\tabcolsep}{4pt}
\begin{tabular}{@{}lcccccccc@{}}
\toprule
 & \multicolumn{3}{c}{Held-out camera} & \multicolumn{3}{c}{Future frames} & \multicolumn{2}{c}{Future 3D (cm) $\downarrow$} \\
\cmidrule(lr){2-4}\cmidrule(lr){5-7}\cmidrule(l){8-9}
Twin & IoU $\uparrow$ & PSNR $\uparrow$ & SSIM $\uparrow$ & IoU $\uparrow$ & PSNR $\uparrow$ & SSIM $\uparrow$ & Centroid & Chamfer \\
\midrule
""" + '\n'.join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
(OUT / 'table.tex').write_text(tex)
print('\nwrote', OUT / 'RESULTS.json', OUT / 'table.tex')
