"""V3l: assemble docs/v3l/RESULTS.json (+ markdown tables on stdout) from data/v3l/** and the v3g official outputs.
  .venv/bin/python -I scripts/v3l/make_results.py
"""
import json, math
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PRIMARY = ['torus', 'duck', 'apple']
EXTRA = ['C', 'E', 'V', 'banana', 'chess', 'cream', 'cross', 'doraemon', 'droplet', 'rope', 'toothpaste']
KEYS = ['fit_cd_mean', 'future_cd_mean', 'fit_cd2_x1e3_mean', 'future_cd2_x1e3_mean', 'fit_track_l2_mean', 'future_track_l2_mean',
        'fit_psnr_mean', 'future_psnr_mean', 'fit_ssim_mean', 'future_ssim_mean', 'fit_mask_iou_mean', 'future_mask_iou_mean',
        'future_mask_iou_exact_splat_mean']


def rd(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def arm_row(st, tag='final'):
    if not st or tag not in st.get('rollouts', {}):
        return None
    r = st['rollouts'][tag]
    if r.get('failed'):
        return {'log10E': r['log10E'], 'failed': True}
    row = {'log10E': r['log10E'], 'nu': r.get('nu'), 'abs_dlog10E': r.get('abs_dlog10E'), 'rollout_s': r.get('seconds')}
    row.update({k: r['summary'].get(k) for k in KEYS})
    return row


def main():
    prior = rd(ROOT / 'data/v3l/prior20.json') or {}
    sgo = rd(ROOT / 'docs/v3g/SPRINGGAUS_OFFICIAL.json')['scenes']
    sgi = rd(ROOT / 'docs/v3l/SG_OFFICIAL_IMAGES.json') or {}
    pt = prior.get('timing_s', {})
    n_sc = len(prior.get('scenes', {})) or 14
    res = {'protocol': 'docs/v3l/PROTOCOL.json', 'summary_zh': 'docs/v3l/RESULTS_ZH.md', 'figure': 'docs/v3l/cd_per_frame.png', 'caveats': [
        'Part 1 geometry is PRIVILEGED: our MPM starts from the dataset frame-0 GT particles; Spring-Gaus reconstructs geometry from images. Early fit-frame CD is at the metric noise floor for us by construction.',
        'Different simulators: ours = MPM jelly (fixed corotated), rho 1000 assumed, nu from the prior; Spring-Gaus = spring-mass; data = a third MPM. GT-E replay (gtE_DIAGNOSTIC) shows the engine-mismatch floor.',
        'Views: ours 3 (t0 triplet [2,3,8]); Spring-Gaus all 10. Eval cam 5 is unseen for us and a training view for Spring-Gaus.',
        'Renderer: ours = unshaded particle-colour disk splats (ceiling PSNR 25.6/19.7/29.4 with GT particles), so PSNR gaps are largely renderer gaps.',
        'CD metric (v3g sg_eval, 8192-pt subsample) has a noise floor of 0.0052/0.0090/0.0076 (torus/duck/apple) even for GT particles.',
        'prior20 and the refinement use frames 0-19 only; prior30_v3f saw all 30 frames (reference only). Spring-Gaus image metrics for future frames use nearest-time 60 Hz frames (|dt|<=6.7 ms).',
        'Wall-clock: shared A800s at ~100% util from other jobs; Spring-Gaus times are from v3g (upper bounds); our prior forward excludes ~4 s one-off model loading; final 30-frame rollout (8-38 s) excluded from wall_s_estimator.',
        'Part 2: the no-physics rigid_follow baseline (object translated with the hand centroid) beats both PhysTwin and our MPM on test chamfer/track; the case is dominated by rigid sliding, so E is weakly identifiable and our refined E (~7e6 Pa, near the bound) means \'as stiff as possible\', not a material measurement.'],
        'part1': {}, 'part2': None}
    for s in PRIMARY + EXTRA:
        D = ROOT / 'data/v3l/sg' / s
        st = {a: rd(D / f'{a}.json') for a in ('prior20', 'prior20_refine', 'prior30_v3f', 'gtE')}
        any_st = next((v for v in st.values() if v), None)
        e = {'gt_log10E': any_st.get('gt_log10E') if any_st else None, 'arms': {}}
        # forward-pass cost of the prior (per scene): decode + 20-frame encode + share of 9-run heads + amortised model loads excluded
        fwd = None
        if pt:
            fwd = pt.get(f'{s}_decode_s', 0) + pt.get(f'{s}_n20_encode_s', 0) + pt.get('heads_9runs_total_all_samples', 0) / (2 * n_sc)
        e['prior_forward_s'] = fwd
        if st['prior20']:
            e['arms']['prior20'] = arm_row(st['prior20']);
            if e['arms']['prior20']:
                e['arms']['prior20']['wall_s_estimator'] = fwd
        if st['prior20_refine']:
            R = st['prior20_refine']
            for k in ('12', '4'):
                tag = 'final' if k == '12' else f'budget{k}'
                row = arm_row(R, tag)
                if row:
                    n = min(int(k), len(R['evals']))
                    row['sims'] = n
                    row['search_s'] = float(sum(x['seconds'] for x in R['evals'][:n]))
                    row['wall_s_estimator'] = (fwd or 0) + row['search_s']
                    row['wall_s_incl_final_rollout'] = row['wall_s_estimator'] + (row.get('rollout_s') or 0)
                    e['arms'][f'prior20_refine_b{k}'] = row
            e['refine_trace'] = [{'n': x['n'], 'log10E': x['x'], 'loss_0_19': x['loss'], 's': round(x['seconds'], 1)} for x in R['evals']]
        if st['prior30_v3f']:
            e['arms']['prior30_v3f_REFERENCE_saw_future'] = arm_row(st['prior30_v3f'])
        if st['gtE']:
            e['arms']['gtE_DIAGNOSTIC'] = arm_row(st['gtE'])
        if any_st and 'static_baseline' in any_st:
            e['static_baseline'] = {k: any_st['static_baseline'].get(k) for k in KEYS}
            e['renderer_ceiling_gt_particles'] = {k: any_st['renderer_ceiling'].get(k) for k in KEYS}
            e['splat_radius_512'] = any_st.get('splat_radius_hi')
        v3k = rd(ROOT / 'data/v3k/sg' / s / 'prior.json')
        if v3k and v3k.get('budgets'):
            e['v3k_all_frames_refine_REFERENCE'] = {k: v3k['budgets'][k]['log10E'] for k in ('0', '4', '12') if k in v3k['budgets']}
        if s in sgo:
            m = sgo[s].get('metrics', {})
            e['springgaus_official'] = {'fit_cd_mean': m.get('fit_cd_mean'), 'future_cd_mean': m.get('future_cd_mean'),
                                        'fit_cd2_x1e3_mean': m.get('fit_cd2_x1e3_mean'), 'future_cd2_x1e3_mean': m.get('future_cd2_x1e3_mean'),
                                        'future_cd_static_baseline_mean_v3g': m.get('future_cd_static_baseline_mean'),
                                        'wall_clock_total_s': sgo[s].get('wall_clock_total_s'),
                                        'log10_k_median': sgo[s].get('learned_params', {}).get('log10_k_median'),
                                        **(sgi.get(s, {}).get('summary', {}))}
        res['part1'][s] = e
    # aggregate over primary and all scenes
    agg = {}
    for name, scenes in (('primary3', PRIMARY), ('all14', PRIMARY + EXTRA)):
        for arm in ('prior20', 'prior20_refine_b4', 'prior20_refine_b12', 'prior30_v3f_REFERENCE_saw_future', 'gtE_DIAGNOSTIC'):
            rows = [res['part1'][s]['arms'].get(arm) for s in scenes]
            if all(r and not r.get('failed') for r in rows):
                agg.setdefault(name, {})[arm] = {k: float(np.mean([r[k] for r in rows])) for k in KEYS + ['abs_dlog10E'] if all(r.get(k) is not None for r in rows)}
        rows = [res['part1'][s].get('static_baseline') for s in scenes]
        if all(rows):
            agg.setdefault(name, {})['static_baseline'] = {k: float(np.mean([r[k] for r in rows])) for k in KEYS if all(r.get(k) is not None for r in rows)}
    res['part1_mean'] = agg
    # part 2
    P = ROOT / 'data/v3l/pt'
    p2 = {'official_phystwin': {'chamfer_train': 0.015509363132245515, 'chamfer_test': 0.037271304499535335,
                                'track_train': 0.027187777268687124, 'track_test': 0.08916416602987824, 'wall_s_physics': 1876.9,
                                'source': 'docs/v3g/PHYSTWIN_OFFICIAL.json (re-evaluated with scripts/v3l/pt_eval.py: identical)'}}
    for tag in ('static', 'rigid_follow', 'mpm_prior_b0', 'mpm_prior_b12', 'mpm_jellymean_b0', 'mpm_jellymean_b12'):
        r = rd(P / f'eval_{tag}.json')
        if r:
            p2[tag] = {k: v for k, v in r.items() if k != 'chamfer_per_frame'}
    pp = rd(P / 'pt_prior.json')
    if pp:
        p2['prior_on_rgb'] = {k: pp.get(k) for k in ('log10_E_Pa_at_rho1000', 'nu', 'per_run_log10E', 'law_votes', 'seconds_model', 'n_frames')}
    for start in ('prior', 'jellymean'):
        srch = rd(P / f'search_{start}.json')
        if srch:
            d = {k: srch.get(k) for k in ('n_tracer', 'n_filler', 'n_driven', 'dx', 'r_ctrl', 'x0', 'x0_source', 'best', 'search_seconds', 'complete')}
            d['trace'] = [{'n': x['n'], 'log10E': x['x'], 'train_chamfer': x['loss'], 's': round(x['seconds'], 1), 'failed': x.get('failed', False)} for x in srch.get('evals', [])]
            p2[f'search_{start}'] = d
    smoke = rd(P / 'smoke.json')
    if smoke:
        p2['smoke'] = smoke.get('smoke')
    res['part2'] = p2
    (ROOT / 'docs/v3l/RESULTS.json').write_text(json.dumps(res, indent=1))
    # markdown tables
    f = lambda v, n=4: '–' if v is None else (f'{v:.{n}f}' if isinstance(v, float) else str(v))
    print('| scene | method | log10E (GT) | fit CD 0-19 | future CD 20-29 | future PSNR | future SSIM | future IoU | fit PSNR | wall |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for s in PRIMARY + EXTRA:
        e = res['part1'][s]
        sg = e.get('springgaus_official')
        if sg:
            print(f"| {s} | Spring-Gaus official | – ({f(e['gt_log10E'],2)}) | {f(sg['fit_cd_mean'])} | {f(sg['future_cd_mean'])} | {f(sg.get('future_psnr_mean'),2)} | {f(sg.get('future_ssim_mean'),3)} | {f(sg.get('future_mask_iou_mean'),3)} | {f(sg.get('fit_psnr_mean'),2)} | {sg['wall_clock_total_s']/3600:.1f} h |")
        for arm, r in e['arms'].items():
            if not r or r.get('failed'):
                print(f'| {s} | {arm} | failed |'); continue
            w = r.get('wall_s_estimator')
            print(f"| {s} | {arm} | {f(r['log10E'],2)} ({f(e['gt_log10E'],2)}) | {f(r['fit_cd_mean'])} | {f(r['future_cd_mean'])} | {f(r['future_psnr_mean'],2)} | {f(r['future_ssim_mean'],3)} | {f(r['future_mask_iou_mean'],3)} | {f(r['fit_psnr_mean'],2)} | {'' if w is None else f'{w/60:.1f} min'} |")
        sb = e.get('static_baseline'); rc = e.get('renderer_ceiling_gt_particles')
        if sb:
            print(f"| {s} | static (GT frame 0) | | {f(sb['fit_cd_mean'])} | {f(sb['future_cd_mean'])} | {f(sb['future_psnr_mean'],2)} | {f(sb['future_ssim_mean'],3)} | {f(sb['future_mask_iou_mean'],3)} | {f(sb['fit_psnr_mean'],2)} | |")
            print(f"| {s} | GT particles, our renderer (ceiling / CD floor) | | {f(rc['fit_cd_mean'])} | {f(rc['future_cd_mean'])} | {f(rc['future_psnr_mean'],2)} | {f(rc['future_ssim_mean'],3)} | {f(rc['future_mask_iou_mean'],3)} | {f(rc['fit_psnr_mean'],2)} | |")
    print(json.dumps(agg, indent=1))
    print(json.dumps(p2, indent=1))


if __name__ == '__main__':
    main()
