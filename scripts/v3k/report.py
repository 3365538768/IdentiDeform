"""V3k report -> docs/v3k/RESULTS.json (A Spring-Gaus in-target refinement, B reconstructed geometry, C simulator
mismatch, D calibration merged from docs/v3k/D_CALIBRATION.json). See docs/v3k/PROTOCOL.json.

  .venv/bin/python scripts/v3k/report.py [--allow-partial]
"""
import argparse, json, math
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
P = json.loads((ROOT / 'docs/v3k/PROTOCOL.json').read_text())
STATS = json.loads((ROOT / 'data/manifests/v2main.specific.stats.json').read_text())
DRAWS, SEED = 2000, 17
KS = [0, 4, 12]


def log10E_from_z(z):
    return math.log10(math.expm1(z * STATS['std'][0] + STATS['mean'][0]))


def zE(l):
    return (math.log1p(10 ** l) - STATS['mean'][0]) / STATS['std'][0]


def pair_acc(pred, gt):
    pairs = [(i, j) for i in range(len(gt)) for j in range(i + 1, len(gt)) if gt[i] != gt[j]]
    return float(np.mean([(pred[i] - pred[j]) * (gt[i] - gt[j]) > 0 for i, j in pairs])), len(pairs)


def rank_metrics(pred, gt):
    acc, n = pair_acc(pred, gt)
    return {'spearman': float(spearmanr(pred, gt).statistic), 'pairwise_order_acc': acc, 'n_pairs': n,
            'mean_abs_dlog10E': float(np.mean(np.abs(np.array(pred) - np.array(gt)))),
            'mean_abs_dz': float(np.mean([abs(zE(p) - zE(g)) for p, g in zip(pred, gt)])),
            'mean_signed_dlog10E': float(np.mean(np.array(pred) - np.array(gt)))}


def boot_mean(d, rng):
    d = np.asarray(d, float); idx = rng.integers(0, len(d), (DRAWS, len(d)))
    m = d[idx].mean(1)
    return [float(d.mean()), float(np.quantile(m, .025)), float(np.quantile(m, .975))]


def boot_spearman_diff(a, b, gt, rng):
    a, b, gt = map(np.asarray, (a, b, gt)); out = []
    for _ in range(DRAWS):
        i = rng.integers(0, len(gt), len(gt))
        if len(set(gt[i])) < 3:
            continue
        out.append(spearmanr(a[i], gt[i]).statistic - spearmanr(b[i], gt[i]).statistic)
    out = np.array([v for v in out if np.isfinite(v)])
    return [float(spearmanr(a, gt).statistic - spearmanr(b, gt).statistic), float(np.quantile(out, .025)), float(np.quantile(out, .975))]


def report_A(allow_partial):
    A = P['A_springgaus_in_target_refinement']; scenes = A['scenes']
    d = ROOT / 'data/v3k/sg'
    st = {arm: {s: json.loads((d / s / f'{arm}.json').read_text()) for s in scenes if (d / s / f'{arm}.json').exists()} for arm in ('prior', 'jellymean', 'diag')}
    done = {arm: [s for s in scenes if st[arm].get(s, {}).get('complete')] for arm in st}
    ok = [s for s in scenes if s in done['prior'] and s in done['jellymean']]
    if not allow_partial:
        assert len(ok) == len(scenes), done
    gt = [st['prior'][s]['gt_log10E'] for s in ok]
    syn = json.loads((ROOT / 'docs/v3f/SPRINGGAUS_SYN.json').read_text())['ensemble9']
    sm = [log10E_from_z(syn['support_mean_baseline']['z_t0'][s][0]) for s in ok]
    rid = [log10E_from_z(syn['ridge_last_layer']['z_t0'][s][0]) for s in ok]
    curves = {arm: {k: [st[arm][s]['budgets'][str(k)]['log10E'] for s in ok] for k in range(13)} for arm in ('prior', 'jellymean')}
    rng = np.random.default_rng(SEED)
    out = {'n_scenes': len(ok), 'scenes': ok, 'missing': [s for s in scenes if s not in ok],
           'per_budget': {arm: {str(k): rank_metrics(curves[arm][k], gt) for k in range(13)} for arm in curves},
           'controls': {'support_mean_no_video_v3f_LOSO': rank_metrics(sm, gt), 'ridge_last_layer_v3f_LOSO': rank_metrics(rid, gt)},
           'per_scene': {}}
    ad = lambda arm, k: np.abs(np.array(curves[arm][k]) - np.array(gt))
    out['paired'] = {
        'prior_k_minus_prior_0_abs_dlog10E': {str(k): boot_mean(ad('prior', k) - ad('prior', 0), rng) for k in (4, 12)},
        'prior_minus_jellymean_abs_dlog10E': {str(k): boot_mean(ad('prior', k) - ad('jellymean', k), rng) for k in KS},
        'prior_k_minus_support_mean_abs_dlog10E': {str(k): boot_mean(ad('prior', k) - np.abs(np.array(sm) - np.array(gt)), rng) for k in KS},
        'jellymean_k_minus_support_mean_abs_dlog10E': {str(k): boot_mean(ad('jellymean', k) - np.abs(np.array(sm) - np.array(gt)), rng) for k in KS},
        'spearman_prior_k_minus_prior_0': {str(k): boot_spearman_diff(curves['prior'][k], curves['prior'][0], gt, rng) for k in (4, 12)},
        'spearman_prior_minus_jellymean': {str(k): boot_spearman_diff(curves['prior'][k], curves['jellymean'][k], gt, rng) for k in (4, 12)}}
    # diagnostics
    diag = {}
    for s in ok:
        dg = st['diag'].get(s, {})
        land = {float(g): v.get('loss') for g, v in dg.get('landscape', {}).items()}
        best_land = min(land, key=land.get) if land else None
        p, j = st['prior'][s], st['jellymean'][s]
        diag[s] = {'gt_log10E': p['gt_log10E'], 'gt_nu': p['gt_nu'], 'prior_log10E': p['x0'], 'prior_nu': p['nu'], 'jellymean_log10E': j['x0'],
                   'prior_k12': curves['prior'][12][ok.index(s)], 'jellymean_k12': curves['jellymean'][12][ok.index(s)],
                   'loss_prior_k0': (p['evals'][0] if p['evals'] else {}).get('loss'),
                   'loss_prior_k12': min(e['loss'] for e in p['evals'][:12]) if p['evals'] else None,
                   'loss_jellymean_k12': min(e['loss'] for e in j['evals'][:12]) if j['evals'] else None,
                   'iou_prior_k12': min(p['evals'][:12], key=lambda e: e['loss']).get('iou') if p['evals'] else None,
                   'splat_radius': p.get('splat_radius'), 'n_particles_sim': p.get('n_particles_sim'),
                   'renderer_upper_bound': dg.get('renderer_upper_bound'), 'gt_replay': {k: dg.get('gt_replay', {}).get(k) for k in ('iou', 'rgb_mae_union', 'loss', 'failed')} if dg.get('gt_replay') else None,
                   'landscape_loss': land, 'landscape_argmin_log10E': best_land,
                   'failed_sims': {'prior': sum(bool(e.get('failed')) for e in p['evals']), 'jellymean': sum(bool(e.get('failed')) for e in j['evals'])}}
    out['per_scene'] = diag
    lam = [diag[s]['landscape_argmin_log10E'] for s in ok if diag[s]['landscape_argmin_log10E'] is not None]
    if len(lam) == len(ok):
        out['diagnostic_landscape_argmin'] = rank_metrics(lam, gt)
    gtl = [diag[s]['gt_replay']['loss'] for s in ok if diag[s]['gt_replay'] and diag[s]['gt_replay'].get('loss') is not None]
    out['diagnostic_summary'] = {
        'renderer_upper_bound_iou_mean': float(np.mean([diag[s]['renderer_upper_bound']['iou'] for s in ok if diag[s]['renderer_upper_bound']])) if any(diag[s]['renderer_upper_bound'] for s in ok) else None,
        'gt_replay_iou_mean': float(np.mean([diag[s]['gt_replay']['iou'] for s in ok if diag[s]['gt_replay'] and diag[s]['gt_replay'].get('iou') is not None])) if gtl else None,
        'gt_replay_loss_mean': float(np.mean(gtl)) if gtl else None,
        'prior_k12_loss_mean': float(np.mean([diag[s]['loss_prior_k12'] for s in ok])),
        'frac_scenes_prior_k12_loss_below_gt_replay': float(np.mean([diag[s]['loss_prior_k12'] < diag[s]['gt_replay']['loss'] for s in ok if diag[s]['gt_replay'] and diag[s]['gt_replay'].get('loss') is not None])) if gtl else None,
        'n_sims_total': int(sum(len(st[a][s]['evals']) for a in ('prior', 'jellymean') for s in ok) + sum(len(st['diag'].get(s, {}).get('landscape', {})) + ('gt_replay' in st['diag'].get(s, {})) for s in ok)),
        'n_failed_refinement_sims': int(sum(v['failed_sims']['prior'] + v['failed_sims']['jellymean'] for v in diag.values()))}
    return out


def ident(abs_err):
    return [v for j, v in enumerate(abs_err) if j in (0, 3) and v is not None]


def obj_boot(groups, vals_fn, rng):
    """groups: {object: [per-element values]} -> pooled mean and object-clustered CI."""
    objs = sorted(groups); arr = [np.asarray(groups[o], float) for o in objs]
    pooled = float(np.concatenate(arr).mean()); ms = []
    for _ in range(DRAWS):
        pick = rng.integers(0, len(objs), len(objs)); ms.append(np.concatenate([arr[i] for i in pick]).mean())
    return [pooled, float(np.quantile(ms, .025)), float(np.quantile(ms, .975))]


def report_BC(variant, idx, allow_partial):
    rng = np.random.default_rng(SEED)
    rows = []
    for i in idx:
        f = ROOT / f'data/v3k/bc/{variant}/{i:06d}/state.json'
        ref = ROOT / f'data/v3b_refine/v2main_C_seed29/{i:06d}/prior_top1/state.json'
        if not f.exists() or not json.loads(f.read_text()).get('complete'):
            continue
        s, r = json.loads(f.read_text()), json.loads(ref.read_text())
        if 'budgets' not in s:
            rows.append({'index': i, 'failed': True, 'state': {k: s.get(k) for k in ('observed_regen',)}}); continue
        row = {'index': i, 'object': s['object'], 'action': s['observed_action'], 'gt_law': s['gt_law'], 'law': s['law'],
               'err': {k: ident(s['budgets'][str(k)]['abs_err_z']) for k in KS},
               'ref_err': {k: ident(r['budgets'][str(k)]['abs_err_z']) for k in KS},
               'obs_iou': {k: (s['budgets'][str(k)].get('observed_fit') or {}).get('iou') for k in KS},
               'ref_obs_iou': {k: (r['budgets'][str(k)].get('observed_fit') or {}).get('iou') for k in KS},
               'failed_sims': sum(bool(e.get('failed')) for e in s['evals']), 'n_sims': len(s['evals'])}
        assert s['x0'] == r['x0'] and s['law'] == r['law'], (i, s['x0'], r['x0'])
        if variant == 'hull':
            row['replay_iou'] = {k: s['replay'].get(str(k), {}).get('iou') for k in KS}
            row['ref_replay_iou'] = {k: r['replay'].get(str(k), {}).get('iou') for k in KS}
            row['gt_params_hull_observed'] = s.get('gt_params_hull_observed')
            meta = ROOT / f'data/v3k/hull_assets/{i:06d}/hull_meta.json'
            row['hull_meta'] = json.loads(meta.read_text()) if meta.exists() else None
        else:
            row['observed_regen_vs_original'] = s.get('observed_regen_vs_original')
        rows.append(row)
    good = [r for r in rows if not r.get('failed')]
    out = {'n_requested': len(idx), 'n_complete': len(good), 'missing': [i for i in idx if i not in [r['index'] for r in rows]],
           'failed_samples': [r['index'] for r in rows if r.get('failed')], 'n_objects': len({r['object'] for r in good})}
    if not good:
        return out
    def grp(key, k):
        g = {}
        for r in good:
            g.setdefault(r['object'], []).extend(r[key][k])
        return g
    def grpdiff(k):
        g = {}
        for r in good:
            g.setdefault(r['object'], []).extend(np.array(r['err'][k]) - np.array(r['ref_err'][k]))
        return g
    out['ident_zmae'] = {str(k): {'variant': obj_boot(grp('err', k), None, rng), 'reference_matched_gt_asset': obj_boot(grp('ref_err', k), None, rng),
                                  'paired_diff_variant_minus_ref': obj_boot(grpdiff(k), None, rng)} for k in KS}
    out['refinement_gain'] = {}
    for k in (4, 12):
        g, gr = {}, {}
        for r in good:
            g.setdefault(r['object'], []).extend(np.array(r['err'][k]) - np.array(r['err'][0]))
            gr.setdefault(r['object'], []).extend(np.array(r['ref_err'][k]) - np.array(r['ref_err'][0]))
        out['refinement_gain'][str(k)] = {'variant_k_minus_0': obj_boot(g, None, rng), 'reference_k_minus_0': obj_boot(gr, None, rng)}
    mean = lambda key, k: float(np.mean([r[key][k] for r in good if r[key][k] is not None])) if any(r[key][k] is not None for r in good) else None
    out['observed_iou_mean'] = {str(k): {'variant': mean('obs_iou', k), 'reference': mean('ref_obs_iou', k)} for k in (4, 12)}
    if variant == 'hull':
        out['replay_iou_mean'] = {}
        for k in KS:
            both = [r for r in good if r['replay_iou'][k] is not None and r['ref_replay_iou'][k] is not None]
            out['replay_iou_mean'][str(k)] = {'n_paired': len(both), 'variant': float(np.mean([r['replay_iou'][k] for r in both])) if both else None,
                                              'reference': float(np.mean([r['ref_replay_iou'][k] for r in both])) if both else None,
                                              'paired_diff': boot_mean([r['replay_iou'][k] - r['ref_replay_iou'][k] for r in both], rng) if len(both) > 1 else None}
        g = [r['gt_params_hull_observed'] for r in good if r.get('gt_params_hull_observed') and 'iou' in r['gt_params_hull_observed']]
        out['gt_params_hull_geometry_observed_iou_mean'] = float(np.mean([x['iou'] for x in g])) if g else None
        out['gt_params_hull_geometry_observed_n'] = len(g)
    else:
        g = [r['observed_regen_vs_original'] for r in good if r.get('observed_regen_vs_original')]
        out['perturbed_observed_vs_original_observed'] = {'iou_mean': float(np.mean([x['iou'] for x in g])), 'rgb_mae_mean': float(np.mean([x['rgb_mae_union'] for x in g])), 'n': len(g)}
    out['failed_sims'] = int(sum(r['failed_sims'] for r in good)); out['n_sims_refinement'] = int(sum(r['n_sims'] for r in good))
    out['per_sample'] = rows
    return out


def main():
    a = argparse.ArgumentParser(); a.add_argument('--allow-partial', action='store_true'); args = a.parse_args()
    res = {'protocol': 'docs/v3k/PROTOCOL.json', 'split': 'V2 val only (A: Spring-Gaus synthetic; D: val + test calibration only)'}
    res['A_springgaus_in_target_refinement'] = report_A(args.allow_partial)
    res['B_reconstructed_geometry'] = report_BC('hull', P['B_reconstructed_geometry']['samples'], args.allow_partial)
    res['C_simulator_mismatch'] = report_BC('mismatch', P['C_simulator_mismatch']['samples'], args.allow_partial)
    for key, v in (('B_reconstructed_geometry', 'hull'), ('C_simulator_mismatch', 'mismatch')):  # addendum 1: also without 212
        idx = [i for i in P[key]['samples'] if i != 212]
        sub = report_BC(v, idx, args.allow_partial); sub.pop('per_sample', None)
        res[key]['excluding_212'] = sub
    dc = ROOT / 'docs/v3k/D_CALIBRATION.json'
    res['D_calibration'] = {'file': 'docs/v3k/D_CALIBRATION.json', 'present': dc.exists()}
    (ROOT / 'docs/v3k/RESULTS.json').write_text(json.dumps(res, indent=1, default=float) + '\n')
    A = res['A_springgaus_in_target_refinement']
    for arm in ('prior', 'jellymean'):
        print(arm, {k: {m: round(A['per_budget'][arm][k][m], 3) for m in ('spearman', 'pairwise_order_acc', 'mean_abs_dlog10E')} for k in ('0', '4', '12')})
    print('controls', {c: {m: round(v[m], 3) for m in ('spearman', 'pairwise_order_acc', 'mean_abs_dlog10E')} for c, v in A['controls'].items()})
    for v in ('B_reconstructed_geometry', 'C_simulator_mismatch'):
        r = res[v]; print(v, r.get('n_complete'), {k: [round(x, 3) for x in r['ident_zmae'][k]['variant']] + [round(x, 3) for x in r['ident_zmae'][k]['reference_matched_gt_asset']] for k in r.get('ident_zmae', {})})


if __name__ == '__main__':
    main()
