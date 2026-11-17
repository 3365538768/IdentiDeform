"""V3f: merge Spring-Gaus results into docs/v3f/SPRINGGAUS.json (python -I; inputs are our own JSON outputs).
  springgaus_merge.py <SYN.json> <REAL_PRED.json> <replay.json>... --v2-val-preds runs/v3c_ens_bcd9_val/predictions.jsonl --out docs/v3f/SPRINGGAUS.json
"""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ST = json.loads(Path('data/manifests/v2main.specific.stats.json').read_text())


def lg(z):
    return float(np.log10(np.expm1(z * ST['std'][0] + ST['mean'][0])))


def main(a):
    syn = json.loads(Path(a.syn).read_text()); pred = json.loads(Path(a.pred).read_text())
    rep = {}
    for r in a.replay:
        rep.update(json.loads(Path(r).read_text()))
    e = syn['ensemble9']
    # V2 val reference: jelly E/rho range + drop-only within-law Spearman (same ensemble)
    rows = [json.loads(l) for l in Path(a.v2_val_preds).read_text().splitlines()]
    ref = {}
    for law, name in enumerate(['jelly', 'metal', 'plasticine']):
        sel = [r for r in rows if r['law_id'] == law]
        g = np.array([lg(r['target_z'][0]) for r in sel])
        drop = [r for r in sel if r['action'] == 'drop']
        ref[name] = {'n_val': len(sel), 'log10_E_over_rho_x1000_range': [float(g.min()), float(g.max())],
                     'drop_only_spearman_E': float(spearmanr([r['target_z'][0] for r in drop], [r['prediction_z'][0] for r in drop]).statistic), 'n_drop': len(drop),
                     'drop_only_law_acc': float(np.mean([r['predicted_law_id'] == law for r in drop]))}
    real = {}
    for s, p in pred['predictions'].items():
        o = s.split('/')[1]; R = rep.get(o)
        d = {'law_majority': p['law_majority'], 'law_votes': p['law_votes'], 'log10_E_over_rho_x1000_pred': p['log10_E_over_rho_x1000'],
             'log10_sd_over_runs': float(np.std([lg(z) for z in p['per_run_z_E']])), 'nu_pred': p['nu']}
        if R:
            c = R['candidates']; b = R['refined_best_grid']
            pick = lambda k: {kk: c[k].get(kk) for kk in ('log10_E_over_rho_x1000', 'post_hit_up_rmse_cm', 'mask_iou_mean', 'centroid_err_cm_mean')} | {
                'restitution_e_20f': (c[k].get('bounce_sim_20frames') or {}).get('restitution_e'),
                'apex_above_contact_cm_full': (c[k].get('bounce_sim_full') or {}).get('apex_minus_contact_cm')}
            d['replay'] = {'observed': R['observed_bounce'], 'ground_up_m': R['ground_up_m'], 'hull_extent_cm': R['hull_extent_cm'],
                           'hull_particles': R['hull_particles'], 'predicted': pick('predicted') if 'loss' in c['predicted'] else c['predicted'],
                           'refined_best_grid': pick(b) if b else None,
                           'grid': {k: pick(k) if 'loss' in v else v for k, v in c.items() if k.startswith('grid')}}
        real[o] = d
    objs = sorted(real)
    out = {'source': 'Spring-Gaus (Zhong et al., ECCV 2024), data from the official Google Drive folder linked in the README '
                     '(mpm_synthetic.zip, real_capture.zip); Apache-2.0 code repo cloned to external/springgaus for reading only (nothing executed)',
           'synthetic': {'protocol': syn['protocol'],
                         'ensemble9_canonical_triplet': {m: e[m]['canonical_t0'] for m in ('K0', 'offset', 'ridge_last_layer', 'support_mean_baseline')},
                         'ensemble9_mean_over_triplets': {m: e[m]['mean_over_triplets'] for m in ('K0', 'offset', 'ridge_last_layer', 'support_mean_baseline')},
                         'law': e['law'], 'K0_spearman_E_per_run': syn['K0_spearman_E_per_run'],
                         'K0_decoded_t0': e['K0']['decoded_t0'],
                         'v2_val_reference_same_ensemble': ref,
                         'detail_file': 'docs/v3f/SPRINGGAUS_SYN.json'},
           'real': {'protocol': {'input': pred['protocol'], 'replay': 'scripts/v3f/springgaus_replay.py docstring', 'gt': 'none released (no material parameters for real objects); qualitative + rebound consistency only'},
                    'objects': real, 'prior_ranking_soft_to_stiff': pred['E_over_rho_ranking_soft_to_stiff']},
           'figures': ['docs/v3f/springgaus_syn_scatter.png', 'docs/v3f/springgaus_real_prior.png', 'docs/v3f/springgaus_real_replay.png']}
    ok = [o for o in objs if 'replay' in real[o] and real[o]['replay']['observed']]
    if len(ok) >= 3:
        eo = [real[o]['replay']['observed']['restitution_e'] for o in ok]
        out['real']['consistency'] = {
            'objects': ok, 'observed_restitution': dict(zip(ok, eo)),
            'spearman_pred_E_vs_observed_restitution': float(spearmanr([real[o]['log10_E_over_rho_x1000_pred'] for o in ok], eo).statistic),
            'spearman_refined_E_vs_observed_restitution': float(spearmanr([real[o]['replay']['refined_best_grid']['log10_E_over_rho_x1000'] for o in ok], eo).statistic),
            'mean_abs_restitution_err_predicted': float(np.mean([abs((real[o]['replay']['predicted'].get('restitution_e_20f') or np.nan) - real[o]['replay']['observed']['restitution_e']) for o in ok])),
            'mean_abs_restitution_err_refined': float(np.mean([abs((real[o]['replay']['refined_best_grid'].get('restitution_e_20f') or np.nan) - real[o]['replay']['observed']['restitution_e']) for o in ok])),
            'mean_post_hit_rmse_cm_predicted': float(np.mean([real[o]['replay']['predicted']['post_hit_up_rmse_cm'] for o in ok])),
            'mean_post_hit_rmse_cm_refined': float(np.mean([real[o]['replay']['refined_best_grid']['post_hit_up_rmse_cm'] for o in ok])),
            'mean_iou_predicted': float(np.mean([real[o]['replay']['predicted']['mask_iou_mean'] for o in ok])),
            'mean_iou_refined': float(np.mean([real[o]['replay']['refined_best_grid']['mask_iou_mean'] for o in ok]))}
    Path(a.out).write_text(json.dumps(out, indent=1, default=float) + '\n')
    print(json.dumps(out['real'].get('consistency'), indent=1)); print(json.dumps(ref, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('syn'); p.add_argument('pred'); p.add_argument('replay', nargs='*')
    p.add_argument('--v2-val-preds', required=True); p.add_argument('--out', required=True)
    main(p.parse_args())
