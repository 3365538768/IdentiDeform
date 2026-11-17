"""V3f: score the PIXIE static baseline (scripts/v3f/pixie_features.py outputs) on V2 main val / test.

One prediction per OBJECT (static; PIXIE sees only geometry/appearance), broadcast to every sample of that object.
Class mapping and fill rules are fixed a priori in PROTOCOL and written to the JSON before any scoring.
Scoring = scripts/v3c/baselines_report.score (standardized MAE in specific space with train stats, law accuracy,
within-law r) + scripts/bootstrap_object_ci.compare vs each V-JEPA C seed (val: best-val epoch, test: test_once).
  .venv/bin/python scripts/v3f/pixie_score.py --split val      (writes docs/v3f/PIXIE_BASELINE.json['val'])
  .venv/bin/python scripts/v3f/pixie_score.py --split test     (only after val is written; once)
"""
import argparse, json, sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, 'scripts'); sys.path.insert(0, 'scripts/v3c')
import numpy as np
from fieldbridge.data import ParameterCodec, specific_parameters
from baselines_report import score
from bootstrap_object_ci import compare
from r1_report import best_dir

OUT = Path('docs/v3f/PIXIE_BASELINE.json')
STATS = 'data/manifests/v2main.specific.stats.json'
MANIFEST = {'val': 'data/manifests/v2main.jsonl', 'test': 'data/manifests/v2main_withtest.jsonl'}
FEAT = {'val': 'data/v3f_pixie/val', 'test': 'data/v3f_pixie/test', 'train': 'data/v3f_pixie/train'}
LAWS = {'jelly': 0, 'metal': 1, 'plasticine': 2}
PROTOCOL = {
    'method': 'PIXIE (Le et al., github.com/vlongle/pixie) released UNets: checkpoints_discrete/20250505_013424/epoch_510.pth '
              '(8-class material segmentation) + checkpoints_continuous_mse/20250805_040343/epoch_250.pth (density, E, nu); '
              'input = 64^3 CLIP ViT-L/14@336 feature voxel grid of the static object (see pixie_features.py docstring for faithful vs approximated steps)',
    'class_mapping_a_priori': {
        'jelly': 'jelly', 'metal': 'metal', 'stationary': 'metal (rigid -> stiffest of our laws)',
        'sand': 'plasticine', 'visplas': 'plasticine', 'snow': 'plasticine', 'fluid': 'plasticine',
        'background': 'jelly (majority train law; only if every mask voxel were background)',
        'rationale': 'our laws: jelly = elastic (FCR), metal = elastic + von Mises yield (StVK), plasticine = elastoplastic/irreversible; '
                     'PIXIE ids follow PhysGaussian MPM (0 jelly, 1 metal, 2 sand, 3 visplas, 4 fluid, 5 snow, 6 stationary, 7 background); '
                     'every PIXIE flow/plastic class goes to plasticine'},
    'aggregation': 'object = majority PIXIE class over occupied voxels (background excluded); E, nu, rho = medians over occupied voxels',
    'targets': 'specific space as V2: E/(rho/1000), nu, rho, sigma_y/(rho/1000); ln1p + z-score with train stats (data/manifests/v2main.specific.stats.json). '
               'PIXIE has no yield stress -> sigma_y/rho prediction = train mean (z = 0), same as the constant baseline',
    'primary_variant': 'filled (occupancy with interior filled); shell variant reported as sensitivity',
    'metric': 'identical to docs/v3c/BASELINES_VAL.json (scripts/v3c/baselines_report.score); bootstrap = scripts/bootstrap_object_ci.compare vs V-JEPA C seeds',
    'test_rule': 'test scored once, only after val numbers were written to this file',
    'calibrated_variant': 'filled_train_calibrated: PIXIE also run on the 87 TRAIN objects; per parameter (E/rho, nu, rho) an OLS map '
                          'z_target = a + b * z_pixie is fitted on train samples only and applied to val/test (fixes PIXIE\'s real-world '
                          'absolute scale vs our sampled parameter ranges; laws unchanged). Added after the first val scoring showed the scale '
                          'mismatch; fixed before test.'}


def predictions(split, variant, codec, calib=None, tag=None):
    rows = [json.loads(l) for l in open(MANIFEST.get(split, MANIFEST['val']))]
    rows = [r for r in rows if r['split'] == split]
    objs = sorted({r['object_id'] for r in rows})
    per = {}
    for o in objs:
        v = json.loads((Path(FEAT[split]) / f'{o}.json').read_text())['variants'][variant]
        cls = v['majority_class_excl_background']
        law = PROTOCOL['class_mapping_a_priori'][cls].split(' ')[0]
        z, m = codec.encode(specific_parameters({'E': v['median_E_Pa'], 'nu': v['median_nu'], 'rho': v['median_rho'], 'sigma_y': None}))
        if calib:
            z = z.clone()
            for j in range(3):
                z[j] = calib[j][0] + calib[j][1] * z[j]
        per[o] = {'pixie_class': cls, 'law': law, 'E_Pa': v['median_E_Pa'], 'nu': v['median_nu'], 'rho': v['median_rho'],
                  'z': [float(x) for x in z], 'class_hist': v['class_hist_mask_voxels'],
                  'gt_law_counts': dict(Counter(r['law_id'] for r in rows if r['object_id'] == o))}
    d = Path(f'runs/v3f_pixie_{tag or variant}_{split}')
    d.mkdir(parents=True, exist_ok=True)
    with (d / 'predictions.jsonl').open('w') as f:
        for r in rows:
            t, m = codec.encode(specific_parameters(r['parameters']))
            p = per[r['object_id']]
            f.write(json.dumps({'sample_id': r['sample_id'], 'object_id': r['object_id'], 'source': r.get('source', ''),
                                'action': r['action'], 'law_id': r['law_id'], 'predicted_law_id': LAWS[p['law']],
                                'prediction_z': p['z'], 'logvar': [0.] * 4, 'target_z': t.tolist(), 'target_mask': m.tolist()}) + '\n')
    return d, per


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--split', choices=['val', 'test'], required=True); a = ap.parse_args()
    rep = json.loads(OUT.read_text()) if OUT.exists() else {}
    if 'protocol' not in rep:
        rep['protocol'] = PROTOCOL; OUT.write_text(json.dumps(rep, indent=2) + '\n')  # mapping fixed before scoring
    assert rep['protocol']['class_mapping_a_priori'] == PROTOCOL['class_mapping_a_priori']
    if 'calibrated_variant' not in rep['protocol']:
        rep['protocol']['calibrated_variant'] = PROTOCOL['calibrated_variant']
    if a.split == 'test':
        if 'val' not in rep:
            raise SystemExit('val numbers must be written first')
        if 'test' in rep or Path('runs/v3f_pixie_filled_test').exists():
            raise SystemExit('test is scored once')
    codec = ParameterCodec(json.loads(Path(STATS).read_text()))
    ref = ({s: best_dir(Path(f'runs/v2main_C_seed{s}'))[0] for s in (17, 29, 43)} if a.split == 'val'
           else {s: Path(f'runs/v2main_C_seed{s}/test_once') for s in (17, 29, 43)})
    res = {}
    # train-only affine calibration of PIXIE's z (filled variant)
    _, trper = predictions('train', 'filled', codec)
    trrows = [r for r in (json.loads(l) for l in open(MANIFEST['val'])) if r['split'] == 'train']
    calib = []
    for j in range(3):
        x, y = [], []
        for r in trrows:
            t, mm = codec.encode(specific_parameters(r['parameters']))
            if bool(mm[j]):
                x.append(trper[r['object_id']]['z'][j]); y.append(float(t[j]))
        b, a0 = np.polyfit(x, y, 1); calib.append((float(a0), float(b)))
    res['train_calibration'] = {'params': ['E_over_rho', 'nu', 'rho'], 'intercept_slope': calib, 'train_objects': len(trper),
                                'train_object_law_counts': dict(Counter(p['law'] for p in trper.values()))}
    for variant, cal, tag in (('filled', None, None), ('shell', None, None), ('filled', calib, 'filled_train_calibrated')):
        d, per = predictions(a.split, variant, codec, cal, tag)
        m = score(d)
        m['vs_vjepa_C'] = {str(s): compare(d / 'predictions.jsonl', r / 'predictions.jsonl') for s, r in ref.items()}
        ens = Path(f'runs/v3c_ens_bcd9_{a.split}')
        if (ens / 'predictions.jsonl').exists():
            m['vs_vjepa_ensemble_BCD9'] = compare(d / 'predictions.jsonl', ens / 'predictions.jsonl')
        zE = np.array([p['z'][0] for p in per.values()])
        m['across_object_spread_of_predicted_z'] = {'E_over_rho_sd': float(zE.std()), 'nu_sd': float(np.std([p['z'][1] for p in per.values()])),
                                                    'rho_sd': float(np.std([p['z'][2] for p in per.values()]))}
        m['predicted_law_counts_objects'] = dict(Counter(p['law'] for p in per.values()))
        m['objects'] = per; m['predictions'] = str(d / 'predictions.jsonl')
        res[tag or variant] = m
    # comparison rows (same metric, same samples) from the existing frozen reports + the BCD9 ensemble
    comp = {}
    src = Path('docs/v3c/BASELINES_VAL.json' if a.split == 'val' else 'docs/v3c/BASELINES_TEST.json')
    comp['source'] = str(src); comp['table'] = json.loads(src.read_text()).get('table')
    ens = Path(f'runs/v3c_ens_bcd9_{a.split}')
    if (ens / 'predictions.jsonl').exists():
        e = score(ens); comp['vjepa2_ensemble_BCD9 (ours, rescored here)'] = {k: e[k] for k in ('standardized_mae', 'object_macro_mae', 'law_accuracy', 'per_parameter_mae_z', 'within_law_r')}
    res['comparison'] = comp
    rep[a.split] = res
    OUT.write_text(json.dumps(rep, indent=2) + '\n')
    for v in ('filled', 'shell', 'filled_train_calibrated'):
        r = res[v]
        print(a.split, v, 'MAE %.4f law_acc %.3f' % (r['standardized_mae'], r['law_accuracy']),
              {k: x.get('mean_over_laws') for k, x in r['within_law_r'].items()}, r['predicted_law_counts_objects'])


if __name__ == '__main__':
    main()
