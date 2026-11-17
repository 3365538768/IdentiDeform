"""V3k D: calibration of the 9-run B/C/D ensemble's uncertainty (val; test reported once, calibration only).

Interval types per target (z space, train-standardised):
  ens   sigma = std over the 9 runs of prediction_z (np.std, ddof=0; as docs/v3b/ENSEMBLE_BCD9_VAL.json)
  total sigma = sqrt(mean_r exp(logvar_r) + ens_var)  (per-run heteroscedastic NLL head + across-run spread)
mu = mean over runs (identical to the saved ensembles). Targets masked by target_mask.
Temperature: one scalar T per target (and pooled) fitted on VAL ONLY, closed-form Gaussian-NLL minimiser
T^2 = mean(r^2 / sigma^2). Test coverage is reported with raw sigma and with the val-fitted T only (no selection).
  .venv/bin/python scripts/v3k/d_calibration.py        -> docs/v3k/D_CALIBRATION.json
"""
import json, math, sys
from pathlib import Path
import numpy as np
from scipy.stats import norm
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from r1_report import best_dir  # noqa: E402
RUNS = [f'v2main_{g}_seed{s}' for g in 'BCD' for s in (17, 29, 43)]
TARGETS = ['E_over_rho', 'nu', 'rho', 'sigma_y_over_rho']
LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.683, 0.7, 0.8, 0.9, 0.954, 0.99]


def load(split):
    rows = {}
    for r in RUNS:
        d = best_dir(ROOT / 'runs' / r)[0] if split == 'val' else ROOT / 'runs' / r / 'test_once'
        for x in map(json.loads, (d / 'predictions.jsonl').read_text().splitlines()):
            e = rows.setdefault(x['sample_id'], {'z': [], 'lv': [], 't': x['target_z'], 'm': x['target_mask'], 'obj': x['object_id']})
            e['z'].append(x['prediction_z']); e['lv'].append(x['logvar'])
    sids = sorted(rows); assert all(len(rows[s]['z']) == 9 for s in sids)
    Z = np.array([rows[s]['z'] for s in sids]); LV = np.array([rows[s]['lv'] for s in sids])
    mu = Z.mean(1); ens = Z.std(1); tot = np.sqrt(np.exp(LV).mean(1) + ens ** 2)
    return sids, mu, {'ens': ens, 'total': tot}, np.array([rows[s]['t'] for s in sids]), np.array([rows[s]['m'] for s in sids], bool)


def check(split, sids, mu):
    ref = (json.loads((ROOT / 'docs/v3b/ENSEMBLE_BCD9_VAL.json').read_text())['samples'] if split == 'val' else
           {x['sample_id']: x for x in map(json.loads, (ROOT / 'runs/v3c_ens_bcd9_test/predictions.jsonl').read_text().splitlines())})
    assert set(ref) == set(sids), split
    assert np.allclose(mu, np.array([ref[s]['prediction_z'] for s in sids]), atol=1e-9), split


def stats(r, s):
    a = np.abs(r) / s
    return {'n': int(len(r)), 'cov_1sigma': float((a <= 1).mean()), 'cov_2sigma': float((a <= 2).mean()),
            'nll': float(np.mean(0.5 * (r / s) ** 2 + np.log(s) + 0.5 * math.log(2 * math.pi))),
            'median_sigma': float(np.median(s)), 'rmse': float(np.sqrt(np.mean(r ** 2))),
            'reliability': {str(l): float((a <= norm.ppf(0.5 + l / 2)).mean()) for l in LEVELS}}


def main():
    data = {sp: load(sp) for sp in ('val', 'test')}
    for sp, (sids, mu, *_ ) in data.items():
        check(sp, sids, mu)
    out = {'protocol': __doc__, 'runs': RUNS, 'expected': {'cov_1sigma': 0.6827, 'cov_2sigma': 0.9545},
           'temperature_fit': 'val only; T^2 = mean(r^2/sigma^2) (closed-form Gaussian NLL minimiser); scaled val is in-sample',
           'n_samples': {sp: len(data[sp][0]) for sp in data}, 'results': {}}
    for kind in ('ens', 'total'):
        res = {}
        vs, vmu, vsig, vt, vm = data['val']; ts, tmu, tsig, tt, tm = data['test']
        rv_all, sv_all = [], []
        for j, name in enumerate(TARGETS):
            rv = (vt - vmu)[vm[:, j], j]; sv = np.maximum(vsig[kind][vm[:, j], j], 1e-6)
            rt = (tt - tmu)[tm[:, j], j]; st = np.maximum(tsig[kind][tm[:, j], j], 1e-6)
            T = float(np.sqrt(np.mean((rv / sv) ** 2)))
            res[name] = {'T_val': T, 'val_raw': stats(rv, sv), 'val_scaled_in_sample': stats(rv, T * sv),
                         'test_raw': stats(rt, st), 'test_val_scaled': stats(rt, T * st)}
            rv_all.append(rv / sv)
        Tp = float(np.sqrt(np.mean(np.concatenate(rv_all) ** 2)))
        res['pooled_T_val'] = Tp
        res['pooled_T_applied'] = {}
        for j, name in enumerate(TARGETS):
            rv = (vt - vmu)[vm[:, j], j]; sv = np.maximum(vsig[kind][vm[:, j], j], 1e-6)
            rt = (tt - tmu)[tm[:, j], j]; st = np.maximum(tsig[kind][tm[:, j], j], 1e-6)
            res['pooled_T_applied'][name] = {'val': stats(rv, Tp * sv), 'test': stats(rt, Tp * st)}
        out['results'][kind] = res
    p = ROOT / 'docs/v3k/D_CALIBRATION.json'; p.write_text(json.dumps(out, indent=1) + '\n')
    for kind in out['results']:
        for name in TARGETS:
            r = out['results'][kind][name]
            f = lambda d: f"{d['cov_1sigma']:.3f}/{d['cov_2sigma']:.3f}"
            print(kind, name, 'n', r['val_raw']['n'], r['test_raw']['n'], 'val', f(r['val_raw']), 'T', round(r['T_val'], 2),
                  'valT', f(r['val_scaled_in_sample']), 'test', f(r['test_raw']), 'testT', f(r['test_val_scaled']),
                  'nll', round(r['test_raw']['nll'], 3), round(r['test_val_scaled']['nll'], 3))
        print(kind, 'pooled T', out['results'][kind]['pooled_T_val'])


if __name__ == '__main__':
    main()
