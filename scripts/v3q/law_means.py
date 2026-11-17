"""V3q: V3 train law means -> docs/v3q/LAW_MEANS.json.

The V3 train parameters are deterministic draws of scripts/v2/gen_v2.sample(i, 'main-train') with the V3 range
overrides of scripts/v3n/gen_v3.py (the same call that writes data/v3_main/train/<i>/gt.json). The means are taken
over the PLANNED train indices (docs/v3n/queue.json, split train), so they equal the train gt.json means up to
samples whose generation fails; they are cross-checked against the train gt.json files that exist (--check).
Units as V2 law_means: per law, mean of ln1p(E/rho specific), ln1p(sigma_y/rho specific) (non-censored only,
V2 manifests drop censored sigma), and mean nu. Also writes provisional V3 z statistics (train, specific,
ln1p / linear nu) used only if the V3 prior hand-off does not ship its own.

  .venv/bin/python scripts/v3q/law_means.py [--check]
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v2'))
import gen_v2  # noqa: E402
gen_v2.SPECIFIC_E.update(jelly=(3.5, 6.0), plasticine=(3.5, 5.5), metal=(4.5, 6.0))
gen_v2.SPECIFIC_SIGMA.update(plasticine=(3.2, 4.5), metal=(3.2, 4.5))


def row(i):
    law, _, p, labels, _ = gen_v2.sample(i, 'main-train')
    if 'sigma_over_rho_specific' in labels:
        labels['censored_sigma'] = bool(labels['sigma_over_rho_specific'] > 0.5 * labels['E_over_rho_specific'])
    return law, labels['E_over_rho_specific'], p['nu'], p['density'], (None if labels['censored_sigma'] else labels.get('sigma_over_rho_specific'))


def means_of(rows):
    out = {}
    for law in gen_v2.LAWS:
        r = [x for x in rows if x[0] == law]
        s = [math.log1p(x[4]) for x in r if x[4] is not None]
        out[law] = {'n': len(r), 'logE': float(np.mean([math.log1p(x[1]) for x in r])), 'nu': float(np.mean([x[2] for x in r])),
                    'logS': float(np.mean(s)) if s else None, 'n_sigma': len(s),
                    'log10E_geo': float(np.mean([math.log10(x[1]) for x in r])),
                    'log10S_geo': float(np.mean([math.log10(x[4]) for x in r if x[4] is not None])) if s else None}
    return out


def main():
    a = argparse.ArgumentParser(); a.add_argument('--check', action='store_true'); args = a.parse_args()
    q = json.loads((ROOT / 'docs/v3n/queue.json').read_text())['jobs']
    idx = sorted({int(j['cmd'].split('--indices ')[1].split()[0]) for j in q if '--split train' in j['cmd']})
    a4 = ROOT / 'docs/v3n/jobs_a4.json'
    if a4.exists():
        idx = sorted(set(idx) | {j[2] for j in json.loads(a4.read_text()) if j[1] == 'train'})
    rows = [row(i) for i in idx]
    out_path = ROOT / 'docs/v3q/LAW_MEANS.json'
    if out_path.exists() and not args.check:
        print('exists'); return
    if args.check:
        old = json.loads(out_path.read_text())
        have = []
        for i in idx:
            d = ROOT / f'data/v3_main/train/{i:06d}'
            try:
                if json.loads((d / 'result.json').read_text()).get('exit') == 0:
                    g = json.loads((d / 'gt.json').read_text()); lab = g['labels']
                    r = row(i)
                    assert g['law'] == r[0] and abs(lab['E_over_rho_specific'] - r[1]) < 1e-6 * r[1], i
                    have.append(r)
            except FileNotFoundError:
                pass
        old['check_against_generated_train'] = {'n_generated': len(have), 'means_generated': means_of(have) if have else None,
                                                'all_rows_match_sampler': True}
        out_path.write_text(json.dumps(old, indent=1) + '\n'); print(json.dumps(old['check_against_generated_train'], indent=1)); return
    allS = [math.log1p(x[4]) for x in rows if x[4] is not None]
    cols = [[math.log1p(x[1]) for x in rows], [x[2] for x in rows], [math.log1p(x[3]) for x in rows], allS]
    stats = {'mean': [float(np.mean(c)) for c in cols], 'std': [float(np.std(c)) for c in cols], 'count': [len(c) for c in cols],
             'transform': 'ln1p_E_rho_sigma_y;linear_nu', 'target_space': 'specific', 'fit_split': 'train (planned V3 indices, sampler)',
             'note': 'provisional; the V3 prior hand-off stats replace it if shipped'}
    out = {'source': 'gen_v2.sample(i, "main-train") with V3 ranges (scripts/v3n/gen_v3.py) over the planned train indices',
           'n_train_planned': len(idx), 'means': means_of(rows), 'all_law_logS': float(np.mean(allS)), 'stats_provisional': stats}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=1) + '\n'); print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
