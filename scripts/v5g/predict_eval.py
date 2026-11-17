"""V5g primary metric (docs/v5g/PREREG_ADDENDUM.md): per-candidate grasp-outcome prediction on paired-complete samples.
  .venv/bin/python scripts/v5g/predict_eval.py [--runs data/v5g/runs/val] [--out docs/v5g/PREDICTION.json]
Predictors: twins (prior_vt, prior_vision, lawmean_pred, gt = self-consistency 1.0) classify each candidate from their own simulation
with the pre-registered outcome rule; baselines: (i) LOO majority class, (ii) per-candidate LOO majority, (iii) per-candidate x
vision-predicted-law LOO majority. Metrics: macro-F1, accuracy, per-class recall, feasibility accuracy, safe-grasp precision/recall;
object bootstrap (2000, seed 0) with paired contrasts on the same draws.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v5g'))
import plan_eval as P  # noqa: E402  (outcome rule + thresholds)

TW = ['prior_vt', 'prior_vision', 'lawmean_pred']
BASE = ['base_majority', 'base_cand', 'base_cand_law']
CLS = ['success', 'drop', 'damage']
CONTRASTS = [('prior_vt', 'prior_vision'), ('prior_vision', 'lawmean_pred'), ('prior_vt', 'lawmean_pred'),
             ('prior_vt', 'base_cand'), ('prior_vision', 'base_cand'), ('prior_vt', 'base_cand_law'), ('prior_vision', 'base_cand_law'),
             ('lawmean_pred', 'base_cand'), ('prior_vt', 'base_majority')]


def metrics(pairs):
    """pairs: list of (true, pred). Returns dict."""
    if not pairs:
        return None
    t = np.array([p[0] for p in pairs]); q = np.array([p[1] for p in pairs])
    rec, f1 = {}, []
    for c in CLS:
        tp = np.sum((t == c) & (q == c)); fn = np.sum((t == c) & (q != c)); fp = np.sum((t != c) & (q == c))
        r = tp / (tp + fn) if tp + fn else np.nan; pr = tp / (tp + fp) if tp + fp else np.nan
        rec[c] = float(r) if np.isfinite(r) else None
        if tp + fn:  # macro over classes present in the truth
            f1.append(0.0 if not (tp) else 2 * pr * r / (pr + r))
    ps = q == 'success'; ts = t == 'success'
    return {'macro_f1': float(np.mean(f1)), 'accuracy': float(np.mean(t == q)), 'recall': rec,
            'safe_precision': float(np.mean(ts[ps])) if ps.any() else None, 'safe_recall': float(np.mean(ps[ts])) if ts.any() else None,
            'n_pairs': int(len(t))}


def main():
    a = argparse.ArgumentParser(); a.add_argument('--runs', default='data/v5g/runs/val'); a.add_argument('--out', default='docs/v5g/PREDICTION.json')
    args = a.parse_args()
    S = []
    for sd in sorted((ROOT / args.runs).iterdir()):
        if not sd.name.isdigit():
            continue
        st = {}
        for m in ['gt'] + TW:
            s = P.load_state(sd, m)
            if s is not None:
                st[m] = s
        if len(st) < 4:
            continue  # paired-complete only
        g = json.loads((ROOT / f'data/v5b_main/val/{sd.name}/gt.json').read_text())
        S.append({'index': int(sd.name), 'object': g['object'], 'law': g['law'], 'vis_law': st['prior_vision']['meta'].get('law'),
                  'true': {c: P.outcome(r) for c, r in st['gt']['runs'].items() if c != 'none:0'},
                  'pred': {m: {c: P.outcome(r) for c, r in st[m]['runs'].items() if c != 'none:0'} for m in TW}})
    cands = sorted({c for s in S for c in s['true']})
    # baselines (leave-one-OBJECT-out)
    for s in S:
        oth = [o for o in S if o['object'] != s['object']] or [o for o in S if o is not s]
        allc = Counter(v for o in oth for v in o['true'].values())
        maj = max(CLS, key=lambda c: (allc[c], -CLS.index(c)))
        s['pred']['base_majority'] = {c: maj for c in cands}
        s['pred']['base_cand'] = {}; s['pred']['base_cand_law'] = {}
        for c in cands:
            cc = Counter(o['true'][c] for o in oth)
            s['pred']['base_cand'][c] = max(CLS, key=lambda k: (cc[k], -CLS.index(k)))
            cl = Counter(o['true'][c] for o in oth if o['vis_law'] == s['vis_law'])
            s['pred']['base_cand_law'][c] = max(CLS, key=lambda k: (cl[k], -CLS.index(k))) if cl else s['pred']['base_cand'][c]
    preds = TW + BASE

    def evaluate(samples):
        out = {}
        for m in preds:
            pairs = [(s['true'][c], s['pred'][m][c]) for s in samples for c in cands]
            e = metrics(pairs)
            if e is None:
                continue
            feas = [(any(v == 'success' for v in s['true'].values()), any(v == 'success' for v in s['pred'][m].values())) for s in samples]
            e['feasibility_accuracy'] = float(np.mean([x == y for x, y in feas]))
            out[m] = e
        return out
    point = evaluate(S)
    keys = ['macro_f1', 'accuracy', 'safe_precision', 'safe_recall', 'feasibility_accuracy']
    rng = np.random.default_rng(0); by = defaultdict(list)
    for s in S:
        by[s['object']].append(s)
    objs = sorted(by); draws = []
    for _ in range(2000 if S else 0):
        draws.append(evaluate([s for o in rng.choice(objs, len(objs)) for s in by[o]]))
    def ci(vals):
        v = [x for x in vals if x is not None and np.isfinite(x)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [None, None]
    for m in point:
        point[m]['ci95'] = {k: ci([d[m][k] if m in d and d[m].get(k) is not None else None for d in draws]) for k in keys}
        point[m]['ci95']['recall'] = {c: ci([d[m]['recall'][c] if m in d else None for d in draws]) for c in CLS}
    paired = {}
    for x, y in CONTRASTS:
        if x in point and y in point:
            paired[f'{x}-minus-{y}'] = {k: {'diff': (point[x][k] - point[y][k]) if None not in (point[x][k], point[y][k]) else None,
                                            'ci95': ci([(d[x][k] - d[y][k]) if (x in d and y in d and None not in (d[x][k], d[y][k])) else None for d in draws])}
                                        for k in keys}
    res = {'addendum': 'docs/v5g/PREREG_ADDENDUM.md', 'n_samples': len(S), 'n_objects': len(objs), 'n_candidates': len(cands),
           'thresholds': {'T_DROP_left': P.T_DROP, 'T_DMG_crush': P.T_DMG}, 'class_counts_gt': dict(Counter(v for s in S for v in s['true'].values())),
           'predictors': point, 'paired': paired, 'samples': [{k: s[k] for k in ('index', 'object', 'law', 'vis_law', 'true', 'pred')} for s in S]}
    (ROOT / args.out).write_text(json.dumps(res, indent=1))
    print('n samples', len(S), 'objects', len(objs), 'gt classes', res['class_counts_gt'])
    for m, e in point.items():
        print(f"{m:15s} F1={e['macro_f1']:.3f} {e['ci95']['macro_f1']} acc={e['accuracy']:.3f} safeP={e['safe_precision']} safeR={e['safe_recall']} feas={e['feasibility_accuracy']:.3f} rec={e['recall']}")
    for k, v in paired.items():
        print(k, 'F1', v['macro_f1'], 'safeP', v['safe_precision'])


if __name__ == '__main__':
    main()
