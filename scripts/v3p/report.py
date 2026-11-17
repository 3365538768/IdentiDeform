"""V3p: V3 vs V2 feed-forward prior comparison -> docs/v3p/RESULTS.json.

Every method is scored from evaluation-format predictions.jsonl (fieldbridge.evaluation output):
  * z-MAE (V2 tab:prior metric): pooled MAE over all scored (sample, parameter) entries in standardized specific
    space (E/rho, nu, rho, sigma_y/rho; ln1p + train z-score of THAT dataset's train split);
  * identifiable z-MAE: pooled over E/rho and sigma_y/rho entries only (refinement metric);
  * per-parameter z-MAE; class (law) accuracy;
  * within-class Pearson r (V2 protocol) and Spearman rho for E/rho and sigma_y/rho, mean over GT laws with >=6 entries;
  * decade errors: |log10(pred) - log10(GT)| of E/rho and sigma_y/rho (z decoded with the dataset's train stats),
    comparable across V2/V3 although their z-units differ (V3 ranges are narrower).
CIs: 95% percentile object bootstrap (2000 draws, seed 17): objects are resampled, every metric recomputed on the
pooled resample. For a 3-seed group the statistic is the mean over seeds of the per-seed metric on the same resample.
Trained runs: val = best-val epoch dir (scripts/r1_report.best_dir, = best_material.pt); test = <run>/test_once.
  .venv/bin/python scripts/v3p/report.py --v3-prefix v3main --v3-stats data/manifests/v3main_t2000.specific.stats.json [--v3-prefix-4k v3main4k ...]
"""
import argparse, json, math, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from r1_report import best_dir  # noqa: E402

P = ['E_over_rho', 'nu', 'rho', 'sigma_y_over_rho']; IDENT = [0, 3]; SEEDS = (17, 29, 43); DRAWS, SEED = 2000, 17


def load(d):
    rows = [json.loads(l) for l in (Path(d) / 'predictions.jsonl').read_text().splitlines() if l.strip()]
    return {r['sample_id']: r for r in rows}


def arrays(sets, stats):
    """sets: list of {sid: row} with identical sample ids -> stacked arrays (S, N, ...)."""
    sids = sorted(sets[0])
    for s in sets[1:]:
        if set(s) != set(sids):
            raise ValueError('member coverage mismatch')
    base = [sets[0][k] for k in sids]
    t = np.array([r['target_z'] for r in base]); m = np.array([r['target_mask'] for r in base], bool)
    z = np.array([[s[k]['prediction_z'] for k in sids] for s in sets])
    law = np.array([r['law_id'] for r in base]); pl = np.array([[s[k]['predicted_law_id'] for k in sids] for s in sets])
    obj = np.array([r['object_id'] for r in base])
    mean, std = np.array(stats['mean']), np.array(stats['std'])
    dec = lambda x: np.log10(np.maximum(np.expm1(np.clip(x * std + mean, 0, 40)), 1e-12))
    return dict(t=t, m=m, z=z, law=law, pl=pl, obj=obj, n=len(sids), dec_t=dec(t), dec_z=dec(z))


def corr(x, y, kind):
    if len(x) < 6 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return None
    return float(np.corrcoef(x, y)[0, 1]) if kind == 'pearson' else float(spearmanr(x, y)[0])


def metrics_one(A, s, idx):
    t, m, z, law = A['t'][idx], A['m'][idx], A['z'][s][idx], A['law'][idx]
    e = np.abs(z - t)
    out = {'z_mae': float(e[m].mean()), 'ident_z_mae': float(e[:, IDENT][m[:, IDENT]].mean()),
           'law_accuracy': float((A['pl'][s][idx] == law).mean())}
    for j, p in enumerate(P):
        out[f'z_mae_{p}'] = float(e[m[:, j], j].mean()) if m[:, j].any() else None
    for j in IDENT:
        de = np.abs(A['dec_z'][s][idx][:, j] - A['dec_t'][idx][:, j])[m[:, j]]
        out[f'decade_mae_{P[j]}'] = float(de.mean()) if len(de) else None
        out[f'decade_median_{P[j]}'] = float(np.median(de)) if len(de) else None
        for kind in ('pearson', 'spearman'):
            vals = [corr(z[(law == L) & m[:, j], j], t[(law == L) & m[:, j], j], kind) for L in sorted(set(law.tolist()))]
            vals = [v for v in vals if v is not None]
            out[f'within_law_{kind}_{P[j]}'] = float(np.mean(vals)) if vals else None
    de = np.abs(A['dec_z'][s][idx] - A['dec_t'][idx])[:, IDENT][m[:, IDENT]]
    out['decade_mae_ident'] = float(de.mean())
    return out


def score(sets, stats):
    """Point estimate (mean over members if several runs) + object-bootstrap CI + per-member values."""
    A = arrays(sets, stats); S = len(sets); allidx = np.arange(A['n'])
    per = [metrics_one(A, s, allidx) for s in range(S)]
    keys = per[0].keys()
    point = {k: (float(np.mean([p[k] for p in per if p[k] is not None])) if any(p[k] is not None for p in per) else None) for k in keys}
    objs = sorted(set(A['obj'].tolist())); groups = [np.where(A['obj'] == o)[0] for o in objs]
    rng = np.random.default_rng(SEED); boot = {k: [] for k in keys}
    for _ in range(DRAWS):
        idx = np.concatenate([groups[i] for i in rng.integers(len(groups), size=len(groups))])
        res = [metrics_one(A, s, idx) for s in range(S)]
        for k in keys:
            v = [r[k] for r in res if r[k] is not None]
            if v:
                boot[k].append(np.mean(v))
    out = {'samples': A['n'], 'objects': len(objs), 'members': S}
    for k in keys:
        out[k] = {'value': point[k], 'ci95': np.quantile(boot[k], [.025, .975]).tolist() if boot[k] else None}
        if S > 1:
            vals = [p[k] for p in per]
            out[k]['per_member'] = vals
            v = [x for x in vals if x is not None]
            out[k]['sd_over_members'] = float(np.std(v, ddof=1)) if len(v) > 1 else None
    return out


def ensemble(sets):
    """Equal-weight ensemble (rule of scripts/v3c/ensembles.py): mean z, majority-vote law (ties -> lowest id)."""
    out = {}
    for sid in sets[0]:
        rs = [s[sid] for s in sets]; votes = np.bincount([r['predicted_law_id'] for r in rs], minlength=3)
        out[sid] = dict(rs[0], prediction_z=np.mean([r['prediction_z'] for r in rs], 0).tolist(), logvar=[0.] * 4,
                        predicted_law_id=int(np.argmax(votes)), law_votes=votes.tolist())
    return out


def run_dir(run, split):
    return best_dir(ROOT / 'runs' / run)[0] if split == 'val' else ROOT / 'runs' / run / 'test_once'


def dataset_block(prefix, stats, splits, baselines, extra_groups=None):
    st = json.loads(Path(ROOT / stats).read_text()); res = {'stats': stats, 'train_count': st['count'],
        'train_std_ln': st['std'], 'train_std_decades': {'E_over_rho': st['std'][0] / math.log(10), 'sigma_y_over_rho': st['std'][3] / math.log(10)},
        'splits': {}}
    for split in splits:
        blk = {}
        runs = {g: [f'{prefix}_{g}_seed{s}' for s in SEEDS] for g in 'BCD'}
        avail = {g: [r for r in rs if (run_dir(r, split) / 'predictions.jsonl').exists()] if all((ROOT / 'runs' / r / 'timing.jsonl').exists() for r in rs) else [] for g, rs in runs.items()}
        sets = {g: [load(run_dir(r, split)) for r in rs] for g, rs in avail.items() if rs}
        for g, ss in sets.items():
            blk[f'{g} (3 seeds)' if len(ss) == 3 else f'{g} ({len(ss)} seeds)'] = score(ss, st)
        allm = [s for g in 'BCD' for s in sets.get(g, [])]
        if len(allm) > 1:
            blk[f'ensemble_{"".join(g for g in "BCD" if g in sets)}{len(allm)}'] = score([ensemble(allm)], st)
        for name, pat in (extra_groups or {}).items():
            rs = [pat.format(s) for s in SEEDS]
            if all((ROOT / 'runs' / r / 'timing.jsonl').exists() and (run_dir(r, split) / 'predictions.jsonl').exists() for r in rs):
                blk[f'{name} (3 seeds)'] = score([load(run_dir(r, split)) for r in rs], st)
        for name, d in baselines.get(split, {}).items():
            if (ROOT / d / 'predictions.jsonl').exists():
                blk[name] = score([load(ROOT / d)], st)
        res['splits'][split] = blk
    return res


def main():
    a = argparse.ArgumentParser(); a.add_argument('--v3-prefix', default='v3main'); a.add_argument('--v3-stats', default='data/manifests/v3main_t2000.specific.stats.json')
    a.add_argument('--v3-prefix-4k'); a.add_argument('--v3-stats-4k'); a.add_argument('--splits', default='val,test')
    a.add_argument('--output', default='docs/v3p/RESULTS_raw.json'); args = a.parse_args()
    splits = args.splits.split(',')
    rep = {'protocol': __doc__.strip().splitlines()[0], 'metrics_doc': __doc__}
    rep['V2'] = dataset_block('v2main', 'data/manifests/v2main.specific.stats.json', splits,
                              {'val': {'constant_train_mean': 'runs/v3c_const', 'class_mean_gt_privileged': 'runs/v3c_lawmean_gt'},
                               'test': {'constant_train_mean': 'runs/v3c_const_test', 'class_mean_gt_privileged': 'runs/v3c_lawmean_gt_test'}},
                              {'dinov3_first_frame_B': 'v3c_dino_S_seed{}'})
    rep['V3'] = dataset_block(args.v3_prefix, args.v3_stats, splits,
                              {s: {'constant_train_mean': f'runs/{args.v3_prefix}_const_{s}', 'class_mean_gt_privileged': f'runs/{args.v3_prefix}_lawmean_gt_{s}'} for s in splits},
                              {'dinov3_first_frame_B': args.v3_prefix + '_dinoS_seed{}'})
    if args.v3_prefix_4k:
        rep['V3_4k'] = dataset_block(args.v3_prefix_4k, args.v3_stats_4k, splits,
                                     {s: {'constant_train_mean': f'runs/{args.v3_prefix_4k}_const_{s}', 'class_mean_gt_privileged': f'runs/{args.v3_prefix_4k}_lawmean_gt_{s}'} for s in splits})
    Path(ROOT / args.output).write_text(json.dumps(rep, indent=1) + '\n'); print('wrote', args.output)


if __name__ == '__main__':
    main()
