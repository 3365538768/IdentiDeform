"""V3f follow-up: do FROZEN V-JEPA2 features carry stiffness information in external domains? (CPU only)

Domains (cached features only, no new downloads):
  springgaus : data/prepared/v3f_springgaus_syn_vjl  (14 homogeneous elastic scenes x 3 camera triplets)
  pacnerf    : data/prepared/v3c_pacnerf_vjl, variant bgsub (4 scenes x 3 triplets; nu is 0.3 everywhere -> E only)
  v2jelly    : data/prepared/v2main_vjl, V2 val jelly samples (sanity reference; 1 row per sample, sample = "scene")
Features:
  frozen : mean over views x time x 14x14 tokens of the cached V-JEPA2 ViT-L last hidden state (1024-d)
  head   : input of the last Linear of param_head (penultimate) of each V2 main run (9 runs B/C/D x 3 seeds);
           one probe per run, predictions averaged over the 9 runs
Probes (target = train-standardised z of ln1p(E/rho*1000) and of nu, V2 stats):
  ridge (linear kernel X X^T / d on support-standardised features), krr (RBF, gamma = 1/median sq. distance),
  support_mean (no video). alpha in logspace(-3, 3, 13), chosen by inner leave-one-scene-out on the support only.
  Training rows = all triplets of the support scenes; held-out prediction = canonical triplet t0.
Protocol: outer leave-one-scene-out (K = all other scenes) + K curve (K = 3, 6, 9, 13; 20 random support subsets).
Metrics over held-out scenes: MAE_z, Spearman, pairwise order accuracy (ties in GT excluded); bootstrap CI over scenes.
  python -I scripts/v3f/fewshot_probe.py <out.json>
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import springgaus_fewshot as sf  # noqa: E402

ROOT = sf.ROOT
ALPHAS = np.logspace(-3, 3, 13)
RNG = np.random.default_rng(17)
torch.set_num_threads(16)


def pooled(feat):
    return feat.float().mean(dim=(0, 1, 3, 4)).numpy()  # [V,T,C,H,W] -> C


def load_domain(name):
    rows = []
    if name == 'springgaus':
        d = ROOT / 'data/prepared/v3f_springgaus_syn_vjl'; idx = json.loads((d / 'index.json').read_text())
        for sid, f in sorted(idx['files'].items()):
            sc = sid.split('/')[1]
            if not idx['scenes'][sc]['homogeneous']:
                continue
            rows.append((sc, int(sid[-1]), torch.load(d / f, weights_only=True)))
    elif name == 'pacnerf':
        d = ROOT / 'data/prepared/v3c_pacnerf_vjl'; idx = json.loads((d / 'index.json').read_text())
        for sid, f in sorted(idx['files'].items()):
            if '/bgsub/' in sid:
                rows.append((sid.split('/')[1], int(sid[-1]), torch.load(d / f, weights_only=True)))
    elif name == 'v2jelly':
        d = ROOT / 'data/prepared/v2main_vjl'; idx = json.loads((d / 'index.json').read_text())['files']
        preds = [json.loads(l) for l in (ROOT / 'runs/v3c_ens_bcd9_val/predictions.jsonl').read_text().splitlines()]
        for p in preds:
            if p['law_id'] == 0:
                s = torch.load(d / idx[p['sample_id']], weights_only=False)
                s = {k: s[k] for k in ('features', 'rgb', 'valid', 'times', 'prior', 'prior_mask', 'target', 'target_mask', 'action')}
                rows.append((p['sample_id'], 0, s))
    return rows


def head_features(rows):
    out = []
    for r in sf.RUNS:
        model, _ = sf.load_model(r, 'cpu')
        _, _, h = sf.run(model, [s for _, _, s in rows], 'cpu')
        out.append(h.numpy()); del model
        print('head', r, flush=True)
    return np.stack(out)  # [R,N,d]


def kfit_predict(Xs, ys, Xq, kind, alpha):
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-6
    A, B = (Xs - mu) / sd, (Xq - mu) / sd
    if kind == 'ridge':
        K, k = A @ A.T / A.shape[1], B @ A.T / A.shape[1]
    else:
        d2 = ((A[:, None] - A[None]) ** 2).sum(-1); g = 1.0 / max(np.median(d2[d2 > 0]), 1e-9) if (d2 > 0).any() else 1.0
        K = np.exp(-g * d2); k = np.exp(-g * ((B[:, None] - A[None]) ** 2).sum(-1))
    ym = ys.mean()
    w = np.linalg.solve(K + alpha * np.eye(len(K)), ys - ym)
    return ym + k @ w


def fit_predict(X, y, scene, sup, q_rows, kind):
    """X rows, y targets, scene labels; sup = list of support scenes; inner LOSO chooses alpha."""
    S = np.isin(scene, sup)
    if kind == 'support_mean':
        return np.full(len(q_rows), y[S].mean()), None
    best, ba = np.inf, ALPHAS[len(ALPHAS) // 2]
    if len(sup) >= 2:
        for a in ALPHAS:
            err = []
            for s in sup:
                tr = S & (scene != s); te = scene == s
                err.append(np.abs(kfit_predict(X[tr], y[tr], X[te], kind, a) - y[te]).mean())
            if np.mean(err) < best:
                best, ba = np.mean(err), a
    return kfit_predict(X[S], y[S], X[q_rows], kind, ba), float(ba)


def metrics(pred, gt):
    from scipy.stats import spearmanr
    pred, gt = np.asarray(pred), np.asarray(gt)
    pairs = [(i, j) for i in range(len(gt)) for j in range(i + 1, len(gt)) if gt[i] != gt[j]]
    pa = float(np.mean([(pred[i] - pred[j]) * (gt[i] - gt[j]) > 0 for i, j in pairs])) if pairs else None
    sp = float(spearmanr(pred, gt).statistic) if np.std(pred) > 0 and np.std(gt) > 0 else None
    return {'mae_z': float(np.abs(pred - gt).mean()), 'spearman': sp, 'pairwise_order_acc': pa, 'n_pairs': len(pairs)}


def boot(pred, gt, n=2000):
    pred, gt = np.asarray(pred), np.asarray(gt); out = {'mae_z': [], 'spearman': [], 'pairwise_order_acc': []}
    rng = np.random.default_rng(0)
    for _ in range(n):
        i = rng.integers(0, len(gt), len(gt)); m = metrics(pred[i], gt[i])
        for k in out:
            if m[k] is not None and not np.isnan(m[k]):
                out[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else None for k, v in out.items()}


def method_preds(feats, y, scene, trip, names, sup_of, kind):
    """feats: dict name->X [N,d] or [R,N,d]; returns held-out t0 prediction per scene in names."""
    preds = {}
    for q in names:
        qr = np.where((scene == q) & (trip == 0))[0]
        X = feats
        if X.ndim == 3:
            p = np.mean([fit_predict(X[r], y, scene, sup_of(q), qr, kind)[0] for r in range(X.shape[0])], 0)
        else:
            p = fit_predict(X, y, scene, sup_of(q), qr, kind)[0]
        preds[q] = float(p[0])
    return preds


def run_loso(F, H, y, scene, trip, names, label):
    res = {}
    for meth, X, kind in [('frozen_ridge', F, 'ridge'), ('frozen_krr', F, 'krr'), ('head_ridge', H, 'ridge'), ('head_krr', H, 'krr'),
                          ('support_mean', F, 'support_mean')]:
        p = method_preds(X, y, scene, trip, names, lambda q: [s for s in names if s != q], kind)
        gt = [float(y[np.where((scene == s) & (trip == 0))[0][0]]) for s in names]
        pr = [p[s] for s in names]
        res[meth] = {'metrics': metrics(pr, gt), 'ci95_bootstrap_over_scenes': boot(pr, gt), 'pred': p}
        print(label, meth, json.dumps(res[meth]['metrics']), flush=True)
    return res


def k_curve(F, H, y, scene, trip, names, Ks=(3, 6, 9, 13), reps=20):
    out = {}
    for K in Ks:
        out[K] = {}
        for meth, X, kind in [('frozen_ridge', F, 'ridge'), ('frozen_krr', F, 'krr'), ('head_ridge', H, 'ridge'), ('support_mean', F, 'support_mean')]:
            ms = []
            for r in range(1 if K >= len(names) - 1 else reps):
                rng = np.random.default_rng(1000 * K + r)
                sup = {q: list(rng.choice([s for s in names if s != q], size=K, replace=False)) for q in names}
                p = method_preds(X, y, scene, trip, names, lambda q: sup[q], kind)
                gt = [float(y[np.where((scene == s) & (trip == 0))[0][0]]) for s in names]
                ms.append(metrics([p[s] for s in names], gt))
            agg = {}
            for k in ('mae_z', 'spearman', 'pairwise_order_acc'):
                v = [m[k] for m in ms if m[k] is not None]
                agg[k] = {'mean': float(np.mean(v)), 'p2.5': float(np.percentile(v, 2.5)), 'p97.5': float(np.percentile(v, 97.5))}
            out[K][meth] = {'repeats': len(ms), **agg}
            print('Kcurve', K, meth, json.dumps({k: round(v['mean'], 3) for k, v in agg.items()}), flush=True)
    return out


def main(outp):
    t0 = time.time(); report = {'protocol': __doc__, 'domains': {}}
    for dom in ('springgaus', 'pacnerf'):
        rows = load_domain(dom)
        scene = np.array([r[0] for r in rows]); trip = np.array([r[1] for r in rows])
        F = np.stack([pooled(r[2]['features']) for r in rows]); H = head_features(rows)
        T = np.stack([r[2]['target'].numpy() for r in rows]); names = sorted(set(scene))
        dres = {'n_scenes': len(names), 'rows': len(rows)}
        for pi, pname in [(0, 'E_over_rho'), (1, 'nu')]:
            if np.std(T[:, pi]) < 1e-6:
                dres[pname] = 'constant GT (skipped)'; continue
            dres[pname] = {'loso': run_loso(F, H, T[:, pi], scene, trip, names, f'{dom}/{pname}')}
            if dom == 'springgaus':
                dres[pname]['k_curve'] = k_curve(F, H, T[:, pi], scene, trip, names)
        report['domains'][dom] = dres
        Path(outp).write_text(json.dumps(report, indent=1, default=float))
    # V2 val jelly sanity reference: 14 random samples, LOO over samples; 20 draws; plus 1-decade range-matched draws
    rows = load_domain('v2jelly')
    F = np.stack([pooled(r[2]['features']) for r in rows]); H = head_features(rows)
    T = np.stack([r[2]['target'].numpy() for r in rows]); ids = np.array([r[0] for r in rows])
    lg = (T[:, 0] * sf.json.loads(sf.STATS.read_text())['std'][0] + json.loads(sf.STATS.read_text())['mean'][0]) / np.log(10)
    v2 = {'n_val_jelly': len(rows)}
    for variant, pool in [('random14', np.arange(len(rows))), ('range_matched_1decade_3.5_4.5', np.where((lg >= 3.5) & (lg <= 4.5))[0])]:
        per = {m: [] for m in ('frozen_ridge', 'frozen_krr', 'head_ridge', 'support_mean')}
        for r in range(20):
            sel = np.random.default_rng(500 + r).choice(pool, size=min(14, len(pool)), replace=False)
            names = list(ids[sel]); scene = ids[sel]; trip = np.zeros(len(sel), int)
            for meth, X, kind in [('frozen_ridge', F[sel], 'ridge'), ('frozen_krr', F[sel], 'krr'), ('head_ridge', H[:, sel], 'ridge'), ('support_mean', F[sel], 'support_mean')]:
                p = method_preds(X, T[sel, 0], scene, trip, names, lambda q: [s for s in names if s != q], kind)
                per[meth].append(metrics([p[s] for s in names], list(T[sel, 0])))
        v2[variant] = {'pool_size': int(len(pool)), **{m: {k: {'mean': float(np.mean([x[k] for x in v if x[k] is not None])),
                                                           'p2.5': float(np.percentile([x[k] for x in v if x[k] is not None], 2.5)),
                                                           'p97.5': float(np.percentile([x[k] for x in v if x[k] is not None], 97.5))}
                                                       for k in ('mae_z', 'spearman', 'pairwise_order_acc')} for m, v in per.items()}}
        print('v2jelly', variant, json.dumps({m: round(v2[variant][m]['spearman']['mean'], 3) for m in per}), flush=True)
    report['v2_val_jelly_reference_E_over_rho'] = v2
    report['seconds'] = time.time() - t0
    Path(outp).write_text(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    main(sys.argv[1])
