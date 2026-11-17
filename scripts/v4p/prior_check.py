"""V4 prior usefulness check: V4 (gravity 2 on every action) vs V3 (paired: same val indices, objects, parameters),
both trained on train indices 0-999 with the unchanged V2 recipe (configs/v2main/*, own train standardization).
Val predictions = best val epoch of each run (scripts/r1_report.best_dir). Writes docs/v4/PRIOR_CHECK.json + _ZH.md.

Metrics per arm/group (3-seed groups: per-seed metric averaged over available seeds; ensemble = mean z, vote law):
  decade MAE of E/rho and sigma_y/rho, gain over the constant (train-mean) baseline = 1 - MAE_model / MAE_const,
  class accuracy, within-class Spearman (mean over classes; per action; per class), z-MAE.
CIs: object bootstrap (2000, seed 17). V4-V3 paired differences resample the same val objects (index-paired).
  .venv/bin/python scripts/v4p/prior_check.py
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from r1_report import best_dir  # noqa: E402
SEEDS = (17, 29, 43); LAWS = ['jelly', 'metal', 'plasticine']; DRAWS = 2000
ARMS = {'V3_t1000': ('v3t1k', 'data/manifests/v3main_t1000'), 'V4_t1000': ('v4t1k', 'data/manifests/v4main_t1000')}


def load(d):
    return {r['sample_id']: r for r in map(json.loads, (Path(d) / 'predictions.jsonl').read_text().splitlines())}


def idx(sid):
    return sid.split('/', 1)[1]  # 'val/000123' -> pairing key


def sp(x, y):
    return float(spearmanr(x, y)[0]) if len(x) >= 6 and np.std(x) > 1e-9 and np.std(y) > 1e-9 else None


class Arm:
    def __init__(self, sets, stats, keys, manifest):
        mean, std = np.array(stats['mean']), np.array(stats['std'])
        dec = lambda z: np.log10(np.maximum(np.expm1(np.clip(z * std + mean, 0, 40)), 1e-12))
        rows = {r['sample_id']: r for r in map(json.loads, Path(manifest + '.jsonl').read_text().splitlines())}
        base = [{idx(k): v for k, v in s.items()} for s in sets]
        b0 = base[0]
        self.t = np.array([b0[k]['target_z'] for k in keys]); self.m = np.array([b0[k]['target_mask'] for k in keys], bool)
        self.z = np.array([[s[k]['prediction_z'] for k in keys] for s in base]); self.pl = np.array([[s[k]['predicted_law_id'] for k in keys] for s in base])
        self.law = np.array([b0[k]['law_id'] for k in keys]); self.obj = np.array([b0[k]['object_id'] for k in keys])
        self.act = np.array([b0[k]['action'] for k in keys])
        self.dt, self.dz = dec(self.t), dec(self.z); self.dc = dec(np.zeros_like(self.t))
        self.S = len(sets)

    def metrics(self, ii, detail=False):
        out = {}
        m, t, law = self.m[ii], self.t[ii], self.law[ii]
        per = []
        for s in range(self.S):
            z = self.z[s][ii]; r = {'z_mae': float(np.abs(z - t)[m].mean()), 'law_accuracy': float((self.pl[s][ii] == law).mean())}
            for j, n in ((0, 'E_over_rho'), (3, 'sigma_y_over_rho')):
                mm = m[:, j]; e = np.abs(self.dz[s][ii][mm, j] - self.dt[ii][mm, j]); c = np.abs(self.dc[ii][mm, j] - self.dt[ii][mm, j])
                r[f'decade_mae_{n}'] = float(e.mean()); r[f'const_decade_mae_{n}'] = float(c.mean()); r[f'gain_over_const_{n}'] = float(1 - e.mean() / c.mean())
                vals = [sp(z[(law == L) & mm, j], t[(law == L) & mm, j]) for L in range(3)]
                v = [x for x in vals if x is not None]; r[f'within_class_spearman_{n}'] = float(np.mean(v)) if v else None
                if detail:
                    r[f'per_class_spearman_{n}'] = {LAWS[L]: vals[L] for L in range(3)}
                    pa = {}
                    for a in sorted(set(self.act[ii])):
                        k = self.act[ii] == a
                        vv = [sp(z[k & (law == L) & mm, j], t[k & (law == L) & mm, j]) for L in range(3)]
                        vv = [x for x in vv if x is not None]
                        pa[a] = float(np.mean(vv)) if vv else None
                    r[f'per_action_spearman_{n}'] = pa
            if detail:
                r['per_class_accuracy'] = {LAWS[L]: float((self.pl[s][ii][law == L] == L).mean()) for L in range(3)}
                r['per_action_accuracy'] = {a: float((self.pl[s][ii][self.act[ii] == a] == law[self.act[ii] == a]).mean()) for a in sorted(set(self.act[ii]))}
            per.append(r)
        def avg(x):
            if isinstance(x[0], dict):
                return {k: avg([y[k] for y in x]) for k in x[0]}
            v = [y for y in x if y is not None]
            return float(np.mean(v)) if v else None
        return avg(per)


def ens(sets):
    out = {}
    for sid in sets[0]:
        rs = [s[sid] for s in sets]; votes = np.bincount([r['predicted_law_id'] for r in rs], minlength=3)
        out[sid] = dict(rs[0], prediction_z=np.mean([r['prediction_z'] for r in rs], 0).tolist(), predicted_law_id=int(np.argmax(votes)))
    return out


def boot(arms, keys, fn, seed=17):
    obj = arms[0].obj; objs = sorted(set(obj)); g = [np.where(obj == o)[0] for o in objs]
    rng = np.random.default_rng(seed); vals = []
    for _ in range(DRAWS):
        ii = np.concatenate([g[i] for i in rng.integers(len(g), size=len(g))]); vals.append(fn(ii))
    return vals


def flat(d, pre=''):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(flat(v, pre + k + '.'))
        else:
            out[pre + k] = v
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument('--arm', action='append', default=[], metavar='NAME=RUNPREFIX:MANIFEST',
        help='arm definition; default V3_t1000=v3t1k:data/manifests/v3main_t1000 and V4_t1000=v4t1k:data/manifests/v4main_t1000. '
             'The data root / per-action source of each arm is fixed by its manifest (scripts/v3p/build_manifest.py --root/--alt).')
    ap.add_argument('--base', default='V3_t1000', help='arm subtracted in the paired differences'); ap.add_argument('--out', default='docs/v4/PRIOR_CHECK')
    a = ap.parse_args()
    if a.arm:
        ARMS.clear()
        for spec in a.arm:
            n, rest = spec.split('='); pre, man = rest.split(':', 1); ARMS[n] = (pre, man)
    groups = {}
    for arm, (prefix, manifest) in ARMS.items():
        if not (ROOT / f'{manifest}.specific.stats.json').exists():
            continue
        stats = json.loads((ROOT / f'{manifest}.specific.stats.json').read_text())
        for g in 'BCD':
            sets = []
            for s in SEEDS:
                r = ROOT / 'runs' / f'{prefix}_{g}_seed{s}'
                if (r / 'status.json').exists() and json.loads((r / 'status.json').read_text()).get('status') in ('early_stopped', 'training_finished'):
                    sets.append((s, load(best_dir(r)[0])))
            if sets:
                groups.setdefault(arm, {})[g] = (sets, stats, manifest)
    # pairing keys: val indices present in every arm
    keys = None
    for arm, gs in groups.items():
        for g, (sets, _, _) in gs.items():
            k = {idx(x) for x in sets[0][1]}
            keys = k if keys is None else keys & k
    keys = sorted(keys or [])
    rep = {'arms_def': {k: list(v) for k, v in ARMS.items()}, 'written_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'doc': __doc__, 'paired_val_samples': len(keys), 'arms': {}, 'paired_v4_minus_v3': {}}
    objs = {}
    for arm, gs in groups.items():
        rep['arms'][arm] = {}
        allsets = []
        for g, (sets, stats, manifest) in gs.items():
            A = Arm([x[1] for x in sets], stats, keys, manifest); objs[arm] = A
            rep['arms'][arm][g] = {'seeds': [x[0] for x in sets], 'train_std_decades': {'E_over_rho': stats['std'][0] / math.log(10), 'sigma_y_over_rho': stats['std'][3] / math.log(10)}}
            allsets += [x[1] for x in sets]
        if len(allsets) > 1:
            gs = dict(gs); gs[f'ensemble_{len(allsets)}'] = ([(0, ens(allsets))], stats, manifest)
        for g, (sets, stats, manifest) in gs.items():
            A = Arm([x[1] for x in sets], stats, keys, manifest)
            allii = np.arange(len(keys)); point = A.metrics(allii, detail=True)
            bs = boot([A], keys, lambda ii: flat(A.metrics(ii)))
            cis = {k: np.quantile([b[k] for b in bs if b.get(k) is not None], [.025, .975]).tolist() for k in bs[0] if any(b.get(k) is not None for b in bs)}
            rep['arms'][arm].setdefault(g, {}).update(point=point, ci95=cis, seeds=[x[0] for x in sets], train_std_decades=rep['arms'][arm].get(g, {}).get('train_std_decades'))
            objs[(arm, g)] = A
    # paired V4 - V3 (same group)
    for other in [n for n in rep['arms'] if n != a.base]:
      for g in sorted(set(rep['arms'].get(a.base, {})) & set(rep['arms'][other])):
          a3, a4 = objs[(a.base, g)], objs[(other, g)]
          if list(a3.obj) != list(a4.obj):
              raise ValueError('pairing mismatch')
          KS = ('decade_mae_E_over_rho', 'decade_mae_sigma_y_over_rho', 'gain_over_const_E_over_rho', 'gain_over_const_sigma_y_over_rho', 'law_accuracy',
                'within_class_spearman_E_over_rho', 'within_class_spearman_sigma_y_over_rho')
          def diff(ii, a3=a3, a4=a4, KS=KS):
              x4, x3 = flat(a4.metrics(ii)), flat(a3.metrics(ii))
              return {k: (x4[k] - x3[k]) if x4[k] is not None and x3[k] is not None else None for k in KS}
          pt = diff(np.arange(len(keys))); bs = boot([a3], keys, diff)
          rep['paired_v4_minus_v3'][f'{other} - {a.base} / {g}'] = {k: {'value': pt[k], 'ci95': np.quantile([b[k] for b in bs if b[k] is not None], [.025, .975]).tolist()} for k in pt if pt[k] is not None}
    # privileged GT-class-mean reference per arm (decades; shows how much of the gain the class ranges alone give)
    for arm, (prefix, manifest) in ARMS.items():
        if arm not in rep['arms']:
            continue
        st = json.loads((ROOT / f'{manifest}.specific.stats.json').read_text()); mean, std = np.array(st['mean']), np.array(st['std'])
        from fieldbridge.data import ParameterCodec, specific_parameters
        codec = ParameterCodec(st); rows = [json.loads(l) for l in (ROOT / f'{manifest}.jsonl').read_text().splitlines()]
        enc = lambda r: [x.numpy() for x in codec.encode(specific_parameters(r['parameters']))]
        tr = [(r, *enc(r)) for r in rows if r['split'] == 'train']; va = [(r, *enc(r)) for r in rows if r['split'] == 'val' and idx(r['sample_id']) in set(keys)]
        dec = lambda z: np.log10(np.maximum(np.expm1(np.clip(z * std + mean, 0, 40)), 1e-12))
        lm = {}
        for L in range(3):
            z = np.array([t for r, t, m in tr if r['law_id'] == L]); m = np.array([m for r, t, m in tr if r['law_id'] == L])
            lm[L] = np.array([z[m[:, j], j].mean() if m[:, j].any() else 0. for j in range(4)])
        ref_arm = {}
        for j, n in ((0, 'E_over_rho'), (3, 'sigma_y_over_rho')):
            t = np.array([dec(tt)[j] for r, tt, m in va if m[j]]); cm = np.array([dec(lm[r['law_id']])[j] for r, tt, m in va if m[j]]); c0 = dec(np.zeros(4))[j]
            ref_arm[f'decade_mae_{n}'] = float(np.abs(cm - t).mean()); ref_arm[f'const_decade_mae_{n}'] = float(np.abs(c0 - t).mean())
            ref_arm[f'gain_over_const_{n}'] = 1 - ref_arm[f'decade_mae_{n}'] / ref_arm[f'const_decade_mae_{n}']
        rep['arms'][arm]['class_mean_gt_privileged'] = {'point_only': ref_arm}
    # references
    rr = json.loads((ROOT / 'docs/v3p/RESULTS_raw.json').read_text())
    ref = {}
    for ds, label in (('V2', 'V2 main (1955 train)'), ('V3', 'V3 main (2000 train)')):
        for m in ('B (3 seeds)', 'ensemble_BCD9', 'constant_train_mean'):
            x = rr[ds]['splits']['val'][m]
            ref[f'{label} / {m}'] = {k: x[k]['value'] for k in ('decade_mae_E_over_rho', 'decade_mae_sigma_y_over_rho', 'law_accuracy', 'within_law_spearman_E_over_rho', 'within_law_spearman_sigma_y_over_rho', 'z_mae')}
        for m in ('B (3 seeds)', 'ensemble_BCD9'):
            for n in ('E_over_rho', 'sigma_y_over_rho'):
                ref[f'{label} / {m}'][f'gain_over_const_{n}'] = 1 - ref[f'{label} / {m}'][f'decade_mae_{n}'] / ref[f'{label} / constant_train_mean'][f'decade_mae_{n}']
    rep['references_val'] = ref
    (ROOT / f'{a.out}.json').write_text(json.dumps(rep, indent=1) + '\n')
    # ZH summary
    f = lambda x, ci=None: ('–' if x is None else (f'{x:.3f}' + (f' [{ci[0]:.3f}, {ci[1]:.3f}]' if ci else '')))
    L = [f'# 先验可用性快检（{", ".join(ARMS)}；train 0–999，val 配对）', '', f'生成 {rep["written_utc"]}；配对 val 样本 {len(keys)}。源 `{a.out}.json`。V2 配方不变（configs/v2main/*，各自 train 标准化）。', '',
         '| 组 | 种子 | E/ρ 误差 dec | 常数 dec | 相对常数增益 E/ρ | σ/ρ 误差 dec | 常数 dec | 相对常数增益 σ/ρ | 类别准确率 | 族内 r_S(E/ρ) | 族内 r_S(σ/ρ) |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for arm, gs in rep['arms'].items():
        for g, x in gs.items():
            if 'point_only' in x:
                p = x['point_only']
                L.append(f"| {arm} GT类均值(特权) | – | {f(p['decade_mae_E_over_rho'])} | {f(p['const_decade_mae_E_over_rho'])} | {f(p['gain_over_const_E_over_rho'])} | {f(p['decade_mae_sigma_y_over_rho'])} | {f(p['const_decade_mae_sigma_y_over_rho'])} | {f(p['gain_over_const_sigma_y_over_rho'])} | 1 | – | – |")
                continue
            if 'point' not in x:
                continue
            p, c = x['point'], x['ci95']
            L.append(f"| {arm} {g} | {x['seeds']} | {f(p['decade_mae_E_over_rho'], c.get('decade_mae_E_over_rho'))} | {f(p['const_decade_mae_E_over_rho'])} | {f(p['gain_over_const_E_over_rho'], c.get('gain_over_const_E_over_rho'))} | "
                     f"{f(p['decade_mae_sigma_y_over_rho'], c.get('decade_mae_sigma_y_over_rho'))} | {f(p['const_decade_mae_sigma_y_over_rho'])} | {f(p['gain_over_const_sigma_y_over_rho'], c.get('gain_over_const_sigma_y_over_rho'))} | "
                     f"{f(p['law_accuracy'], c.get('law_accuracy'))} | {f(p['within_class_spearman_E_over_rho'], c.get('within_class_spearman_E_over_rho'))} | {f(p['within_class_spearman_sigma_y_over_rho'], c.get('within_class_spearman_sigma_y_over_rho'))} |")
    L += ['', '## 配对差（同组；物体 bootstrap 95% CI；按 val 索引配对）', '', '| 组 | 指标 | 差值 |', '|---|---|---|']
    for g, d in rep['paired_v4_minus_v3'].items():
        for k, x in d.items():
            L.append(f"| {g} | {k} | {f(x['value'], x['ci95'])} |")
    L += ['', '## 每动作 / 每类（点估计）', '']
    for arm, gs in rep['arms'].items():
        for g, x in gs.items():
            if 'point' not in x or not g.startswith('ens') and g != 'B':
                continue
            p = x['point']
            L.append(f"**{arm} {g}** 类别准确率/动作 {json.dumps({k: round(v, 2) for k, v in p['per_action_accuracy'].items()})}；/类 {json.dumps({k: round(v, 2) for k, v in p['per_class_accuracy'].items()})}")
            for n in ('E_over_rho', 'sigma_y_over_rho'):
                L.append(f"- r_S({n}) /动作 {json.dumps({k: (round(v, 2) if v is not None else None) for k, v in p[f'per_action_spearman_{n}'].items()})}；/类 {json.dumps({k: (round(v, 2) if v is not None else None) for k, v in p[f'per_class_spearman_{n}'].items()})}")
            L.append('')
    L += ['## 参考（val，全量训练）', '', '| 参考 | E/ρ dec | σ/ρ dec | 增益 E/ρ | 增益 σ/ρ | 类别准确率 | r_S(E/ρ) | r_S(σ/ρ) |', '|---|---|---|---|---|---|---|---|']
    for k, x in ref.items():
        L.append(f"| {k} | {f(x['decade_mae_E_over_rho'])} | {f(x['decade_mae_sigma_y_over_rho'])} | {f(x.get('gain_over_const_E_over_rho'))} | {f(x.get('gain_over_const_sigma_y_over_rho'))} | {f(x['law_accuracy'])} | {f(x['within_law_spearman_E_over_rho'])} | {f(x['within_law_spearman_sigma_y_over_rho'])} |")
    (ROOT / f'{a.out}_ZH.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
