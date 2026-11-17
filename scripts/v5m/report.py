"""V5m report: vision / vision+action / vision+touch+action / touch+action (+ V5f-mixed) multimodal priors on V5b val/test
and the V5f force-press val subset. Decade errors for E/rho, sigma_y/rho, rho, E (constant = train mean, privileged
GT-class-mean oracle = train mean of the true class), class accuracy, within-class Spearman (per action / per class),
3-seed groups (per-seed metric averaged) and 3-seed ensembles (mean z, vote law); paired differences vs the vision arm
(object bootstrap, 2000 draws, seed 17). Writes docs/v5m/RESULTS.json and RESULTS_ZH.md.
"""
import json, time
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[2]
ARMS = ['vision', 'vision_action', 'vision_touch_action', 'touch_action', 'vta_mixv5f']
LAB = {'vision': '视觉', 'vision_action': '视觉+动作', 'vision_touch_action': '视觉+触觉+动作', 'touch_action': '触觉+动作', 'vta_mixv5f': '视觉+触觉+动作（混入 V5f）'}
SEEDS = (29, 17, 43); TG = {'E_over_rho': 0, 'sigma_y_over_rho': 3, 'rho': 2, 'E': 4}; LAWS = ['jelly', 'metal', 'plasticine']; DRAWS = 2000


def sp(x, y):
    return float(spearmanr(x, y)[0]) if len(x) >= 6 and np.std(x) > 1e-9 and np.std(y) > 1e-9 else None


def load(p):
    return [json.loads(l) for l in open(p)]


class Set:
    def __init__(self, members, st, lawmean):
        r0 = members[0]
        self.t = np.array([r['target_z'] for r in r0]); self.m = np.array([r['target_mask'] for r in r0], bool)
        self.z = np.array([[r['prediction_z'] for r in mem] for mem in members]); self.pl = np.array([[r['predicted_law_id'] for r in mem] for mem in members])
        self.law = np.array([r['law_id'] for r in r0]); self.obj = np.array([r['object_id'] for r in r0]); self.act = np.array([r['action'] for r in r0])
        mean, std = st['y_mean'].numpy(), st['y_std'].numpy()
        self.dec = lambda z: np.log10(np.maximum(np.expm1(np.clip(z * std + mean, 0, 40)), 1e-12))
        self.dt = self.dec(self.t); self.dz = self.dec(self.z); self.dc = self.dec(np.zeros(5)); self.do = self.dec(np.array([lawmean[l] for l in self.law]))

    def metrics(self, ii, detail=False):
        per = []
        for s in range(len(self.z)):
            z = self.z[s][ii]; t, m, law = self.t[ii], self.m[ii], self.law[ii]; r = {'law_accuracy': float((self.pl[s][ii] == law).mean())}
            for n, j in TG.items():
                mm = m[:, j]; e = np.abs(self.dz[s][ii][mm, j] - self.dt[ii][mm, j])
                c = np.abs(self.dc[j] - self.dt[ii][mm, j]); o = np.abs(self.do[ii][mm, j] - self.dt[ii][mm, j])
                r[f'dec_{n}'] = float(e.mean()) if len(e) else None; r[f'const_{n}'] = float(c.mean()) if len(e) else None; r[f'oracle_{n}'] = float(o.mean()) if len(e) else None
                r[f'gain_{n}'] = (1 - e.mean() / c.mean()) if len(e) else None
                v = [sp(z[(law == L) & mm, j], t[(law == L) & mm, j]) for L in range(3)]; vv = [x for x in v if x is not None]
                r[f'rs_{n}'] = float(np.mean(vv)) if vv else None
                if detail:
                    r[f'rs_class_{n}'] = {LAWS[L]: v[L] for L in range(3)}
                    pa = {}
                    for a in sorted(set(self.act[ii])):
                        k = self.act[ii] == a; w = [sp(z[k & (law == L) & mm, j], t[k & (law == L) & mm, j]) for L in range(3)]; w = [x for x in w if x is not None]
                        pa[a] = float(np.mean(w)) if w else None
                    r[f'rs_action_{n}'] = pa
            if detail:
                r['acc_action'] = {a: float((self.pl[s][ii][self.act[ii] == a] == law[self.act[ii] == a]).mean()) for a in sorted(set(self.act[ii]))}
                r['acc_class'] = {LAWS[L]: float((self.pl[s][ii][law == L] == L).mean()) for L in range(3) if (law == L).any()}
            per.append(r)
        def avg(xs):
            if isinstance(xs[0], dict):
                return {k: avg([x[k] for x in xs]) for k in xs[0]}
            v = [x for x in xs if x is not None]; return float(np.mean(v)) if v else None
        return avg(per)


def ens(members):
    out = []
    for rows in zip(*members):
        r = dict(rows[0]); r['prediction_z'] = np.mean([x['prediction_z'] for x in rows], 0).tolist()
        r['predicted_law_id'] = int(np.argmax(np.bincount([x['predicted_law_id'] for x in rows], minlength=3))); out.append(r)
    return out


def boot(obj, fn):
    objs = sorted(set(obj)); g = [np.where(obj == o)[0] for o in objs]; rng = np.random.default_rng(17); out = []
    for _ in range(DRAWS):
        out.append(fn(np.concatenate([g[i] for i in rng.integers(len(g), size=len(g))])))
    return out


def ci(vals):
    v = [x for x in vals if x is not None]; return np.quantile(v, [.025, .975]).tolist() if v else None


def main():
    D = torch.load(ROOT / 'data/prepared/v5m/data.pt', weights_only=False); st = D['stats']
    tr = [i for i, m in enumerate(D['meta']) if m['split'] == 'train']
    y = ((D['y'][tr] - st['y_mean']) / st['y_std']).numpy(); ym = D['ym'][tr].numpy(); law = D['law'][tr].numpy()
    lawmean = {L: np.array([y[(law == L) & ym[:, j], j].mean() if ((law == L) & ym[:, j]).any() else 0. for j in range(5)]) for L in range(3)}
    rep = {'written_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'doc': __doc__, 'stats': {k: (v.tolist() if torch.is_tensor(v) else v) for k, v in st.items()}, 'splits': {}}
    for split in ('val', 'test', 'fval'):
        R = {}; sets = {}
        for arm in ARMS:
            mem = [load(ROOT / f'runs/v5m_{arm}_seed{s}/pred_{split}.jsonl') for s in SEEDS if (ROOT / f'runs/v5m_{arm}_seed{s}/pred_{split}.jsonl').exists()]
            if not mem:
                continue
            for name, members in ((f'{arm} (seeds {len(mem)})', mem), (f'{arm} ensemble', [ens(mem)] if len(mem) > 1 else None)):
                if not members:
                    continue
                S = Set(members, st, lawmean); sets[name] = S
                pt = S.metrics(np.arange(len(S.t)), detail=True)
                bs = boot(S.obj, lambda ii: S.metrics(ii))
                R[name] = {'arm': arm, 'members': len(members) if 'ensemble' not in name else len(mem), 'point': pt,
                           'ci95': {k: ci([b[k] for b in bs]) for k in bs[0]}}
        base = {k: v for k, v in sets.items() if k.startswith('vision ') or k.startswith('vision (')}
        paired = {}
        for name, S in sets.items():
            if name.startswith('vision ') or name.startswith('vision ('):
                continue
            kind = 'ensemble' if name.endswith('ensemble') else 'seeds'
            bname = next((b for b in base if (b.endswith('ensemble')) == (kind == 'ensemble')), None)
            if bname is None:
                continue
            B = base[bname]
            KS = [f'dec_{n}' for n in TG] + [f'gain_{n}' for n in TG] + [f'rs_{n}' for n in TG] + ['law_accuracy']
            def diff(ii, S=S, B=B):
                x, y_ = S.metrics(ii), B.metrics(ii); return {k: (x[k] - y_[k]) if x[k] is not None and y_[k] is not None else None for k in KS}
            pt = diff(np.arange(len(S.t))); bs = boot(S.obj, diff)
            paired[f'{name} - {bname}'] = {k: {'value': pt[k], 'ci95': ci([b[k] for b in bs])} for k in KS}
        rep['splits'][split] = {'arms': R, 'paired_vs_vision': paired}
    (ROOT / 'docs/v5m/RESULTS.json').write_text(json.dumps(rep, indent=1) + '\n')
    f = lambda v, c=None: '–' if v is None else (f'{v:.3f}' + (f' [{c[0]:.3f}, {c[1]:.3f}]' if c else ''))
    L = ['# V5m 多模态先验（视觉 + 触觉力 + 动作），V5b train 0–1999', '', f'生成 {rep["written_utc"]}；源 `docs/v5m/RESULTS.json`。误差为 log10 decade；增益 = 1 − 误差/常数误差；方括号为物体 bootstrap 95% CI。', '']
    for split, r in rep['splits'].items():
        title = {'val': 'V5b val（283）', 'test': 'V5b test（300，一次评估）', 'fval': 'V5f 力控 press val（48）'}[split]
        L += [f'## {title}', '', '| 方法 | E/ρ dec（增益） | σ/ρ dec（增益） | ρ dec（增益） | E dec（增益） | 类别准确率 | r_S(E/ρ) | r_S(σ/ρ) | r_S(ρ) | r_S(E) |', '|---|---|---|---|---|---|---|---|---|---|']
        first = next(iter(r['arms'].values()), None)
        if first:
            p = first['point']
            L.append('| 常数 | ' + ' | '.join(f"{f(p[f'const_{n}'])}" for n in TG) + ' | – | – | – | – | – |')
            L.append('| GT 类均值 oracle | ' + ' | '.join(f"{f(p[f'oracle_{n}'])} ({f(1 - p[f'oracle_{n}'] / p[f'const_{n}']) if p[f'oracle_{n}'] is not None else '–'})" for n in TG) + ' | 1 | – | – | – | – |')
        for name, x in r['arms'].items():
            p, c = x['point'], x['ci95']
            L.append(f"| {LAB[x['arm']]} {name.split(' ', 1)[1]} | " + ' | '.join(f"{f(p[f'dec_{n}'])} ({f(p[f'gain_{n}'], c[f'gain_{n}'])})" for n in TG) +
                     f" | {f(p['law_accuracy'], c['law_accuracy'])} | " + ' | '.join(f(p[f'rs_{n}'], c[f'rs_{n}']) for n in TG) + ' |')
        L += ['', '配对差（相对视觉单模态，同类型：3 种子平均 / 3 种子集成）：', '', '| 对比 | ΔE/ρ dec | Δσ/ρ dec | Δρ dec | ΔE dec | Δ类别准确率 | Δr_S(ρ) | Δr_S(E) |', '|---|---|---|---|---|---|---|---|']
        for k, d in r['paired_vs_vision'].items():
            g = lambda m: f(d[m]['value'], d[m]['ci95'])
            L.append(f"| {k} | {g('dec_E_over_rho')} | {g('dec_sigma_y_over_rho')} | {g('dec_rho')} | {g('dec_E')} | {g('law_accuracy')} | {g('rs_rho')} | {g('rs_E')} |")
        L.append('')
        for name, x in r['arms'].items():
            if name.endswith('ensemble'):
                p = x['point']
                L.append(f"- **{LAB[x['arm']]} 集成** 准确率/动作 {json.dumps({k: round(v, 2) for k, v in p['acc_action'].items()})}；r_S(E/ρ)/动作 {json.dumps({k: (round(v, 2) if v is not None else None) for k, v in p['rs_action_E_over_rho'].items()})}；"
                         f"r_S(ρ)/动作 {json.dumps({k: (round(v, 2) if v is not None else None) for k, v in p['rs_action_rho'].items()})}；r_S(E)/动作 {json.dumps({k: (round(v, 2) if v is not None else None) for k, v in p['rs_action_E'].items()})}")
        L.append('')
    (ROOT / 'docs/v5m/RESULTS_ZH.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
