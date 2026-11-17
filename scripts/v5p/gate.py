"""V5b gate report (paper-line decision): full V5b prior (train 0-1999, B/C/D x3, 9-run ensemble) on V5b val vs the
privileged GT-class-mean oracle and the constant (train-mean) baseline, in log10 decades; within-class Spearman;
class accuracy; V2 main val as reference (prior vs V2's own class-mean oracle). Scoring = scripts/v3p/report.py
(object bootstrap 2000, seed 17). Writes docs/v5/GATE.json and docs/v5/GATE_ZH.md.
  .venv/bin/python scripts/v5p/gate.py [--prefix v5b2k --stats data/manifests/v5bmain_t2000.specific.stats.json]
"""
import argparse, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v3p')); sys.path.insert(0, str(ROOT / 'scripts'))
from report import dataset_block  # noqa: E402

K = ['decade_mae_E_over_rho', 'decade_mae_sigma_y_over_rho', 'within_law_spearman_E_over_rho', 'within_law_spearman_sigma_y_over_rho',
     'within_law_pearson_E_over_rho', 'within_law_pearson_sigma_y_over_rho', 'law_accuracy', 'z_mae', 'ident_z_mae']


def summarize(blk, ens):
    pick = lambda m: {k: {'value': blk[m][k]['value'], 'ci95': blk[m][k]['ci95']} for k in K}
    e, o, c = pick(ens), pick('class_mean_gt_privileged'), pick('constant_train_mean')
    out = {'prior_ensemble': e, 'class_mean_oracle': o, 'constant': c, 'groups': {g: pick(g) for g in blk if g.endswith('seeds)')}}
    for n in ('E_over_rho', 'sigma_y_over_rho'):
        k = f'decade_mae_{n}'
        out[f'gain_over_constant_{n}'] = 1 - e[k]['value'] / c[k]['value']
        out[f'oracle_gain_over_constant_{n}'] = 1 - o[k]['value'] / c[k]['value']
        out[f'prior_minus_oracle_decades_{n}'] = e[k]['value'] - o[k]['value']
    return out


def main():
    a = argparse.ArgumentParser(); a.add_argument('--prefix', default='v5b2k'); a.add_argument('--stats', default='data/manifests/v5bmain_t2000.specific.stats.json')
    a.add_argument('--splits', default='val'); args = a.parse_args()
    rep = {'written_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'doc': __doc__, 'splits': {}}
    for sp in args.splits.split(','):
        v5 = dataset_block(args.prefix, args.stats, [sp], {sp: {'constant_train_mean': f'runs/{args.prefix}_const_{sp}', 'class_mean_gt_privileged': f'runs/{args.prefix}_lawmean_gt_{sp}'}})
        v2 = dataset_block('v2main', 'data/manifests/v2main.specific.stats.json', [sp],
                           {sp: {'constant_train_mean': 'runs/v3c_const' + ('' if sp == 'val' else '_test'), 'class_mean_gt_privileged': 'runs/v3c_lawmean_gt' + ('' if sp == 'val' else '_test')}})
        rep['splits'][sp] = {'V5b_2000': summarize(v5['splits'][sp], 'ensemble_BCD9'), 'V2_main': summarize(v2['splits'][sp], 'ensemble_BCD9'),
                             'train_std_decades': {'V5b': v5['train_std_decades'], 'V2': v2['train_std_decades']}}
    (ROOT / 'docs/v5/GATE.json').write_text(json.dumps(rep, indent=1) + '\n')
    f = lambda x: '–' if x is None or x.get('value') is None else (f"{x['value']:.3f}" + (f" [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]" if x.get('ci95') else ''))
    L = ['# V5b 门槛报告（2000 样本先验，9 头集成）', '', f'生成 {rep["written_utc"]}；源 `docs/v5/GATE.json`；方括号为物体 bootstrap 95% CI。GT 类均值为特权 oracle（已知真实类别，预测该类 train 均值）。', '']
    for sp, r in rep['splits'].items():
        L += [f'## {sp}', '', '| 数据 | 方法 | E/ρ 误差 dec | σ/ρ 误差 dec | 族内 r_S(E/ρ) | 族内 r_S(σ/ρ) | 类别准确率 |', '|---|---|---|---|---|---|---|']
        for ds in ('V5b_2000', 'V2_main'):
            for m, lab in (('prior_ensemble', '先验（9 头集成）'), ('class_mean_oracle', 'GT 类均值 oracle'), ('constant', '常数')):
                x = r[ds][m]
                L.append(f"| {ds} | {lab} | {f(x['decade_mae_E_over_rho'])} | {f(x['decade_mae_sigma_y_over_rho'])} | {f(x['within_law_spearman_E_over_rho'])} | {f(x['within_law_spearman_sigma_y_over_rho'])} | {f(x['law_accuracy'])} |")
        L += ['', '| 数据 | 先验增益 E/ρ（相对常数） | oracle 增益 E/ρ | 先验−oracle E/ρ (dec) | 先验增益 σ/ρ | oracle 增益 σ/ρ | 先验−oracle σ/ρ (dec) |', '|---|---|---|---|---|---|---|']
        for ds in ('V5b_2000', 'V2_main'):
            x = r[ds]
            L.append(f"| {ds} | {x['gain_over_constant_E_over_rho']:.3f} | {x['oracle_gain_over_constant_E_over_rho']:.3f} | {x['prior_minus_oracle_decades_E_over_rho']:+.3f} | "
                     f"{x['gain_over_constant_sigma_y_over_rho']:.3f} | {x['oracle_gain_over_constant_sigma_y_over_rho']:.3f} | {x['prior_minus_oracle_decades_sigma_y_over_rho']:+.3f} |")
        L.append('')
    (ROOT / 'docs/v5/GATE_ZH.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
