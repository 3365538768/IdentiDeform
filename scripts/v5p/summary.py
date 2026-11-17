"""docs/v5/SUMMARY_ZH.md: compact V3/V4/V5a/V5b quick-check + V5b gate summary (from docs/v5/PRIOR_CHECK.json, docs/v5/GATE.json)."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
pc = json.loads((ROOT / 'docs/v5/PRIOR_CHECK.json').read_text()); gt = json.loads((ROOT / 'docs/v5/GATE.json').read_text())
f = lambda x, c=None: '–' if x is None else (f'{x:.3f}' + (f' [{c[0]:.3f}, {c[1]:.3f}]' if c else ''))
D = {'V3_t1000': 'V3（压板/夹爪接触，V3 范围）', 'V4_t1000': 'V4（V3 + 弱动作 g=2）', 'V5a_t1000': 'V5a（弱动作改托举式，V3 范围）', 'V5b_t1000': 'V5b（托举式 + 类别可分新范围）'}
L = ['# V3→V5 数据改造：先验可用性汇总（补充材料用）', '',
     '协议：V2 原前馈先验（冻结 V-JEPA 2 ViT-L + 2 层 transformer 头，B/C/D×3 种子，best-val），各数据集自己的 train 标准化；误差为 log10 decade；方括号为物体 bootstrap 95% CI（2000 次，seed 17）。'
     '“GT 类均值 oracle”为特权基线：已知真实类别，预测该类 train 均值。', '',
     '## 1. 快检（train 0–999，val 283，按索引配对；9 头集成）', '',
     '| 数据 | E/ρ dec | 常数 | oracle | 增益 E/ρ（相对常数） | oracle 增益 | σ/ρ dec | 常数 | oracle | 增益 σ/ρ | oracle 增益 | 类别准确率 | 族内 r_S(E/ρ) | 族内 r_S(σ/ρ) |',
     '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
for arm, gs in pc['arms'].items():
    e = gs.get('ensemble_9'); o = gs.get('class_mean_gt_privileged', {}).get('point_only')
    if not e or not o:
        continue
    p, c = e['point'], e['ci95']
    L.append(f"| {D.get(arm, arm)} | {f(p['decade_mae_E_over_rho'])} | {f(p['const_decade_mae_E_over_rho'])} | {f(o['decade_mae_E_over_rho'])} | {f(p['gain_over_const_E_over_rho'], c.get('gain_over_const_E_over_rho'))} | {f(o['gain_over_const_E_over_rho'])} | "
             f"{f(p['decade_mae_sigma_y_over_rho'])} | {f(p['const_decade_mae_sigma_y_over_rho'])} | {f(o['decade_mae_sigma_y_over_rho'])} | {f(p['gain_over_const_sigma_y_over_rho'], c.get('gain_over_const_sigma_y_over_rho'))} | {f(o['gain_over_const_sigma_y_over_rho'])} | "
             f"{f(p['law_accuracy'], c.get('law_accuracy'))} | {f(p['within_class_spearman_E_over_rho'], c.get('within_class_spearman_E_over_rho'))} | {f(p['within_class_spearman_sigma_y_over_rho'], c.get('within_class_spearman_sigma_y_over_rho'))} |")
L += ['', '配对差（相对 V3_t1000，9 头集成）：', '', '| 对比 | Δ增益 E/ρ | Δ增益 σ/ρ | Δ类别准确率 | Δ族内 r_S(E/ρ) | Δ族内 r_S(σ/ρ) |', '|---|---|---|---|---|---|']
for k, d in pc['paired_v4_minus_v3'].items():
    if not k.endswith('ensemble_9'):
        continue
    g = lambda m: f(d[m]['value'], d[m]['ci95']) if m in d else '–'
    L.append(f"| {k.split(' / ')[0]} | {g('gain_over_const_E_over_rho')} | {g('gain_over_const_sigma_y_over_rho')} | {g('law_accuracy')} | {g('within_class_spearman_E_over_rho')} | {g('within_class_spearman_sigma_y_over_rho')} |")
L += ['', '## 2. V5b 门槛（train 0–1999 全量先验，9 头集成；V2 主线作参照）', '']
for sp, r in gt['splits'].items():
    L += [f'**{sp}**', '', '| 数据 | 方法 | E/ρ dec | σ/ρ dec | 族内 r_S(E/ρ) | 族内 r_S(σ/ρ) | 类别准确率 |', '|---|---|---|---|---|---|---|']
    for ds in ('V5b_2000', 'V2_main'):
        for m, lab in (('prior_ensemble', '先验'), ('class_mean_oracle', 'GT 类均值 oracle'), ('constant', '常数')):
            x = r[ds][m]; h = lambda k: f(x[k]['value'], x[k]['ci95']) if x[k]['value'] is not None else '–'
            L.append(f"| {ds} | {lab} | {h('decade_mae_E_over_rho')} | {h('decade_mae_sigma_y_over_rho')} | {h('within_law_spearman_E_over_rho')} | {h('within_law_spearman_sigma_y_over_rho')} | {h('law_accuracy')} |")
    L += ['', '| 数据 | 先验−oracle E/ρ (dec) | 先验−oracle σ/ρ (dec) | 先验增益 E/ρ / oracle 增益 | 先验增益 σ/ρ / oracle 增益 |', '|---|---|---|---|---|']
    for ds in ('V5b_2000', 'V2_main'):
        x = r[ds]
        L.append(f"| {ds} | {x['prior_minus_oracle_decades_E_over_rho']:+.3f} | {x['prior_minus_oracle_decades_sigma_y_over_rho']:+.3f} | {x['gain_over_constant_E_over_rho']:.3f} / {x['oracle_gain_over_constant_E_over_rho']:.3f} | {x['gain_over_constant_sigma_y_over_rho']:.3f} / {x['oracle_gain_over_constant_sigma_y_over_rho']:.3f} |")
    L.append('')
L += ['## 3. 结论与注意事项',
      '- V3（外部压板/夹爪接触、窄范围）使视频先验大幅变弱：2000 样本 9 头集成相对常数只降 17%/13% 误差（V2 52%/50%），见 docs/v3p/RESULTS_ZH.md。',
      '- 只改动作（V4 加重力、V5a 托举式）在 1000 样本快检中只带来小幅、不显著的 E/ρ 提升，σ/ρ 不变或变差。',
      '- V5b 的大幅“增益”主要来自类别可分的新范围：其 GT 类均值 oracle 本身就把常数误差降低约 0.38（E/ρ）/ 0.23（σ/ρ）。全量 V5b 先验在 E/ρ 上与 oracle 持平（val 差 +0.035 dec，test −0.021 dec，CI 重叠），σ/ρ 略优于 oracle；族内 r_S ≈ 0.5（V2 0.65–0.88）。',
      '- 相比之下 V2 先验远超其 oracle（E/ρ −0.22～−0.25 dec，σ/ρ −0.52～−0.65 dec），因为 V2 的类别范围不可分、oracle 几乎无用——所以跨数据集比较必须相对各自的常数和 oracle，而不是看绝对 decade 误差或 z-MAE。',
      '- 决定：论文主线保持 V2；V5b（快检 + 门槛 + V5b 修正）作为补充分析。V5b 先验交接见 docs/v5p/HANDOFF.json（main_run v5b2k_C_seed29 + 9 头集成）。',
      '', '源文件：docs/v5/PRIOR_CHECK.json、docs/v5/GATE.json、docs/v4/PRIOR_CHECK.json、docs/v3p/RESULTS.json。']
(ROOT / 'docs/v5/SUMMARY_ZH.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))
