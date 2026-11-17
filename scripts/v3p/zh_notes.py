import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
n = json.loads((ROOT / 'docs/v3p/notes.json').read_text()); raw = json.loads((ROOT / 'docs/v3p/RESULTS_raw.json').read_text())
v = lambda ds, sp, m, k: raw[ds]['splits'][sp][m][k]['value']
c = lambda ds, sp, m, k: raw[ds]['splits'][sp][m][k]['ci95']
f = lambda ds, sp, m, k: '%.3f [%.3f, %.3f]' % (v(ds, sp, m, k), *c(ds, sp, m, k))
E = 'ensemble_BCD9'
rows = []
for sp in ('val', 'test'):
    for ds in ('V2', 'V3'):
        rows.append(f"| {sp} | {ds} | {f(ds,sp,E,'z_mae')} | {f(ds,sp,E,'ident_z_mae')} | {f(ds,sp,E,'law_accuracy')} | {f(ds,sp,E,'within_law_spearman_E_over_rho')} | {f(ds,sp,E,'within_law_spearman_sigma_y_over_rho')} | {f(ds,sp,E,'decade_mae_E_over_rho')} | {v(ds,sp,'constant_train_mean','decade_mae_E_over_rho'):.3f} | {f(ds,sp,E,'decade_mae_sigma_y_over_rho')} | {v(ds,sp,'constant_train_mean','decade_mae_sigma_y_over_rho'):.3f} |")
g = n['headline']
sec = [
'## 结论（2000 条训练样本，V2 原管线原超参）',
'- **V3 上前馈先验明显变弱。** 9 头集成 test：z-MAE %.3f（V2 %.3f），类别准确率 %.2f（V2 %.2f），族内 Spearman r(E/ρ) %.2f（V2 %.2f），r(σ/ρ) %.2f（V2 %.2f）。单个 C 头 test z-MAE %.3f（V2 %.3f）。' % (
    v('V3','test',E,'z_mae'), v('V2','test',E,'z_mae'), v('V3','test',E,'law_accuracy'), v('V2','test',E,'law_accuracy'),
    v('V3','test',E,'within_law_spearman_E_over_rho'), v('V2','test',E,'within_law_spearman_E_over_rho'),
    v('V3','test',E,'within_law_spearman_sigma_y_over_rho'), v('V2','test',E,'within_law_spearman_sigma_y_over_rho'),
    v('V3','test','C (3 seeds)','z_mae'), v('V2','test','C (3 seeds)','z_mae')),
'- **按 decade 比较（跨数据集公平口径）：** V3 test E/ρ 误差 %.3f dec（V2 %.3f），σ/ρ %.3f dec（V2 %.3f）。绝对误差看似相近甚至更小，是因为 V3 的范围窄：常数基线本身就只有 %.3f / %.3f dec（V2 %.3f / %.3f）。相对常数基线，V3 先验只把 E/ρ 误差降低 %.0f%%、σ/ρ 降低 %.0f%%（V2 为 %.0f%% / %.0f%%）。' % (
    v('V3','test',E,'decade_mae_E_over_rho'), v('V2','test',E,'decade_mae_E_over_rho'), v('V3','test',E,'decade_mae_sigma_y_over_rho'), v('V2','test',E,'decade_mae_sigma_y_over_rho'),
    v('V3','test','constant_train_mean','decade_mae_E_over_rho'), v('V3','test','constant_train_mean','decade_mae_sigma_y_over_rho'),
    v('V2','test','constant_train_mean','decade_mae_E_over_rho'), v('V2','test','constant_train_mean','decade_mae_sigma_y_over_rho'),
    100*g['V3/test']['ensemble_gain_over_constant_rel']['decade_mae_E_over_rho'], 100*g['V3/test']['ensemble_gain_over_constant_rel']['decade_mae_sigma_y_over_rho'],
    100*g['V2/test']['ensemble_gain_over_constant_rel']['decade_mae_E_over_rho'], 100*g['V2/test']['ensemble_gain_over_constant_rel']['decade_mae_sigma_y_over_rho']),
'- 仍明显优于外观基线：首帧 DINOv3 ≈ 常数（V3 test z-MAE %.3f vs 常数 %.3f），视频先验也优于特权的 GT 类均值（%.3f）。B/C/D 之间仍无可分辨差异（与 V2 一致，场监督/反馈无收益）。' % (
    v('V3','test','dinov3_first_frame_B (3 seeds)','z_mae'), v('V3','test','constant_train_mean','z_mae'), v('V3','test','class_mean_gt_privileged','z_mae')),
'- 每种动作都退化（docs/v3p/PER_ACTION_VAL.json，val 集成）：族准确率从 0.60–0.84 降到 0.48–0.60，plasticine 最难（val 仅 26/94 判对）；shear 下 r(σ/ρ) 几乎为 0。drop 不经过夹板也退化，说明不只是接触方式的问题：可能与重力减小（drop 重力 2）、族间参数范围重叠（GT 类均值仅比常数好 %.3f z）、以及物体在画面中更小有关（抽查 val 前 24 个样本，物体像素占比约为 V2 的 1/3）。这些是推测，未做受控验证。' % (
    v('V3','val','constant_train_mean','z_mae') - v('V3','val','class_mean_gt_privileged','z_mae')),
'',
'## V2 vs V3 主表（9 头集成；方括号为物体 bootstrap 95% CI）',
'',
'| split | 数据 | z-MAE(4) | 可辨识 z-MAE | 类别准确率 | 族内 r_S(E/ρ) | 族内 r_S(σ/ρ) | E/ρ 误差 dec | 常数 E/ρ dec | σ/ρ 误差 dec | 常数 σ/ρ dec |',
'|---|---|---|---|---|---|---|---|---|---|---|', *rows,
'',
'## 与 V2 的差异 / 偏离',
*['- ' + x for x in n['differences_vs_v2']],
'- 协议：' + n['protocol_identical_to_v2'],
*['- 偏离：' + x for x in n['deviations']],
'- 交接：' + n['hand_off'],
]
n['zh_sections'] = sec
(ROOT / 'docs/v3p/notes.json').write_text(json.dumps(n, indent=1, ensure_ascii=False)); print('\n'.join(sec))
