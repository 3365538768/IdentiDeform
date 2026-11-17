"""docs/v5q/RESULTS.json -> docs/v5q/RESULTS_ZH.md (Chinese summary tables)."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
R = json.loads((ROOT / 'docs/v5q/RESULTS.json').read_text())
NAME = {'prior_top1': '先验 top-1（无探测）', 'prior_top2': '先验 top-2（2 次律探测）', 'lawmean_gt': '律均值-GT（特权）', 'lawmean_best3': '律均值 best-of-3'}
f3 = lambda v: '–' if v is None else f'{v:.3f}'
L = ['# V5q：V5b 数据上的先验引导少量仿真修正（C1 原实验复跑）', '',
     f"生成时间 {R['generated_utc']}；协议 `docs/v5q/PROTOCOL.json`（预注册）；执行记录 `docs/v5q/DEVIATIONS.md`。", '',
     '- 仿真：与 V5b 观测完全相同的配置/重力/环境（scripts/v3n/gen_v3.py --g-all -2 --weak hold --ranges v5 同一路径），仅材料参数不同；每个候选都核对非材料配置与 contact_bc 边界条件与观测一致。',
     '- 目标/优化器/预算/失败规则与 V2 C1 相同（(1−IoU)+RGB MAE/255，罗盘搜索，k=0..24 嵌套，失败候选 loss 9 计入预算）；BOUNDS 与律均值改为 V5b。',
     f"- 误差：可辨识 z-MAE（E/ρ、σ_y/ρ），主指标用 V5b z 单位（{R['z_stats']}）；另给 V2 z 单位与 log10 十进位，用于和 V2 数字比较。CI = 物体聚类 bootstrap（2000 次，seed 17）。", '']
for split, title in (('val', '验证集'), ('test', '测试集')):
    r = R.get(split, {})
    if not r.get('curves'):
        L += [f'## {title}', '', '（尚无完成的样本）', '']; continue
    L += [f"## {title}（{r['n_analysed']}/{r['planned']} 样本，{r['n_objects']} 个物体{'，部分' if r['partial'] else ''}）", '',
          '| 臂 | k=0 | k=4 | k=12 | k=24 | k=24 95% CI | k=24（V2 z 单位） | k=24（十进位） |', '|---|---|---|---|---|---|---|---|']
    for a, c in r['curves'].items():
        g = lambda k: c.get(str(k), {}).get('z_mae_v3')
        top = c.get('24') or c.get(str(max(int(k) for k in c)))
        L.append(f"| {NAME[a]} | {f3(g(0))} | {f3(g(4))} | {f3(g(12))} | {f3(g(24))} | [{f3(top['ci95'][0])}, {f3(top['ci95'][1])}] | {f3(top['z_mae_v2units'])} | {f3(top['log10_decades_mae'])} |")
    L.append('')
    ss = r.get('sims_saved_vs_lawmean_gt', {})
    for a, v in ss.items():
        for k, x in v.items():
            m = x['sims_to_match_log_interp']; ci = x['log_interp_bootstrap']['ci95']
            L.append(f"- {NAME[a]} k={k}（误差 {f3(x['prior_mae'])}）：律均值-GT 需 {('>24（截尾）' if m is None else f'{m:.1f}')} 次仿真才能追平"
                     f"（对数插值，bootstrap 95% [{f3(ci[0])}, {f3(ci[1])}]，截尾比例 {x['log_interp_bootstrap']['fraction_censored']:.2f}；线性插值 {f3(x['sims_to_match_linear'])}）")
    fit = r.get('post_vs_initial_error', {})
    for k, v in fit.items():
        p = v['pooled_per_sample']
        L.append(f"- 修正后 vs 初始误差（k={k}，逐样本，无探测臂合并）：斜率 {p['slope']:.2f}，r={p['r']:.2f}（n={p['n']}）")
    fl = r.get('simulation_failures', {})
    L.append('- 失败仿真：' + '；'.join(f"{NAME[a]} {v['failed']}/{v['total']}" for a, v in fl.items()))
    nf = r.get('noise_floor_gt_resim_observed', {})
    if nf.get('n'):
        L.append(f"- GT 参数重仿真噪声底（{nf['n']} 个，每动作 1 个）：IoU {f3(nf['iou'])}，RGB MAE {f3(nf['rgb_mae_union'])}，loss {nf['loss']:.2e}")
    L.append('')
L += ['## 与 V2 对比（V2 为 V2 z 单位；V5b 同时给 V5b z / V2 z / 十进位）', '', '| 项 | k | V2 | V5b（V5b z） | V5b（V2 z） | V5b（十进位） |', '|---|---|---|---|---|---|']
for key, v in R['v2_comparison'].items():
    for k, x in v.items():
        L.append(f"| {key} | {k} | {f3(x['v2_z'])} | {f3(x['v3_z'])} | {f3(x['v3_in_v2_z_units'])} | {f3(x['v3_log10_decades'])} |")
v2f = R.get('v2_per_sample_post_vs_initial_fit_val60', {})
if '12' in v2f:
    L += ['', f"V2 val60 同口径逐样本拟合（k=12）：斜率 {v2f['12']['pooled_per_sample']['slope']:.2f}，r={v2f['12']['pooled_per_sample']['r']:.2f}。"]
L += ['', '图数据：`docs/paper_figs/data/f1_bands_v5b.json`（键 val60/test30 → n_samples, n_objects, arms → <臂> → <k> → mean, ci95；另 sims_saved_prior_top1_k0_vs_lawmean_gt）。', '']
(ROOT / 'docs/v5q/RESULTS_ZH.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
