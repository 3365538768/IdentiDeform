"""V3e: write docs/v3e/GRIPPER_TASK_ZH.md (Chinese summary) from docs/v3e/GRIPPER_TASK*.json."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
M = ['gt', 'refined_top2', 'ensemble_prior', 'refined_best3', 'lawmean_pred', 'fixed_loo']
ZH = {'gt': 'GT 参数（oracle 孪生体）', 'refined_top2': 'prior_top2 + 12 次仿真细化（本文方法）', 'ensemble_prior': '集成先验，无细化（0 次仿真）',
      'refined_best3': 'lawmean_best3 + 12 次仿真细化（无先验）', 'lawmean_pred': '预测材料律的训练集均值参数', 'fixed_loo': '固定深度启发式（留一调参）'}


def pct(x):
    return f'{100 * x:.1f}%'


def table(d, groups=('all', 'plastic_laws', 'plasticine', 'metal', 'jelly')):
    s = d['summary']
    head = '| 方法 | ' + ' | '.join({'all': '全部', 'plastic_laws': '弹塑性(金属+橡皮泥)', 'plasticine': '橡皮泥', 'metal': '金属', 'jelly': '果冻'}[g] for g in groups) + \
           ' | 实现的安全夹深 (oracle) [CI] | 平均遗憾 \\|d−d*\\| | 损坏率 | 漏抓率 | 仿真次数/物体 |\n'
    head += '|' + '---|' * (len(groups) + 6) + '\n'
    for m in M:
        e = s[m]
        if 'all' not in e:
            continue
        cells = []
        for g in groups:
            if g in e:
                lo, hi = e[g]['success_ci95_object_bootstrap']
                cells.append(f"{pct(e[g]['success'])} [{pct(lo)}, {pct(hi)}] (n={e[g]['n']})")
            else:
                cells.append('—')
        a = e['all']
        u = a['utility_ci95_object_bootstrap']
        head += f"| {ZH[m]} | " + ' | '.join(cells) + f" | {a['utility_mean']:.3f} ({a['oracle_utility_mean']:.3f}) [{u[0]:.3f}, {u[1]:.3f}] | {a['regret_mean']:.3f} | {pct(a['damage_rate'])} | {pct(a['missed_grasp_rate'])} | {e['sims_per_object']:.1f} |\n"
    return head


def paired(d):
    out = '| 对比 | 成功率差 [95% CI] | 安全夹深差 [95% CI] | 遗憾差 [95% CI] | n |\n|---|---|---|---|---|\n'
    for k in ['refined_top2-minus-ensemble_prior', 'refined_top2-minus-lawmean_pred', 'refined_top2-minus-fixed_loo', 'refined_top2-minus-refined_best3',
              'ensemble_prior-minus-lawmean_pred', 'ensemble_prior-minus-fixed_loo']:
        if k in d['paired_success_differences']:
            p = d['paired_success_differences'][k]
            out += f"| {k} | {100 * p['success_diff']:+.1f} pp [{100 * p['success_ci95_object_bootstrap'][0]:+.1f}, {100 * p['success_ci95_object_bootstrap'][1]:+.1f}] | " \
                   f"{p['utility_diff']:+.3f} [{p['utility_ci95_object_bootstrap'][0]:+.3f}, {p['utility_ci95_object_bootstrap'][1]:+.3f}] | " \
                   f"{p['regret_diff']:+.3f} [{p['regret_ci95_object_bootstrap'][0]:+.3f}, {p['regret_ci95_object_bootstrap'][1]:+.3f}] | {p['n']} |\n"
    return out


def main():
    main_ = json.loads((ROOT / 'docs/v3e/GRIPPER_TASK.json').read_text())
    w1 = json.loads((ROOT / 'docs/v3e/GRIPPER_TASK_wave1.json').read_text())
    sens = {p.stem.split('_', 2)[-1]: json.loads(p.read_text()) for p in sorted((ROOT / 'docs/v3e').glob('GRIPPER_TASK_sens_*.json'))}
    det = (ROOT / 'docs/v3e/DETERMINISM.json')
    pr = main_['protocol']
    L = []
    L.append('# V3e 下游机器人任务：损伤感知的平行夹爪挤压（数字孪生用于规划）\n')
    L.append('自动生成：`scripts/v3e/write_zh.py`（数据：`docs/v3e/GRIPPER_TASK.json`）。\n')
    L.append('## 任务与协议\n')
    L.append(f"- 动作：两块刚性（粘着）平板沿物体较窄的水平轴对夹，每块向内移动 d·W/2（1.0 s 内），保持到 1.5 s，然后张开（移除平板），自由回弹到 4.0 s。MPM 设置与 V2-main/C1 相同（低阻尼、substep 1e-4、同一物体 3DGS 点云与填充）。\n"
             f"- **重力关闭（g=0）**：试点发现，在 V2-main 的重力设置下，多数弹塑性物体在 <0.5 s 内自行塌成一摊（见 `docs/v3e/figs/pilot_particles_gravity.png`），夹取无从谈起，因此任务在无重力下进行（参数仍是从有重力视频中估计的）。\n"
             f"- 损伤指标（主）：永久压溃 = 释放后（3.5–4.0 s 平均位置）先与同参数 d=0 参考仿真做刚体（Procrustes）对齐，再取沿夹持轴的宽度（粒子 1–99% 分位跨度）减少量 / W。容差 τ={pct(pr['tau'])}。"
             "原计划的未对齐宽度损失（wloss）被释放后的刚体转动污染（薄物体转几度即表现为 10–60% “压溃”，见 `scripts/v3e/diag_rotation.py`），该问题是在中途（约 20 个样本完成、已看过 wloss 下的部分方法对比后）通过检查 GT 仿真的刚体转角发现的；主指标随后改为对齐版本，wloss 及其结果作为敏感性完整报告，请读者据此判断。\n"
             f"- 候选深度 d ∈ {pr['candidate_depths']}，最小可靠抓取深度 d_min={pr['d_min']}。\n"
             '- 规划：在各方法的孪生体（同一观测物体几何 + 该方法的材料律/参数）中仿真全部候选深度（+1 次 d=0 参考），选择预测压溃 ≤ τ 的最大 d；若无可行深度则**放弃抓取**。孪生体仿真失败（MPM 发散）视为不可行。\n'
             '- 执行：用 GT 参数在所选 d 下仿真（GT 在每个候选深度的仿真只做一次，同时也是 oracle 孪生体）。\n'
             '- 成功：所选 d 在 GT 下压溃 ≤ τ；或 GT 下没有任何可行深度且方法选择放弃（正确拒绝）。遗憾 = |d_选 − d_oracle|（放弃记为 0）。\n'
             '- 固定启发式：所有物体用同一深度（或一律放弃），在其余被评估样本的 GT 结果上留一选优（与被测样本不相交）。\n'
             '- 置信区间：按物体身份（10 个 3DGS 物体）做簇自助法，2000 次，百分位 95% CI。\n'
             '- 硬化/软化参数不被预测：孪生体固定 0.25，GT 用真值。\n')
    L.append(f"\n## 主结果：全部 C1 验证样本（n={main_['n_samples']} 可评估，排除 {len(main_['excluded_samples'])} 个 GT 仿真失败样本）\n")
    L.append('成功率 [物体自助 95% CI]：\n\n' + table(main_) + '\n')
    L.append('配对差（同一批样本）：\n\n' + paired(main_) + '\n')
    L.append(f"\n## 预先指定的分层子集（wave 1：10 橡皮泥 + 10 金属 + 6 果冻，种子 20261114；n={w1['n_samples']} 可评估）\n\n" + table(w1) + '\n' + paired(w1) + '\n')
    if sens:
        L.append('\n## 敏感性（全部样本）\n\n| 设置 | ' + ' | '.join(ZH[m] for m in M) + ' |\n|' + '---|' * (len(M) + 1) + '\n')
        for k, d in sens.items():
            L.append(f"| {k} | " + ' | '.join(pct(d['summary'][m]['all']['success']) if 'all' in d['summary'][m] else '—' for m in M) + ' |\n')
    if det.exists():
        L.append('\n## 仿真可重复性\n\n```\n' + det.read_text() + '\n```\n')
    L.append('\n## GT 可行性分布\n\n')
    od = [r['oracle_d'] for r in main_['samples']]
    L.append(f"oracle 深度分布：放弃 {sum(v is None for v in od)}，" + '，'.join(f"{v}: {sum(x == v for x in od)}" for v in pr['candidate_depths']) + '\n')
    L.append(f"\n固定单一深度在全部样本上的成功率：{json.dumps(main_['fixed_depth_success_global'])}\n")
    L.append(f"\n排除样本：{json.dumps(main_['excluded_samples'], ensure_ascii=False)}\n")
    extra = ROOT / 'docs/v3e/GRIPPER_TASK_ZH_discussion.md'
    if extra.exists():
        L.append('\n' + extra.read_text())
    (ROOT / 'docs/v3e/GRIPPER_TASK_ZH.md').write_text(''.join(L))
    print('wrote docs/v3e/GRIPPER_TASK_ZH.md')


if __name__ == '__main__':
    main()
