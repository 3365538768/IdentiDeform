"""docs/v3x/RESULTS.json -> docs/v3x/RESULTS_ZH.md (tables only; conclusions appended by hand in the CONCLUSION block)."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
R = json.loads((ROOT / "docs/v3x/RESULTS.json").read_text())
NAME = {"prior_top1": "prior-top1", "lawmean_best3": "law-mean-best3", "lawmean_gt": "law-mean-GT（特权）", "prior_top2": "prior-top2"}
f = lambda v: "–" if v is None else f"{v:.3f}"
L = [f"# v3x：test 精修扩样（新增 30 样本，合并 60）\n", f"生成时间：{R['generated_utc']}。协议：`docs/v3x/PROTOCOL.json`（运行前写定）。数据：`docs/v3x/RESULTS.json`。指标：可辨识 z-MAE（E/ρ、σ_y/ρ pooled），越低越好。\n",
     "新 30 样本进度：" + "；".join(f"{NAME[a]} 完成 {v['complete']}/30（已仿真 {v['sims_done']} 次）" for a, v in R["progress_new"].items()) + "\n"]
CON = ROOT / "docs/v3x/CONCLUSION_ZH.md"
if CON.exists(): L.append(CON.read_text() + "\n")
for name, title in (("combined_60", "合并 60 样本"), ("new_30", "新 30 样本（复现）"), ("original_30", "原 30 样本（v3d，核对）")):
    S = R[name]; L.append(f"\n## {title}\n\n### 各臂误差（每格样本数 n / 物体数）\n\n| 方法 | k=0 | k=4 | k=12 | k=24 |\n|---|---|---|---|---|")
    for a, t in S["arm_table_reached_k"].items():
        L.append(f"| {NAME[a]} | " + " | ".join((f"{t[str(k)]['mae']:.3f}（{t[str(k)]['n']}/{t[str(k)]['objects']}）" if str(k) in t and not (a == "lawmean_best3" and k == 0) else "–") for k in (0, 4, 12, 24)) + " |")
    L.append("\nlaw-mean-best3 的 k=0 不存在（至少 3 次探测）；配对比较中 k=0 用 best3 的 k=3。prior-top2 的 k=0 与 prior-top1 相同（同一集成起点）。\n\n### 配对差异（仅两臂都已完成 24 次的样本；物体级 bootstrap 95% CI，2000 次，seed 17）\n\n| 比较 | n / 物体 | k=0 | k=4 | k=12 | k=24 |\n|---|---|---|---|---|---|")
    for p, row in S["paired"].items():
        if p.endswith('_reached_k') or not row["n_paired_complete"]: continue
        a, b = p.split("_minus_")
        L.append(f"| {NAME[a]} − {NAME[b]} | {row['n_paired_complete']} / {row['objects']} | " + " | ".join(
            (f"{row[f'k{k}']['mean']:+.3f} [{row[f'k{k}']['ci95'][0]:+.3f}, {row[f'k{k}']['ci95'][1]:+.3f}]" if f"k{k}" in row else "–") for k in (0, 4, 12, 24)) + " |")
    L.append("\n### 配对差异（部分结果：两臂都已达到预算 k 的样本，含未跑完 24 次的搜索；每格 n）\n\n| 比较 | k=0 | k=4 | k=12 | k=24 |\n|---|---|---|---|---|")
    for p, row in S['paired'].items():
        if not p.endswith('_reached_k') or not row: continue
        a, b = p[:-len('_reached_k')].split('_minus_')
        L.append(f"| {NAME[a]} − {NAME[b]} | " + " | ".join((f"{row[f'k{k}']['mean']:+.3f} [{row[f'k{k}']['ci95'][0]:+.3f}, {row[f'k{k}']['ci95'][1]:+.3f}] (n={row[f'k{k}']['n']})" if f'k{k}' in row else "–") for k in (0, 4, 12, 24)) + " |")
    for key, lab in (('lawmean_gt_sims_to_reach_prior_k0', '24 次内'), ('lawmean_gt_sims_to_reach_prior_k0_within12', '只用到 12 次（含部分完成的搜索）')):
      sp = S.get(key)
      if sp:
        hi = f(sp['ci95'][1]) if sp['ci95'][1] else '>%d' % sp['kmax']
        mid = f(sp['sims']) if sp['sims'] else '>%d（删失）' % sp['kmax']
        L.append(f"\n**{lab}，特权 law-mean-GT 追平先验 k=0（{sp['prior_k0_mae']:.3f}）所需仿真次数：** {mid} [{f(sp['ci95'][0])}, {hi}]，n = {sp['n']}，{sp['objects']} 个物体，bootstrap 删失比例 {sp['frac_censored']:.2f}。")
    if False:
        L.append(f"\n**特权 law-mean-GT 追平先验 k=0（{sp['prior_k0_mae']:.3f}）所需仿真次数：** {f(sp['sims']) if sp['sims'] else '>24（删失）'} [{f(sp['ci95'][0])}, {f(sp['ci95'][1]) if sp['ci95'][1] else '>24'}]，n = {sp['n']}，{sp['objects']} 个物体，bootstrap 删失比例 {sp['frac_censored']:.2f}。")
    if S["replay_iou_unseen_action_k24"]:
        L.append("\n**未见动作重放 IoU（k=24）：** " + "；".join(f"{NAME[a]} {v['iou_k24']:.3f}（n={v['n']}）" for a, v in S["replay_iou_unseen_action_k24"].items()))
    if S["law_accuracy"]:
        L.append("\n**k=24 law 准确率：** " + "；".join(f"{NAME[a]} {v['k24_law_acc']:.3f}（n={v['n']}）" for a, v in S["law_accuracy"].items()))
    L.append("\n**失败仿真：** " + "；".join(f"{NAME[a]} {v['failed']}/{v['sims']}" for a, v in S["failures"].items()))
(ROOT / "docs/v3x/RESULTS_ZH.md").write_text("\n".join(L) + "\n")
