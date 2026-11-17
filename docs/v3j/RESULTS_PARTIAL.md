# V3j 审稿人对照实验：部分结果（2026-11-15 04:10 UTC 暂停）

**状态：未完成。** 用户在 04:05 UTC 暂停了 v3j：所有 GPU 改去用修正后的边界条件重新生成数据集，主会话停掉了所有 v3j 任务。
下面的数字来自 `runs/v3j_done` 里已完成的 state（`scripts/v3j/report.py` 只读 `complete` 的 state，等同于 allow-partial），
**不是**预注册的终稿。由于边界条件有误，这些仿真是在旧数据上跑的，重新生成数据后需要整体重跑。
仅用 val 划分，test 未动。预注册：`docs/v3j/PROTOCOL.json`；完整表格：`docs/v3j/tables_partial.md`、`docs/v3j/RESULTS.json`；图：`docs/v3j/curves.png`。

## 覆盖情况
- 子集：v3h 30 样本按分层选取顺序的前 20 个 `[210,219,180,276,297,141,283,202,181,295,118,52,248,113,272,134,119,71,231,228]`。
- Exp 1（CMA-ES）：只有 7–8 个样本完成了 budget 12（cma_prior 8 / cma_gtmean 8 / cma_best3 7），**没有任何 budget-24 结果**。
  这几个样本是分层顺序里最靠前的，大部分是 jelly（1 维搜索），**不具代表性**。03:12 时进度预测会超出截止时间，
  所以按预注册规则先砍掉了 cma_prior 的 budget 24（`docs/v3j/DEVIATIONS.md`）。
- Exp 2（4 维 compass）：u4_prior 有 16/20 个样本跑到 budget 24，u4_gtmean 有 17/20；
  两臂和冻结基线共同的样本有 15 个；回放 IoU 有 10 个共同样本。
- 失败：113、231（两个已知的不稳定样本，C1 中同样失败）、202（u4_gtmean）。按 C1 规则，失败的仿真记 loss 9，不丢样本。
- 与 data/v3b_refine 中参数完全相同的仿真直接复用，并计入预算（u4 复用 128/156 次，主要来自前几步的 E 移动）。

## Exp 2（15 个共同样本）：可辨识 z-MAE；Δ 为相对 prior_top1，括号内为配对 object bootstrap 95% CI
| arm | k=0 | k=4 | k=12 | k=24 |
|---|---|---|---|---|
| V-JEPA prior，2-D compass（冻结） | 0.234 | 0.253 | 0.247 | 0.204 |
| V-JEPA prior，4-D compass（解冻 h、ν） | 0.234 | 0.253 | 0.267 (Δ+0.020 [−0.003,+0.066]) | 0.223 (Δ+0.019 [−0.007,+0.046]) |
| GT 律均值（特权），2-D | 0.630 | 0.617 | 0.321 | 0.189 |
| GT 律均值（特权），4-D | 0.630 | 0.617 | 0.349 | 0.295 (Δ+0.091 [−0.065,+0.211]) |

k=24 时：|hardening 误差| 0.083→0.150（prior，n=8），0.098→0.109（GT 均值）；|ν 误差| 0.082→0.087 / 0.082→0.107；
观测拟合 IoU 0.867→0.850 / 0.904→0.883；未见动作回放 IoU 0.860→0.839 / 0.841→0.783（n=10）。
k≤4 时 4-D 与 2-D 完全相同，因为两者前几步走的都是 E 方向。

**初步结论（Exp 2）：** 在 24 次仿真内，解冻 hardening/ν **没有降低**这个底线。4-D 搜索把预算分到了 h/ν 的轮询上，
E/σ_y 误差、拟合和回放都略微变差，h/ν 本身也没有被找回。所以在这个预算下，底线并不能用冻结的 nuisance 参数来解释；
它与可辨识性和预算都一致（更多预算的 4-D 搜索未测）。prior 在 4-D 下仍保留低预算优势（k=0/4 时 Δ ≈ −0.38 vs GT 均值），
与 2-D 的情况相同；k=24 时 4-D 的 prior 优于 4-D 的 GT 均值（0.223 vs 0.295，CI 未做该配对）。

## Exp 1（≤8 个样本，只到 k=12，以 jelly 为主，仅作参考）
k=12：prior compass 0.085，prior CMA 0.145，GT 均值 compass 0.048，GT 均值 CMA 0.257，best3 compass 0.253，best3 CMA 0.331。
k=4：prior CMA 0.211 vs GT 均值 CMA 0.386 vs best3 CMA 0.749。在这些样本上，CMA-ES 的无先验基线在 k=4/12 都**没有**比 compass 更强
（GT 均值 CMA 在 k=12 比 GT 均值 compass 差），prior 在低预算下的优势在 CMA 下依然存在。但 n 太小、
样本几乎都是 1 维 jelly，**不能下结论**，需要完整重跑。

## 代码与恢复
- 代码：`scripts/v3j/{make_protocol,c1j_refine,pool,report,tables_md,make_figure,fix_complete}.py`；state 位于 `data/v3j_refine/<arm>/<idx>/state.json`，
  可断点续跑，jobs 在 `docs/v3j/jobs.json`（完整列表见 `jobs_full.json`）。
- 已修复的 bug：旧版 `--replay` 会清掉 state 的 `complete` 标志（搜索结果不受影响）；`fix_complete.py` 已恢复了这些标志。
- 清理：123 个被中断的渲染目录和 14 个 GT 回放 npz 已删除（记录在 docs/CLEANUP_2026-11-12.md）。
