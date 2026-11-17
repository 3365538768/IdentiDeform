# P3 / P4a 结果（2026-11-11，A800-2）

原始数据：`docs/p3/P3_RESULTS.json`、`docs/p4/P4A_SPECTRUM.json`。选模只用验证集，test 未读取。

## P3：V-JEPA 2 ViT-L 末 4 块微调（C 组，原论文目标空间）

实现：缓存第 20 块输入（`scripts/prepare_vjepa_cache.py --hidden-layer 20`），训练末 4 块 + 最终 LayerNorm + 头（`TailFinetuneFieldModel`）。与完整模型前向的相对误差 0.8%（bf16 噪声），等价于无增强的末 4 块微调。

| 组 | 验证 MAE（3 seeds） | 族准确率 | 相对冻结 | 物体配对区间 |
|---|---:|---:|---:|---|
| 冻结 P1 Cvjn | 0.576 ± 0.009 | 0.772 | | |
| 微调 backbone_lr 1e-5 | 0.584 ± 0.017 | 0.778 | −1.4% | 3/3 跨 0 |
| 微调 backbone_lr 3e-5 | 0.570 ± 0.008 | 0.774 | +1.0% | 3/3 跨 0 |

结论：未过预登记 2% 门槛，**不做 test 评估**。族分类与族内相关均无变化。冻结 V-JEPA 表征不是当前瓶颈。

## P4a：32 fps 重放（30 个验证样本，3 族 × 5 动作 × 2，原参数/物体/动作）

上游管线已在 a800-2 端到端复现（修复 taichi 粒子填充越界循环，`scripts/v2/patch_filling.py`；使用上游自带 3DGS 光栅器）。单样本约 3.3 分钟（每卡 2 并发），8 卡约 300 样本/小时。

物体区域逐像素亮度的时间功率谱，高于各采样方式奈奎斯特频率的能量占比：

| 族 | 全程 >2Hz（DINO 16 帧） | >4Hz（V-JEPA 32 步） | >8Hz（V1 16fps） | 释放后 1 s 内 >8Hz |
|---|---:|---:|---:|---:|
| jelly | 1.5% | 0.4% | 0.1% | 0.3% |
| plasticine | 2.9% | 0.9% | 0.3% | 1.1% |
| metal | 2.3% | 0.9% | 0.2% | 7.2% |

结论：V1 视频几乎全部是低频的准静态运动，**提高帧率本身几乎不增加信息**。

### 根因：上游阻尼按子步施加

`add_damping_via_grid` 在每个子步（dt=1e-4）把网格速度乘以 `grid_v_damping_scale`。每秒等效衰减：jelly 0.999^10000≈5e-5，plasticine ≈3e-4，metal ≈7e-3；另有 rpic_damping 0.25–0.45。立方体 tap 测试（jelly E=1e4）：V1 阻尼下释放后约 0.3 s 完全静止；低阻尼（各族相同）下以 2.65 Hz 持续振荡 3.5 s，幅度约 ±4 px。

含义：
1. V1 中 press/shear/bend/stretch 是位移控制 + 强阻尼，形变近似与 E 无关，只有塑性（σ_y）和释放瞬间带少量 E/ρ 信息——与 R1/P1 中族内 E 相关仅 0.1–0.2、metal σ_y/ρ 相关 0.6 一致。
2. 阻尼按族取不同常数，本身是一个与族相关、与标签无关的线索。
3. Data V2 的关键改动应为“低且各族相同的阻尼 + 以释放后自由运动为主的激励”，而不是更高帧率或更大骨干。P4b 扫描正在量化这一点。

## P4b：激励 × 阻尼扫描（立方体，相邻半数量级的形变 RMS 差，像素，中位数）

| 量 | 动作 | V1 阻尼 | 低阻尼（各族相同） |
|---|---|---:|---:|
| jelly E/ρ | press | 2.9 | 4.2 |
| plasticine E/ρ | press / tap | 2.16 / 0.82 | 3.53 / 1.76 |
| plasticine σ_y/ρ | press / tap | 0.24 / 0.04 | **2.36 / 0.92** |
| metal σ_y/ρ | press / tap | 1.19 / 0.33 | 1.88 / **2.67** |

- 低阻尼 + tap 下主振荡频率随 E/ρ 单调上升（plasticine 0.3→16 Hz），提供直接的频率读数；V1 阻尼下所有频率都≈0（无振荡）。
- plasticine σ_y/ρ ≥10^3.5（tap）/10^4（press）时差异为 0：不屈服，需作删失标签。
- jelly E/ρ ≥10^4.5 在低阻尼下差异突增（18–46 px），疑为释放后弹跳/离地，需在渲染视频中核实。
- V1 的 plasticine 仿真参数里没有 yield_stress（用上游默认值），与 R1 中“plasticine 无 σ_y 标签”一致，rebuttal 中需说明。
- metal E/ρ 扫描（子步 2e-5）进行中。

## P4c（进行中）：配对 V2 试点

两臂物体、动作（press / tap）、参数、视角完全相同，只差阻尼（low vs V1），各 480 train + 120 val（物体沿用 V1 划分），32 fps × 4 s。参数按可辨识量采样（E/ρ、σ_y/ρ 对数均匀，ρ 独立），plasticine σ_y/ρ>10^3.75 标记删失。生成后：V-JEPA 冻结特征 + C 头，每臂 3 seeds，比较验证集族内 r(E/ρ)、r(σ_y/ρ) 与可辨识量 MAE。生成脚本 `scripts/v2/gen_v2.py`，tap 动作 `scripts/v2/upstream_main.py`。

### P4c 生成过程中的观察（2026-11-11）

- 单样本中位 195 s（每卡 2 并发），1200 个配对样本约 4.5 小时。
- 失败约 2%，均为 plasticine 的 CUDA 700（粒子越出网格）；同一参数在两臂同时失败，属参数相关。manifest 只保留两臂都成功的样本，保持配对；失败与参数相关会带来轻微选择偏差，两臂一致。
- **低阻尼 tap 的漂移**：重力为 0，jelly E/ρ ≥10^3.4 释放后从地面反弹并以恒速向上漂移（4 s 约 50 px，未出画面）。这是“反弹速度”线索而非振荡线索；正式 V2 应开重力或使用粘性地面。
- P4d 已排队（生成完成后自动开始）：两臂分别建 manifest/统计量 → V-JEPA ViT-L 冻结缓存（128 帧均匀取 64 帧）→ C 头 × 3 seeds，可辨识目标；比较验证集族内 r(E/ρ)、r(σ_y/ρ) 与 MAE。

### 11-11 连接中断排查

- 机器与容器均未重启：宿主 uptime 65 天，容器 PID 1 自 11-03 06:18 起运行，实验进程持续运行；sshd 期间一直正常认证。
- 原因：VS Code 远程 Pylance 对整个 `linrui/` 工作区建索引（多个 venv、数据、run 输出、递归符号链接），03:40–05:05 共 95 次 `JavaScript heap out of memory`，约每 50 s 崩溃重启，导致 VS Code 连接反复断开、重连很慢。P4c 每样本写约 1500 个中间 PNG（GPFS 上累计 >100 万个小文件）加重了文件监视与元数据负载。
- 修复：`linrui/.vscode/settings.json` 与 `/root/.vscode-server/data/Machine/settings.json`（原文件备份为 `.before-20261111`）限定 `python.analysis.include` 为 src/scripts，关闭索引，排除 data/runs/venv/weights/cache 等的分析、文件监视与搜索；`scripts/v2/gen_v2.py` 在 sample_pack 写出后删除中间 PNG，已清理 768 个完成样本（每样本约 1540→6 个文件）。
