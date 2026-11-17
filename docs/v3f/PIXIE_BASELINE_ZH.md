# V3f：PIXIE 静态基线（V2 val / test）

数据：`docs/v3f/PIXIE_BASELINE.json`；脚本：`scripts/v3f/pixie_features.py`（特征与推理）、`pixie_score.py`（评分）、`pixie_train_diag.py`（train 诊断）、`pixie_fig.py`（输入图）；日志：`docs/v3f/logs/pixie_*.log`。
图：`docs/v3f/pixie_val_inputs.png`、`docs/v3f/pixie_test_inputs.png`（每个物体 200 个视角中的 6 个，以及 PIXIE 给出的物体级输出）。

## 1. 做了什么

PIXIE（Le 等，github.com/vlongle/pixie）读入**静态**物体 64³ 的 CLIP 特征体素网格，用两个已发布的 3D UNet 预测逐体素材料类别（8 类，沿用 PhysGaussian 的 id：jelly/metal/sand/visplas/fluid/snow/stationary/background）以及 ρ、E、ν。权重来自 HF `vlongle/pixie`：`checkpoints_discrete/20250505_013424/epoch_510.pth` 和 `checkpoints_continuous_mse/20250805_040343/epoch_250.pth`，两者都严格加载（strict），没有缺失键或多余键。

**与原文一致的部分**
- 归一化与 `generate_blendernerf_data.py` 相同：最大包围盒边长缩放到 1，包围盒中心放在原点，坐标转为 Z 轴向上。
- 相机与 BlenderNeRF COS 设置相同：200 个视角，半径 U(0.6, 0.9)，上半球，焦距 20 mm、传感器 36 mm，512²，黑色背景。
- CLIP 特征与 f3rm `clip_extract.py` 一致：使用 OpenAI ViT-L/14@336px 的 MaskCLIP patch 特征，不做 center crop。直接调用 vendored 的 f3rm clip 包，权重通过 sha256 校验。
- 体素网格与 `voxelize.py` 相同（[-0.5, 0.5]³，体素边长 1/64，'ij' 索引），特征为 alpha × feature。
- UNet 结构与发布配置一致，反归一化沿用 `unscale_prediction` 和 `normalization_ranges.yaml`。

**近似的部分**（f3rm/nerfstudio/tinycudann/Blender 都未安装；按时间盒选择了备选方案）
1. 输入图像改为用我们的 3DGS 资产渲染（只用 DC 颜色，diff-gaussian-rasterization depth/alpha 版），没有用 Blender Cycles 渲染 mesh。
2. 没有训练 f3rm 蒸馏 NeRF 特征场，改为多视角反投影：每个占据体素取所有可见视角下双线性采样 patch 特征的平均（带深度测试）。不可见的内部体素复制最近的可见体素特征。掩码内 alpha=1，掩码外为 0。
3. 占据掩码 = 不透明度 > 0.1 的高斯中心所在体素，膨胀 1 格后填充内部（主变体 `filled`）；不填充的 `shell` 作为敏感性对照，两者结果几乎相同。

因此严格来说这是"PIXIE 发布的 UNet + 近似 F3RM 特征场"，不是完整原流程。不过 PIXIE 的核心假设，即"由外观/语义 CLIP 特征推断材料"，被完整保留。

**物体级聚合与类别映射**（先验固定，第一次评分前已写入 JSON 的 `protocol`）
- jelly→jelly；metal→metal；stationary→metal（刚体对应我们最硬的律）。
- sand/visplas/snow/fluid→plasticine。
- 物体类别 = 体素多数类（不计背景）；E、ν、ρ 取体素中位数。
- PIXIE 不输出屈服应力，所以 σ_y/ρ 取训练均值（z=0）。
- 每个物体只有一个预测，广播到该物体的全部样本。

**评分**：与 `docs/v3c/BASELINES_VAL.json` 完全相同：调用 `scripts/v3c/baselines_report.score`，比例空间的标准化 MAE（ln1p + train z-score）、律准确率和律内 r，再加上对 V-JEPA C 各 seed 及 BCD9 集成的物体配对 bootstrap。

## 2. 结果

| split | 方法 | 标准化 MAE ↓ | 律准确率 | E/ρ 律内 r |
|---|---|---|---|---|
| val (283 样本 / 10 物体) | PIXIE（原始输出） | **1.447** | 0.293 | −0.02 |
| val | PIXIE + train 仿射校准 | 0.857 | 0.293 | −0.02 |
| val | 常数（train 均值） | 0.858 | 0.329 | — |
| val | DINOv3 首帧静态 | 0.857 | 0.335 | 0.02 |
| val | **V-JEPA2 集成 BCD9（ours）** | **0.616** | **0.700** | **0.69** |
| test (300 样本 / 10 物体，只评一次) | PIXIE（原始输出） | **1.339** | 0.390 | −0.00 |
| test | PIXIE + train 仿射校准 | 0.853 | 0.390 | −0.00 |
| test | 常数（train 均值） | 0.853 | 0.333 | — |
| test | DINOv3 首帧静态 | 0.852 | 0.342 | — |
| test | **V-JEPA2 集成 BCD9（ours）** | **0.646** | **0.743** | **0.63** |

- 物体配对 bootstrap（2000 次，原始 PIXIE − V-JEPA C seed17）：val 差值为 −0.82，95% CI [−0.97, −0.66]；test 的 CI 为 [−0.86, −0.49]。两者都显著支持我们的方法。相对 BCD9 集成，val CI 为 [−0.98, −0.68]，test 为 [−0.89, −0.52]。
- 原始 PIXIE 的误差主要来自 E/ρ（val 的 z 误差为 2.73）。PIXIE 学到的是真实世界的绝对量级，例如它给塑料瓶的 E 约为 1.8e8 Pa、ρ 约为 2273。这与 V2 采样参数区间的尺度不同，所以比常数基线还差。
- **train 仿射校准**：PIXIE 也在 87 个 train 物体上运行，然后对每个参数用 OLS 拟合 z_target = a + b·z_pixie，只用 train 数据，拟合完再用于 val/test。这一步是在第一次 val 评分后、test 之前加入的，JSON 中有记录。三个参数拟合出的斜率都只有 ≈0.01–0.02，说明 PIXIE 的输出与我们的标签几乎无关。校准后的结果因此退化为常数基线：val 0.857 对 0.858，test 0.853 对 0.853。
- **train 诊断**（87 个物体、1955 个样本，样本数更大）：律准确率 0.326，多数类比例为 0.335；混淆矩阵三行几乎相同。PIXIE z 与目标 z 的相关性为 E/ρ r=0.009、ν r=−0.008、ρ r=0.013，律内 r 都在 |0.013| 以内。
- test 上律准确率 0.39 略高于 1/3，但只有 10 个物体，且每个物体同时含有三种律的样本，可以视为噪声。train 上的 0.326 更可靠。

**结论**：与数据设计一致，V2 中材料参数和律是对每个样本独立采样的，与物体外观无关（同一个物体在不同样本中会对应不同的律），所以任何只看静态外观的方法最多只能达到常数水平。PIXIE 的原始输出比常数还差（存在尺度失配），校准后与常数基线和 DINOv3 首帧基线持平。只有看到运动的 V-JEPA2 模型能从视频中读出律和 E/ρ。这说明 DeformFieldBench 的设计使材料无法从外观得到，而不是说 PIXIE 本身有缺陷：在 PixieVerse 那类"外观即材料"的数据上，PIXIE 的设计是合理的。

## 3. 局限

- 特征场用的是近似：没有训练 f3rm NeRF，渲染用的是 3DGS 而不是 Blender mesh。用反投影得到的 CLIP 特征，其分布可能与 PIXIE 训练时的特征不同，这会影响 PIXIE 的绝对输出。不过从物体级预测来看，类别在语义上是合理的：两个望远镜/水槽判为 metal，靴子判为 jelly，面包、汉堡、砖墙判为 sand。所以特征本身并没有失效。即使用完全忠实的流程，也无法绕过"V2 中外观与材料独立"这一点。
- 类别映射是人为设计的（stationary→metal，snow/sand→plasticine），但映射先验固定，换成任何映射也不能让律准确率超过外观与律独立时的水平。
- PIXIE 不预测屈服应力，σ_y/ρ 一项等同于常数基线。
- 每个 split 只有 10 个物体，物体级 bootstrap CI 较宽；train 诊断只是补充，不是选择依据。
- 原始变体的 val 和 test 都只报告一次。校准变体是在看过 val 之后加入的，test 只评了一次。

## 4. 产物与清理

- 新增的权重（保留）：`external/pixie_hf`（479 MB，两个 UNet）和 `external/clip_vitl14_336`（891 MB）。代码：`external/pixie`（只读，未执行任何脚本，只 import 了 f3rm clip 和 Wavelet-Generation 的网络定义）。隔离环境为 `.engine-envs/pixie`（13 MB）。
- 中间产物见 `docs/CLEANUP_2026-11-12.md` 中的 "V3f PIXIE" 一节。
