# V3a：跨引擎小样本适配试点（Warp-MPM → libuipc FEM）

更新：2026-11-13（UTC）。原始数据：`docs/v3a/UIPC_FEWSHOT.json`（最终版，source 模型训练全部结束后运行）；`docs/v3a/UIPC_FEWSHOT_prelim.json` 为 seed17 训练未结束时的预跑（K=8/32/all，数值与最终版一致或接近）。只用 val，无 test。

## 1. 协议

**目标引擎数据（data/v3_uipc）**：pyuipc 0.0.29，StableNeoHookean FEM（SNK1 系数按 `lambda+mu` 修正），BDF1 隐式积分，solver dt 5 ms，无重力、无接触。悬臂梁/板一端固定，端部受 −y 向力，按"0.2 s 线性加载 → 保持到 1.2 s → 0.1 s 卸载 → 自由振动"施加；同一几何体的力固定（E_min 时线性挠度为 0.25L），因此运动幅度和频率都由 E/ρ 决定。E 在 [2e4, 3e5] Pa 上对数均匀采样，ν∈[0.2, 0.45]，ρ=1000。共 7 种几何体，按几何体划分 train/val：train 用 beam_a/b/c/d 和 plate_e，每种 24 个，共 120 个；val 用 beam_f/g，每种 20 个，共 40 个。每个样本 3 个固定正交视角，224 px，64 帧（帧间隔 0.04 s，共 2.52 s），渲染器与 V2 相同。sample_pack.npz 中 flow/stress/force_mask 是全零占位，**不是**场监督。160/160 个样本全部成功，min Jacobian ≥ 0.938。生成前做过验证：同一几何体在 E=2e4/8e4/3e5 时，与首帧的平均像素差分别为 4.7/1.2/0.43，端部峰值位移分别为 39/8.6/1.6 px（见 `docs/v3a/check_sheets/`）。

**特征与目标空间**：使用冻结的 V-JEPA2 ViT-L 缓存（`data/prepared/v3_uipc_vjl`，配置 `configs/v3/v3_uipc_cache.json`，frames=32），与 V2 MPM 的缓存方式完全相同。目标是 MPM v2main 的 specific 空间，即 ln1p(E/(ρ/1000))，并用 MPM train 的统计量标准化（std=2.458 ln）。由于 ρ=1000，该目标就是 ln E。ν 同样标准化，但报告时还原为原始单位。

**Source 模型**：`runs/v2main_C_seed{17,29,43}/best_material.pt`，分别取第 16、7、11 个 epoch，均已早停结束。C 臂是 aux-field、无 feedback 的 head，在 MPM 上训练（三种本构，jelly/metal/plasticine）。在 MPM 同域 val 的 jelly 子集上，E 的 MAE 为 0.70/0.78/0.84 ln（约 0.31 std 单位），ν 的 MAE 约 0.09。

**比较的方法**（每个 seed 对应一个 source 模型和一种 K 子集；子集按几何体轮流抽取，K=4⊂8⊂16⊂32；超参数事先固定，不在 val 上选择）：
- `zero_shot`（K=0）：直接迁移。
- `affine_K`：用 K 个样本对 source 输出做仿射校准，E 和 ν 分别拟合。
- `headft_K`：只微调 source 的 param_head（2 层 MLP），作用在冻结的 pooled 表示上；300 步，lr 3e-4，MSE。
- `engcond_K`：**引擎条件化 head**。在冻结的 pooled 表示上加一个可学习的引擎 embedding，再对 (lnE, ν) 做逐引擎输出仿射；用 256 个 MPM jelly train 样本（引擎 0）和 K 个 uipc 样本（引擎 1）**联合训练**，每个 batch 两种引擎各 32 个，300 步。
- `scratch_K`：与 source 结构相同的轻量 head（2 层 temporal transformer 加 param head），随机初始化，只用 K 个 uipc 样本训练，600 步。它不使用 MPM 先验，但仍使用冻结的 V-JEPA 特征。
- `trainmean_K`：直接预测 K 个样本的目标均值，作为下界参照。
- "全监督"指 K=all=120（uipc 全部 train）。

**指标**：40 个 val 样本上 lnE 的 MAE（MPM 标准化单位，括号内为 ln 单位）、ν 的 MAE、E 的 Spearman（全部 val 一起算，以及按几何体分别计算后取平均）。所有数字都是 3 个 seed 的均值±标准差。

## 2. 结果（val，3 seed）

| 方法 | K | MAE lnE（std） | MAE lnE（ln） | MAE ν | Spearman E |
|---|---|---|---|---|---|
| zero_shot | 0 | 1.176±0.094 | 2.89 | 0.069 | −0.12±0.61 |
| affine | 8 / 32 / 120 | 0.227 / 0.222 / 0.221 | 0.56 / 0.55 / 0.54 | 0.063 | 0.40 / 0.58 / 0.40 |
| headft | 4 | 0.182±0.025 | 0.45 | 0.084 | 0.74 |
| headft | 8 | 0.120±0.025 | 0.30 | 0.080 | 0.89 |
| headft | 16 | 0.074±0.011 | 0.18 | 0.091 | 0.96 |
| headft | 32 | 0.062±0.004 | 0.15 | 0.089 | 0.96 |
| headft | 120 | 0.058±0.001 | 0.14 | 0.062 | 0.96 |
| engcond | 4 | 0.265±0.021 | 0.65 | 0.087 | 0.36 |
| engcond | 8 | 0.151±0.029 | 0.37 | 0.077 | 0.78 |
| engcond | 16 | 0.087±0.024 | 0.22 | 0.085 | 0.94 |
| engcond | 32 | 0.072±0.013 | 0.18 | 0.085 | 0.94 |
| engcond | 120 | 0.067±0.006 | 0.17 | 0.063 | 0.95 |
| scratch | 4 | 0.099±0.031 | 0.24 | 0.079 | 0.97 |
| scratch | 8 | 0.079±0.017 | 0.20 | 0.072 | 0.97 |
| scratch | 16 | 0.060±0.008 | 0.15 | 0.076 | 0.97 |
| scratch | 32 | 0.055±0.004 | 0.14 | 0.074 | 0.97 |
| **scratch（全监督）** | 120 | **0.045±0.005** | 0.11 | 0.064 | **0.987** |
| trainmean | 8 / 120 | 0.290 / 0.284 | 0.71 / 0.70 | 0.062 / 0.060 | — |

**达到全监督精度所需的 uipc 样本数**：参照为 scratch_all 的 0.045。要求误差不超过参照的 1.5 倍时，scratch 需要 K=16，headft 需要 K=32，engcond 需要 K=120；affine 无论 K 多大都达不到。要求不超过 1.25 倍时，只有 scratch 能做到，需要 K=32。要求不超过 1.1 倍时，所有小样本方法都达不到，只有 K=120 可以。完整曲线见 JSON 中的 `k_to_reach_full_supervision`。

**重仿真检查**：在 10 个 val 样本上（每个 val 几何体各 5 个）把预测的 (E, ν) 代回 uipc 重新仿真。K=0 的平均表面位移误差为 19.9 mm，K=8 的适配方法为 0.75–1.18 mm，K=32 为 0.77–0.92 mm；所有适配方法在 30/30 个配对样本上都优于 K=0。详见第 4 节；数据在 JSON 的 `resimulation_check` 字段。

## 3. 结论

1. **K=0 直接迁移失败**。所有 seed 都系统性低估 E，偏差约 −1.18 std，即 −2.9 ln，约低 18 倍。排序也不稳定，Spearman 在 −0.77 到 +0.70 之间，与 seed 有关。所以不同引擎之间同名参数的标度并不一致，MPM 上学到的 E 不能直接用在 FEM 上。
2. **少量 uipc 样本就能修正**。headft 在 K=8 时 MAE 从 1.18 降到 0.12，K=32 时降到 0.062，已接近 headft 全监督的 0.058。只做输出仿射校准（affine）不够，误差停在约 0.22，说明问题不只是线性的标度偏移，source 表示中 E 的排序本身就不可靠。
3. **在本试点中没有观察到 MPM 先验的收益**。在冻结的 V-JEPA 特征上，只用 uipc 从头训练的轻量 head（scratch）在每个 K 下都优于 headft 和 engcond；K=4 时就达到 0.099，Spearman 0.97。引擎条件化联合训练（engcond）在所有 K 下都比单纯的 headft 略差。学到的逐引擎仿射几乎没有变化（scale≈1，bias≈0.005），MPM 样本起到的是把模型拉回 MPM 映射的正则作用，而不是提供有用的先验。**这一结果不支持"MPM 预训练的引擎条件先验 + K-shot 优于从头训练"的说法**，至少在这个简单任务上不支持。
4. **ν 无法辨识**。所有方法的 ν MAE 都在 0.06–0.09，不优于 trainmean 的 0.060–0.062。这与 MPM 同域的情况一致：同域 ν 的 R²≈0。

## 4. 重仿真检查（预测参数 → uipc 运动）

挑选 10 个 val 样本：beam_f 和 beam_g 各取样本 id 最小的 5 个。把每个 seed 的预测 (E, ν) 代回原 case，网格、载荷和求解器都不变，重新做 uipc 仿真，再与观测到的 uipc 表面位移轨迹（64 帧 × 全部表面节点）比较。3 个 seed，每个方法 30 次仿真；oracle 只跑 10 次。全部成功，没有失败，也没有触发 E 截断。

| 方法 | 平均表面位移误差（mm，均值 / 中位数） | 端部 y 轨迹 RMSE（px，均值 / 中位数） | 相对误差中位数 |
|---|---|---|---|
| oracle（真值参数重跑） | 0.000 / 0.000 | 0.00 / 0.00 | 0.000 |
| zero_shot（K=0） | 19.85 / 17.30 | 71.2 / 72.4 | 12.8 |
| headft_8 | 1.07 / 0.53 | 4.73 / 2.69 | 0.56 |
| engcond_8 | 1.18 / 0.65 | 5.21 / 3.01 | 0.58 |
| scratch_8 | 0.75 / 0.36 | 3.29 / 1.38 | 0.46 |
| headft_32 | 0.92 / 0.20 | 4.12 / 1.06 | 0.25 |
| engcond_32 | 0.77 / 0.33 | 3.54 / 1.59 | 0.46 |
| scratch_32 | 0.77 / 0.27 | 3.50 / 1.05 | 0.27 |

- oracle 的误差约为 1e-11 m，说明 uipc 重仿真是确定性的，上表中的误差全部来自参数误差。
- 逐样本配对比较时，所有适配方法在 **30/30** 个样本上都优于 K=0。K=0 把 E 低估约 18 倍，梁的挠度因此大了一个数量级：误差约 2 cm，而观测位移的均值只有 0.3–6 mm。
- 在适配方法之间，K=32 优于 K=8；scratch ≈ headft ≈ engcond，与第 2 节的参数误差排序一致。由于样本少、误差分布偏斜，均值主要受少数小 E 样本影响，所以同时给出了中位数。
- ν 的误差对运动影响很小，所以这一检查基本只反映 E。

## 5. 局限（论文中必须写明）

- **规模很小，任务过于简单**：只有 160 个样本，val 只有 40 个、2 个几何体；只有一种动作（端部加载），只有弹性材料，E 只覆盖 15 倍范围（2e4–3e5 Pa，约 2.7 ln，相当于 MPM 标准化后 1.1 个 std）。仅凭"与首帧的平均像素差"这一个非学习线索，在 val 上与 E 的 Spearman 就达到 0.98（每个几何体内分别为 0.98 和 0.98），与最好的学习方法相当。因此这里的高 Spearman 主要说明该设置很容易，不能说明模型具备泛化的物理理解。
- **scratch 优于迁移的结论只适用于本设置**。在更难的目标域上（多种动作、接触、塑性、几何体差异更大），MPM 先验仍有可能带来帮助，但本试点没有给出这方面的证据。
- **引擎条件化只在冻结表示之上实现**：做法是 embedding 加到 pooled 表示上，再加逐引擎仿射，没有对 temporal transformer 做联合微调；MPM 锚点只用了 256 个 jelly 样本，原因是 I/O 很慢（约 20 MB/s）。更强的做法，例如在输入 token 级别注入引擎信息并联合训练整个 head，尚未测试。
- **"500 个仿真的域适配"对比**：本试点的全监督只有 120 个 uipc 样本，所以 K=16–32 达到全监督 1.5 倍以内，只能说明"在 ≤120 个样本的尺度上很快饱和"，**不能**直接与 500 个仿真的工作做定量比较。
- 超参数（步数、lr）事先固定，没有调；每个 K 只抽一个子集，seed 之间的方差同时包含 source 模型、子集和初始化三方面的差异。
- **数值阻尼**：BDF1 在 dt=5 ms 时会产生数值阻尼，这不是材料参数。uipc 的视频缺少 MPM 那样的 flow/stress 监督，只能训练参数 head。
- 两个引擎的本构不同：MPM 端是 V2 的本构，uipc 端是 StableNeoHookean。同名的 E/ν 不意味着物理上等价，K=0 时 −2.9 ln 的偏差就说明了这一点。

## 6. 文件

- 生成器：`scripts/v3/gen_uipc_video.py`；数据：`data/v3_uipc/{train,val}/<idx>/`，共 369 MB；验证用的对比图：`docs/v3a/check_sheets/`
- manifest：`scripts/v3/build_uipc_manifest.py` → `data/manifests/v3_uipc.jsonl`、`.stats.json`、`.specific.stats.json`
- 缓存：`configs/v3/v3_uipc_cache.json` → `data/prepared/v3_uipc_vjl`（8.6 GB）
- 小样本实验：`scripts/v3/uipc_fewshot.py` → `docs/v3a/UIPC_FEWSHOT.json`，其中包含每个 seed 的子集 id、逐样本预测和引擎仿射参数
- 重仿真：`scripts/v3/uipc_resim.py` → `docs/v3a/resim/seed*.json`，合并后写入 `UIPC_FEWSHOT.json` 的 `resimulation_check` 字段
- 队列：`docs/v3a/queue.json`（v3a_*）；日志：`runs/rq_logs/v3a_*`
