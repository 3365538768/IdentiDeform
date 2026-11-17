# V4 先验可用性快检（V4 vs V3，train 0–999，val 配对）

生成 2026-11-15T18:10:28Z；配对 val 样本 283。源 `docs/v4/PRIOR_CHECK.json`。V2 配方不变（configs/v2main/*，各自 train 标准化）。

| 组 | 种子 | E/ρ 误差 dec | 常数 dec | 相对常数增益 E/ρ | σ/ρ 误差 dec | 常数 dec | 相对常数增益 σ/ρ | 类别准确率 | 族内 r_S(E/ρ) | 族内 r_S(σ/ρ) |
|---|---|---|---|---|---|---|---|---|---|---|
| V3_t1000 B | [17, 29, 43] | 0.529 [0.485, 0.571] | 0.559 | 0.053 [0.025, 0.077] | 0.311 [0.286, 0.339] | 0.332 | 0.063 [0.035, 0.095] | 0.433 [0.409, 0.457] | 0.218 [0.129, 0.305] | 0.348 [0.202, 0.460] |
| V3_t1000 C | [17, 29, 43] | 0.526 [0.479, 0.570] | 0.559 | 0.059 [0.025, 0.094] | 0.311 [0.286, 0.340] | 0.332 | 0.063 [0.045, 0.082] | 0.466 [0.422, 0.520] | 0.214 [0.121, 0.292] | 0.336 [0.202, 0.455] |
| V3_t1000 D | [17, 29, 43] | 0.513 [0.460, 0.564] | 0.559 | 0.082 [0.044, 0.125] | 0.306 [0.277, 0.340] | 0.332 | 0.078 [0.048, 0.108] | 0.465 [0.427, 0.500] | 0.290 [0.177, 0.395] | 0.403 [0.282, 0.506] |
| V3_t1000 ensemble_9 | [0] | 0.514 [0.466, 0.559] | 0.559 | 0.081 [0.050, 0.115] | 0.304 [0.277, 0.336] | 0.332 | 0.082 [0.056, 0.111] | 0.473 [0.434, 0.513] | 0.278 [0.174, 0.372] | 0.421 [0.264, 0.537] |
| V4_t1000 B | [17, 29, 43] | 0.517 [0.462, 0.571] | 0.559 | 0.075 [0.021, 0.129] | 0.318 [0.291, 0.347] | 0.332 | 0.041 [0.003, 0.081] | 0.469 [0.426, 0.514] | 0.285 [0.188, 0.395] | 0.200 [0.072, 0.362] |
| V4_t1000 C | [17, 29, 43] | 0.514 [0.458, 0.564] | 0.559 | 0.081 [0.041, 0.128] | 0.317 [0.292, 0.345] | 0.332 | 0.043 [0.022, 0.069] | 0.465 [0.424, 0.505] | 0.284 [0.193, 0.402] | 0.256 [0.120, 0.394] |
| V4_t1000 D | [17, 29, 43] | 0.511 [0.454, 0.565] | 0.559 | 0.085 [0.022, 0.145] | 0.310 [0.281, 0.340] | 0.332 | 0.064 [0.024, 0.108] | 0.482 [0.408, 0.557] | 0.334 [0.245, 0.447] | 0.307 [0.168, 0.440] |
| V4_t1000 ensemble_9 | [0] | 0.504 [0.447, 0.556] | 0.559 | 0.099 [0.051, 0.152] | 0.312 [0.284, 0.341] | 0.332 | 0.059 [0.026, 0.099] | 0.498 [0.420, 0.575] | 0.343 [0.235, 0.480] | 0.357 [0.228, 0.482] |

## 配对差（同组；物体 bootstrap 95% CI；按 val 索引配对）

| 组 | 指标 | 差值 |
|---|---|---|
| V4_t1000 - V3_t1000 / B | decade_mae_E_over_rho | -0.012 [-0.031, 0.007] |
| V4_t1000 - V3_t1000 / B | decade_mae_sigma_y_over_rho | 0.007 [0.000, 0.014] |
| V4_t1000 - V3_t1000 / B | gain_over_const_E_over_rho | 0.021 [-0.012, 0.056] |
| V4_t1000 - V3_t1000 / B | gain_over_const_sigma_y_over_rho | -0.023 [-0.041, -0.000] |
| V4_t1000 - V3_t1000 / B | law_accuracy | 0.035 [-0.002, 0.082] |
| V4_t1000 - V3_t1000 / B | within_class_spearman_E_over_rho | 0.067 [0.017, 0.122] |
| V4_t1000 - V3_t1000 / B | within_class_spearman_sigma_y_over_rho | -0.148 [-0.237, -0.050] |
| V4_t1000 - V3_t1000 / C | decade_mae_E_over_rho | -0.012 [-0.023, -0.001] |
| V4_t1000 - V3_t1000 / C | decade_mae_sigma_y_over_rho | 0.007 [-0.000, 0.012] |
| V4_t1000 - V3_t1000 / C | gain_over_const_E_over_rho | 0.021 [0.001, 0.045] |
| V4_t1000 - V3_t1000 / C | gain_over_const_sigma_y_over_rho | -0.020 [-0.037, 0.000] |
| V4_t1000 - V3_t1000 / C | law_accuracy | -0.001 [-0.039, 0.034] |
| V4_t1000 - V3_t1000 / C | within_class_spearman_E_over_rho | 0.070 [0.024, 0.138] |
| V4_t1000 - V3_t1000 / C | within_class_spearman_sigma_y_over_rho | -0.080 [-0.150, -0.004] |
| V4_t1000 - V3_t1000 / D | decade_mae_E_over_rho | -0.002 [-0.020, 0.023] |
| V4_t1000 - V3_t1000 / D | decade_mae_sigma_y_over_rho | 0.004 [-0.013, 0.019] |
| V4_t1000 - V3_t1000 / D | gain_over_const_E_over_rho | 0.004 [-0.042, 0.035] |
| V4_t1000 - V3_t1000 / D | gain_over_const_sigma_y_over_rho | -0.013 [-0.057, 0.040] |
| V4_t1000 - V3_t1000 / D | law_accuracy | 0.016 [-0.031, 0.070] |
| V4_t1000 - V3_t1000 / D | within_class_spearman_E_over_rho | 0.044 [-0.040, 0.137] |
| V4_t1000 - V3_t1000 / D | within_class_spearman_sigma_y_over_rho | -0.096 [-0.205, 0.002] |
| V4_t1000 - V3_t1000 / ensemble_9 | decade_mae_E_over_rho | -0.010 [-0.024, 0.006] |
| V4_t1000 - V3_t1000 / ensemble_9 | decade_mae_sigma_y_over_rho | 0.008 [-0.004, 0.017] |
| V4_t1000 - V3_t1000 / ensemble_9 | gain_over_const_E_over_rho | 0.018 [-0.010, 0.044] |
| V4_t1000 - V3_t1000 / ensemble_9 | gain_over_const_sigma_y_over_rho | -0.023 [-0.052, 0.013] |
| V4_t1000 - V3_t1000 / ensemble_9 | law_accuracy | 0.025 [-0.028, 0.084] |
| V4_t1000 - V3_t1000 / ensemble_9 | within_class_spearman_E_over_rho | 0.066 [0.011, 0.129] |
| V4_t1000 - V3_t1000 / ensemble_9 | within_class_spearman_sigma_y_over_rho | -0.064 [-0.166, 0.038] |

## 每动作 / 每类（点估计）

**V3_t1000 B** 类别准确率/动作 {"bend": 0.42, "drop": 0.44, "press": 0.42, "shear": 0.45, "stretch": 0.44, "tap30hold": 0.41}；/类 {"jelly": 0.27, "metal": 0.43, "plasticine": 0.6}
- r_S(E_over_rho) /动作 {"bend": 0.2, "drop": 0.24, "press": 0.11, "shear": 0.23, "stretch": 0.22, "tap30hold": 0.27}；/类 {"jelly": -0.02, "metal": 0.47, "plasticine": 0.2}
- r_S(sigma_y_over_rho) /动作 {"bend": 0.64, "drop": 0.56, "press": 0.32, "shear": 0.03, "stretch": 0.44, "tap30hold": 0.41}；/类 {"jelly": null, "metal": 0.33, "plasticine": 0.37}

**V3_t1000 ensemble_9** 类别准确率/动作 {"bend": 0.5, "drop": 0.44, "press": 0.38, "shear": 0.5, "stretch": 0.54, "tap30hold": 0.49}；/类 {"jelly": 0.44, "metal": 0.49, "plasticine": 0.49}
- r_S(E_over_rho) /动作 {"bend": 0.27, "drop": 0.33, "press": 0.15, "shear": 0.3, "stretch": 0.25, "tap30hold": 0.29}；/类 {"jelly": 0.04, "metal": 0.51, "plasticine": 0.29}
- r_S(sigma_y_over_rho) /动作 {"bend": 0.71, "drop": 0.57, "press": 0.46, "shear": 0.16, "stretch": 0.41, "tap30hold": 0.49}；/类 {"jelly": null, "metal": 0.39, "plasticine": 0.46}

**V4_t1000 B** 类别准确率/动作 {"bend": 0.52, "drop": 0.49, "press": 0.45, "shear": 0.43, "stretch": 0.46, "tap30hold": 0.46}；/类 {"jelly": 0.65, "metal": 0.52, "plasticine": 0.23}
- r_S(E_over_rho) /动作 {"bend": 0.48, "drop": 0.37, "press": 0.14, "shear": 0.2, "stretch": 0.43, "tap30hold": 0.28}；/类 {"jelly": 0.13, "metal": 0.42, "plasticine": 0.3}
- r_S(sigma_y_over_rho) /动作 {"bend": 0.36, "drop": 0.27, "press": 0.15, "shear": -0.02, "stretch": 0.25, "tap30hold": 0.25}；/类 {"jelly": null, "metal": 0.26, "plasticine": 0.14}

**V4_t1000 ensemble_9** 类别准确率/动作 {"bend": 0.54, "drop": 0.54, "press": 0.42, "shear": 0.44, "stretch": 0.54, "tap30hold": 0.51}；/类 {"jelly": 0.74, "metal": 0.52, "plasticine": 0.23}
- r_S(E_over_rho) /动作 {"bend": 0.5, "drop": 0.34, "press": 0.24, "shear": 0.33, "stretch": 0.47, "tap30hold": 0.35}；/类 {"jelly": 0.22, "metal": 0.47, "plasticine": 0.33}
- r_S(sigma_y_over_rho) /动作 {"bend": 0.65, "drop": 0.42, "press": 0.39, "shear": -0.01, "stretch": 0.23, "tap30hold": 0.42}；/类 {"jelly": null, "metal": 0.32, "plasticine": 0.39}

## 参考（val，全量训练）

| 参考 | E/ρ dec | σ/ρ dec | 增益 E/ρ | 增益 σ/ρ | 类别准确率 | r_S(E/ρ) | r_S(σ/ρ) |
|---|---|---|---|---|---|---|---|
| V2 main (1955 train) / B (3 seeds) | 0.438 | 0.476 | 0.514 | 0.558 | 0.688 | 0.669 | 0.857 |
| V2 main (1955 train) / ensemble_BCD9 | 0.381 | 0.421 | 0.577 | 0.609 | 0.700 | 0.714 | 0.884 |
| V2 main (1955 train) / constant_train_mean | 0.902 | 1.078 | – | – | 0.329 | – | – |
| V3 main (2000 train) / B (3 seeds) | 0.484 | 0.295 | 0.134 | 0.111 | 0.517 | 0.370 | 0.441 |
| V3 main (2000 train) / ensemble_BCD9 | 0.467 | 0.289 | 0.165 | 0.130 | 0.548 | 0.458 | 0.485 |
| V3 main (2000 train) / constant_train_mean | 0.559 | 0.332 | – | – | 0.336 | – | – |

## 结论（v4p，2026-11-15 18:15 UTC，9 个 V4 运行全部完成）
- V4（shear/stretch/bend 加 g=2，其余动作 = V3 原样本）对先验没有实质帮助：9 头集成 E/ρ 相对常数增益 0.099 [0.051, 0.152] vs V3 0.081 [0.050, 0.115]（配对差 B/C 组 +0.02，CI 贴近 0）；σ/ρ 增益反而下降 0.059 vs 0.082（B 组配对差 −0.023 [−0.041, −0.000]），族内 r_S(σ/ρ) 下降 0.08–0.15。
- E/ρ 的小幅提升来自重新生成的 bend/stretch（族内 r_S 提高），与探针结论一致（g=2 只让弱动作的视频变化从 0.02–0.05 升到 0.025–0.07）。
- 两组都远低于 V2（增益 0.58/0.61）和 V3 全量 2000 先验（0.165/0.130）。
- 运行：runs/v3t1k_*、runs/v4t1k_*（best_material.pt + val_epoch_*）；manifest data/manifests/{v3main,v4main}_t1000；作业文件 docs/v4p/。
