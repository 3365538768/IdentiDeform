"""V3c: render docs/v3c/BASELINES_ZH.md from docs/v3c/BASELINES_VAL.json (numbers never typed by hand)."""
import json, sys
from pathlib import Path
src = sys.argv[1] if len(sys.argv) > 1 else 'docs/v3c/BASELINES_VAL.json'
d = json.loads(Path(src).read_text()); M = d['methods']
def g(x, k='mean'):
    return x.get(k) if isinstance(x, dict) else x
def f(x, n=3):
    return '–' if x is None else f'{x:.{n}f}'
def boot(m):
    rows = list(m['vs_vjepa_C'].values()) if 'vs_vjepa_C' in m else [s['vs_vjepa_C_same_seed'] for s in m.get('seeds', {}).values() if 'vs_vjepa_C_same_seed' in s]
    if not rows: return '（参照）'
    deltas = [r['mean_object_delta_B_minus_A'] for r in rows]
    cis = [r['paired_bootstrap_95_ci'] for r in rows]
    fav_c = sum(hi < 0 for lo, hi in cis); fav_m = sum(lo > 0 for lo, hi in cis)
    ci = '; '.join(f'[{lo:+.3f},{hi:+.3f}]' for lo, hi in cis)
    return f'{sum(deltas)/len(deltas):+.3f}（{len(rows)} 对；CI 全负={fav_c}，全正={fav_m}；{ci}）'
ZH = {'constant_train_mean': '常数（训练均值）', 'law_mean_gt_privileged': '律类均值（GT 律类，特权）',
      'qwen3vl8b_zeroshot': 'Qwen3-VL-8B 零样本', 'dinov3_static_first_frame_B': 'DINOv3 首帧静态（B 头）',
      'dinov3_16f_B': 'DINOv3 16 帧 B', 'dinov3_16f_C': 'DINOv3 16 帧 C',
      'vjepa2_32f_B (reference)': 'V-JEPA 2 32 帧 B（主线）', 'vjepa2_32f_C (reference)': 'V-JEPA 2 32 帧 C（主线，比较基准）',
      'vjepa2_32f_D (reference)': 'V-JEPA 2 32 帧 D（主线）'}
L = ['# V2 main 验证集对比基线（V3c）', '',
     f'由 `scripts/v3c/write_zh.py` 从 `{src}` 自动生成。验证集 283 个样本、10 个物体，test 未使用。', '',
     '## 协议', '- 目标空间：specific（E/ρ、ν、ρ、σY/ρ；ln1p 后用训练集统计量标准化）。标准化 MAE 对所有计分的（样本，参数）项取平均。可辨识量为 E/ρ 和 σY/ρ；ρ 是预期不可辨识的对照量。',
     '- 训练出的方法：每个 run 取验证集 MAE 最好的 epoch（与 V2 main 报告的规则相同），对所有训练方法同样偏乐观。DINOv3 与 V-JEPA 的头部、预算、早停、划分都一致，差别只在骨干和帧数（DINOv3 16 帧，V-JEPA 32 步）。',
     '- 物体 bootstrap：成对比较“方法 − V-JEPA C”的物体宏平均 MAE 差（2000 次重采样）；负值表示 V-JEPA C 更好。训练出的方法按同种子配对；确定性基线分别与 C 的 3 个种子配对。只有 10 个物体，置信区间偏宽。', '',
     '## 结果', '| 方法 | 种子数 | 标准化 MAE（均值±sd） | E/ρ MAE(z) | σY/ρ MAE(z) | 律类准确率 | 律内 r：E/ρ | 律内 r：σY/ρ | 相对 V-JEPA C 的物体差（均值；CI） |', '|---|---|---|---|---|---|---|---|---|']
for row in d['table']:
    n = row['method']; m = M[n]; seeds = len(m.get('seeds', {}))
    mae = m['standardized_mae']
    mae_s = f"{f(g(mae))}±{f(g(mae,'sd'))}" if isinstance(mae, dict) else f(mae)
    pe = m['per_parameter_mae_z']; e = g(pe['E_over_rho']); s = g(pe['sigma_y_over_rho'])
    la = g(m['law_accuracy']) if m.get('law_accuracy') is not None else None
    L.append(f"| {ZH.get(n, n)} | {seeds or '–'} | {mae_s} | {f(e)} | {f(s)} | {f(la)} | {f(row['E_over_rho_within_law_r'])} | {f(row['sigma_y_over_rho_within_law_r'])} | {boot(m)} |")
q = M.get('qwen3vl8b_zeroshot', {})
L += ['', '## 说明', f"- VLM：可解析回答 {q.get('samples', 0) - q.get('unparseable_answers', 0)}/{q.get('samples', 0)}。但绝大多数回答只是照抄提示里给的参数范围下界（E=36 Pa, ν=0.1, ρ=300, 判为 jelly），与 R5 在 V1 上的现象一致（R5 为 347/399 条抄下界）。因此 VLM 比常数基线还差，这一点照实报告。提示词与 R5 相比只改了参数范围那一句（换成 V2 训练集范围）。",
      '- 律类均值（特权）使用 GT 律类，不能部署，只作为“只知道律类时能做到多好”的参照；律内 r 无定义。',
      '- 首帧静态基线：只看首帧外观（P1b 协议），检验外观本身能否泄漏材料信息。', '']
Path(sys.argv[2] if len(sys.argv) > 2 else 'docs/v3c/BASELINES_ZH.md').write_text('\n'.join(L) + '\n')
print('\n'.join(L))
