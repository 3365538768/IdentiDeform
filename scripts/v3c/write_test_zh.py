"""V3c: docs/v3c/BASELINES_TEST_ZH.md from BASELINES_TEST.json + BASELINES_VAL.json (numbers not hand-typed)."""
import json
from pathlib import Path
T = json.loads(Path('docs/v3c/BASELINES_TEST.json').read_text()); V = json.loads(Path('docs/v3c/BASELINES_VAL.json').read_text())
VAL_KEY = {'vjepa2_32f_B': 'vjepa2_32f_B (reference)', 'vjepa2_32f_C': 'vjepa2_32f_C (reference)', 'vjepa2_32f_D': 'vjepa2_32f_D (reference)'}
ZH = {'constant_train_mean': '常数（训练均值）', 'law_mean_gt_privileged': '律类均值（GT 律类，特权）', 'qwen3vl8b_zeroshot': 'Qwen3-VL-8B 零样本',
      'dinov3_static_first_frame_B': 'DINOv3 首帧静态（B）', 'dinov3_16f_B': 'DINOv3 16 帧 B', 'dinov3_16f_C': 'DINOv3 16 帧 C',
      'vjepa2_32f_B': 'V-JEPA 2 B', 'vjepa2_32f_C': 'V-JEPA 2 C（比较基准）', 'vjepa2_32f_D': 'V-JEPA 2 D',
      'vjepa2_ensemble_BCD9': 'V-JEPA 2 B/C/D 9 run 等权集成', 'vjepa2_ensemble_C3': 'V-JEPA 2 C 3 种子集成'}
g = lambda x, k='mean': x.get(k) if isinstance(x, dict) else x
f = lambda x, n=3: '–' if x is None else f'{x:.{n}f}'
def boot(m):
    rows = list(m['vs_vjepa_C'].values()) if 'vs_vjepa_C' in m else [s['vs_vjepa_C_same_seed'] for s in m.get('seeds', {}).values() if 'vs_vjepa_C_same_seed' in s]
    if not rows: return '（参照）'
    cis = [r['paired_bootstrap_95_ci'] for r in rows]; mean = sum(r['mean_object_delta_B_minus_A'] for r in rows) / len(rows)
    return f"{mean:+.3f}（C 更好 {sum(hi < 0 for lo, hi in cis)}/{len(cis)}，方法更好 {sum(lo > 0 for lo, hi in cis)}/{len(cis)}；{'; '.join(f'[{lo:+.3f},{hi:+.3f}]' for lo, hi in cis)}）"
L = ['# V2 main 测试集结果（V3c，一次性评估）', '',
     '由 `scripts/v3c/write_test_zh.py` 从 `docs/v3c/BASELINES_TEST.json` 和 `docs/v3c/BASELINES_VAL.json` 生成。test 共 300 个样本、10 个物体（与 train/val 无重叠，三类律各 100）。协议见 `docs/v3c/TEST_PROTOCOL.json`（冻结）与 `docs/v3c/TEST_READY.json`。每个 run 用验证集上选出的 best_material.pt，没有在 test 上做任何选择。集成规则沿用 v3b：z 取平均，律类按多数票，预先固定。', '',
     '物体 bootstrap 列：方法 − V-JEPA C 的物体宏平均 MAE 差（2000 次重采样），负值表示 V-JEPA C 更好；训练方法按同种子配对，确定性方法与集成分别与 C 的 3 个种子配对。', '',
     '| 方法 | 种子 | test 标准化 MAE | val 标准化 MAE | test 律类准确率 | val 律类准确率 | test 律内 r E/ρ | test 律内 r σY/ρ | test 物体 bootstrap 相对 V-JEPA C |', '|---|---|---|---|---|---|---|---|---|']
for r in T['table']:
    n = r['method']; m = T['methods'][n]
    if n.startswith('vjepa2_ensemble'):
        vm, vl = m['val']['standardized_mae'], m['val']['law_accuracy']
    else:
        v = V['methods'][VAL_KEY.get(n, n)]; vm, vl = g(v['standardized_mae']), g(v['law_accuracy']) if v.get('law_accuracy') is not None else None
    tm = m['standardized_mae']; ts = f"{f(g(tm))}±{f(g(tm, 'sd'))}" if isinstance(tm, dict) else f(tm)
    L.append(f"| {ZH.get(n, n)} | {len(m.get('seeds', {})) or '–'} | {ts} | {f(vm)} | {f(r['law_accuracy'])} | {f(vl)} | {f(r['E_over_rho_within_law_r'])} | {f(r['sigma_y_over_rho_within_law_r'])} | {boot(m)} |")
Path('docs/v3c/BASELINES_TEST_ZH.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))
