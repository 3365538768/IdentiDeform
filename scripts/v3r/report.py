"""V3r report: V3 gripper task with the V3 prior -> docs/v3r/RESULTS.json, docs/v3r/GRIPPER_TASK*.json, RESULTS_ZH.md.

Runs the unchanged V3e evaluator (scripts/v3e/plan_eval.py, FB_GRIP_DATA=data/v3_main/val) on data/v3r/runs for the main
setting and every V3e sensitivity setting, then adds the V2 (docs/v3e) side-by-side.

  .venv/bin/python scripts/v3r/report.py
docs/v3r/GRIPPER_TASK.json has the exact V3e schema (summary.<method>.all.{success, success_ci95_object_bootstrap,
regret_mean, regret_ci95_object_bootstrap}, n_samples) -> F6 in scripts/paper_figs/make_figs.py reads it by swapping
'docs/v3e/GRIPPER_TASK.json' for 'docs/v3r/GRIPPER_TASK.json'.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODS = ['gt', 'refined_top2', 'ensemble_prior', 'refined_best3', 'fixed_loo', 'lawmean_pred']
ZH = {'gt': 'GT 参数（oracle 孪生体）', 'refined_top2': '先验 top-2 + 12 次仿真修正（本文）', 'ensemble_prior': 'V3 先验，无修正（0 次仿真）',
      'refined_best3': '三律均值择优 + 12 次修正（无先验）', 'lawmean_pred': '预测律的训练均值参数', 'fixed_loo': '固定深度（留一调参）'}
SENS = {'sens_abscrush': ['--abs'], 'sens_dentp95': ['--metric', 'dent_p95'], 'sens_dmin010': ['--dmin', '0.10'],
        'sens_dmin014': ['--dmin', '0.14'], 'sens_tau002': ['--tau', '0.02'], 'sens_tau005': ['--tau', '0.05'],
        'sens_wloss_unaligned': ['--metric', 'wloss']}
V2_HEADLINE = {'ensemble_prior': 0.818, 'refined_top2': 0.818, 'refined_best3': 0.80, 'lawmean_pred': 0.618, 'fixed_loo': 0.855, 'gt': 1.0}


def run(out, extra):
    env = dict(os.environ, FB_GRIP_DATA='data/v3_main/val', FB_GRIP_TWINS='v3')
    cmd = [sys.executable, 'scripts/v3e/plan_eval.py', '--runs', 'data/v3r/runs', '--samples', 'docs/v3r/SAMPLES.json', '--out', out, *extra]
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stderr)
    return json.loads((ROOT / out).read_text()), r.stdout


def pct(v):
    return '–' if v is None else f'{100 * v:.1f}%'


def main():
    main_, log = run('docs/v3r/GRIPPER_TASK.json', [])
    wave1, _ = run('docs/v3r/GRIPPER_TASK_wave1.json', ['--only', 'wave1'])
    sens = {k: run(f'docs/v3r/GRIPPER_TASK_{k}.json', v)[0] for k, v in SENS.items()}
    v2 = json.loads((ROOT / 'docs/v3e/GRIPPER_TASK.json').read_text())
    v2s = {k: json.loads((ROOT / f'docs/v3e/GRIPPER_TASK_{k}.json').read_text()) for k in SENS}
    S = main_['summary']; S2 = v2['summary']
    groups = ['all', 'plastic_laws', 'plasticine', 'metal', 'jelly']
    tab = {}
    for m in METHODS:
        if 'all' not in S.get(m, {}):
            continue
        tab[m] = {'label': S[m]['label'], 'n': S[m]['n'], 'sims_per_object': S[m].get('sims_per_object'),
                  **{g: {k: S[m][g][k] for k in ('n', 'success', 'success_ci95_object_bootstrap', 'regret_mean', 'regret_ci95_object_bootstrap',
                                                  'utility_mean', 'utility_ci95_object_bootstrap', 'oracle_utility_mean', 'damage_rate',
                                                  'missed_grasp_rate')} for g in groups if g in S[m]},
                  'v2': {g: {'success': S2[m][g]['success'], 'success_ci95_object_bootstrap': S2[m][g]['success_ci95_object_bootstrap'],
                             'regret_mean': S2[m][g]['regret_mean'], 'n': S2[m][g]['n']} for g in groups if g in S2.get(m, {})},
                  'delta_success_v3_minus_v2_all': S[m]['all']['success'] - S2[m]['all']['success'],
                  'delta_regret_v3_minus_v2_all': S[m]['all']['regret_mean'] - S2[m]['all']['regret_mean']}
    # prior quality on the gripper set: law accuracy of the twins and |dlog10 E| vs GT
    import math
    acc = {}
    for m in ('ensemble_prior', 'refined_top2', 'refined_best3', 'lawmean_pred'):
        rows = [r for r in main_['samples'] if m in r['methods']]
        if not rows:
            continue
        law_ok = [r['methods'][m]['twin_law'] == r['law'] for r in rows]
        dE = [abs(math.log10(r['methods'][m]['twin_params']['E']) - math.log10(r['methods']['gt']['twin_params']['E'] / (r['methods']['gt']['twin_params']['density'] / 1000.)))
              for r in rows if 'gt' in r['methods']]
        acc[m] = {'n': len(rows), 'twin_law_accuracy': sum(law_ok) / len(rows), 'mean_abs_dlog10_E_over_rho': sum(dE) / max(len(dE), 1)}
    res = {'stage': 'v3r', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
           'question': 'V3e damage-aware gripper task re-run on V3 val samples with the V3 prior (and V3q refinement)',
           'protocol': main_['protocol'], 'protocol_changes_vs_v2': [
               'samples: V3 val C1 set (docs/v3r/SAMPLES.json = docs/v3q/PROTOCOL.json val, same c1_select rule) instead of V2 val',
               'twins: V3 prior ensemble (docs/v3q/PRIOR_ENSEMBLE_val.json from docs/v3p/HANDOFF.json); refined twins from V3q '
               '(V3 contact setup, per-law V3 bounds, V3 train law means)', 'law-mean twin: V3 train means (docs/v3q/LAW_MEANS.json)',
               'squeeze: unchanged (own jaws, g = 0, same depths, tau, d_min, crush metric); stray particles outside the grid interior are '
               'clamped in before the run (port of scripts/v2/contact_bc._clamp_particles, FB_SQUEEZE_CLAMP=1)'],
           'n_samples': main_['n_samples'], 'n_complete': main_['n_complete'], 'excluded_samples': main_['excluded_samples'],
           'table': tab, 'paired_success_differences': main_['paired_success_differences'],
           'fixed_depth_success_global': main_['fixed_depth_success_global'],
           'v2_fixed_depth_success_global': v2['fixed_depth_success_global'],
           'twin_quality_on_gripper_set': acc,
           'wave1': {m: wave1['summary'][m].get('all') for m in METHODS if 'all' in wave1['summary'].get(m, {})},
           'sensitivity': {k: {m: {'success': d['summary'][m]['all']['success'], 'regret_mean': d['summary'][m]['all']['regret_mean'],
                                   'v2_success': v2s[k]['summary'][m]['all']['success']}
                               for m in METHODS if 'all' in d['summary'].get(m, {})} for k, d in sens.items()},
           'oracle_depths': [r['oracle_d'] for r in main_['samples']],
           'v2_headline_success': V2_HEADLINE,
           'files': {'per_sample': 'docs/v3r/GRIPPER_TASK.json', 'wave1': 'docs/v3r/GRIPPER_TASK_wave1.json',
                     'sensitivity': [f'docs/v3r/GRIPPER_TASK_{k}.json' for k in SENS], 'samples': 'docs/v3r/SAMPLES.json',
                     'runs': 'data/v3r/runs/<index>/<method>/state.json',
                     'F6': "scripts/paper_figs/make_figs.py fig6 reads summary.<m>.all.{success, success_ci95_object_bootstrap, regret_mean, "
                           "regret_ci95_object_bootstrap} and n_samples: point it at docs/v3r/GRIPPER_TASK.json (same schema)"},
           'evaluator_stdout': log}
    (ROOT / 'docs/v3r/RESULTS.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
    # ---------------------------------------------------------------- markdown
    L = [f"# V3r：V3 夹爪任务（损伤感知挤压，V3 先验）", f"自动生成：`scripts/v3r/report.py`，{res['created_utc']}。数据：`docs/v3r/RESULTS.json`、`docs/v3r/GRIPPER_TASK.json`。", '',
         '## 协议', '- 与 V3e（V2）完全相同：两块刚性粘着夹板沿较窄水平轴各内移 d·W/2（1.0 s），保持到 1.5 s，松开，自由回弹到 4.0 s；g=0；候选深度 0.06–0.32 + d=0 参考；'
         '选预测永久压溃（Procrustes 对齐）≤3% 的最大深度，否则放弃；成功 = GT 执行压溃 ≤3%（或无可行深度时正确放弃）；物体簇自助 2000 次 95% CI。',
         '- 改动：样本 = V3 val 的 C1 集合（与 V3q 相同的 60 个，规则同 V2）；孪生体参数来自 V3 先验 / V3q 修正 / V3 训练均值；挤压前把网格外的游离粒子钳回网格内（移植 contact_bc 的修复）。', '',
         f"## 主结果（n={res['n_samples']} 可评估；排除 {len(res['excluded_samples'])} 个 GT 仿真失败样本）", '',
         '| 方法 | 成功率 [95% CI] | V2 成功率 | Δ(V3−V2) | 弹塑性 | 果冻 | 平均遗憾 [CI] | V2 遗憾 | 安全夹深 (oracle) | 损坏率 | 漏抓率 | 仿真/物体 |',
         '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for m, t in tab.items():
        a = t['all']; ci = a['success_ci95_object_bootstrap']; rc = a['regret_ci95_object_bootstrap']
        L.append(f"| {ZH[m]} | {pct(a['success'])} [{pct(ci[0])}, {pct(ci[1])}] | {pct(t['v2'].get('all', {}).get('success'))} | "
                 f"{100 * t['delta_success_v3_minus_v2_all']:+.1f} pp | {pct(t.get('plastic_laws', {}).get('success'))} | {pct(t.get('jelly', {}).get('success'))} | "
                 f"{a['regret_mean']:.3f} [{rc[0]:.3f}, {rc[1]:.3f}] | {t['v2'].get('all', {}).get('regret_mean', float('nan')):.3f} | "
                 f"{a['utility_mean']:.3f} ({a['oracle_utility_mean']:.3f}) | {pct(a['damage_rate'])} | {pct(a['missed_grasp_rate'])} | {t['sims_per_object']:.1f} |")
    L += ['', '配对差（同一批样本，物体自助 95% CI）：', '', '| 对比 | 成功率差 | 遗憾差 | n |', '|---|---|---|---|']
    for k, v in main_['paired_success_differences'].items():
        if k.startswith('gt-'):
            continue
        c = v['success_ci95_object_bootstrap']; g = v['regret_ci95_object_bootstrap']
        L.append(f"| {k} | {100 * v['success_diff']:+.1f} pp [{100 * c[0]:+.1f}, {100 * c[1]:+.1f}] | {v['regret_diff']:+.3f} [{g[0]:+.3f}, {g[1]:+.3f}] | {v['n']} |")
    L += ['', '孪生体质量（夹爪样本上）：', '', '| 方法 | 律正确率 | 平均 \\|Δlog10 E/ρ\\| |', '|---|---|---|']
    for m, v in acc.items():
        L.append(f"| {ZH[m]} | {pct(v['twin_law_accuracy'])} | {v['mean_abs_dlog10_E_over_rho']:.2f} |")
    L += ['', '## 敏感性（全部样本，成功率；括号内为 V2）', '', '| 设置 | ' + ' | '.join(ZH[m] for m in METHODS if m in tab) + ' |',
          '|---|' + '---|' * len(tab)]
    for k, d in res['sensitivity'].items():
        L.append(f'| {k} | ' + ' | '.join(f"{pct(d[m]['success'])} ({pct(d[m]['v2_success'])})" for m in METHODS if m in d) + ' |')
    L += ['', '## 固定深度（全局，非留一）', '', '| 深度 | V3 | V2 |', '|---|---|---|']
    for k, v in main_['fixed_depth_success_global'].items():
        L.append(f"| {k} | {pct(v)} | {pct(v2['fixed_depth_success_global'].get(k))} |")
    if res['excluded_samples']:
        L += ['', '排除样本：' + '；'.join(f"#{e['index']} {e['law']} {e['object']}（{e['reason']}）" for e in res['excluded_samples'])]
    L += ['', 'F6：`scripts/paper_figs/make_figs.py` 的 fig6 读 `summary.<方法>.all.{success, success_ci95_object_bootstrap, regret_mean, regret_ci95_object_bootstrap}` 与 `n_samples`；'
          '`docs/v3r/GRIPPER_TASK.json` 结构与 V2 完全相同，把路径换成它即可。']
    (ROOT / 'docs/v3r/RESULTS_ZH.md').write_text('\n'.join(L) + '\n')
    print(log)


if __name__ == '__main__':
    main()
