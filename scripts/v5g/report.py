"""V5g report: runs scripts/v5g/plan_eval.py (main + pre-registered sensitivities) -> docs/v5g/RESULTS.json, RESULTS_ZH.md.
  .venv/bin/python scripts/v5g/report.py [--runs data/v5g/runs/val]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ZH = {'gt': 'GT oracle', 'prior_vt': '(c) 视觉+触觉先验孪生体', 'prior_vision': '(b) 视觉先验孪生体', 'lawmean_pred': '(a) 类均值孪生体',
      'fixed_loo': '(d) 固定抓取（留一）'}
SENS = {'drop0.4': {'V5G_T_DROP': '0.4'}, 'drop0.6': {'V5G_T_DROP': '0.6'}, 'dmg0.03': {'V5G_T_DMG': '0.03'}, 'dmg0.08': {'V5G_T_DMG': '0.08'}}


def ev(runs, out, env=None):
    r = subprocess.run([sys.executable, 'scripts/v5g/plan_eval.py', '--runs', runs, '--out', out], cwd=ROOT,
                       env=dict(os.environ, **(env or {})), capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stderr)
    return json.loads((ROOT / out).read_text()), r.stdout


def pct(v):
    return '–' if v is None else f'{100 * v:.1f}%'


def ci(c, p=True):
    if c is None or c[0] is None:
        return ''
    return f' [{pct(c[0])}, {pct(c[1])}]' if p else f' [{c[0]:+.2f}, {c[1]:+.2f}]'


def main():
    a = argparse.ArgumentParser(); a.add_argument('--runs', default='data/v5g/runs/val'); args = a.parse_args()
    M, log = ev(args.runs, 'docs/v5g/RESULTS_core.json')
    sens = {k: ev(args.runs, f'docs/v5g/RESULTS_sens_{k}.json', e)[0] for k, e in SENS.items()}
    S = M['summary']; n = M['n_samples']
    res = {'stage': 'v5g', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'prereg': 'docs/v5g/PREREG_2026-11-16.md',
           'n_samples': n, 'thresholds': M['thresholds'], 'candidates': M['candidates'], 'summary': S, 'paired': M['paired'],
           'gt_outcome_by_candidate': M['gt_outcome_by_candidate'], 'oracle_histogram': M['oracle_histogram'],
           'fixed_global_success': M['fixed_global_success'],
           'sensitivity': {k: {m: {x: v['summary'][m][x] for x in ('success', 'drop', 'damage', 'level_regret')} for m in v['summary']}
                           for k, v in sens.items()},
           'per_sample': 'docs/v5g/RESULTS_core.json', 'figure': 'docs/paper_figs/F6b_grasp.pdf (scripts/v5g/make_fig.py)', 'evaluator_stdout': log}
    import numpy as np
    feas = [r for r in M['samples'] if r['oracle']]
    def bootm(rs, f, n=2000):
        rng = np.random.default_rng(0); objs = sorted({r['object'] for r in rs}); by = {o: [r for r in rs if r['object'] == o] for o in objs}
        v = [f([r for o in rng.choice(objs, len(objs)) for r in by[o]]) for _ in range(n)] if objs else []
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [None, None]
    fs = {'definition': 'samples with at least one GT-feasible grasp (descriptive; selected on GT outcomes)', 'n': len(feas), 'arms': {}, 'paired': {}}
    for m in S:
        rs = [r for r in feas if m in r['methods']]
        if rs:
            g = lambda k: (lambda p: float(np.mean([x['methods'][m][k] for x in p])))
            fs['arms'][m] = {'n': len(rs), **{k: g(k)(rs) for k in ('success', 'drop', 'damage', 'missed')}, 'success_ci95': bootm(rs, g('success'))}
    for nm, (x, y) in {'vt-minus-vision': ('prior_vt', 'prior_vision'), 'vision-minus-classmean': ('prior_vision', 'lawmean_pred'),
                       'vt-minus-fixed': ('prior_vt', 'fixed_loo')}.items():
        rs = [r for r in feas if x in r['methods'] and y in r['methods']]
        if rs:
            d = lambda p: float(np.mean([float(r['methods'][x]['success']) - float(r['methods'][y]['success']) for r in p]))
            fs['paired'][nm] = {'n': len(rs), 'success_diff': d(rs), 'ci95': bootm(rs, d)}
    res['feasible_subset'] = fs
    (ROOT / 'docs/v5g/RESULTS.json').write_text(json.dumps(res, indent=1, ensure_ascii=False))
    G = M['gt_outcome_by_candidate']; nfeas = sum(1 for r in M['samples'] if r['oracle'])
    L = ['# V5g：数字孪生规划抓取位置与夹持力（两侧失败：掉落 vs 损伤）', f"自动生成 `scripts/v5g/report.py`，{res['created_utc']}；预注册 `docs/v5g/PREREG_2026-11-16.md`。", '',
         f"- 任务：V5b 物体放在支撑床上，平行夹爪在 {{端部, 中部}} × 6 个夹持深度（局部厚度的 3–40%）中选一个抓取，抬起 30% 高度并保持（g=2），放下后松开。"
         f"掉落 = 保持期间超过一半的粒子没被带起（left>{M['thresholds']['T_DROP_left']}）；损伤 = 松开后夹持段永久宽度变化 >{100*M['thresholds']['T_DMG_crush']:.0f}%；其余为成功。"
         "孪生体中仿真全部 12 个候选，选预测成功中夹持最轻的；都不行则放弃。用 GT 参数执行。",
         f"- n = {n} 个样本（四个孪生体都完成）；GT 下至少有一个可行抓取的样本 {nfeas}/{n}。", '',
         '| 方法 | n | 成功率 [95% CI] | 掉落率 | 损伤率 | 错过（可行却放弃） | 夹持级别遗憾 | 相对夹持力遗憾 (n) | 律正确率 |', '|---|---|---|---|---|---|---|---|---|']
    for m in ('gt', 'prior_vt', 'prior_vision', 'lawmean_pred', 'fixed_loo'):
        if m not in S:
            continue
        v = S[m]; fr = v['force_regret_rel']
        L.append(f"| {ZH[m]} | {v['n']} | {pct(v['success'])}{ci(v['success_ci95'])} | {pct(v['drop'])} | {pct(v['damage'])} | {pct(v['missed'])} | "
                 f"{v['level_regret']:.2f} | {'–' if fr is None else f'{100 * fr:+.0f}%'} ({v['n_force_regret']}) | {pct(v['law_accuracy'])} |")
    L += ['', '## 配对对比（同一批样本，物体自助 95% CI）', '', '| 对比 | n (物体) | 成功率差 | 掉落率差 | 损伤率差 | 级别遗憾差 |', '|---|---|---|---|---|---|']
    for k, v in M['paired'].items():
        L.append(f"| {k} | {v['n']} ({v['n_objects']}) | {100 * v['success_diff']:+.1f} pp{ci([x * 100 if x is not None else None for x in v['success_diff_ci95']], False).replace('%', '')} | "
                 f"{100 * v['drop_diff']:+.1f} pp | {100 * v['damage_diff']:+.1f} pp | {v['level_regret_diff']:+.2f}{ci(v['level_regret_diff_ci95'], False)} |")
    L += ['', '## GT 结果按候选（成功 / 掉落 / 损伤）', '', ' · '.join(f"{c}: {v['success']}/{v['drop']}/{v['damage']}" for c, v in G.items()), '',
          'GT oracle 选择分布：' + '，'.join(f"{k}: {v}" for k, v in M['oracle_histogram'].items() if v), '',
          '固定选择（全局，非留一）成功率：' + ' · '.join(f"{k}: {pct(v)}" for k, v in M['fixed_global_success'].items()), '',
          f"## 至少有一个 GT 可行抓取的子集（n={fs['n']}/{n}，按 GT 结果选的子集，仅描述）", '',
          '| 方法 | n | 成功率 [CI] | 掉落 | 损伤 | 错过 |', '|---|---|---|---|---|---|'] + [
          f"| {ZH[m]} | {v['n']} | {pct(v['success'])}{ci(v['success_ci95'])} | {pct(v['drop'])} | {pct(v['damage'])} | {pct(v['missed'])} |" for m, v in fs['arms'].items()] + [
          ''] + [f"- 配对 {k}: n={v['n']}，成功率差 {100*v['success_diff']:+.1f} pp [{'–' if v['ci95'][0] is None else f'{100*v[chr(99)+chr(105)+chr(57)+chr(53)][0]:+.1f}'}, {'–' if v['ci95'][1] is None else f'{100*v[chr(99)+chr(105)+chr(57)+chr(53)][1]:+.1f}'}]" for k, v in fs['paired'].items()] + [
          '', '## 敏感性（成功率）', '', '| 设置 | ' + ' | '.join(ZH[m] for m in S) + ' |', '|---|' + '---|' * len(S)]
    for k, d in res['sensitivity'].items():
        L.append(f'| {k} | ' + ' | '.join(pct(d[m]['success']) if m in d else '–' for m in S) + ' |')
    (ROOT / 'docs/v5g/RESULTS_ZH.md').write_text('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    main()
