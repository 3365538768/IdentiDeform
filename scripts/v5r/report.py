"""V5r report: probe-then-grasp on V5b val -> docs/v5r/GRID_FINAL.json (step 'grid'), docs/v5r/RESULTS.json + RESULTS_ZH.md.

  .venv/bin/python scripts/v5r/report.py grid     # GT-only final-grid rule (docs/v5r/PREREG_2026-11-16.md)
  .venv/bin/python scripts/v5r/report.py report [--allow-partial]
Uses the unchanged V3e evaluator (scripts/v3e/plan_eval.py) with FB_GRIP_DATA=data/v5b_main/val, FB_GRIP_DEPTHS=<final grid>.
"""
import json
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FULL = [0.06, 0.10, 0.14, 0.18, 0.24, 0.32, 0.40, 0.48, 0.56, 0.64]
V2GRID = FULL[:6]
METHODS = ['gt', 'prior_vt', 'prior_vision', 'prior_vis3', 'lawmean_pred', 'fixed_loo', 'refined_vt']
ZH = {'gt': 'GT 参数（oracle）', 'prior_vt': '(c) 视觉 + 触觉先验', 'prior_vision': '(b) 视觉先验（9-run）', 'prior_vis3': '补充：视觉先验（v5m 3-seed，同族）',
      'lawmean_pred': '(a) 类均值（视觉预测的律）', 'fixed_loo': '(d) 固定深度（留一）', 'refined_vt': '(e) 视觉+触觉 + 12 次修正'}
SENS = {'sens_abscrush': ['--abs'], 'sens_dentp95': ['--metric', 'dent_p95'], 'sens_dmin010': ['--dmin', '0.10'],
        'sens_dmin014': ['--dmin', '0.14'], 'sens_tau002': ['--tau', '0.02'], 'sens_tau005': ['--tau', '0.05'],
        'sens_wloss_unaligned': ['--metric', 'wloss']}


def run(out, depths, methods, extra=()):
    env = dict(os.environ, FB_GRIP_DATA='data/v5b_main/val', FB_GRIP_TWINS='v5', FB_GRIP_DEPTHS=','.join(f'{d:.2f}' for d in depths),
               FB_GRIP_METHODS=','.join(methods), FB_GRIP_PAIRED='prior_vision,lawmean_pred,fixed_loo,prior_vis3')
    r = subprocess.run([sys.executable, 'scripts/v3e/plan_eval.py', '--runs', os.environ.get('V5R_RUNS', 'data/v5r/runs'), '--samples', 'docs/v5r/SAMPLES.json',
                        '--out', out, *extra], cwd=ROOT, env=env, capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(r.stderr)
    return json.loads((ROOT / out).read_text()), r.stdout


def grid():
    d, log = run('docs/v5r/GT_FULLGRID.json', FULL, ['gt', 'fixed_loo'])
    fails = Counter(); n = len(d['samples'])
    for p in sorted((ROOT / 'data/v5r/runs').glob('*/gt/state.json')):
        for k, v in json.loads(p.read_text())['runs'].items():
            if v.get('failed'):
                fails[k] += 1
    hist = Counter(r['oracle_d'] for r in d['samples'])
    excluded = [float(k) for k, c in fails.items() if float(k) > 0 and c >= 0.1 * n]
    cand = [x for x in (0.40, 0.48, 0.56, 0.64) if x not in excluded]
    chosen = None
    for top in cand:
        g = [x for x in FULL if x <= top and x not in excluded]
        dd, _ = run('.cache/v5r_grid_probe.json', g, ['gt'])
        at_top = sum(r['oracle_d'] == top for r in dd['samples'])
        if at_top <= n / 3:
            chosen = (top, g, at_top); break
    if chosen is None:
        top = cand[-1]; g = [x for x in FULL if x <= top and x not in excluded]
        dd, _ = run('.cache/v5r_grid_probe.json', g, ['gt']); chosen = (top, g, sum(r['oracle_d'] == top for r in dd['samples']))
    out = {'rule': 'docs/v5r/PREREG_2026-11-16.md (= docs/v3r/PREREG_GRID_2026-11-15.md rule on V5b GT)', 'n_samples': n,
           'gt_failures_per_depth': dict(fails), 'excluded_depths': excluded,
           'oracle_histogram_full_grid': {str(k): v for k, v in sorted(hist.items(), key=lambda kv: (kv[0] is None, kv[0] or 0))},
           'D_top': chosen[0], 'grid': chosen[1], 'n_at_top_final': chosen[2], 'excluded_gt_samples': d['excluded_samples'],
           'written_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    (ROOT / 'docs/v5r/GRID_FINAL.json').write_text(json.dumps(out, indent=1)); print(json.dumps(out, indent=1))


def pct(v):
    return '–' if v is None else f'{100 * v:.1f}%'


def report(allow_partial):
    G = json.loads((ROOT / os.environ.get('V5R_GRIDFILE', 'docs/v5r/GRID_FINAL.json')).read_text()); grid_ = G['grid']
    have = [m for m in METHODS if m in ('gt', 'fixed_loo') or any((ROOT / 'data/v5r/runs').glob(f'*/{m}/state.json'))]
    main_, log = run(os.environ.get('V5R_PREFIX', 'docs/v5r/') + 'GRIPPER_TASK.json', grid_, have)
    if main_['n_complete'] < main_['n_samples'] and not allow_partial:
        raise SystemExit(f"incomplete: {main_['n_complete']}/{main_['n_samples']}")
    v2grid, _ = run(os.environ.get('V5R_PREFIX', 'docs/v5r/') + 'GRIPPER_TASK_v2grid.json', V2GRID, have)
    wave1, _ = run(os.environ.get('V5R_PREFIX', 'docs/v5r/') + 'GRIPPER_TASK_wave1.json', grid_, have, ['--only', 'wave1'])
    sens = {k: run(os.environ.get('V5R_PREFIX', 'docs/v5r/') + f'GRIPPER_TASK_{k}.json', grid_, have, v)[0] for k, v in SENS.items()}
    S = main_['summary']; groups = ['all', 'plastic_laws', 'plasticine', 'metal', 'jelly']
    tab = {m: {'label': S[m]['label'], 'n': S[m]['n'], **{g: S[m][g] for g in groups if g in S[m]}} for m in have if 'all' in S.get(m, {})}
    acc = {}
    for m in have:
        rows = [r for r in main_['samples'] if m in r['methods'] and 'twin_law' in r['methods'][m]]
        if m in ('gt', 'fixed_loo') or not rows:
            continue
        import math
        acc[m] = {'law_accuracy': sum(r['methods'][m]['twin_law'] == r['law'] for r in rows) / len(rows),
                  'mean_abs_dlog10_E_over_rho': sum(abs(math.log10(r['methods'][m]['twin_params']['E']) - math.log10(
                      r['methods']['gt']['twin_params']['E'] / (r['methods']['gt']['twin_params']['density'] / 1000.))) for r in rows) / len(rows)}
    P = main_['paired_success_differences']
    key = {'(c)-(b)': 'prior_vt-minus-prior_vision', '(b)-(a)': 'prior_vision-minus-lawmean_pred', '(c)-(a)': 'prior_vt-minus-lawmean_pred',
           '(c)-(d)': 'prior_vt-minus-fixed_loo', '(b)-(d)': 'prior_vision-minus-fixed_loo', '(c)-vis3 (touch, same family)': 'prior_vt-minus-prior_vis3',
           '(e)-(b)': 'refined_vt-minus-prior_vision'}
    contrasts = {k: P[v] for k, v in key.items() if v in P}
    import numpy as np
    top = max(grid_); rows_all = main_['samples']; below = [r for r in rows_all if (r['oracle_d'] or 0) < top - 1e-9]
    def bootm(rs, f, n=2000, seed=0):
        rng = np.random.default_rng(seed); objs = sorted({r['object'] for r in rs}); by = {o: [r for r in rs if r['object'] == o] for o in objs}
        v = [f([r for o in rng.choice(objs, len(objs)) for r in by[o]]) for _ in range(n)]
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    sub = {'definition': f'samples whose GT oracle depth < top of grid ({top}); here twins can differ from a fixed depth',
           'n': len(below), 'fraction': len(below) / max(len(rows_all), 1), 'methods': {}, 'paired': {}}
    for m in have:
        rs = [r for r in below if m in r['methods']]
        if rs:
            g = lambda k: (lambda p: float(np.mean([float(x['methods'][m][k] or 0) for x in p])))
            sub['methods'][m] = {'n': len(rs), **{k: g(k)(rs) for k in ('success', 'regret', 'damaged', 'chosen_d', 'utility')},
                                 'success_ci95': bootm(rs, g('success')), 'regret_ci95': bootm(rs, g('regret'))}
    for name, (x, y) in {'(c)-(b)': ('prior_vt', 'prior_vision'), '(b)-(a)': ('prior_vision', 'lawmean_pred'),
                         '(c)-vis3': ('prior_vt', 'prior_vis3'), '(c)-(d)': ('prior_vt', 'fixed_loo'), '(b)-(d)': ('prior_vision', 'fixed_loo')}.items():
        for lab, rs0 in (('all', rows_all), ('below_ceiling', below)):
            rs = [r for r in rs0 if x in r['methods'] and y in r['methods']]
            if not rs:
                continue
            d = {k: (lambda p, k=k: float(np.mean([float(r['methods'][x][k]) - float(r['methods'][y][k]) for r in p]))) for k in ('success', 'regret', 'utility')}
            sub['paired'].setdefault(name, {})[lab] = {'n': len(rs), 'n_objects': len({r['object'] for r in rs}),
                                                       **{f'{k}_diff': d[k](rs) for k in d}, **{f'{k}_ci95': bootm(rs, d[k]) for k in d}}
    res = {'stage': 'v5r', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'prereg': 'docs/v5r/PREREG_2026-11-16.md',
           'grid': G, 'protocol': main_['protocol'], 'n_samples': main_['n_samples'], 'n_complete': main_['n_complete'],
           'excluded_samples': main_['excluded_samples'], 'table': tab, 'primary_contrasts': contrasts, 'paired_all': P,
           'twin_quality': acc, 'below_ceiling': sub, 'oracle_histogram': {str(k): sum(1 for r in rows_all if r['oracle_d'] == k) for k in [None] + grid_}, 'fixed_depth_success_global': main_['fixed_depth_success_global'],
           'v2_grid_reference': {m: v2grid['summary'][m].get('all') for m in have if 'all' in v2grid['summary'].get(m, {})},
           'wave1': {m: wave1['summary'][m].get('all') for m in have if 'all' in wave1['summary'].get(m, {})},
           'sensitivity': {k: {m: {'success': d['summary'][m]['all']['success'], 'regret_mean': d['summary'][m]['all']['regret_mean'],
                                   'utility_mean': d['summary'][m]['all']['utility_mean']} for m in have if 'all' in d['summary'].get(m, {})} for k, d in sens.items()},
           'F6': "docs/v5r/GRIPPER_TASK.json has the V3e schema (summary.<m>.all.{success, success_ci95_object_bootstrap, regret_mean, regret_ci95_object_bootstrap}, n_samples)",
           'evaluator_stdout': log}
    (ROOT / (os.environ.get('V5R_PREFIX', 'docs/v5r/') + 'RESULTS.json')).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    L = ['# V5r：先探后抓（V5b，损伤感知平行夹爪）', f"自动生成 `scripts/v5r/report.py`，{res['created_utc']}。预注册：`docs/v5r/PREREG_2026-11-16.md`；网格：`docs/v5r/GRID_FINAL.json`。", '',
         f"- 每个 V5b 样本是一次探测交互（视频 + 接触力）。由此建孪生体，在孪生体里规划夹深（预测压溃 ≤3% 的最大深度，否则放弃），用 GT 参数执行。挤压任务与 V3e 相同（g=0，对齐压溃，τ=3%，d_min=0.06，粒子钳制）。",
         f"- 深度网格（按预注册规则由 GT 选定）：{G['grid']}（D_top={G['D_top']}，GT oracle 落在 D_top 的样本 {G['n_at_top_final']}/{G['n_samples']}）；GT 失败/深度：{G['gt_failures_per_depth'] or '无'}。",
         f"- n = {res['n_samples']} 可评估（完整 {res['n_complete']}），排除 {len(res['excluded_samples'])}。", '',
         '| 方法 | 成功率 [95% CI] | 遗憾 [CI] | 安全夹深 (oracle) [CI] | 平均所选深度 | 损坏率 | 漏抓率 | 弹塑性 | 果冻 | 律正确率 | \\|Δlog10 E/ρ\\| |', '|---|---|---|---|---|---|---|---|---|---|---|']
    for m, t in tab.items():
        a = t['all']; c = a['success_ci95_object_bootstrap']; r = a['regret_ci95_object_bootstrap']; u = a['utility_ci95_object_bootstrap']
        q = acc.get(m, {})
        L.append(f"| {ZH[m]} | {pct(a['success'])} [{pct(c[0])}, {pct(c[1])}] | {a['regret_mean']:.3f} [{r[0]:.3f}, {r[1]:.3f}] | "
                 f"{a['utility_mean']:.3f} ({a['oracle_utility_mean']:.3f}) [{u[0]:.3f}, {u[1]:.3f}] | {a['mean_chosen_d']:.3f} | {pct(a['damage_rate'])} | {pct(a['missed_grasp_rate'])} | "
                 f"{pct(t.get('plastic_laws', {}).get('success'))} | {pct(t.get('jelly', {}).get('success'))} | {pct(q.get('law_accuracy'))} | "
                 f"{'–' if not q else format(q['mean_abs_dlog10_E_over_rho'], '.2f')} |")
    L += ['', '## 主对比（配对，同一批样本，物体自助 95% CI）', '', '| 对比 | 成功率差 | 遗憾差 | 安全夹深差 | n |', '|---|---|---|---|---|']
    for k, v in contrasts.items():
        c = v['success_ci95_object_bootstrap']; g = v['regret_ci95_object_bootstrap']; u = v['utility_ci95_object_bootstrap']
        L.append(f"| {k} | {100 * v['success_diff']:+.1f} pp [{100 * c[0]:+.1f}, {100 * c[1]:+.1f}] | {v['regret_diff']:+.3f} [{g[0]:+.3f}, {g[1]:+.3f}] | "
                 f"{v['utility_diff']:+.3f} [{u[0]:+.3f}, {u[1]:+.3f}] | {v['n']} |")
    L += ['', f"## GT 深度接近天花板：oracle 深度 < {top} 的子集（{sub['n']}/{len(rows_all)} = {pct(sub['fraction'])}）", '',
          'oracle 深度分布：' + '，'.join(f"{k}: {v}" for k, v in res['oracle_histogram'].items() if v), '',
          '| 方法 | n | 成功率 [CI] | 遗憾 [CI] | 损坏率 | 平均所选深度 |', '|---|---|---|---|---|---|']
    for m, v in sub['methods'].items():
        L.append(f"| {ZH[m]} | {v['n']} | {pct(v['success'])} [{pct(v['success_ci95'][0])}, {pct(v['success_ci95'][1])}] | {v['regret']:.3f} [{v['regret_ci95'][0]:.3f}, {v['regret_ci95'][1]:.3f}] | {pct(v['damaged'])} | {v['chosen_d']:.3f} |")
    L += ['', '配对对比（全部 / 天花板以下子集；物体自助 95% CI）：', '', '| 对比 | 子集 | n (物体) | 成功率差 [CI] | 遗憾差 [CI] |', '|---|---|---|---|---|']
    for k, dd in sub['paired'].items():
        for lab, v in dd.items():
            L.append(f"| {k} | {lab} | {v['n']} ({v['n_objects']}) | {100*v['success_diff']:+.1f} pp [{100*v['success_ci95'][0]:+.1f}, {100*v['success_ci95'][1]:+.1f}] | {v['regret_diff']:+.3f} [{v['regret_ci95'][0]:+.3f}, {v['regret_ci95'][1]:+.3f}] |")
    L += ['', '## 其他设置（成功率 / 遗憾）', '', '| 设置 | ' + ' | '.join(ZH[m] for m in tab) + ' |', '|---|' + '---|' * len(tab)]
    rows = {'V2 网格（≤0.32）': {m: v for m, v in res['v2_grid_reference'].items()}, 'wave 1（26）': res['wave1'], **res['sensitivity']}
    for k, d in rows.items():
        L.append(f'| {k} | ' + ' | '.join((f"{pct(d[m]['success'])} / {d[m]['regret_mean']:.3f}" if m in d and d[m] else '–') for m in tab) + ' |')
    L += ['', '## 固定深度（全局）', '', ' · '.join(f"{k}: {pct(v)}" for k, v in main_['fixed_depth_success_global'].items())]
    (ROOT / (os.environ.get('V5R_PREFIX', 'docs/v5r/') + 'RESULTS_ZH.md')).write_text('\n'.join(L) + '\n'); print('\n'.join(L))


if __name__ == '__main__':
    {'grid': grid, 'report': lambda: report('--allow-partial' in sys.argv)}[sys.argv[1]]()
