"""V5g analysis on the PAIRED-COMPLETE samples (GT, vision+touch, vision, class mean all complete; fixed LOO from GT):
  .venv/bin/python scripts/v5g/analyze.py [--tag interim_0815]
Runs scripts/v5g/report.py with V5G_REQUIRE_ALL=1 (outputs docs/v5g/RESULTS*.json / RESULTS_ZH.md), copies them to
docs/v5g/interim/<tag>/ when --tag is given, and prints n and the per-arm summary."""
import argparse, json, os, shutil, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
a = argparse.ArgumentParser(); a.add_argument('--tag', default=''); args = a.parse_args()
r = subprocess.run([sys.executable, 'scripts/v5g/report.py'], cwd=ROOT, env=dict(os.environ, V5G_REQUIRE_ALL='1'), capture_output=True, text=True)
if r.returncode:
    raise SystemExit(r.stderr)
R = json.loads((ROOT / 'docs/v5g/RESULTS.json').read_text()); R['analysis_set'] = 'paired-complete samples (all 4 twins complete)'
q = subprocess.run([sys.executable, 'scripts/v5g/predict_eval.py', '--out', 'docs/v5g/PREDICTION.json'], cwd=ROOT, capture_output=True, text=True)
if q.returncode:
    raise SystemExit(q.stderr)
PR = json.loads((ROOT / 'docs/v5g/PREDICTION.json').read_text())
R['primary_prediction'] = {k: PR[k] for k in ('addendum', 'n_samples', 'n_objects', 'n_candidates', 'class_counts_gt', 'predictors', 'paired')}
R['secondary_planning_note'] = 'planning metrics as pre-registered; always-abstain wins on raw success because most samples have no GT-feasible grasp'
ZHN = {'prior_vt': '(c) 视觉+触觉孪生体', 'prior_vision': '(b) 视觉孪生体', 'lawmean_pred': '(a) 类均值孪生体', 'base_majority': '基线 (i) 总体多数类',
       'base_cand': '基线 (ii) 按候选的多数类（留一物体）', 'base_cand_law': '基线 (iii) 按候选 × 视觉预测律的多数类'}
f = lambda v: '–' if v is None else f'{v:.3f}'
c = lambda x: '' if not x or x[0] is None else f' [{x[0]:.2f}, {x[1]:.2f}]'
L = ['## 主指标（预注册补充 docs/v5g/PREREG_ADDENDUM.md）：逐候选抓取结果预测', '',
     f"n = {PR['n_samples']} 个配对完整样本（{PR['n_objects']} 个物体）× {PR['n_candidates']} 个候选；GT 类别计数 {PR['class_counts_gt']}。", '',
     '| 预测器 | 宏 F1 [CI] | 准确率 [CI] | 召回 成功/掉落/损伤 | 安全抓取精确率 [CI] | 安全抓取召回 [CI] | 可行性准确率 [CI] |', '|---|---|---|---|---|---|---|']
for m, e in PR['predictors'].items():
    r = e['recall']; ci_ = e['ci95']
    L.append(f"| {ZHN[m]} | {f(e['macro_f1'])}{c(ci_['macro_f1'])} | {f(e['accuracy'])}{c(ci_['accuracy'])} | {f(r['success'])}/{f(r['drop'])}/{f(r['damage'])} | "
             f"{f(e['safe_precision'])}{c(ci_['safe_precision'])} | {f(e['safe_recall'])}{c(ci_['safe_recall'])} | {f(e['feasibility_accuracy'])}{c(ci_['feasibility_accuracy'])} |")
L += ['', '配对对比（同一批 bootstrap 抽样）：', '', '| 对比 | 宏 F1 差 [CI] | 准确率差 [CI] | 安全精确率差 [CI] | 可行性准确率差 [CI] |', '|---|---|---|---|---|']
for k, v in PR['paired'].items():
    L.append(f"| {k} | " + ' | '.join(f"{'–' if v[x]['diff'] is None else format(v[x]['diff'], '+.3f')}{c(v[x]['ci95'])}" for x in ('macro_f1', 'accuracy', 'safe_precision', 'feasibility_accuracy')) + ' |')
L += ['', '## 次要指标：预注册的规划指标（原样报告；多数样本没有 GT 可行抓取，因此“一律放弃”在原始成功率上最高）', '']
zh = (ROOT / 'docs/v5g/RESULTS_ZH.md').read_text().split('\n')
(ROOT / 'docs/v5g/RESULTS_ZH.md').write_text('\n'.join(zh[:3] + L + zh[3:]))
(ROOT / 'docs/v5g/RESULTS.json').write_text(json.dumps(R, indent=1, ensure_ascii=False))
print('\n'.join(L))
if args.tag:
    d = ROOT / 'docs/v5g/interim' / args.tag; d.mkdir(parents=True, exist_ok=True)
    for f in (ROOT / 'docs/v5g').glob('RESULTS*'):
        shutil.copy2(f, d / f.name)
print('n paired-complete =', R['n_samples'])
for m, v in R['summary'].items():
    print(f"{m:13s} n={v['n']} success={v['success']:.3f} {v['success_ci95']} drop={v['drop']:.3f} damage={v['damage']:.3f} missed={v['missed']:.3f} "
          f"force_regret={v['force_regret_rel']} (n={v['n_force_regret']})")
