"""V3p: docs/v3p/RESULTS_raw.json -> docs/v3p/RESULTS.json (headline tables + notes) and docs/v3p/RESULTS_ZH.md.
  .venv/bin/python scripts/v3p/write_results.py [--notes docs/v3p/notes.json]
"""
import argparse, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
KEYS = [('z_mae', 'z-MAE(4)'), ('ident_z_mae', '可辨识 z-MAE'), ('law_accuracy', '类别准确率'),
        ('within_law_pearson_E_over_rho', 'r_P(E/ρ)'), ('within_law_spearman_E_over_rho', 'r_S(E/ρ)'),
        ('within_law_pearson_sigma_y_over_rho', 'r_P(σ/ρ)'), ('within_law_spearman_sigma_y_over_rho', 'r_S(σ/ρ)'),
        ('decade_mae_E_over_rho', 'E/ρ 误差(dec)'), ('decade_mae_sigma_y_over_rho', 'σ/ρ 误差(dec)')]


def fmt(x, ci=True):
    if x is None or x.get('value') is None:
        return '–'
    s = f"{x['value']:.3f}"
    if ci and x.get('ci95'):
        s += f" [{x['ci95'][0]:.3f}, {x['ci95'][1]:.3f}]"
    return s


def main():
    a = argparse.ArgumentParser(); a.add_argument('--raw', default='docs/v3p/RESULTS_raw.json'); a.add_argument('--notes', default='docs/v3p/notes.json')
    a.add_argument('--out', default='docs/v3p/RESULTS.json'); a.add_argument('--zh', default='docs/v3p/RESULTS_ZH.md'); args = a.parse_args()
    raw = json.loads((ROOT / args.raw).read_text()); notes = json.loads((ROOT / args.notes).read_text()) if (ROOT / args.notes).exists() else {}
    table = []
    for ds in [k for k in ('V2', 'V3', 'V3_4k') if k in raw]:
        for split, blk in raw[ds]['splits'].items():
            for method, m in blk.items():
                table.append({'dataset': ds, 'split': split, 'method': method, 'samples': m['samples'], 'objects': m['objects'],
                              **{k: {'value': m[k]['value'], 'ci95': m[k]['ci95'], **({'sd_over_seeds': m[k]['sd_over_members']} if 'sd_over_members' in m[k] else {})} for k, _ in KEYS},
                              'z_mae_per_parameter': {p: m[f'z_mae_{p}']['value'] for p in ('E_over_rho', 'nu', 'rho', 'sigma_y_over_rho')}})
    out = {'stage': 'v3p', 'written_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'protocol': raw['metrics_doc'],
           'train_stats': {ds: {'stats': raw[ds]['stats'], 'train_count': raw[ds]['train_count'], 'train_std_decades': raw[ds]['train_std_decades']} for ds in raw if ds.startswith('V')},
           'table': table, **notes}
    (ROOT / args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')
    L = ['# V3p：V2 前馈先验管线在 V3 数据上的复现与对比', '', f'生成时间 {out["written_utc"]}；源数据 `docs/v3p/RESULTS.json`（明细 `RESULTS_raw.json`）。', '']
    for sec in notes.get('zh_sections', []):
        L += [sec, '']
    for split in ('val', 'test'):
        L += [f'## {split}（方括号为物体级 bootstrap 95% CI，2000 次，seed 17；3 种子组为种子平均）', '',
              '| 数据 | 方法 | ' + ' | '.join(n for _, n in KEYS) + ' |', '|' + '---|' * (len(KEYS) + 2)]
        for r in table:
            if r['split'] == split:
                L.append(f"| {r['dataset']} | {r['method']} | " + ' | '.join(fmt(r[k], k in ('z_mae', 'ident_z_mae', 'law_accuracy', 'within_law_pearson_E_over_rho', 'within_law_pearson_sigma_y_over_rho', 'decade_mae_E_over_rho', 'decade_mae_sigma_y_over_rho')) for k, _ in KEYS) + ' |')
        L.append('')
    L += ['## 训练集标准化尺度（z 单位不同）', '', '| 数据 | train 样本 | std(ln E/ρ) 折合 decade | std(ln σ/ρ) 折合 decade |', '|---|---|---|---|']
    for ds, s in out['train_stats'].items():
        L.append(f"| {ds} | {s['train_count'][0]} | {s['train_std_decades']['E_over_rho']:.3f} | {s['train_std_decades']['sigma_y_over_rho']:.3f} |")
    (ROOT / args.zh).write_text('\n'.join(L) + '\n'); print('wrote', args.out, args.zh, len(table))


if __name__ == '__main__':
    main()
