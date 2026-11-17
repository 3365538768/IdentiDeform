"""V3l: per-frame Chamfer (sg_eval definition) on torus/duck/apple: Spring-Gaus official vs ours. -> docs/v3l/cd_per_frame.png
  .venv/bin/python -I scripts/v3l/make_figure.py
"""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SER = [('Spring-Gaus (official, 10 views)', '#2a78d6', '-'), ('Ours: prior only (frames 0-19)', '#eb6834', '-'),
       ('Ours: prior + 12 sims (frames 0-19)', '#1baf7a', '-'), ('Static (GT frame 0)', '#8a8984', '--')]


def main():
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), sharex=True)
    for ax, s in zip(axs, ('torus', 'duck', 'apple')):
        sg = json.loads((ROOT / f'docs/v3g/logs/sg_eval_{s}.json').read_text())['per_frame']
        sgc = {int(k): v['cd'] for part in ('fit', 'future') for k, v in sg[part].items()}
        ys = [sgc]
        for arm, tag in (('prior20', 'final'), ('prior20_refine', 'final')):
            p = ROOT / f'data/v3l/sg/{s}/{arm}.json'
            st = json.loads(p.read_text()) if p.exists() else {}
            r = st.get('rollouts', {}).get(tag, {})
            ys.append({int(k): v['cd'] for k, v in r.get('per_frame', {}).items()})
        sf = sg['future']
        ys.append({int(k): v['cd_static_baseline'] for k, v in sf.items()})
        for (lab, c, ls), y in zip(SER, ys):
            if y:
                k = sorted(y); ax.plot(k, [y[i] for i in k], color=c, ls=ls, lw=2, label=lab)
        ax.axvline(19.5, color='#c3c2b7', lw=1); ax.text(19.7, ax.get_ylim()[1] * 0.92, 'future →', fontsize=8, color='#52514e')
        ax.set_yscale('log'); ax.set_title(s, fontsize=10, color='#0b0b0b'); ax.set_xlabel('frame (dt 0.03 s)')
        ax.grid(alpha=0.25, lw=0.5); [sp.set_visible(False) for n, sp in ax.spines.items() if n in ('top', 'right')]
    axs[0].set_ylabel('Chamfer (scene units, log)')
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(ROOT / 'docs/v3l/cd_per_frame.png', dpi=150, facecolor='#fcfcfb')


if __name__ == '__main__':
    main()
