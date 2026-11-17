"""Website: particle animations of the V3e damage-aware squeeze. GT-parameter execution at the depth chosen by
the class-mean twin vs by our twin (refined_top2). Side view: squeeze axis horizontal, z vertical."""
import json, os, sys, subprocess
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
OUT = sys.argv[1]; IDX = [int(v) for v in sys.argv[2].split(',')]
os.makedirs(OUT, exist_ok=True)
J = json.load(open('docs/v3e/GRIPPER_TASK.json')); res = {r['index']: r for r in J['samples']}; tau = J['protocol']['tau']
BG = '#0d1014'; GREEN = '#3ecf8e'; RED = '#ff5d5d'; GREY = '#5b6470'
W_PX, H_PX, DPI = 960, 440, 100
meta = {}
for i in IDX:
    r = res[i]; key = json.load(open(f'data/v3e/runs/{i:06d}/gt/twin.json'))['cache']
    cache = f'data/v3e/runs/{i:06d}/cache/{key}'
    panels = []
    for m, lab in (('lawmean_pred', 'class-mean twin'), ('refined_top2', 'our twin (prior + refinement)')):
        d = r['methods'][m]['chosen_d']; ok = r['methods'][m]['success']
        z = np.load(f'{cache}/{d:.3f}/subset.npz'); bc = json.load(open(f'{cache}/{d:.3f}/squeeze_bc.json'))
        panels.append(dict(d=d, ok=ok, lab=lab, x=z['subset'].astype(np.float32), t=z['subset_t'] * 0.01, x0=z['x0'], bc=bc,
                           crush=r['gt_dent'][f'{d:.2f}']))
    meta[i] = {'law': r['law'], 'oracle_d': r['oracle_d'], 'panels': [{k: p[k] for k in ('d', 'ok', 'lab', 'crush')} for p in panels]}
    ax_a = panels[0]['bc']['axis']
    # common view box
    allx = np.concatenate([p['x'].reshape(-1, 3) for p in panels])
    lo = np.percentile(allx, 0.5, 0); hi = np.percentile(allx, 99.5, 0)
    cx, cz = (lo[ax_a] + hi[ax_a]) / 2, (lo[2] + hi[2]) / 2
    half = max(hi[ax_a] - lo[ax_a], hi[2] - lo[2]) * 0.62 + 0.06
    # interpolate 26 stored frames x3
    nf = len(panels[0]['t']); up = 3
    ts = np.linspace(0, nf - 1, (nf - 1) * up + 1)
    frames_dir = f'{OUT}/_frames_{i}'; os.makedirs(frames_dir, exist_ok=True)
    rng = np.random.default_rng(0); sel = rng.permutation(panels[0]['x'].shape[1])[:4096]
    for k, s in enumerate(ts):
        j0 = int(np.floor(s)); j1 = min(j0 + 1, nf - 1); w = s - j0
        fig, axes = plt.subplots(1, 2, figsize=(W_PX / DPI, H_PX / DPI), dpi=DPI, facecolor=BG)
        for ax, p in zip(axes, panels):
            x = (1 - w) * p['x'][j0] + w * p['x'][j1]; t = (1 - w) * p['t'][j0] + w * p['t'][j1]
            ax.set_facecolor(BG); ax.set_xlim(cx - half, cx + half); ax.set_ylim(cz - half * 0.85, cz + half * 0.85)
            ax.set_aspect('equal'); ax.axis('off')
            ax.scatter(p['x0'][sel, ax_a], p['x0'][sel, 2], s=1.2, c=GREY, alpha=0.35, lw=0)
            done = t >= 3.0
            col = (GREEN if p['ok'] else RED) if done else '#d9dee5'
            ax.scatter(x[sel, ax_a], x[sel, 2], s=1.3, c=col, lw=0)
            bc = p['bc']; Wd = bc['width']; d = p['d']
            if t <= bc['t_open']:
                s_in = min(t / bc['t_close'], 1.0) * d * Wd / 2
                zlo, zhi = bc['lo'][2] - 0.03, bc['hi'][2] + 0.03; th = 0.035
                for xf, sgn in ((bc['lo'][ax_a] + s_in, -1), (bc['hi'][ax_a] - s_in, +1)):
                    ax.add_patch(Rectangle((xf if sgn > 0 else xf - th, zlo), th, zhi - zlo, color='#8fa3ff', alpha=0.85, lw=0))
            phase = 'closing' if t < bc['t_close'] else 'holding' if t < bc['t_open'] else 'released'
            ax.text(0.02, 0.97, p['lab'], transform=ax.transAxes, color='white', fontsize=12, va='top', weight='bold')
            ax.text(0.02, 0.89, f'chosen depth d = {d:.2f}', transform=ax.transAxes, color='#c9d1db', fontsize=10.5, va='top')
            ax.text(0.98, 0.97, f't = {t:.1f} s  ({phase})', transform=ax.transAxes, color='#8b95a1', fontsize=9.5, ha='right', va='top')
            q0 = np.percentile(p['x0'][:, ax_a], [1, 99]); qt = np.percentile(x[:, ax_a], [1, 99])
            for v in q0:
                ax.axvline(v, color=GREY, lw=1.0, ls=(0, (4, 3)), alpha=0.9)
            for v in qt:
                ax.axvline(v, color=col, lw=1.2, alpha=0.9, ymax=0.12)
            if done:
                verdict = f"{'INTACT' if p['ok'] else 'CRUSHED'}: permanent crush {max(0.0, 100 * p['crush']):.1f}% of width (limit {100 * tau:.0f}%)"
                ax.text(0.02, 0.81, verdict, transform=ax.transAxes, color=(GREEN if p['ok'] else RED), fontsize=10.5, weight='bold')
        fig.subplots_adjust(0, 0, 1, 1, 0.02, 0)
        fig.savefig(f'{frames_dir}/{k:04d}.png', facecolor=BG); plt.close(fig)
    # hold last frame ~1.2 s
    last = len(ts) - 1
    for e in range(1, 18):
        os.link(f'{frames_dir}/{last:04d}.png', f'{frames_dir}/{last + e:04d}.png')
    path = f'{OUT}/grasp_{i:06d}.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', '15', '-i', f'{frames_dir}/%04d.png', '-c:v', 'libx264', '-preset', 'slow',
                    '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', path], check=True)
    subprocess.run(['cp', f'{frames_dir}/{last:04d}.png', f'{OUT}/grasp_{i:06d}.png'], check=True)
    subprocess.run(['rm', '-rf', frames_dir], check=True)
    print(path, os.path.getsize(path) // 1024, 'KB', flush=True)
json.dump(meta, open(f'{OUT}/grasp_meta.json', 'w'), indent=1)
