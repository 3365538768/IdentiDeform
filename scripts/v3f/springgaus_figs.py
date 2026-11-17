"""V3f: figures for the Spring-Gaus check (matplotlib, .engine-envs/plot, run with -I; data read as images/JSON only).
  syn   <SPRINGGAUS_SYN.json> <out.png>
  real  <real_capture.zip> <prepared_real_dir> <SPRINGGAUS_REAL_PRED.json> <replay_dir|-> <out_prefix>
"""
import io, json, sys, zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

BLUE, ORANGE, AQUA, GRAY, INK, MUTED = '#2a78d6', '#eb6834', '#1baf7a', '#9a9893', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.size': 9, 'axes.edgecolor': GRAY, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': 'white'})
STATS_MEAN, STATS_STD = 8.983360716557394, 2.4582887452126303  # v2main E/rho z-stats (ln1p)
CAMS = ['C0733', 'C0787', 'C0801']
P = 'real_capture/dynamic/'


def z2lg(z):
    return np.log10(np.expm1(np.asarray(z) * STATS_STD + STATS_MEAN))


def syn(rep_path, out):
    r = json.loads(Path(rep_path).read_text()); e = r['ensemble9']; gt = r['protocol']['gt']
    names = r['protocol']['scored_scenes']
    gx = np.log10([gt[s]['E'] for s in names])
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, mode, title in zip(axs, ['K0', 'ridge_last_layer', 'support_mean_baseline'],
                               ['zero-shot (K=0)', 'LOSO ridge on last layer', 'LOSO support mean (no video)']):
        z = np.array([e[mode]['z_t0'][s][0] for s in names]); sd = np.array([e[mode]['z_sd_over_triplets'][s][0] for s in names])
        y = z2lg(z); yerr = np.abs(z2lg(z + sd) - y)
        ax.axhspan(2.0, 4.98, color=BLUE, alpha=0.08, lw=0)
        ax.text(6.12, 4.93, 'V2 train jelly range', ha='right', va='top', color=MUTED, fontsize=8)
        ax.plot([4.9, 6.1], [4.9, 6.1], color=GRAY, lw=1, ls='--')
        jit = (np.arange(len(names)) % 3 - 1) * 0.02
        ax.errorbar(gx + jit, y, yerr=yerr if mode != 'support_mean_baseline' else None, fmt='o', ms=5, color=ORANGE if mode == 'K0' else BLUE,
                    ecolor=GRAY, elinewidth=1, capsize=0, mec='white', mew=1)
        s = e[mode]['canonical_t0']
        ax.set_title(f'{title}\nSpearman {s["E_over_rho_spearman"]:+.2f}, MAE_z(E) {s["mae_z_E"]:.2f}', fontsize=9, color=INK)
        ax.set_xlabel('GT log10 E/ρ·1000 (ρ=1000 assumed)')
        ax.set_xlim(4.85, 6.15); ax.set_ylim(3.5, 6.2); ax.grid(axis='y', color='#e6e5e1', lw=0.6)
    axs[0].set_ylabel('predicted log10 E/ρ·1000')
    fig.suptitle('Spring-Gaus synthetic (14 elastic MPM scenes), V2 9-run ensemble; bars = sd over 3 camera triplets', fontsize=9.5, color=INK)
    fig.tight_layout(); fig.savefig(out, dpi=170); print('wrote', out)


def rodrigues(r):
    r = np.asarray(r, float).reshape(3); th = np.linalg.norm(r)
    if th < 1e-12:
        return np.eye(3)
    k = r / th; Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * Kx + (1 - np.cos(th)) * Kx @ Kx


def project(X, K, d, rv, tv):
    Xc = X @ rodrigues(rv).T + np.asarray(tv).reshape(3)
    x, y = Xc[:, 0] / Xc[:, 2], Xc[:, 1] / Xc[:, 2]
    k1, k2, p1, p2, k3 = np.asarray(d).reshape(-1)[:5]; r2 = x * x + y * y
    rad = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    xd = x * rad + 2 * p1 * x * y + p2 * (r2 + 2 * x * x); yd = y * rad + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return np.stack([K[0, 0] * xd + K[0, 2], K[1, 1] * yd + K[1, 2]], 1)


def real(zip_path, prep, pred_path, replay_dir, prefix):
    z = zipfile.ZipFile(zip_path); idx = json.loads((Path(prep) / 'index.json').read_text())
    pred = json.loads(Path(pred_path).read_text())
    objs = pred['E_over_rho_ranking_soft_to_stiff']
    cal = json.loads(z.read(P + 'cameras_calib.json')); K = np.array(cal['camera_matrix']); d = np.array(cal['distortion_coefficient'])
    rep = json.loads((Path(replay_dir) / 'replay.json').read_text()) if replay_dir != '-' and (Path(replay_dir) / 'replay.json').exists() else {}
    cam = 'C0733'
    # Figure 1: inputs + predicted stiffness / law
    fig = plt.figure(figsize=(11, 1.65 * len(objs) + 0.6))
    gs = fig.add_gridspec(len(objs), 6, width_ratios=[1, 1, 1, 1, 0.45, 2.2], wspace=0.05, hspace=0.25)
    ax_bar = fig.add_subplot(gs[:, 5])
    for i, obj in enumerate(objs):
        o = idx['objects'][obj]; hit = o['hit_frame']; x0, y0, side = o['crop_xy_side'][cam]
        for j, k in enumerate([0, hit, min(hit + 5, 19), 19]):
            f = o['frames'][cam][k]
            im = Image.open(io.BytesIO(z.read(f'{P}videos_images/{cam}/{f}'))).convert('RGB').crop((x0, y0, x0 + side, y0 + side)).resize((200, 200))
            ax = fig.add_subplot(gs[i, j]); ax.imshow(im); ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if i == 0:
                ax.set_title(['frame 0', f'hit ({hit})', f'frame {min(hit + 5, 19)}', 'frame 19'][j], fontsize=8, color=MUTED)
            if j == 0:
                ax.set_ylabel(obj, fontsize=10, color=INK)
    ys = np.arange(len(objs))[::-1]
    for y, obj in zip(ys, objs):
        p = pred['predictions'][f'springgaus_real/{obj}/seq20']
        runs = z2lg(p['per_run_z_E'])
        ax_bar.scatter(runs, np.full(len(runs), y) + (np.arange(9) - 4) * 0.03, s=10, color=GRAY, zorder=2)
        ax_bar.scatter([p['log10_E_over_rho_x1000']], [y], s=60, color=ORANGE, zorder=3, edgecolor='white', lw=1)
        votes = p['law_votes']
        ax_bar.text(5.05, y, f"{max(votes, key=votes.get)} {max(votes.values())}/9", va='center', fontsize=8, color=INK)
        if obj in rep:
            b = rep[obj].get('refined_best_grid')
            if b:
                ax_bar.scatter([rep[obj]['candidates'][b]['log10_E_over_rho_x1000']], [y], marker='D', s=36, color=BLUE, zorder=3, edgecolor='white', lw=1)
    ax_bar.set_yticks(ys); ax_bar.set_yticklabels(objs); ax_bar.set_xlim(2.9, 5.6)
    ax_bar.set_xlabel('log10 E/ρ·1000  (orange: 9-run mean, gray: runs' + (', blue: MPM-refined)' if rep else ')'))
    ax_bar.axvspan(2.0, 4.98, color=BLUE, alpha=0.06, lw=0); ax_bar.grid(axis='x', color='#e6e5e1', lw=0.6)
    ax_bar.set_title('prior: law (majority) and stiffness, soft → stiff', fontsize=9, color=INK)
    fig.suptitle('Spring-Gaus real drops (camera C0733 crops; prior input = 3 views × 20 frames, masked)', fontsize=10, color=INK)
    fig.savefig(prefix + '_prior.png', dpi=150, bbox_inches='tight'); print('wrote', prefix + '_prior.png')
    if not rep:
        return
    # Figure 2: replay overlays + up trajectories
    shown = [o for o in objs if o in rep]
    fig = plt.figure(figsize=(12, 2.1 * len(shown) + 0.6))
    gs = fig.add_gridspec(len(shown), 6, width_ratios=[1, 1, 1, 1, 0.45, 1.9], wspace=0.05, hspace=0.5)
    for i, obj in enumerate(shown):
        o = idx['objects'][obj]; R = rep[obj]; hit = o['hit_frame']; x0, y0, side = o['crop_xy_side'][cam]
        best = R['refined_best_grid']
        tr = {t: np.load(Path(replay_dir) / f'{obj}_{t}_traj.npy') for t in ['predicted', best] if (Path(replay_dir) / f'{obj}_{t}_traj.npy').exists()}
        for j, k in enumerate([0, hit, min(hit + 5, 19), 19]):
            f = o['frames'][cam][k]
            im = Image.open(io.BytesIO(z.read(f'{P}videos_images/{cam}/{f}'))).convert('RGB')
            ax = fig.add_subplot(gs[i, j]); ax.imshow(np.asarray(im)[y0:y0 + side, x0:x0 + side]); ax.set_xticks([]); ax.set_yticks([])
            for t, col in [('predicted', ORANGE), (best, BLUE)]:
                if t in tr:
                    uv = project(tr[t][k], K, d, cal[cam]['rvecs'], cal[cam]['tvecs']) - [x0, y0]
                    ax.scatter(uv[:, 0], uv[:, 1], s=0.3, color=col, alpha=0.25, lw=0)
            ax.set_xlim(0, side); ax.set_ylim(side, 0)
            if i == 0:
                ax.set_title(['frame 0', 'hit', '+5', 'frame 19'][j], fontsize=8, color=MUTED)
            if j == 0:
                ax.set_ylabel(obj, fontsize=10, color=INK)
        ax = fig.add_subplot(gs[i, 5])
        t = np.arange(len(R['observed_up_cm'])) / 119.88 * 1000
        g = R['observed_up_cm'][-1] * 0
        ax.plot(t, np.array(R['observed_up_cm']) - R['ground_up_m'] * 100, color=INK, lw=2, label='observed (triangulated)')
        for tag, col, lab in [('predicted', ORANGE, 'MPM, predicted E'), (best, BLUE, 'MPM, refined E')]:
            c = R['candidates'].get(tag, {})
            if 'sim_up_cm' in c:
                ts = np.arange(len(c['sim_up_cm'])) / 119.88 * 1000
                ax.plot(ts, np.array(c['sim_up_cm']) - R['ground_up_m'] * 100, color=col, lw=2,
                        label=f"{lab} (10^{c['log10_E_over_rho_x1000']:.2f}), e={((c.get('bounce_sim_20frames') or {}).get('restitution_e') or float('nan')):.2f}")
        ob = R['observed_bounce'] or {}
        ax.set_title(f"centroid height; observed e={ob.get('restitution_e', float('nan')):.2f} (e from first 20 frames)", fontsize=8.5, color=INK)
        ax.set_xlabel('ms'); ax.set_ylabel('cm above ground'); ax.legend(fontsize=6.5, frameon=False, loc='upper right')
        ax.grid(color='#e6e5e1', lw=0.6)
    fig.suptitle('MPM replay (our engine, 3-view visual-hull geometry) vs real video; particles projected into C0733', fontsize=10, color=INK)
    fig.savefig(prefix + '_replay.png', dpi=150, bbox_inches='tight'); print('wrote', prefix + '_replay.png')


if __name__ == '__main__':
    if sys.argv[1] == 'syn':
        syn(sys.argv[2], sys.argv[3])
    else:
        real(*sys.argv[2:7])
