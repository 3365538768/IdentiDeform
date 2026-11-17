"""Contact-force curves for review pages: per sample, |F| of each rigid body (grouped by role) and the press plate's
vertical force over the 4 s clip. Usage: force_plots.py <root> <split> <idx,...> <outdir> <tag>"""
import sys, json, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
root, split, idxs, out, tag = sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3].split(',')], sys.argv[4], sys.argv[5]
os.makedirs(out, exist_ok=True)
for i in idxs:
    f = f'{root}/{split}/{i:06d}/contact_forces.npz'
    if not os.path.exists(f): continue
    z = np.load(f); F = z['F']; t = z['t']; B = json.loads(str(z['bodies'])); Fp = z['F_plate']
    fig, ax = plt.subplots(figsize=(4.2, 1.6), dpi=110)
    mags = np.linalg.norm(F, axis=2) if F.size else np.zeros((len(t), 0))
    if mags.shape[1]:
        ax.plot(t[:len(mags)], mags.sum(1), lw=1.0, color='tab:blue', label='grippers / plates, total |F|')
    if len(Fp):
        ax.plot(t[:len(Fp)], -Fp, lw=1.2, color='k', label='plate (down)')
    fl = f'{root}/{split}/{i:06d}/press_force_log.json'
    if os.path.exists(fl):  # force-controlled press: measured reaction R and the target force
        d = json.load(open(fl)); L = np.array(d['log_t_z_R'] or [[0, 0, 0]], float)
        ax.plot(L[:, 0], L[:, 2], lw=1.2, color='k', marker='.', ms=2, label='plate reaction R')
        ax.axhline(d['force'], color='tab:red', lw=0.8, ls='--', label=f"target F = {d['force']:g}")
    ax.set_xlim(0, 4); ax.set_xlabel('t [s]', fontsize=7); ax.set_ylabel('|F|', fontsize=7); ax.tick_params(labelsize=6)
    ax.legend(fontsize=5, ncol=4, frameon=False, loc='upper right'); ax.grid(alpha=0.25)
    fig.tight_layout(pad=0.2); fig.savefig(f'{out}/{tag}_{split}_{i:06d}.png'); plt.close(fig)
print('done')
