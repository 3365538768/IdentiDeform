"""Review videos (one setup, three views per clip); was the V3 contact-setup review: for each index, old (v2_main) vs new (v3_main) clip, 3 views, plus fragment/fall metrics.
  .venv/bin/python scripts/v3n/review.py val 0,1,2 data/v3n_review"""
import sys, os, json, glob, subprocess, numpy as np, cv2
sys.path.insert(0, 'scripts/v3m'); from common import crop_box
split, idxs, out = sys.argv[1], [int(x) for x in sys.argv[2].split(',')], sys.argv[3]
ROOT3 = sys.argv[4] if len(sys.argv) > 4 else 'data/v5b_main'
TAG = sys.argv[5] if len(sys.argv) > 5 else 'V5b'
VARIANTS = {k: [(TAG, ROOT3)] for k in ('press', 'shear', 'bend', 'drop')}
VARIANTS['tap30hold'] = VARIANTS['press']; VARIANTS['stretch'] = VARIANTS['shear']
os.makedirs(out, exist_ok=True)


def pack(root, i):
    p = glob.glob(f'{root}/{split}/{i:06d}/*/sample_pack.npz')
    return np.load(p[0]) if p else None


def metrics(m):
    frag = 0.0
    for v in range(m.shape[0]):
        for t in range(0, m.shape[1], 4):
            n, _, st, _ = cv2.connectedComponentsWithStats(m[v, t].astype(np.uint8), connectivity=8)
            if n > 2:
                a = st[1:, cv2.CC_STAT_AREA]; frag = max(frag, 1 - a.max() / a.sum())
    H = m.shape[2]; ys = np.arange(H)[None, None, :, None]
    cy = (m * ys).sum((2, 3)) / np.maximum(m.sum((2, 3)), 1)
    h0 = np.array([max(np.ptp(np.nonzero(m[v, 0].any(1))[0]), 1) if m[v, 0].any() else 1 for v in range(m.shape[0])])
    return {'frag': round(float(frag), 4), 'drop_0_0.5s': round(float(np.median((cy[:, 16] - cy[:, 0]) / h0)), 3)}


meta = {}
for i in idxs:
    g = json.load(open(f'{ROOT3}/{split}/{i:06d}/gt.json'))
    row = {'action': g['action'], 'law': g['law'], 'object': g['object'], 'log10_E_over_rho': round(float(np.log10(g['labels']['E_over_rho_specific'])), 2)}
    if 'sigma_over_rho_specific' in g['labels']:
        row['log10_sigma_over_rho'] = round(float(np.log10(g['labels']['sigma_over_rho_specific'])), 2)
    rows = []
    for tag, root in VARIANTS[g['action']]:
        z = pack(root, i)
        if z is None:
            row[tag] = None; continue
        rgb, m = z['rgb'], z['object_mask'][:, :, 0] > 127
        row[tag] = metrics(m)
        tiles = []
        for v in range(3):
            y0, y1, x0, x1 = crop_box(m[v])
            fr = np.stack([cv2.resize(f.transpose(1, 2, 0)[y0:y1, x0:x1], (200, 200), interpolation=cv2.INTER_AREA) for f in rgb[v]])
            tiles += [fr, np.zeros((len(fr), 200, 4, 3), np.uint8)]
        r = np.concatenate(tiles[:-1], 2).copy()
        for f in r:
            cv2.putText(f, tag, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA); cv2.putText(f, tag, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        rows.append(r)
    if not rows:
        continue
    T = min(len(r) for r in rows)
    fr = rows[0][:T]
    H, W = fr.shape[1:3]; path = f'{out}/{split}_{i:06d}.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', '32', '-i', '-',
                    '-c:v', 'libx264', '-preset', 'medium', '-crf', '26', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', path],
                   input=np.ascontiguousarray(fr).tobytes(), check=True)
    meta[f'{split}_{i:06d}'] = row; print(i, row, flush=True)
json.dump(meta, open(f'{out}/review_{split}.json', 'w'), indent=1)
