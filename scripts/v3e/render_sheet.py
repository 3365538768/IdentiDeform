"""V3e qualitative figure: GT-parameter execution of the depths chosen by different planners, rendered with the
3DGS pipeline (one camera view), contact sheet t = 0 / closed 1.0 s / hold end 1.5 s / final 4.0 s.

  .venv/bin/python scripts/v3e/render_sheet.py --index 134 --rows "oracle (GT twin):0.16,ours (refined):0.16,prior (0 sims):0.32" --out docs/v3e/figs/x.png
Render directories are deleted after the frames are extracted (logged in docs/CLEANUP_2026-11-12.md); the
extracted frames are kept as a small npz next to the figure.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import squeeze_sim as S  # noqa: E402

ROOT = S.ROOT
TIMES = [0.0, 1.0, 1.5, 4.0]


def frames_for(index, d, view, work):
    tag = f'{d:.3f}'; dst = work / tag
    cache = work / f'{tag}_frames.npz'
    if cache.exists():
        return dict(np.load(cache))
    gt = json.loads((ROOT / f'data/v2_main/val/{index:06d}/gt.json').read_text())
    ok, sec = S.simulate(dst, S.gt_params(index), gt['object'], d, index, render=True)
    if not ok:
        raise RuntimeError(f'render sim failed {dst}')
    root = next(dst.glob('*/images'))
    views = sorted(p.name for p in root.iterdir() if p.is_dir())
    out = {}
    for v in views:
        pngs = sorted((root / v).glob('*.png'))
        idx = [min(len(pngs) - 1, int(round(t * (len(pngs) - 1) / 4.0))) for t in TIMES]
        out[v] = np.stack([cv2.imread(str(pngs[i]))[:, :, ::-1] for i in idx])
    np.savez_compressed(cache, **out)
    n = sum(1 for _ in dst.rglob('*.png'))
    shutil.rmtree(dst, ignore_errors=True)
    with (ROOT / 'docs/CLEANUP_2026-11-12.md').open('a') as f:
        f.write(f'- {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())} v3e render_sheet: deleted {dst.relative_to(ROOT)} '
                f'({n} PNGs, frames kept in {cache.relative_to(ROOT)})\n')
    return out


def main():
    a = argparse.ArgumentParser(); a.add_argument('--index', type=int, required=True); a.add_argument('--rows', required=True)
    a.add_argument('--view', type=int, default=0); a.add_argument('--out', required=True); a.add_argument('--crop', type=float, default=0.0)
    a.add_argument('--dent', default='')  # optional "label=value" annotations, comma separated
    args = a.parse_args()
    work = ROOT / 'data/v3e/render' / f'{args.index:06d}'; work.mkdir(parents=True, exist_ok=True)
    rows = [(r.rsplit(':', 1)[0], float(r.rsplit(':', 1)[1])) for r in args.rows.split(',')]
    notes = dict(x.split('=', 1) for x in args.dent.split(';') if '=' in x)
    tiles = []
    allf = [frames_for(args.index, d, args.view, work) for _, d in rows]
    v = sorted(allf[0])[args.view]
    # common square crop around the object (union over all rows/frames of non-background pixels), 20 % margin
    m = np.concatenate([f[v] for f in allf]).max(-1).max(0) > 12
    ys, xs = np.where(m); cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = int(0.6 * max(ys.max() - ys.min(), xs.max() - xs.min())) + 4
    H, Wd = m.shape
    y0, x0 = int(max(0, cy - half)), int(max(0, cx - half)); y1, x1 = int(min(H, cy + half)), int(min(Wd, cx + half))
    for (label, d), fr in zip(rows, allf):
        imgs = fr[v][:, y0:y1, x0:x1]
        imgs = [cv2.resize(np.ascontiguousarray(i), (320, 320), interpolation=cv2.INTER_AREA) for i in imgs]
        row = np.concatenate(imgs, 1).copy()
        txt = f'{label}: d={d:.2f}' + (f'  {notes[label]}' if label in notes else '')
        cv2.putText(row, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(row, txt, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(row)
    head = np.full((28, tiles[0].shape[1], 3), 255, np.uint8)
    for k, t in enumerate(['t=0', 'closed (1.0 s)', 'hold end (1.5 s)', 'after release (4.0 s)']):
        cv2.putText(head, t, (k * 320 + 8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    sheet = np.concatenate([head] + tiles, 0)
    Path(ROOT / args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(ROOT / args.out), sheet[:, :, ::-1])
    print('wrote', args.out)


if __name__ == '__main__':
    main()
