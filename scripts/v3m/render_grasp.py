"""V3m (project website): 3DGS render of the V3e squeeze executed with GT parameters at a given depth d
(same path as scripts/v3e/render_sheet.py, but all 128 frames of every view are kept for a video).
Output: data/v3m_site/grasp_gs/<idx>_<d>.npz (view_k: T,H,W,3 uint8). Render PNGs are deleted afterwards."""
import argparse, json, shutil, sys, time
from pathlib import Path
import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v3e'))
import squeeze_sim as S  # noqa: E402
ROOT = S.ROOT
a = argparse.ArgumentParser(); a.add_argument('--index', type=int, required=True); a.add_argument('--d', type=float, required=True)
args = a.parse_args()
out = ROOT / 'data/v3m_site/grasp_gs'; out.mkdir(parents=True, exist_ok=True)
dst = out / f'{args.index:06d}_{args.d:.3f}_work'
gt = json.loads((ROOT / f'data/v2_main/val/{args.index:06d}/gt.json').read_text())
ok, sec = S.simulate(dst, S.gt_params(args.index), gt['object'], args.d, args.index, render=True)
if not ok:
    sys.exit(1)
root = next(dst.glob('*/images'))
frames = {}
for v in sorted(p.name for p in root.iterdir() if p.is_dir()):
    pngs = sorted((root / v).glob('*.png'))
    frames[v] = np.stack([cv2.imread(str(p))[:, :, ::-1] for p in pngs])
np.savez_compressed(out / f'{args.index:06d}_{args.d:.3f}.npz', **frames)
n = sum(1 for _ in dst.rglob('*.png')); shutil.rmtree(dst, ignore_errors=True)
with (ROOT / 'docs/CLEANUP_2026-11-12.md').open('a') as f:
    f.write(f'- {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())} v3m render_grasp: deleted {dst.relative_to(ROOT)} ({n} PNGs, frames kept in npz)\n')
print('done', args.index, args.d, {k: v.shape for k, v in frames.items()}, round(sec, 1), flush=True)
