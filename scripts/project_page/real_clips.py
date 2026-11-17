"""Project-page clips for the Spring-Gaus real drops (V3v): held-out camera C0787, 20 frames, 4 tiles.

Tiles: observed real video | prior (0 sims) | prior + refine E with dissipation (LOO confirm arm) | class-mean E.
Rendering = real_eval.splat with real_eval.colorize (training cams of the C0787 fold), i.e. the evaluation's projection and
colour code. Trajectories come from data/v3v/sims/<obj>/*.npy (re-simulated by real_eval.traj2 only if missing).
Run: scripts/v3f/replay_env.sh scripts/project_page/real_clips.py
Out: docs/project_assets/real/<obj>.mp4/.jpg and real.json
"""
import json, shutil, subprocess, sys, tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v3v'))
import real_eval as re_  # noqa: E402  (sets sys.path for numpy/cv2/torch/PIL)
import numpy as np  # noqa: E402
import cv2  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

CAM = 'C0787'; TILE = 256; STRIP = 28; FPS_OUT = 8; LOOPS = 2
OUT = ROOT / 'docs/project_assets/real'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
ARMS = [('prior', 'Our prior (0 sims)'), ('prior_refine_damp_loo', 'Prior + refine E, dissip. (LOO)'),
        ('class_mean', 'Class-mean E (uninformed)')]


def font(sz):
    return ImageFont.truetype(FONT, sz)


def strip(labels):
    """labels: list of (line1, line2) per tile -> RGB array STRIP x (TILE*n)."""
    im = Image.new('RGB', (TILE * len(labels), STRIP), (255, 255, 255)); dr = ImageDraw.Draw(im)
    f1, f2 = font(11), font(10)
    for i, (a, b) in enumerate(labels):
        x = i * TILE + TILE // 2
        dr.text((x, 1), a, fill=(20, 20, 20), font=f1, anchor='mt')
        dr.text((x, 15), b, fill=(90, 90, 90), font=f2, anchor='mt')
    return np.asarray(im)


def contour(tile, m):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(tile, cs, -1, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    R = json.loads((ROOT / 'docs/v3v/RESULTS.json').read_text()); RC = json.loads((ROOT / 'docs/v3v/RESULTS_confirm.json').read_text())
    summary = {'camera': CAM, 'frames': 20, 'fps': FPS_OUT, 'loops': LOOPS,
               'iou_metric': 'future_frames IoU (silhouette IoU, frames 10-19, all 3 cameras) from data/v3v/eval/<obj>.json, fold C0787; '
                             'iou_fold_mean = the per-object value in RESULTS.json / RESULTS_confirm.json (mean over 3 held-out folds)',
               'objects': {}}
    for obj in re_.OBJS:
        o = re_.Obs(obj); S = re_.Sims(obj, o); ev = json.loads((re_.D / 'eval' / f'{obj}.json').read_text())
        F = ev['folds'][CAM]; tc = F['train_cams']; col, _ = re_.colorize(o, o.X0, tc)
        rend, rec = {}, {}
        for arm, _ in ARMS:
            e = F['arms'][arm]; r = e.get('rpic') or 0.0
            tr = re_.traj2(S, e['lg'], e['nu'], r)
            rend[arm] = [re_.splat(o, CAM, tr[k], col)[:2] for k in range(o.n)]
            fm = (RC['summary']['prior_refine_damp_loo']['future_frames']['iou']['per_object'][obj] if arm == 'prior_refine_damp_loo'
                  else R['per_object'][obj][arm]['future_frames']['iou'])
            rec[arm] = {'log10_E_over_rho_x1000': e['lg'], 'nu': e['nu'], 'rpic_dissipation': r,
                        'iou_future_fold_C0787': e['future_frames']['iou'], 'iou_future_fold_mean': fm,
                        'iou_heldout_cam_C0787_frames1_19': e['heldout_camera']['iou']}
        # identical square crop for all tiles: union of observed + rendered masks over all frames
        ms = [o.msk[CAM][k] for k in range(o.n)] + [m for arm in rend for (m, _) in rend[arm]]
        u = np.any(np.stack(ms), 0); ys, xs = np.nonzero(u)
        cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
        half = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 0.55) + 8
        pad = half + 2

        def crop(a):
            p = np.pad(a, ((pad, pad), (pad, pad)) + ((0, 0),) * (a.ndim - 2))
            return p[cy - half + pad:cy + half + pad, cx - half + pad:cx + half + pad]

        def rs(a, nearest=False):
            return cv2.resize(a, (TILE, TILE), interpolation=cv2.INTER_NEAREST if nearest else cv2.INTER_AREA)
        # observed tile: real frame; pad with the frame's edge colour instead of black
        labels = [('Observed (held-out cam)', '')] + [(name, f'future IoU {rec[a]["iou_future_fold_C0787"]:.2f}') for a, name in ARMS]
        tmp = Path(tempfile.mkdtemp(prefix=f'real_{obj}_', dir=ROOT / 'tmp')) if (ROOT / 'tmp').exists() else Path(tempfile.mkdtemp(prefix=f'real_{obj}_', dir=OUT))
        frames = []
        for k in range(o.n):
            img = o.img[CAM][k]
            p = np.pad(img, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
            obs = rs(p[cy - half + pad:cy + half + pad, cx - half + pad:cx + half + pad])
            om = rs(crop(o.msk[CAM][k]).astype(np.uint8), True)
            tiles = [obs]
            for arm, _ in ARMS:
                m, im = rend[arm][k]
                tiles.append(contour(np.ascontiguousarray(rs(crop(im))), om))
            phase = 'fit window' if k < 10 else 'future'
            lab = list(labels); lab[0] = ('Observed (held-out cam)', f'frame {k:02d}/19 - {phase}')
            frames.append(np.vstack([strip(lab), np.hstack(tiles)]))
        seq = frames * LOOPS
        for i, fr in enumerate(seq):
            Image.fromarray(fr).save(tmp / f'{i:04d}.png')
        mp4 = OUT / f'{obj}.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS_OUT), '-i', str(tmp / '%04d.png'), '-an',
                        '-c:v', 'libx264', '-preset', 'slow', '-crf', '20', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(mp4)], check=True)
        Image.fromarray(seq[int(0.4 * len(seq))]).save(OUT / f'{obj}.jpg', quality=90)
        shutil.rmtree(tmp)
        summary['objects'][obj] = {'hit_frame': o.hit, 'crop_roi_px': [int(cx - half), int(cy - half), int(2 * half)], 'arms': rec,
                                   'mp4_bytes': mp4.stat().st_size}
        print(obj, {a: round(rec[a]['iou_future_fold_C0787'], 3) for a in rec}, mp4.stat().st_size, flush=True)
    (OUT / 'real.json').write_text(json.dumps(summary, indent=1, default=float))


if __name__ == '__main__':
    main()
