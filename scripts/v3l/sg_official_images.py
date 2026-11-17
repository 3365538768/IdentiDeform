"""V3l: image metrics of the OFFICIAL Spring-Gaus runs (v3g) on eval camera 5 with the same PSNR/SSIM/IoU code as sg_metric.py.

  scripts/v3f/replay_env.sh scripts/v3l/sg_official_images.py <mpm_synthetic.zip> <out.json>

fit    : evaluations/images_fit/5_kk.png  (left half = Spring-Gaus render of frame k, right half = its GT image)
future : evaluations/images_pred/5_jj.png (60 Hz rollout from frame 0, image j at t = j/60 s) at j = round(1.8 k) for GT frame k
         in 20..29 (|dt| <= 1/120 s); 'future_exact' = k in {20, 25} where 1.8 k is an integer.
GT     : render/<scene>/camera_005/<k>.png RGB*alpha on black (Spring-Gaus loader).  CPU only.  Also checks that our GT
         composite equals the right half of images_fit (PSNR of the two GT versions).
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v3l'))
import sg_metric as M  # noqa: E402
import zipfile  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

EXP = {s: json.loads((ROOT / f'docs/v3g/logs/sg_eval_{s}.json').read_text())['exp'] for s in ('torus', 'duck', 'apple')}


def load(p):
    return np.asarray(Image.open(p).convert('RGB'), np.float32) / 255


def main(a):
    z = zipfile.ZipFile(a.zip); out = {}
    for s, exp in EXP.items():
        M.SCENE = s
        ev = ROOT / exp / 'evaluations'
        cams = json.loads(z.read(f'mpm_synthetic/render/{s}/camera.json'))
        gt = {k: M.gt_image(z, cams, M.EVAL_CAM, k) for k in range(30)}
        per, chk = {}, []
        for k in range(20):
            im = load(ev / 'images_fit' / f'5_{k:02d}.png'); W = im.shape[1] // 2
            per[k] = M.img_metrics(im[:, :W], *gt[k]); per[k]['source'] = f'images_fit/5_{k:02d}.png'
            chk.append(10 * np.log10(1 / max(((im[:, W:] - gt[k][0]) ** 2).mean(), 1e-12)))
        for k in range(20, 30):
            j = int(round(1.8 * k))
            im = load(ev / 'images_pred' / f'5_{j:02d}.png')
            per[k] = M.img_metrics(im, *gt[k]); per[k].update(source=f'images_pred/5_{j:02d}.png', time_offset_s=j / 60 - 0.03 * k)
        summ = {}
        for name, ks in (('fit', range(20)), ('future', range(20, 30)), ('future_exact', (20, 25))):
            for key in ('psnr', 'ssim', 'mask_iou'):
                summ[f'{name}_{key}_mean'] = float(np.mean([per[k][key] for k in ks]))
        out[s] = {'exp': exp, 'summary': summ, 'per_frame': per,
                  'gt_check_psnr_ours_vs_their_gt_half_min': float(np.min(chk))}
        print(s, json.dumps(summ), 'gt-check min PSNR', round(float(np.min(chk)), 2), flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('zip'); p.add_argument('out'); main(p.parse_args())
