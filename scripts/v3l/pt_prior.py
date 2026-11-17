"""V3l Part 2: our zero-shot prior (9-run V2-main ensemble, K0) applied to PhysTwin's RGB views (train frames 0-46 only).

  .venv/bin/python -I scripts/v3l/pt_prior.py <rgb_case_dir> <out.json>

Input (untrusted, only PNG decoded): color/<cam>/<f>.png (848x480) and mask/<cam>/<obj_id>/<f>.png for cams 0,1,2.
The object mask id is read from mask/mask_info_<cam>.json (the non-'hand' entry).  Each view: object pixels on black
(hand and background removed), square crop around the union object bbox over frames 0-46 (+15% margin), resized to
224 (bicubic); 47 frames -> 64 by nearest repetition (as v3f), times f/30 s.  Same model code path as v3f/prior20.
Domain gap is large (real RGB-D capture, hand push, partial visibility, no ground in the image after masking).
"""
import argparse, io, json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/v3f')]
import springgaus_fewshot as F  # noqa: E402


def main(a):
    from PIL import Image
    from fieldbridge.data import ParameterCodec
    codec = ParameterCodec(json.loads(F.STATS.read_text()))
    d = Path(a.case); T = a.n_frames
    per_view, info = [], {}
    for cam in (0, 1, 2):
        mi = json.loads((d / 'mask' / f'mask_info_{cam}.json').read_text())
        obj_ids = [k for k, v in mi.items() if 'hand' not in str(v).lower()]
        assert len(obj_ids) == 1, mi
        oid = obj_ids[0]
        rgbs, masks = [], []
        for f in range(T):
            rgbs.append(np.asarray(Image.open(d / 'color' / str(cam) / f'{f}.png').convert('RGB'), np.float32) / 255)
            masks.append(np.asarray(Image.open(d / 'mask' / str(cam) / oid / f'{f}.png').convert('L')) > 127)
        u = np.any(np.stack(masks), 0); ys, xs = np.where(u)
        cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
        half = 0.5 * 1.15 * max(ys.max() - ys.min(), xs.max() - xs.min())
        H, W = u.shape
        y0, y1 = int(max(0, cy - half)), int(min(H, cy + half)); x0, x1 = int(max(0, cx - half)), int(min(W, cx + half))
        fs = []
        for im, m in zip(rgbs, masks):
            c = (im * m[..., None])[y0:y1, x0:x1]
            s = max(c.shape[:2]); pad = np.zeros((s, s, 3), np.float32)
            oy, ox = (s - c.shape[0]) // 2, (s - c.shape[1]) // 2; pad[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
            fs.append(np.asarray(Image.fromarray((pad * 255).round().astype(np.uint8)).resize((224, 224), Image.BICUBIC), np.float32) / 255)
        per_view.append(fs)
        info[cam] = {'obj_mask_id': oid, 'mask_info': mi, 'crop': [y0, y1, x0, x1], 'mask_area_frac_f0': float(masks[0].mean())}
        Image.fromarray((np.concatenate([fs[0], fs[T // 2], fs[-1]], 1) * 255).astype(np.uint8)).save(Path(a.out).with_name(f'pt_prior_input_cam{cam}.png'))
    t0 = time.time()
    vj = F.vjepa_model()
    rgb = F.frames_to_tensor(per_view, T)
    idx = np.round(np.linspace(0, T - 1, 64)).astype(int)
    times = torch.from_numpy((np.arange(T, dtype=np.float32) / 30.0)[idx])
    feats = F.encode(vj, rgb); del vj; torch.cuda.empty_cache()
    sample = F.make_sample('phystwin/single_push_sloth/t0', 'sloth', 'PhysTwin', [0, 1, 2], feats, times, torch.zeros(4), torch.zeros(4, dtype=torch.bool))
    mus, laws = [], []
    for r in F.RUNS:
        model, _ = F.load_model(r, 'cuda')
        mu, logits, h = F.run(model, [sample], 'cuda'); mus.append(mu[0]); laws.append(int(logits[0].argmax())); del model; torch.cuda.empty_cache()
    Z = torch.stack(mus).mean(0); dec = codec.decode(Z[None])[0]
    out = {'n_frames': T, 'views': [0, 1, 2], 'z': Z.tolist(), 'log10_E_Pa_at_rho1000': float(np.log10(float(dec[0]))), 'nu': float(dec[1]),
           'per_run_log10E': [float(np.log10(float(codec.decode(m[None])[0, 0]))) for m in mus],
           'law_votes': [F.LAW_NAMES[k] for k in laws], 'seconds_model': time.time() - t0, 'input': info}
    Path(a.out).write_text(json.dumps(out, indent=1, default=str)); print(json.dumps({k: v for k, v in out.items() if k != 'input'}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('case'); p.add_argument('out'); p.add_argument('--n-frames', type=int, default=47)
    main(p.parse_args())
