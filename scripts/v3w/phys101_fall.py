"""V3w: Physics 101 fall scenario -> zero-shot stiffness ordering with the V2 main 9-run ensemble (protocol docs/v3w/PROTOCOL.json).
Videos are untrusted: only decoded with cv2.VideoCapture (mp4), nothing executed. Run with python -I.
  build   <fall_root> <out_dir>          background per surface, masks, release, crops, V-JEPA features (motion + static first frame)
  predict <out_dir> <pred.json>          ensemble predictions per trial
"""
import argparse, json, sys, random
from pathlib import Path
import numpy as np
import cv2
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v3f')); sys.path.insert(0, str(ROOT / 'src'))
import springgaus_fewshot as sf  # noqa: E402
import os
CAM = os.environ.get('V3W_CAM', 'Camera_1.mp4')


def read(p):
    cap = cv2.VideoCapture(str(p)); fps = cap.get(cv2.CAP_PROP_FPS); fr = []
    while True:
        ok, f = cap.read()
        if not ok: break
        fr.append(f[..., ::-1].copy())
    cap.release(); return fr, fps


def raw_fg(f, bg):
    d = np.abs(cv2.GaussianBlur(f, (5, 5), 0).astype(np.int16) - bg.astype(np.int16)).max(-1)
    return d > 30


def components(raw, persistent):
    """-> list of (mask, centroid, area) object candidates (not touching / within 25 px of top-border hand components)."""
    m = (raw & ~persistent).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
    top = [k for k in range(1, n) if st[k][1] <= 1]  # touches top border: hand / arm
    hand = cv2.dilate(np.isin(lab, top).astype(np.uint8), np.ones((51, 51), np.uint8)) > 0 if top else np.zeros(m.shape, bool)
    near = set(np.unique(lab[hand]).tolist())
    return [(lab == k, cen[k], int(st[k][4])) for k in range(1, n) if k not in top and k not in near and st[k][4] >= 150]


def track(cands, H):
    """Backward tracking from the resting object in the last frame (deviation 3, logged)."""
    T = len(cands); masks = [None] * T
    last = [c for c in cands[-1] if c[1][1] > H * 0.4]
    if not last: return masks
    cur = max(last, key=lambda c: c[2]); masks[-1] = cur[0]
    for t in range(T - 2, -1, -1):
        near = [c for c in cands[t] if np.linalg.norm(c[1] - cur[1]) < 200 and 0.33 < c[2] / cur[2] < 3]
        if not near: break
        cur = min(near, key=lambda c: np.linalg.norm(c[1] - cur[1])); masks[t] = cur[0]
    return masks


def build(a):
    from fieldbridge.train import atomic_save
    root = Path(a.root); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    trials = sorted(p.parent for p in root.glob(f'*/*/*/{CAM}'))
    by_surf = {}
    for t in trials: by_surf.setdefault(t.parent.name, []).append(t)
    rng = random.Random(0); bgs = {}
    for s, ts in by_surf.items():
        if (out / f'bg_{s}.npy').exists():
            bgs[s] = np.load(out / f'bg_{s}.npy'); continue
        smp = []
        for t in rng.sample(ts, min(40, len(ts))):
            fr, _ = read(t / CAM); smp += [cv2.GaussianBlur(fr[k], (5, 5), 0) for k in range(0, len(fr), 4)]
        bgs[s] = np.median(np.stack(smp), 0).astype(np.uint8)
        cv2.imwrite(str(out / f'bg_{s}.jpg'), bgs[s][..., ::-1]); np.save(out / f'bg_{s}.npy', bgs[s])
        print('bg', s, len(ts), len(smp), flush=True)
    import os, re
    if os.environ.get('V3W_ONLY'): trials = [t for t in trials if re.search(os.environ['V3W_ONLY'], str(t))]
    model = sf.vjepa_model(); index = {}
    for i, t in enumerate(trials):
        obj, surf, rep = t.parts[-3], t.parts[-2], t.parts[-1]; tid = f'{obj}/{surf}/{rep}'
        fr, fps = read(t / CAM); rec = {'object': obj, 'surface': surf, 'trial': rep, 'n_frames': len(fr), 'fps': fps}
        bg = bgs[surf]; H = bg.shape[0]
        d0 = np.abs(cv2.GaussianBlur(fr[0], (5, 5), 0).astype(np.int16) - bg.astype(np.int16)).mean(-1)[H // 4:]
        if np.median(d0) > 15:
            bg = np.median(np.stack([cv2.GaussianBlur(f, (5, 5), 0) for f in fr]), 0).astype(np.uint8); rec['bg_fallback'] = True
        raws = [raw_fg(f, bg) for f in fr]
        persistent = cv2.dilate((np.mean(raws, 0) > 0.95).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0  # static misregistration (dev. logged)
        masks = track([components(r, persistent) for r in raws], H)
        rel = next((k for k in range(len(fr)) if masks[k] is not None), None)
        if rel is None: rec['excluded'] = 'no_release'; index[tid] = rec; continue
        clip = list(range(rel, len(fr))); present = [masks[k] is not None for k in clip]
        rec.update(release=rel, clip_len=len(clip), present_frac=float(np.mean(present)))
        if len(clip) < 8: rec['excluded'] = 'short_clip'; index[tid] = rec; continue
        if np.mean(present) < 0.8: rec['excluded'] = 'mask_missing'; index[tid] = rec; continue
        u = np.any(np.stack([masks[k] for k in clip if masks[k] is not None]), 0); ys, xs = np.nonzero(u)
        cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
        side = max(96, int(1.35 * max(xs.max() - xs.min(), ys.max() - ys.min()))); Hh, W = u.shape; side = min(side, Hh)
        x0 = int(np.clip(cx - side / 2, 0, W - side)); y0 = int(np.clip(cy - side / 2, 0, Hh - side))
        fs, hs = [], []
        for k in clip:
            m = masks[k] if masks[k] is not None else np.zeros_like(u)
            x = (fr[k] * m[..., None]).astype(np.uint8)[y0:y0 + side, x0:x0 + side]
            fs.append(cv2.resize(x, (224, 224), interpolation=cv2.INTER_CUBIC).astype(np.float32) / 255)
            if masks[k] is not None:
                yy = np.nonzero(m.any(1))[0]; hs.append(float(yy.max() - yy.min() + 1))
            else: hs.append(np.nan)
        hs = np.array(hs); h0 = np.nanmedian(hs[:3])
        rec['squash'] = float(np.clip(np.nanmax(1 - hs / h0), 0, None)); rec['crop_xy_side'] = [x0, y0, side]
        rec['mask_area_px_first'] = int(masks[rel].sum())
        n = len(fs); idx = np.round(np.linspace(0, n - 1, 64)).astype(int)
        times = torch.from_numpy((idx / fps).astype(np.float32))
        rgb = sf.frames_to_tensor([fs], n); stat = sf.frames_to_tensor([[fs[0]] * n], n)
        if i % 25 == 0 or os.environ.get('V3W_ONLY'):
            cv2.imwrite(str(out / f'prev_{obj}_{surf}_{rep}.jpg'), (rgb[0, ::8].permute(2, 0, 3, 1).reshape(224, -1, 3).numpy()[..., ::-1] * 255).astype(np.uint8))
        t_, m_ = sf.encode_target(None)
        for kind, x in (('motion', rgb), ('static', stat)):
            sid = f'phys101_fall/{tid}/{kind}'; name = sid.replace('/', '_') + '.pt'
            atomic_save(sf.make_sample(sid, obj, 'phys101-fall', [CAM[:-4]], sf.encode(model, x), times, t_, m_), out / name)
            rec[f'file_{kind}'] = name
        index[tid] = rec
        if i % 20 == 0:
            print(i, len(trials), tid, 'rel', rel, 'clip', len(clip), flush=True)
            (out / 'index.json').write_text(json.dumps(index, indent=1))
    (out / 'index.json').write_text(json.dumps(index, indent=1)); print('done', len(index))


def predict(a):
    from fieldbridge.data import ParameterCodec
    codec = ParameterCodec(json.loads(sf.STATS.read_text()))
    d = Path(a.out); index = json.loads((d / 'index.json').read_text())
    keys = [(tid, kind) for tid, r in sorted(index.items()) if 'excluded' not in r and 'file_motion' in r for kind in ('motion', 'static')]
    samples = [torch.load(d / index[t][f'file_{k}'], weights_only=True) for t, k in keys]
    Z, P = [], []
    for r in sf.RUNS:
        model, _ = sf.load_model(r, 'cuda'); mu, logits, _ = sf.run(model, samples, 'cuda')
        Z.append(mu); P.append(logits.softmax(-1)); del model; torch.cuda.empty_cache(); print(r, flush=True)
    Z = torch.stack(Z); P = torch.stack(P); res = {}
    for i, (t, k) in enumerate(keys):
        z = Z[:, i].mean(0); dec = codec.decode(z[None])[0]
        res.setdefault(t, {})[k] = {'z_mean': z.tolist(), 'z_sd_runs': Z[:, i].std(0).tolist(),
            'log10_E_over_rho_x1000': float(np.log10(dec[0])), 'rho_decoded': float(dec[2]),
            'law_prob_mean': dict(zip(sf.LAW_NAMES, P[:, i].mean(0).tolist()))}
    Path(a.pred).write_text(json.dumps(res, indent=1)); print('pred', len(res))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='stage', required=True)
    b = sub.add_parser('build'); b.add_argument('root'); b.add_argument('out')
    e = sub.add_parser('predict'); e.add_argument('out'); e.add_argument('pred')
    a = p.parse_args(); {'build': build, 'predict': predict}[a.stage](a)
