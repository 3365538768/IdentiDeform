"""V3f: Spring-Gaus real captures (burger, bun, dog, pig, potato) -> V-JEPA2 features + 3D centroid trajectories.

Input: real_capture.zip (untrusted; only JPG/PNG/JSON decoded, nothing executed). Run with python -I.
  build  <real_capture.zip> <out_dir>
  - sequences/<obj>/0.json: 20 synchronized frames per camera (C0733, C0787, C0801; 119.88 fps), hit_frame.
  - V-JEPA input 'seq20': provided frames x provided masks (object on black, == V2 background), fixed square
    crop per camera = union mask bbox over the 20 frames (side x1.35, min 160 px), 224 px, 20 -> 64 frames by
    nearest repetition, times in seconds.
  - trajectory: per frame and camera the mask centroid + lowest mask point, undistorted with the released
    calibration (cameras_calib.json: shared K/dist, per-camera rvec/tvec, checkerboard world frame), DLT-
    triangulated over the 3 cameras; reprojection error reported.
  predict <out_dir> <report.json>   9-run B/C/D ensemble K0 (no GT exists for these objects)
"""
import argparse, io, json, sys, zipfile
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import springgaus_fewshot as sf  # noqa: E402

CAMS = ['C0733', 'C0787', 'C0801']
OBJS = ['bun', 'burger', 'dog', 'pig', 'potato']
FPS = 119.88
P = 'real_capture/dynamic/'


def cam_mats(calib):
    import cv2
    K = np.array(calib['camera_matrix']); dist = np.array(calib['distortion_coefficient'])
    Ps = {}
    for c in CAMS:
        R, _ = cv2.Rodrigues(np.array(calib[c]['rvecs'], dtype=np.float64).reshape(3))
        t = np.array(calib[c]['tvecs'], dtype=np.float64).reshape(3, 1)
        Ps[c] = (R, t)
    return K, dist, Ps


def triangulate(pts, K, dist, Ps):
    import cv2
    A = []; norm = {}
    for c, uv in pts.items():
        n = cv2.undistortPoints(np.array(uv, np.float64).reshape(1, 1, 2), K, dist).reshape(2)
        R, t = Ps[c]; M = np.hstack([R, t]); norm[c] = n
        A.append(n[0] * M[2] - M[0]); A.append(n[1] * M[2] - M[1])
    X = np.linalg.svd(np.array(A))[2][-1]; X = X[:3] / X[3]
    err = []
    for c, uv in pts.items():
        R, t = Ps[c]
        proj, _ = cv2.projectPoints(X.reshape(1, 3), cv2.Rodrigues(R)[0], t, K, dist)
        err.append(float(np.linalg.norm(proj.reshape(2) - np.array(uv))))
    return X, err


def build(a):
    from PIL import Image
    from fieldbridge.train import atomic_save
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(a.zip)
    calib = json.loads(z.read(P + 'cameras_calib.json')); K, dist, Ps = cam_mats(calib)
    model = sf.vjepa_model()
    index = {'objects': {}, 'files': {}, 'fps': FPS}
    for obj in OBJS:
        seq = json.loads(z.read(f'{P}sequences/{obj}/0.json')); n = len(seq[CAMS[0]])
        imgs, masks = {}, {}
        for c in CAMS:
            imgs[c] = [np.asarray(Image.open(io.BytesIO(z.read(f'{P}videos_images/{c}/{f}'))).convert('RGB')) for f in seq[c]]
            masks[c] = [np.asarray(Image.open(io.BytesIO(z.read(f'{P}videos_masks/{c}/{f[:-4]}.png'))).convert('L')) > 127 for f in seq[c]]
        # trajectory
        traj, low, errs, crops = [], [], [], {}
        for k in range(n):
            cen = {}; bot = {}
            for c in CAMS:
                ys, xs = np.nonzero(masks[c][k])
                cen[c] = (xs.mean(), ys.mean())
            X, e = triangulate(cen, K, dist, Ps); traj.append(X.tolist()); errs.append(e)
        # per-camera mask area (pixels) and bbox height for squash proxy
        area = {c: [int(m.sum()) for m in masks[c]] for c in CAMS}
        frames_view = []
        for c in CAMS:
            u = np.any(np.stack(masks[c]), 0); ys, xs = np.nonzero(u)
            cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
            side = max(160, int(1.35 * max(xs.max() - xs.min(), ys.max() - ys.min())))
            H, W = u.shape
            x0 = int(np.clip(cx - side / 2, 0, W - side)); y0 = int(np.clip(cy - side / 2, 0, H - side))
            crops[c] = [x0, y0, side]
            fs = []
            for im, m in zip(imgs[c], masks[c]):
                x = (im * m[..., None]).astype(np.uint8)[y0:y0 + side, x0:x0 + side]
                fs.append(np.asarray(Image.fromarray(x).resize((224, 224), Image.BICUBIC), np.float32) / 255)
            frames_view.append(fs)
        rgb = sf.frames_to_tensor(frames_view, n)
        strip = (rgb[:, ::4].permute(0, 3, 1, 4, 2).reshape(3 * 224, -1, 3).numpy() * 255).astype(np.uint8)
        Image.fromarray(strip).save(out / f'preview_{obj}.png')
        idx = np.round(np.linspace(0, n - 1, 64)).astype(int)
        times = torch.from_numpy((idx / FPS).astype(np.float32))
        feats = sf.encode(model, rgb)
        sid = f'springgaus_real/{obj}/seq20'; name = sid.replace('/', '_') + '.pt'
        t, m = sf.encode_target(None)
        atomic_save(sf.make_sample(sid, obj, 'Spring-Gaus-real', CAMS, feats, times, t, m), out / name)
        index['files'][sid] = name
        index['objects'][obj] = {'hit_frame': seq['hit_frame'], 'frames': {c: seq[c] for c in CAMS}, 'crop_xy_side': crops,
                                 'centroid_xyz': traj, 'reproj_err_px': errs, 'mask_area_px': area}
        print(obj, 'reproj px mean', round(float(np.mean(errs)), 2), flush=True)
    (out / 'index.json').write_text(json.dumps(index, indent=1))


def predict(a):
    from fieldbridge.data import ParameterCodec
    codec = ParameterCodec(json.loads(sf.STATS.read_text()))
    d = Path(a.out); index = json.loads((d / 'index.json').read_text())
    sids = sorted(index['files']); samples = [torch.load(d / index['files'][s], weights_only=True) for s in sids]
    res = {}
    for r in sf.RUNS:
        model, _ = sf.load_model(r, 'cuda')
        mu, logits, _ = sf.run(model, samples, 'cuda')
        res[r] = (mu, logits.softmax(-1))
        del model; torch.cuda.empty_cache()
    Z = torch.stack([res[r][0] for r in sf.RUNS]); Pr = torch.stack([res[r][1] for r in sf.RUNS])
    out = {}
    for i, s in enumerate(sids):
        z = Z[:, i].mean(0); dec = codec.decode(z[None])[0]
        votes = np.bincount(Pr[:, i].argmax(-1).numpy(), minlength=3)
        out[s] = {'z_mean': z.tolist(), 'z_sd_over_runs': Z[:, i].std(0).tolist(),
                  'E_over_rho_x1000_decoded': float(dec[0]), 'log10_E_over_rho_x1000': float(np.log10(dec[0])),
                  'nu': float(dec[1]), 'sigma_y_over_rho_x1000_decoded': float(dec[3]),
                  'law_votes': dict(zip(sf.LAW_NAMES, votes.tolist())), 'law_majority': sf.LAW_NAMES[int(votes.argmax())],
                  'law_prob_mean': dict(zip(sf.LAW_NAMES, Pr[:, i].mean(0).tolist())),
                  'per_run_z_E': Z[:, i, 0].tolist()}
    order = sorted(out, key=lambda s: out[s]['z_mean'][0])
    rep = {'protocol': 'K0, V2 main 9-run B/C/D ensemble; input seq20 (provided masks, black bg, fixed crop); no GT stiffness exists',
           'predictions': out, 'E_over_rho_ranking_soft_to_stiff': [s.split('/')[1] for s in order]}
    Path(a.report).write_text(json.dumps(rep, indent=1))
    print(json.dumps({s.split('/')[1]: (round(v['log10_E_over_rho_x1000'], 2), v['law_majority'], v['law_votes']) for s, v in out.items()}, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='stage', required=True)
    b = sub.add_parser('build'); b.add_argument('zip'); b.add_argument('out')
    e = sub.add_parser('predict'); e.add_argument('out'); e.add_argument('report')
    a = p.parse_args(); {'build': build, 'predict': predict}[a.stage](a)
