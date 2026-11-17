"""V3v: real-video replay evaluation on the Spring-Gaus real captures (protocol: docs/v3v/PROTOCOL.json).

Run with scripts/v3f/replay_env.sh (gsim python -I; this file adds sys.path itself). Stages:
  prep  <real_capture.zip>            decode masks/images (ROI crops), per-frame 3-view hulls, initial state -> data/v3v/obs/
  run   --objs ...                    MPM sims (prior, class mean, grid, compass per fold) + IoU tables -> data/v3v/sims/
  eval  --objs ...                    full metrics for every arm and fold -> data/v3v/eval/<obj>.json
  agg                                 RESULTS.json (per object, mean, bootstrap CI, LOO range)
Archive is untrusted: only PNG/JPG/JSON are decoded, nothing is executed.
"""
import argparse, io, json, math, sys, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v3f'))
import springgaus_replay as rp  # noqa: E402  (sets engine sys.path; provides hull/simulate/project/calib)
import numpy as np  # noqa: E402
import cv2  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

CAMS = rp.CAMS; FPS = rp.FPS; P = rp.P; G = rp.G; VOX = rp.VOX
OBJS = ['bun', 'burger', 'dog', 'pig', 'potato']
D = ROOT / 'data/v3v'
PRED = ROOT / 'docs/v3f/SPRINGGAUS_REAL_PRED.json'
CLASS_MEAN_LG, CLASS_MEAN_NU = 3.482, 0.278   # V2 train split, jelly (651 samples)
GRID = [round(3.0 + 0.25 * i, 2) for i in range(13)]
FIT_FRAMES = list(range(0, 10)); FUT_FRAMES = list(range(10, 20)); HO_FRAMES = list(range(1, 20))
BUDGET = 12


def key(lg, nu):
    return f'lg{lg:.4f}_nu{nu:.3f}'


# ------------------------------------------------------------------ geometry helpers
def cam_depth(X, rt):
    R, _ = cv2.Rodrigues(rt[0]); return (X @ R.T + rt[1])[:, 2]


def cam_center(rt):
    R, _ = cv2.Rodrigues(rt[0]); return -R.T @ rt[1]


def triangulate(pts, K, d, rts):
    A = []
    for c, uv in pts.items():
        n = cv2.undistortPoints(np.array(uv, np.float64).reshape(1, 1, 2), K, d).reshape(2)
        R, _ = cv2.Rodrigues(rts[c][0]); M = np.hstack([R, rts[c][1].reshape(3, 1)])
        A.append(n[0] * M[2] - M[0]); A.append(n[1] * M[2] - M[1])
    X = np.linalg.svd(np.array(A))[2][-1]; return X[:3] / X[3]


class Obs:
    def __init__(self, obj):
        z = np.load(D / 'obs' / f'{obj}.npz'); self.meta = json.loads((D / 'obs' / f'{obj}.json').read_text())
        self.K = np.array(self.meta['K']); self.d = np.array(self.meta['dist'])
        self.rts = {c: (np.array(self.meta['rt'][c][0]), np.array(self.meta['rt'][c][1])) for c in CAMS}
        self.roi = self.meta['roi']; self.hit = self.meta['hit']; self.n = self.meta['n']
        self.img = {c: z[f'img_{c}'] for c in CAMS}; self.msk = {c: z[f'msk_{c}'] for c in CAMS}
        self.X0 = z['X0']; self.C = z['C']; self.v0 = z['v0']; self.ground = float(z['ground'])
        self.hulls = [z[f'hull_{k}'] for k in range(self.n)]
        self.spacing = self.meta['spacing_px']


# ------------------------------------------------------------------ prep
def prep(a):
    zf = zipfile.ZipFile(a.zip); K, d, rts = rp.calib(zf)
    (D / 'obs').mkdir(parents=True, exist_ok=True)
    for obj in OBJS:
        seq = json.loads(zf.read(f'{P}sequences/{obj}/0.json')); hit = seq['hit_frame']; n = len(seq[CAMS[0]])
        M = {c: rp.masks_of(zf, seq, c) for c in CAMS}
        I = {c: [np.asarray(Image.open(io.BytesIO(zf.read(f'{P}videos_images/{c}/{f}'))).convert('RGB')) for f in seq[c]] for c in CAMS}
        C = []
        for k in range(n):
            cen = {}
            for c in CAMS:
                ys, xs = np.nonzero(M[c][k]); cen[c] = (xs.mean(), ys.mean())
            C.append(triangulate(cen, K, d, rts))
        C = np.array(C)
        hulls = [rp.hull({c: M[c][k] for c in CAMS}, C[k], K, d, rts) for k in range(n)]
        X0 = hulls[0]
        Xh = rp.hull({c: M[c][hit] for c in CAMS}, C[hit], K, d, rts); ground_u = float((-Xh[:, 2]).min())
        t = np.arange(n) / FPS; pre = np.arange(0, hit)
        vx = np.polyfit(t[pre], C[pre, 0], 1)[0]; vy = np.polyfit(t[pre], C[pre, 1], 1)[0]
        vu = np.polyfit(t[pre], -C[pre, 2] + 0.5 * G * t[pre] ** 2, 1)[0]
        v0 = np.array([vx, vy, -vu])
        roi, arrs, spacing = {}, {}, {}
        for c in CAMS:
            u = np.any(np.stack(M[c]), 0); ys, xs = np.nonzero(u); H, W = u.shape
            mx = max(120, int(0.5 * (xs.max() - xs.min()))); my = max(120, int(0.5 * (ys.max() - ys.min())))
            x0, x1 = max(0, xs.min() - mx), min(W, xs.max() + mx + 1); y0, y1 = max(0, ys.min() - my), min(H, ys.max() + my + 1)
            roi[c] = [int(x0), int(y0), int(x1), int(y1), int(W), int(H)]
            arrs[f'img_{c}'] = np.stack([im[y0:y1, x0:x1] for im in I[c]]).astype(np.uint8)
            arrs[f'msk_{c}'] = np.stack([m[y0:y1, x0:x1] for m in M[c]])
            spacing[c] = float(K[0, 0] * VOX / np.median(cam_depth(X0, rts[c])))
        for k in range(n):
            arrs[f'hull_{k}'] = hulls[k].astype(np.float32)
        np.savez_compressed(D / 'obs' / f'{obj}.npz', X0=X0.astype(np.float64), C=C, v0=v0, ground=ground_u, **arrs)
        meta = {'obj': obj, 'hit': int(hit), 'n': int(n), 'K': K.tolist(), 'dist': d.tolist(),
                'rt': {c: [rts[c][0].tolist(), rts[c][1].tolist()] for c in CAMS}, 'roi': roi, 'spacing_px': spacing,
                'hull_particles': [int(len(h)) for h in hulls], 'ground_up_m': ground_u, 'v0_world_mps': v0.tolist()}
        (D / 'obs' / f'{obj}.json').write_text(json.dumps(meta, indent=1))
        print(obj, 'hit', hit, 'X0', len(X0), 'spacing px', {c: round(s, 2) for c, s in spacing.items()}, flush=True)


# ------------------------------------------------------------------ rendering
def splat(o, c, X, colors=None):
    """Project particles X into camera c (ROI coords). Returns (mask, image or None, n_outside_roi_px_estimate)."""
    x0, y0, x1, y1, W, H = o.roi[c]; h, w = y1 - y0, x1 - x0
    s = o.spacing[c]; r = max(1, int(round(0.75 * s))); kc = 2 * int(math.ceil(s)) + 1
    uv = rp.project(X, o.K, o.d, o.rts[c]); dep = cam_depth(X, o.rts[c])
    u = np.round(uv[:, 0]).astype(np.int64) - x0; v = np.round(uv[:, 1]).astype(np.int64) - y0
    oy, ox = np.mgrid[-r:r + 1, -r:r + 1]; disc = (ox ** 2 + oy ** 2) <= r * r + 0.25
    ox, oy = ox[disc], oy[disc]
    U = (u[:, None] + ox[None]).ravel(); V = (v[:, None] + oy[None]).ravel(); pid = np.repeat(np.arange(len(X)), len(ox))
    ok = (U >= 0) & (U < w) & (V >= 0) & (V < h)
    outside = int((~ok).sum() // max(len(ox), 1))
    U, V, pid = U[ok], V[ok], pid[ok]
    cov = np.zeros((h, w), np.uint8); cov[V, U] = 1
    msk = cv2.morphologyEx(cov, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))).astype(bool)
    if colors is None:
        return msk, None, outside
    pix = V * w + U; order = np.lexsort((dep[pid], pix)); pix_s = pix[order]
    first = np.ones(len(pix_s), bool); first[1:] = pix_s[1:] != pix_s[:-1]
    img = np.zeros((h * w, 3), np.uint8); img[pix_s[first]] = colors[pid[order][first]]
    img = img.reshape(h, w, 3)
    hole = (msk & (cov == 0)).astype(np.uint8)
    if hole.any():
        img = cv2.inpaint(img, hole, 3, cv2.INPAINT_TELEA)
    img[~msk] = 0
    return msk, img, outside


def colorize(o, X0, train_cams):
    """Per-particle RGB at t0 from the training cameras (z-buffer visibility, inside observed mask, nearest camera)."""
    N = len(X0); col = np.zeros((N, 3), np.uint8); best = np.full(N, np.inf)
    for c in train_cams:
        x0, y0, x1, y1, W, H = o.roi[c]; h, w = y1 - y0, x1 - x0
        uv = rp.project(X0, o.K, o.d, o.rts[c]); dep = cam_depth(X0, o.rts[c])
        u = np.round(uv[:, 0]).astype(int) - x0; v = np.round(uv[:, 1]).astype(int) - y0
        ok = (u >= 0) & (u < w) & (v >= 0) & (v < h)
        zb = np.full((h, w), np.inf); np.minimum.at(zb, (v[ok], u[ok]), dep[ok])
        r = max(1, int(round(0.75 * o.spacing[c])))
        zb = cv2.erode(zb.astype(np.float32), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        vis = np.zeros(N, bool); vis[ok] = dep[ok] <= zb[v[ok], u[ok]] + 1.5 * VOX
        vis[ok] &= o.msk[c][0][v[ok], u[ok]]
        dist = np.linalg.norm(X0 - cam_center(o.rts[c]), axis=1)
        take = vis & (dist < best)
        col[take] = o.img[c][0][v[take], u[take]]; best[take] = dist[take]
    have = np.isfinite(best)
    if (~have).any() and have.any():
        A = torch.tensor(X0[~have], dtype=torch.float32); B = torch.tensor(X0[have], dtype=torch.float32)
        nn = torch.cat([torch.cdist(A[i:i + 4096], B).argmin(1) for i in range(0, len(A), 4096)]).numpy()
        col[~have] = col[have][nn]
    return col, float(have.mean())


def ssim_masked(a, b, m):
    a = a.astype(np.float64); b = b.astype(np.float64); C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    bl = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    vals = []
    for ch in range(3):
        x, y = a[..., ch], b[..., ch]; mx, my = bl(x), bl(y)
        sxx = bl(x * x) - mx * mx; syy = bl(y * y) - my * my; sxy = bl(x * y) - mx * my
        s = ((2 * mx * my + C1) * (2 * sxy + C2)) / ((mx * mx + my * my + C1) * (sxx + syy + C2))
        vals.append(s[m].mean())
    return float(np.mean(vals))


def img_metrics(o, c, k, msk, img):
    om = o.msk[c][k]; oi = o.img[c][k].copy(); oi[~om] = 0
    un = msk | om; inter = msk & om
    iou = float(inter.sum() / max(un.sum(), 1))
    ys, xs = np.nonzero(un); y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    A = oi[y0:y1, x0:x1]; B = img[y0:y1, x0:x1]; mm = un[y0:y1, x0:x1]
    mse = float(np.mean((A[mm].astype(np.float64) - B[mm].astype(np.float64)) ** 2))
    psnr = 10 * math.log10(255 ** 2 / max(mse, 1e-10))
    return {'iou': iou, 'psnr': psnr, 'ssim': ssim_masked(A, B, mm)}


def iou_table(o, traj):
    """IoU per camera per frame (silhouette only)."""
    T = {}
    for c in CAMS:
        T[c] = []
        for k in range(o.n):
            m, _, out = splat(o, c, traj[k]); om = o.msk[c][k]
            T[c].append(float((m & om).sum() / max((m | om).sum() + out, 1)))
    return T


# ------------------------------------------------------------------ run (simulations)
class Sims:
    def __init__(self, obj, o):
        self.obj, self.o = obj, o; self.dir = D / 'sims' / obj; self.dir.mkdir(parents=True, exist_ok=True)
        self.tab_path = self.dir / 'iou_tables.json'
        self.tab = json.loads(self.tab_path.read_text()) if self.tab_path.exists() else {}
        self.n_new = 0

    def traj(self, lg, nu):
        p = self.dir / f'{key(lg, nu)}.npy'
        if p.exists():
            return np.load(p)
        t0 = time.time()
        tr, info = rp.simulate(self.o.X0, self.o.v0, self.o.ground, 10 ** lg, nu, self.o.n, 1 / FPS)
        if tr is None:
            raise RuntimeError(f'sim failed {self.obj} {lg} {info}')
        np.save(p, tr.astype(np.float32)); self.n_new += 1
        print(f'  sim {self.obj} {key(lg, nu)} {time.time() - t0:.1f}s', flush=True)
        return tr.astype(np.float32)

    def ious(self, lg, nu):
        k = key(lg, nu)
        if k not in self.tab:
            self.tab[k] = {'lg': float(lg), 'nu': float(nu), 'iou': iou_table(self.o, self.traj(lg, nu))}
            self.tab_path.write_text(json.dumps(self.tab))
        return self.tab[k]['iou']


def fit_iou(T, train_cams, frames=FIT_FRAMES):
    return float(np.mean([T[c][k] for c in train_cams for k in frames]))


def compass(S, lg0, nu, train_cams):
    seen = {}

    def f(x):
        x = round(x, 4)
        if x not in seen:
            seen[x] = 1 - fit_iou(S.ious(x, nu), train_cams)
        return seen[x]
    x, step, path = round(lg0, 4), 0.5, []
    fx = f(x); path.append((x, fx))
    while step >= 0.0625 and len(seen) < BUDGET:
        cand = []
        for dx in (+step, -step):
            if len(seen) >= BUDGET and round(x + dx, 4) not in seen:
                continue
            if 2.0 <= x + dx <= 6.5:
                cand.append((f(x + dx), round(x + dx, 4)))
        if cand and min(cand)[0] < fx:
            fx, x = min(cand); path.append((x, fx))
        else:
            step /= 2
    return {'lg': x, 'loss': fx, 'n_sims': len(seen), 'path': path, 'evaluated': sorted(seen.items())}


def run(a):
    pred = json.loads(PRED.read_text())['predictions']
    for obj in a.objs:
        o = Obs(obj); S = Sims(obj, o); p = pred[f'springgaus_real/{obj}/seq20']
        lgp, nup = round(p['log10_E_over_rho_x1000'], 4), p['nu']
        S.ious(lgp, nup); S.ious(CLASS_MEAN_LG, CLASS_MEAN_NU)
        for g in GRID:
            S.ious(g, CLASS_MEAN_NU)
        folds = {}
        for h in CAMS:
            tc = [c for c in CAMS if c != h]
            folds[h] = compass(S, lgp, nup, tc)
            print(obj, 'fold', h, 'refined lg', folds[h]['lg'], 'sims', folds[h]['n_sims'], flush=True)
        json.dump({'obj': obj, 'prior_lg': lgp, 'prior_nu': nup, 'folds': folds, 'unique_sims_total': len(S.tab),
                   'refine_unique_sims_union_over_folds': len({x for f in folds.values() for x, _ in f['evaluated']})},
                  open(D / 'sims' / obj / 'refine.json', 'w'), indent=1)


# ------------------------------------------------------------------ eval
def metrics_3d(o, traj, frames):
    out = []
    for k in frames:
        X = traj[k]; cen = {}
        for c in CAMS:
            m, _, _ = splat(o, c, X); ys, xs = np.nonzero(m)
            if len(xs) == 0:
                cen = None; break
            x0, y0 = o.roi[c][:2]; cen[c] = (xs.mean() + x0, ys.mean() + y0)
        ce = float(np.linalg.norm(triangulate(cen, o.K, o.d, o.rts) - o.C[k]) * 100) if cen else float('nan')
        Hk = o.hulls[k]
        low = float(abs((-X[:, 2]).min() - (-Hk[:, 2]).min()) * 100)
        rng = np.random.default_rng(k)
        A = torch.tensor(X[rng.choice(len(X), min(4000, len(X)), replace=False)], dtype=torch.float64)
        B = torch.tensor(Hk[rng.choice(len(Hk), min(4000, len(Hk)), replace=False)], dtype=torch.float64)
        dm = torch.cdist(A, B); ch = float((dm.min(1)[0].mean() + dm.min(0)[0].mean()) / 2 * 100)
        out.append({'frame': k, 'centroid_err_cm': ce, 'lowest_err_cm': low, 'chamfer_cm': ch})
    return out


def agg_frames(recs):
    ks = [k for k in recs[0] if k not in ('frame', 'cam')]
    return {k: float(np.nanmean([r[k] for r in recs])) for k in ks}


def eval_obj(a):
    pred = json.loads(PRED.read_text())['predictions']
    tabs = {ob: json.loads((D / 'sims' / ob / 'iou_tables.json').read_text()) for ob in OBJS}
    for obj in a.objs:
        o = Obs(obj); S = Sims(obj, o); R = json.loads((D / 'sims' / obj / 'refine.json').read_text())
        p = pred[f'springgaus_real/{obj}/seq20']; lgp, nup = round(p['log10_E_over_rho_x1000'], 4), p['nu']
        # oracle: best IoU over all frames x all cams among all simulated E for this object
        allk = {k: np.mean([v['iou'][c][f] for c in CAMS for f in range(o.n)]) for k, v in tabs[obj].items() if '_r' not in k}
        ko = max(allk, key=allk.get); oracle = (tabs[obj][ko]['lg'], tabs[obj][ko]['nu'])
        ceiling = {c: tabs[obj][key(lgp, nup)]['iou'][c][0] for c in CAMS}
        res = {'obj': obj, 'prior_lg': lgp, 'oracle_lg': oracle[0], 'frame0_iou_ceiling': ceiling, 'folds': {}}
        for h in CAMS:
            tc = [c for c in CAMS if c != h]
            loo = {g: np.mean([fit_iou(tabs[ob][key(g, CLASS_MEAN_NU)]['iou'], tc) for ob in OBJS if ob != obj]) for g in GRID}
            g_loo = max(loo, key=loo.get)
            arms = {'prior': (lgp, nup), 'prior_refine': (R['folds'][h]['lg'], nup), 'class_mean': (CLASS_MEAN_LG, CLASS_MEAN_NU),
                    'global_constant_loo': (g_loo, CLASS_MEAN_NU), 'oracle': oracle}
            col, frac_vis = colorize(o, o.X0, tc)
            F = {'heldout_cam': h, 'train_cams': tc, 'color_visible_frac': frac_vis, 'arms': {}}
            # 2-cam triangulated centroid for the rigid-track reference
            C2 = []
            for k in range(o.n):
                cen = {}
                for c in tc:
                    ys, xs = np.nonzero(o.msk[c][k]); cen[c] = (xs.mean() + o.roi[c][0], ys.mean() + o.roi[c][1])
                C2.append(triangulate(cen, o.K, o.d, o.rts))
            C2 = np.array(C2)
            rigid = np.stack([o.X0 + (C2[k] - C2[0]) for k in range(o.n)])
            for arm, (lg, nu) in list(arms.items()) + [('rigid_track_2cam', (None, None)), ('persist_last', (None, None))]:
                if arm == 'persist_last':
                    recs = []
                    for c in CAMS:
                        oi = o.img[c][9].copy(); om = o.msk[c][9]; oi[~om] = 0
                        for k in FUT_FRAMES:
                            r = img_metrics(o, c, k, om, oi); r.update(cam=c, frame=k); recs.append(r)
                    m3 = []
                    for k in FUT_FRAMES:
                        Hk, H9 = o.hulls[k], o.hulls[9]
                        rng = np.random.default_rng(k)
                        A = torch.tensor(H9[rng.choice(len(H9), min(4000, len(H9)), replace=False)], dtype=torch.float64)
                        B = torch.tensor(Hk[rng.choice(len(Hk), min(4000, len(Hk)), replace=False)], dtype=torch.float64); dm = torch.cdist(A, B)
                        m3.append({'frame': k, 'centroid_err_cm': float(np.linalg.norm(o.C[9] - o.C[k]) * 100),
                                   'lowest_err_cm': float(abs((-H9[:, 2]).min() - (-Hk[:, 2]).min()) * 100),
                                   'chamfer_cm': float((dm.min(1)[0].mean() + dm.min(0)[0].mean()) / 2 * 100)})
                    F['arms'][arm] = {'future_frames': agg_frames(recs), 'future_3d': agg_frames(m3)}
                    continue
                traj = rigid if arm == 'rigid_track_2cam' else S.traj(lg, nu)
                ho = []
                for k in HO_FRAMES:
                    m, im, _ = splat(o, h, traj[k], col); r = img_metrics(o, h, k, m, im); r.update(cam=h, frame=k); ho.append(r)
                e = {'lg': lg, 'nu': nu, 'heldout_camera': agg_frames(ho),
                     'heldout_camera_post_hit': agg_frames([r for r in ho if r['frame'] >= o.hit]),
                     'heldout_per_frame_iou': [r['iou'] for r in ho]}
                if arm != 'rigid_track_2cam':
                    fu = []
                    for c in CAMS:
                        for k in FUT_FRAMES:
                            m, im, _ = splat(o, c, traj[k], col); r = img_metrics(o, c, k, m, im); r.update(cam=c, frame=k); fu.append(r)
                    e['future_frames'] = agg_frames(fu)
                    e['fit_set_iou'] = fit_iou(tabs[obj][key(lg, nu)]['iou'], tc)
                    if h == CAMS[0] or arm in ('prior_refine', 'global_constant_loo'):
                        m3 = metrics_3d(o, traj, list(range(1, o.n)))
                        e['future_3d'] = agg_frames([r for r in m3 if r['frame'] in FUT_FRAMES])
                        e['all_3d'] = agg_frames(m3)
                F['arms'][arm] = e
                print(obj, h, arm, None if lg is None else round(lg, 3), 'HO', {k: round(v, 3) for k, v in e['heldout_camera'].items()},
                      'FU', {k: round(v, 3) for k, v in e.get('future_frames', {}).items()}, flush=True)
            res['folds'][h] = F
        # 3D metrics that do not depend on the fold: copy from fold 0
        for h in CAMS[1:]:
            for arm in ('prior', 'class_mean', 'oracle'):
                for kk in ('future_3d', 'all_3d'):
                    res['folds'][h]['arms'][arm][kk] = res['folds'][CAMS[0]]['arms'][arm][kk]
        (D / 'eval').mkdir(exist_ok=True)
        (D / 'eval' / f'{obj}.json').write_text(json.dumps(res, indent=1, default=float))


# ------------------------------------------------------------------ exploratory (post hoc): E + strain-rate damping (RPIC)
def simulate_rpic(Xw, v0w, ground_u, E, nu, rpic, n_frames, dt_frame, device='cuda:0'):
    """Same as springgaus_replay.simulate but with rpic_damping (damps the symmetric part of the APIC affine velocity
    = strain-rate dissipation; rigid translation/rotation untouched)."""
    import warp as wp
    wp.init()
    from mpm_solver_warp import MPM_Simulator_WARP
    GL, NG = rp.GRID_LIM, rp.N_GRID
    cx, cy = Xw[:, 0].mean(), Xw[:, 1].mean()
    to_sim = lambda X: np.stack([X[:, 0] - cx + GL / 2, X[:, 1] - cy + GL / 2, -X[:, 2] - ground_u + 0.08], 1)
    to_world = lambda S_: np.stack([S_[:, 0] + cx - GL / 2, S_[:, 1] + cy - GL / 2, -(S_[:, 2] + ground_u - 0.08)], 1)
    S_ = torch.tensor(to_sim(Xw), dtype=torch.float32, device=device)
    vol = torch.full((len(S_),), VOX ** 3, dtype=torch.float32, device=device)
    sim = MPM_Simulator_WARP(10, device=device)
    sim.load_initial_data_from_torch(S_, vol, None, n_grid=NG, grid_lim=GL, device=device)
    sim.set_parameters_dict({'material': 'jelly', 'E': float(E), 'nu': float(nu), 'density': 1000.0, 'g': [0.0, 0.0, -G],
                             'rpic_damping': float(rpic), 'grid_v_damping_scale': 0.99999, 'n_grid': NG, 'grid_lim': GL}, device=device)
    sim.add_bounding_box()
    sim.add_surface_collider([GL / 2, GL / 2, 0.08], [0.0, 0.0, 1.0], surface='slip', friction=0.3)
    sim.finalize_mu_lam(device=device)
    v = torch.tensor([v0w[0], v0w[1], -v0w[2]], dtype=torch.float32, device=device).expand(len(S_), 3).contiguous()
    sim.import_particle_v_from_torch(v, device=device)
    lam_mu = E / (2 * (1 + nu)) + E * nu / ((1 + nu) * (1 - 2 * nu))
    c = math.sqrt((lam_mu + E / (2 * (1 + nu))) / 1000.)
    dt = min(1e-4, 0.25 * (GL / NG) / max(c, 1e-3))
    steps = max(1, int(math.ceil(dt_frame / dt))); dt = dt_frame / steps
    out = [Xw.copy()]; step = 0
    for f in range(1, n_frames):
        for _ in range(steps):
            sim.p2g2p(step, dt, device=device); step += 1
        Sx = sim.export_particle_x_to_torch().detach().cpu().numpy()
        if not np.isfinite(Sx).all():
            return None
        out.append(to_world(Sx))
    return np.stack(out)


def key2(lg, nu, r):
    return key(lg, nu) if abs(r) < 1e-9 else f'{key(lg, nu)}_r{r:.4f}'


def traj2(S, lg, nu, r):
    p = S.dir / f'{key2(lg, nu, r)}.npy'
    if p.exists():
        return np.load(p)
    if abs(r) < 1e-9:
        return S.traj(lg, nu)
    t0 = time.time(); tr = simulate_rpic(S.o.X0, S.o.v0, S.o.ground, 10 ** lg, nu, r, S.o.n, 1 / FPS)
    if tr is None:
        raise RuntimeError(f'sim failed {S.obj} {lg} {r}')
    np.save(p, tr.astype(np.float32)); print(f'  sim {S.obj} {key2(lg, nu, r)} {time.time() - t0:.1f}s', flush=True)
    return tr.astype(np.float32)


def ious2(S, lg, nu, r):
    k = key2(lg, nu, r)
    if k not in S.tab:
        S.tab[k] = {'lg': float(lg), 'nu': float(nu), 'rpic': float(r), 'iou': iou_table(S.o, traj2(S, lg, nu, r))}
        S.tab_path.write_text(json.dumps(S.tab))
    return S.tab[k]['iou']


def compass2(S, lg0, nu, train_cams):
    seen = {}

    def f(x, r):
        x, r = round(x, 4), round(r, 4)
        if (x, r) not in seen:
            seen[(x, r)] = 1 - fit_iou(ious2(S, x, nu, r), train_cams)
        return seen[(x, r)]
    x, r, sx, sr = round(lg0, 4), 0.0, 0.5, 0.5
    fx = f(x, r); path = [(x, r, fx)]
    while sx >= 0.0625 and len(seen) < BUDGET:
        moved = False
        for dx, dr in ((0, sr), (sx, 0), (-sx, 0), (0, -sr)):
            nx, nr = round(x + dx, 4), round(r + dr, 4)
            if not (2.0 <= nx <= 6.5 and 0.0 <= nr <= 1.0) or (len(seen) >= BUDGET and (nx, nr) not in seen):
                continue
            fn = f(nx, nr)
            if fn < fx:
                x, r, fx, moved = nx, nr, fn, True; path.append((x, r, fx)); break
        if not moved:
            sx /= 2; sr /= 2
    return {'lg': x, 'rpic': r, 'loss': fx, 'n_sims': len(seen), 'path': path, 'evaluated': [[k[0], k[1], v] for k, v in sorted(seen.items())]}


def run2(a):
    pred = json.loads(PRED.read_text())['predictions']
    for obj in a.objs:
        o = Obs(obj); S = Sims(obj, o); p = pred[f'springgaus_real/{obj}/seq20']
        lgp, nup = round(p['log10_E_over_rho_x1000'], 4), p['nu']
        folds = {}
        for h in CAMS:
            folds[h] = compass2(S, lgp, nup, [c for c in CAMS if c != h])
            print(obj, 'fold', h, 'refined2 lg', folds[h]['lg'], 'rpic', folds[h]['rpic'], 'sims', folds[h]['n_sims'], flush=True)
        json.dump({'obj': obj, 'folds': folds}, open(D / 'sims' / obj / 'refine_damp.json', 'w'), indent=1)


def eval2(a):
    """Adds arm 'prior_refine_damp' (exploratory) to data/v3v/eval/<obj>.json with the same metrics."""
    for obj in a.objs:
        o = Obs(obj); S = Sims(obj, o); R2 = json.loads((D / 'sims' / obj / 'refine_damp.json').read_text())
        ev = json.loads((D / 'eval' / f'{obj}.json').read_text())
        for h in CAMS:
            tc = [c for c in CAMS if c != h]; col, _ = colorize(o, o.X0, tc)
            lg, r = R2['folds'][h]['lg'], R2['folds'][h]['rpic']; nu = ev['folds'][h]['arms']['prior']['nu']
            traj = traj2(S, lg, nu, r)
            ho, fu = [], []
            for k in HO_FRAMES:
                m, im, _ = splat(o, h, traj[k], col); rr = img_metrics(o, h, k, m, im); rr.update(cam=h, frame=k); ho.append(rr)
            for c in CAMS:
                for k in FUT_FRAMES:
                    m, im, _ = splat(o, c, traj[k], col); rr = img_metrics(o, c, k, m, im); rr.update(cam=c, frame=k); fu.append(rr)
            m3 = metrics_3d(o, traj, list(range(1, o.n)))
            ev['folds'][h]['arms']['prior_refine_damp'] = {
                'lg': lg, 'rpic': r, 'nu': nu, 'heldout_camera': agg_frames(ho),
                'heldout_camera_post_hit': agg_frames([x for x in ho if x['frame'] >= o.hit]), 'heldout_per_frame_iou': [x['iou'] for x in ho],
                'future_frames': agg_frames(fu), 'fit_set_iou': fit_iou(S.tab[key2(lg, nu, r)]['iou'], tc),
                'future_3d': agg_frames([x for x in m3 if x['frame'] in FUT_FRAMES]), 'all_3d': agg_frames(m3)}
            if a.with_oracle:
                tab = json.loads(S.tab_path.read_text())
                allk = {kk: np.mean([v['iou'][c][f] for c in CAMS for f in range(o.n)]) for kk, v in tab.items()}
                ko = max(allk, key=allk.get); v = tab[ko]; tr = traj2(S, v['lg'], v['nu'], v.get('rpic', 0.0))
                hoo = []
                for k in HO_FRAMES:
                    m, im, _ = splat(o, h, tr[k], col); rr = img_metrics(o, h, k, m, im); rr.update(cam=h, frame=k); hoo.append(rr)
                fuo = []
                for c in CAMS:
                    for k in FUT_FRAMES:
                        m, im, _ = splat(o, c, tr[k], col); rr = img_metrics(o, c, k, m, im); rr.update(cam=c, frame=k); fuo.append(rr)
                m3o = metrics_3d(o, tr, list(range(1, o.n)))
                ev['folds'][h]['arms']['oracle_damp'] = {'lg': v['lg'], 'rpic': v.get('rpic', 0.0), 'nu': v['nu'], 'heldout_camera': agg_frames(hoo),
                    'heldout_camera_post_hit': agg_frames([x for x in hoo if x['frame'] >= o.hit]), 'future_frames': agg_frames(fuo),
                    'future_3d': agg_frames([x for x in m3o if x['frame'] in FUT_FRAMES]), 'all_3d': agg_frames(m3o)}
            e = ev['folds'][h]['arms']['prior_refine_damp']
            print(obj, h, 'prior_refine_damp', lg, r, 'HO', {k: round(v, 3) for k, v in e['heldout_camera'].items()},
                  'FU', {k: round(v, 3) for k, v in e['future_frames'].items()}, flush=True)
        (D / 'eval' / f'{obj}.json').write_text(json.dumps(ev, indent=1, default=float))


# ------------------------------------------------------------------ confirmation: LOO damping (PROTOCOL_confirm.json)
def loo_rpic(obj, h):
    vals = [json.loads((D / 'sims' / ob / 'refine_damp.json').read_text())['folds'][h]['rpic'] for ob in OBJS if ob != obj]
    return float(np.median(vals))


def compass_fixed_r(S, lg0, nu, r, train_cams):
    seen = {}

    def f(x):
        x = round(x, 4)
        if x not in seen:
            seen[x] = 1 - fit_iou(ious2(S, x, nu, r), train_cams)
        return seen[x]
    x, step, path = round(lg0, 4), 0.5, []
    fx = f(x); path.append((x, fx))
    while step >= 0.0625 and len(seen) < BUDGET:
        cand = []
        for dx in (+step, -step):
            if len(seen) >= BUDGET and round(x + dx, 4) not in seen:
                continue
            if 2.0 <= x + dx <= 6.5:
                cand.append((f(x + dx), round(x + dx, 4)))
        if cand and min(cand)[0] < fx:
            fx, x = min(cand); path.append((x, fx))
        else:
            step /= 2
    return {'lg': x, 'rpic': r, 'loss': fx, 'n_sims': len(seen), 'path': path, 'evaluated': sorted(seen.items())}


def run3(a):
    pred = json.loads(PRED.read_text())['predictions']
    for obj in a.objs:
        o = Obs(obj); S = Sims(obj, o); p = pred[f'springgaus_real/{obj}/seq20']
        lgp, nup = round(p['log10_E_over_rho_x1000'], 4), p['nu']
        folds = {}
        for h in CAMS:
            r = loo_rpic(obj, h)
            folds[h] = compass_fixed_r(S, lgp, nup, r, [c for c in CAMS if c != h])
            print(obj, 'fold', h, 'loo rpic', r, 'refined lg', folds[h]['lg'], 'sims', folds[h]['n_sims'], flush=True)
        json.dump({'obj': obj, 'folds': folds}, open(D / 'sims' / obj / 'refine_damp_loo.json', 'w'), indent=1)


def eval3(a):
    for obj in a.objs:
        o = Obs(obj); S = Sims(obj, o); R3 = json.loads((D / 'sims' / obj / 'refine_damp_loo.json').read_text())
        ev = json.loads((D / 'eval' / f'{obj}.json').read_text())
        for h in CAMS:
            tc = [c for c in CAMS if c != h]; col, _ = colorize(o, o.X0, tc)
            lg, r = R3['folds'][h]['lg'], R3['folds'][h]['rpic']; nu = ev['folds'][h]['arms']['prior']['nu']
            traj = traj2(S, lg, nu, r); ho, fu = [], []
            for k in HO_FRAMES:
                m, im, _ = splat(o, h, traj[k], col); rr = img_metrics(o, h, k, m, im); rr.update(cam=h, frame=k); ho.append(rr)
            for c in CAMS:
                for k in FUT_FRAMES:
                    m, im, _ = splat(o, c, traj[k], col); rr = img_metrics(o, c, k, m, im); rr.update(cam=c, frame=k); fu.append(rr)
            m3 = metrics_3d(o, traj, list(range(1, o.n)))
            e = {'lg': lg, 'rpic': r, 'nu': nu, 'heldout_camera': agg_frames(ho),
                 'heldout_camera_post_hit': agg_frames([x for x in ho if x['frame'] >= o.hit]), 'heldout_per_frame_iou': [x['iou'] for x in ho],
                 'future_frames': agg_frames(fu), 'fit_set_iou': fit_iou(S.tab[key2(lg, nu, r)]['iou'], tc),
                 'future_3d': agg_frames([x for x in m3 if x['frame'] in FUT_FRAMES]), 'all_3d': agg_frames(m3)}
            ev['folds'][h]['arms']['prior_refine_damp_loo'] = e
            print(obj, h, 'prior_refine_damp_loo', lg, r, 'HO', {k: round(v, 3) for k, v in e['heldout_camera'].items()},
                  'FU', {k: round(v, 3) for k, v in e['future_frames'].items()}, flush=True)
        (D / 'eval' / f'{obj}.json').write_text(json.dumps(ev, indent=1, default=float))


def figdump(a):
    """Crops for the qualitative strip: held-out camera a.cam, frames a.frames, arms prior / prior_refine / oracle."""
    (D / 'fig').mkdir(exist_ok=True)
    for obj in a.objs:
        o = Obs(obj); S = Sims(obj, o); ev = json.loads((D / 'eval' / f'{obj}.json').read_text())
        F = ev['folds'][a.cam]; tc = F['train_cams']; col, _ = colorize(o, o.X0, tc)
        frames = [o.hit if f < 0 else f for f in a.frames]
        out = {'frames': np.array(frames)}
        ms = []
        for arm in ('prior', 'prior_refine', 'oracle'):
            e = F['arms'][arm]; traj = S.traj(e['lg'], e['nu'])
            for k in frames:
                m, im, _ = splat(o, a.cam, traj[k], col); out[f'{arm}_{k}_img'] = im; out[f'{arm}_{k}_msk'] = m; ms.append(m)
        for k in frames:
            oi = o.img[a.cam][k].copy(); oi[~o.msk[a.cam][k]] = 0; out[f'obs_{k}_img'] = oi; out[f'obs_{k}_msk'] = o.msk[a.cam][k]
            ms.append(o.msk[a.cam][k])
        u = np.any(np.stack(ms), 0); ys, xs = np.nonzero(u)
        cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2; half = int(max(ys.max() - ys.min(), xs.max() - xs.min()) * 0.55) + 8
        H, W = u.shape; y0, x0 = max(0, cy - half), max(0, cx - half); y1, x1 = min(H, cy + half), min(W, cx + half)
        for kk in list(out):
            if kk != 'frames':
                out[kk] = out[kk][y0:y1, x0:x1]
        out['lg'] = np.array([F['arms'][arm]['lg'] for arm in ('prior', 'prior_refine', 'oracle')])
        np.savez_compressed(D / 'fig' / f'{obj}_{a.cam}.npz', **out)
        print('figdump', obj, a.cam, frames, flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='stage', required=True)
    s = sub.add_parser('prep'); s.add_argument('zip')
    s = sub.add_parser('run'); s.add_argument('--objs', nargs='+', default=OBJS)
    s = sub.add_parser('eval'); s.add_argument('--objs', nargs='+', default=OBJS)
    s = sub.add_parser('figdump'); s.add_argument('--objs', nargs='+', default=OBJS); s.add_argument('--cam', default='C0787')
    s.add_argument('--frames', nargs='+', type=int, default=[-1, 13, 19])
    s = sub.add_parser('run2'); s.add_argument('--objs', nargs='+', default=OBJS)
    s = sub.add_parser('eval2'); s.add_argument('--objs', nargs='+', default=OBJS); s.add_argument('--with_oracle', type=int, default=1)
    s = sub.add_parser('run3'); s.add_argument('--objs', nargs='+', default=OBJS)
    s = sub.add_parser('eval3'); s.add_argument('--objs', nargs='+', default=OBJS)
    a = ap.parse_args(); {'prep': prep, 'run': run, 'eval': eval_obj, 'figdump': figdump, 'run2': run2, 'eval2': eval2,
                          'run3': run3, 'eval3': eval3}[a.stage](a)
