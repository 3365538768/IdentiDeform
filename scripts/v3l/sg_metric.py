"""V3l Part 1: Spring-Gaus synthetic, metric-matched (Chamfer to GT particles + image metrics on eval cam 5).

  scripts/v3f/replay_env.sh scripts/v3l/sg_metric.py <mpm_synthetic.zip> <scene> --arm prior20|prior20_refine|prior30_v3f|gtE
        [--budget 12] [--prior data/v3l/prior20.json] [--out data/v3l/sg]

Spec: docs/v3l/PROTOCOL.json part1.  Engine / data loading / 128-px splat renderer / C1 objective / compass search
are those of scripts/v3k/a_sg_refine.py (imported unchanged); the only change of the search is that the observation
and the simulations are restricted to frames 0..N_OBS-1 (=19).  The chosen (E, nu) is then rolled out for all 30
frames and scored with:
  * Chamfer exactly as scripts/v3g/sg_eval.py (sub/cd copied verbatim), GT = full simulation/<scene>/<k>.ply;
  * image metrics on camera 5 at 512 px (PSNR as sg_eval.py, Gaussian SSIM, mask IoU) vs RGB*alpha-on-black GT.
Untrusted zip: only PLY/PNG/JSON decoded, nothing executed, python -I.
"""
import argparse, io, json, math, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v3k'))
import a_sg_refine as A  # noqa: E402  (sets up sim paths, numpy, cv2, torch)
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.ndimage import gaussian_filter  # noqa: E402

N_OBS = 20
EVAL_CAM = 5
HI = 512
MASK_THR = 0.02


# ---- verbatim from scripts/v3g/sg_eval.py
def sub(x, n=8192, seed=0):
    if len(x) <= n: return x
    return x[np.random.default_rng(seed).choice(len(x), n, replace=False)]


def cd(a, b):
    a, b = sub(a), sub(b)
    d1, _ = cKDTree(b).query(a); d2, _ = cKDTree(a).query(b)
    return 0.5 * (d1.mean() + d2.mean()), 0.5 * ((d1**2).mean() + (d2**2).mean())
# ----


def ssim(a, b):
    """a, b: HxWx3 in [0,1]. Gaussian window sigma 1.5 (truncate 3.5 -> 11x11), per channel, mean (Wang et al. 2004)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2; vals = []
    for c in range(3):
        x, y = a[..., c].astype(np.float64), b[..., c].astype(np.float64)
        f = lambda z: gaussian_filter(z, 1.5, truncate=3.5)
        mx, my = f(x), f(y); sxx = f(x * x) - mx * mx; syy = f(y * y) - my * my; sxy = f(x * y) - mx * my
        vals.append(((2 * mx * my + C1) * (2 * sxy + C2) / ((mx * mx + my * my + C1) * (sxx + syy + C2))).mean())
    return float(np.mean(vals))


def img_metrics(pred, gt_rgb, gt_mask, pred_mask_exact=None):
    mse = ((pred - gt_rgb) ** 2).mean()
    pm = pred.max(-1) > MASK_THR
    iou = lambda m: float((m & gt_mask).sum() / max((m | gt_mask).sum(), 1))
    out = {'psnr': float(10 * np.log10(1.0 / max(mse, 1e-12))), 'ssim': ssim(pred, gt_rgb), 'mask_iou': iou(pm)}
    if pred_mask_exact is not None:
        out['mask_iou_exact_splat'] = iou(pred_mask_exact)
    return out


def render_hi(X, col, cam, r, res=HI):
    """render() of a_sg_refine generalised to a given resolution, single camera; col 0-255 -> (mask, rgb in [0,1])."""
    K, w2c = cam
    offs = [(du, dv) for du in range(-r, r + 1) for dv in range(-r, r + 1) if du * du + dv * dv <= r * r + r]
    Xh = np.concatenate([X, np.ones((len(X), 1))], 1); Xc = Xh @ w2c.T
    zc = -Xc[:, 2]; ok = zc > 1e-4
    u = K[0, 0] * Xc[:, 0] / np.where(ok, zc, 1) + K[0, 2]; v = -K[1, 1] * Xc[:, 1] / np.where(ok, zc, 1) + K[1, 2]
    order = np.argsort(-zc); order = order[ok[order]]
    ui = np.floor(u[order]).astype(np.int64); vi = np.floor(v[order]).astype(np.int64); c = col[order]
    img = np.zeros((res, res, 3), np.float32); m = np.zeros((res, res), bool)
    U = np.concatenate([ui + du for du, dv in offs]); V = np.concatenate([vi + dv for du, dv in offs])
    rank = np.concatenate([np.arange(len(ui))] * len(offs)); C = np.concatenate([c] * len(offs))
    s = np.argsort(rank, kind='stable'); U, V, C = U[s], V[s], C[s]
    inb = (U >= 0) & (U < res) & (V >= 0) & (V < res); U, V, C = U[inb], V[inb], C[inb]
    m[V, U] = True; img[V, U] = C
    return m, np.clip(img / 255., 0, 1)


def gt_image(z, cams, cam_idx, f):
    im = np.asarray(Image.open(io.BytesIO(z.read(f'{A.PREFIX}render/{SCENE}/{cams[cam_idx]["camera"]}/{f:03d}.png'))).convert('RGBA'), np.float32) / 255
    return im[..., :3] * im[..., 3:4], im[..., 3] > 0.5


def hi_cam(cams, i):
    return np.array(cams[i]['K'], np.float64), np.linalg.inv(np.array(cams[i]['c2w'], np.float64))


def score_traj(traj, sc, cams, col, r_hi, z, gt_cache):
    """traj [T,N,3] (our particle subset, SG world) -> geometry + image metrics per frame."""
    T = len(sc['times']); keep = sc['keep']
    per = {}
    for k in range(T):
        G = gt_cache['ply'][k]
        c, c2 = cd(traj[k], G[:, :3].astype(np.float64))
        trk = float(np.linalg.norm(traj[k] - G[keep, :3], axis=1).mean()) if gt_cache['corr'] else None
        m, img = render_hi(traj[k], col, hi_cam(cams, EVAL_CAM), r_hi)
        gi, gm = gt_cache['img'][k]
        per[k] = {'cd': float(c), 'cd2_x1e3': float(c2 * 1e3), 'track_l2': trk, **img_metrics(img, gi, gm, m)}
    return per


def summarize(per):
    out = {}
    for name, ks in (('fit', range(0, N_OBS)), ('future', range(N_OBS, 30))):
        ks = [k for k in ks if k in per]
        for key in ('cd', 'cd2_x1e3', 'track_l2', 'psnr', 'ssim', 'mask_iou', 'mask_iou_exact_splat'):
            vals = [per[k][key] for k in ks if per[k].get(key) is not None]
            if vals:
                out[f'{name}_{key}_mean'] = float(np.mean(vals))
        out[f'{name}_n_frames'] = len(ks)
    return out


def main(a):
    global SCENE
    SCENE = a.scene
    t_all = time.time()
    sc = A.load_scene(a.zip, a.scene)
    z = sc['z']; cams = json.loads(z.read(f'{A.PREFIX}render/{a.scene}/camera.json'))
    out_dir = Path(a.out) / a.scene; out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / f'{a.arm}.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'evals': []}

    def save():
        tmp = state_path.with_suffix('.tmp'); tmp.write_text(json.dumps(state, indent=1)); os.replace(tmp, state_path)
    P = sc['P'][sc['keep']]; X0, col = P[:, :3].astype(np.float64), P[:, 3:].astype(np.float32)
    phys = sc['phys']; v0 = np.array(phys[1]['INIT_VELOCITY'], np.float64); gy = float(phys[1]['BC']['GROUND'][0][1])
    T = len(sc['times']); dtf = float(sc['times'][1] - sc['times'][0]); assert T == 30 and abs(dtf - 0.03) < 1e-6, (T, dtf)
    obs_m, obs_rgb = sc['obs']; obs20 = (obs_m[:, :N_OBS], obs_rgb[:, :N_OBS])
    gt_ok = isinstance(phys[0].get('E'), (int, float))
    gtE, gtnu = (math.log10(float(phys[0]['E'])), float(phys[0]['NU'])) if gt_ok else (None, None)
    # splat radius for the 128-px objective (frame-0 geometry, observed views; as v3k-A)
    if 'splat_radius' not in state:
        f0 = {r: A.compare((obs_m[:, :1], obs_rgb[:, :1]), tuple(x[:, None] for x in A.render(X0, col, sc['cams'], r)))['iou'] for r in (0, 1, 2)}
        state['splat_radius'] = max(f0, key=f0.get); state['frame0_iou_by_radius_128'] = f0; save()
    r = state['splat_radius']
    # splat radius for 512-px images: frame-0 IoU of GT particles on the OBSERVED t0 views at 512 (geometry only)
    if 'splat_radius_hi' not in state:
        ious = {}
        for rr in range(5):
            vals = []
            for v in sc['views']:
                gi, gm = gt_image(z, cams, v, 0); m, _ = render_hi(X0, col, hi_cam(cams, v), rr)
                vals.append((m & gm).sum() / max((m | gm).sum(), 1))
            ious[rr] = float(np.mean(vals))
        state['splat_radius_hi'] = max(ious, key=ious.get); state['frame0_iou_by_radius_512_t0views'] = ious; save()
    r_hi = state['splat_radius_hi']

    # estimator
    if a.arm in ('prior20', 'prior20_refine'):
        pr = json.loads(Path(a.prior).read_text())['scenes'][a.scene]['f20']
        x0, nu = pr['log10_E_Pa_at_rho1000'], pr['nu']
    elif a.arm == 'prior30_v3f':
        zp = json.loads((ROOT / 'docs/v3f/SPRINGGAUS_SYN.json').read_text())['ensemble9']['K0']['z_t0'][a.scene]
        x0, nu = A.log10E_from_z(zp[0]), float(A.nu_from_z(zp[1]))
    else:
        assert gt_ok; x0, nu = gtE, gtnu
    x0 = round(float(np.clip(x0, *A.BOUNDS)), 6) if a.arm != 'gtE' else x0
    state.update(scene=a.scene, arm=a.arm, x0=x0, nu=nu, gt_log10E=gtE, gt_nu=gtnu, n_obs=N_OBS, views=sc['views'],
                 n_particles_sim=int(len(X0)), n_particles_total=sc['n_particles_total'], eval_cam=EVAL_CAM)
    save()

    def sim(log10E, n):
        t0 = time.time()
        try:
            traj, info = A.simulate(X0, v0, gy, 10 ** log10E, nu, n, dtf)
        except Exception as e:  # noqa: BLE001
            traj, info = None, {'exception': repr(e)[:300]}
        return traj, info, time.time() - t0

    best_x = x0
    if a.arm == 'prior20_refine':
        def evaluate(x):
            x = round(float(x), 6)
            for e in state['evals']:
                if e['x'] == x:
                    return e
            if len(state['evals']) >= a.budget:
                return None
            traj, info, secs = sim(x, N_OBS)
            res = {'failed': True, 'loss': A.FAILED_LOSS} if traj is None else A.compare(obs20, A.render_traj(traj, col, sc['cams'], r))
            e = {'n': len(state['evals']) + 1, 'x': x, **res, 'info': info, 'seconds': secs}
            state['evals'].append(e); save()
            print(a.scene, 'eval', e['n'], x, round(e['loss'], 4), round(secs, 1), flush=True)
            return e
        best = evaluate(x0); step = A.STEP0; last = None
        while best is not None and len(state['evals']) < a.budget and step >= A.MIN_STEP:
            improved = False
            for s in ([last] if last else []) + [d for d in (+1, -1) if d != last]:
                y = round(float(np.clip(best['x'] + s * step, *A.BOUNDS)), 6)
                if y == best['x']:
                    continue
                e = evaluate(y)
                if e is None:
                    break
                if e['loss'] < best['loss']:
                    best, last, improved = e, s, True
                    break
            if not improved:
                step /= 2; last = None
        bud = {}
        for k in (0, 4, 12):
            n = min(k, len(state['evals']))
            bud[str(k)] = x0 if n == 0 else min(state['evals'][:n], key=lambda e: e['loss'])['x']
        state['best_x_at_budget'] = bud
        best_x = bud[str(a.budget)] if str(a.budget) in bud else min(state['evals'], key=lambda e: e['loss'])['x']
        state['search_seconds'] = float(sum(e['seconds'] for e in state['evals']))
        save()

    # GT cache for scoring
    gt_cache = {'ply': [A.read_ply(z, a.scene, k) for k in range(T)], 'img': [gt_image(z, cams, EVAL_CAM, k) for k in range(T)]}
    gt_cache['corr'] = all(len(p) == len(gt_cache['ply'][0]) for p in gt_cache['ply'])
    if 'renderer_ceiling' not in state:
        traj_gt = np.stack([gt_cache['ply'][k][sc['keep'], :3] for k in range(T)]).astype(np.float64)
        state['renderer_ceiling'] = summarize(score_traj(traj_gt, sc, cams, col, r_hi, z, gt_cache))
        state['static_baseline'] = summarize(score_traj(np.stack([X0] * T), sc, cams, col, r_hi, z, gt_cache))
        save()

    # final rollouts at the chosen E (and at budget 0/4 for the refine arm)
    rolls = {'final': best_x}
    if a.arm == 'prior20_refine':
        rolls.update({f'budget{k}': v for k, v in state['best_x_at_budget'].items()})
    state.setdefault('rollouts', {})
    done_x = {}
    for tag, x in rolls.items():
        if tag in state['rollouts']:
            continue
        if x in done_x:
            state['rollouts'][tag] = state['rollouts'][done_x[x]]; save(); continue
        traj, info, secs = sim(x, T)
        if traj is None:
            state['rollouts'][tag] = {'log10E': x, 'failed': True, 'info': info, 'seconds': secs}
        else:
            per = score_traj(traj, sc, cams, col, r_hi, z, gt_cache)
            state['rollouts'][tag] = {'log10E': x, 'nu': nu, 'abs_dlog10E': abs(x - gtE) if gtE is not None else None,
                                      'summary': summarize(per), 'per_frame': per, 'info': info, 'seconds': secs}
            if tag == 'final':
                np.savez_compressed(out_dir / f'{a.arm}_traj_f16.npz', traj=traj.astype(np.float16)[:, ::10])
        done_x[x] = tag; save()
        print(a.scene, a.arm, tag, round(x, 3), json.dumps(state['rollouts'][tag].get('summary'), default=float), flush=True)
    state['wall_s_this_invocation'] = time.time() - t_all
    state['complete'] = True; save()


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('zip'); p.add_argument('scene')
    p.add_argument('--arm', choices=('prior20', 'prior20_refine', 'prior30_v3f', 'gtE'), required=True)
    p.add_argument('--budget', type=int, default=12)
    p.add_argument('--prior', default=str(ROOT / 'data/v3l/prior20.json'))
    p.add_argument('--out', default=str(ROOT / 'data/v3l/sg'))
    main(p.parse_args())
