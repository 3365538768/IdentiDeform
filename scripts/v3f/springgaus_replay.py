"""V3f: MPM replay of the Spring-Gaus real drops with OUR engine (DeformFieldBench mpm_solver_warp, jelly =
fixed-corotated, same solver as V2 data generation) using predicted vs refined stiffness.

  replay <real_capture.zip> <prepared_real_dir> <pred.json> <out_dir> --objs bun burger ...

Per object (all from released data; nothing from the archive executed, only JPG/PNG/JSON decoded):
  geometry   3-view visual hull of frame 0 (object in free flight) from the released masks + calibration
             (world frame = checkerboard, metres, up = -z), voxel 2.5 mm -> MPM particles (8 per 5 mm cell).
  ground     height = lowest visual-hull point at the hit frame (3-view hull, coarse; reported).
  init state centroid offset from the hull, velocity from a fit of the 7 pre-hit centroids with |g| = 9.8 m/s^2.
  physics    rho = 1000 (only E/rho matters for MPM motion), E = predicted E/rho x 1000, nu predicted,
             slip ground, friction 0.3, rpic_damping 0, grid_v_damping 0.99999 (as V2 C1 refine), dt from CFL.
  compare    simulated particle centroid vs triangulated mask centroid (cm), simulated silhouettes (particles
             projected with the real calibration, splatted) vs released masks (IoU), and rebound metrics:
             e = v_up(after)/v_down(before) from parabola fits (g = 9.8 fixed) and extrapolated apex height.
  refine     grid log10(E/rho x 1000) in {3.0..6.0 step 0.25} plus the prediction; pick min post-hit centroid
             RMSE (up axis) + (1 - mean IoU) — a small simulation-in-the-loop search, 14 simulations/object.
"""
import argparse, io, json, math, sys, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / 'upstream/DeformFieldBench/simulation'
sys.path[:0] = [str(SIM / 'mpm_solver_warp'), str(SIM), str(ROOT / '.engine-envs/gsim/lib/python3.12/site-packages'),
                str(ROOT / '.venv/lib/python3.12/site-packages')]
import numpy as np  # noqa: E402
import cv2  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

CAMS = ['C0733', 'C0787', 'C0801']
FPS = 119.88
P = 'real_capture/dynamic/'
G = 9.8
VOX = 0.0025
GRID_LIM, N_GRID = 0.5, 100


def calib(z):
    c = json.loads(z.read(P + 'cameras_calib.json'))
    K = np.array(c['camera_matrix']); d = np.array(c['distortion_coefficient'])
    return K, d, {k: (np.array(c[k]['rvecs'], np.float64).reshape(3), np.array(c[k]['tvecs'], np.float64).reshape(3)) for k in CAMS}


def project(X, K, d, rt):
    uv, _ = cv2.projectPoints(X.reshape(-1, 1, 3).astype(np.float64), rt[0], rt[1], K, d)
    return uv.reshape(-1, 2)


def masks_of(z, seq, cam):
    return [np.asarray(Image.open(io.BytesIO(z.read(f'{P}videos_masks/{cam}/{f[:-4]}.png'))).convert('L')) > 127 for f in seq[cam]]


def hull(masks_k, center, K, d, rts, half=0.08, vox=VOX):
    ax = np.arange(-half, half + 1e-9, vox)
    X = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), -1).reshape(-1, 3) + center
    keep = np.ones(len(X), bool)
    for c in CAMS:
        uv = np.round(project(X, K, d, rts[c])).astype(int); m = masks_k[c]; H, W = m.shape
        ok = (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        inside = np.zeros(len(X), bool); inside[ok] = m[uv[ok, 1], uv[ok, 0]]
        keep &= inside
    return X[keep]


def silhouette(X, K, d, rt, shape, radius=3):
    m = np.zeros(shape, np.uint8)
    uv = np.round(project(X, K, d, rt)).astype(int)
    ok = (uv[:, 0] >= 0) & (uv[:, 0] < shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < shape[0])
    m[uv[ok, 1], uv[ok, 0]] = 1
    m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return m.astype(bool)


def bounce_metrics(u, hit, fps=FPS):
    """u: up-coordinate of centroid per frame. Fit pre (0..hit-1) and post (first rising frame..) parabolas, g fixed."""
    t = np.arange(len(u)) / fps
    pre = np.arange(0, hit)
    a = np.polyfit(t[pre], u[pre] + 0.5 * G * t[pre] ** 2, 1)  # u + g t^2/2 = v0 t + u0
    # post: frames after the minimum of u following the hit
    lo = hit + int(np.argmin(u[hit:hit + 6]))
    post = np.arange(lo + 1, len(u))
    if len(post) < 3:
        return None
    b = np.polyfit(t[post], u[post] + 0.5 * G * t[post] ** 2, 1)
    t_c = t[lo]
    v_in = a[0] - G * t_c; v_out = b[0] - G * t_c
    u_at = lambda tt, c: c[1] + c[0] * tt - 0.5 * G * tt ** 2
    t_ap = b[0] / G; apex = u_at(t_ap, b)
    return {'contact_frame': int(lo), 'v_in_mps': float(v_in), 'v_out_mps': float(v_out), 'restitution_e': float(-v_out / v_in) if v_in < 0 else None,
            'apex_minus_contact_cm': float((apex - u[lo]) * 100), 'drop_frame0_minus_contact_cm': float((u[0] - u[lo]) * 100)}


def simulate(Xw, v0w, ground_u, E, nu, n_frames, dt_frame, device='cuda:0'):
    """Xw: world particles (m), up = -z. Returns world positions per frame [n_frames, N, 3]."""
    import warp as wp
    wp.init()
    from mpm_solver_warp import MPM_Simulator_WARP
    # sim frame: s = (x - cx + L/2, y - cy + L/2, (-z) - ground_u + 0.08)
    cx, cy = Xw[:, 0].mean(), Xw[:, 1].mean()
    to_sim = lambda X: np.stack([X[:, 0] - cx + GRID_LIM / 2, X[:, 1] - cy + GRID_LIM / 2, -X[:, 2] - ground_u + 0.08], 1)
    to_world = lambda S: np.stack([S[:, 0] + cx - GRID_LIM / 2, S[:, 1] + cy - GRID_LIM / 2, -(S[:, 2] + ground_u - 0.08)], 1)
    S = torch.tensor(to_sim(Xw), dtype=torch.float32, device=device)
    vol = torch.full((len(S),), VOX ** 3, dtype=torch.float32, device=device)
    sim = MPM_Simulator_WARP(10, device=device)
    sim.load_initial_data_from_torch(S, vol, None, n_grid=N_GRID, grid_lim=GRID_LIM, device=device)
    sim.set_parameters_dict({'material': 'jelly', 'E': float(E), 'nu': float(nu), 'density': 1000.0, 'g': [0.0, 0.0, -G],
                             'rpic_damping': 0.0, 'grid_v_damping_scale': 0.99999, 'n_grid': N_GRID, 'grid_lim': GRID_LIM}, device=device)
    sim.add_bounding_box()
    sim.add_surface_collider([GRID_LIM / 2, GRID_LIM / 2, 0.08], [0.0, 0.0, 1.0], surface='slip', friction=0.3)
    sim.finalize_mu_lam(device=device)
    v = torch.tensor([v0w[0], v0w[1], -v0w[2]], dtype=torch.float32, device=device).expand(len(S), 3).contiguous()
    sim.import_particle_v_from_torch(v, device=device)
    lam_mu = E / (2 * (1 + nu)) + E * nu / ((1 + nu) * (1 - 2 * nu))
    c = math.sqrt((lam_mu + E / (2 * (1 + nu))) / 1000.)
    dt = min(1e-4, 0.25 * (GRID_LIM / N_GRID) / max(c, 1e-3))
    steps = max(1, int(math.ceil(dt_frame / dt))); dt = dt_frame / steps
    out = [Xw.copy()]; step = 0
    for f in range(1, n_frames):
        for _ in range(steps):
            sim.p2g2p(step, dt, device=device); step += 1
        Sx = sim.export_particle_x_to_torch().detach().cpu().numpy()
        if not np.isfinite(Sx).all():
            return None, {'dt': dt, 'failed_frame': f}
        out.append(to_world(Sx))
    return np.stack(out), {'dt': dt, 'steps_per_frame': steps}


def main(a):
    z = zipfile.ZipFile(a.zip); K, d, rts = calib(z)
    idx = json.loads((Path(a.prepared) / 'index.json').read_text())
    pred = json.loads(Path(a.pred).read_text())['predictions']
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    report = {}
    for obj in a.objs:
        seq = json.loads(z.read(f'{P}sequences/{obj}/0.json')); hit = seq['hit_frame']; n = len(seq[CAMS[0]])
        M = {c: masks_of(z, seq, c) for c in CAMS}
        C = np.array(idx['objects'][obj]['centroid_xyz'])
        X0 = hull({c: M[c][0] for c in CAMS}, C[0], K, d, rts)
        Xh = hull({c: M[c][hit] for c in CAMS}, C[hit], K, d, rts)
        u0 = -X0[:, 2]; ground_u = float((-Xh[:, 2]).min())
        # free-flight state from pre-hit centroids (g fixed); hull centroid vs mask-centroid offset kept
        t = np.arange(n) / FPS; pre = np.arange(0, hit)
        vx = np.polyfit(t[pre], C[pre, 0], 1)[0]; vy = np.polyfit(t[pre], C[pre, 1], 1)[0]
        vu = np.polyfit(t[pre], -C[pre, 2] + 0.5 * G * t[pre] ** 2, 1)[0]
        v0 = np.array([vx, vy, -vu])
        obs_u = -C[:, 2]
        obs = bounce_metrics(obs_u, hit)
        p = pred[f'springgaus_real/{obj}/seq20']
        cands = [('predicted', p['log10_E_over_rho_x1000'])] + [(f'grid_{g:.2f}', g) for g in np.arange(3.0, 6.01, 0.25)]
        n_sim = int(round(0.45 * FPS))
        res = {}
        for tag, lg in cands:
            t0 = time.time()
            traj, info = simulate(X0, v0, ground_u, 10 ** lg, p['nu'], n_sim, 1 / FPS)
            if traj is None:
                res[tag] = {'log10_E_over_rho_x1000': lg, 'failed': info}; continue
            cen = traj.mean(1)
            err = np.linalg.norm(cen[:n] - C, axis=1) * 100
            ious = []
            for k in range(n):
                for c in CAMS:
                    s = silhouette(traj[k], K, d, rts[c], M[c][k].shape)
                    ious.append((s & M[c][k]).sum() / max((s | M[c][k]).sum(), 1))
            u_sim = -cen[:, 2]
            rmse_post = float(np.sqrt(np.mean((u_sim[hit:n] - obs_u[hit:n]) ** 2)) * 100)
            res[tag] = {'log10_E_over_rho_x1000': float(lg), 'centroid_err_cm_mean': float(err.mean()), 'post_hit_up_rmse_cm': rmse_post,
                        'mask_iou_mean': float(np.mean(ious)), 'loss': rmse_post + (1 - float(np.mean(ious))),
                        'bounce_sim_20frames': bounce_metrics(u_sim[:n], hit), 'bounce_sim_full': bounce_metrics(u_sim, hit),
                        'sim_up_cm': (u_sim * 100).round(2).tolist(), 'sim_info': info, 'seconds': time.time() - t0}
            np.save(out / f'{obj}_{tag}_traj.npy', traj[:: 1].astype(np.float32)[:, ::4])  # subsampled particles for figures
            print(obj, tag, round(lg, 2), {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[tag].items() if k in ('post_hit_up_rmse_cm', 'mask_iou_mean', 'centroid_err_cm_mean')},
                  (res[tag]['bounce_sim_20frames'] or {}).get('restitution_e'), flush=True)
        ok = {k: v for k, v in res.items() if 'loss' in v and k.startswith('grid')}
        best = min(ok, key=lambda k: ok[k]['loss']) if ok else None
        report[obj] = {'hit_frame': hit, 'ground_up_m': ground_u, 'hull_particles': int(len(X0)),
                       'hull_extent_cm': ((X0.max(0) - X0.min(0)) * 100).round(2).tolist(),
                       'init_velocity_world_mps': v0.tolist(), 'observed_up_cm': (obs_u * 100).round(2).tolist(),
                       'observed_bounce': obs, 'candidates': res, 'refined_best_grid': best}
        np.save(out / f'{obj}_hull0.npy', X0.astype(np.float32))
        (out / 'replay.json').write_text(json.dumps(report, indent=1, default=float))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('zip'); p.add_argument('prepared'); p.add_argument('pred'); p.add_argument('out')
    p.add_argument('--objs', nargs='+', default=['bun', 'burger', 'dog', 'pig', 'potato'])
    main(p.parse_args())
