"""V3k-A: cross-dataset rescue by in-target simulation refinement on the Spring-Gaus MPM synthetic scenes.

  scripts/v3f/replay_env.sh scripts/v3k/a_sg_refine.py <mpm_synthetic.zip> <scene> --arm prior|jellymean|diag --out data/v3k/sg

Data (untrusted, read-only; only PLY/PNG/JSON decoded from the zip, nothing executed, python -I):
  geometry  = the scene's released simulation point cloud at frame 0 (simulation/<scene>/000.ply, x y z r g b),
              randomly subsampled (seed 0) to <= MAX_PARTICLES particles.
  initial v = physical.json INIT_VELOCITY, gravity -9.8 along +y-up, ground plane y = BC.GROUND (slip, friction 0.3).
  observed  = the 3 canonical-triplet (v3f 't0') cameras, 30 RGBA frames (dt 0.03 s), alpha-composited on black,
              resized 512 -> RES px (INTER_AREA); mask = alpha > 0.5.
Engine: our mpm_solver_warp (V2 data-generation solver), jelly = fixed corotated, rho = 1000 (assumed, as v3f),
  nu fixed (prior arm: ensemble-predicted nu; jellymean arm: train-mean jelly nu), rpic 0, grid_v_damping 0.99999,
  grid_lim 1.2 / n_grid 120 (dx = 1 cm in SG units), CFL dt.  Spring-Gaus world (y up) -> sim (x+0.1, z+0.1, y+0.1).
Render: particles projected with the released pinhole K / Blender c2w (camera looks along -z), z-buffered disk splats
  of radius r px (r chosen per scene from {0,1,2} maximising the frame-0 IoU of the GT particles: geometry only).
Objective (as V3b C1): L = (1 - mean mask IoU) + RGB-MAE(union)/255 over 3 views x 30 frames.
Search (as V3b C1, 1-D since elastic): deterministic opportunistic compass search on log10(E/rho*1000), step 0.5
  decade halved after an unsuccessful poll (min 0.02), bounds [1.5, 6.5]; every simulation counts; budget k = best of
  the first k sims (nested), k = 0 is the start point.  Failed (non-finite / crashed) sims: loss 9, counted, never best.
Arms: prior     = 9-run ensemble zero-shot (K0, canonical triplet) log10 E and nu from docs/v3f/SPRINGGAUS_SYN.json
      jellymean = uninformed: V2 train mean of the jelly law (E/rho, nu)
      diag      = (not an estimator) GT-E/GT-nu replay in our engine, renderer upper bound (GT SG particles of every
                  frame rendered), and a coarse loss landscape log10 E in {3.5,...,6.5} step 0.5 with prior nu.
"""
import argparse, io, json, math, os, sys, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SIM = ROOT / 'upstream/DeformFieldBench/simulation'
sys.path[:0] = [str(SIM / 'mpm_solver_warp'), str(SIM), str(ROOT / '.engine-envs/gsim/lib/python3.12/site-packages'),
                str(ROOT / '.venv/lib/python3.12/site-packages')]
import numpy as np  # noqa: E402
import cv2  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

PREFIX = 'mpm_synthetic/'
STATS = json.loads((ROOT / 'data/manifests/v2main.specific.stats.json').read_text())
RES = 128
MAX_PARTICLES = 150000
OFF, GRID_LIM, N_GRID = 0.1, 1.2, 120
G = 9.8
BOUNDS = (1.5, 6.5)
STEP0, MIN_STEP = 0.5, 0.02
FAILED_LOSS = 9.0
LANDSCAPE = [3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5]


def read_ply(z, scene, f):
    b = z.read(f'{PREFIX}simulation/{scene}/{f:03d}.ply'); i = b.find(b'end_header') + len('end_header') + 1
    head = b[:i].decode(errors='replace')
    assert 'property float red' in head and 'binary_little_endian' in head, head
    return np.frombuffer(b[i:], np.float32).reshape(-1, 6).copy()


def load_scene(zpath, scene):
    z = zipfile.ZipFile(zpath)
    syn = json.loads((ROOT / 'docs/v3f/SPRINGGAUS_SYN.json').read_text())
    views = syn['protocol']['views'][scene]['triplets'][0]
    phys = json.loads(z.read(f'{PREFIX}render/{scene}/physical.json'))
    cams = json.loads(z.read(f'{PREFIX}render/{scene}/camera.json'))
    fr = json.loads(z.read(f'{PREFIX}render/{scene}/frame.json'))
    times = np.array([list(d.values())[0] for d in fr], np.float64)
    masks, rgbs = [], []
    for v in views:
        mv, rv = [], []
        for f in range(len(times)):
            im = np.asarray(Image.open(io.BytesIO(z.read(f'{PREFIX}render/{scene}/{cams[v]["camera"]}/{f:03d}.png'))).convert('RGBA'), np.float32) / 255
            small = cv2.resize(im, (RES, RES), interpolation=cv2.INTER_AREA)
            mv.append(small[..., 3] > 0.5); rv.append(small[..., :3] * small[..., 3:4] * 255)
        masks.append(np.stack(mv)); rgbs.append(np.stack(rv))
    P = read_ply(z, scene, 0)
    rng = np.random.RandomState(0)
    keep = np.sort(rng.choice(len(P), MAX_PARTICLES, replace=False)) if len(P) > MAX_PARTICLES else np.arange(len(P))
    cam = []
    for v in views:
        K = np.array(cams[v]['K'], np.float64) * np.array([[RES / 512], [RES / 512], [1]])
        cam.append((K, np.linalg.inv(np.array(cams[v]['c2w'], np.float64))))
    return {'z': z, 'phys': phys, 'views': views, 'times': times, 'obs': (np.stack(masks), np.stack(rgbs).astype(np.float32)),
            'P': P, 'keep': keep, 'cams': cam, 'n_particles_total': int(len(P))}


def render(X, col, cams, r):
    """X [N,3] SG world, col [N,3] 0-255 -> (mask [V,H,W] bool, rgb [V,H,W,3] float)."""
    offs = [(du, dv) for du in range(-r, r + 1) for dv in range(-r, r + 1) if du * du + dv * dv <= r * r + r]
    ms, rs = [], []
    Xh = np.concatenate([X, np.ones((len(X), 1))], 1)
    for K, w2c in cams:
        Xc = Xh @ w2c.T
        zc = -Xc[:, 2]; ok = zc > 1e-4
        u = K[0, 0] * Xc[:, 0] / np.where(ok, zc, 1) + K[0, 2]
        v = -K[1, 1] * Xc[:, 1] / np.where(ok, zc, 1) + K[1, 2]
        order = np.argsort(-zc)  # far first; later writes (nearer) win
        order = order[ok[order]]
        ui = np.floor(u[order]).astype(np.int64); vi = np.floor(v[order]).astype(np.int64); c = col[order]
        img = np.zeros((RES, RES, 3), np.float32); m = np.zeros((RES, RES), bool)
        U = np.concatenate([ui + du for du, dv in offs]); V = np.concatenate([vi + dv for du, dv in offs])
        rank = np.concatenate([np.arange(len(ui))] * len(offs)); C = np.concatenate([c] * len(offs))
        s = np.argsort(rank, kind='stable'); U, V, C = U[s], V[s], C[s]
        inb = (U >= 0) & (U < RES) & (V >= 0) & (V < RES)
        U, V, C = U[inb], V[inb], C[inb]
        m[V, U] = True
        img[V, U] = C  # numpy assigns in order: the last (nearest) write wins
        ms.append(m); rs.append(img)
    return np.stack(ms), np.stack(rs)


def compare(ref, sim):
    (ma, ra), (mb, rb) = ref, sim
    t = min(ma.shape[1], mb.shape[1]); ma, ra, mb, rb = ma[:, :t], ra[:, :t], mb[:, :t], rb[:, :t]
    inter = (ma & mb).sum((-1, -2)); union = (ma | mb).sum((-1, -2))
    iou = np.where(union > 0, inter / np.maximum(union, 1), 1.0)
    u = ma | mb
    rgb = float((np.abs(ra - rb).mean(-1) * u).sum() / max(u.sum(), 1))
    return {'iou': float(iou.mean()), 'iou_last': float(iou[:, -1].mean()), 'rgb_mae_union': rgb, 'loss': float(1 - iou.mean() + rgb / 255.)}


def render_traj(traj, col, cams, r):
    out = [render(X, col, cams, r) for X in traj]  # list over T of ([V,H,W],[V,H,W,3])
    return np.stack([o[0] for o in out], 1), np.stack([o[1] for o in out], 1)


_WP = None


def simulate(X0, v0, ground_y, E, nu, n_frames, dt_frame, device='cuda:0'):
    global _WP
    import warp as wp
    if _WP is None:
        wp.init(); _WP = wp
    from mpm_solver_warp import MPM_Simulator_WARP
    to_sim = lambda X: np.stack([X[:, 0] + OFF, X[:, 2] + OFF, X[:, 1] + OFF], 1)
    to_world = lambda S: np.stack([S[:, 0] - OFF, S[:, 2] - OFF, S[:, 1] - OFF], 1)
    S = torch.tensor(to_sim(X0), dtype=torch.float32, device=device)
    dx = GRID_LIM / N_GRID
    vol = torch.full((len(S),), (dx / 2) ** 3, dtype=torch.float32, device=device)
    sim = MPM_Simulator_WARP(10, device=device)
    sim.load_initial_data_from_torch(S, vol, None, n_grid=N_GRID, grid_lim=GRID_LIM, device=device)
    sim.set_parameters_dict({'material': 'jelly', 'E': float(E), 'nu': float(nu), 'density': 1000.0, 'g': [0.0, 0.0, -G],
                             'rpic_damping': 0.0, 'grid_v_damping_scale': 0.99999, 'n_grid': N_GRID, 'grid_lim': GRID_LIM}, device=device)
    sim.add_bounding_box()
    sim.add_surface_collider([GRID_LIM / 2, GRID_LIM / 2, ground_y + OFF], [0.0, 0.0, 1.0], surface='slip', friction=0.3)
    sim.finalize_mu_lam(device=device)
    v = torch.tensor([v0[0], v0[2], v0[1]], dtype=torch.float32, device=device).expand(len(S), 3).contiguous()
    sim.import_particle_v_from_torch(v, device=device)
    lam_2mu = E * nu / ((1 + nu) * (1 - 2 * nu)) + E / (1 + nu)
    c = math.sqrt(lam_2mu / 1000.)
    dt = min(1e-4, 0.25 * dx / max(c, 1e-3))
    steps = max(1, int(math.ceil(dt_frame / dt))); dt = dt_frame / steps
    out = [X0.copy()]; step = 0
    for f in range(1, n_frames):
        for _ in range(steps):
            sim.p2g2p(step, dt, device=device); step += 1
        Sx = sim.export_particle_x_to_torch().detach().cpu().numpy()
        if not np.isfinite(Sx).all() or np.abs(Sx).max() > 10:
            return None, {'dt': dt, 'failed_frame': f}
        out.append(to_world(Sx))
    return np.stack(out), {'dt': dt, 'steps_per_frame': steps}


def zE(log10E):
    return (math.log1p(10 ** log10E) - STATS['mean'][0]) / STATS['std'][0]


def log10E_from_z(z):
    return math.log10(math.expm1(z * STATS['std'][0] + STATS['mean'][0]))


def nu_from_z(z):
    return z * STATS['std'][1] + STATS['mean'][1]


def jelly_train_mean():
    s, n = np.zeros(2), 0
    for r in map(json.loads, (ROOT / 'data/manifests/v2main.jsonl').read_text().splitlines()):
        if r['split'] != 'train' or r['law_id'] != 0:
            continue
        p = r['parameters']; s += [(math.log1p(p['E'] / (p['rho'] / 1000.)) - STATS['mean'][0]) / STATS['std'][0], p['nu']]; n += 1
    zE_, nu = s / n
    return log10E_from_z(zE_), float(nu), n


def main(a):
    sc = load_scene(a.zip, a.scene)
    out = Path(a.out) / a.scene; out.mkdir(parents=True, exist_ok=True)
    state_path = out / f'{a.arm}.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'evals': []}
    def save():
        tmp = state_path.with_suffix('.tmp'); tmp.write_text(json.dumps(state, indent=1)); os.replace(tmp, state_path)
    P = sc['P'][sc['keep']]; X0, col = P[:, :3].astype(np.float64), P[:, 3:].astype(np.float32)
    phys = sc['phys']; v0 = np.array(phys[1]['INIT_VELOCITY'], np.float64); gy = float(phys[1]['BC']['GROUND'][0][1])
    assert phys[1]['BC']['GROUND'][1] == [0, 1, 0] and phys[1]['GRAVITY'] == [0, -9.8, 0], phys[1]
    T = len(sc['times']); dtf = float(sc['times'][1] - sc['times'][0])
    obs = sc['obs']
    # splat radius from frame-0 geometry only
    if 'splat_radius' not in state:
        f0 = {r: compare((obs[0][:, :1], obs[1][:, :1]), tuple(x[:, None] for x in render(X0, col, sc['cams'], r)))['iou'] for r in (0, 1, 2)}
        state['splat_radius'] = max(f0, key=f0.get); state['frame0_iou_by_radius'] = f0; save()
    r = state['splat_radius']
    gtE = math.log10(float(phys[0]['E'])) if isinstance(phys[0].get('E'), (int, float)) else None
    syn = json.loads((ROOT / 'docs/v3f/SPRINGGAUS_SYN.json').read_text())['ensemble9']
    zp = syn['K0']['z_t0'].get(a.scene)
    prior_log10E, prior_nu = (log10E_from_z(zp[0]), float(nu_from_z(zp[1]))) if zp else (None, None)
    if a.prior_json:  # V3s: prior from scripts/v3s/prior_v3.py output (30 frames, same t0 triplet / K0 path)
        pj = json.loads(Path(a.prior_json).read_text())['scenes'][a.scene]['f30_recomputed']
        prior_log10E, prior_nu = float(pj['log10_E_Pa_at_rho1000']), float(pj['nu'])
    jm_log10E, jm_nu, n_jelly = jelly_train_mean()
    state.update(scene=a.scene, arm=a.arm, gt_log10E=gtE, gt_nu=float(phys[0]['NU']) if gtE is not None else None,
                 prior_log10E=prior_log10E, prior_nu=prior_nu, jellymean_log10E=jm_log10E, jellymean_nu=jm_nu, n_train_jelly=n_jelly,
                 n_particles_total=sc['n_particles_total'], n_particles_sim=int(len(X0)), views=sc['views'], res=RES,
                 init_velocity=v0.tolist(), ground_y=gy, frames=T, frame_dt=dtf)
    save()

    def run_sim(tag, log10E, nu):
        pend = out / f'{a.arm}.pending.json'
        if pend.exists():  # previous attempt crashed hard inside this simulation: deterministic failure
            prev = json.loads(pend.read_text())
            if prev.get('tag') == tag:
                pend.unlink(); return {'failed': True, 'crashed': True, 'loss': FAILED_LOSS, 'seconds': prev.get('seconds', 0)}
        pend.write_text(json.dumps({'tag': tag, 'started': time.time()}))
        t0 = time.time()
        try:
            traj, info = simulate(X0, v0, gy, 10 ** log10E, nu, T, dtf)
        except Exception as e:  # noqa: BLE001
            traj, info = None, {'exception': repr(e)[:300]}
        if traj is None:
            res = {'failed': True, 'loss': FAILED_LOSS, 'info': info}
        else:
            res = {**compare(obs, render_traj(traj, col, sc['cams'], r)), 'info': info,
                   'centroid_y': traj.mean(1)[:, 1].round(4).tolist()}
        res['seconds'] = time.time() - t0
        pend.unlink(missing_ok=True)
        print(a.scene, a.arm, tag, round(log10E, 4), {k: res.get(k) for k in ('iou', 'rgb_mae_union', 'loss')}, round(res['seconds'], 1), flush=True)
        return res

    if a.arm == 'diag':
        if 'renderer_upper_bound' not in state:
            traj = np.stack([read_ply(sc['z'], a.scene, f)[sc['keep'], :3] for f in range(T)]).astype(np.float64)
            state['renderer_upper_bound'] = compare(obs, render_traj(traj, col, sc['cams'], r))
            state['observed_centroid_y'] = traj.mean(1)[:, 1].round(4).tolist(); save()
        if gtE is not None and 'gt_replay' not in state:
            state['gt_replay'] = {'log10E': gtE, 'nu': state['gt_nu'], **run_sim('gt', gtE, state['gt_nu'])}; save()
        state.setdefault('landscape', {})
        nu_l = prior_nu if prior_nu is not None else jm_nu
        for g in LANDSCAPE:
            if str(g) not in state['landscape']:
                state['landscape'][str(g)] = {'nu': nu_l, **run_sim(f'land_{g}', g, nu_l)}; save()
        state['complete'] = True; save(); return

    x0, nu = (prior_log10E, prior_nu) if a.arm == 'prior' else (jm_log10E, jm_nu)
    x0 = round(float(np.clip(x0, *BOUNDS)), 6)
    state.update(x0=x0, nu=nu); save()

    def evaluate(x):
        x = round(float(x), 6)
        for e in state['evals']:
            if e['x'] == x:
                return e
        if len(state['evals']) >= a.budget:
            return None
        res = run_sim(f'eval_{len(state["evals"]):02d}', x, nu)
        e = {'n': len(state['evals']) + 1, 'x': x, **{k: v for k, v in res.items() if k != 'centroid_y'}}
        state['evals'].append(e); save()
        return e

    best = evaluate(x0); step = STEP0; last = None
    while best is not None and len(state['evals']) < a.budget and step >= MIN_STEP:
        improved = False
        for s in ([last] if last else []) + [d for d in (+1, -1) if d != last]:
            y = round(float(np.clip(best['x'] + s * step, *BOUNDS)), 6)
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
    budgets = {}
    for k in range(0, a.budget + 1):
        n = min(k, len(state['evals']))
        x = x0 if n == 0 else min(state['evals'][:n], key=lambda e: e['loss'])['x']
        b = {'sims': n, 'log10E': x}
        if gtE is not None:
            b.update(abs_dlog10E=abs(x - gtE), abs_dz=abs(zE(x) - zE(gtE)))
        fit = next((e for e in state['evals'][:max(n, 1)] if e['x'] == x), None) if n else None
        b['observed_fit'] = {kk: fit.get(kk) for kk in ('iou', 'rgb_mae_union', 'loss')} if fit else None
        budgets[str(k)] = b
    state['budgets'] = budgets; state['complete'] = True; save()
    print('done', a.scene, a.arm, {k: round(v['log10E'], 3) for k, v in budgets.items() if k in ('0', '4', '12')}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('zip'); p.add_argument('scene')
    p.add_argument('--arm', choices=('prior', 'jellymean', 'diag'), required=True)
    p.add_argument('--budget', type=int, default=12); p.add_argument('--out', default=str(ROOT / 'data/v3k/sg'))
    p.add_argument('--prior-json', default=None)
    main(p.parse_args())
