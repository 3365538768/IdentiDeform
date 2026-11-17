"""V3l Part 2 (stretch): our MPM driven by PhysTwin's released controller (hand) trajectory on single_push_sloth.

  scripts/v3f/replay_env.sh scripts/v3l/pt_mpm.py <case_dir> --mode search|baselines [--budget 12] [--out data/v3l/pt]

Geometry : PhysTwin vertex layout = object_points[0] (4829, observed, frame-0) + surface_points (535) + interior_points (906)
           (all released, frame 0).  These 6270 points are MASSLESS TRACERS (volume 1e-3 h^3) advected by the MPM grid;
           the mass is carried by filler particles on an h = dx/2 voxel grid: voxels containing any of the 6270 points,
           morphologically closed (dilate 2 / fill holes / erode 1) -> one particle per voxel centre, volume h^3.
Engine   : our mpm_solver_warp jelly (fixed corotated), rho 1000 assumed, nu 0.3 fixed, grid_lim 1.0 / n_grid 140
           (dx 7.1 mm), grid_v_damping 0.99999, CFL dt; gravity +z_world (PhysTwin reverse_z), ground plane z_world = 0
           (slip, friction 0.3 = PhysTwin collide_fric); world -> sim: (x+0.3, y+0.3, -z+0.1).
Boundary : moving kinematic 'sticky' boundary: filler particles within R_CTRL = 0.02 m of any controller point at frame 0
           are attached to the controller-point centroid translation C(t) (linear in time within a frame): before every
           substep their x is set to x0 + C(t) - C(0) and v to dC/dt.  (PhysTwin instead connects controller points to object
           points within 0.04 m by learned springs.)
Search   : 1-D compass search on log10 E exactly as v3k-A (start, step 0.5 halved on failure, min 0.02, bounds [2.5, 7.0]),
           objective = PhysTwin train Chamfer (frames 1..46, single-direction L1 NN of visible GT object points to predicted
           object+surface points; cKDTree p=1 == pytorch3d chamfer_distance(single_directional, norm=1)); budget 12.
           Start = V2 train mean of the jelly law (our RGB-video prior is NOT applied: the released processed case has no
           RGB frames; see RESULTS).  Each simulation runs all 68 frames; ONLY frames 0..46 enter the objective.
Output   : vertices [68, 6270, 3] (world) of the best E for scripts/v3l/pt_eval.py (verbatim PhysTwin evaluation).
"""
import argparse, json, math, os, pickle, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v3k'))
import a_sg_refine as A  # noqa: E402  (sim paths, numpy, torch)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy import ndimage  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402

GRID_LIM, N_GRID = 1.0, 140
SHIFT = np.array([0.3, 0.3, 0.1])
R_CTRL = 0.02
NU, RHO, FPS = 0.3, 1000.0, 30.0
BOUNDS = (2.5, 7.0)
OK = {('numpy.core.multiarray', '_reconstruct'), ('numpy', 'ndarray'), ('numpy', 'dtype'), ('numpy.core.multiarray', 'scalar'),
      ('numpy._core.multiarray', '_reconstruct'), ('numpy._core.multiarray', 'scalar')}


class R(pickle.Unpickler):
    def find_class(self, m, n):
        if (m, n) in OK:
            return super().find_class(m, n)
        raise pickle.UnpicklingError(f'blocked {m}.{n}')


def rload(p):
    with open(p, 'rb') as f:
        return R(f).load()


def to_sim(X):
    return np.stack([X[:, 0] + SHIFT[0], X[:, 1] + SHIFT[1], -X[:, 2] + SHIFT[2]], 1)


def to_world(S):
    return np.stack([S[:, 0] - SHIFT[0], S[:, 1] - SHIFT[1], -(S[:, 2] - SHIFT[2])], 1)


def build(case):
    d = rload(Path(case) / 'final_data.pkl')
    split = json.loads((Path(case) / 'split.json').read_text())
    tracer = np.concatenate([d['object_points'][0], d['surface_points'], d['interior_points']]).astype(np.float64)
    dx = GRID_LIM / N_GRID; h = dx / 2
    S = to_sim(tracer); lo = S.min(0) - 4 * h
    ijk = np.floor((S - lo) / h).astype(int); shape = ijk.max(0) + 5
    occ = np.zeros(shape, bool); occ[tuple(ijk.T)] = True
    occ = ndimage.binary_dilation(occ, iterations=2); occ = ndimage.binary_fill_holes(occ); occ = ndimage.binary_erosion(occ, iterations=1)
    filler = lo + (np.argwhere(occ) + 0.5) * h
    filler = filler[filler[:, 2] >= SHIFT[2] + 0.25 * h]  # nothing below the ground plane
    ctrl = d['controller_points'].astype(np.float64)  # [T,30,3] world
    C = ctrl.mean(1)  # centroid translation
    dist, _ = cKDTree(to_sim(ctrl[0])).query(filler)
    driven = np.where(dist < R_CTRL)[0]
    return {'d': d, 'split': split, 'tracer': tracer, 'filler': filler, 'driven': driven, 'C': C, 'h': h, 'dx': dx}


def obj_chamfer(V, d, frames, n_surf):
    """PhysTwin chamfer (single direction, L1) per frame, numpy/cKDTree version for the search objective."""
    out = []
    for f in frames:
        gt = d['object_points'][f][d['object_visibilities'][f]]
        dd, _ = cKDTree(V[f][:n_surf]).query(gt, p=1)
        out.append(float(dd.mean()))
    return out


_WP = None


def simulate(B, E, T, device='cuda:0'):
    global _WP
    import warp as wp
    if _WP is None:
        wp.init(); _WP = wp
    from mpm_solver_warp import MPM_Simulator_WARP
    nt = len(B['tracer']); Sf = B['filler']; St = to_sim(B['tracer'])
    X = torch.tensor(np.concatenate([St, Sf]), dtype=torch.float32, device=device)
    vol = torch.cat([torch.full((nt,), 1e-3 * B['h'] ** 3), torch.full((len(Sf),), B['h'] ** 3)]).float().to(device)
    sim = MPM_Simulator_WARP(10, device=device)
    sim.load_initial_data_from_torch(X, vol, None, n_grid=N_GRID, grid_lim=GRID_LIM, device=device)
    sim.set_parameters_dict({'material': 'jelly', 'E': float(E), 'nu': NU, 'density': RHO, 'g': [0.0, 0.0, -9.8],
                             'rpic_damping': 0.0, 'grid_v_damping_scale': 0.99999, 'n_grid': N_GRID, 'grid_lim': GRID_LIM}, device=device)
    sim.add_bounding_box()
    sim.add_surface_collider([GRID_LIM / 2, GRID_LIM / 2, SHIFT[2]], [0.0, 0.0, 1.0], surface='slip', friction=0.3)
    sim.finalize_mu_lam(device=device)
    sim.import_particle_v_from_torch(torch.zeros_like(X), device=device)
    lam_2mu = E * NU / ((1 + NU) * (1 - 2 * NU)) + E / (1 + NU)
    c = math.sqrt(lam_2mu / RHO)
    dt = min(1e-4, 0.25 * B['dx'] / max(c, 1e-3))
    dtf = 1.0 / FPS
    steps = max(1, int(math.ceil(dtf / dt))); dt = dtf / steps
    idx = torch.tensor(B['driven'] + nt, dtype=torch.long, device=device)
    x0d = X[idx].clone()
    Csim = torch.tensor(B['C'] * np.array([1, 1, -1]), dtype=torch.float32, device=device)  # translation in sim axes
    out = [B['tracer'].copy()]; step = 0
    for f in range(1, T):
        vel = (Csim[f] - Csim[f - 1]) * FPS
        for s in range(steps):
            tt = s / steps
            xs = sim.export_particle_x_to_torch(); vs = sim.export_particle_v_to_torch()
            wp.synchronize()
            xs[idx] = x0d + (Csim[f - 1] + tt * (Csim[f] - Csim[f - 1]) - Csim[0])
            vs[idx] = vel
            torch.cuda.synchronize()
            sim.p2g2p(step, dt, device=device); step += 1
        Sx = sim.export_particle_x_to_torch()[:nt].detach().cpu().numpy()
        if not np.isfinite(Sx).all() or np.abs(Sx).max() > 10:
            return None, {'dt': dt, 'failed_frame': f}
        out.append(to_world(Sx.astype(np.float64)))
    return np.stack(out), {'dt': dt, 'steps_per_frame': steps}


def main(a):
    B = build(a.case); d = B['d']; T = B['split']['frame_len']; tr = B['split']['train'][1]
    n_surf = d['object_points'].shape[1] + d['surface_points'].shape[0]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sp = out / (f'search_{a.start}.json' if a.mode == 'search' else f'{a.mode}.json')
    state = json.loads(sp.read_text()) if sp.exists() else {'evals': []}

    def save():
        tmp = sp.with_suffix('.tmp'); tmp.write_text(json.dumps(state, indent=1)); os.replace(tmp, sp)
    state.update(n_tracer=int(len(B['tracer'])), n_filler=int(len(B['filler'])), n_driven=int(len(B['driven'])), dx=B['dx'],
                 r_ctrl=R_CTRL, train_frames=[1, tr], test_frames=[tr, T])
    save()
    print('build', state['n_tracer'], state['n_filler'], state['n_driven'], flush=True)
    if a.mode == 'baselines':
        V0 = np.stack([B['tracer']] * T)
        Vr = np.stack([B['tracer'] + (B['C'][f] - B['C'][0]) for f in range(T)])
        np.save(out / 'vertices_static.npy', V0.astype(np.float32)); np.save(out / 'vertices_rigid_follow.npy', Vr.astype(np.float32))
        state['complete'] = True; save(); return
    if a.mode == 'smoke':
        t0 = time.time(); V, info = simulate(B, 10 ** a.log10E, 6)
        state['smoke'] = {'seconds': time.time() - t0, 'info': info, 'ok': V is not None,
                          'cham': obj_chamfer(V, d, range(1, 6), n_surf) if V is not None else None}
        save(); print(state['smoke']); return
    if a.start == 'prior':
        pr = json.loads((out / 'pt_prior.json').read_text())
        x0 = round(float(np.clip(pr['log10_E_Pa_at_rho1000'], *BOUNDS)), 6)
        state.update(x0=x0, x0_source='our 9-run K0 prior on PhysTwin RGB views, train frames 0-46 (data/v3l/pt/pt_prior.json); nu kept 0.3')
    else:
        jm, _, _ = A.jelly_train_mean()
        x0 = round(float(np.clip(jm, *BOUNDS)), 6); state.update(x0=x0, x0_source='V2 train mean of the jelly law (uninformed)')
    save()
    trajs = {}

    def evaluate(x):
        x = round(float(x), 6)
        for e in state['evals']:
            if e['x'] == x:
                return e
        if len(state['evals']) >= a.budget:
            return None
        t0 = time.time()
        try:
            V, info = simulate(B, 10 ** x, T)
        except Exception as ex:  # noqa: BLE001
            V, info = None, {'exception': repr(ex)[:300]}
        if V is None:
            e = {'x': x, 'failed': True, 'loss': 9.0, 'info': info}
        else:
            ch = obj_chamfer(V, d, range(1, tr), n_surf)
            e = {'x': x, 'loss': float(np.mean(ch)), 'info': info}
            np.save(out / f'vertices_{a.start}_{x:.6f}.npy', V.astype(np.float32))
        e.update(n=len(state['evals']) + 1, seconds=time.time() - t0)
        state['evals'].append(e); save()
        print('eval', e['n'], x, round(e['loss'], 5), round(e['seconds'], 1), flush=True)
        return e

    best = evaluate(x0); step = A.STEP0; last = None
    while best is not None and len(state['evals']) < a.budget and step >= A.MIN_STEP:
        improved = False
        for s in ([last] if last else []) + [dd for dd in (+1, -1) if dd != last]:
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
    ok = [e for e in state['evals'] if not e.get('failed')]
    b = min(ok, key=lambda e: e['loss'])
    state['best'] = {'log10E': b['x'], 'train_chamfer_objective': b['loss'], 'vertices': f'vertices_{a.start}_{b["x"]:.6f}.npy'}
    state['budget0'] = {'log10E': x0, 'vertices': f'vertices_{a.start}_{x0:.6f}.npy'}
    state['search_seconds'] = float(sum(e['seconds'] for e in state['evals']))
    state['complete'] = True; save()
    print('best', state['best'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('case')
    p.add_argument('--mode', choices=('search', 'baselines', 'smoke'), required=True)
    p.add_argument('--budget', type=int, default=12); p.add_argument('--start', choices=('jellymean', 'prior'), default='jellymean'); p.add_argument('--log10E', type=float, default=5.0)
    p.add_argument('--out', default=str(ROOT / 'data/v3l/pt'))
    main(p.parse_args())
