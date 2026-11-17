"""V3 contact setup (FB_BC_FIX=1). Every action acts on the object through rigid bodies that start outside it and
touch it (as in the gripper task, scripts/v3e/squeeze_main.py); nothing grabs the object's own particles at t = 0.

  press     : a rigid, frictionless plate above the object moves down by 70 % of the height over 0-3 s, then lifts
              back over 3-4 s; it only pushes (one-sided contact), so the object springs back following it instead of
              being launched when the plate disappears. The object rests on a support bed shaped to its underside.
  press-hold-release (FB_ACTION=tap30hold): the same plate presses 30 % over 0-0.5 s, holds until 1.0 s, lifts back
              over 1.0-1.5 s and is then gone (free motion until 4 s).
  FB_RELEASE=fast: the plate lifts away in 0.1 s and the support bed is sticky (the whole underside is held), so the
              object springs back and rings freely without jumping off.
  shear     : the underside is held by a sticky support bed; a parallel-jaw gripper closes on the top 20 % of the
              object, holds, then drags it horizontally (25 % of the object's extent), and opens at 3 s.
  stretch   : a fixed gripper holds one end of the object's long axis and a second gripper pulls the other end
              outwards (15 % of the length), like a tensile tester.
  bend      : fixed grippers clamp both ends of the long axis and a plate pushes the middle sideways
              (35 % of the thickness).
  long axis : the longest axis, unless an end segment there is too thin to grip (< 6 cells); then the longer
              horizontal axis.
  drop      : upstream drop, floor FB_DROP_GAP below the object.
  press with fixed force (FB_PRESS_FORCE=F): the same one-sided plate, but force-controlled like a robot arm in
              admittance mode: the object's reaction R on the plate is measured from the momentum the plate removes
              (mean over 20 substeps), and the plate speed is set to v_max * (F - R) / F (capped at v_max = 70 % of the height per
              3 s; travel capped at 70 % of the height). It presses until 3 s, then lifts back over 1 s. With a fixed
              force the deformation depends on E itself, not only on E/rho.
  grippers  : two rigid sticky jaws placed just outside an object segment, sized to that segment's local cross-section;
              they close by 15 % of the local thickness per side over 0-0.5 s, hold until 0.75 s, then move or stay.
  support bed: for every grid column under the object the support rises to just below the object's lowest point in
              that column, so feet, rims and overhangs are carried and the object cannot tip or sink. Columns without
              the object are plain floor. Sliding is allowed for press (only downward motion is blocked); shear's bed
              is sticky. Raised columns (under overhangs) are removed for good once the press plate comes down to
              their height, so they never squeeze the object against the plate.
Weak-action redesign (FB_WEAK=hold, with gravity): the object is held up by one gripper like a robot arm would.
  cantilever (replaces bend): a gripper closes on one end of the long horizontal axis; the support bed is removed at
              0.75 s, so the free part sags under its own weight and oscillates (deflection ~ rho g L^4 / (E I)).
  hang (replaces stretch): a gripper closes on the top 25 % of the object; at 0.75 s the support is removed and the
              gripper lifts by 6 % of the height over 0.5 s and holds, so the object hangs and stretches under its weight.
  shake (replaces shear): cantilever hold, then at 1.0 s the gripper steps sideways by 15 % of the object depth over
              0.25 s and stops; the free end swings (frequency ~ sqrt(E / rho) / L).
  Grip: outer 25 % of the length, jaws close 30 % of the local thickness per side (firm enough to carry the
  object's weight). The floor is 0.2 below the object in these actions.
Contact forces (FB_FORCES=1, default on): for every rigid body (gripper jaw, plate, pusher, support plate; the press
plate included) the force it exerts on the object is measured every substep as the momentum it imposes on the grid
nodes it covers, m (v_body - v) / dt, summed over its box, sampled every 4th substep and averaged into 10 ms bins. Written at exit to
<output_path>/contact_forces.npz: t [s] (bin centres), F [n_bins, n_bodies, 3] (force on the object, simulation units:
kg m s^-2 with the config's density in kg/m^3 and lengths in grid units), F_plate [n_bins] (vertical force of the press
plate on the object), bodies (JSON: point, size, velocity, start/end time, in creation order = order of the cuboid
boundary conditions in contact_bc.json).
Stray particles outside the grid are moved inside before the simulation starts (they made p2g index out of bounds).
Gravity is set in the config (scripts/v3n/gen_v3.py)."""
import os
import numpy as np
import my_utils.sim_utils as su

T, T_CLOSE, T_HOLD = 3.0, 0.5, 0.75
BED = {}
PLATE = {}


def _geom(mpm_init_pos, mp, prefix):
    pos = su._filter_points_for_bbox(mpm_init_pos.detach().cpu().numpy(), mp, prefix=prefix)
    g = float(mp.get('grid_lim', 2.0)); dx = g / float(mp.get('n_grid', 150))
    return pos, g, dx, pos.min(0), pos.max(0)


def _floor(lo, hi, g, surface='slip', dx=2.0 / 150):
    return {'type': 'surface_collider', 'point': [float(0.5 * (lo[0] + hi[0])), float(0.5 * (lo[1] + hi[1])), float(lo[2] - 0.25 * dx)],
            'normal': [0.0, 0.0, 1.0], 'surface': surface, 'friction': 0.5 if surface == 'slip' else 0.0, 'start_time': 0.0, 'end_time': 1e3}


def _plate(center, half, vel, t0=0.0, t1=T):
    return {'type': 'cuboid', 'point': [float(c) for c in center], 'size': [float(s) for s in half],
            'velocity': [float(v) for v in vel], 'start_time': float(t0), 'end_time': float(t1), 'reset': 0}


def _set_bed(mpm_init_pos, mp, sticky):
    """Height of the support top for every grid column: just below the object's lowest particle in that column."""
    x = mpm_init_pos.detach().cpu().numpy(); g = float(mp.get('grid_lim', 2.0)); n = int(mp.get('n_grid', 150)); dx = g / n
    pos, *_ = _geom(mpm_init_pos, mp, 'press')
    H = np.full((n + 1, n + 1), float(pos[:, 2].min()) - 0.5 * dx, np.float32)  # plain floor
    keep = (x[:, 2] >= pos[:, 2].min() - dx)  # ignore stray particles below the object
    ij = np.clip(np.round(x[keep, :2] / dx).astype(int), 0, n); z = x[keep, 2]
    order = np.lexsort((z, ij[:, 1], ij[:, 0])); ij, z = ij[order], z[order]
    first = np.ones(len(z), bool); first[1:] = (ij[1:] != ij[:-1]).any(1)
    for (i, j), zz in zip(ij[first], z[first]):
        H[i, j] = max(H[i, j], zz - 0.5 * dx)
    BED.update(H=H, dx=dx, sticky=int(sticky))


def _gripper(pos, seg_axis, a0, a1, F, dx, g, after, fixed=False, depth_frac=0.15):
    """Two jaws closing along F on the slab a0 <= x[seg_axis] <= a1; after: velocity once the hold ends."""
    seg = pos[(pos[:, seg_axis] >= a0) & (pos[:, seg_axis] <= a1)]
    W = [a for a in (0, 1, 2) if a not in (seg_axis, F)][0]
    flo, fhi = np.quantile(seg[:, F], [0.005, 0.995]); wlo, whi = np.quantile(seg[:, W], [0.005, 0.995])
    th = 0.02 * g; eps = 0.25 * dx; depth = max(depth_frac * (fhi - flo), 3.0 * dx); bcs = []
    for side, face in ((+1, flo - th + eps), (-1, fhi + th - eps)):
        c = [0.0, 0.0, 0.0]; c[seg_axis] = 0.5 * (a0 + a1); c[W] = 0.5 * (wlo + whi); c[F] = face
        half = [0.0, 0.0, 0.0]; half[seg_axis] = 0.5 * (a1 - a0); half[W] = 0.5 * (whi - wlo) + 0.01 * g; half[F] = th
        vc = [0.0, 0.0, 0.0]; vc[F] = side * depth / T_CLOSE
        closed = list(c); closed[F] = face + side * depth
        bcs += [_plate(c, half, vc, 0.0, T_CLOSE), _plate(closed, half, [0.0, 0.0, 0.0], T_CLOSE, T if fixed else T_HOLD)]
        if not fixed:
            bcs.append(_plate(closed, half, after, T_HOLD, T))
    return bcs, {'segment_axis': seg_axis, 'segment': [float(a0), float(a1)], 'jaw_axis': F, 'thickness': float(fhi - flo), 'close_depth': float(depth)}


def _end_thickness(pos, L, F, lo, hi):
    ext = hi[L] - lo[L]; t = []
    for a0, a1 in ((lo[L], lo[L] + 0.15 * ext), (hi[L] - 0.15 * ext, hi[L])):
        seg = pos[(pos[:, L] >= a0) & (pos[:, L] <= a1)]
        t.append(float(np.ptp(np.quantile(seg[:, F], [0.005, 0.995]))) if len(seg) > 10 else 0.0)
    return min(t)


def _axes(lo, hi, pos=None, dx=2.0 / 150):
    """Long axis L and jaw axis F (the thinner horizontal axis != L); falls back to the longer horizontal axis when
    an end segment along the longest axis is too thin to grip."""
    ext = hi - lo

    def jaw(L):
        hor = [a for a in (0, 1) if a != L]
        return min(hor, key=lambda a: ext[a]) if len(hor) > 1 else hor[0]
    L = int(np.argmax(ext)); F = jaw(L)
    if pos is not None and _end_thickness(pos, L, F, lo, hi) < 6 * dx:
        L2 = int(np.argmax(ext[:2])); F2 = jaw(L2)
        if L2 != L and _end_thickness(pos, L2, F2, lo, hi) > _end_thickness(pos, L, F, lo, hi):
            L, F = L2, F2
    return ext, L, F


def press(mpm_init_pos, mp):
    pos, g, dx, lo, hi = _geom(mpm_init_pos, mp, 'press')
    ext = hi - lo; z0 = float(hi[2] + 0.5 * dx)
    if os.environ.get('FB_PRESS_FORCE'):
        PLATE.update(z0=z0, force=float(os.environ['FB_PRESS_FORCE']), vmax=0.7 * float(ext[2]) / T, zmin=z0 - 0.7 * float(ext[2]),
                     profile=[], mode='force')
        _set_bed(mpm_init_pos, mp, sticky=False)
        return [{'type': 'bounding_box'}, _floor(lo, hi, g, 'slip', dx)]
    fast = os.environ.get('FB_RELEASE') == 'fast'  # plate leaves in 0.1 s; sticky bed keeps the base, so the object rings freely
    tr = 0.1 if fast else None
    if os.environ.get('FB_ACTION') == 'tap30hold':
        d = 0.3 * ext[2]; prof = [(0.0, 0.5, -d / 0.5), (0.5, 1.0, 0.0), (1.0, 1.0 + (tr or 0.5), d / (tr or 0.5))]
    else:
        d = 0.7 * ext[2]; prof = [(0.0, T, -d / T), (T, T + (tr or 1.0), d / (tr or 1.0))]
    PLATE.update(z0=z0, profile=prof)
    _set_bed(mpm_init_pos, mp, sticky=fast)
    return [{'type': 'bounding_box'}, _floor(lo, hi, g, 'slip', dx)]


def shear(mpm_init_pos, mp):
    pos, g, dx, lo, hi = _geom(mpm_init_pos, mp, 'shear')
    ext = hi - lo; F = int(np.argmin(ext[:2])); M = 1 - F  # jaws close across the thinner side, drag along the other
    v = [0.0, 0.0, 0.0]; v[M] = max(0.25 * ext[M], 0.06) / (T - T_HOLD)
    jaws, info = _gripper(pos, 2, hi[2] - 0.2 * ext[2], hi[2], F, dx, g, v)
    _set_bed(mpm_init_pos, mp, sticky=True)
    return [{'type': 'bounding_box'}, _floor(lo, hi, g, 'sticky', dx)] + jaws


def stretch(mpm_init_pos, mp):
    pos, g, dx, lo, hi = _geom(mpm_init_pos, mp, 'stretch')
    ext, L, F = _axes(lo, hi, pos, dx)
    v = [0.0, 0.0, 0.0]; v[L] = 0.15 * ext[L] / (T - T_HOLD)
    fixed, _ = _gripper(pos, L, lo[L], lo[L] + 0.15 * ext[L], F, dx, g, None, fixed=True)
    pull, _ = _gripper(pos, L, hi[L] - 0.15 * ext[L], hi[L], F, dx, g, v)
    return [{'type': 'bounding_box'}, _floor(lo, hi, g, 'slip', dx)] + fixed + pull


def bend(mpm_init_pos, mp):
    pos, g, dx, lo, hi = _geom(mpm_init_pos, mp, 'bend')
    ext, L, F = _axes(lo, hi, pos, dx)
    bcs = [{'type': 'bounding_box'}, _floor(lo, hi, g, 'slip', dx)]
    for a0, a1 in ((lo[L], lo[L] + 0.15 * ext[L]), (hi[L] - 0.15 * ext[L], hi[L])):
        bcs += _gripper(pos, L, a0, a1, F, dx, g, None, fixed=True)[0]
    mid = pos[np.abs(pos[:, L] - 0.5 * (lo[L] + hi[L])) <= 0.05 * ext[L]]
    W = [a for a in (0, 1, 2) if a not in (L, F)][0]
    th = 0.02 * g; eps = 0.25 * dx
    c = [0.0, 0.0, 0.0]; c[L] = 0.5 * (lo[L] + hi[L]); c[W] = 0.5 * (mid[:, W].min() + mid[:, W].max()); c[F] = mid[:, F].min() - th + eps
    half = [0.0, 0.0, 0.0]; half[L] = 0.05 * ext[L]; half[W] = 0.5 * np.ptp(mid[:, W]) + 0.01 * g; half[F] = th
    v = [0.0, 0.0, 0.0]; v[F] = max(0.35 * ext[F], 0.06) / T
    return bcs + [_plate(c, half, v)]



HOLD_SEG, HOLD_DEPTH = 0.25, 0.30  # firm grip for the held actions (thin grips tore or let go under the object's weight)


def _hold(pos, g, dx, lo, hi, seg_axis, a0, a1, F, after=None, t_move=(T_HOLD, T)):
    """Gripper that closes, then stays (after=None) or moves with velocity `after` during t_move."""
    jaws, info = _gripper(pos, seg_axis, a0, a1, F, dx, g, None, fixed=True, depth_frac=HOLD_DEPTH)
    if after is None:  # keep holding until the end of the clip
        return [dict(b, end_time=4.5) if b['start_time'] == T_CLOSE else b for b in jaws]
    out = []
    for b in jaws:
        if b['start_time'] == T_CLOSE:  # closed jaw: hold until the move, move, then hold again
            b0 = dict(b, end_time=t_move[0]); out.append(b0)
            moved = list(b['point']); out.append(dict(b, start_time=t_move[0], end_time=t_move[1], velocity=[float(x) for x in after]))
            for k in range(3):
                moved[k] += after[k] * (t_move[1] - t_move[0])
            out.append(dict(b, point=[float(x) for x in moved], start_time=t_move[1], end_time=4.5))
        else:
            out.append(b)
    return out


def _deep_floor(lo, hi):
    return {'type': 'surface_collider', 'point': [float(0.5 * (lo[0] + hi[0])), float(0.5 * (lo[1] + hi[1])), float(max(lo[2] - 0.2, 0.05))],
            'normal': [0.0, 0.0, 1.0], 'surface': 'slip', 'friction': 0.5, 'start_time': 0.0, 'end_time': 1e3}


def cantilever(mpm_init_pos, mp, shake=False):
    pos, g, dx, lo, hi = _geom(mpm_init_pos, mp, 'bend')
    ext = hi - lo; L = int(np.argmax(ext[:2])); F = 1 - L
    after = None; tm = (1.0, 1.25)
    if shake:
        after = [0.0, 0.0, 0.0]; after[F] = 0.15 * ext[F] / (tm[1] - tm[0])
    jaws = _hold(pos, g, dx, lo, hi, L, lo[L], lo[L] + HOLD_SEG * ext[L], F, after, tm)
    _set_bed(mpm_init_pos, mp, sticky=False); BED['t_end'] = T_HOLD
    return [{'type': 'bounding_box'}, _deep_floor(lo, hi)] + jaws


def hang(mpm_init_pos, mp):
    pos, g, dx, lo, hi = _geom(mpm_init_pos, mp, 'stretch')
    ext = hi - lo; F = int(np.argmin(ext[:2]))
    up = [0.0, 0.0, 0.06 * ext[2] / 0.5]
    jaws = _hold(pos, g, dx, lo, hi, 2, hi[2] - HOLD_SEG * ext[2], hi[2], F, up, (T_HOLD, T_HOLD + 0.5))
    _set_bed(mpm_init_pos, mp, sticky=False); BED['t_end'] = T_HOLD
    return [{'type': 'bounding_box'}, _deep_floor(lo, hi)] + jaws


def drop(mpm_init_pos, mp, _orig=su.build_drop_boundary_conditions):
    pos, g, dx, lo, hi = _geom(mpm_init_pos, mp, 'drop')
    bcs = _orig(mpm_init_pos, mp)
    for b in bcs:
        if b.get('type') == 'surface_collider':
            b['point'] = [b['point'][0], b['point'][1], float(lo[2] - float(os.environ.get('FB_DROP_GAP', '0.25')))]
    return bcs


def _register_bed():
    """Attach the support bed to the solver's grid boundary-condition loop (after the bounding box)."""
    import warp as wp
    from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
    from mpm_solver_warp.warp_utils import MPMStateStruct, MPMModelStruct, Dirichlet_collider

    @wp.kernel
    def noop(time: float, dt: float, state: MPMStateStruct, model: MPMModelStruct, param: Dirichlet_collider):
        pass

    @wp.kernel
    def bed(state: MPMStateStruct, H: wp.array2d(dtype=float), removed: wp.array2d(dtype=int), dx: float, sticky: int,
            zp: float, hfloor: float):
        i, j, k = wp.tid()
        h = H[i, j]
        if h > hfloor + dx and zp <= h + 2.0 * dx:
            removed[i, j] = 1
        if removed[i, j] == 0 and float(k) * dx < h:
            v = state.grid_v_out[i, j, k]
            if sticky == 1:
                state.grid_v_out[i, j, k] = wp.vec3(0.0, 0.0, 0.0)
            elif v[2] < 0.0:
                state.grid_v_out[i, j, k] = wp.vec3(v[0], v[1], 0.0)

    @wp.kernel
    def plate(state: MPMStateStruct, zp: float, vp: float, dx: float, dt: float, R: wp.array(dtype=float)):
        i, j, k = wp.tid()
        if float(k) * dx >= zp:
            v = state.grid_v_out[i, j, k]
            if v[2] > vp:
                state.grid_v_out[i, j, k] = wp.vec3(v[0], v[1], vp)
                wp.atomic_add(R, 0, state.grid_m[i, j, k] * (v[2] - vp) / dt)

    def plate_state(t):
        z = PLATE['z0']
        for t0, t1, v in PLATE['profile']:
            if t < t0:
                break
            if t < t1:
                return z + v * (t - t0), v
            z += v * (t1 - t0)
        return None

    add_box = MPM_Simulator_WARP.add_bounding_box

    def add_bounding_box(self, *a, **k):
        add_box(self, *a, **k)
        dims = (self.mpm_model.grid_dim_x, self.mpm_model.grid_dim_y, self.mpm_model.grid_dim_z)
        solver = self
        if PLATE:
            pdx = float(self.mpm_model.dx)

            Fw = wp.zeros(1, dtype=float, device='cuda:0')  # plate reaction force accumulator
            PLATE['_Fw'] = Fw if PLATE.get('mode') != 'force' else None
            if PLATE.get('mode') == 'force':
                PLATE.update(z=PLATE['z0'], v=0.0, R=0.0, log=[])

            def apply_plate(time, dt, param):
                if PLATE.get('mode') == 'force':
                    if time < T:  # admittance control towards the target force
                        v = -PLATE['vmax'] * max(-1.0, min(1.0, (PLATE['force'] - PLATE['R']) / PLATE['force']))
                        if PLATE['z'] <= PLATE['zmin'] and v < 0:
                            v = 0.0
                    elif PLATE['z'] < PLATE['z0']:
                        v = PLATE['vmax'] * 3.0  # lift back over about 1 s
                    else:
                        return
                    PLATE['z'] += v * dt; PLATE['v'] = v
                    wp.launch(kernel=plate, dim=dims, inputs=[solver.mpm_state, float(PLATE['z']), float(v), pdx, float(dt), Fw], device='cuda:0')
                    PLATE['n'] = PLATE.get('n', 0) + 1
                    if PLATE['n'] % 20 == 0:  # read the reaction (mean over 20 substeps) without syncing every substep
                        PLATE['R'] = float(Fw.numpy()[0]) / 20.0; Fw.zero_()
                    if int(round(time / dt)) % 1000 == 0:
                        PLATE['log'].append([round(time, 3), round(PLATE['z'], 5), round(PLATE['R'], 3)])
                    return
                st = plate_state(time)
                if st is not None:
                    wp.launch(kernel=plate, dim=dims, inputs=[solver.mpm_state, float(st[0]), float(st[1]), pdx, float(dt), Fw], device='cuda:0')
            p = Dirichlet_collider(); p.start_time = 0.0; p.end_time = 1e3
            self.collider_params.append(p); self.grid_postprocess.append(noop); self.modify_bc.append(apply_plate)
        if not BED:
            return
        Hw = wp.array(BED['H'], dtype=float, device='cuda:0'); dx = float(BED['dx']); sticky = int(BED['sticky'])
        Rw = wp.zeros(BED['H'].shape, dtype=int, device='cuda:0'); hfloor = float(BED['H'].min())

        def apply(time, dt, param):
            if time >= BED.get('t_end', 1e9):
                return
            st = plate_state(time) if PLATE else None
            zp = float(st[0]) if st is not None else 1e9
            wp.launch(kernel=bed, dim=dims, inputs=[solver.mpm_state, Hw, Rw, dx, sticky, zp, hfloor], device='cuda:0')
        p = Dirichlet_collider(); p.start_time = 0.0; p.end_time = 1e3
        self.collider_params.append(p); self.grid_postprocess.append(noop); self.modify_bc.append(apply)
    MPM_Simulator_WARP.add_bounding_box = add_bounding_box


def _clamp_particles():
    from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
    load = MPM_Simulator_WARP.load_initial_data_from_torch

    def wrapped(self, tensor_x, tensor_volume, tensor_cov=None, n_grid=100, grid_lim=1.0, device='cuda:0'):
        m = 3.0 * grid_lim / n_grid
        bad = int(((tensor_x < m) | (tensor_x > grid_lim - m)).any(1).sum())
        if bad:
            print(f'[contact_bc] clamped {bad} particles into the grid interior', flush=True)
            tensor_x = tensor_x.clamp(m, grid_lim - m)
        return load(self, tensor_x, tensor_volume, tensor_cov, n_grid=n_grid, grid_lim=grid_lim, device=device)
    MPM_Simulator_WARP.load_initial_data_from_torch = wrapped


def _dump(build):
    import json, sys
    from pathlib import Path

    def wrapped(mpm_init_pos, mp):
        bcs = json.loads(json.dumps(build(mpm_init_pos, mp), default=float))
        try:
            out = Path(sys.argv[sys.argv.index('--output_path') + 1]); out.mkdir(parents=True, exist_ok=True)
            pos = mpm_init_pos.detach().cpu().numpy()
            info = {'bcs': bcs, 'particles_lo': pos.min(0).tolist(), 'particles_hi': pos.max(0).tolist(),
                    'grid_lim': mp.get('grid_lim'), 'n_grid': mp.get('n_grid')}
            if PLATE:
                info['press_plate'] = PLATE
            if BED:
                H = BED['H']; info['bed'] = {'sticky': BED['sticky'], 'columns_raised': int((H > H.min() + 1e-6).sum()), 'top_max': float(H.max())}
                np.save(out / 'support_bed.npy', H)
            (out / 'contact_bc.json').write_text(json.dumps(info, indent=1, default=float))
        except Exception as e:
            print('contact_bc dump failed', e, flush=True)
        return bcs
    return wrapped


def _save_force_log():
    import atexit, json, sys
    from pathlib import Path

    def write():
        if PLATE.get('mode') == 'force':
            try:
                out = Path(sys.argv[sys.argv.index('--output_path') + 1])
                (out / 'press_force_log.json').write_text(json.dumps({'force': PLATE['force'], 'z0': PLATE['z0'], 'zmin': PLATE['zmin'],
                                                                      'log_t_z_R': PLATE.get('log', [])}, default=float))
            except Exception as e:
                print('force log failed', e, flush=True)
    atexit.register(write)



FORCES = {'bodies': [], 'bins': [], 'plate': [], 'n': 0}


def _register_forces():
    """Measure the force each rigid body (velocity cuboid) and the press plate exert on the object."""
    if os.environ.get('FB_FORCES', '1') != '1':
        return
    import atexit, json, sys
    from pathlib import Path
    import warp as wp
    from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
    from mpm_solver_warp.warp_utils import MPMStateStruct, MPMModelStruct, Dirichlet_collider

    @wp.kernel
    def noop(time: float, dt: float, state: MPMStateStruct, model: MPMModelStruct, param: Dirichlet_collider):
        pass

    @wp.kernel
    def measure(state: MPMStateStruct, i0: int, j0: int, k0: int, c: wp.vec3, h: wp.vec3, vb: wp.vec3, dx: float, dt: float,
                acc: wp.array(dtype=wp.vec3), b: int):
        a, bb, cc = wp.tid()
        i = i0 + a; j = j0 + bb; k = k0 + cc
        off = wp.vec3(float(i) * dx - c[0], float(j) * dx - c[1], float(k) * dx - c[2])
        if wp.abs(off[0]) < h[0] and wp.abs(off[1]) < h[1] and wp.abs(off[2]) < h[2]:
            m = state.grid_m[i, j, k]
            if m > 1e-15:
                wp.atomic_add(acc, b, m * (vb - state.grid_v_out[i, j, k]) / dt)

    ACC = {}
    cuboid = MPM_Simulator_WARP.set_velocity_on_cuboid

    def set_velocity_on_cuboid(self, point, size, velocity, start_time=0.0, end_time=999.0, reset=0):
        b = len(FORCES['bodies'])
        FORCES['bodies'].append({'point': [float(x) for x in point], 'size': [float(x) for x in size], 'velocity': [float(x) for x in velocity],
                                 'start_time': float(start_time), 'end_time': float(end_time)})
        if 'acc' not in ACC:
            ACC['acc'] = wp.zeros(64, dtype=wp.vec3, device='cuda:0')
        holder, solver = {}, self
        n = int(self.mpm_model.grid_dim_x); dx = float(self.mpm_model.dx)

        def hook(time, dt, _p):
            cp = holder.get('param')
            if FORCES['n'] % 4 or cp is None or b >= 64 or not (time >= cp.start_time and time < cp.end_time):
                return  # sample every 4th substep (25 samples per 10 ms bin) to keep the overhead small
            c = cp.point; h = cp.size
            lo = [max(0, int((c[d] - h[d]) / dx)) for d in range(3)]; hi = [min(n - 1, int((c[d] + h[d]) / dx) + 1) for d in range(3)]
            dims = tuple(max(hi[d] - lo[d] + 1, 0) for d in range(3))
            if min(dims) > 0:
                wp.launch(kernel=measure, dim=dims, inputs=[solver.mpm_state, lo[0], lo[1], lo[2], c, h, cp.velocity, dx, float(dt), ACC['acc'], b],
                          device='cuda:0')
        p = Dirichlet_collider(); p.start_time = 0.0; p.end_time = 1e3
        self.collider_params.append(p); self.grid_postprocess.append(noop); self.modify_bc.append(hook)
        cuboid(self, point, size, velocity, start_time, end_time, reset)
        holder['param'] = self.collider_params[-1]
    MPM_Simulator_WARP.set_velocity_on_cuboid = set_velocity_on_cuboid

    box = MPM_Simulator_WARP.add_bounding_box

    def add_bounding_box(self, *a, **k):  # first BC of every action: register the per-substep binning tick
        box(self, *a, **k)
        if 'acc' not in ACC:
            ACC['acc'] = wp.zeros(64, dtype=wp.vec3, device='cuda:0')

        def tick(time, dt, _p):
            FORCES['n'] += 1
            if FORCES['n'] % 100 == 0:  # 10 ms at dt = 1e-4
                nb = max(len(FORCES['bodies']), 1)
                FORCES['bins'].append((ACC['acc'].numpy()[:nb] / 25.0).tolist()); ACC['acc'].zero_()
                fw = PLATE.get('_Fw')
                if fw is not None:
                    FORCES['plate'].append(-float(fw.numpy()[0]) / 100.0); fw.zero_()
        p = Dirichlet_collider(); p.start_time = 0.0; p.end_time = 1e3
        self.collider_params.append(p); self.grid_postprocess.append(noop); self.modify_bc.append(tick)
    MPM_Simulator_WARP.add_bounding_box = add_bounding_box

    def write():
        try:
            out = Path(sys.argv[sys.argv.index('--output_path') + 1])
            nb = max(len(FORCES['bodies']), 1)
            F = np.array([row + [[0.0, 0.0, 0.0]] * (nb - len(row)) for row in FORCES['bins']], np.float32) if FORCES['bins'] else np.zeros((0, nb, 3), np.float32)
            t = (np.arange(len(F)) + 0.5) * 0.01
            np.savez_compressed(out / 'contact_forces.npz', t=t, F=F, F_plate=np.array(FORCES['plate'], np.float32),
                                bodies=json.dumps(FORCES['bodies']))
        except Exception as e:
            print('contact force export failed', e, flush=True)
    atexit.register(write)


def install():
    _save_force_log()
    _register_forces()
    _clamp_particles()
    _register_bed()
    su.build_press_boundary_conditions = _dump(press)
    if os.environ.get('FB_WEAK') == 'hold':
        su.build_shear_boundary_conditions = _dump(lambda p, m: cantilever(p, m, shake=True))
        su.build_bend_boundary_conditions = _dump(cantilever)
        su.build_stretch_boundary_conditions = _dump(hang)
    else:
        su.build_shear_boundary_conditions = _dump(shear)
        su.build_bend_boundary_conditions = _dump(bend)
        su.build_stretch_boundary_conditions = _dump(stretch)
    su.build_drop_boundary_conditions = _dump(drop)
