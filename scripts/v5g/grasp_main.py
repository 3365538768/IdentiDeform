"""V5g: upstream modified_simulation.py with a GRASP-AND-LIFT action (two-sided downstream task). Run with --sim_type stretch
(its builder is replaced; unlike press/shear/bend the clip length comes from the config, FRAMES = 260 = 2.6 s).

Env: FB_GRASP_POS = end | centre | none (none = reference run without jaws), FB_GRASP_LEVEL = closing depth per jaw as a fraction
of the local thickness of the gripped segment (min 3 grid cells, as scripts/v2/contact_bc._gripper).
Object rests on the V3 support bed (non-sticky, removed at T_LIFT0) and a slip floor at its lowest point; gravity from the config.
Gripper: two rigid sticky jaws with the geometry of contact_bc._gripper (sized to the local cross-section of a segment of 25 % of the
long horizontal axis L, closing along the other horizontal axis F); segment = the outer 25 % at the low-L end ('end') or the middle
25 % ('centre'). The jaws are applied by ONE small kernel per jaw and substep over the jaw's swept region (grid velocity of the nodes
inside the moving jaw box = jaw velocity, as upstream set_velocity_on_cuboid, without its full-grid kernels per jaw phase), which
also accumulates the jaw's force on the object m (v_jaw - v) / dt over the hold window (grip force; config density units).
Timeline: close 0-T_CLOSE; hold; bed removed at T_LIFT0; lift by LIFT * height over T_LIFT0-T_LIFT1; hold to T_LOWER0; lower back over
T_LOWER0-T_LOWER1; jaws open (removed) at T_OPEN; free relaxation on the floor to the end (2.6 s).
Recorded: x0, a 4096-particle subset, mean dz / left-behind fraction per frame, mean positions over the last 0.2 s, grasp_force.json.
"""
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v2'))
import my_utils.sim_utils as su  # noqa: E402
import mpm_solver_warp.engine_utils as eu  # noqa: E402
import contact_bc as C  # noqa: E402

T_CLOSE, T_LIFT0, T_LIFT1, T_LOWER0, T_LOWER1, T_OPEN = 0.4, 0.6, 1.0, 1.6, 2.0, 2.1
FRAMES, FRAME_DT = 260, 0.01
LIFT = 0.30
SEG = 0.25
MIN_CELLS = float(os.environ.get('FB_GRASP_MIN_CELLS', '1'))
HOLD_WIN = (1.3, 1.6)
TAIL0 = FRAMES - 20
OUT = Path(sys.argv[sys.argv.index('--output_path') + 1])
POS = os.environ.get('FB_GRASP_POS', 'none'); LEVEL = float(os.environ.get('FB_GRASP_LEVEL', '0'))
INFO = {}
JAWS = []
REC = {'zmean': [], 'left': [], 'subset': [], 'subset_t': [], 'tail': None, 'tail_n': 0}


def lift_at(t):
    if t < T_LIFT0 or 'lift' not in INFO:
        return 0.0, 0.0
    vz = INFO['lift'] / (T_LIFT1 - T_LIFT0)
    if t < T_LIFT1:
        return vz * (t - T_LIFT0), vz
    if t < T_LOWER0:
        return INFO['lift'], 0.0
    if t < T_LOWER1:
        return INFO['lift'] - vz * (t - T_LOWER0), -vz
    return 0.0, 0.0


def grasp(mpm_init_pos, mp):
    pos, g, dx, lo, hi = C._geom(mpm_init_pos, mp, 'press')
    ext = hi - lo; L = int(np.argmax(ext[:2])); F = 1 - L
    C._set_bed(mpm_init_pos, mp, sticky=False); C.BED['t_end'] = T_LIFT0
    bcs = [{'type': 'bounding_box'}, C._floor(lo, hi, g, 'slip', dx)]
    INFO.update(pos=POS, level=LEVEL, L=L, F=F, lo=lo.tolist(), hi=hi.tolist(), ext=ext.tolist(), lift=LIFT * float(ext[2]), dx=dx,
                times=[T_CLOSE, T_LIFT0, T_LIFT1, T_LOWER0, T_LOWER1, T_OPEN])
    if POS != 'none':
        a0 = lo[L] if POS == 'end' else 0.5 * (lo[L] + hi[L]) - 0.5 * SEG * ext[L]
        a1 = a0 + SEG * ext[L]
        C.T_CLOSE = T_CLOSE
        jaws, ji = C._gripper(pos, L, a0, a1, F, dx, g, None, fixed=True, depth_frac=LEVEL)
        INFO.update(**ji)  # segment, jaw_axis, thickness, close_depth
        depth = max(LEVEL * ji['thickness'], MIN_CELLS * dx)  # gentler floor than _gripper's 3 cells (fine grip levels)
        INFO['close_depth'] = depth
        for b in jaws:
            if b['start_time'] == 0.0:  # closing cuboid: start centre, half size, closing velocity (re-scaled to `depth`)
                vc = list(b['velocity']); vc[F] = float(np.sign(vc[F]) * depth / T_CLOSE)
                JAWS.append({'c0': b['point'], 'h': b['size'], 'vc': vc})
        INFO['jaws'] = JAWS
    OUT.mkdir(parents=True, exist_ok=True); (OUT / 'grasp_bc.json').write_text(json.dumps(INFO, indent=1, default=float))
    return json.loads(json.dumps(bcs, default=float))


def _register_jaws():
    import warp as wp
    from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP
    from mpm_solver_warp.warp_utils import MPMStateStruct, MPMModelStruct, Dirichlet_collider

    @wp.kernel
    def noop(time: float, dt: float, state: MPMStateStruct, model: MPMModelStruct, param: Dirichlet_collider):
        pass

    @wp.kernel
    def jaw(state: MPMStateStruct, i0: int, j0: int, k0: int, c: wp.vec3, h: wp.vec3, v: wp.vec3, dx: float, dt: float,
            acc: wp.array(dtype=wp.vec3), b: int, meas: int):
        a, bb, cc = wp.tid()
        i = i0 + a; j = j0 + bb; k = k0 + cc
        off = wp.vec3(float(i) * dx - c[0], float(j) * dx - c[1], float(k) * dx - c[2])
        if wp.abs(off[0]) < h[0] and wp.abs(off[1]) < h[1] and wp.abs(off[2]) < h[2]:
            if meas == 1:
                m = state.grid_m[i, j, k]
                if m > 1e-15:
                    wp.atomic_add(acc, b, m * (v - state.grid_v_out[i, j, k]) / dt)
            state.grid_v_out[i, j, k] = v

    box = MPM_Simulator_WARP.add_bounding_box

    def add_bounding_box(self, *a, **k):
        box(self, *a, **k)
        if not JAWS:
            return
        solver = self; n = int(self.mpm_model.grid_dim_x); dx = float(self.mpm_model.dx)
        acc = wp.zeros(len(JAWS), dtype=wp.vec3, device='cuda:0'); cnt = {'n': 0}
        regions = []
        for jw in JAWS:
            c0, h, vc = np.array(jw['c0']), np.array(jw['h']), np.array(jw['vc'])
            c1 = c0 + vc * T_CLOSE
            lo_ = np.minimum(c0, c1) - h; hi_ = np.maximum(c0, c1) + h; hi_[2] += INFO['lift']
            i0 = [max(0, int(lo_[d] / dx) - 1) for d in range(3)]; i1 = [min(n - 1, int(hi_[d] / dx) + 2) for d in range(3)]
            regions.append((i0, tuple(i1[d] - i0[d] + 1 for d in range(3))))

        def apply(time, dt, param):
            if time >= T_OPEN:
                return
            zoff, vz = lift_at(time)
            meas = 1 if HOLD_WIN[0] <= time < HOLD_WIN[1] else 0
            for b, (jw, (i0, dims)) in enumerate(zip(JAWS, regions)):
                c0, vc = np.array(jw['c0']), np.array(jw['vc'])
                c = c0 + vc * min(time, T_CLOSE); c[2] += zoff
                v = vc.copy() if time < T_CLOSE else np.zeros(3); v[2] += vz
                wp.launch(kernel=jaw, dim=dims, inputs=[solver.mpm_state, i0[0], i0[1], i0[2], wp.vec3(*[float(x) for x in c]),
                                                        wp.vec3(*[float(x) for x in jw['h']]), wp.vec3(*[float(x) for x in v]),
                                                        dx, float(dt), acc, b, meas], device='cuda:0')
            cnt['n'] += meas
        p = Dirichlet_collider(); p.start_time = 0.0; p.end_time = 1e3
        self.collider_params.append(p); self.grid_postprocess.append(noop); self.modify_bc.append(apply)
        REC['acc'] = acc; REC['cnt'] = cnt
    MPM_Simulator_WARP.add_bounding_box = add_bounding_box


def record(mpm_solver, dir_name, frame, save_to_ply=True, save_to_h5=False):
    x = mpm_solver.mpm_state.particle_x.numpy().astype(np.float64)
    if frame == 0:
        REC['x0'] = x.astype(np.float32); REC['z0'] = x[:, 2].copy()
        REC['idx'] = np.random.default_rng(0).choice(len(x), min(4096, len(x)), replace=False)
    dz = x[:, 2] - REC['z0']; li = lift_at(frame * FRAME_DT)[0]
    REC['zmean'].append(float(dz.mean()))
    REC['left'].append(float((dz < 0.5 * li).mean()) if li > 0 else 0.0)
    if frame % 4 == 0:
        REC['subset_t'].append(frame); REC['subset'].append(x[REC['idx']].astype(np.float32))
    if frame > TAIL0:
        REC['tail'] = x if REC['tail'] is None else REC['tail'] + x; REC['tail_n'] += 1
    if frame == FRAMES:
        if 'acc' in REC:
            f = (REC['acc'].numpy() / max(REC['cnt']['n'], 1)).tolist()
            (OUT / 'grasp_force.json').write_text(json.dumps({'mean_force_per_jaw_hold': f, 'substeps': REC['cnt']['n'], 'hold_window_s': HOLD_WIN}))
        np.savez_compressed(OUT / 'traj.npz', x0=REC['x0'], x_tail=(REC['tail'] / REC['tail_n']).astype(np.float32), idx=REC['idx'],
                            subset=np.stack(REC['subset']), subset_t=np.array(REC['subset_t']), zmean=np.array(REC['zmean']),
                            left=np.array(REC['left']))


C._clamp_particles()
C._register_bed()
_register_jaws()
su.build_stretch_boundary_conditions = grasp
eu.save_data_at_frame = record
sys.argv = ['modified_simulation.py', *sys.argv[1:]]
runpy.run_path('modified_simulation.py', run_name='__main__')
