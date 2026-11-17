"""V3e: upstream modified_simulation.py with a parallel-jaw SQUEEZE action (gripper task).

Env FB_SQUEEZE_D = closing depth d (fraction of the object width W along the squeeze axis; 0 = no plates,
the gravity-only reference run). Run with --sim_type press (4 s total). The press builder is replaced by:
  bounding box + floor collider (identical to press) + two rigid, sticky plates on the horizontal axis with the
  smaller extent (the gripper closes across the narrower side). Each plate starts just outside the object and
  moves inward by d*W/2 over [0, T_CLOSE); plates are held still on [T_CLOSE, T_OPEN); at T_OPEN the jaws
  open (plates removed) and the object relaxes freely until 4 s.
Particle positions are recorded in-process (no h5 files): full positions at frame 0, the end of closing, the end
of the hold, the mean over the last 0.5 s, a 4096-particle subset every 4 frames, and the width trace.
"""
import json
import os
import runpy
import sys
from pathlib import Path

import numpy as np
import my_utils.sim_utils as su
import mpm_solver_warp.engine_utils as eu

T_CLOSE, T_OPEN, FRAMES, FRAME_DT = 1.0, 1.5, 400, 0.01
OUT = Path(sys.argv[sys.argv.index('--output_path') + 1])
D = float(os.environ['FB_SQUEEZE_D'])
INFO = {}
press = su.build_press_boundary_conditions


def squeeze(mpm_init_pos, material_params):
    bcs = press(mpm_init_pos, material_params)[:2]  # bounding box + floor collider
    pos = su._filter_points_for_bbox(mpm_init_pos.detach().cpu().numpy(), material_params, prefix='press')
    g = float(material_params.get('grid_lim', 2.0))
    lo, hi = pos.min(0), pos.max(0); ext = hi - lo
    a = int(np.argmin(ext[:2])); b = 1 - a
    W = float(ext[a]); th = 0.02 * g; eps = 0.002 * g
    speed = D * W / 2 / T_CLOSE
    INFO.update(axis=a, width=W, extents=ext.tolist(), lo=lo.tolist(), hi=hi.tolist(), d=D, grid_lim=g,
                plate_half_thickness=th, t_close=T_CLOSE, t_open=T_OPEN, n_filtered=int(len(pos)))
    if D > 0:
        for sign, face in ((+1, lo[a] - th + eps), (-1, hi[a] + th - eps)):
            point = [0.0, 0.0, 0.5 * (lo[2] + hi[2])]; point[a] = float(face); point[b] = 0.5 * g
            size = [0.0, 0.0, 0.5 * ext[2] + 0.02 * g]; size[a] = th; size[b] = 0.5 * g
            vel = [0.0, 0.0, 0.0]; vel[a] = sign * speed
            moving = {'type': 'cuboid', 'point': point, 'size': [float(s) for s in size], 'velocity': vel,
                      'start_time': 0.0, 'end_time': T_CLOSE, 'reset': 0}
            held_point = list(point); held_point[a] = float(face + sign * speed * T_CLOSE)
            held = dict(moving, point=held_point, velocity=[0.0, 0.0, 0.0], start_time=T_CLOSE, end_time=T_OPEN)
            bcs += [moving, held]
    bcs = json.loads(json.dumps(bcs, default=float))  # numpy scalars -> python floats
    INFO['bcs'] = bcs
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'squeeze_bc.json').write_text(json.dumps(INFO, indent=1, default=float))
    return bcs


REC = {'subset_t': [], 'subset': [], 'width': [], 'zmax': [], 'tail': None, 'tail_n': 0}
TAIL0 = FRAMES - int(round(0.5 / FRAME_DT))  # last 0.5 s


def record(mpm_solver, dir_name, frame, save_to_ply=True, save_to_h5=False):
    x = mpm_solver.mpm_state.particle_x.numpy().astype(np.float64)
    if frame == 0:
        REC['x0'] = x.astype(np.float32)
        REC['idx'] = np.random.default_rng(0).choice(len(x), min(4096, len(x)), replace=False)
    a = INFO.get('axis', 0)
    q = np.quantile(x[:, a], [0.01, 0.99]); REC['width'].append(float(q[1] - q[0]))
    REC['zmax'].append(float(np.quantile(x[:, 2], 0.99)))
    if frame % 4 == 0:
        REC['subset_t'].append(frame); REC['subset'].append(x[REC['idx']].astype(np.float32))
    if frame == int(round(T_CLOSE / FRAME_DT)):
        REC['x_closed'] = x.astype(np.float32)
    if frame == int(round(T_OPEN / FRAME_DT)):
        REC['x_open'] = x.astype(np.float32)
    if frame > TAIL0:
        REC['tail'] = x if REC['tail'] is None else REC['tail'] + x; REC['tail_n'] += 1
    if frame == FRAMES:
        REC['x_last'] = x.astype(np.float32)
        np.savez_compressed(OUT / 'traj.npz', x0=REC['x0'], x_closed=REC['x_closed'], x_open=REC['x_open'],
                            x_last=REC['x_last'], x_tail=(REC['tail'] / REC['tail_n']).astype(np.float32),
                            idx=REC['idx'], subset=np.stack(REC['subset']), subset_t=np.array(REC['subset_t']),
                            width=np.array(REC['width']), zmax=np.array(REC['zmax']))


su.build_press_boundary_conditions = squeeze
eu.save_data_at_frame = record
sys.argv = ['modified_simulation.py', *sys.argv[1:]]
runpy.run_path('modified_simulation.py', run_name='__main__')
