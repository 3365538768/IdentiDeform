"""R2: parameter identifiability of the original DeformFieldBench Warp-MPM (press / drop).

A cube is simulated with the dataset's press/drop boundary builders and material settings, while
E, rho, nu and yield stress are varied around a reference. Each case reports how much the particle
trajectories differ from the reference, both raw and after removing per-frame rigid translation.
Hypothesis under test: scaling (E, rho, sigma_y) together leaves motion unchanged, so only E/rho,
sigma_y/rho and nu are visible in video.

Run with the warp engine env, borrowing torch from the training venv:
  PYTHONPATH=.venv/lib/python3.12/site-packages .engine-envs/warp/bin/python scripts/r2_mpm_identifiability.py
"""
import argparse
import math
import json
import sys
import time
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'upstream/DeformFieldBench'
sys.modules.setdefault('h5py', types.ModuleType('h5py'))  # only used by upstream file export
import torch  # noqa: E402
import warp as wp  # noqa: E402

wp.config.quiet = True
# The solver imports its helpers as top-level modules, while decode_param imports the solver as
# the namespace package mpm_solver_warp.*: preload the helpers, then drop the inner directory so
# its mpm_solver_warp.py cannot shadow the package.
inner = str(UPSTREAM / 'simulation/mpm_solver_warp')
sys.path.insert(0, inner)
import engine_utils, warp_utils, mpm_utils  # noqa: E401,E402,F401
sys.path.remove(inner)
sys.path[:0] = [str(UPSTREAM / 'simulation'), str(UPSTREAM)]
from mpm_solver_warp.mpm_solver_warp import MPM_Simulator_WARP  # noqa: E402
from utils.decode_param import set_boundary_conditions  # noqa: E402
from my_utils.sim_utils import build_drop_boundary_conditions, build_press_boundary_conditions  # noqa: E402

SPACE = json.loads((UPSTREAM / 'configs/simulation/material_space_example.json').read_text())['material_spaces']
REFERENCE = {
    'jelly': {'E': 1e4, 'nu': 0.3, 'density': 1000.0},
    'plasticine': {'E': 1e4, 'nu': 0.3, 'density': 1000.0, 'yield_stress': 1e3, 'hardening': 0.2, 'softening': 0.2},
    'metal': {'E': 2e6, 'nu': 0.3, 'density': 5000.0, 'yield_stress': 1e4, 'hardening': 0.3},
}


def cases(material):
    ref = REFERENCE[material]; scaled = ('E', 'density', 'yield_stress')
    out = {'reference': {}}
    for k in (4.0, 0.25):
        out[f'E_x{k:g}'] = {'E': ref['E'] * k}
        out[f'rho_x{k:g}'] = {'density': ref['density'] * k}
        out[f'E_rho_sigma_x{k:g}'] = {key: ref[key] * k for key in scaled if key in ref}
        if 'yield_stress' in ref:
            out[f'sigma_x{k:g}'] = {'yield_stress': ref['yield_stress'] * k}
    out['nu_0.4'] = {'nu': 0.4}
    return out


def cube(spacing, lo=0.75, hi=1.25):
    axis = torch.arange(lo + spacing / 2, hi, spacing)
    grid = torch.stack(torch.meshgrid(axis, axis, axis, indexing='ij'), -1).reshape(-1, 3)
    return grid.float().cuda(), torch.full((len(grid),), spacing ** 3).float().cuda()


def simulate(material, overrides, action, seconds, frame_dt, substep_dt, damping_mode="legacy"):
    per_frame = int(math.ceil(frame_dt / substep_dt))
    effective_dt = frame_dt / per_frame
    params = {**{k: v for k, v in SPACE[material].items() if not isinstance(v, dict)},
              **REFERENCE[material], **overrides, 'material': material,
              'g': [0.0, 0.0, -9.8] if action == 'drop' else [0.0, 0.0, 0.0]}
    if damping_mode == 'physical_time':
        # Reference damping is defined at 100us; preserve continuous decay rate.
        params['grid_v_damping_scale'] = float(params['grid_v_damping_scale']) ** (effective_dt / 1e-4)
        # RPIC mixing is not a continuous-time damping law; disable for this controlled convergence test.
        params['rpic_damping'] = 0.0
    elif damping_mode != 'legacy':
        raise ValueError(damping_mode)
    x, volume = cube(params['grid_lim'] / params['n_grid'])
    solver = MPM_Simulator_WARP(10)
    solver.load_initial_data_from_torch(x, volume, n_grid=params['n_grid'], grid_lim=params['grid_lim'])
    solver.set_parameters_dict(params)
    builder = build_press_boundary_conditions if action == 'press' else build_drop_boundary_conditions
    bcs = builder(x, params)
    set_boundary_conditions(solver, bcs, {'substep_dt': substep_dt, 'frame_dt': frame_dt})
    solver.finalize_mu_lam()
    frames, per_frame = int(round(seconds / frame_dt)), int(math.ceil(frame_dt / substep_dt))
    effective_dt = frame_dt / per_frame
    trajectory = [solver.export_particle_x_to_torch().clone().cpu()]
    for frame in range(frames):
        for _ in range(per_frame):
            solver.p2g2p(frame, effective_dt, device='cuda:0')
        trajectory.append(solver.export_particle_x_to_torch().clone().cpu())
    out = torch.stack(trajectory).numpy().astype(np.float64)
    if not np.isfinite(out).all():
        raise FloatingPointError(f'non-finite trajectory: {material} {action} {overrides}')
    return out


def compare(traj, ref):
    motion = ref - ref[:1]
    centered = lambda a: a - a.mean(1, keepdims=True)
    deformation = centered(ref) - centered(ref[:1])
    return {'relative_rms': float(np.sqrt(((traj - ref) ** 2).mean() / max((motion ** 2).mean(), 1e-30))),
            'relative_rms_deformation': float(np.sqrt(((centered(traj) - centered(ref)) ** 2).mean()
                                                      / max((deformation ** 2).mean(), 1e-30))),
            'rms_grid_units': float(np.sqrt(((traj - ref) ** 2).mean())),
            'reference_motion_rms': float(np.sqrt((motion ** 2).mean())),
            'reference_deformation_rms': float(np.sqrt((deformation ** 2).mean()))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--materials', default='jelly,plasticine,metal'); p.add_argument('--actions', default='press,drop')
    p.add_argument('--frame-dt', type=float, default=1 / 30); p.add_argument('--substep-dt', type=float, default=1e-4)
    p.add_argument('--output', default='docs/r2/MPM_IDENTIFIABILITY.json'); a = p.parse_args()
    wp.init(); Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    report = {'scope': 'Original DeformFieldBench Warp-MPM on a 0.5-unit cube (not GS assets); trajectories of all '
                       'particles at frame_dt; relative_rms normalised by the reference motion RMS.',
              'frame_dt': a.frame_dt, 'substep_dt': a.substep_dt, 'results': []}
    for material in a.materials.split(','):
        for action in a.actions.split(','):
            seconds = 4.0 if action == 'press' else 2.0
            started = time.time(); ref = simulate(material, {}, action, seconds, a.frame_dt, a.substep_dt)
            for name, overrides in cases(material).items():
                traj = ref if name == 'reference' else simulate(material, overrides, action, seconds, a.frame_dt, a.substep_dt)
                row = {'material': material, 'action': action, 'case': name, 'overrides': overrides, **compare(traj, ref)}
                report['results'].append(row); print(json.dumps(row), flush=True)
                Path(a.output).write_text(json.dumps(report, indent=2))
            print(f'{material}/{action} done in {time.time() - started:.0f}s', flush=True)
    report['status'] = 'complete'; Path(a.output).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
