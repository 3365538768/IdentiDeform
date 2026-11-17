"""V3a: libuipc (FEM, StableNeoHookean, BDF1) cantilever videos for the cross-engine few-shot pilot.

Elastic only. Each case: one beam geometry (geometry split train/val), log-uniform E, nu in [0.2,0.45],
density 1000 kg/m^3, tip load in -y with a ramp-hold-release schedule, then free vibration. Dynamics are
run with pyuipc 0.0.29; 64 frames at 0.04 s are rendered from 3 fixed orthographic views at 224 px with the
deterministic depth-tested rasterizer (scripts/physical_renderer.py). No intermediate images are written.

Output per case (data/v3_uipc/<split>/<idx>/): sample_pack.npz (rgb [3,64,3,224,224] uint8, object_mask,
zero placeholder flow/stress/force_mask at 56 px so the legacy_visual loader works; these are NOT field
supervision), traj.npz (surface displacement per frame), gt.json, result.json (written last).

  .engine-envs/uipc/bin/python scripts/v3/gen_uipc_video.py --split train --shard 0 --num-shards 4
  .engine-envs/uipc/bin/python scripts/v3/gen_uipc_video.py --split val --indices 0 --young 20000,300000 --check
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from engine_fixture import atomic, block_mesh, tip_weights  # noqa: E402
from physical_renderer import exterior_triangles, rasterize  # noqa: E402

# (name, [L, h(y), w(z)] metres). Geometry split: no geometry is shared between train and val.
GEOMETRIES = {
    'train': [('beam_a', [0.10, 0.015, 0.015]), ('beam_b', [0.08, 0.012, 0.020]), ('beam_c', [0.12, 0.020, 0.020]),
              ('beam_d', [0.07, 0.010, 0.014]), ('plate_e', [0.09, 0.010, 0.030])],
    'val': [('beam_f', [0.11, 0.016, 0.012]), ('beam_g', [0.085, 0.014, 0.024])],
}
COUNTS = {'train': 120, 'val': 40}
E_RANGE = (2.0e4, 3.0e5)
NU_RANGE = (0.2, 0.45)
RHO = 1000.
FRAME_DT, FRAMES, SUBSTEPS = 0.04, 64, 8           # solver dt = 0.005 s; 63 frame intervals = 2.52 s
FORCE_HISTORY = [[0., 0.], [0.2, 1.], [1.2, 1.], [1.3, 0.], [10., 0.]]   # ramp, hold, release, free vibration
DELTA_AT_EMIN = 0.25                                # linear Euler-Bernoulli tip deflection / L at E_min
IMAGE, FIELD = 224, 56
VIEWS = [('side', 0.0, 0.0), ('oblique', 0.55, -0.30), ('high', -0.85, -0.55)]


def case_for(split, idx):
    rng = np.random.default_rng(int(hashlib.sha256(f'v3a-uipc-{split}-{idx}'.encode()).hexdigest()[:12], 16))
    geoms = GEOMETRIES[split]
    name, dims = geoms[idx % len(geoms)]
    L, h, w = dims
    inertia = w * h ** 3 / 12
    force = DELTA_AT_EMIN * L * 3 * E_RANGE[0] * inertia / L ** 3   # same force for every E on this geometry
    young = float(math.exp(rng.uniform(math.log(E_RANGE[0]), math.log(E_RANGE[1]))))
    nu = float(rng.uniform(*NU_RANGE))
    nx = 16; ny = max(3, round(nx * h / L)); nz = max(3, round(nx * w / L))
    return dict(split=split, index=idx, geometry=name, dimensions_m=dims, cells=[nx, ny, nz], young_pa=young,
                poisson=nu, density_kg_m3=RHO, force_n=float(force), axis=1, direction=-1.,
                force_history=FORCE_HISTORY, frame_dt_s=FRAME_DT, frames=FRAMES, substeps=SUBSTEPS,
                texture_seed=int(rng.integers(1, 10 ** 6)))


def simulate(case, workspace):
    from uipc import Engine, Scene, Timer, World, builtin, view
    from uipc.constitution import ElasticModuli, FiniteElementExternalForce, StableNeoHookean
    from uipc.core import FiniteElementStateAccessorFeature
    from uipc.geometry import label_surface, label_triangle_orient, tetmesh
    from interaction_protocol import force_scale
    Timer.disable_all(); workspace.mkdir(parents=True, exist_ok=True)
    rest, tets = block_mesh(case['cells'], case['dimensions_m'])
    young, nu = case['young_pa'], case['poisson']
    mu = young / (2 * (1 + nu)); lam = young * nu / ((1 + nu) * (1 - 2 * nu))
    mesh = tetmesh(rest, tets); label_surface(mesh); label_triangle_orient(mesh)
    StableNeoHookean().apply_to(mesh, ElasticModuli.youngs_poisson(young, nu), mass_density=RHO)
    # Same explicit SNK1 coefficient correction as scripts/solve_engine_fixture.py (pyuipc 0.0.29 map differs).
    view(mesh.tetrahedra().find('mu'))[:] = mu
    view(mesh.tetrahedra().find('lambda'))[:] = lam + mu
    view(mesh.vertices().find(builtin.is_fixed))[np.isclose(rest[:, 0], 0, atol=1e-12)] = 1
    FiniteElementExternalForce().apply_to(mesh, np.zeros(3))
    weights = tip_weights(rest, tets, case['dimensions_m'][0])
    nodal = np.zeros_like(rest); nodal[:, case['axis']] = case['direction'] * weights * case['force_n']
    dt = case['frame_dt_s'] / case['substeps']
    cfg = Scene.default_config(); cfg['gravity'] = [[0.], [0.], [0.]]; cfg['contact']['enable'] = False
    cfg['dt'] = dt; cfg['newton']['velocity_tol'] = 1e-6
    cfg['newton']['semi_implicit']['enable'] = False; cfg['linear_system']['tol_rate'] = 1e-7
    engine = Engine('cuda', str(workspace)); world = World(engine); scene = Scene(cfg)
    obj = scene.objects().create('beam'); obj.geometries().create(mesh)
    scale = [0.]

    def animate(info):
        geom = info.geo_slots()[0].geometry()
        view(geom.vertices().find('external_force'))[:] = (nodal * scale[0]).reshape(-1, 3, 1)
        view(geom.vertices().find('external_force/is_constrained'))[:] = (weights > 0).reshape(-1, 1)
    scene.animator().insert(obj, animate); world.init(scene)
    accessor = world.features().find(FiniteElementStateAccessorFeature)
    frames = [np.zeros_like(rest)]; completed = True; max_speed = 0.; loads = [0.]
    total = (case['frames'] - 1) * case['substeps']
    for step in range(total):
        t = (step + 1) * dt
        scale[0] = force_scale(t, case['force_history'])
        world.advance(); world.retrieve()
        completed &= bool(engine.frame_stats()['completed'])
        if not completed:
            raise RuntimeError(f'uipc step failed at t={t:.3f}s')
        if (step + 1) % case['substeps'] == 0:
            state = accessor.create_geometry(); state.vertices().create(builtin.position, np.zeros(3))
            state.vertices().create(builtin.velocity, np.zeros(3)); accessor.copy_to(state)
            x = np.array(view(state.vertices().find(builtin.position))).reshape(-1, 3)
            v = np.array(view(state.vertices().find(builtin.velocity))).reshape(-1, 3)
            if not np.isfinite(x).all():
                raise RuntimeError('non-finite positions')
            max_speed = max(max_speed, float(np.linalg.norm(v, axis=1).max()))
            frames.append(x - rest); loads.append(scale[0] * case['force_n'])
    disp = np.stack(frames)
    basis = np.stack([rest[tets[:, i]] - rest[tets[:, 0]] for i in range(1, 4)], -1)
    min_j = 1.
    for d in disp[::4]:
        x = rest + d
        cur = np.stack([x[tets[:, i]] - x[tets[:, 0]] for i in range(1, 4)], -1)
        min_j = min(min_j, float(np.linalg.det(cur @ np.linalg.inv(basis)).min()))
    health = dict(completed=completed, max_speed_m_s=max_speed, min_jacobian=min_j, solver_dt_s=dt, steps=total,
                  native_coefficients_pa={'mu': mu, 'lambda_snk1': lam + mu}, integrator='bdf1',
                  numerical_dissipation_note='Implicit BDF1 damping is numerical, not a material damping parameter.')
    return rest, tets, disp, np.array(loads), health


def render(case, rest, tets, disp):
    tri = exterior_triangles(tets); surface = np.unique(tri)
    inverse = np.full(len(rest), -1, int); inverse[surface] = np.arange(len(surface))
    tri = inverse[tri]; rest_s = rest[surface]; disp_s = disp[:, surface]
    L, h, w = case['dimensions_m']
    cams = [dict(name=n, yaw=y, pitch=p, image_size=IMAGE, field_of_view_m=1.45 * L,
                 center_m=[0.5 * L, -0.12 * L, 0.5 * w]) for n, y, p in VIEWS]
    # Material-coordinate texture frequencies in renderer are tuned for ~cm objects; scale so stripes are visible.
    rgb = np.zeros((len(cams), len(disp), 3, IMAGE, IMAGE), np.uint8)
    mask = np.zeros((len(cams), len(disp), 1, FIELD, FIELD), np.uint8)
    for v, cam in enumerate(cams):
        for t, d in enumerate(disp_s):
            r = rasterize(rest_s, rest_s + d, tri, cam, case['texture_seed'])
            rgb[v, t] = r['rgb'].transpose(2, 0, 1)
            mask[v, t, 0] = r['mask'].reshape(FIELD, IMAGE // FIELD, FIELD, IMAGE // FIELD).max(axis=(1, 3))
    return rgb, mask * 255, cams, surface, disp_s


def run_one(split, idx, root, young=None, check=False):
    out = root / split / f'{idx:06d}'
    if (out / 'result.json').exists() and young is None:
        return 'skip'
    case = case_for(split, idx)
    if young is not None:
        case['young_pa'] = float(young)
    ws = ROOT / '.cache/v3a_uipc_ws' / f'{split}_{idx:06d}_{os.getpid()}'
    started = time.time()
    try:
        out.mkdir(parents=True, exist_ok=True)
        rest, tets, disp, loads, health = simulate(case, ws)
        t_sim = time.time() - started
        rgb, mask, cams, surface, disp_s = render(case, rest, tets, disp)
        z = np.zeros((rgb.shape[0], rgb.shape[1], 3, FIELD, FIELD), np.uint8)
        tmp = out / 'sample_pack.tmp.npz'
        np.savez_compressed(tmp, rgb=rgb, object_mask=mask, flow=z, stress=z, force_mask=z,
                            timestamps=np.arange(FRAMES, dtype=np.float32) * FRAME_DT)
        tmp.replace(out / 'sample_pack.npz')
        tip = np.isclose(rest[:, 0], case['dimensions_m'][0])
        tip_y = disp[:, tip, 1].mean(axis=1)
        np.savez_compressed(out / 'traj.npz', surface_rest_m=rest[surface].astype(np.float32),
                            surface_displacement_m=disp_s.astype(np.float32), tip_y_m=tip_y, force_n=loads)
        px = IMAGE / cams[0]['field_of_view_m']
        motion = dict(tip_peak_deflection_m=float(-tip_y.min()), tip_peak_deflection_over_L=float(-tip_y.min() / case['dimensions_m'][0]),
                      tip_peak_px_side=float(-tip_y.min() * px), tip_final_px_side=float(abs(tip_y[-1]) * px),
                      tip_ptp_after_release_px=float(np.ptp(tip_y[int(1.3 / FRAME_DT):]) * px),
                      rgb_absdiff_vs_first_mean=float(np.abs(rgb.astype(np.float32) - rgb[:, :1]).mean()),
                      rgb_absdiff_vs_first_max_t=float(np.abs(rgb.astype(np.float32) - rgb[:, :1]).mean(axis=(0, 2, 3, 4)).max()))
        gt = dict(engine='libuipc', engine_version='0.0.29', law='elastic', law_id=0, geometry=case['geometry'],
                  object=f"uipc_{case['geometry']}", regression=dict(E=case['young_pa'], nu=case['poisson'], density=RHO),
                  action='tip_load_hold_release', case=case, cameras=cams, timestamps_seconds=(np.arange(FRAMES) * FRAME_DT).tolist(),
                  constitutive_model='StableNeoHookean (SNK1, lambda corrected to lam+mu)')
        atomic(out / 'gt.json', gt)
        result = dict(exit=0, health=health, motion=motion, sim_seconds=t_sim, wall_seconds=time.time() - started,
                      pack_bytes=(out / 'sample_pack.npz').stat().st_size)
        if check:
            from PIL import Image
            sheet = np.concatenate([np.concatenate([rgb[v, t].transpose(1, 2, 0) for t in (0, 8, 16, 32, 36, 40, 48, 63)], 1)
                                    for v in range(len(cams))], 0)
            Image.fromarray(sheet).save(ROOT / '.cache' / f'v3a_check_{split}_{idx}_{int(case["young_pa"])}.png')
        atomic(out / 'result.json', result)
        return result
    except Exception as exc:
        atomic(out / 'error.json', dict(exit=1, error=repr(exc), traceback=traceback.format_exc(), case=case))
        raise
    finally:
        shutil.rmtree(ws, ignore_errors=True)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--split', required=True, choices=list(COUNTS)); a.add_argument('--shard', type=int, default=0)
    a.add_argument('--num-shards', type=int, default=1); a.add_argument('--indices', default='')
    a.add_argument('--young', default='', help='check mode: comma list of E overriding the sampled one')
    a.add_argument('--check', action='store_true'); a.add_argument('--root', default='data/v3_uipc')
    args = a.parse_args(); os.chdir(ROOT); root = ROOT / args.root
    indices = [int(i) for i in args.indices.split(',')] if args.indices else \
        [i for i in range(COUNTS[args.split]) if i % args.num_shards == args.shard]
    failures = 0
    for i in indices:
        for young in ([float(y) for y in args.young.split(',')] if args.young else [None]):
            try:
                r = run_one(args.split, i, root, young, args.check)
                print(json.dumps({'split': args.split, 'index': i, 'young': young, 'result': r}), flush=True)
            except Exception as exc:
                failures += 1; print(f'FAILED {args.split}/{i}: {exc!r}', flush=True)
    print(json.dumps({'done': len(indices), 'failures': failures}), flush=True)
    sys.exit(1 if failures > len(indices) // 4 else 0)


if __name__ == '__main__':
    main()
