"""V3e gripper task: squeeze simulation wrapper, parameter sets per method, and the damage metric.

simulate(): one upstream Warp-MPM run of the parallel-jaw squeeze (scripts/v3e/squeeze_main.py) at closing
depth d, same object/physics/time step as V2 main and C1 (low damping, gravity, density 1000 + specific moduli
for twins; GT config verbatim). Rendering is off unless render=True (qualitative figures only).
Failure rule as C1: a crash within 60 s is retried once; a repeated or mid-run crash is a failed simulation.
"""
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/v3b'), str(ROOT / 'scripts/v2'), str(ROOT / 'scripts'), str(ROOT / 'upstream/DeformFieldBench')]
from gen_v2 import ASSETS, LAWS, TEMPLATE, V1_RUN_PARAMETERS  # noqa: E402
from upstream_cli import build  # noqa: E402

QUICK_FAIL_SECONDS, QUICK_RETRIES = 60, 5  # CUDA error 700 in the first frames is intermittent (ice_telescope)
DEPTHS = [0.06, 0.10, 0.14, 0.18, 0.24, 0.32]   # candidate closing depths (fraction of width W)
D_MIN = 0.06                                  # minimum depth for a secure grasp
TAU = 0.03                                    # tolerance on permanent crush (width loss, fraction of W)


GRAVITY = 0.0  # task runs without gravity: under V2-main gravity most elastoplastic objects slump to a puddle in <0.5 s
              # (pilot, docs/v3e/figs/pilot_particles_gravity.png), which makes a grasp ill-posed


def config_for(params):
    cfg = json.loads(TEMPLATE.read_text())
    cfg.update(params, rpic_damping=0.0, grid_v_damping_scale=0.99999, substep_dt=1e-4, frame_dt=0.01, frame_num=400,
               g=[0.0, 0.0, -GRAVITY])
    return cfg


def simulate(dst, params, obj, d, index, render=False):
    """Returns (ok, seconds). On success dst/traj.npz and dst/squeeze_bc.json exist."""
    shutil.rmtree(dst, ignore_errors=True); dst.mkdir(parents=True)
    (dst / 'config.json').write_text(json.dumps(config_for(params), indent=1))
    rp = json.loads(V1_RUN_PARAMETERS.read_text())
    over = {'sim_type': 'press', 'render_outputs_per_sim_second': 32.0 if render else 4.0, 'compile_video': False,
            'delete_png_sequences_after_compile_video': False, 'random_render_views': True,
            'random_render_views_seed': index, 'camera_distance_scale': 0.65, 'pack_sample_pack': False,
            'output_view_stress_gaussian': False, 'output_view_flow_gaussian': False, 'output_view_force_mask': False,
            'output_view_object_mask': bool(render), 'compress_render_pngs': False, 'pack_arch4_lmdb': False,
            'render_export_max_side': 0, 'render_export_scale': 1.0, 'render_img': bool(render), 'output_h5': True}
    cli = build(rp, ASSETS / f'{obj}.ply', dst / 'config.json', dst, over)
    cli += ['--num_views', '3', '--num_render_views', '3' if render else '1', '--no_compress_render_pngs']
    env = dict(os.environ); env.pop('FB_ACTION', None); env['FB_SQUEEZE_D'] = repr(float(d))
    started = time.time()
    for attempt in range(QUICK_RETRIES + 1):
        launched = time.time()
        with (dst / 'run.log').open('w') as log:
            code = subprocess.call([str(ROOT / 'scripts/v3e/run_squeeze.sh'), *cli], stdout=log, stderr=subprocess.STDOUT, env=env)
        if code == 0 or time.time() - launched > QUICK_FAIL_SECONDS:
            break
        print('quick failure', dst, attempt, flush=True)
        if attempt < QUICK_RETRIES:
            time.sleep(20)
    ok = code == 0 and (dst / 'traj.npz').exists()
    if not ok:
        print('simulation failed', dst, code, flush=True)
    return ok, time.time() - started


def procrustes(src, dst):
    """Rigid (R,t) minimising |R src + t - dst|; returns aligned src."""
    ms, md = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - ms).T @ (dst - md))
    s = np.eye(3); s[2, 2] = np.sign(np.linalg.det(u @ vt))
    r = (u @ s @ vt).T
    return (src - ms) @ r.T + md


def measure(sq, ref, bc):
    """Damage of a squeeze run vs the d=0 reference run of the SAME parameters (cancels gravity sag/settling).
    dent  = P95 over particles of |rigidly aligned final-mean position - reference final-mean position| / W
    wloss = permanent width loss along the squeeze axis (1-99 % quantile span), fraction of W
    peak  = max compression of the width during the hold (contact check), fraction of W."""
    W = bc['width']; a = bc['axis']
    xs, xr = sq['x_tail'].astype(np.float64), ref['x_tail'].astype(np.float64)
    al = procrustes(xs, xr); res = np.linalg.norm(al - xr, axis=1)
    span = lambda x: float(np.subtract(*np.quantile(x[:, a], [0.99, 0.01])))
    w0 = sq['width'][0]
    crush = float((span(xr) - span(al)) / W)  # PRIMARY: permanent width loss after rigid alignment
    x0 = sq['x0'].astype(np.float64); r0 = np.linalg.norm(procrustes(xs, x0) - x0, axis=1)
    return {'crush': crush, 'dent_p95_vs_x0': float(np.quantile(r0, .95) / W), 'dent_mean_vs_x0': float(r0.mean() / W),
            'wloss_vs_x0': float((span(x0) - span(xs)) / W),'dent_p95': float(np.quantile(res, .95) / W), 'dent_mean': float(res.mean() / W),
            'wloss': float((span(xr) - span(xs)) / W), 'peak_compression': float((w0 - sq['width'].min()) / W),
            'final_width_vs_ref': float((ref['width'][-50:].mean() - sq['width'][-50:].mean()) / W),
            'zmax_drop': float((ref['zmax'][-50:].mean() - sq['zmax'][-50:].mean()) / W),
            'n_particles': int(len(xs)), 'W': W, 'axis': a}


# ----------------------------------------------------------------------------------------- parameter sets
def _c1():
    import c1_refine
    return c1_refine


def gt_params(index):
    g = json.loads((ROOT / f'data/v2_main/val/{index:06d}/config.json').read_text())
    return {k: g[k] for k in ('material', 'E', 'nu', 'density', 'yield_stress', 'hardening', 'softening') if k in g}


def twin_params(index, method):
    """Material twin of each method (object geometry is the observed one for every method)."""
    c1 = _c1()
    sid = f'v2main/val/{index:06d}'
    base = ROOT / f'data/v3b_refine/v2main_C_seed29/{index:06d}'
    if method == 'gt':
        p = gt_params(index); return p, {'law': p['material']}
    if method in ('refined_top2', 'refined_best3', 'ensemble_prior'):
        arm, k = {'refined_top2': ('prior_top2', '12'), 'refined_best3': ('lawmean_best3', '12'),
                  'ensemble_prior': ('prior_top2', '0')}[method]
        st = json.loads((base / arm / 'state.json').read_text())
        b = st['budgets'][k]; law, x = b['law'], b['x']
        nu = c1.decode(b['prediction_z'])[1]
        return c1.candidate(law, x[0], x[1], nu), {'law': law, 'x': x, 'sims_refine': b['sims'],
                                                   'refine_wall_seconds': b['wall_seconds']}
    if method == 'lawmean_pred':
        ens = json.loads((ROOT / 'docs/v3b/ENSEMBLE_BCD9_VAL.json').read_text())['samples'][sid]
        votes = ens['law_posterior_votes']; means = c1.law_means()
        order = sorted(range(3), key=lambda k: (-votes[k], abs(means[LAWS[k]][0] - ens['prediction_z'][0])))
        law = LAWS[order[0]]; e, nu, _, s = c1.decode(means[law])
        return c1.candidate(law, math.log10(e), math.log10(max(s, 1e-3)), nu), {'law': law}
    raise ValueError(method)
