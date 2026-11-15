"""P4c: paired Data-V2 pilot. Same objects/actions/parameters in both arms; arms differ only in damping.

  .venv/bin/python scripts/v2/gen_v2.py --arm low --split train --count 480 --shard i --num-shards n
Arms: 'low' (rpic 0, grid 0.99999 per substep, identical for all laws), 'v1' (per-law dataset damping).
Parameters are drawn in identifiable form: E/rho and sigma_y/rho log-uniform over the ranges P4b found
visible, rho log-uniform independent of them; plasticine sigma_y/rho above 10^3.75 is flagged censored.
Actions: V1 press, or tap (10 % over 0.5 s, release). 32 fps x 4 s = 128 frames. Objects keep their V1 split.
"""
import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v2'))
from upstream_cli import build  # noqa: E402

TEMPLATE = ROOT / 'upstream/DeformFieldBench/simulation/config/press_cube_jelly.json'
ASSETS = ROOT / 'data/geometry/kaggle_v2/3DGS_model/3DGS_model'
# Renderer/camera settings of the DeformFieldBench v1 dataset (shipped in this repo).
V1_RUN_PARAMETERS = ROOT / 'configs/sim/run_parameters.json'
LAWS = ['jelly', 'metal', 'plasticine']          # law_id order of the V1 manifest
V1_DAMPING = {'jelly': (0.35, 0.999), 'plasticine': (0.45, 0.9992), 'metal': (0.25, 0.9995)}
# log10 of value / (rho / 1000)
SPECIFIC_E = {'jelly': (2.0, 5.0), 'plasticine': (2.0, 4.5), 'metal': (4.0, 6.0)}
SPECIFIC_SIGMA = {'plasticine': (0.0, 4.0), 'metal': (0.0, 4.5)}
CENSOR = {'plasticine': 3.75}


# Rendered PNG sequences are only an intermediate for sample_pack.npz; ~1500 small files per sample
# overload the shared GPFS workspace (and editors watching it), so they are removed once the pack exists.
PNG_DIRS = ('images', 'flow_gaussian', 'stress_gaussian', 'object_mask', 'force_mask')


def prune_pngs(sample_dir):
    for pack in sample_dir.glob('*/sample_pack.npz'):
        for name in PNG_DIRS:
            shutil.rmtree(pack.parent / name, ignore_errors=True)


def sample(index, split):
    rng = random.Random(f'p4c-{split}-{index}')
    law = LAWS[index % 3]; action = ['press', 'tap'][(index // 3) % 2]
    rho = 10 ** rng.uniform(math.log10(300), math.log10(3000))
    se = 10 ** rng.uniform(*SPECIFIC_E[law])
    p = {'material': law, 'E': se * rho / 1000., 'nu': rng.uniform(0.1, 0.46), 'density': rho}
    labels = {'E_over_rho_specific': se, 'censored_sigma': False}
    if law in SPECIFIC_SIGMA:
        ss = 10 ** rng.uniform(*SPECIFIC_SIGMA[law])
        p['yield_stress'] = ss * rho / 1000.; labels['sigma_over_rho_specific'] = ss
        labels['censored_sigma'] = law in CENSOR and math.log10(ss) > CENSOR[law]
        p['hardening'] = rng.uniform(0.05, 0.45)
        if law == 'plasticine':
            p['softening'] = rng.uniform(0.05, 0.45)
    return law, action, p, labels, rng


def main():
    a = argparse.ArgumentParser(); a.add_argument('--arm', choices=['low', 'v1'], required=True)
    a.add_argument('--split', choices=['train', 'val'], required=True); a.add_argument('--count', type=int, required=True)
    a.add_argument('--shard', type=int, default=0); a.add_argument('--num-shards', type=int, default=1); args = a.parse_args()
    rows = [json.loads(l) for l in (ROOT / 'data/manifests/deformfield_v1.jsonl').read_text().splitlines() if l.strip()]
    objects = sorted({r['object_id'] for r in rows if r['split'] == args.split})
    rp = json.loads(V1_RUN_PARAMETERS.read_text())
    out = ROOT / f'data/v2_pilot/p4c_{args.arm}/{args.split}'
    for i in range(args.shard, args.count, args.num_shards):
        law, action, params, labels, rng = sample(i, args.split)
        obj = objects[rng.randrange(len(objects))]
        dst = out / f'{i:06d}'
        # result.json is written last, so a pack from an interrupted run is never mistaken for a finished one.
        if (dst / 'result.json').exists() and json.loads((dst / 'result.json').read_text()).get('exit') == 0:
            continue
        shutil.rmtree(dst, ignore_errors=True)
        dst.mkdir(parents=True, exist_ok=True)
        rpic, grid = (0.0, 0.99999) if args.arm == 'low' else V1_DAMPING[law]
        cfg = json.loads(TEMPLATE.read_text())
        cfg.update(params, rpic_damping=rpic, grid_v_damping_scale=grid, substep_dt=1e-4, frame_dt=0.01, frame_num=400, g=[0.0, 0.0, 0])
        (dst / 'config.json').write_text(json.dumps(cfg, indent=1))
        gt = {'object': obj, 'action': action, 'law': law, 'law_id': LAWS.index(law), 'arm': args.arm, 'split': args.split,
              'index': i, 'regression': {'E': params['E'], 'nu': params['nu'], 'density': params['density'],
                                         'yield_stress': params.get('yield_stress', 0.0)}, 'labels': labels,
              'damping': {'rpic_damping': rpic, 'grid_v_damping_scale': grid}, 'fps': 32}
        (dst / 'gt.json').write_text(json.dumps(gt, indent=1))
        cli = build(rp, ASSETS / f'{obj}.ply', dst / 'config.json', dst,
                    {'sim_type': 'press', 'render_outputs_per_sim_second': 32.0, 'compile_video': False,
                     'delete_png_sequences_after_compile_video': False, 'random_render_views': True,
                     'random_render_views_seed': i})
        cli += ['--num_views', '3', '--num_render_views', '3']
        started = time.time()
        with (dst / 'run.log').open('w') as log:
            code = subprocess.call([str(ROOT / 'scripts/v2/run_upstream_v2.sh'), *cli], stdout=log, stderr=subprocess.STDOUT,
                                   env=dict(os.environ, FB_ACTION=action))
        (dst / 'result.json').write_text(json.dumps({'exit': code, 'seconds': time.time() - started}))
        if code == 0:
            prune_pngs(dst)
        print(args.arm, args.split, i, law, action, code, round(time.time() - started), flush=True)


if __name__ == '__main__':
    main()
