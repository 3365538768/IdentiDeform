"""V2 main data: low damping (identical for all laws), gravity on, six actions, camera scale 0.65.

  .venv/bin/python scripts/v2/gen_v2_main.py --split train --count 2000 --shard i --num-shards n
Parameters as in gen_v2.sample (E/rho, sigma_y/rho log-uniform; rho independent; plasticine sigma censoring).
Actions cycle press/drop/shear/stretch/bend/tap30hold; laws cycle jelly/metal/plasticine. Objects keep their
V1 split. Camera scale 0.65 passed the Quality-V2 coverage gate (no empty/border frames, p10 area >=2 %).
Output data/v2_main/<split>/<index>/ ; result.json is written last.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v2'))
from gen_v2 import ASSETS, LAWS, TEMPLATE, V1_RUN_PARAMETERS, prune_pngs, sample  # noqa: E402
from upstream_cli import build  # noqa: E402

ACTIONS = ['press', 'drop', 'shear', 'stretch', 'bend', 'tap30hold']


def main():
    a = argparse.ArgumentParser(); a.add_argument('--split', choices=['train', 'val', 'test'], required=True)
    a.add_argument('--count', type=int, required=True); a.add_argument('--shard', type=int, default=0)
    a.add_argument('--num-shards', type=int, default=1); args = a.parse_args()
    rows = [json.loads(l) for l in (ROOT / 'data/manifests/deformfield_v1.jsonl').read_text().splitlines() if l.strip()]
    objects = sorted({r['object_id'] for r in rows if r['split'] == args.split})
    rp = json.loads(V1_RUN_PARAMETERS.read_text())
    for i in range(args.shard, args.count, args.num_shards):
        law, _, params, labels, rng = sample(i, f'main-{args.split}')
        action = ACTIONS[(i // 3) % len(ACTIONS)]
        obj = objects[rng.randrange(len(objects))]
        dst = ROOT / f'data/v2_main/{args.split}/{i:06d}'
        if (dst / 'result.json').exists() and json.loads((dst / 'result.json').read_text()).get('exit') == 0:
            continue
        shutil.rmtree(dst, ignore_errors=True); dst.mkdir(parents=True)
        cfg = json.loads(TEMPLATE.read_text())
        cfg.update(params, rpic_damping=0.0, grid_v_damping_scale=0.99999, substep_dt=1e-4, frame_dt=0.01, frame_num=400,
                   g=[0.0, 0.0, -9.8])
        (dst / 'config.json').write_text(json.dumps(cfg, indent=1))
        gt = {'object': obj, 'action': action, 'law': law, 'law_id': LAWS.index(law), 'arm': 'v2main', 'split': args.split,
              'index': i, 'regression': {'E': params['E'], 'nu': params['nu'], 'density': params['density'],
                                         'yield_stress': params.get('yield_stress', 0.0)}, 'labels': labels,
              'hardening': params.get('hardening'), 'softening': params.get('softening'),
              'damping': {'rpic_damping': 0.0, 'grid_v_damping_scale': 0.99999}, 'gravity': -9.8, 'fps': 32,
              'camera_distance_scale': 0.65}
        (dst / 'gt.json').write_text(json.dumps(gt, indent=1))
        cli = build(rp, ASSETS / f'{obj}.ply', dst / 'config.json', dst,
                    {'sim_type': 'press' if action == 'tap30hold' else action, 'render_outputs_per_sim_second': 32.0,
                     'compile_video': False, 'delete_png_sequences_after_compile_video': False, 'random_render_views': True,
                     'random_render_views_seed': i, 'camera_distance_scale': 0.65})
        cli += ['--num_views', '3', '--num_render_views', '3']
        env = dict(os.environ); env.pop('FB_ACTION', None)
        if action == 'tap30hold':
            env['FB_ACTION'] = 'tap30hold'
        started = time.time()
        with (dst / 'run.log').open('w') as log:
            code = subprocess.call([str(ROOT / 'scripts/v2/run_upstream_v2.sh'), *cli], stdout=log, stderr=subprocess.STDOUT, env=env)
        if code == 0:
            prune_pngs(dst)
        (dst / 'result.json').write_text(json.dumps({'exit': code, 'seconds': time.time() - started}))
        print(args.split, i, law, action, code, round(time.time() - started), flush=True)


if __name__ == '__main__':
    main()
