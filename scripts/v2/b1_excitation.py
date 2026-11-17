"""B1: which excitation makes jelly E/rho visible on real (3DGS) objects? Low damping, gravity on.

For each action and object, two simulations differ only in E/rho by sqrt(10) (base 10^2.75 or 10^3.75 in
units of rho/1000; rho=1000, nu=0.3). Everything else (object, views, damping, gravity) is identical, so any
rendered difference is caused by stiffness. 32 fps x 4 s.
  .venv/bin/python scripts/v2/b1_excitation.py --shard i --num-shards n
Output: data/v2_pilot/b1/<action>/<object>_<base>_<a|b>/
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v2'))
from gen_v2 import ASSETS, TEMPLATE, V1_RUN_PARAMETERS, prune_pngs  # noqa: E402
from upstream_cli import build  # noqa: E402

ACTIONS = ['press', 'drop', 'shear', 'stretch', 'bend', 'tap30hold']
BASES = [2.75, 3.75]


def jobs():
    rows = [json.loads(l) for l in (ROOT / 'data/manifests/deformfield_v1.jsonl').read_text().splitlines() if l.strip()]
    objects = sorted({r['object_id'] for r in rows if r['split'] == 'train'})
    objects = random.Random(11).sample(objects, 10)
    return [(a, o, b, k) for a in ACTIONS for o in objects for b in BASES for k in ('a', 'b')]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--shard', type=int, default=0); p.add_argument('--num-shards', type=int, default=1)
    args = p.parse_args(); rp = json.loads(V1_RUN_PARAMETERS.read_text())
    for i, (action, obj, base, k) in enumerate(jobs()):
        if i % args.num_shards != args.shard:
            continue
        dst = ROOT / f'data/v2_pilot/b1/{action}/{obj}_{base}_{k}'
        if (dst / 'result.json').exists() and json.loads((dst / 'result.json').read_text()).get('exit') == 0:
            continue
        dst.mkdir(parents=True, exist_ok=True)
        spec = 10 ** (base + (0.5 if k == 'b' else 0.0))
        cfg = json.loads(TEMPLATE.read_text())
        cfg.update(material='jelly', E=spec, nu=0.3, density=1000.0, rpic_damping=0.0, grid_v_damping_scale=0.99999,
                   substep_dt=1e-4, frame_dt=0.01, frame_num=400, g=[0.0, 0.0, -9.8])
        (dst / 'config.json').write_text(json.dumps(cfg, indent=1))
        (dst / 'gt.json').write_text(json.dumps({'object': obj, 'action': action, 'base': base, 'member': k,
                                                 'E_over_rho_specific': spec, 'gravity': -9.8}, indent=1))
        sim_type = 'press' if action == 'tap30hold' else action
        cli = build(rp, ASSETS / f'{obj}.ply', dst / 'config.json', dst,
                    {'sim_type': sim_type, 'render_outputs_per_sim_second': 32.0, 'compile_video': False,
                     'delete_png_sequences_after_compile_video': False, 'random_render_views': True,
                     'random_render_views_seed': 7})
        cli += ['--num_views', '3', '--num_render_views', '3']
        env = dict(os.environ); env.pop('FB_ACTION', None)
        if action == 'tap30hold':
            env['FB_ACTION'] = 'tap30hold'
        started = time.time()
        with (dst / 'run.log').open('w') as log:
            code = subprocess.call([str(ROOT / 'scripts/v2/run_upstream_v2.sh'), *cli], stdout=log, stderr=subprocess.STDOUT, env=env)
        (dst / 'result.json').write_text(json.dumps({'exit': code, 'seconds': time.time() - started}))
        if code == 0:
            prune_pngs(dst)
        print(action, obj, base, k, code, round(time.time() - started), flush=True)


if __name__ == '__main__':
    main()
