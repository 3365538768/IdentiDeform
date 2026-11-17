"""V3 main data = V2 main protocol (same per-index parameters, objects, actions, cameras) with corrected contact setup:
every action except drop has a floor under the object and the press plate starts above it (FB_BC_FIX=1 in
scripts/v2/upstream_main.py).

  .venv/bin/python scripts/v3n/gen_v3.py --split val --indices 0-282

V2 docstring follows.
V2 main data: low damping (identical for all laws), gravity on, six actions, camera scale 0.65.

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
    a.add_argument('--indices', required=True)  # e.g. 0-282 or 5,12,40
    a.add_argument('--root', default='data/v3_main'); a.add_argument('--ranges', default='v3', choices=['v2', 'v3', 'v5'])
    a.add_argument('--press-mode', default='gsmall', choices=['sticky', 'gafter', 'gsmall']); a.add_argument('--g-small', type=float, default=-2.0)
    a.add_argument('--objects', default='v3', choices=['v1', 'v3']); a.add_argument('--e-scale', type=float, default=1.0); a.add_argument('--press-force', type=float, default=None); a.add_argument('--weak', default=None, choices=[None, 'hold']); a.add_argument('--release', default=None, choices=[None, 'fast']); a.add_argument('--g-all', type=float, default=None); a.add_argument('--g-drop', type=float, default=-2.0)
    args = a.parse_args()
    idx = [int(x) for x in args.indices.split(',')] if '-' not in args.indices else list(range(int(args.indices.split('-')[0]), int(args.indices.split('-')[1]) + 1))
    if args.ranges == 'v3':  # fewer samples that splat on impact or flow like sand; same random draws as V2
        import gen_v2
        gen_v2.SPECIFIC_E.update(jelly=(3.5, 6.0), plasticine=(3.5, 5.5), metal=(4.5, 6.0))
        gen_v2.SPECIFIC_SIGMA.update(plasticine=(3.2, 4.5), metal=(3.2, 4.5))  # yield must hold the object's own weight at g = 2
    if args.ranges == 'v5':  # classes separated again (plasticine softer and weaker than metal), yield still carries the weight at g = 2
        import gen_v2
        gen_v2.SPECIFIC_E.update(jelly=(3.0, 5.5), plasticine=(3.0, 4.5), metal=(4.5, 6.0))
        gen_v2.SPECIFIC_SIGMA.update(plasticine=(2.5, 4.0), metal=(3.0, 4.5))
    rows = [json.loads(l) for l in (ROOT / 'data/manifests/deformfield_v1.jsonl').read_text().splitlines() if l.strip()]
    objects = sorted({r['object_id'] for r in rows if r['split'] == args.split})
    if args.objects == 'v3':  # screened objects (data/manifests/v3_split.json)
        objects = json.loads((ROOT / 'data/manifests/v3_split.json').read_text())[args.split]
    rp = json.loads(V1_RUN_PARAMETERS.read_text())
    for i in idx:
        law, _, params, labels, rng = sample(i, f'main-{args.split}')
        if args.ranges in ('v3', 'v5') and 'sigma_over_rho_specific' in labels:  # censored only if the action cannot make it yield
            labels['censored_sigma'] = bool(labels['sigma_over_rho_specific'] > 0.5 * labels['E_over_rho_specific'])
        action = ACTIONS[(i // 3) % len(ACTIONS)]
        obj = objects[rng.randrange(len(objects))]
        dst = ROOT / f'{args.root}/{args.split}/{i:06d}'
        if (dst / 'result.json').exists() and json.loads((dst / 'result.json').read_text()).get('exit') == 0:
            continue
        shutil.rmtree(dst, ignore_errors=True); dst.mkdir(parents=True)
        cfg = json.loads(TEMPLATE.read_text())
        # gravity only for drop; press variants may add gravity to keep the object on the floor after release
        if args.e_scale != 1.0:  # sensitivity probe: same sample with E (and so E/rho) scaled
            params['E'] *= args.e_scale; labels['E_over_rho_specific'] *= args.e_scale
        gz = args.g_all if args.g_all is not None else args.g_drop if action == 'drop' else (args.g_small if args.press_mode == 'gsmall' and action in ('press', 'tap30hold') else 0.0)
        cfg.update(params, rpic_damping=0.0, grid_v_damping_scale=0.99999, substep_dt=1e-4, frame_dt=0.01, frame_num=400,
                   g=[0.0, 0.0, gz])
        (dst / 'config.json').write_text(json.dumps(cfg, indent=1))
        gt = {'object': obj, 'action': action, 'law': law, 'law_id': LAWS.index(law), 'arm': 'v3main', 'contact_fix': True, 'split': args.split,
              'index': i, 'regression': {'E': params['E'], 'nu': params['nu'], 'density': params['density'],
                                         'yield_stress': params.get('yield_stress', 0.0)}, 'labels': labels,
              'hardening': params.get('hardening'), 'softening': params.get('softening'),
              'damping': {'rpic_damping': 0.0, 'grid_v_damping_scale': 0.99999}, 'gravity': gz, 'ranges': args.ranges, 'press_force': args.press_force if action == 'press' else None, 'weak_design': args.weak if action in ('shear', 'stretch', 'bend') else None, 'release': args.release if action in ('press', 'tap30hold') else None, 'objects_split': args.objects, 'press_mode': args.press_mode if action in ('press', 'tap30hold') else None, 'fps': 32,
              'camera_distance_scale': 0.65}
        (dst / 'gt.json').write_text(json.dumps(gt, indent=1))
        cli = build(rp, ASSETS / f'{obj}.ply', dst / 'config.json', dst,
                    {'sim_type': 'press' if action == 'tap30hold' else action, 'render_outputs_per_sim_second': 32.0,
                     'compile_video': False, 'delete_png_sequences_after_compile_video': False, 'random_render_views': True,
                     'random_render_views_seed': i, 'camera_distance_scale': 0.65})
        cli += ['--num_views', '3', '--num_render_views', '3']
        env = dict(os.environ); env.pop('FB_ACTION', None); env['FB_BC_FIX'] = '1'; env['FB_PRESS_MODE'] = args.press_mode
        env['FB_G_ON_AT'] = '1.0' if action == 'tap30hold' else '3.0'; env.setdefault('FB_DROP_GAP', '0.25')
        if args.release:
            env['FB_RELEASE'] = args.release
        if args.weak:  # redesigned shear/stretch/bend (contact_bc: shake / hang / cantilever)
            env['FB_WEAK'] = args.weak
        if args.press_force and action == 'press':  # force-controlled press (contact_bc: admittance plate)
            env['FB_PRESS_FORCE'] = repr(args.press_force)
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
