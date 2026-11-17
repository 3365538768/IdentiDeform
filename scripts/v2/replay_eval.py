"""Replay test: are predicted parameters good enough to simulate an *unseen* interaction?

For each P4c low-damping validation sample, the observed action is swapped (press <-> tap) and the object is
simulated three times with identical physics (g=0, rpic 0, grid 0.99999, same views):
  gt   : true E, rho, nu, sigma_y, hardening/softening (regenerated deterministically by gen_v2.sample)
  pred : model prediction decoded from the identifiable space (rho fixed to 1000, so E = E/rho specific value)
  lawmean : train mean of the *true* law (privileged reference, no video information)
pred/lawmean use hardening=softening=0.25 (not predicted). Predictions come from the best validation epoch
of runs/<run> (selection used validation MAE only).
  .venv/bin/python scripts/v2/replay_eval.py --run p4d_Clow_seed17 --shard i --num-shards n
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'scripts/v2'), str(ROOT / 'scripts'), str(ROOT / 'src')]
from gen_v2 import ASSETS, LAWS, TEMPLATE, V1_RUN_PARAMETERS, prune_pngs, sample  # noqa: E402
from upstream_cli import build  # noqa: E402
from r1_report import best_dir  # noqa: E402
from fieldbridge.data import ParameterCodec, specific_parameters, transform_parameters  # noqa: E402

SWAP = {'press': 'tap', 'tap': 'press'}


def law_means(manifest, stats):
    codec = ParameterCodec(stats); sums = {}
    for r in manifest:
        if r['split'] != 'train':
            continue
        z, m = codec.encode(specific_parameters(r['parameters']))
        s = sums.setdefault(r['law_id'], [torch.zeros(4), torch.zeros(4)]); s[0] += torch.where(m, z, 0); s[1] += m
    return {k: v[0] / v[1].clamp(min=1) for k, v in sums.items()}


def candidate(law, values):
    e, nu, _, sigma = [float(v) for v in values]
    p = {'material': law, 'E': e, 'nu': float(np.clip(nu, 0.05, 0.48)), 'density': 1000.0}
    if law != 'jelly':
        p.update(yield_stress=sigma, hardening=0.25)
        if law == 'plasticine':
            p['softening'] = 0.25
    return p


def main():
    a = argparse.ArgumentParser(); a.add_argument('--run', required=True); a.add_argument('--shard', type=int, default=0)
    a.add_argument('--num-shards', type=int, default=1); args = a.parse_args()
    manifest = [json.loads(l) for l in (ROOT / 'data/manifests/v2pilot_low.jsonl').read_text().splitlines() if l.strip()]
    stats = json.loads((ROOT / 'data/manifests/v2pilot_low.specific.stats.json').read_text()); codec = ParameterCodec(stats)
    means = law_means(manifest, stats)
    pred_dir, _ = best_dir(ROOT / f'runs/{args.run}')
    preds = [json.loads(l) for l in (pred_dir / 'predictions.jsonl').read_text().splitlines() if l.strip()]
    rp = json.loads(V1_RUN_PARAMETERS.read_text())
    for n, pr in enumerate(sorted(preds, key=lambda r: r['sample_id'])):
        if n % args.num_shards != args.shard:
            continue
        index = int(pr['sample_id'].rsplit('/', 1)[1])
        law, action, gt_params, _, rng = sample(index, 'val')
        obj = json.loads((ROOT / f'data/v2_pilot/p4c_low/val/{index:06d}/gt.json').read_text())['object']
        cands = {'gt': gt_params,
                 'pred': candidate(LAWS[pr['predicted_law_id']], codec.decode(torch.tensor(pr['prediction_z']))),
                 'lawmean': candidate(law, codec.decode(means[LAWS.index(law)]))}
        for name, params in cands.items():
            dst = ROOT / f'data/v2_replay/{args.run}/{index:06d}/{name}'
            if (dst / 'result.json').exists() and json.loads((dst / 'result.json').read_text()).get('exit') == 0:
                continue
            shutil.rmtree(dst, ignore_errors=True); dst.mkdir(parents=True)
            cfg = json.loads(TEMPLATE.read_text())
            cfg.update(params, rpic_damping=0.0, grid_v_damping_scale=0.99999, substep_dt=1e-4, frame_dt=0.01, frame_num=400, g=[0.0, 0.0, 0])
            (dst / 'config.json').write_text(json.dumps(cfg, indent=1))
            (dst / 'replay.json').write_text(json.dumps({'sample_id': pr['sample_id'], 'observed_action': action,
                                                        'replay_action': SWAP[action], 'object': obj, 'params': params}, indent=1))
            cli = build(rp, ASSETS / f'{obj}.ply', dst / 'config.json', dst,
                        {'sim_type': 'press', 'render_outputs_per_sim_second': 32.0, 'compile_video': False,
                         'delete_png_sequences_after_compile_video': False, 'random_render_views': True,
                         'random_render_views_seed': index})
            cli += ['--num_views', '3', '--num_render_views', '3']
            env = dict(os.environ, FB_ACTION=SWAP[action]) if SWAP[action] == 'tap' else {k: v for k, v in os.environ.items() if k != 'FB_ACTION'}
            started = time.time()
            with (dst / 'run.log').open('w') as log:
                code = subprocess.call([str(ROOT / 'scripts/v2/run_upstream_v2.sh'), *cli], stdout=log, stderr=subprocess.STDOUT, env=env)
            if code == 0:
                prune_pngs(dst)
            (dst / 'result.json').write_text(json.dumps({'exit': code, 'seconds': time.time() - started}))
            print(index, name, code, round(time.time() - started), flush=True)


if __name__ == '__main__':
    main()
