"""P4a: re-simulate V1 samples with the upstream pipeline at a higher render rate (same params/action/object).

  .venv/bin/python scripts/v2/replay_v1.py --per-cell 2 --fps 32 --out data/v2_pilot/replay32 [--shard i --num-shards n]
Picks --per-cell val samples per (law, action) with a fixed seed, writes per-sample config.json + cli and runs
scripts/v2/run_upstream.sh; skips samples whose sample_pack.npz exists. Results: <out>/<sample_id>/...
"""
import argparse
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v2'))
from upstream_cli import build  # noqa: E402

sys.path.insert(0, str(ROOT / 'src'))
from fieldbridge.paths import path_map, relocate_path  # noqa: E402

TEMPLATES = ROOT / 'upstream/DeformFieldBench/simulation/config'
ASSETS = ROOT / 'data/geometry/kaggle_v2/3DGS_model/3DGS_model'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--per-cell', type=int, default=2); p.add_argument('--fps', type=float, default=32)
    p.add_argument('--out', required=True); p.add_argument('--shard', type=int, default=0); p.add_argument('--num-shards', type=int, default=1)
    a = p.parse_args(); out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(l) for l in (ROOT / 'data/manifests/deformfield_v1.jsonl').read_text().splitlines() if l.strip()]
    cells = {}
    for r in sorted((r for r in rows if r['split'] == 'val'), key=lambda r: r['sample_id']):
        cells.setdefault((r['law_id'], r['action']), []).append(r)
    rng = random.Random(0); chosen = []
    for key in sorted(cells):
        chosen += rng.sample(cells[key], min(a.per_cell, len(cells[key])))
    (out / 'selection.json').write_text(json.dumps([r['sample_id'] for r in chosen], indent=1))
    for i, r in enumerate(chosen):
        if i % a.num_shards != a.shard:
            continue
        src = Path(relocate_path(r["pack"], path_map())).parent; dst = out / r['sample_id'].replace('/', '_')
        gt = json.loads((src / 'gt.json').read_text()); rp = json.loads((src / 'meta/run_parameters.json').read_text())
        if list(dst.glob('*/sample_pack.npz')):
            continue
        dst.mkdir(parents=True, exist_ok=True)
        cfg = json.loads((TEMPLATES / f"{rp['sim_type']}_cube_jelly.json").read_text())
        cfg.update({k: v for k, v in rp['material_params'].items() if k != 'g' or rp['sim_type'] == 'drop'}); cfg.update(rp['time_params'])
        (dst / 'config.json').write_text(json.dumps(cfg, indent=1)); (dst / 'gt.json').write_text(json.dumps(gt, indent=1))
        cli = build(rp, ASSETS / f"{gt['object']}.ply", dst / 'config.json', dst,
                    {'render_outputs_per_sim_second': a.fps, 'compile_video': False, 'delete_png_sequences_after_compile_video': False})
        cli += ['--num_views', '3', '--num_render_views', '3']
        started = time.time()
        with (dst / 'run.log').open('w') as log:
            code = subprocess.call([str(ROOT / 'scripts/v2/run_upstream.sh'), *cli], stdout=log, stderr=subprocess.STDOUT)
        (dst / 'replay.json').write_text(json.dumps({'source': str(src), 'exit': code, 'seconds': time.time() - started, 'fps': a.fps}))
        print(r['sample_id'], code, round(time.time() - started), flush=True)


if __name__ == '__main__':
    main()
