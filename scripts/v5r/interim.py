"""V5r interim / deadline measurement: squeeze_run.py measures only after ALL depths of a job are simulated. This script measures every
depth whose simulation has finished (cache/<key>/<d>/traj.npz next to the d=0 reference traj.npz, or already-stored metrics), with the
SAME measure() (scripts/v3e/squeeze_sim.py), and writes read-only copies data/v5r/interim/<idx>/<method>/state.json marked complete
for the requested depth set only if every requested depth is available. Live run directories are never modified.

  FB_GRIP_TWINS=v5 FB_GRIP_DATA=data/v5b_main/val .venv/bin/python scripts/v5r/interim.py --depths 0.06,...,0.32
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v3e'))
import squeeze_sim as S  # noqa: E402


def load(p):
    try:
        return dict(np.load(p))
    except Exception:  # noqa: BLE001  (file still being written)
        return None


def main():
    a = argparse.ArgumentParser(); a.add_argument('--depths', required=True); a.add_argument('--out', default='data/v5r/interim')
    args = a.parse_args(); depths = [float(v) for v in args.depths.split(',')]
    n = {}
    for td in sorted((ROOT / 'data/v5r/runs').glob('*/*/twin.json')):
        m = td.parent.name; idx = td.parent.parent.name; tw = json.loads(td.read_text())
        cache = td.parent.parent / 'cache' / tw['cache']
        try:
            st = json.loads((cache / 'state.json').read_text())
        except (OSError, ValueError):
            continue
        ref = None; runs = {}
        for d in depths:
            k = f'{d:.3f}'; r = st['runs'].get(k)
            if r is None:
                break
            if r.get('failed'):
                runs[k] = r; continue
            if 'metrics' in r:
                runs[k] = r; continue
            if ref is None:
                ref = load(cache / '0.000' / 'traj.npz')
                if ref is None:
                    break
            sq = load(cache / k / 'traj.npz')
            if sq is None or not (cache / k / 'squeeze_bc.json').exists():
                break
            runs[k] = {**r, 'metrics': S.measure(sq, ref, json.loads((cache / k / 'squeeze_bc.json').read_text()))}
        if len(runs) < len(depths):
            continue
        o = ROOT / args.out / idx / m; o.mkdir(parents=True, exist_ok=True)
        (o / 'state.json').write_text(json.dumps({'params': tw['params'], 'object': tw['object'], 'runs': runs, 'complete': True,
                                                  'method': m, 'meta': tw['meta'], 'cache': tw['cache'], 'interim': True}, indent=1))
        n[m] = n.get(m, 0) + 1
    print(json.dumps(n))


if __name__ == '__main__':
    main()
