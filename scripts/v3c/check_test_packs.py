"""V3c: integrity check of data/v2_main/test before building the test manifest (no labels read except gt.json schema).

For every sample dir: result.json exit==0, exactly one */sample_pack.npz, zip CRC test, every array loads,
rgb is uint8 [V>=3, T, 3, H, W] and finite fields. Corrupt packs (bad zip / unreadable) are MOVED to
.cache/bad_samples/test_<dir>/ (only with --quarantine). Writes a JSON report.
  .venv/bin/python scripts/v3c/check_test_packs.py --output docs/v3c/test_pack_check.json --quarantine
"""
import argparse, json, shutil, time, zipfile
from pathlib import Path
import numpy as np

ROOT = Path('data/v2_main/test')


def check(d):
    out = {'dir': d.name}
    try:
        out['exit'] = json.loads((d / 'result.json').read_text()).get('exit')
    except Exception as e:
        out['exit'] = None; out['result_error'] = type(e).__name__
    packs = sorted(d.glob('*/sample_pack.npz')); out['packs'] = len(packs)
    if len(packs) != 1:
        out['ok'] = False; out['reason'] = f'{len(packs)} packs'; return out
    p = packs[0]; out['pack'] = str(p); out['bytes'] = p.stat().st_size
    try:
        with zipfile.ZipFile(p) as z:
            bad = z.testzip()
        if bad:
            raise ValueError('crc ' + bad)
        with np.load(p, allow_pickle=False) as z:
            shapes = {k: list(z[k].shape) for k in z.files}
            rgb = z['rgb']
            if rgb.dtype != np.uint8 or rgb.ndim != 5 or rgb.shape[0] < 3 or rgb.shape[2] != 3:
                raise ValueError('rgb shape/dtype ' + str(rgb.shape) + str(rgb.dtype))
            for k in z.files:
                a = z[k]
                if a.dtype.kind == 'f' and not np.isfinite(a).all():
                    raise ValueError('non-finite ' + k)
        out['shapes'] = shapes; out['corrupt'] = False
        out['ok'] = out['exit'] == 0 and (d / 'gt.json').exists() and (p.parent / 'meta/run_parameters.json').exists()
        if not out['ok']:
            out['reason'] = 'exit/gt/meta'
    except Exception as e:
        out['ok'] = False; out['corrupt'] = True; out['reason'] = f'{type(e).__name__}: {e}'[:200]
    return out


def main():
    a = argparse.ArgumentParser(); a.add_argument('--output', required=True); a.add_argument('--quarantine', action='store_true'); a = a.parse_args()
    t = time.time(); rows = [check(d) for d in sorted(ROOT.glob('[0-9]*')) if d.is_dir()]
    moved = []
    if a.quarantine:
        for r in rows:
            if r.get('corrupt'):
                dst = Path('.cache/bad_samples') / f"test_{r['dir']}"; dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(ROOT / r['dir']), str(dst)); moved.append({'from': str(ROOT / r['dir']), 'to': str(dst), 'reason': r['reason']})
    rep = {'dirs': len(rows), 'exit0': sum(r['exit'] == 0 for r in rows), 'ok': sum(r['ok'] for r in rows),
           'corrupt': sum(bool(r.get('corrupt')) for r in rows), 'not_ok': [r for r in rows if not r['ok']],
           'quarantined': moved, 'rgb_shapes': sorted({str(r['shapes']['rgb']) for r in rows if r.get('shapes')}),
           'seconds': time.time() - t, 'samples': rows}
    Path(a.output).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != 'samples'}, indent=1))


if __name__ == '__main__':
    main()
