"""V3e cleanup: shrink kept subset trajectories (every 4th stored frame, float16) and delete leftover full particle dumps."""
import glob, os, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
before = after = 0; n = 0
for f in glob.glob(str(ROOT / 'data/v3e/runs/*/cache/*/*/subset.npz')):
    t = dict(np.load(f))
    if t['subset'].dtype == np.float16:
        continue
    before += os.path.getsize(f)
    t['subset'] = t['subset'][::4].astype(np.float16); t['subset_t'] = t['subset_t'][::4]
    tmp = f[:-4] + '.tmp.npz'; np.savez_compressed(tmp, **t); os.replace(tmp, f)
    after += os.path.getsize(f); n += 1
dumps = glob.glob(str(ROOT / 'data/v3e/runs/*/cache/*/*/traj.npz')) + glob.glob(str(ROOT / 'data/v3e/pilot/*/cache/*/*/traj.npz'))
sz = sum(os.path.getsize(p) for p in dumps)
for p in dumps:
    os.remove(p)
msg = (f'- {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())} v3e: compacted {n} data/v3e/runs/**/subset.npz '
       f'({before / 2**30:.2f} GiB -> {after / 2**30:.2f} GiB: kept every 4th subset frame as float16; x0/x_tail/width trace unchanged); '
       f'deleted {len(dumps)} full particle dumps traj.npz ({sz / 2**20:.0f} MiB) under data/v3e/runs and data/v3e/pilot '
       f'(metrics already in state.json; pilot figures kept in docs/v3e/figs)\n')
with open(ROOT / 'docs/CLEANUP_2026-11-12.md', 'a') as fh:
    fh.write(msg)
print(msg)
