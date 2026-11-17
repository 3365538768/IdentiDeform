"""Summarize P4b sweeps: adjacent half-decade deformation difference (px) per action/damping/sampling."""
import glob
import json

import numpy as np

for path in sorted(glob.glob('docs/p4/P4B_SWEEP_*.json')):
    r = json.load(open(path)); tag = f"{r['material']}/{r['sweep']}/{r['damping']}{'' if r.get('status') == 'complete' else ' (partial)'}"
    for action, rows in r['actions'].items():
        d = [x for x in rows if 'diff_px_all' in x]
        if not d:
            continue
        fail = sum(not x['ok'] for x in rows)
        med = lambda k: float(np.median([x[k] for x in d]))
        low = lambda k: float(np.min([x[k] for x in d]))
        hz = [round(x.get('dominant_height_hz') or 0, 1) for x in rows if x['ok']]
        print(f"{tag:34s} {action:5s} adj diff px median all/32/16/8/4fps "
              f"{med('diff_px_all'):5.2f} {med('diff_px_32fps'):5.2f} {med('diff_px_16fps'):5.2f} {med('diff_px_8fps'):5.2f} {med('diff_px_4fps'):5.2f}"
              f" | min {low('diff_px_all'):5.2f} | fail {fail} | hz {hz}")
