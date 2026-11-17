"""Website: extra metal / plasticine clips (3 views each), screened with select_media.py (no fragments)."""
import sys, os, json, numpy as np
sys.path.insert(0, 'scripts/v3m')
from make_videos_lib import encode, views
OUT = sys.argv[1]; os.makedirs(OUT + '/more', exist_ok=True)
MORE = json.loads(sys.argv[2])  # {name: [split, index, (frames)]}
meta = {}
for name, (sp, i, *lim) in MORE.items():
    vv, g = views(i, split=sp)
    if lim: vv = [v[:lim[0]] for v in vv]
    sep = np.zeros((vv[0].shape[0], 256, 4, 3), np.uint8)
    encode(np.concatenate([vv[0], sep, vv[1], sep, vv[2]], 2), f'{OUT}/more/{name}.mp4')
    meta[name] = {'split': sp, 'index': i, 'law': g['law'], 'action': g['action'], 'object': g['object']}
json.dump(meta, open(f'{OUT}/more_meta.json', 'w'), indent=1)
