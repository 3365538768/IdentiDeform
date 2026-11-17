"""Gravity/grip audit of V2-main: per action, fragment rate and vertical motion of the object's mask centroid."""
import sys, json, glob, collections, numpy as np
sys.path.insert(0, 'scripts/v3m'); from common import load
R = json.load(open('data/v3m_site/select_val.json'))
agg = collections.defaultdict(list)
for r in R:
    z, g = load(r['index'], 'val'); m = z['object_mask'][:, :, 0] > 127  # V,T,H,W
    H = m.shape[2]; ys = np.arange(H)[None, None, :, None]
    cnt = m.sum((2, 3)); cy = (m * ys).sum((2, 3)) / np.maximum(cnt, 1)  # V,T  (image y grows downward)
    h0 = np.array([np.ptp(np.nonzero(m[v, 0].any(1))[0]) if m[v, 0].any() else 1 for v in range(m.shape[0])])
    d = lambda t0, t1: float(np.median((cy[:, t1] - cy[:, t0]) / np.maximum(h0, 1)))  # downward shift in object heights
    agg[r['action']].append(dict(frag=r['frag'] > 0.01, d_0_3=d(0, 3 * 32 - 1), d_3_4=d(3 * 32 - 1, 127), d_0_05=d(0, 16)))
for a, L in agg.items():
    f = lambda k: np.median([x[k] for x in L])
    print(f"{a:10s} n={len(L):3d} frag>1%={np.mean([x['frag'] for x in L]):.2f}  centroid drop (object heights): 0-0.5s {f('d_0_05'):+.2f}  0-3s {f('d_0_3'):+.2f}  3-4s {f('d_3_4'):+.2f}")
