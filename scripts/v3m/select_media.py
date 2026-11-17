"""Website: score V2-main val/train samples for intact (non-tearing) deformation per action."""
import sys, glob, json, numpy as np, cv2
sys.path.insert(0, 'scripts/v3m')
split = sys.argv[1]; out = sys.argv[2]
rows = []
for d in sorted(glob.glob(f'data/v2_main/{split}/*/gt.json')):
    i = int(d.split('/')[-2])
    try:
        g = json.load(open(d)); p = glob.glob(d[:-7] + '*/sample_pack.npz')
        if not p: continue
        m = np.load(p[0])['object_mask'][:, :, 0] > 127  # V,T,H,W
    except Exception as e:
        continue
    V, T = m.shape[:2]
    frag = 0.0; area = m.sum((2, 3)).astype(float)
    for v in range(V):
        for t in range(0, T, 4):
            n, lab, st, _ = cv2.connectedComponentsWithStats(m[v, t].astype(np.uint8), connectivity=8)
            if n <= 2: continue
            a = st[1:, cv2.CC_STAT_AREA]; frag = max(frag, 1 - a.max() / a.sum())
    a0 = area[:, :2].mean(1); ratio = (area / np.maximum(a0[:, None], 1))
    inter = (m & m[:, :1]).sum((2, 3)); uni = (m | m[:, :1]).sum((2, 3))
    motion = float((1 - inter / np.maximum(uni, 1)).mean())
    r = g['regression']; s = r['density'] / 1000
    rows.append(dict(index=i, action=g['action'], law=g['law'], object=g['object'], frag=round(float(frag), 4),
                     area_max=round(float(ratio.max()), 2), area_min=round(float(ratio.min()), 2), area_end=round(float(ratio[:, -1].mean()), 2),
                     motion=round(motion, 3), logEr=round(float(np.log10(r['E'] / s)), 2), a0=float(a0.mean())))
json.dump(rows, open(out, 'w'), indent=0)
print(len(rows))
