import glob, json, numpy as np, cv2
def frag(m):
    f = 0.0
    for v in range(m.shape[0]):
        for t in range(0, m.shape[1], 4):
            n, _, st, _ = cv2.connectedComponentsWithStats(m[v, t].astype(np.uint8), connectivity=8)
            if n > 2:
                a = st[1:, cv2.CC_STAT_AREA]; f = max(f, 1 - a.max() / a.sum())
    return f
for root in ('data/v3n_probe/H1', 'data/v3n_probe/H4'):
    for d in sorted(glob.glob(f'{root}/val/*')):
        p = glob.glob(d + '/*/sample_pack.npz')
        if not p: continue
        g = json.load(open(d + '/gt.json')); m = np.load(p[0])['object_mask'][:, :, 0] > 127
        H = m.shape[2]; ys = np.arange(H)[None, None, :, None]
        cy = (m * ys).sum((2, 3)) / np.maximum(m.sum((2, 3)), 1); h0 = max(np.ptp(np.nonzero(m[0, 0].any(1))[0]), 1)
        drop = float(np.median((cy[:, -1] - cy[:, 0])) / h0)
        print(root[-2:], d[-3:], g['action'][:7], g['law'][:5], 'frag %.3f' % frag(m), 'centroid drop %.2f heights' % drop, 'area_end/start %.2f' % (m[:, -1].sum() / max(m[:, 0].sum(), 1)))
