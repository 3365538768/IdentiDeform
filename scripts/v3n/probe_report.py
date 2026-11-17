"""Identifiability probe: video change when E/rho is x4 (mask 1-IoU and RGB MAE over the union), per action,
for the V3 setup (data/v3_main vs data/v3n_probe/A4) and V3 + g=2 everywhere (G1 vs G4). Also V2 for reference
is not possible here (different objects/params); compare with Fig. 3c numbers instead."""
import glob, json, collections, numpy as np
def pack(root, i):
    p = glob.glob(f'{root}/val/{i:06d}/*/sample_pack.npz')
    return np.load(p[0]) if p else None
def change(a, b):
    ma, mb = a['object_mask'][:, :, 0] > 127, b['object_mask'][:, :, 0] > 127
    T = min(ma.shape[1], mb.shape[1]); ma, mb = ma[:, :T], mb[:, :T]
    inter = (ma & mb).sum((2, 3)); uni = (ma | mb).sum((2, 3))
    iou = np.where(uni > 0, inter / np.maximum(uni, 1), 1.0).mean()
    ra, rb = a['rgb'][:, :T].astype(np.float32), b['rgb'][:, :T].astype(np.float32)
    u = (ma | mb)[:, :, None]
    mae = (np.abs(ra - rb) * u).sum() / max(u.sum() * 3, 1)
    return 1 - iou, mae
IDX = [0, 1, 2, 18, 3, 4, 5, 21, 6, 7, 8, 24, 9, 10, 11, 27, 12, 13, 14, 30, 15, 16, 17, 33]
res = collections.defaultdict(lambda: collections.defaultdict(list)); rows = []
for i in IDX:
    g = json.load(open(f'data/v3_main/val/{i:06d}/gt.json')); a = g['action']
    for tag, r1, r4 in (('V3', 'data/v3_main', 'data/v3n_probe/A4'), ('V3+g2', 'data/v3n_probe/G1', 'data/v3n_probe/G4')):
        p1, p4 = pack(r1, i), pack(r4, i)
        if p1 is None or p4 is None: continue
        d, m = change(p1, p4); res[a][tag].append((d, m)); rows.append((i, a, g['law'], tag, round(float(d), 3), round(float(m), 1)))
out = {a: {t: {'n': len(v), 'mask_change': float(np.mean([x[0] for x in v])), 'rgb_mae': float(np.mean([x[1] for x in v]))} for t, v in d.items()} for a, d in res.items()}
json.dump({'per_action': out, 'rows': rows}, open('docs/v3n/IDENT_PROBE.json', 'w'), indent=1)
for a, d in out.items(): print(a, {t: (v['n'], round(v['mask_change'], 3), round(v['rgb_mae'], 1)) for t, v in d.items()})
