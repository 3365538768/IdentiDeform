"""Score val samples for project-page clip selection: fragmentation, height/area change, final residual.
Usage: score_touch.py <root> <out.json>"""
import sys, json, glob, os, numpy as np, cv2
from concurrent.futures import ProcessPoolExecutor
root, out = sys.argv[1], sys.argv[2]
def one(d):
    g = json.load(open(f"{d}/gt.json")); p = glob.glob(f"{d}/*/sample_pack.npz")
    if not p: return None
    m = np.load(p[0])["object_mask"][:, :, 0] > 127
    frag = 0.0
    for v in range(3):
        for t in range(0, m.shape[1], 8):
            n, _, st, _ = cv2.connectedComponentsWithStats(m[v, t].astype(np.uint8), connectivity=8)
            if n > 2:
                a = st[1:, cv2.CC_STAT_AREA]; frag = max(frag, 1 - a.max() / a.sum())
    area = m.sum((2, 3)).astype(float)  # V,T
    def h(v, t):
        ys = np.nonzero(m[v, t].any(1))[0]; return np.ptp(ys) + 1 if len(ys) else 0
    H = np.array([[h(v, t) for t in range(m.shape[1])] for v in range(3)], float)
    hr = H / np.maximum(H[:, :1], 1)
    def iou(a, b): return (a & b).sum() / max((a | b).sum(), 1)
    iou_min = np.median([min(iou(m[v, 0], m[v, t]) for t in range(0, m.shape[1], 4)) for v in range(3)])
    iou_end = np.median([iou(m[v, 0], m[v, -1]) for v in range(3)])
    lab = g["labels"]
    return dict(path=d, idx=g["index"], action=g["action"], law=g["law"], obj=g["object"],
                le=round(float(np.log10(lab["E_over_rho_specific"])), 2), lr=round(float(np.log10(g["regression"]["density"])), 2),
                frag=round(float(frag), 3), hmin=round(float(np.median(hr.min(1))), 3), hend=round(float(np.median(hr[:, -1])), 3),
                amax=round(float(np.median(area.max(1) / np.maximum(area[:, 0], 1))), 3), a0=int(np.median(area[:, 0])),
                iou_min=round(float(iou_min), 3), iou_end=round(float(iou_end), 3), press_force=g.get("press_force"))
ds = sorted(glob.glob(f"{root}/val/*/"))
ds = [d.rstrip("/") for d in ds]
with ProcessPoolExecutor(32) as ex: rows = [r for r in ex.map(one, ds) if r]
json.dump(rows, open(out, "w"), indent=0); print(len(rows))
