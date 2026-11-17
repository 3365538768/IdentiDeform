import sys, json, collections, numpy as np, cv2
sys.path.insert(0, "scripts/v3m"); from common import load, cropped_view
R = [dict(r, split=s) for s in ("train", "test", "val") for r in json.load(open(f"data/v3m_site/select_{s}.json"))]
g = collections.defaultdict(list)
for r in R:
    if r["frag"] < 0.001 and r["a0"] > 900 and r["area_max"] < 1.6 and r["area_min"] > 0.55 and r["law"] == "jelly":
        g[(r["object"], r["action"])].append(r)
P = []
for k, v in g.items():
    v.sort(key=lambda r: r["logEr"])
    if len(v) >= 2 and v[-1]["logEr"] - v[0]["logEr"] > 1.0 and k[1] not in ("stretch",):
        P.append((v[0], v[-1]))
print(len(P))
rows = []
for a, b in P[:14]:
    tiles = []
    for r in (a, b):
        z, _ = load(r["index"], r["split"]); fr, _ = cropped_view(z["rgb"][0], z["object_mask"][0], out=80)
        tiles += [fr[t] for t in np.linspace(0, 127, 6).astype(int)]
    row = np.concatenate(tiles, 1).copy()
    cv2.putText(row, f"{a['action']} {a['split']}{a['index']}({a['logEr']}) vs {b['split']}{b['index']}({b['logEr']})", (2, 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 80, 80), 1)
    rows.append(row)
cv2.imwrite("data/v3m_site/look/c_pairs.jpg", cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
