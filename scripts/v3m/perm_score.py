"""Website: permanent-deformation score for metal/plasticine clips (object stays in place: press/shear/bend)."""
import sys, glob, json, numpy as np
sys.path.insert(0, 'scripts/v3m'); from common import load
R = [dict(r, split=s) for s in ("train", "test", "val") for r in json.load(open(f"data/v3m_site/select_{s}.json"))]
C = [r for r in R if r["law"] in ("metal", "plasticine") and r["action"] in ("press", "shear", "bend", "tap30hold")
     and r["frag"] < 0.002 and r["a0"] > 700 and 0.6 < r["area_end"] < 1.4 and r["area_max"] < 1.6]
out = []
for r in C:
    z, g = load(r["index"], r["split"]); m = z["object_mask"][:, :, 0] > 127
    ch = 1 - (m & m[:, :1]).sum((2, 3)) / np.maximum((m | m[:, :1]).sum((2, 3)), 1)  # V,T
    fin = float(ch[:, -8:].mean()); mx = float(ch.max(1).mean())
    late = float(np.abs(np.diff(m[:, -16:].astype(np.int8), axis=1)).sum((2, 3)).mean() / max(m[:, -1].sum((1, 2)).mean(), 1))
    out.append(dict(r, final=round(fin, 3), peak=round(mx, 3), keep=round(fin / max(mx, 1e-6), 2), still=round(late, 4)))
json.dump(out, open("data/v3m_site/perm.json", "w"))
for law in ("metal", "plasticine"):
    for a in ("press", "shear", "bend", "tap30hold"):
        c = [o for o in out if o["law"] == law and o["action"] == a and o["keep"] > 0.7 and o["final"] > 0.2]
        c.sort(key=lambda o: -o["final"])
        print(law, a, len(c), [(o["split"], o["index"], o["object"][2:18], o["final"], o["keep"]) for o in c[:7]])
