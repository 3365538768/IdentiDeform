"""V5q: V5b train law means from the generated train gt.json (data/v5b_main/train 0-1999, exit 0) -> docs/v5q/LAW_MEANS.json.
Units as V2 law_means: per law mean ln1p(E/rho specific), ln1p(sigma_y/rho specific) (non-censored only; V2/V3 manifests drop
censored sigma), mean nu. Also provisional z stats (replaced by the v5p hand-off stats in the report if shipped).
  .venv/bin/python scripts/v5q/law_means.py
"""
import json, math
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
LAWS = ["jelly", "metal", "plasticine"]


def main():
    rows = []
    for i in range(2000):
        d = ROOT / f"data/v5b_main/train/{i:06d}"
        try:
            if json.loads((d / "result.json").read_text()).get("exit") != 0:
                continue
        except FileNotFoundError:
            continue
        g = json.loads((d / "gt.json").read_text()); lab = g["labels"]
        assert g["ranges"] == "v5" and g["gravity"] == -2.0
        s = None if lab.get("censored_sigma") else lab.get("sigma_over_rho_specific")
        rows.append((g["law"], lab["E_over_rho_specific"], g["regression"]["nu"], g["regression"]["density"], s))
    out = {}
    for law in LAWS:
        r = [x for x in rows if x[0] == law]; s = [math.log1p(x[4]) for x in r if x[4] is not None]
        out[law] = {"n": len(r), "logE": float(np.mean([math.log1p(x[1]) for x in r])), "nu": float(np.mean([x[2] for x in r])),
                    "logS": float(np.mean(s)) if s else None, "n_sigma": len(s),
                    "log10E_geo": float(np.mean([math.log10(x[1]) for x in r])),
                    "log10S_geo": float(np.mean([math.log10(x[4]) for x in r if x[4] is not None])) if s else None}
    allS = [math.log1p(x[4]) for x in rows if x[4] is not None]
    cols = [[math.log1p(x[1]) for x in rows], [x[2] for x in rows], [math.log1p(x[3]) for x in rows], allS]
    stats = {"mean": [float(np.mean(c)) for c in cols], "std": [float(np.std(c)) for c in cols], "count": [len(c) for c in cols],
             "transform": "ln1p_E_rho_sigma_y;linear_nu", "target_space": "specific", "fit_split": "train 0-1999 (generated gt.json)"}
    res = {"source": "data/v5b_main/train/000000-001999 gt.json (exit 0)", "n_train": len(rows), "means": out,
           "all_law_logS": float(np.mean(allS)), "stats_provisional": stats}
    f = ROOT / "docs/v5q/LAW_MEANS.json"
    if f.exists():
        print("exists"); return
    f.write_text(json.dumps(res, indent=1) + "\n"); print(json.dumps(res["means"], indent=1), len(rows))


if __name__ == "__main__":
    main()
