"""V5r: a800-5 withdrawn (04:00 UTC). Move every not-finished job of the a800-5 samples (jobs_a5.json + jobs_a5b.json) to a new
a800-4 pool list docs/v5r/jobs_a4c.json (whole samples; their twins already on a800-4 stay in jobs_a4.json). Twin jobs use the full
grid (V2 depths first). Squeeze states are resumable: finished depths are kept, the killed in-flight sim is re-run."""
import json, os
from pathlib import Path
R = Path('/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test')
FULL = "0.06,0.10,0.14,0.18,0.24,0.32,0.40,0.48,0.56,0.64"; V2 = "0.06,0.10,0.14,0.18,0.24,0.32"
jobs = []
for f in ("a5", "a5b"):
    for j in json.loads((R / f"docs/v5r/jobs_{f}.json").read_text()):
        if not (R / "runs/v5r_done" / j["id"]).exists():
            j["cmd"] = j["cmd"].replace("FB_GRIP_DEPTHS=" + V2, "FB_GRIP_DEPTHS=" + FULL); jobs.append(j)
order = {"gt": 0, "prior_vision": 1, "prior_vt": 2, "lawmean_pred": 3, "prior_vis3": 4}
jobs.sort(key=lambda j: (order[j["id"][4:].rsplit("_", 1)[0]], j["id"]))
(R / "docs/v5r/jobs_a4c.json").write_text(json.dumps(jobs, indent=1))
for f in ("a5", "a5b"):
    os.replace(R / f"docs/v5r/jobs_{f}.json", R / f"docs/v5r/jobs_{f}.WITHDRAWN.json")
print(len(jobs), "jobs moved;", sorted({j["id"].rsplit("_", 1)[1] for j in jobs}))
