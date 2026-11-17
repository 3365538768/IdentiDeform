"""V5r rebalance (deadline 05:55): move not-yet-started twin jobs (whole samples only) from the a800-3/5 second pools to an a800-4
pool, switch unstarted twin jobs to the full depth grid (V2 depths are simulated first), raise the a800-2 queue slot cap."""
import json, re
from pathlib import Path
R = Path('/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test')
FULL = "0.06,0.10,0.14,0.18,0.24,0.32,0.40,0.48,0.56,0.64"; V2 = "0.06,0.10,0.14,0.18,0.24,0.32"
started = set()
for f in ("a3", "a3b", "a5", "a5b"):
    started |= set(re.findall(r"start (\S+) gpu", (R / f"docs/v5r/logs/pool_{f}.log").read_text()))
st = json.loads((R / "docs/rq_state.json").read_text())["jobs"]
move = []
for f in ("a3b", "a5b"):
    p = R / f"docs/v5r/jobs_{f}.json"; L = json.loads(p.read_text()); by = {}
    for j in L:
        by.setdefault(j["id"].rsplit("_", 1)[1], []).append(j)
    keep = []
    for i, js in by.items():
        tw = [j for j in js if not j["id"].startswith("v5r_gt_")]
        if tw and not any(j["id"] in started for j in tw):
            move += tw; keep += [j for j in js if j not in tw]
        else:
            keep += js
    p.write_text(json.dumps(keep, indent=1))
order = {"prior_vision": 0, "prior_vt": 1, "lawmean_pred": 2, "prior_vis3": 3}
for j in move:
    j["cmd"] = j["cmd"].replace("FB_GRIP_DEPTHS=" + V2, "FB_GRIP_DEPTHS=" + FULL)
move.sort(key=lambda j: (order[j["id"][4:].rsplit("_", 1)[0]], j["id"]))
(R / "docs/v5r/jobs_a4.json").write_text(json.dumps(move, indent=1))
q = json.loads((R / "docs/v5r/queue.json").read_text()); q["max_per_gpu"] = 16; n = 0
for j in q["jobs"]:
    if not j["id"].startswith("v5r_gt_") and st.get(j["id"], {}).get("status") == "pending":
        j["cmd"] = j["cmd"].replace("FB_GRIP_DEPTHS=" + V2, "FB_GRIP_DEPTHS=" + FULL); n += 1
(R / "docs/v5r/queue.json").write_text(json.dumps(q, indent=1))
print("moved to a4", len(move), "| a2 pending twins -> full grid", n)
