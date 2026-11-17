"""V5r (deadline): not-yet-started twin jobs in the a800-4 lists go back to the V2 grid (priority 1 = GT + 3 main twins on the V2 grid
for as many samples as possible); prior_vis3 jobs stay last. Started jobs are untouched (pool re-reads the list every cycle)."""
import json, re
from pathlib import Path
R = Path('/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test')
FULL = "0.06,0.10,0.14,0.18,0.24,0.32,0.40,0.48,0.56,0.64"; V2 = "0.06,0.10,0.14,0.18,0.24,0.32"
for f in ("a4", "a4c"):
    started = set(re.findall(r"start (\S+) gpu", (R / f"docs/v5r/logs/pool_{f}.log").read_text()))
    p = R / f"docs/v5r/jobs_{f}.json"; L = json.loads(p.read_text()); n = 0
    for j in L:
        if j["id"] not in started and not j["id"].startswith("v5r_gt_") and "FB_GRIP_DEPTHS=" + FULL in j["cmd"]:
            j["cmd"] = j["cmd"].replace("FB_GRIP_DEPTHS=" + FULL, "FB_GRIP_DEPTHS=" + V2); n += 1
    p.write_text(json.dumps(L, indent=1)); print(f, "started", len(started), "switched to V2 grid", n, "of", len(L))
