"""V5r (deadline): samples in docs/v5r/jobs_a4.json with NO started job are moved whole to the a800-2 r_queue (phase v5r, 16/GPU)."""
import json, re
from pathlib import Path
R = Path('/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test')
started = set(re.findall(r"start (\S+) gpu", (R / "docs/v5r/logs/pool_a4.log").read_text()))
p = R / "docs/v5r/jobs_a4.json"; L = json.loads(p.read_text()); by = {}
for j in L:
    by.setdefault(j["id"].rsplit("_", 1)[1], []).append(j)
move = [j for i, js in by.items() if not any(j["id"] in started for j in js) for j in js]
keep = [j for j in L if j not in move]
q = json.loads((R / "docs/v5r/queue.json").read_text()); q["max_per_gpu"] = 16
order = {"prior_vision": 0, "prior_vt": 1, "lawmean_pred": 2, "prior_vis3": 3}
for j in sorted(move, key=lambda j: (order[j["id"][4:].rsplit("_", 1)[0]], j["id"])):
    q["jobs"].append({"id": j["id"], "kind": "cmd", "group": "v5r-twin", "mem_mib": 5000, "retries": 2, "cmd": j["cmd"]})
p.write_text(json.dumps(keep, indent=1)); (R / "docs/v5r/queue.json").write_text(json.dumps(q, indent=1))
print("moved", len(move), "jobs of samples", sorted({j['id'].rsplit('_', 1)[1] for j in move}))
