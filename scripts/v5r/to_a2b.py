"""V5r (deadline): move not-started a800-4 twin jobs to the a800-2 r_queue when their parameter set (cache key) is unique within the
sample, so no cache directory can be written from two hosts (the only cross-host hazard: flock is not coherent across hosts)."""
import hashlib, json, re, sys
from pathlib import Path
R = Path('/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test')
sys.path.insert(0, str(R / 'scripts/v5r')); import twins  # noqa: E402
key = lambda p: hashlib.sha1(json.dumps(p, sort_keys=True).encode()).hexdigest()[:12]
started = set()
for f in ("a4", "a4c"):
    started |= set(re.findall(r"start (\S+) gpu", (R / f"docs/v5r/logs/pool_{f}.log").read_text()))
p = R / "docs/v5r/jobs_a4.json"; L = json.loads(p.read_text())
q = json.loads((R / "docs/v5r/queue.json").read_text()); q["max_per_gpu"] = 16
move = []
for j in L:
    if j["id"] in started:
        continue
    m, i = j["id"][4:].rsplit("_", 1); i = int(i)
    k = key(twins.twin_params(i, m)[0])
    others = [key(twins.twin_params(i, mm)[0]) for mm in ("gt", "prior_vision", "prior_vt", "lawmean_pred", "prior_vis3") if mm != m]
    if k not in others:
        move.append(j)
for j in move:
    q["jobs"].append({"id": j["id"], "kind": "cmd", "group": "v5r-twin", "mem_mib": 5000, "retries": 2, "cmd": j["cmd"]})
p.write_text(json.dumps([j for j in L if j not in move], indent=1)); (R / "docs/v5r/queue.json").write_text(json.dumps(q, indent=1))
print("moved", len(move), "kept on a800-4", len(L) - len(move))
