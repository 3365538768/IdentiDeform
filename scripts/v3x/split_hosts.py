"""V3x host split (run on a800-2 at docs/v3j/DONE): samples new_order[15:] go entirely to a800-3.
Rewrites docs/v3x/jobs_a2.json (first 15 samples: top1, best3, then lawmean_gt, then prior_top2) and
docs/v3x/jobs_a3.json (last 15 samples, same order), all with --out data/v3x_refine_test (each sample on one host).
Prints the process-group ids (pool bash pids) of RUNNING a2 jobs of moved samples; kill them only AFTER this rewrite."""
import json, re
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
order = json.loads((ROOT / "docs/v3x/SAMPLES.json").read_text())["new_order"]
base = ".venv/bin/python scripts/v3d/c1_refine.py --run v2main_C_seed29 --split test --ensemble docs/v3d/ENSEMBLE_BCD9_TEST.json"
job = lambda i, arm: {"id": f"v3x_{arm}_{i:06d}", "cmd": f"{base} --out data/v3x_refine_test --index {i} --arm {arm} --budget 24 --replay-budgets 24"}
def jobs(samples):
    return ([job(i, a) for i in samples for a in ("prior_top1", "lawmean_best3")] + [job(i, "lawmean_gt") for i in samples]
            + [job(i, "prior_top2") for i in samples])
A2, A3 = order[:15], order[15:]
old = (ROOT / "docs/v3x/jobs_a2.json").read_text(); (ROOT / "docs/v3x/jobs_a2.before_split.json").write_text(old)
(ROOT / "docs/v3x/jobs_a3.before_split.json").write_text((ROOT / "docs/v3x/jobs_a3.json").read_text())
(ROOT / "docs/v3x/jobs_a2.json").write_text(json.dumps(jobs(A2), indent=1)); (ROOT / "docs/v3x/jobs_a3.json").write_text(json.dumps(jobs(A3), indent=1))
moved = {j["id"] for j in jobs(A3)}
log = (ROOT / "docs/v3x/logs/pool_a2.log").read_text().splitlines()
run = {}
for l in log:
    m = re.match(r"\S+ start (\S+) gpu \d+ pid (\d+)", l)
    if m: run[m.group(1)] = int(m.group(2))
    m = re.match(r"\S+ exit (\S+) ", l)
    if m: run.pop(m.group(1), None)
print(" ".join(str(p) for j, p in run.items() if j in moved))
