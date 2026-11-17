"""(Re)write the C1 jobs of docs/v3b/queue.json: prior/uninformed arms, priority = list order (idempotent)."""
import json, os
q = json.load(open("docs/v3b/queue.json"))
q["max_per_gpu"] = 6
plan = json.load(open("docs/v3b/C1_SAMPLES.json"))["samples"]
base = ".venv/bin/python scripts/v3b/c1_refine.py --run v2main_C_seed29 --index {i}"
def job(jid, cmd, after=None):
    j = {"id": jid, "kind": "cmd", "group": "v3b-C1", "mem_mib": 14000, "retries": 6, "cmd": cmd}
    if after:
        j["after"] = after
    return j
# 48-sim subset: first sample of every (law, action) cell + 2 more, in plan order.
seen, sub = set(), []
for p in plan:
    if (p["law"], p["action"]) not in seen:
        seen.add((p["law"], p["action"])); sub.append(p["index"])
sub += [p["index"] for p in plan if p["index"] not in sub][:20 - len(sub)]
jobs = [j for j in q["jobs"] if not j["id"].startswith("v3b_C1")]
prior = []
for p in plan:
    i = p["index"]
    prior.append(job(f"v3b_C1_{i:06d}", base.format(i=i) + " --budget 24 --replay-budgets 0,4,12,24" + (" --gt-observed" if p["gt_observed"] else "")))
second = []
for p in plan:
    i = p["index"]
    second.append(job(f"v3b_C1p24_{i:06d}", base.format(i=i) + " --budget 24 --replay-budgets 0,4,12,24", [f"v3b_C1_{i:06d}"]))
    second.append(job(f"v3b_C1g_{i:06d}", base.format(i=i) + " --arm lawmean_gt --budget 24 --replay-budgets 12,24"))
    second.append(job(f"v3b_C1b_{i:06d}", base.format(i=i) + " --arm lawmean_best3 --budget 24 --replay-budgets 12,24"))
third = [job(f"v3b_C1u_{p['index']:06d}", base.format(i=p['index']) + " --arm prior_sigma --budget 12 --replay-budgets 12") for p in plan]
fourth = []
for i in sub:
    fourth.append(job(f"v3b_C1g48_{i:06d}", base.format(i=i) + " --arm lawmean_gt --budget 48 --replay-budgets 12,24,48", [f"v3b_C1g_{i:06d}"]))
    fourth.append(job(f"v3b_C1b48_{i:06d}", base.format(i=i) + " --arm lawmean_best3 --budget 48 --replay-budgets 12,24,48", [f"v3b_C1b_{i:06d}"]))
# Per-sample interleaving (prior, prior-24 extension, lawmean_gt, lawmean_best3) so that a balanced common
# sample set accumulates early (interim go/no-go checkpoint), then prior_sigma, then the 48-sim subset.
inter = []
for j, p in enumerate(plan):
    inter += [prior[j]] + second[3 * j:3 * j + 3]
top2 = [job(f"v3b_C1t_{p['index']:06d}", base.format(i=p['index']) + " --arm prior_top2 --budget 24 --replay-budgets 0,4,12,24") for p in plan]
q["jobs"] = jobs + inter + top2 + third + fourth
q["c1_subset48"] = sub
open("docs/v3b/queue.json.tmp", "w").write(json.dumps(q, indent=1)); os.replace("docs/v3b/queue.json.tmp", "docs/v3b/queue.json")
print(len(q["jobs"]), len(sub))
