"""Create/update only the v3x entry of docs/v3_pipeline/PLAN.json. done = finished v3x jobs (runs/v3x_done)."""
import argparse, calendar, fcntl, json, os, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; F = ROOT / "docs/v3_pipeline/PLAN.json"
a = argparse.ArgumentParser(); a.add_argument("--status"); a.add_argument("--detail"); a.add_argument("--eta-lo"); a.add_argument("--eta-hi"); a.add_argument("--host"); args = a.parse_args()
jobs = json.loads((ROOT / "docs/v3x/jobs_a2.json").read_text()) + json.loads((ROOT / "docs/v3x/jobs_a3.json").read_text())
done = sum((ROOT / "runs/v3x_done" / j["id"]).exists() for j in jobs)
ep = lambda s: calendar.timegm(time.strptime(s, "%Y-%m-%d %H:%M"))
with open(str(F) + ".v3x.lock", "w") as lk:
    fcntl.flock(lk, fcntl.LOCK_EX)
    P = json.loads(F.read_text())
    e = next((s for s in P["stages"] if s["id"] == "v3x"), None)
    if e is None:
        e = {"id": "v3x", "label": "v3x test 精修扩样：新增 30 个 test 样本（覆盖全部 10 个物体），合并为 60 样本（prior-top1 / law-mean-best3 / law-mean-GT / prior-top2）",
             "host": "a800-2", "unit": "作业", "deliverable": "docs/v3x/RESULTS.json"}
        P["stages"].insert(0, e)
    e.update(done=done, total=len(jobs), updated_utc=time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()))
    for k, v in (("status", args.status), ("detail", args.detail), ("host", args.host)):
        if v is not None: e[k] = v
    if args.eta_lo: e["eta_lo"] = ep(args.eta_lo)
    if args.eta_hi: e["eta_hi"] = ep(args.eta_hi)
    tmp = F.with_suffix(".v3x.tmp"); tmp.write_text(json.dumps(P, indent=1, ensure_ascii=False)); os.replace(tmp, F)
print(e)
