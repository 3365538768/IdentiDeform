import json, shutil, time
from pathlib import Path
J = json.load(open("docs/v3j/jobs.json")); done = set(p.name for p in Path("runs/v3j_done").iterdir())
log = open("docs/v3j/logs/pool.log").read().splitlines()
started = set(l.split()[2] for l in log if " start " in l)
cand = [j for j in J if j["id"] not in done and j["id"] not in started and not j.get("after")]
move = cand[len(cand) // 3:]  # tail two-thirds -> a800-2 (the pool launches in file order)
print("pending-no-after-not-started", len(cand), "move", len(move))
mv = {j["id"] for j in move}
shutil.copy("docs/v3j/jobs.json", "docs/v3j/jobs_before_split.json")
json.dump([j for j in J if j["id"] not in mv], open("docs/v3j/jobs.json", "w"), indent=1)
Path("docs/v3j_a2").mkdir(exist_ok=True)
q = {"phase": "v3j_a2", "max_per_gpu": 3, "gpus": [0, 1, 2, 3, 4, 5],
     "note": "v3j jobs moved from the a800-4 pool (user: use a800-2 but keep 2 GPUs free -> 6,7 untouched)",
     "jobs": [dict(id=j["id"], kind="cmd", group="v3j", mem_mib=26000 if j.get("slots", 1) > 1 else 7000, retries=2,
                   cmd=j["cmd"] + " && touch runs/v3j_done/" + j["id"]) for j in move]}
json.dump(q, open("docs/v3j_a2/queue.json", "w"), indent=1)
open("docs/v3j/DEVIATIONS.md", "a").write(time.strftime("- %H:%M UTC", time.gmtime()) + f": moved {len(move)} not-yet-started jobs (no dependencies) "
    "to the a800-2 r_queue (docs/v3j_a2/queue.json, GPUs 0-5; 6-7 left free for the user) to speed up; same commands and done markers; "
    "pre-split list in docs/v3j/jobs_before_split.json\n")
print(sorted(mv)[:5])
