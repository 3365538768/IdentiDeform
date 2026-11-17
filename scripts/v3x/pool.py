"""V3x worker pool (nohup/setsid): adapted from scripts/v3j/pool.py. A job is done when runs/v3x_done/<id> exists
(resumable: c1_refine.py resumes from state.json). Re-reads the jobs file AND the control file every cycle
(control json: {"slots": N, "per_gpu": {"0": 9, ...}}). Stops launching when the stop file exists.
  setsid nohup .venv/bin/python scripts/v3x/pool.py docs/v3x/jobs_a2.json --ctl docs/v3x/pool_ctl_a2.json --stop-file docs/v3x/POOL_STOP_a2 --log docs/v3x/logs/pool_a2.log &
"""
import argparse, json, subprocess, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]

def main():
    a = argparse.ArgumentParser(); a.add_argument("jobs"); a.add_argument("--ctl", required=True)
    a.add_argument("--retries", type=int, default=2); a.add_argument("--stop-file", required=True); args = a.parse_args()
    done_dir = ROOT / "runs/v3x_done"; done_dir.mkdir(parents=True, exist_ok=True)
    logs = ROOT / "docs/v3x/logs"; logs.mkdir(parents=True, exist_ok=True)
    running, attempts = {}, {}
    while True:
        try:
            jobs = json.loads((ROOT / args.jobs).read_text()); ctl = json.loads((ROOT / args.ctl).read_text())
        except Exception as e:
            print("read error", e, flush=True); time.sleep(15); continue
        per = {int(k): int(v) for k, v in ctl["per_gpu"].items()}; slots = int(ctl["slots"])
        for jid, (p, g) in list(running.items()):
            if p.poll() is not None:
                print(time.strftime("%H:%M:%S"), "exit", jid, p.returncode, "gpu", g, flush=True); running.pop(jid)
        stop = (ROOT / args.stop_file).exists()
        pending = [] if stop else [j for j in jobs if j["id"] not in running and not (done_dir / j["id"]).exists()
                                   and attempts.get(j["id"], 0) <= args.retries]
        if not pending and not running:
            print("all done" if not stop else "stopped", flush=True); break
        for j in pending:
            if len(running) >= slots:
                break
            load = {g: sum(1 for v in running.values() if v[1] == g) for g in per}
            free = [g for g in per if load[g] < per[g]]
            if not free:
                break
            g = min(free, key=lambda x: load[x] / max(per[x], 1))
            attempts[j["id"]] = attempts.get(j["id"], 0) + 1
            cmd = f"cd {ROOT} && source scripts/r1_env.sh && export CUDA_VISIBLE_DEVICES={g} && ( {j['cmd']} ) && touch runs/v3x_done/{j['id']}"
            p = subprocess.Popen(["bash", "-c", cmd], stdout=(logs / f"{j['id']}.log").open("a"), stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, start_new_session=True)
            running[j["id"]] = (p, g); print(time.strftime("%H:%M:%S"), "start", j["id"], "gpu", g, "pid", p.pid, flush=True)
            time.sleep(2)
        time.sleep(15)

if __name__ == "__main__":
    main()
