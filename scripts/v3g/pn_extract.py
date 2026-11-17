"""Parse v3g PAC-NeRF run logs (all attempts) + final dynamic checkpoint into JSON.
Run with the pacnerf venv python: pn_extract.py <name> [<name> ...]"""
import glob, json, math, os, re, sys

R = "/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test"
GT = {"torus": dict(E=1e6, nu=0.3), "bird": dict(E=3e5, nu=0.3),
      "playdoh": dict(E=2e6, nu=0.3, tau_y=1.54e4), "cat": dict(E=1e6, nu=0.3, tau_y=3.85e3)}
NUM = r"([0-9.eE+-]+|nan|inf)"
INFO = re.compile(r"velocity: \[([^\]]*)\]; rho: " + NUM + r"(?:; E: " + NUM + r")?(?:; nu: " + NUM + r")?(?:; yield_stress: " + NUM + r")?")


def fl(x):
    return float(x) if x is not None else None


def summarize(name):
    scene = name.split("_")[0]
    logs = sorted(glob.glob(os.path.join(R, "docs/v3g/logs", f"pn_{name}.a*.log")),
                  key=lambda x: int(x.rsplit(".a", 1)[1][:-4]))
    main = os.path.join(R, "docs/v3g/logs", f"pn_{name}.log")
    if os.path.exists(main):
        logs.append(main)
    attempts, gpu = [], None
    stage = None
    traj = {"static": [], "velocity": [], "dynamic": []}
    dyn_loss, last_fwd, pending, retries = [], None, None, 0
    for lg in logs:
        a = dict(log=os.path.basename(lg), stages={}, t_start=None, t_end=None, rc=None, t_last=None)
        pending, last_fwd = None, None
        for ln in open(lg, errors="replace").read().splitlines():
            if ln.startswith("V3G_START"):
                a["t_start"] = int(ln.split()[1]); gpu = ln.split("gpu=")[1]; continue
            if ln.startswith("V3G_END"):
                a["t_end"] = int(ln.split()[1]); a["rc"] = ln.split("rc=")[1]; continue
            sp = ln.split(" ", 1)
            if len(sp) < 2 or not sp[0].isdigit():
                continue
            t, msg = int(sp[0]), sp[1]
            a["t_last"] = t
            m = re.search(r"======= (\w+) Train from (\w+)", msg)
            if m:
                stage = m.group(1); a["stages"][stage] = t; last_fwd = None; pending = None
                if stage == "dynamic":
                    traj["dynamic"] = []; dyn_loss = []
                continue
            m = INFO.search(msg)
            if m and stage:
                rec = dict(t=t, E=fl(m.group(3)), nu=fl(m.group(4)), tau_y=fl(m.group(5)))
                if stage == "dynamic":
                    if pending is None:
                        pending, retries = rec, 0
                    else:
                        retries += 1  # NaN -> official code halves dt and re-simulates
                elif stage != "static" or not traj["static"]:
                    traj[stage].append(rec)
                continue
            m = re.search(r"\[Forward\] loss: " + NUM, msg)
            if m and stage == "dynamic":
                last_fwd = float(m.group(1)); continue
            if msg.startswith("Time elaspsed") and stage == "dynamic" and pending is not None:
                pending.update(t_eval_end=t, loss=last_fwd, nan_retries=retries)
                traj["dynamic"].append(pending); dyn_loss.append(last_fwd); pending, last_fwd = None, None
        a["dynamic_iteration_in_progress"] = pending
        attempts.append(a)
    stage_wall = {}
    for a in attempts:
        st = sorted(a["stages"].items(), key=lambda kv: kv[1])
        end = a["t_end"] or a["t_last"]
        if a["t_start"] and st:
            stage_wall["data_prep_matting"] = stage_wall.get("data_prep_matting", 0) + st[0][1] - a["t_start"]
        for i, (k, t0) in enumerate(st):
            t1 = st[i + 1][1] if i + 1 < len(st) else end
            stage_wall[k] = stage_wall.get(k, 0) + (t1 - t0)
    out = dict(name=name, scene=scene, views=11 if "v11" in name else 3, gpu=gpu,
               rc=attempts[-1]["rc"] if attempts else None, n_attempts=len(attempts), attempts=attempts,
               wall_s_sum_attempts=sum(((a["t_end"] or a["t_last"] or 0) - (a["t_start"] or 0)) for a in attempts),
               stage_wall_s=stage_wall)
    last_dyn = None
    for a in attempts:
        if "dynamic" in a["stages"]:
            last_dyn = (a["t_end"] or a["t_last"]) - a["stages"]["dynamic"]
    out["dynamic_wall_s_last_attempt"] = last_dyn
    out["dynamic_iters_logged"] = len(traj["dynamic"])
    out["dynamic_nan_retries_total"] = sum(r.get("nan_retries", 0) for r in traj["dynamic"])
    out["velocity_evals"] = len(traj["velocity"])
    out["dynamic_loss"] = dyn_loss
    out["dynamic_traj"] = traj["dynamic"]
    ck = os.path.join(R, "runs/v3g/pacnerf", name, "ckpt/model/train_dynamic.pt")
    est = None
    c = None
    if os.path.exists(ck):
        import torch
        c = torch.load(ck, map_location="cpu", weights_only=False)
    finished = attempts and (attempts[-1]["rc"] == "0" or (attempts[-1]["rc"] is None and c is not None and int(c["epoch"]) == 2
                                                           and len(traj["dynamic"]) >= 149))
    if finished and attempts[-1]["rc"] is None:
        out["note"] = "wrapper V3G_END line missing (wrapper shell lost); train_dynamic.pt has epoch 2 and 150 dynamic iterations are logged -> treated as completed; end time = last log line"
    out["completed"] = bool(finished)
    if finished:
        sd = c["model_state_dict"]
        est = dict(E=10 ** float(sd["global_E"]), nu=0.45 * math.tanh(float(sd["global_nu"]) / 0.45),
                   source="final train_dynamic.pt", ckpt_stage_epoch=int(c["epoch"]))
        if GT[scene].get("tau_y"):
            est["tau_y"] = 10 ** float(sd["yield_stress"])
    elif traj["dynamic"]:
        r = traj["dynamic"][-1]
        est = dict(E=r["E"], nu=r["nu"], source="last logged dynamic-iteration params (run not finished)")
        if GT[scene].get("tau_y"):
            est["tau_y"] = r["tau_y"]
    out["estimate"] = est
    out["gt"] = GT[scene]
    if est and est.get("E") and est["E"] > 0 and math.isfinite(est["E"]):
        err = dict(log10_E_abs=abs(math.log10(est["E"]) - math.log10(GT[scene]["E"])),
                   E_rel=abs(est["E"] - GT[scene]["E"]) / GT[scene]["E"],
                   nu_abs=abs(est["nu"] - GT[scene]["nu"]) if est.get("nu") is not None else None)
        if est.get("tau_y"):
            err["log10_tau_y_abs"] = abs(math.log10(est["tau_y"]) - math.log10(GT[scene]["tau_y"]))
            err["tau_y_rel"] = abs(est["tau_y"] - GT[scene]["tau_y"]) / GT[scene]["tau_y"]
        out["error"] = err
    return out


if __name__ == "__main__":
    os.makedirs(os.path.join(R, "docs/v3g/pn_results"), exist_ok=True)
    for n in sys.argv[1:]:
        o = summarize(n)
        json.dump(o, open(os.path.join(R, "docs/v3g/pn_results", f"{n}.json"), "w"), indent=1)
        print(n, "rc", o["rc"], "att", o["n_attempts"], "stages", o["stage_wall_s"], "dyn_iters", o["dynamic_iters_logged"],
              "est", o["estimate"], "err", o.get("error"))
