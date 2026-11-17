"""Compare dynamic-stage convergence of PAC-NeRF from its default init vs. an init taken from our model's prediction.
Inputs: docs/v3g/pn_results/<name>.json produced by pn_extract.py.  Usage: pn_hybrid_analyze.py <scene> <default_name> <hybrid_name> [...]
Pairs are given as scene:default:hybrid triples."""
import json, math, os, sys

R = "/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test"
N_STAGE1 = 50  # official schedule: 50 iters on hit_frame+warmup frames, then 100 iters on all frames


def load(n):
    return json.load(open(os.path.join(R, "docs/v3g/pn_results", f"{n}.json")))


def smooth(x, k=5):
    out = []
    for i in range(len(x)):
        w = [v for v in x[max(0, i - k + 1): i + 1] if v is not None and math.isfinite(v)]
        out.append(sum(w) / len(w) if w else None)
    return out


def errs(rec, gt):
    e = dict(log10_E_abs=abs(math.log10(rec["E"]) - math.log10(gt["E"])) if rec.get("E") else None)
    if gt.get("tau_y") and rec.get("tau_y"):
        e["log10_tau_y_abs"] = abs(math.log10(rec["tau_y"]) - math.log10(gt["tau_y"]))
    return e


def analyze(scene, dn, hn):
    d, h = load(dn), load(hn)
    gt = d["gt"]
    res = dict(scene=scene, default_run=dn, hybrid_run=hn)
    for tag, r in (("default", d), ("hybrid", h)):
        tr = r["dynamic_traj"]
        loss = r["dynamic_loss"]
        t0 = tr[0]["t"] if tr else None
        res[tag] = dict(n_iters=len(tr), dynamic_wall_s=r.get("dynamic_wall_s_last_attempt"),
                        init=dict(E=tr[0]["E"], nu=tr[0]["nu"], tau_y=tr[0]["tau_y"]) if tr else None,
                        init_error=errs(tr[0], gt) if tr else None,
                        final_estimate=r.get("estimate"), final_error=r.get("error"),
                        stage1_final_loss=smooth(loss)[N_STAGE1 - 1] if len(loss) >= N_STAGE1 else None,
                        stage2_final_loss_smoothed=smooth(loss)[-1] if len(loss) > N_STAGE1 else None,
                        t_rel=[(x["t"] - t0) for x in tr] if tr else [],
                        err_traj=[errs(x, gt) for x in tr])
    # iterations / time to reach the default run's final (smoothed) stage-2 loss, and default's final E error
    ld, lh = smooth(d["dynamic_loss"]), smooth(h["dynamic_loss"])
    tgt = ld[-1] if len(ld) > N_STAGE1 else None
    def first_reach(ls, t_rel, thr):
        for i in range(N_STAGE1, len(ls)):
            if ls[i] is not None and thr is not None and ls[i] <= thr:
                return i + 1, (t_rel[i + 1] if i + 1 < len(t_rel) else None)
        return None, None
    res["target_loss_default_final_smoothed"] = tgt
    for tag, ls in (("default", ld), ("hybrid", lh)):
        it, t = first_reach(ls, res[tag]["t_rel"], tgt)
        res[tag]["iters_to_default_final_loss"] = it
        res[tag]["time_s_to_default_final_loss"] = t
    # E-error criterion: first iteration whose |log10 E err| <= default final |log10 E err|
    def first_err(et, thr):
        for i, e in enumerate(et):
            if e.get("log10_E_abs") is not None and thr is not None and e["log10_E_abs"] <= thr:
                return i
        return None
    thr = (d.get("error") or {}).get("log10_E_abs")
    res["E_err_threshold_default_final"] = thr
    for tag in ("default", "hybrid"):
        i = first_err(res[tag]["err_traj"], thr)
        res[tag]["iters_to_default_final_E_err"] = i
        res[tag]["time_s_to_default_final_E_err"] = res[tag]["t_rel"][i] if i is not None and i < len(res[tag]["t_rel"]) else None
        # reach-and-stay criteria (robust to the noisy single-camera loss): first iteration after which the estimate stays
        # within tol (log10) of (a) GT E, (b) the default run's own final E (GT-free convergence criterion); tau_y likewise if present
        dE = d["estimate"]["E"] if d.get("estimate") else None
        trj = (d if tag == "default" else h)["dynamic_traj"]
        for tol in (0.05, 0.1):
            for refname, ref in (("gt", gt["E"]), ("default_final", dE)):
                if ref is None: continue
                k = None
                for i in range(len(trj) - 1, -1, -1):
                    if trj[i]["E"] is None or abs(math.log10(trj[i]["E"]) - math.log10(ref)) > tol: break
                    k = i
                res[tag][f"iter_stay_within_{tol}_of_{refname}_E"] = k
                res[tag][f"time_s_stay_within_{tol}_of_{refname}_E"] = res[tag]["t_rel"][k] if k is not None else None
            if gt.get("tau_y"):
                k = None
                for i in range(len(trj) - 1, -1, -1):
                    if not trj[i]["tau_y"] or abs(math.log10(trj[i]["tau_y"]) - math.log10(gt["tau_y"])) > tol: break
                    k = i
                res[tag][f"iter_stay_within_{tol}_of_gt_tau_y"] = k
        del res[tag]["t_rel"]
        res[tag]["err_traj_every10"] = res[tag].pop("err_traj")[::10]
    return res


if __name__ == "__main__":
    out = {}
    for trip in sys.argv[1:]:
        s, dn, hn = trip.split(":")
        out[hn] = analyze(s, dn, hn)
        r = out[hn]
        print(s, "init_err", r["default"]["init_error"], r["hybrid"]["init_error"], "final_err", r["default"]["final_error"], r["hybrid"]["final_error"],
              "iters->loss", r["default"]["iters_to_default_final_loss"], r["hybrid"]["iters_to_default_final_loss"],
              "iters->Eerr", r["default"]["iters_to_default_final_E_err"], r["hybrid"]["iters_to_default_final_E_err"])
    json.dump(out, open(os.path.join(R, "docs/v3g/pn_results", "HYBRID.json"), "w"), indent=1)
