"""V3x report: 60-sample test refinement analysis (docs/v3d original 30 + docs/v3x new 30) and new-30 replication.
Metric functions from scripts/v3b/c1_report.py (unchanged). For jobs still running, budget-k errors are reconstructed
from state.json evals with the exact best_after/z logic of scripts/v3d/c1_refine.py (nested budgets), so a sample
counts at k once its search has done >= k simulations; completed samples use their stored budgets."""
import json, math, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/v3b"))
from c1_report import boot_mean, crossing  # noqa: E402
STATS = json.loads((ROOT / "data/manifests/v2main.specific.stats.json").read_text())
LAWS = ["jelly", "metal", "plasticine"]
ARMS = ["prior_top1", "lawmean_best3", "lawmean_gt", "prior_top2"]
PROBES = {"prior_top1": 0, "prior_top2": 2, "lawmean_best3": 3, "lawmean_gt": 0}
KS = [0, 4, 12, 24]

def law_means():
    sums = {}
    for r in map(json.loads, (ROOT / "data/manifests/v2main.jsonl").read_text().splitlines()):
        if r["split"] != "train": continue
        p = r["parameters"]; sc = p["rho"] / 1000.
        vals = [p["E"] / sc, p["nu"], p["rho"], None if p.get("sigma_y") is None else p["sigma_y"] / sc]
        acc = sums.setdefault(r["law_id"], [np.zeros(4), np.zeros(4)])
        for j, v in enumerate(vals):
            if v is None: continue
            acc[0][j] += ((v if j == 1 else math.log1p(v)) - STATS["mean"][j]) / STATS["std"][j]; acc[1][j] += 1
    return {LAWS[k]: (v[0] / np.maximum(v[1], 1)).tolist() for k, v in sums.items()}
MEANS = law_means()
enc = lambda j, v: (math.log1p(v) - STATS["mean"][j]) / STATS["std"][j]

def err_at(s, k):
    """identifiable abs errors [E, sigma?] at budget k, or None if not reached."""
    if s.get("complete") and str(k) in s.get("budgets", {}):
        e = s["budgets"][str(k)]["abs_err_z"]; return [v for j, v in enumerate(e) if j in (0, 3) and v is not None]
    arm = s["arm"]; ev = s["evals"]
    if arm == "lawmean_best3" and k == 0: return None
    if "x0" not in s or s.get("law") is None: return None
    if len(ev) < k: return None
    if k == 0: lk, x = s["law_start"], s["x0"]
    else:
        e = min(ev[:k], key=lambda e: e["loss"]); lk, x = e.get("law", s["law"]), e["x"]
    z = list(s["base_z"]) if (lk == s["law"] or arm == "prior_top2") else list(MEANS[lk])
    z[0] = enc(0, 10 ** x[0])
    if lk != "jelly": z[3] = enc(3, 10 ** x[1])
    elif arm not in ("prior", "prior_sigma", "prior_top2", "prior_top1"): z[3] = 0.0
    m = np.array(s["target_mask"], bool); err = np.abs(np.array(z) - np.array(s["target_z"]))
    return [float(err[j]) for j in (0, 3) if m[j]]

def load(root, idx, arm):
    f = ROOT / root / "v2main_C_seed29" / f"{idx:06d}" / arm / "state.json"
    return json.loads(f.read_text()) if f.exists() else None

def main():
    orig = [s["index"] for s in json.loads((ROOT / "docs/v3d/C1_TEST_SAMPLES.json").read_text())["samples"]]
    new = json.loads((ROOT / "docs/v3x/SAMPLES.json").read_text())["new_order"]
    S = {}
    for i in orig:
        S[i] = {a: load("data/v3d_refine_test", i, a) for a in ARMS}
    for i in new:
        S[i] = {a: load("data/v3x_refine_test", i, a) for a in ARMS[:3]}
        S[i]["prior_top2"] = load("data/v3x_refine_test", i, "prior_top2")
    obj = {i: next(s for s in S[i].values() if s)["object"] if any(S[i].values()) else None for i in S}
    def E(i, a, k):
        s = S[i][a]; return err_at(s, k) if s else None
    def done(i, a):
        s = S[i][a]; return bool(s and s.get("complete"))
    out = {"generated_utc": __import__("time").strftime("%Y-%m-%d %H:%M UTC", __import__("time").gmtime()),
           "protocol": "docs/v3x/PROTOCOL.json", "metric": "pooled identifiable z-MAE (E/rho, sigma_y/rho)",
           "progress_new": {a: {"complete": sum(done(i, a) for i in new),
                                "sims_done": sum(len(S[i][a]["evals"]) for i in new if S[i][a])} for a in ARMS}}
    sets = {"combined_60": orig + new, "new_30": new, "original_30": orig}
    for name, idx in sets.items():
        R = {}
        # per-arm table on samples where that arm reached k (n reported per cell)
        tab = {}
        for a in ARMS:
            for k in KS + [3]:
                have = [i for i in idx if E(i, a, k) is not None]
                if have:
                    vals = [v for i in have for v in E(i, a, k)]
                    tab.setdefault(a, {})[str(k)] = {"mae": float(np.mean(vals)), "n": len(have), "objects": len({obj[i] for i in have})}
        R["arm_table_reached_k"] = tab
        # paired differences on samples where both arms reached the budget; complete-pair set reported too
        pairs = [("prior_top1", "lawmean_best3"), ("prior_top1", "lawmean_gt"), ("prior_top2", "lawmean_best3"),
                 ("prior_top2", "lawmean_gt"), ("prior_top1", "prior_top2")]
        P = {}
        for a, b in pairs:
            both = [i for i in idx if done(i, a) and done(i, b)]
            row = {"n_paired_complete": len(both), "objects": len({obj[i] for i in both})}
            for k in KS:
                kb = 3 if (k == 0 and b == "lawmean_best3") else k
                sel = [i for i in both]
                if not sel: continue
                d = [np.mean(E(i, a, k)) - np.mean(E(i, b, kb)) for i in sel]
                row[f"k{k}"] = {**boot_mean(d, [obj[i] for i in sel]),
                                "mae_a": float(np.mean([v for i in sel for v in E(i, a, k)])),
                                "mae_b": float(np.mean([v for i in sel for v in E(i, b, kb)]))}
            P[f"{a}_minus_{b}"] = row
            # partial variant: samples where both arms have REACHED budget k (incomplete searches included), n per k
            rp = {}
            for k in KS:
                kb = 3 if (k == 0 and b == "lawmean_best3") else k
                sel = [i for i in idx if E(i, a, k) is not None and E(i, b, kb) is not None]
                if sel:
                    d = [np.mean(E(i, a, k)) - np.mean(E(i, b, kb)) for i in sel]
                    rp[f"k{k}"] = {**boot_mean(d, [obj[i] for i in sel]), "mae_a": float(np.mean([v for i in sel for v in E(i, a, k)])),
                                   "mae_b": float(np.mean([v for i in sel for v in E(i, b, kb)]))}
            P[f"{a}_minus_{b}_reached_k"] = rp
        R["paired"] = P
        # Table-2 style rows: top1/best3 on the paired-complete set; GT / top2 on their own complete sets
        Pset = [i for i in idx if done(i, "prior_top1") and done(i, "lawmean_best3")]
        T2 = {"paired_set": {"n": len(Pset), "objects": len({obj[i] for i in Pset})}}
        for a, sel in (("prior_top1", Pset), ("lawmean_best3", Pset),
                       ("lawmean_gt", [i for i in idx if done(i, "lawmean_gt")]), ("prior_top2", [i for i in idx if done(i, "prior_top2")])):
            if not sel: continue
            T2[a] = {"n": len(sel), "objects": len({obj[i] for i in sel}),
                     **{f"k{k}": float(np.mean([v for i in sel for v in E(i, a, k)])) for k in KS + [3] if not (a == "lawmean_best3" and k == 0)}}
        R["table2"] = T2
        # sims for lawmean_gt to reach the prior k=0 error (prior k=0 is identical for top1/top2)
        for KMAX in (24, 12):
            sp = [i for i in idx if (done(i, "lawmean_gt") or E(i, "lawmean_gt", KMAX) is not None) and E(i, "prior_top1", 0) is not None]
            if sp:
                ks = [k for k in [3, 4, 8, 12, 16, 24] if k <= KMAX]
                def point(sel):
                    pm = float(np.mean([v for i in sel for v in E(i, "prior_top1", 0)]))
                    uc = {k: float(np.mean([v for i in sel for v in E(i, "lawmean_gt", k)])) for k in [0] + ks}
                    return crossing(pm, {k: v for k, v in uc.items() if k > 0}), pm, uc
                est, pm, uc = point(sp)
                groups = {}
                for i in sp: groups.setdefault(obj[i], []).append(i)
                g = list(groups.values()); rng = np.random.default_rng(17); bs = []
                for _ in range(1000):
                    bs.append(point([i for j in rng.integers(0, len(g), len(g)) for i in g[j]])[0])
                bs = np.array(bs); q = np.quantile(bs, [.025, .975])
                f = lambda v: None if not np.isfinite(v) else float(v)
                R["lawmean_gt_sims_to_reach_prior_k0" + ("" if KMAX == 24 else f"_within{KMAX}")] = {"kmax": KMAX, "n": len(sp), "objects": len(g), "prior_k0_mae": pm, "lawmean_gt_curve": uc,
                                                         "sims": f(est), "ci95": [f(q[0]), f(q[1])], "frac_censored": float(np.mean(~np.isfinite(bs)))}
        # unseen-action replay IoU at k=24
        rep = {}
        for a in ARMS:
            v = [S[i][a]["replay"]["24"]["iou"] for i in idx if done(i, a) and isinstance(S[i][a]["replay"].get("24"), dict) and "iou" in S[i][a]["replay"]["24"]]
            if v: rep[a] = {"iou_k24": float(np.mean(v)), "n": len(v)}
        R["replay_iou_unseen_action_k24"] = rep
        lacc = {}
        for a in ARMS:
            c = [i for i in idx if done(i, a)]
            if c: lacc[a] = {"k24_law_acc": float(np.mean([S[i][a]["budgets"]["24"]["law"] == S[i][a]["gt_law"] for i in c])), "n": len(c)}
        R["law_accuracy"] = lacc
        R["failures"] = {a: {"failed": sum(bool(e.get("failed")) for i in idx if S[i][a] for e in S[i][a]["evals"]),
                             "sims": sum(len(S[i][a]["evals"]) for i in idx if S[i][a])} for a in ARMS}
        out[name] = R
    out["per_sample_new"] = {str(i): {"object": obj[i], **{a: ({"complete": done(i, a), "sims": len(S[i][a]["evals"]),
                              **{f"k{k}": (None if E(i, a, k) is None else float(np.mean(E(i, a, k)))) for k in KS}} if S[i][a] else None) for a in ARMS}} for i in new}
    (ROOT / "docs/v3x/RESULTS.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["progress_new"]))
    for name in sets:
        print(name, json.dumps(out[name]["arm_table_reached_k"]))

if __name__ == "__main__":
    main()
