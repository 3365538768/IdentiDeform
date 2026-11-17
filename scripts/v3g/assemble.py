"""Assemble docs/v3g/OFFICIAL_BASELINES.json from per-run results (run from project root with python3 -I)."""
import json, math, os, glob

R = "/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test"
P = os.path.join(R, "docs/v3g/pn_results")


def J(p):
    p = os.path.join(R, p)
    return json.load(open(p)) if os.path.exists(p) else None


scenes = ["torus", "bird", "playdoh", "cat"]
ours = J("docs/v3g/pn_results/OURS.json")
timing = J("docs/v3g/pn_results/OURS_TIMING_torus.json")
hyb = J("docs/v3g/pn_results/HYBRID.json") or {}
notes = J("docs/v3g/pn_results/RUN_NOTES.json") or {}

pac = []
for sc in scenes:
    for v in ("v11", "v3t0"):
        n = f"{sc}_{v}"
        r = J(f"docs/v3g/pn_results/{n}.json")
        if r is None:
            pac.append(dict(scene=sc, run=n, status="not run")); continue
        sw = r["stage_wall_s"]
        prod = sum(x for k, x in sw.items() if x)
        status = "completed (150/150 dynamic iterations)" if r.get("completed") else f"not completed: {r['dynamic_iters_logged']}/150 dynamic iterations"
        if n in notes:
            status += "; " + notes[n]
        pac.append(dict(
            scene=sc, run=n, inputs="PAC-NeRF data.zip renders, 800x800 RGB, 14 frames @24 fps, official background matting",
            views=(11 if v == "v11" else [4, 6, 7]), config=f"external/pacnerf/configs/{sc}.py (official) + scripts/v3g/pn_cfgs/{n}.py (paths, n_cameras, taichi pool only)",
            iterations=dict(static=6000, velocity_lbfgs_evals=r["velocity_evals"], dynamic_completed=r["dynamic_iters_logged"], dynamic_schedule="50 (hit_frame+warmup frames) + 100 (all frames), Adam",
                            dynamic_nan_dt_halving_retries=r.get("dynamic_nan_retries_total")),
            wall_clock_s=dict(stage=sw, total_productive=prod, attempts=r["n_attempts"],
                              note="shared A800 at 100% util from other tenants; several PAC-NeRF runs co-located -> upper bound"),
            gpu=r.get("gpu"), estimate=r.get("estimate"), gt=r["gt"], error=r.get("error"), status=status))

ours_rows = {}
for key in ("bgsub/K0", "bgsub/offset", "bgsub/support_mean_baseline", "raw/K0", "raw/offset"):
    ours_rows[key] = ours["methods"][key]

out = dict(
    created="2026-11-14 (v3g)", deadline="2026-11-15T06:00Z",
    protocol_notes=[
        "All PAC-NeRF numbers come from the official code (github.com/xuan-li/PAC-NeRF @ b613048), official per-scene configs; only data/output paths, n_cameras (3-view variant) and the Taichi memory pool fraction (0.6 -> 0.12/0.2, needed on shared GPUs) were changed.",
        "rho is fixed to 1000 by the official configs (not optimised) and assumed 1000 by our model; GT = PAC-NeRF paper Table 1.",
        "3-view variant uses the same canonical triplet t0 = cameras [4,6,7] as our model (docs/v3c/PACNERF_FEWSHOT.json), renumbered 0..2.",
        "Engines differ: PAC-NeRF fits its own differentiable MPM to its own MPM renders (in-engine); our model was trained on a different simulator/action family and is applied zero-shot (K0) or with leave-one-scene-out offset calibration.",
        "Wall-clock measured on shared A800-80GB GPUs (100% utilisation by other agents, up to 5 co-located v3g runs); treat as upper bounds; the README of PhysTwin and PAC-NeRF quote faster single-job times.",
        "Errors: |log10 E_hat - log10 E_gt|, |nu_hat - nu_gt|, |log10 tauY_hat - log10 tauY_gt|.",
    ],
    pacnerf_official=pac,
    ours_on_pacnerf_scenes=dict(rows=ours_rows, wall_clock=timing,
                                note="decoded from docs/v3c/PACNERF_FEWSHOT.json (3-seed ensemble, triplet t0); K0 = zero-shot, offset = leave-one-scene-out calibration using the other 3 scenes' GT; support_mean_baseline = no-video reference"),
    hybrid_prior_init=dict(results=hyb, notes=J("docs/v3g/pn_results/HYBRID_NOTES.json")),
    springgaus_official=J("docs/v3g/SPRINGGAUS_OFFICIAL.json"),
    phystwin_official=J("docs/v3g/PHYSTWIN_OFFICIAL.json"),
)
json.dump(out, open(os.path.join(R, "docs/v3g/OFFICIAL_BASELINES.json"), "w"), indent=1)
for p in pac:
    e = p.get("error") or {}
    print(p["run"], p["status"][:60], round(p["wall_clock_s"]["total_productive"]) if "wall_clock_s" in p else None,
          {k: round(v, 3) for k, v in e.items() if v is not None and ("log10" in k or "nu" in k)})
