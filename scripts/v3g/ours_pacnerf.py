"""Our model's predictions on the 4 PAC-NeRF GT scenes, taken from docs/v3c/PACNERF_FEWSHOT.json (bgsub input, canonical triplet t0,
3-seed ensemble). Ensemble = mean over seeds in log10 for E and tau_y (== mean of z), arithmetic mean for nu.
Writes docs/v3g/pn_results/OURS.json"""
import json, math, os

R = "/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test"
GT = {"torus": dict(E=1e6, nu=0.3), "bird": dict(E=3e5, nu=0.3),
      "playdoh": dict(E=2e6, nu=0.3, tau_y=1.54e4), "cat": dict(E=1e6, nu=0.3, tau_y=3.85e3)}
d = json.load(open(os.path.join(R, "docs/v3c/PACNERF_FEWSHOT.json")))
out = dict(source="docs/v3c/PACNERF_FEWSHOT.json variants.bgsub seeds 17/29/43, decoded_t0 (views [4,6,7]), rho assumed 1000",
           methods={})
for var in ("bgsub", "raw"):
    seeds = d["variants"][var]["seeds"]
    for m in ("K0", "offset", "support_mean_baseline"):
        key = f"{var}/{m}"
        out["methods"][key] = {}
        for sc in GT:
            dec = [seeds[s]["scenes"][sc][m]["decoded_t0"] for s in sorted(seeds)]
            law = [seeds[s]["scenes"][sc][m]["law_pred_t0"] for s in sorted(seeds)]
            E = 10 ** (sum(math.log10(x["E_over_rho_x1000"]) for x in dec) / len(dec))
            nu = sum(x["nu"] for x in dec) / len(dec)
            est = dict(E=E, nu=nu)
            if GT[sc].get("tau_y"):
                est["tau_y"] = 10 ** (sum(math.log10(x["sigma_y_over_rho_x1000"]) for x in dec) / len(dec))
            err = dict(log10_E_abs=abs(math.log10(E) - math.log10(GT[sc]["E"])), E_rel=abs(E - GT[sc]["E"]) / GT[sc]["E"],
                       nu_abs=abs(nu - GT[sc]["nu"]))
            if "tau_y" in est:
                err["log10_tau_y_abs"] = abs(math.log10(est["tau_y"]) - math.log10(GT[sc]["tau_y"]))
                err["tau_y_rel"] = abs(est["tau_y"] - GT[sc]["tau_y"]) / GT[sc]["tau_y"]
            per_seed_err = [abs(math.log10(x["E_over_rho_x1000"]) - math.log10(GT[sc]["E"])) for x in dec]
            out["methods"][key][sc] = dict(estimate=est, gt=GT[sc], error=err, law_pred_per_seed=law,
                                           per_seed_log10_E_abs=per_seed_err)
os.makedirs(os.path.join(R, "docs/v3g/pn_results"), exist_ok=True)
json.dump(out, open(os.path.join(R, "docs/v3g/pn_results", "OURS.json"), "w"), indent=1)
for k, v in out["methods"].items():
    print(k, {s: (round(x["estimate"]["E"]), round(x["estimate"]["nu"], 3), round(x["estimate"].get("tau_y", 0)),
                  round(x["error"]["log10_E_abs"], 3)) for s, x in v.items()})
