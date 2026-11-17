"""v3g: summarise the learned Spring-Gaus parameters of a finished run (weights_only load).
global_k = per-anchor log10 spring stiffness (init log10 GLOBAL_K = 3); fric/rebound/soft are scalar logits."""
import sys, json, torch, numpy as np
d = torch.load(sys.argv[1], map_location="cpu", weights_only=True)
k = d["global_k"].float().numpy()
out = {"n_anchors": int(k.size), "log10_k_median": float(np.median(k)), "log10_k_mean": float(k.mean()),
       "log10_k_p10": float(np.percentile(k, 10)), "log10_k_p90": float(np.percentile(k, 90)),
       "fric_k_eff": float(np.clip(1 / (1 + np.exp(-float(d["fric_k"][0]))) * 1.2 - 0.1, 0, 1)),
       "rebound_k_sigmoid": float(1 / (1 + np.exp(-float(d["rebound_k"][0])))),
       "soft_vector": float(d["soft_vector"][0]) if "soft_vector" in d else None}
json.dump(out, open(sys.argv[2], "w"), indent=1); print(json.dumps(out))
