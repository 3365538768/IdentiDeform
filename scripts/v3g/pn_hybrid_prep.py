"""Hybrid / restart prep: copy static+velocity checkpoints of a finished v3g PAC-NeRF run into a new run dir and
(optionally) overwrite the physical-parameter initialisation (E, nu, tau_y) used by the dynamic stage.
Usage (pacnerf venv python): pn_hybrid_prep.py <src_name> <dst_name> <E|keep> <nu|keep> [<tau_y|keep>]"""
import math, os, shutil, sys, json
import torch
R = "/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test"
src, dst = sys.argv[1], sys.argv[2]
E, nu = sys.argv[3], sys.argv[4]; tau = sys.argv[5] if len(sys.argv) > 5 else "keep"
sd_dir = os.path.join(R, "runs/v3g/pacnerf", src, "ckpt/model"); dd = os.path.join(R, "runs/v3g/pacnerf", dst, "ckpt/model")
os.makedirs(dd, exist_ok=True); os.makedirs(os.path.join(R, "runs/v3g/pacnerf", dst, "ckpt/image"), exist_ok=True)
log = {}
for st in ("static", "velocity"):
    ck = torch.load(os.path.join(sd_dir, f"train_{st}.pt"), map_location="cpu", weights_only=False)
    m = ck["model_state_dict"]
    before = dict(E=10 ** float(m["global_E"]), nu=0.45 * math.tanh(float(m["global_nu"]) / 0.45), tau_y=10 ** float(m["yield_stress"]))
    if E != "keep": m["global_E"] = torch.tensor([math.log10(float(E))]).float()
    if nu != "keep": m["global_nu"] = torch.tensor([0.45 * math.atanh(float(nu) / 0.45)]).float()
    if tau != "keep": m["yield_stress"] = torch.tensor([math.log10(float(tau))]).float()
    after = dict(E=10 ** float(m["global_E"]), nu=0.45 * math.tanh(float(m["global_nu"]) / 0.45), tau_y=10 ** float(m["yield_stress"]))
    torch.save(ck, os.path.join(dd, f"train_{st}.pt")); log[st] = dict(epoch=int(ck["epoch"]), before=before, after=after)
json.dump(dict(src=src, dst=dst, args=sys.argv[3:], ckpts=log), open(os.path.join(R, "runs/v3g/pacnerf", dst, "hybrid_init.json"), "w"), indent=1)
print(json.dumps(log))
