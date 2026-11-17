"""v3g: score a Spring-Gaus official run against the GT MPM simulation point clouds.
Usage: python -I sg_eval.py --exp <exp dir> --gt <simulation/<obj>> --out <json> [--n-fit 20 --dt 0.03 --eval-freq 60]
Fit frames (0..n_fit-1, observed in training) use evaluations/simulate_fit/fit_{k}.ply (time k*dt).
Future frames (n_fit..29, never observed) use evaluations/simulate_pred/pred_{j}.ply at time j/eval_freq,
linearly interpolated in time to the GT times (particles correspond 1:1 across pred frames).
CD = mean of the two directed mean nearest-neighbour L2 distances (and its squared variant, x1e3),
both clouds subsampled to 8192 points with a fixed seed. PSNR on evaluations/images_fit (pred|gt side by side)."""
import argparse, json, os, glob
import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
import imageio.v2 as imageio

def load(p):
    v = PlyData.read(p)["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64)

def sub(x, n=8192, seed=0):
    if len(x) <= n: return x
    return x[np.random.default_rng(seed).choice(len(x), n, replace=False)]

def cd(a, b):
    a, b = sub(a), sub(b)
    d1, _ = cKDTree(b).query(a); d2, _ = cKDTree(a).query(b)
    return 0.5 * (d1.mean() + d2.mean()), 0.5 * ((d1**2).mean() + (d2**2).mean())

ap = argparse.ArgumentParser()
ap.add_argument("--exp"); ap.add_argument("--gt"); ap.add_argument("--out")
ap.add_argument("--n-fit", type=int, default=20); ap.add_argument("--n-all", type=int, default=30)
ap.add_argument("--dt", type=float, default=0.03); ap.add_argument("--eval-freq", type=float, default=60)
a = ap.parse_args()
ev = os.path.join(a.exp, "evaluations")
res = {"fit": {}, "future": {}}
for k in range(a.n_fit):
    p = os.path.join(ev, "simulate_fit", f"fit_{k}.ply")
    if os.path.exists(p):
        c, c2 = cd(load(p), load(os.path.join(a.gt, f"{k:03}.ply")))
        res["fit"][k] = {"cd": c, "cd2_x1e3": c2 * 1e3}
for k in range(a.n_fit, a.n_all):
    t = k * a.dt * a.eval_freq; j0 = int(np.floor(t)); w = t - j0
    p0 = os.path.join(ev, "simulate_pred", f"pred_{j0}.ply"); p1 = os.path.join(ev, "simulate_pred", f"pred_{j0+1}.ply")
    if os.path.exists(p0) and os.path.exists(p1):
        x = (1 - w) * load(p0) + w * load(p1)
        c, c2 = cd(x, load(os.path.join(a.gt, f"{k:03}.ply")))
        cs, _ = cd(load(os.path.join(a.gt, "000.ply")), load(os.path.join(a.gt, f"{k:03}.ply")))
        res["future"][k] = {"cd": c, "cd2_x1e3": c2 * 1e3, "pred_time_index": t, "cd_static_baseline": cs}
ps = []
for f in sorted(glob.glob(os.path.join(ev, "images_fit", "*.png"))):
    im = imageio.imread(f).astype(np.float64) / 255.0; W = im.shape[1] // 2
    mse = ((im[:, :W] - im[:, W:]) ** 2).mean(); ps.append(10 * np.log10(1.0 / max(mse, 1e-12)))
summ = {}
for key in ("fit", "future"):
    if res[key]:
        summ[key + "_cd_mean"] = float(np.mean([v["cd"] for v in res[key].values()]))
        summ[key + "_cd2_x1e3_mean"] = float(np.mean([v["cd2_x1e3"] for v in res[key].values()]))
        summ[key + "_n_frames"] = len(res[key])
if res["future"]: summ["future_cd_static_baseline_mean"] = float(np.mean([v["cd_static_baseline"] for v in res["future"].values()]))
if ps: summ["fit_psnr_mean_eval_cam"] = float(np.mean(ps)); summ["fit_psnr_n_images"] = len(ps)
out = {"summary": summ, "per_frame": res, "exp": a.exp}
json.dump(out, open(a.out, "w"), indent=1)
print(json.dumps(summ, indent=1))
