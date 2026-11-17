"""v3g: merge Spring-Gaus official runs (logs + sg_eval + sg_params) and our model (docs/v3f) into docs/v3g/SPRINGGAUS_OFFICIAL.json."""
import json, os, re, sys, math
R = "/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test"
L = f"{R}/docs/v3g/logs"
scenes = sys.argv[1:] or ["torus", "duck", "apple"]
ANSI = re.compile(r"\x1b\[[0-9;]*m")
def secs(t):
    s = 0
    for x in t.split(":"): s = s * 60 + int(x)
    return s
def bar_times(log, pat):
    out = []
    for line in open(log, errors="ignore").read().replace("\r", "\n").split("\n"):
        line = ANSI.sub("", line)
        if re.search(pat, line):
            m = re.search(r"\[(\d+(?::\d+){1,2})<", line)
            if m: out.append(secs(m.group(1)))
    return out
def last(x): return x[-1] if x else None
v3f = json.load(open(f"{R}/docs/v3f/SPRINGGAUS_SYN.json"))
gt_all = v3f["protocol"]["gt"]; ens = v3f["ensemble9"]
res = {"method": "Spring-Gaus (official code, github.com/Colmar-zlicheng/Spring-Gaus @ 62a1bb5, Apache-2.0)",
       "protocol": {
         "data": "Spring-Gaus mpm_synthetic (official Google Drive zip downloaded by v3f; 3 scenes extracted to data/external_datasets/v3g_springgaus_extract)",
         "config": "official config/mpm_synthetic/<scene>.yaml + default.yaml unchanged; train.py -g <gpu> --cfg ... --eval_cam 5",
         "views": "all 10 cameras used for training (official protocol); eval_cam 5 is one of the training cameras (render-only)",
         "frames": "first 20 of 30 frames (N_FRAME 20, dt 0.03 s) observed; frames 20-29 (0.60-0.87 s) are future prediction",
         "stages": "static 3DGS 30000 it -> velocity 100 it (+3000 Gaussian refine) -> dynamic spring-mass 300 it (+3000 refine first; n_step 100->150->200 at it 100/200) -> eval",
         "metrics": "Chamfer distance (mean of the two directed mean NN L2 distances, 8192-pt subsample, scene units; also squared x1e3) between Spring-Gaus anchor-interpolated Gaussian centres and GT MPM particles. Fit = frames 0-19 at training dt; future = frames 20-29 from its 60 Hz prediction rollout linearly interpolated in time; static baseline = GT frame 0 vs GT frame k. PSNR on eval cam 5 fit renders vs GT images. Metric code is ours (scripts/v3g/sg_eval.py): the official repo writes point clouds but ships no CD/EMD script, so numbers are not directly comparable to the paper table.",
         "params": "Spring-Gaus does not output E or nu: it learns per-anchor spring stiffness log10 k (2048 anchors, unit mass per anchor, init 3.0), friction, rebound, soft-k. Only the ranking of median log10 k vs GT E can be compared.",
         "patches": "build-compat only: #include <cfloat> added to submodules/simple-knn/simple_knn.cu (CUDA 12.6); env torch 2.4.1+cu121, pytorch3d 0.7.8, open3d 0.17.0, pillow 9.5.0 (pillow>=10 breaks its np.byte image loader); no algorithmic change",
         "hardware": "1x NVIDIA A800-SXM4-80GB (GPU 7), shared with other jobs at ~100% util, so wall-clock is an upper bound",
         "restart_note": "torus was first started on GPU 0 at 01:30Z and aborted in the dynamic stage (GPU reassignment); restarted from scratch on GPU 7; the GPU-7 run is the one reported"},
       "scenes": {}}
for s in scenes:
    e = {"gt": gt_all.get(s), "status": "not_started"}
    tf = f"{L}/sg_time_{s}.txt"
    if os.path.exists(tf):
        t = open(tf).read().split("\n")
        st = [x for x in t if x.startswith("START")]; en = [x for x in t if x.startswith("END")]
        if en:
            rc = int(re.search(r"rc=(\d+)", en[0]).group(1))
            e["wall_clock_total_s"] = int(en[0].split()[-1]) - int(st[0].split()[-1])
            e["status"] = "completed" if rc == 0 else ("timeout" if rc == 124 else f"failed rc={rc}")
        else: e["status"] = "running"
    import glob
    log = f"{L}/sg_train_{s}.log"
    if os.path.exists(log):
        logs = [log] + sorted(glob.glob(f"{L}/sg_train_{s}_*.log"))
        def first_found(pat):
            for lg in logs:
                v = last(bar_times(lg, pat))
                if v is not None: return v, os.path.basename(lg)
            return None, None
        st_t, st_src = first_found(r"Static loss: .*30000/30000")
        ve_t, ve_src = first_found(r"Velocity frame: .*100/100")
        refs = {}
        for lg in logs:
            for v in bar_times(lg, r"REFINE loss: .*3000/3000"): refs.setdefault(os.path.basename(lg), set()).add(v)
        e["stage_wall_clock_s"] = {"static_30000it": st_t, "static_source_log": st_src,
                                   "velocity_100it": ve_t, "velocity_source_log": ve_src,
                                   "dynamic_300it": last(bar_times(log, r"Dynamic frame: .*300/300")),
                                   "refines_3000it": {k: sorted(v) for k, v in refs.items()}}
        if st_src and st_src != os.path.basename(log) and "wall_clock_total_s" in e:
            e["wall_clock_successful_attempt_s"] = e["wall_clock_total_s"]
            vref = max(refs.get(ve_src, {0}))
            e["wall_clock_total_s"] = e["wall_clock_total_s"] + st_t + ve_t + vref
            e["wall_clock_note"] = ("static+velocity(+its refine) ran in an earlier attempt that later hit CUDA OOM in the dynamic stage "
                                    "(GPU co-located with other jobs); their checkpoints were reused, so total = successful attempt + those stage times. "
                                    "Failed attempts' wasted compute is excluded; see logs/sg_time_<scene>_*.txt")
    e["iterations"] = {"static": 30000, "velocity": 100, "velocity_refine": 3000, "dynamic": 300, "dynamic_refine": 3000}
    for k, f in (("metrics", f"{L}/sg_eval_{s}.json"), ("learned_params", f"{L}/sg_params_{s}.json")):
        if os.path.exists(f):
            d = json.load(open(f)); e[k] = d.get("summary", d)
    our = {}
    for mode in ("K0", "offset", "ridge_last_layer", "support_mean_baseline"):
        dec = ens[mode]["decoded_t0"].get(s)
        if dec and gt_all.get(s):
            our[mode] = {"E_Pa_at_rho1000": dec["E_Pa_at_rho1000"], "nu": dec["nu"],
                         "abs_log10_E_err": abs(math.log10(dec["E_Pa_at_rho1000"] / gt_all[s]["E"])),
                         "abs_nu_err": abs(dec["nu"] - gt_all[s]["NU"])}
    e["our_model_v3f"] = {"source": "docs/v3f/SPRINGGAUS_SYN.json ensemble9 (V2 main B/C/D x 3 seeds), canonical 3-view triplet t0, 30 frames; offset/ridge = leave-one-scene-out over 14 scenes; support_mean_baseline ignores the video; cost = one forward pass per run, no per-scene optimisation", **our}
    res["scenes"][s] = e
done = [s for s in scenes if "learned_params" in res["scenes"][s]]
if len(done) >= 2:
    res["ranking_check"] = {"scenes": done, "gt_E": [gt_all[s]["E"] for s in done],
                            "springgaus_median_log10_k": [res["scenes"][s]["learned_params"]["log10_k_median"] for s in done],
                            "ours_K0_E": [res["scenes"][s]["our_model_v3f"]["K0"]["E_Pa_at_rho1000"] for s in done]}
json.dump(res, open(f"{R}/docs/v3g/SPRINGGAUS_OFFICIAL.json", "w"), indent=1)
print(json.dumps({s: {k: res["scenes"][s].get(k) for k in ("status", "wall_clock_total_s", "stage_wall_clock_s", "metrics", "learned_params", "our_model_v3f")} for s in scenes}, indent=1))
print(json.dumps(res.get("ranking_check"), indent=1))
