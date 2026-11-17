"""IdentiDeform-Touch project-page clips (v2): one mp4 per sample, 768x416 = 3 cropped views (3 x 256x256) on top and the
synchronized contact-force panel (768x160, moving cursor) below; H.264 yuv420p 30 fps CRF 23 faststart, no audio.
Also a poster JPEG at 40%, clips.json, and the supplement strip figure supp_touch_strip.pdf/.png.
Usage (from project root; needs matplotlib, so system python3): python3 scripts/project_page/make_touch_clips.py"""
import sys, os, json, glob, subprocess, numpy as np, cv2
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, "scripts/v3m"); from common import crop_box
OUT = "docs/project_assets/touch"; os.makedirs(OUT, exist_ok=True)
FPS_IN, FPS_OUT, S, FH, DUR = 32, 30, 256, 160, 4.0
# (root, val index, page action name, description)
SEL = [
    ("data/v5b_main", 198, "press", "Plate press on the support bed: the jelly duck squashes under the plate and springs back after release."),
    ("data/v5b_main", 163, "press", "Plate press: the metal bottle buckles and stays crumpled (permanent plastic deformation)."),
    ("data/v5b_main", 79, "shear_held", "Cantilever hold: the jaws grip the metal chair and step sideways; the chair folds over and keeps the bend."),
    ("data/v5b_main", 25, "shear_held", "Cantilever hold: the jaws grip one edge of the metal tray, then step sideways; the free end bends and stays bent."),
    ("data/v5b_main", 81, "stretch_held", "Hang: the jaws lift the jelly chair by its top rail; the chair hangs and stretches under its own weight."),
    ("data/v5b_main", 48, "bend_held", "Swing: the jaws hold the jelly duck and step sideways; the body swings and oscillates elastically."),
    ("data/v5b_main", 35, "press_hold_release", "Press-hold-release: the plate presses the plasticine tray, holds, and lifts; the dent remains."),
    ("data/v5f_main", 38, "force_press", "Force-controlled press (target 30): the plate pushes the plasticine duck until the reaction reaches the target, then lifts; the head and back stay dented."),
]
STRIP = ["press_jelly_000198", "shear_held_metal_000025", "stretch_held_jelly_000081", "force_press_plasticine_000038"]
PRETTY = {"press": "plate press", "shear_held": "cantilever", "stretch_held": "hang", "bend_held": "swing",
          "press_hold_release": "press-hold-release", "force_press": "force-ctrl press"}
COL = {"plate": "#222222", "jaw A": "#1f77b4", "jaw B": "#d62728"}


def forces(d):
    """-> (list of (label, t, f)), force_source string, optional target force."""
    z = np.load(f"{d}/contact_forces.npz"); t = z["t"]; Fp = z["F_plate"]; F = z["F"]
    fl = f"{d}/press_force_log.json"
    if os.path.exists(fl):  # force-controlled press: plate reaction logged at 10 Hz
        L = json.load(open(fl)); A = np.array(L["log_t_z_R"], float)
        return [("plate", A[:, 0], A[:, 2])], "press_force_log.json: plate reaction R (10 Hz), target dashed", L["force"]
    if Fp.size:
        return [("plate", t[:len(Fp)], -Fp)], "contact_forces.npz F_plate (downward plate force)", None
    mags = np.linalg.norm(F, axis=2)
    if F.shape[1] >= 2 and mags.max() > 0:  # jaws: BC segments of jaw A then jaw B, time-gated, so sum per jaw
        k = F.shape[1] // 2
        return [("jaw A", t, mags[:, :k].sum(1)), ("jaw B", t, mags[:, k:].sum(1))], "contact_forces.npz F: |F| summed per jaw", None
    return None, None, None


def force_panel(series, target, W=768, H=FH, dpi=100):
    fig, ax = plt.subplots(figsize=(W / dpi, H / dpi), dpi=dpi); fig.patch.set_facecolor("white")
    for lab, t, f in series:
        ax.plot(t, f, lw=1.4, color=COL[lab], label=lab)
    ymax = max(f.max() for _, _, f in series)
    if target is not None:
        ax.axhline(target, color="#888888", lw=0.9, ls="--", label=f"target {target:g}"); ymax = max(ymax, target)
    ax.set_xlim(0, DUR); ax.set_ylim(0, ymax * 1.12 + 1e-9)
    ax.set_xlabel("time [s]", fontsize=8, labelpad=1); ax.set_ylabel("force", fontsize=8, labelpad=2)
    ax.tick_params(labelsize=7, length=2, pad=1); ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=3)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.25, top=0.96); fig.canvas.draw()  # fixed margins: same plot box in every clip
    bg = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    (x0, y0), (x1, y1) = ax.transData.transform([(0, ax.get_ylim()[0]), (DUR, ax.get_ylim()[1])]); plt.close(fig)
    return bg, x0, x1, int(round(H - y1)), int(round(H - y0))


def enc(frames, path, crf=23):
    T, H, W = frames.shape[:3]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS_OUT), "-i", "-",
                    "-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", path],
                   input=np.ascontiguousarray(frames).tobytes(), check=True)


clips, cache = [], {}
for root, i, act, desc in SEL:
    d = f"{root}/val/{i:06d}"; g = json.load(open(f"{d}/gt.json"))
    series, src, target = forces(d)
    assert series is not None, f"no tool force for {d}"
    Z = np.load(glob.glob(f"{d}/*/sample_pack.npz")[0]); rgb = Z["rgb"]; m = Z["object_mask"][:, :, 0] > 127
    T_in = rgb.shape[1]; T_out = int(round(DUR * FPS_OUT))
    idx = np.minimum(np.round(np.arange(T_out) * FPS_IN / FPS_OUT).astype(int), T_in - 1)
    boxes, views = [], []
    for v in range(3):
        y0, y1, x0, x1 = crop_box(m[v], margin=0.08); boxes.append((y0, y1, x0, x1))
        interp = cv2.INTER_CUBIC if (y1 - y0) < S else cv2.INTER_AREA
        views.append(np.stack([cv2.resize(rgb[v, t].transpose(1, 2, 0)[y0:y1, x0:x1], (S, S), interpolation=interp) for t in idx]))
    top = np.concatenate(views, 2)
    bg, fx0, fx1, ya, yb = force_panel(series, target)
    fr = np.empty((T_out, S + FH, 3 * S, 3), np.uint8); fr[:, :S] = top
    for k in range(T_out):
        f = bg.copy(); x = int(round(fx0 + (fx1 - fx0) * k / FPS_OUT / DUR))
        cv2.line(f, (x, ya), (x, yb), (90, 90, 90), 1, cv2.LINE_AA); fr[k, S:] = f
    name = "%s_%s_%06d" % (act, g["law"], i)
    crf = 23
    while True:
        enc(fr, f"{OUT}/{name}.mp4", crf)
        if os.path.getsize(f"{OUT}/{name}.mp4") < 400_000 or crf >= 32: break
        crf += 1
    cv2.imwrite(f"{OUT}/{name}.jpg", fr[int(0.4 * T_out)][:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])
    row = dict(file=f"{name}.mp4", poster=f"{name}.jpg", action=act, sim_action=g["action"], **{"class": g["law"]}, object=g["object"],
               log10_E_over_rho=round(float(np.log10(g["labels"]["E_over_rho_specific"])), 3),
               log10_rho=round(float(np.log10(g["regression"]["density"])), 3), description=desc, force_source=src,
               sample_path=os.path.abspath(d), frames=T_out, fps=FPS_OUT, crf=crf, size_bytes=os.path.getsize(f"{OUT}/{name}.mp4"))
    if g.get("press_force") is not None: row["press_force_target"] = g["press_force"]
    clips.append(row); cache[name] = (rgb, boxes[1], series, target, act, g["law"])
    print(name, crf, row["size_bytes"], src, flush=True)
json.dump({"regime": "IdentiDeform-Touch",
           "format": "768x416: 3 views x 256x256 on top, contact-force panel 768x160 below; H.264 yuv420p, 30 fps, 120 frames, faststart, no audio",
           "clips": clips}, open(f"{OUT}/clips.json", "w"), indent=1)
for f in glob.glob(f"{OUT}/*_force.mp4"): os.remove(f)

# ---- supplement strip: 4 rows; 5 frames of view 1 at TS over the force curve, dots at TS ----
TS = [0.4, 1.2, 2.0, 2.8, 3.6]; HALF = 0.4
plt.rcParams.update({"font.size": 7, "font.family": "DejaVu Sans"})
# manual layout in inches so the 5 frames (each 0.8 s wide) tile exactly the force axis 0..4 s
FW, L, R = 7.0, 1.05, 0.08; AW = FW - L - R; IH = AW / 5; PH, G1, G2, TOP, BOT = 0.62, 0.04, 0.12, 0.04, 0.36
FH_ = TOP + BOT + 4 * (IH + G1 + PH) + 3 * G2
fig = plt.figure(figsize=(FW, FH_))
for r, name in enumerate(STRIP):
    rgb, (y0, y1, x0, x1), series, target, act, law = cache[name]
    ytop = FH_ - TOP - r * (IH + G1 + PH + G2)
    ai = fig.add_axes([L / FW, (ytop - IH) / FH_, AW / FW, IH / FH_])
    af = fig.add_axes([L / FW, (ytop - IH - G1 - PH) / FH_, AW / FW, PH / FH_])
    for t in TS:
        k = min(int(round(t * FPS_IN)), rgb.shape[1] - 1)
        im = cv2.resize(rgb[1, k].transpose(1, 2, 0)[y0:y1, x0:x1], (S, S), interpolation=cv2.INTER_CUBIC)
        ai.imshow(im, extent=(t - HALF, t + HALF, 0, 2 * HALF), interpolation="lanczos")
    ai.set_xlim(0, DUR); ai.set_ylim(0, 2 * HALF); ai.set_aspect("auto"); ai.axis("off")
    for lab, t, f in series:
        af.plot(t, f, lw=0.9, color=COL[lab], label=lab)
        af.plot(TS, np.interp(TS, t, f), "o", ms=2.6, color=COL[lab])
    if target is not None: af.axhline(target, color="#888888", lw=0.7, ls="--", label=f"target {target:g}")
    af.set_xlim(0, DUR); af.set_ylim(bottom=0); af.grid(alpha=0.25); af.tick_params(labelsize=6, length=2, pad=1)
    af.set_ylabel("force", fontsize=6, labelpad=1)
    af.legend(fontsize=5.5, frameon=False, loc="upper right", ncol=3, handlelength=1.2)
    if r == len(STRIP) - 1: af.set_xlabel("time [s]", fontsize=7, labelpad=1)
    else: plt.setp(af.get_xticklabels(), visible=False)
    fig.text(0.06 / FW, (ytop - (IH + G1 + PH) / 2) / FH_, f"{PRETTY[act]}\n({law})", fontsize=7.5, va="center", ha="left", weight="bold")
fig.savefig(f"{OUT}/supp_touch_strip.pdf"); fig.savefig(f"{OUT}/supp_touch_strip.png", dpi=200); plt.close(fig)
print("strip done")
