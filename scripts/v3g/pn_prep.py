"""Prepare per-run PAC-NeRF data dirs (copies; official train.py writes m_*.png and data.pt into data_dir).
views: all keeps 11 cameras; otherwise comma list of camera ids, renumbered 0..k-1 in that order."""
import json, os, re, shutil, sys
src_root, dst_root, scene, views = sys.argv[1:5]
src = os.path.join(src_root, scene)
info = json.load(open(os.path.join(src, "all_data.json")))
keep = None if views == "all" else [int(v) for v in views.split(",")]
dst = dst_root
os.makedirs(os.path.join(dst, "data"), exist_ok=True)
out = []
pat = re.compile(r"^\./data/r_(\d+)_(-?\d+)\.png$")
for e in info:
    m = pat.match(e["file_path"]); assert m, e["file_path"]
    cam, fr = int(m.group(1)), int(m.group(2))
    if keep is not None:
        if cam not in keep: continue
        new = keep.index(cam)
    else:
        new = cam
    e2 = dict(e); e2["file_path"] = f"./data/r_{new}_{fr}.png"
    shutil.copyfile(os.path.join(src, "data", f"r_{cam}_{fr}.png"), os.path.join(dst, "data", f"r_{new}_{fr}.png"))
    out.append(e2)
json.dump(out, open(os.path.join(dst, "all_data.json"), "w"), indent=1)
print(scene, views, len(out), "entries ->", dst)
