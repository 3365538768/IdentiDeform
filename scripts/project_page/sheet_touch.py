"""Contact sheets for clip selection: one row per candidate (view 0 + view 1 at 6 times). Usage: sheet_touch.py <score.json> <action> <out.jpg> [max]"""
import sys, json, glob, numpy as np, cv2
sys.path.insert(0, "scripts/v3m"); from common import crop_box
rows = json.load(open(sys.argv[1])); act = sys.argv[2]; mx = int(sys.argv[4]) if len(sys.argv) > 4 else 30
rows = [r for r in rows if r["action"] == act and r["frag"] < 0.01 and r["a0"] > 500][:mx]
TS = [0, 25, 50, 75, 100, 127]; out = []
for r in rows:
    Z = np.load(glob.glob(r["path"] + "/*/sample_pack.npz")[0]); z = Z["rgb"]; m = Z["object_mask"][:, :, 0] > 127
    tiles = []
    for v in (0, 1):
        y0, y1, x0, x1 = crop_box(m[v]); tiles += [cv2.resize(z[v, t].transpose(1, 2, 0)[y0:y1, x0:x1], (112, 112)) for t in TS]
    row = np.concatenate(tiles, 1)[:, :, ::-1].copy()
    lab = np.zeros((112, 150, 3), np.uint8)
    for k, s in enumerate([str(r["idx"]), r["law"], r["obj"][2:16], r["obj"][16:30], "le %s" % r["le"], "F %s" % r["press_force"]]):
        cv2.putText(lab, s, (3, 16 + 17 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    out.append(np.concatenate([lab, row], 1))
cv2.imwrite(sys.argv[3], np.concatenate(out, 0), [cv2.IMWRITE_JPEG_QUALITY, 80]); print(act, len(rows))
