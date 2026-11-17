import sys, json, numpy as np, cv2
sys.path.insert(0, "scripts/v3m"); from common import load, cropped_view
C = json.load(open("data/v3m_site/cand_ps.json"))
for a, L in C.items():
    rows = []
    for sp, i in L:
        z, g = load(i, sp); fr, _ = cropped_view(z["rgb"][0], z["object_mask"][0], out=96)
        row = np.concatenate([fr[t] for t in np.linspace(0, 127, 10).astype(int)], 1).copy()
        cv2.putText(row, f"{sp}{i} {g['law']}", (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 80, 80), 1); rows.append(row)
    cv2.imwrite(f"data/v3m_site/look/c_{a}.jpg", cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
