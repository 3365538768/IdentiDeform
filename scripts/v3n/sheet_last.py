import cv2, numpy as np, glob, json, sys
m = json.load(open('data/v3n_review2/review_val.json')); acts = sys.argv[1].split(','); out = sys.argv[2]
rows = []
for k, r in sorted(m.items()):
    if r['action'] not in acts: continue
    cap = cv2.VideoCapture(f'data/v3n_review2/{k}.mp4'); fr = []
    while True:
        ok, x = cap.read()
        if not ok: break
        fr.append(x[-200:, :200])
    x = np.concatenate([fr[t] for t in np.linspace(0, len(fr) - 1, 8).astype(int)], 1).copy()
    cv2.putText(x, f"{k[-3:]} {r['action']} {r['law']} {r.get('log10_E_over_rho_v3')}", (4, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 255), 1)
    rows.append(cv2.resize(x, (0, 0), fx=0.4, fy=0.4))
cv2.imwrite(out, np.concatenate(rows, 0))
