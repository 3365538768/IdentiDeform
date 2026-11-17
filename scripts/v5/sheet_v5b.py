import cv2, numpy as np, json
m = json.load(open('data/v5_review4/v5b/review_val.json')); rows = []
for k, r in sorted(m.items()):
    cap = cv2.VideoCapture(f'data/v5_review4/v5b/{k}.mp4'); fr = []
    while True:
        ok, x = cap.read()
        if not ok: break
        fr.append(x[:, :200])
    x = np.concatenate([fr[t] for t in np.linspace(0, len(fr) - 1, 8).astype(int)], 1).copy()
    cv2.putText(x, f"{k[-3:]} {r['action'][:6]} {r['law'][:5]}", (4, 195), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 255), 1)
    rows.append(cv2.resize(x, (0, 0), fx=0.4, fy=0.4))
cv2.imwrite('data/v5_review4/sheet_v5b.jpg', np.concatenate(rows, 0))
