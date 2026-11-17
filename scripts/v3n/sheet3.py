import cv2, numpy as np, sys
idx = [int(x) for x in sys.argv[1].split(',')]; out = sys.argv[2]
rows = []
for i in idx:
    cap = cv2.VideoCapture(f'data/v3n_review/val_{i:06d}.mp4'); fr = []
    while True:
        ok, x = cap.read()
        if not ok: break
        fr.append(x)
    if not fr: continue
    x = np.concatenate([fr[t][:, :200] for t in np.linspace(0, len(fr) - 1, 8).astype(int)], 1).copy()
    cv2.putText(x, str(i), (170, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 255), 2)
    rows.append(cv2.resize(x, (0, 0), fx=0.42, fy=0.42)); rows.append(np.full((6, rows[-1].shape[1], 3), 255, np.uint8))
cv2.imwrite(out, np.concatenate(rows, 0))
