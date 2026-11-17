"""Website: 3DGS view of the V3e squeeze (GT parameters) at the depth chosen by the class-mean twin vs by our twin.
Plates are not rendered by the 3DGS pipeline; the phase is printed instead. Input: data/v3m_site/grasp_gs/<idx>_<d>.npz."""
import json, os, sys, subprocess, numpy as np, cv2
OUT = sys.argv[1]; IDX = [int(v) for v in sys.argv[2].split(',')]
os.makedirs(OUT, exist_ok=True)
J = json.load(open('docs/v3e/GRIPPER_TASK.json')); res = {r['index']: r for r in J['samples']}; tau = J['protocol']['tau']
GREEN, RED = (62, 207, 142), (255, 93, 93); S = 400
def put(img, txt, xy, col=(255, 255, 255), sc=0.6, th=1):
    cv2.putText(img, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)
for i in IDX:
    r = res[i]; panels = []
    for m, lab in (('lawmean_pred', 'class-mean twin'), ('refined_top2', 'our twin (prior + refinement)')):
        d = r['methods'][m]['chosen_d']
        panels.append(dict(lab=lab, d=d, ok=r['methods'][m]['success'], crush=r['gt_dent'][f'{d:.2f}'],
                           z=np.load(f'data/v3m_site/grasp_gs/{i:06d}_{d:.3f}.npz')))
    views = sorted(panels[0]['z'].files)
    # view where the squeeze is most visible: largest change between t=0 and the end of the hold (1.5 s)
    def change(v):
        return sum(np.abs(p['z'][v][48].astype(float) - p['z'][v][0].astype(float)).mean() for p in panels)
    v = max(views, key=change)
    fr = [p['z'][v] for p in panels]; T = min(len(f) for f in fr)
    u = np.concatenate([f[:T:4].max(-1) > 40 for f in fr]).mean(0) > 0.05  # ignore faint floaters
    ys, xs = np.where(u); ylo, yhi = np.percentile(ys, [0.5, 99.5]); xlo, xhi = np.percentile(xs, [0.5, 99.5])
    cy, cx = (ylo + yhi) / 2, (xlo + xhi) / 2
    half = int(0.78 * max(yhi - ylo, xhi - xlo)) + 6
    H, W = u.shape; y0, x0 = int(max(0, cy - half)), int(max(0, cx - half)); y1, x1 = int(min(H, cy + half)), int(min(W, cx + half))
    out = []
    for t in list(range(T)) + [T - 1] * 40:
        tiles = []
        for p, f in zip(panels, fr):
            im = cv2.resize(np.ascontiguousarray(f[t, y0:y1, x0:x1]), (S, S), interpolation=cv2.INTER_CUBIC if y1 - y0 < S else cv2.INTER_AREA)
            im = np.concatenate([np.zeros((84, S, 3), np.uint8), im], 0)
            sec = t / 32.0; phase = 'closing' if sec < 1.0 else 'holding' if sec < 1.5 else 'released'
            put(im, p['lab'], (10, 26), sc=0.62, th=2); put(im, f"chosen depth d = {p['d']:.2f}", (10, 50), (200, 210, 220), 0.55)
            put(im, f't = {sec:.1f} s ({phase})', (S - 190, S + 84 - 12), (160, 170, 180), 0.5)
            if sec >= 3.0:
                put(im, ('INTACT' if p['ok'] else 'CRUSHED') + f": crush {max(0.0, 100 * p['crush']):.1f}% (limit {100 * tau:.0f}%)",
                    (10, 76), GREEN if p['ok'] else RED, 0.55, 2)
            tiles.append(im)
        out.append(np.concatenate([tiles[0], np.zeros((S + 84, 8, 3), np.uint8), tiles[1]], 1))
    fr_ = np.stack(out); Hh, Ww = fr_.shape[1:3]
    path = f'{OUT}/grasp_gs_{i:06d}.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{Ww}x{Hh}', '-r', '32', '-i', '-',
                    '-c:v', 'libx264', '-preset', 'slow', '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', path],
                   input=np.ascontiguousarray(fr_).tobytes(), check=True)
    cv2.imwrite(path[:-4] + '.jpg', cv2.cvtColor(fr_[T - 1], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(path, v, Ww, Hh, os.path.getsize(path) // 1024, 'KB', flush=True)
