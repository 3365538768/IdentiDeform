"""Project-page clips for the V5g grasp-outcome study (benchT val): GT execution of 4 candidates per sample, 4096-particle subset.

Sample rule (stated in grasp_pred.json): GT outcomes over the 8 candidates contain all three classes; video+touch twin (prior_vt)
gets the most candidates right; ties -> largest margin over the class-mean twin (lawmean_pred); then lowest index; distinct objects.
Candidate rule: per GT class (drop, success, damage) the first candidate (end before centre, shallow before deep) on which the
class-mean twin is wrong, else the first of that class; 4th = remaining candidate where the class-mean twin is wrong (fewest tiles
at its grip position first), else the first remaining. Tiles ordered end -> centre, shallow -> deep.
Render: view at 30 deg yaw from the jaw closing axis (jaws separate in the image), particles coloured by vertical displacement (0 -> lift height),
jaws (grasp_bc.json geometry + grasp_main timeline) as translucent grey boxes until release at 2.1 s.
Run: .venv/bin/python scripts/project_page/grasp_clips.py
"""
import json, shutil, subprocess, tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
VAL = ROOT / 'data/v5g/runs/val'; OUT = ROOT / 'docs/project_assets/grasp_pred'
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'; FONTB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
TILE, STRIP, FPS_OUT = 256, 28, 20
CANDS = ['end:0.03', 'end:0.10', 'end:0.25', 'end:0.40', 'centre:0.03', 'centre:0.10', 'centre:0.25', 'centre:0.40']
COL = {'success': (30, 150, 60), 'drop': (230, 120, 0), 'damage': (210, 30, 30)}
T_CLOSE, T_LIFT0, T_LIFT1, T_LOWER0, T_LOWER1, T_OPEN = 0.4, 0.6, 1.0, 1.6, 2.0, 2.1
YAW = np.deg2rad(30)


def lift_at(t, lift):
    if t < T_LIFT0:
        return 0.0
    vz = lift / (T_LIFT1 - T_LIFT0)
    if t < T_LIFT1:
        return vz * (t - T_LIFT0)
    if t < T_LOWER0:
        return lift
    if t < T_LOWER1:
        return lift - vz * (t - T_LOWER0)
    return 0.0


def phase(t):
    for t1, n in ((T_CLOSE, 'close'), (T_LIFT0, 'hold'), (T_LIFT1, 'lift'), (T_LOWER0, 'hold'), (T_LOWER1, 'lower'), (T_OPEN, 'hold')):
        if t < t1:
            return n
    return 'released'


def select_samples(P):
    rows = []
    for s in P['samples']:
        t = s['true']
        if len({t[c] for c in CANDS}) < 3:
            continue
        a = sum(s['pred']['prior_vt'][c] == t[c] for c in CANDS); b = sum(s['pred']['lawmean_pred'][c] == t[c] for c in CANDS)
        rows.append((-a, -(a - b), s['index'], s))
    rows.sort(key=lambda r: r[:3]); out, objs = [], set()
    for r in rows:
        if r[3]['object'] not in objs:
            out.append(r[3]); objs.add(r[3]['object'])
        if len(out) == 2:
            break
    return out


def select_cands(s):
    t, lm = s['true'], s['pred']['lawmean_pred']; pick = []
    for cls in ('drop', 'success', 'damage'):
        cs = [c for c in CANDS if t[c] == cls]
        w = [c for c in cs if lm[c] != cls]
        pick.append((w or cs)[0])
    rest = [c for c in CANDS if c not in pick]
    w = [c for c in rest if lm[c] != t[c]] or rest
    npos = lambda c: sum(p.split(':')[0] == c.split(':')[0] for p in pick)
    pick.append(sorted(w, key=lambda c: (npos(c), CANDS.index(c)))[0])
    return sorted(pick, key=CANDS.index)


def view(X, Lax, Fax):
    """(u, v, depth) for points X (N,3): yaw YAW from looking along the jaw axis F."""
    u = np.cos(YAW) * X[:, Lax] + np.sin(YAW) * X[:, Fax]
    d = -np.sin(YAW) * X[:, Lax] + np.cos(YAW) * X[:, Fax]
    return u, X[:, 2], d


def jaw_polys(bc, t):
    if t >= T_OPEN:
        return []
    out = []
    for jw in bc['jaws']:
        c = np.array(jw['c0']) + np.array(jw['vc']) * min(t, T_CLOSE); c[2] += lift_at(t, bc['lift']); h = np.array(jw['h'])
        corners = np.array([c + h * np.array(s) for s in np.stack(np.meshgrid([-1, 1], [-1, 1], [-1, 1]), -1).reshape(-1, 3)])
        out.append((corners, float(view(c[None], bc['L'], bc['F'])[2][0])))
    return out


def text_strip(w, lines, fonts):
    im = Image.new('RGB', (w, STRIP), (255, 255, 255)); dr = ImageDraw.Draw(im)
    for x, y, parts in lines:  # parts: [(text, colour, font)], centred at x
        widths = [dr.textlength(p[0], font=fonts[p[2]]) for p in parts]; xx = x - sum(widths) / 2
        for (tx, cl, f), wd in zip(parts, widths):
            dr.text((xx, y), tx, fill=cl, font=fonts[f]); xx += wd
    return np.asarray(im)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P = json.loads((ROOT / 'docs/v5g/PREDICTION.json').read_text())
    fonts = {'r': ImageFont.truetype(FONT, 11), 'b': ImageFont.truetype(FONTB, 11), 's': ImageFont.truetype(FONT, 10)}
    summary = {'source': 'docs/v5g/PREDICTION.json + data/v5g/runs/val/<sample>/gt (GT simulator), 4096-particle subsets',
               'twins': {'prior_vt': 'video+touch twin', 'lawmean_pred': 'class-mean twin'},
               'sample_rule': 'GT outcomes over the 8 candidates contain success, drop and damage; maximise video+touch twin '
                              'accuracy over the 8 candidates; ties by margin over the class-mean twin, then lowest index; distinct objects',
               'candidate_rule': 'one candidate per GT outcome (first where the class-mean twin is wrong, else first), plus one more where '
                                 'the class-mean twin is wrong (balancing grip positions)',
               'overall': {k: {m: P['predictors'][k][m] for m in ('macro_f1', 'accuracy')} for k in ('prior_vt', 'lawmean_pred')},
               'samples': {}}
    for s in select_samples(P):
        sid = f"{s['index']:06d}"; sd = VAL / sid
        gtc = json.loads((sd / 'gt/twin.json').read_text())['cache']; state = json.loads((sd / 'gt/state.json').read_text())
        picks = select_cands(s); data = []
        for c in picks:
            d = sd / 'cache' / gtc / c.replace(':', '_')
            z = np.load(d / 'subset.npz'); bc = json.loads((d / 'grasp_bc.json').read_text())
            data.append((c, z['subset'].astype(np.float64), z['subset_t'], bc))
        Lax, Fax = data[0][3]['L'], data[0][3]['F']; floor = min(b[3]['lo'][2] for b in data)
        # common square window over all tiles and frames (particles + jaws)
        U, V = [], []
        for c, S, T, bc in data:
            u, v, _ = view(S.reshape(-1, 3), Lax, Fax); U += [u.min(), u.max()]; V += [v.min(), v.max()]
            for t in T * 0.01:
                for corners, _ in jaw_polys(bc, t):
                    u, v, _ = view(corners, Lax, Fax); U += [u.min(), u.max()]; V += [v.min(), v.max()]
        cu, cv_ = (min(U) + max(U)) / 2, (min(V) + max(V)) / 2; half = max(max(U) - min(U), max(V) - min(V)) / 2 * 1.06
        zmax = max(S[:, :, 2].max() for _, S, _, _ in data)
        px = lambda u, v: (np.round((u - cu + half) / (2 * half) * TILE).astype(int), np.round((cv_ + half - v) / (2 * half) * TILE).astype(int))
        lut = cv2.applyColorMap(np.arange(256, dtype=np.uint8)[:, None], cv2.COLORMAP_VIRIDIS)[:, 0, ::-1]
        top = text_strip(TILE * 4, [(i * TILE + TILE / 2, y, parts) for i, (c, *_ ) in enumerate(data) for y, parts in (
            (1, [(f"{c.split(':')[0]} grip, depth {c.split(':')[1]}", (20, 20, 20), 'r')]),
            (14, [('GT: ', (20, 20, 20), 's'), (s['true'][c], COL[s['true'][c]], 'b')]))], fonts)

        def pred_parts(name, p, gt):
            return [(f'{name}: ', (60, 60, 60), 's'), (p, COL[p], 'b'), (' ✓' if p == gt else ' ✗', (20, 20, 20), 's')]
        bot = text_strip(TILE * 4, [(i * TILE + TILE / 2, y, pred_parts(nm, s['pred'][arm][c], s['true'][c]))
                                    for i, (c, *_ ) in enumerate(data)
                                    for y, nm, arm in ((1, 'twin predicts (video+touch)', 'prior_vt'), (14, 'class-mean twin', 'lawmean_pred'))], fonts)
        nF = min(len(d[2]) for d in data); frames = []
        for k in range(nF):
            tiles = []
            for c, S, T, bc in data:
                t = T[k] * 0.01
                img = np.full((TILE, TILE, 3), 255, np.uint8)
                fy = px(np.array([0.0]), np.array([floor]))[1][0]
                cv2.line(img, (0, fy + 2), (TILE, fy + 2), (170, 170, 170), 1)
                jaws = sorted(jaw_polys(bc, t), key=lambda j: j[1])
                d0 = view(S[k], Lax, Fax)[2].mean()

                def draw_jaw(corners, alpha):
                    u, v, _ = view(corners, Lax, Fax); x, y = px(u, v)
                    hull = cv2.convexHull(np.stack([x, y], 1).astype(np.int32)); ov = img.copy()
                    cv2.fillConvexPoly(ov, hull, (150, 150, 150)); img[:] = cv2.addWeighted(ov, alpha, img, 1 - alpha, 0)
                    cv2.polylines(img, [hull], True, (90, 90, 90), 1, cv2.LINE_AA)
                for corners, dj in jaws:  # depth larger = farther (view direction +d)
                    if dj > d0:
                        draw_jaw(corners, 0.55)
                u, v, dep = view(S[k], Lax, Fax); x, y = px(u, v)
                hcol = lut[np.clip((S[k][:, 2] - S[0][:, 2]) / bc['lift'] * 255, 0, 255).astype(int)]
                for i in np.argsort(-dep):
                    cv2.circle(img, (int(x[i]), int(y[i])), 2, tuple(int(q) for q in hcol[i]), -1, cv2.LINE_AA)
                for corners, dj in jaws:
                    if dj <= d0:
                        draw_jaw(corners, 0.35)
                tiles.append(img)
            frame = np.vstack([top, np.hstack(tiles), bot])
            lab = Image.fromarray(frame); ImageDraw.Draw(lab).text((5, STRIP + 4), f't = {t:.2f} s  {phase(t)}', fill=(80, 80, 80), font=fonts['s'])
            ImageDraw.Draw(lab).text((5, STRIP + TILE - 14), 'colour: rise 0 → lift height', fill=(80, 80, 80), font=fonts['s'])
            frames.append(np.asarray(lab))
        frames += [frames[-1]] * 10  # short hold on the final (relaxed) state
        tmp = Path(tempfile.mkdtemp(prefix=f'grasp_{sid}_', dir=OUT))
        for i, fr in enumerate(frames):
            Image.fromarray(fr).save(tmp / f'{i:04d}.png')
        mp4 = OUT / f'{sid}.mp4'
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-framerate', str(FPS_OUT), '-i', str(tmp / '%04d.png'), '-an', '-c:v', 'libx264',
                        '-preset', 'slow', '-crf', '23', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(mp4)], check=True)
        Image.fromarray(frames[int(0.4 * len(frames))]).save(OUT / f'{sid}.jpg', quality=90)
        shutil.rmtree(tmp)
        acc = lambda arm: sum(s['pred'][arm][c] == s['true'][c] for c in CANDS)
        summary['samples'][sid] = {
            'object': s['object'], 'law': s['law'], 'vis_law': s.get('vis_law'), 'gt_cache': gtc, 'shown': picks,
            'acc_8': {'prior_vt': acc('prior_vt'), 'lawmean_pred': acc('lawmean_pred')},
            'candidates': {c: {'gt': s['true'][c], 'video_touch_twin': s['pred']['prior_vt'][c], 'class_mean_twin': s['pred']['lawmean_pred'][c],
                               'gt_left_frac': state['runs'].get(c, {}).get('metrics', {}).get('left'),
                               'gt_crush': state['runs'].get(c, {}).get('metrics', {}).get('crush')} for c in CANDS},
            'mp4_bytes': mp4.stat().st_size, 'frames': len(frames), 'fps': FPS_OUT}
        print(sid, s['object'], picks, summary['samples'][sid]['acc_8'], mp4.stat().st_size, flush=True)
    (OUT / 'grasp_pred.json').write_text(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
