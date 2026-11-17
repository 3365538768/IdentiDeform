"""Website: observed video vs replays of the observed action, one camera view, five columns:
observed | PIXIE feed-forward (no prior) | our prior, 0 sims | prior + 4 sims | no prior, uninformed search (<=24 sims).
Each tile has a header band (column name, time cost, mask IoU vs observed averaged over the sample's 3 views).
Usage: make_refine.py OUT_DIR 5,210,... [VIEW]"""
import sys, os, json, glob, subprocess, numpy as np, cv2
sys.path.insert(0, 'scripts/v3m'); from common import crop_box
OUT = sys.argv[1]; IDX = [int(v) for v in sys.argv[2].split(',')]; VIEW = int(sys.argv[3]) if len(sys.argv) > 3 else 0
os.makedirs(OUT, exist_ok=True)
TAGS = ['pixie', 'prior', 'prior4', 'best3_24']
PRIOR_SECONDS = 6.0  # feed-forward prior (V-JEPA2 features + head), see the page text
PIXIE_SECONDS = json.load(open('docs/v3m/pixie_seconds.json'))  # per-object PIXIE wall time (render 200 views + CLIP + UNet)
S, BAND = 256, 66


def put(img, txt, xy, col=(255, 255, 255), sc=0.5, th=1):
    cv2.putText(img, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, (0, 0, 0), th + 2, cv2.LINE_AA)
    cv2.putText(img, txt, xy, cv2.FONT_HERSHEY_SIMPLEX, sc, col, th, cv2.LINE_AA)


def fmt(sec):
    if sec < 90: return f'~{sec:.0f} s'
    if sec < 5400: return f'~{sec / 60:.0f} min'
    return f'~{sec / 3600:.1f} h'


def iou(m, ref):  # mean over views and frames of the per-frame mask IoU (as c1_refine.compare)
    t = min(m.shape[1], ref.shape[1]); m, ref = m[:, :t], ref[:, :t]
    inter = (m & ref).sum((-1, -2)); uni = (m | ref).sum((-1, -2))
    return float(np.where(uni > 0, inter / np.maximum(uni, 1), 1.0).mean())


meta = {}
for i in IDX:
    idx = f'{i:06d}'
    z = np.load(glob.glob(f'data/v2_main/val/{idx}/*/sample_pack.npz')[0])
    obs_rgb_all = z['rgb'].transpose(0, 1, 3, 4, 2); obs_m_all = z['object_mask'][:, :, 0] > 127  # V,T,H,W
    cols = [dict(tag='observed', rgb=obs_rgb_all[VIEW], m=obs_m_all[VIEW], name='Observed (GT)', time='simulator ground truth')]
    for t in TAGS:
        n = np.load(f'data/v3m_site/{idx}/{t}.npz'); shape = tuple(n['shape'])
        m = np.unpackbits(n['mask'], -1)[..., :shape[-1]].astype(bool)
        p = json.load(open(f'data/v3m_site/{idx}/{t}.json')); c = p.get('cost', {})
        if t == 'pixie':
            sec = PIXIE_SECONDS[p['object']]; name = 'PIXIE, no sims'; time = f'{fmt(sec)} feed-forward'
        elif t == 'prior':
            sec = PRIOR_SECONDS; name = 'Our prior, 0 sims'; time = f'{fmt(sec)} feed-forward'
        elif t == 'prior4':
            sec = PRIOR_SECONDS + c['sim_seconds']; name = f"Prior + {c['sims']} sims"; time = fmt(sec)
        else:
            sec = c['sim_seconds']; name = f"No prior, {c['sims']} sims"; time = fmt(sec)
        cols.append(dict(tag=t, rgb=n['rgb'][VIEW], m=m[VIEW], m_all=m, name=name, time=time, seconds=sec, params=p,
                         iou3=iou(m, obs_m_all), iou_view=iou(m[VIEW:VIEW + 1], obs_m_all[VIEW:VIEW + 1]),
                         state_iou3=c.get('state_iou3')))
    T = min(c['rgb'].shape[0] for c in cols)
    union = np.zeros_like(cols[0]['m'][0])
    for c in cols:  # crop to observed + our columns; a PIXIE replay that spreads far is clipped by the tile
        if c['tag'] != 'pixie': union |= c['m'][:T].any(0)
    y0, y1, x0, x1 = crop_box(union[None])
    sep = np.zeros((T, S + BAND, 6, 3), np.uint8); parts = []
    for k, c in enumerate(cols):
        tile = np.zeros((T, S + BAND, S, 3), np.uint8)
        tile[:, BAND:] = np.stack([cv2.resize(f[y0:y1, x0:x1], (S, S), interpolation=cv2.INTER_CUBIC) for f in c['rgb'][:T]])
        band = np.full((BAND, S, 3), 18, np.uint8)
        put(band, c['name'], (8, 20), (255, 255, 255), 0.52, 1)
        put(band, c['time'], (8, 40), (150, 200, 255), 0.45, 1)
        if 'iou3' in c:
            put(band, f"mask IoU {c['iou3']:.3f} (3 views)", (8, 59), (140, 225, 160), 0.42, 1)
        tile[:, :BAND] = band
        parts += ([sep] if k else []) + [tile]
    frames = np.concatenate(parts, 2)
    H, W = frames.shape[1:3]
    path = f'{OUT}/refine_{idx}.mp4'
    subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', '32', '-i', '-',
                    '-c:v', 'libx264', '-preset', 'slow', '-crf', '21', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', path],
                   input=np.ascontiguousarray(frames).tobytes(), check=True)
    cv2.imwrite(path[:-4] + '.jpg', cv2.cvtColor(frames[min(T - 1, 64)], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
    meta[idx] = {'width': W, 'height': H, 'view': VIEW, 'columns': [
        {k: c[k] for k in ('tag', 'name', 'time', 'seconds', 'iou3', 'iou_view', 'state_iou3', 'params') if k in c} for c in cols]}
    print(path, os.path.getsize(path) // 1024, 'KB', [(c['tag'], round(c.get('iou3', 1), 3), c['time']) for c in cols], flush=True)
json.dump(meta, open(f'{OUT}/refine_meta.json', 'w'), indent=1)
