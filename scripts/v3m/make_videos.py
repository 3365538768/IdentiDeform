"""Website media from V2 sample packs (RGB) -> H.264 MP4 + JPG posters."""
import sys, os, json, subprocess, numpy as np, cv2
sys.path.insert(0, 'scripts/v3m'); from common import load, cropped_view, crop_box
OUT = sys.argv[1]; os.makedirs(OUT + '/actions', exist_ok=True); os.makedirs(OUT + '/pairs', exist_ok=True)

def encode(frames, path, fps=32, crf=21):
    T, H, W, _ = frames.shape
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-',
           '-c:v', 'libx264', '-preset', 'slow', '-crf', str(crf), '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', path]
    subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(), check=True)
    cv2.imwrite(path[:-4] + '.jpg', cv2.cvtColor(frames[min(len(frames) - 1, 40)], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 82])
    print(path, os.path.getsize(path) // 1024, 'KB', flush=True)

def views(i, vs=(0, 1, 2), size=256, split="val"):
    z, g = load(i, split); rgb, m = z['rgb'], z['object_mask']
    return [cropped_view(rgb[v], m[v], out=size)[0] for v in vs], g

meta = {}
# v2 selection (2026-11-15): no tearing/fragmenting clips (scripts/v3m/select_media.py: largest-component share >= 99.8%)
ACT = {'press': ('train', 1495), 'drop': ('train', 1245), 'shear': ('train', 692), 'stretch': ('train', 1287, 100), 'bend': ('val', 12), 'tap30hold': ('train', 33)}
for a, (sp, i, *lim) in ACT.items():
    vv, g = views(i, split=sp)
    if lim:  # stretch: keep the part before the grip is released (the object then drops)
        vv = [v[:lim[0]] for v in vv]
    sep = np.zeros((vv[0].shape[0], 256, 4, 3), np.uint8)
    encode(np.concatenate([vv[0], sep, vv[1], sep, vv[2]], 2), f'{OUT}/actions/{a}.mp4')
    meta[a] = {'split': sp, 'index': i, 'law': g['law'], 'object': g['object']}
PAIRS = {'drop_bottle': (75, 201, (0, 1), 'val'), 'hold_basket': (1959, 1941, (0, 1), 'train')}
for name, (soft, stiff, vs, sp) in PAIRS.items():
    out = []
    for i in (soft, stiff):
        vv, g = views(i, vs, 256, sp)
        out.append(np.concatenate([vv[0], np.zeros((128, 256, 4, 3), np.uint8), vv[1]], 2))
        r = g['regression']; s = r['density'] / 1000
        meta[name + ('_soft' if i == soft else '_stiff')] = {'index': i, 'law': g['law'], 'action': g['action'],
            'log10_E_over_rho': round(float(np.log10(r['E'] / s)), 2), 'object': g['object']}
    sep = np.zeros((128, 256, 12, 3), np.uint8)
    encode(np.concatenate([out[0], sep, out[1]], 2), f'{OUT}/pairs/{name}.mp4')
json.dump(meta, open(f'{OUT}/media_meta.json', 'w'), indent=1)
