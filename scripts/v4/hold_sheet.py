import glob, numpy as np, cv2, sys
sys.path.insert(0, 'scripts/v3m'); from common import crop_box
rows = []
for i in (6, 8, 9, 11, 13, 27, 30, 24):
    for root in ('data/v3n_probe/H1', 'data/v3n_probe/H4'):
        p = glob.glob(f'{root}/val/{i:06d}/*/sample_pack.npz')
        if not p: continue
        z = np.load(p[0]); rgb, m = z['rgb'][0], z['object_mask'][0, :, 0] > 127
        y0, y1, x0, x1 = crop_box(m)
        tiles = [cv2.resize(rgb[t].transpose(1, 2, 0)[y0:y1, x0:x1], (110, 110)) for t in np.linspace(0, len(rgb) - 1, 8).astype(int)]
        r = np.concatenate(tiles, 1).copy(); cv2.putText(r, f'{i} {root[-2:]}', (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 80, 80), 1); rows.append(r)
cv2.imwrite('data/v3n_probe/hold_sheet.jpg', cv2.cvtColor(np.concatenate(rows, 0), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
