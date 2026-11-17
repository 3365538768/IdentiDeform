"""Object review page assets: first frame of 3 views for every object (from one V2-main sample per object)."""
import json, glob, os, collections, numpy as np, cv2, sys
sys.path.insert(0, 'scripts/v3m'); from common import crop_box
out = 'docs/review_v3/objects'; os.makedirs(out, exist_ok=True)
rows = [json.loads(l) for l in open('data/manifests/deformfield_v1.jsonl') if l.strip()]
split_of = {r['object_id']: r['split'] for r in rows}
first, count = {}, collections.Counter()
for sp in ('train', 'val', 'test'):
    for gtp in sorted(glob.glob(f'data/v2_main/{sp}/*/gt.json')):
        g = json.load(open(gtp)); count[g['object']] += 1
        if g['object'] not in first and glob.glob(os.path.dirname(gtp) + '/*/sample_pack.npz'):
            first[g['object']] = os.path.dirname(gtp)
meta = []
for obj in sorted(split_of):
    d = first.get(obj)
    if d is None:
        meta.append({'object': obj, 'split': split_of[obj], 'samples': 0, 'img': None}); continue
    z = np.load(glob.glob(d + '/*/sample_pack.npz')[0]); rgb, m = z['rgb'], z['object_mask'][:, :1, 0] > 127
    tiles = []
    for v in range(3):
        y0, y1, x0, x1 = crop_box(m[v])
        tiles.append(cv2.resize(rgb[v, 0].transpose(1, 2, 0)[y0:y1, x0:x1], (180, 180), interpolation=cv2.INTER_AREA))
    img = np.concatenate([tiles[0], np.zeros((180, 3, 3), np.uint8), tiles[1], np.zeros((180, 3, 3), np.uint8), tiles[2]], 1)
    cv2.imwrite(f'{out}/{obj}.jpg', cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
    meta.append({'object': obj, 'split': split_of[obj], 'samples': count[obj], 'img': f'objects/{obj}.jpg'})
json.dump(meta, open('docs/review_v3/objects.json', 'w'), indent=1)
print(len(meta), sum(1 for x in meta if x['img']))
