"""V3f PIXIE: contact sheet of the rendered inputs (first 6 of 200 views) per object with PIXIE's object-level output."""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw
src, out = Path(sys.argv[1]), Path(sys.argv[2])
rows = []
for j in sorted(src.glob('*.json')):
    r = json.loads(j.read_text()); v = r['variants']['filled']
    im = Image.open(src / f"{r['object_id']}_views.png").convert('RGB')
    canvas = Image.new('RGB', (im.width + 330, im.height), 'white'); canvas.paste(im, (0, 0))
    d = ImageDraw.Draw(canvas)
    d.text((im.width + 6, 10), r['object_id'][:40], fill='black')
    d.text((im.width + 6, 40), f"PIXIE: {v['majority_class_excl_background']}", fill='black')
    d.text((im.width + 6, 60), f"E={v['median_E_Pa']:.3g} Pa  nu={v['median_nu']:.3f}  rho={v['median_rho']:.0f}", fill='black')
    rows.append(canvas)
sheet = Image.new('RGB', (rows[0].width, sum(r.height for r in rows)), 'white')
y = 0
for r in rows:
    sheet.paste(r, (0, y)); y += r.height
sheet.save(out); print('saved', out, sheet.size)
