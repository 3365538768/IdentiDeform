"""v3l: range-download the RGB colour frames + object masks of ONE PhysTwin case (train frames only) from the official HF
data.zip into a NEW directory data/external_datasets/v3l_phystwin_rgb/<case>.  Nothing is executed or unpickled; only
PNG/JPG/JSON bytes are written (paths checked to stay inside the target dir).  Reuses the v3g HTTPRange helper.
  .venv/bin/python -I scripts/v3l/pt_rgb_download.py <case> [--list] [--max-frame 47]
"""
import hashlib, json, sys, time, zipfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from fetch_sample_subset import HTTPRange  # noqa: E402
from acquire_external_pilots import configure_proxy  # noqa: E402

CASE = sys.argv[1]
LIST = '--list' in sys.argv
MAXF = int(sys.argv[sys.argv.index('--max-frame') + 1]) if '--max-frame' in sys.argv else 47
src = json.loads((ROOT / 'docs/DATA_SOURCE_PILOTS.json').read_text())['sources']['phystwin']
url = src['source'] + '/resolve/' + src['revision'] + '/data.zip'
configure_proxy(ROOT)
out = ROOT / 'data/external_datasets/v3l_phystwin_rgb' / CASE
stream = HTTPRange(url, src['archive_bytes'])
prefix = f'data/different_types/{CASE}/'


def frame_of(rel):
    stem = Path(rel).stem
    return int(stem) if stem.isdigit() else None


rows = []
with zipfile.ZipFile(stream) as z:
    members = [i for i in z.infolist() if i.filename.startswith(prefix) and not i.is_dir()]
    if LIST:
        import collections
        c = collections.Counter(); b = collections.Counter()
        for i in members:
            parts = i.filename[len(prefix):].split('/')
            key = '/'.join(parts[:-1]) if len(parts) > 1 else parts[0]
            c[key] += 1; b[key] += i.file_size
        for k in sorted(c):
            print(k, c[k], b[k])
        sys.exit(0)
    out.mkdir(parents=True, exist_ok=True)
    for i in members:
        rel = i.filename[len(prefix):]
        parts = rel.split('/')
        if parts[0] not in ('color', 'mask') or not rel.lower().endswith(('.png', '.jpg', '.jpeg', '.json')):
            continue
        f = frame_of(rel)
        if f is not None and f >= MAXF:
            continue
        if i.file_size > 64 * 1024**2:
            raise ValueError('too large ' + i.filename)
        target = (out / rel).resolve()
        assert target.is_relative_to(out.resolve()), target
        if target.exists() and target.stat().st_size == i.file_size:  # resume: complete file from an interrupted run
            data = target.read_bytes()
            rows.append(dict(member=i.filename, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), resumed=True)); continue
        data = z.read(i)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        rows.append(dict(member=i.filename, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
(out / 'v3l_download_manifest.json').write_text(json.dumps(dict(source=src['source'], revision=src['revision'], case=CASE,
    max_frame_exclusive=MAXF, files=rows, range_downloaded_bytes=stream.downloaded, time_unix=time.time()), indent=1))
print('done', len(rows), stream.downloaded)
