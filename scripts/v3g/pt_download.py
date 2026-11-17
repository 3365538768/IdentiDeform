"""v3g: range-download ONE PhysTwin processed case from the official HF data.zip (no unpickling here)."""
import hashlib, json, os, sys, time, zipfile, pickletools, io
from pathlib import Path
ROOT = Path('/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test')
sys.path.insert(0, str(ROOT / 'scripts'))
from fetch_sample_subset import HTTPRange
from acquire_external_pilots import configure_proxy

CASE = sys.argv[1] if len(sys.argv) > 1 else 'single_push_sloth'
LIST_ONLY = '--list' in sys.argv
src = json.loads((ROOT / 'docs/DATA_SOURCE_PILOTS.json').read_text())['sources']['phystwin']
url = src['source'] + '/resolve/' + src['revision'] + '/data.zip'
configure_proxy(ROOT)
out = ROOT / 'data/external_datasets/v3g_phystwin'
out.mkdir(parents=True, exist_ok=True)
stream = HTTPRange(url, src['archive_bytes'])
WANT = {'final_data.pkl', 'calibrate.pkl', 'metadata.json', 'split.json', 'gt_track_3d.pkl'}
SAFE_GLOBALS = {('numpy.core.multiarray', '_reconstruct'), ('numpy', 'ndarray'), ('numpy', 'dtype'),
                ('numpy.core.multiarray', 'scalar'), ('numpy._core.multiarray', '_reconstruct'),
                ('numpy._core.multiarray', 'scalar'), ('builtins', 'slice')}

def pickle_globals(data):
    found = set(); last = []
    for op, arg, _ in pickletools.genops(io.BytesIO(data)):
        if op.name in ('SHORT_BINUNICODE', 'BINUNICODE', 'UNICODE'):
            last = (last + [arg])[-2:]
        if op.name == 'GLOBAL':
            found.add(tuple(arg.split(' ', 1)))
        elif op.name == 'STACK_GLOBAL':
            found.add(tuple(last))
    return found

rows = []
with zipfile.ZipFile(stream) as z:
    prefix = f'data/different_types/{CASE}/'
    members = [i for i in z.infolist() if i.filename.startswith(prefix)]
    if LIST_ONLY:
        for i in members:
            if '/color/' not in i.filename and '/depth/' not in i.filename and '/mask/' not in i.filename:
                print(i.filename, i.file_size)
        print('n_members', len(members), 'total', sum(i.file_size for i in members))
        sys.exit(0)
    for i in members:
        rel = i.filename[len(prefix):]
        if rel not in WANT:
            continue
        if i.file_size > 512 * 1024**2:
            raise ValueError('too large ' + i.filename)
        data = z.read(i)
        target = (out / CASE / rel).resolve()
        assert target.is_relative_to(out)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        row = dict(member=i.filename, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(), zip_crc_verified=True)
        if rel.endswith('.pkl'):
            g = pickle_globals(data)
            row['pickle_globals'] = sorted('.'.join(x) for x in g)
            row['pickle_globals_safe'] = g <= SAFE_GLOBALS
        rows.append(row); print(json.dumps(row), flush=True)
(out / CASE / 'v3g_download_manifest.json').write_text(json.dumps(dict(
    source=src['source'], revision=src['revision'], case=CASE, files=rows,
    range_downloaded_bytes=stream.downloaded, time_unix=time.time()), indent=1))
print('done', stream.downloaded)
