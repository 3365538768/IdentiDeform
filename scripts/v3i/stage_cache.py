"""Stage the train+val part of the V2-main V-JEPA2 feature cache into /dev/shm on a800-4 (read-only copy;
test files are not copied, so the ablations cannot touch the test split). Delete with --remove."""
import argparse, json, shutil, os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT = Path('/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test')
p = argparse.ArgumentParser(); p.add_argument('--src', default=str(ROOT / 'data/prepared/v2main_vjl'))
p.add_argument('--dst', default='/dev/shm/rsi-v3i/v2main_vjl'); p.add_argument('--remove', action='store_true'); a = p.parse_args()
if a.remove:
    shutil.rmtree(a.dst, ignore_errors=True); print('removed', a.dst); raise SystemExit
src, dst = Path(a.src), Path(a.dst); dst.mkdir(parents=True, exist_ok=True)
index = json.loads((src / 'index.json').read_text())
rows = [json.loads(l) for l in open(ROOT / 'data/manifests/v2main.jsonl')]
keep = {r['sample_id'] for r in rows if r['split'] in ('train', 'val')}
files = [index['files'][s] for s in keep]
def copy(name):
    out = dst / name
    if out.exists() and out.stat().st_size == (src / name).stat().st_size:
        return 0
    tmp = dst / (name + '.part'); shutil.copyfile(src / name, tmp); os.replace(tmp, out); return 1
with ThreadPoolExecutor(24) as ex:
    n = sum(ex.map(copy, files))
staged = dict(index, files={s: index['files'][s] for s in keep})
(dst / 'index.json').write_text(json.dumps(staged, indent=2))
print(json.dumps({'copied': n, 'files': len(files), 'dst': str(dst)}))
