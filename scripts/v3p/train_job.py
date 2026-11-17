"""Run fieldbridge.train for a config exactly like scripts/r_queue.py does (resume from last.pt after a crash/pause).

--preload (V3p, 2026-11-15, IO fix): before training, every cached train+val feature file of the config's prepared
index is read ONCE into this process's RAM (16 reader threads) and torch.load is wrapped so that the dataset's
reads of those exact paths return the in-RAM sample (the dataset copies valid/field_mask before any in-place edit,
so the shared objects are never modified; forked DataLoader workers share them copy-on-write). Training math,
sampling order, checkpoints and outputs are unchanged; only the per-epoch re-reading from the shared GPFS is removed.
  .venv/bin/python scripts/v3p/train_job.py configs/v3p/v3main_C_seed29.json [--preload]
"""
import json, subprocess, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
args = [a for a in sys.argv[1:] if not a.startswith('--')]; preload = '--preload' in sys.argv
config = args[0]; cfg = json.loads((ROOT / config).read_text())
out = ROOT / cfg['train']['output']
status = out / 'status.json'
if status.exists() and json.loads(status.read_text()).get('status') in {'training_finished', 'early_stopped'}:
    sys.exit(0)
control = out / 'control.json'
if control.exists() and json.loads(control.read_text()).get('desired') == 'paused':
    control.unlink()  # a V3p IO-fix pause is lifted by the relaunch
resume = out / 'last.pt'
if resume.exists():
    cfg['train']['resume'] = str(resume)
    config = f'docs/v3p/logs/{Path(config).stem}.resume.json'; (ROOT / config).write_text(json.dumps(cfg, indent=2))
if not preload:
    sys.exit(subprocess.call([sys.executable, '-m', 'fieldbridge.train', '--config', config], cwd=ROOT))
import os
os.chdir(ROOT)
from concurrent.futures import ThreadPoolExecutor
import torch
d = Path(cfg['data']['prepared_dir']); index = json.loads((d / 'index.json').read_text())['files']
rows = [json.loads(l) for l in Path(cfg['data']['manifest']).read_text().splitlines() if l.strip()]
paths = [str(d / index[r['sample_id']]) for r in rows if r['split'] in ('train', 'val')]
t0 = time.time(); _load = torch.load
with ThreadPoolExecutor(16) as ex:
    PRE = dict(zip([os.path.basename(p) for p in paths], ex.map(lambda p: _load(p, weights_only=True), paths)))
print(json.dumps({'v3p_preload_samples': len(PRE), 'seconds': round(time.time() - t0, 1)}), flush=True)


def load(f, *a, **k):
    if isinstance(f, (str, Path)) and os.path.basename(str(f)) in PRE:
        return PRE[os.path.basename(str(f))]
    return _load(f, *a, **k)


torch.load = load
from fieldbridge.train import run
run(config)
