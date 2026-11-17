"""V3i training wrapper: unchanged fieldbridge.train on cached V-JEPA2 features, optionally restricted to a
fixed camera subset (view-count ablation).

The config may carry an extra top-level section {"v3i": {"views": [0, 1]}} which is stripped before the
standard loader sees it. Views outside the subset are masked exactly like the established masked
single-view layout: valid=False (excluded from the temporal-view transformer via the padding mask and
from every pooled average) and field_mask=0 (no field loss), in train AND val. view_dropout stays 0.
A restart resumes from <output>/last.pt.

  python scripts/v3i/train_views.py --config configs/v3i/v3i_C_v1_seed17.json
"""
import argparse
import json
from pathlib import Path

import torch

import fieldbridge.data as fbdata
from fieldbridge.train import run

p = argparse.ArgumentParser(); p.add_argument('--config', required=True); a = p.parse_args()
raw = json.loads(Path(a.config).read_text())
views = raw.pop('v3i', {}).get('views')
out = Path(raw['train']['output']); out.mkdir(parents=True, exist_ok=True)
if (out / 'last.pt').exists():
    raw['train']['resume'] = str(out / 'last.pt')
if views is not None:
    keep = torch.zeros(raw['data']['max_views'], dtype=torch.bool); keep[list(views)] = True
    original = fbdata.MaterialDataset.__getitem__

    def getitem(self, index):
        s = original(self, index)
        k = keep[:s['valid'].shape[0]]
        s['valid'] = s['valid'] & k[:, None]
        s['field_mask'] = s['field_mask'] * k[:, None, None, None, None].to(s['field_mask'].dtype)
        return s
    fbdata.MaterialDataset.__getitem__ = getitem  # DataLoader workers fork, so they inherit the patch
(out / 'v3i_views.json').write_text(json.dumps({'views': views, 'source_config': a.config}))
effective = Path('runs/v3i_logs') / (Path(a.config).stem + '.effective.json'); effective.parent.mkdir(parents=True, exist_ok=True)
effective.write_text(json.dumps(raw, indent=2))
run(str(effective))
