"""V3p: training configs = V2 main configs (configs/v2main/v2main_{B,C,D}_seed{17,29,43}.json) with ONLY the
data paths and output re-pointed to V3 (all hyper-parameters identical).
  .venv/bin/python scripts/v3p/make_configs.py --manifest v3main_t2000 --prefix v3main
"""
import argparse, json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
a = argparse.ArgumentParser(); a.add_argument('--manifest', required=True); a.add_argument('--prefix', required=True); args = a.parse_args()
for g in 'BCD':
    for s in (17, 29, 43):
        cfg = json.loads((ROOT / f'configs/v2main/v2main_{g}_seed{s}.json').read_text())
        cfg['data'].update(manifest=f'data/manifests/{args.manifest}.jsonl', stats=f'data/manifests/{args.manifest}.specific.stats.json',
                           prepared_dir=f'data/prepared/{args.manifest}_vjl')
        cfg['train']['output'] = f'runs/{args.prefix}_{g}_seed{s}'
        out = ROOT / f'configs/v3p/{args.prefix}_{g}_seed{s}.json'; out.write_text(json.dumps(cfg, indent=2)); print(out)
