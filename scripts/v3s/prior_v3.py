"""V3s: zero-shot prior with the V3 prior checkpoints on Spring-Gaus synthetic (0-19 and 0-29 frames) and PhysTwin.

  .venv/bin/python -I scripts/v3s/prior_v3.py sg <mpm_synthetic.zip> data/v3s/prior20.json
  .venv/bin/python -I scripts/v3s/prior_v3.py pt <rgb_case_dir> data/v3s/pt/pt_prior.json

Identical code path to scripts/v3l/prior20.py and scripts/v3l/pt_prior.py (imported unchanged; same frames, views,
V-JEPA2 ViT-L encoding, 64-frame repetition, K0 = mean of the run ensemble). Only the run list, configs/checkpoints
and the parameter stats used for decoding are swapped for the V3 prior from docs/v3s/PRIOR_SOURCES.json
(derived from docs/v3p/HANDOFF.json):
  {"runs": [{"name":..., "config": "configs/...json", "checkpoint": "runs/.../best_material.pt"}, ...],
   "stats": "data/manifests/<v3>.specific.stats.json"}
The 'sanity_abs_dz_E_vs_v3f' field of the sg output compares with the V2 prior and is therefore NOT a sanity check here.
"""
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/v3f'), str(ROOT / 'scripts/v3l')]
import springgaus_fewshot as F  # noqa: E402

SRC = json.loads((ROOT / os.environ.get('FB_PRIOR_SOURCES', 'docs/v3s/PRIOR_SOURCES.json')).read_text())
RUNMAP = {r['name']: r for r in SRC['runs']}
F.RUNS = [r['name'] for r in SRC['runs']]
F.STATS = ROOT / SRC['stats']


def load_model(run, device):
    from fieldbridge.config import load_config
    from fieldbridge.local_model import build_model
    from fieldbridge.runtime import load_weights, sha256
    r = RUNMAP[run]
    cfg = load_config(str(ROOT / r['config']))
    model = build_model(cfg.model)
    ckpt = torch.load(ROOT / r['checkpoint'], map_location='cpu', weights_only=False)
    load_weights(model, ckpt, sha256(str(ROOT / cfg.model.weights)))
    return model.to(device).eval(), {'checkpoint_epoch': ckpt.get('epoch')}


F.load_model = load_model


def main():
    mode, inp, out = sys.argv[1:4]
    import argparse
    if mode == 'sg':
        import prior20
        prior20.main(argparse.Namespace(zip=inp, out=out, n_obs=20, scenes=['torus', 'duck', 'apple', 'C', 'E', 'V', 'banana', 'chess',
                                                                              'cream', 'cross', 'doraemon', 'droplet', 'rope', 'toothpaste']))
    elif mode == 'pt':
        import pt_prior
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        pt_prior.main(argparse.Namespace(case=inp, out=out, n_frames=47))
    else:
        raise SystemExit(mode)
    d = json.loads(Path(out).read_text()); d['prior_sources'] = SRC; d['note'] = 'V3 prior (scripts/v3s/prior_v3.py)'
    Path(out).write_text(json.dumps(d, indent=1, default=str))


if __name__ == '__main__':
    main()
