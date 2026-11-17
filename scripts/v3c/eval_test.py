"""V3c: ONE-TIME V2 main test evaluation of a trained run (frozen protocol docs/v3c/TEST_PROTOCOL.json).

Identical to `python -m fieldbridge.cli evaluate --split test` except for manifest/stats wiring:
the training config points at data/manifests/v2main.jsonl (no test rows). Here the config is
re-pointed at data/manifests/v2main_withtest.jsonl (exact v2main bytes + test rows), its stats
sibling, and a test-only feature cache. The checkpoint codec check ignores ONLY manifest_sha256,
and is guarded by: (a) the new manifest is a strict byte-extension of the training manifest,
(b) checkpoint stats == the frozen training stats file, (c) new stats == frozen stats apart from the hash.
Refuses to overwrite an existing output (run once).
  .venv/bin/python scripts/v3c/eval_test.py --config configs/v2main/v2main_C_seed17.json \
      --checkpoint runs/v2main_C_seed17/best_material.pt --prepared-dir data/prepared/v2main_test_vjl \
      --output runs/v2main_C_seed17/test_once
"""
import argparse, json
from dataclasses import asdict
from pathlib import Path
import torch
from fieldbridge.config import load_config, ModelConfig
from fieldbridge.data import MaterialDataset
from fieldbridge.evaluation import evaluate
from fieldbridge.local_model import build_model
from fieldbridge.runtime import load_weights, sha256

NEW_MANIFEST = 'data/manifests/v2main_withtest.jsonl'


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--checkpoint', required=True)
    p.add_argument('--prepared-dir', required=True); p.add_argument('--output', required=True)
    p.add_argument('--check-only', action='store_true', help='run every guard and build the dataset, but do not evaluate')
    a = p.parse_args()
    if Path(a.output).exists():
        raise SystemExit('output exists: test is evaluated once')
    cfg = load_config(a.config)
    base_manifest, base_stats_path = cfg.data.manifest, cfg.data.stats
    if base_manifest != 'data/manifests/v2main.jsonl':
        raise SystemExit('only V2 main runs are covered by this protocol')
    base = Path(base_manifest).read_bytes(); new = Path(NEW_MANIFEST).read_bytes()
    if new[:len(base)] != base or any(json.loads(l)['split'] != 'test' for l in new[len(base):].decode().splitlines()):
        raise SystemExit('new manifest is not train/val bytes + test rows')
    new_stats_path = base_stats_path.replace('data/manifests/v2main.', 'data/manifests/v2main_withtest.')
    frozen = json.loads(Path(base_stats_path).read_text()); fresh = json.loads(Path(new_stats_path).read_text())
    strip = lambda s: {k: v for k, v in s.items() if k != 'manifest_sha256'}
    ckpt = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if ckpt['stats'] != frozen or strip(fresh) != strip(frozen) or frozen['manifest_sha256'] != sha256(base_manifest):
        raise SystemExit('codec/statistics mismatch')
    cfg.data.manifest, cfg.data.stats, cfg.data.prepared_dir = NEW_MANIFEST, new_stats_path, a.prepared_dir
    dataset = MaterialDataset(cfg.data, 'test', cfg.model.field_size, cfg.model.field_mode)
    if strip(dataset.codec.stats) != strip(ckpt['stats']):
        raise SystemExit('dataset codec mismatch')
    if asdict(ModelConfig(**ckpt['config']['model'])) != cfg.dump()['model']:
        raise SystemExit('checkpoint/config model mismatch')
    backbone_sha = sha256(cfg.model.weights) if cfg.model.weights else None
    prep = dataset.prepared
    if prep.get('backbone_checkpoint_sha256') != backbone_sha or prep.get('hub_name') != cfg.model.hub_name \
            or prep.get('encoder_source_sha256') != sha256('src/fieldbridge/model.py'):
        raise SystemExit('prepared feature backbone/implementation mismatch')
    if a.check_only:
        print(json.dumps({'check_only': True, 'test_samples': len(dataset), 'config': a.config, 'checkpoint': a.checkpoint,
                          'checkpoint_epoch': ckpt.get('epoch'), 'best_material': ckpt.get('best_material')}))
        return
    model = build_model(cfg.model).to(cfg.train.device); load_weights(model, ckpt, backbone_sha)
    result = evaluate(model, dataset, torch.device(cfg.train.device), a.output, cfg.train.val_batch_size,
                      cfg.train.val_workers, cfg.train.val_bf16)
    print(json.dumps({'output': a.output, 'samples': result['samples'], 'standardized_mae': result['standardized_mae']}))


if __name__ == '__main__':
    main()
