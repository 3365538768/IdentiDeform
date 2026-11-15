"""Frozen V-JEPA 2 feature cache in the DINO cache format (independent video-backbone baseline).

All 64 frames per view enter V-JEPA 2 (tubelet 2), giving 32 time steps of 14x14x1024 tokens at
224 px. Field targets, masks and validity are taken at the even frames and times are tubelet
centres, so B/C/D heads train unchanged with data.frames=32. Sharded like prepare_feature_cache.py:
run --shard i --num-shards n on n GPUs, then once without sharding to merge and write index.json.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import torch
from transformers import AutoModel

from fieldbridge.config import load_config
from fieldbridge.data import MaterialDataset
from fieldbridge.runtime import atomic_json, sha256
from fieldbridge.train import atomic_save

MEAN = torch.tensor([.485, .456, .406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([.229, .224, .225]).view(1, 1, 3, 1, 1)


def sha256_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True)
    p.add_argument('--splits', default='train,val'); p.add_argument('--shard', type=int, default=0); p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--status-output', required=True)
    # >=0: cache the input of encoder block k (pre-norm) so the last blocks can be finetuned on top (P3).
    p.add_argument('--hidden-layer', type=int, default=-1); a = p.parse_args()
    cfg = load_config(a.config); directory = Path(cfg.data.prepared_dir); directory.mkdir(parents=True, exist_ok=True)
    weights_dir = Path(cfg.model.repo)
    meta = {'schema': 1, 'manifest_sha256': sha256(cfg.data.manifest), 'frames': cfg.data.frames,
            'max_views': cfg.data.max_views, 'image_size': cfg.data.image_size,
            'field_size': cfg.model.field_size, 'field_mode': cfg.model.field_mode,
            'backbone_checkpoint_sha256': sha256(cfg.model.weights), 'hub_name': cfg.model.hub_name,
            'encoder_source_sha256': sha256(Path(__file__).resolve().parents[1] / 'src/fieldbridge/model.py'),
            'precision': 'bf16', 'source_frames': 64, 'tubelet': 2, 'hidden_layer': a.hidden_layer, 'files': {}}
    if cfg.data.frames != 32:
        raise ValueError('V-JEPA 2 cache expects data.frames=32 (64 input frames, tubelet 2)')
    sharded = a.num_shards > 1
    partial = directory / (f'index.shard{a.shard}of{a.num_shards}.partial.json' if sharded else 'index.partial.json')
    for source in [partial] if sharded else [partial, *sorted(directory.glob('index.shard*.partial.json'))]:
        existing = json.loads(source.read_text()) if source.exists() else None
        if existing and any(existing.get(k) != v for k, v in meta.items() if k != 'files'):
            raise ValueError('Partial cache key mismatch: ' + str(source))
        if existing:
            meta['files'].update(existing['files'])
    raw_cfg = load_config(a.config).data; raw_cfg.prepared_dir = ''; raw_cfg.frames = 64
    # Cached targets stay native (like the DINO caches); target_space is applied at training load.
    raw_cfg.target_space = 'native'; raw_cfg.stats = raw_cfg.stats.replace('.specific.stats.json', '.stats.json')
    datasets = [MaterialDataset(raw_cfg, s, cfg.model.field_size, cfg.model.field_mode) for s in a.splits.split(',')]
    mine = lambda row: int(sha256_text(row['sample_id']), 16) % a.num_shards == a.shard
    todo = [(ds, i) for ds in datasets for i, row in enumerate(ds.rows)
            if mine(row) and not (row['sample_id'] in meta['files'] and (directory / meta['files'][row['sample_id']]).exists())]
    total = sum(mine(row) for ds in datasets for row in ds.rows); done = total - len(todo)
    model = AutoModel.from_pretrained(weights_dir, dtype=torch.bfloat16).cuda().eval()
    started = time.time()
    loader = torch.utils.data.DataLoader([x for x in todo], batch_size=None, num_workers=6,
                                         collate_fn=lambda item: (item[0].rows[item[1]], item[0][item[1]]))
    for row, sample in loader:
        rgb = sample['rgb']; v, t = rgb.shape[:2]
        if t != 64:
            raise ValueError('Expected 64 source frames')
        with torch.no_grad():
            x = ((rgb[None] - MEAN) / STD)[0].cuda().to(torch.bfloat16)
            if a.hidden_layer < 0:
                tokens = model(pixel_values_videos=x, skip_predictor=True).last_hidden_state
            else:  # mirrors VJEPA2Encoder.forward up to block hidden_layer
                tokens = model.encoder.embeddings(x)
                for layer in model.encoder.layer[:a.hidden_layer]:
                    tokens = layer(tokens, None, None, False)[0]
        h = w = cfg.data.image_size // 16
        features = tokens.reshape(v, 32, h, w, -1).permute(0, 1, 4, 2, 3).cpu()
        out = {k: (val[:, 0::2] if torch.is_tensor(val) and val.ndim >= 2 and val.shape[:2] == (v, 64) else val)
               for k, val in sample.items()}
        out['times'] = (sample['times'][:, 0::2] + sample['times'][:, 1::2]) / 2
        out['valid'] = sample['valid'][:, 0::2] & sample['valid'][:, 1::2]
        out['features'] = features
        out['rgb'] = torch.zeros((v, 32, 3, 1, 1), dtype=torch.float32)
        out['field_mask'] = out['field_mask'].bool()
        filename = sha256_text(row['sample_id']) + '.pt'
        atomic_save(out, directory / filename); meta['files'][row['sample_id']] = filename; done += 1
        if done % 20 == 0 or done == total:
            atomic_json(partial, meta)
            atomic_json(a.status_output, {'status': 'building', 'completed': done, 'total': total,
                        'new_samples_per_second': (done - total + len(todo)) / max(time.time() - started, 1e-6),
                        'updated_unix': time.time()})
    atomic_json(partial, meta)
    if sharded:
        atomic_json(a.status_output, {'status': 'shard_complete', 'completed': done, 'total': total, 'updated_unix': time.time()})
        return
    atomic_json(directory / 'index.json', meta)
    atomic_json(a.status_output, {'status': 'complete', 'completed': done, 'total': total, 'updated_unix': time.time(),
                'bytes': sum(f.stat().st_size for f in directory.glob('*.pt'))})


if __name__ == '__main__':
    main()
