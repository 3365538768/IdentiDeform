"""V3p baseline cache: DINOv3 ViT-L/16 features for the FIRST FRAME only (first-frame static baseline, V2 v3c_dino_S).

The V2 baseline trains on the 16-frame DINOv3 cache with data.static_first_frame=true, which masks frames 1..15
(valid=False) so only frame 0 enters the model. Here only frame 0 is encoded (same FrameEncoder, bf16 autocast,
same preprocessing as scripts/prepare_feature_cache.py) and stored as a stride-0 expansion over the 16 frames
(frames 1..15 are masked anyway), which makes the cache ~16x cheaper. Targets/fields/masks/times are the raw
16-frame loader outputs, as in prepare_feature_cache.py. Shared per-sample files + per-manifest index like cache_v3.py.
  build : .venv/bin/python scripts/v3p/dino_ff_cache.py --manifest v3main_t2000 --status-output docs/v3p/dinoff_s0.json [--shard i --num-shards n]
  index : .venv/bin/python scripts/v3p/dino_ff_cache.py --manifest v3main_t2000 --finalize --index-dir data/prepared/v3main_t2000_dinoff
"""
import argparse, hashlib, json, os, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
BASE = 'configs/v3c/v3c_dino_cache.json'; CACHE = 'data/prepared/v3main_dinoff'
sha_t = lambda v: hashlib.sha256(v.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--manifest', required=True); p.add_argument('--splits', default='train,val,test')
    p.add_argument('--shard', type=int, default=0); p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--status-output'); p.add_argument('--finalize', action='store_true'); p.add_argument('--index-dir'); a = p.parse_args()
    from fieldbridge.config import load_config
    from fieldbridge.runtime import atomic_json, sha256
    cfg = load_config(BASE)
    cfg.data.manifest = f'data/manifests/{a.manifest}.jsonl'; cfg.data.stats = f'data/manifests/{a.manifest}.stats.json'
    meta = {'schema': 1, 'manifest_sha256': sha256(cfg.data.manifest), 'frames': cfg.data.frames,
            'max_views': cfg.data.max_views, 'image_size': cfg.data.image_size,
            'field_size': cfg.model.field_size, 'field_mode': cfg.model.field_mode,
            'backbone_checkpoint_sha256': sha256(cfg.model.weights), 'hub_name': cfg.model.hub_name,
            'encoder_source_sha256': sha256('src/fieldbridge/model.py'), 'precision': 'bf16', 'files': {},
            'v3p_first_frame_only': True, 'v3p_shared_cache': CACHE}
    rows = [json.loads(l) for l in Path(cfg.data.manifest).read_text().splitlines() if l.strip()]
    directory = Path(CACHE); directory.mkdir(parents=True, exist_ok=True)
    if a.finalize:
        out = Path(a.index_dir); out.mkdir(parents=True, exist_ok=True); rel = os.path.relpath(directory, out)
        missing = [r['sample_id'] for r in rows if not (directory / (sha_t(r['sample_id']) + '.pt')).exists()]
        if missing:
            raise SystemExit(f'{len(missing)} samples not cached, e.g. {missing[:5]}')
        meta['files'] = {r['sample_id']: f'{rel}/{sha_t(r["sample_id"])}.pt' for r in rows}
        atomic_json(out / 'index.json', meta); print('index', out, len(rows)); return
    import torch
    from fieldbridge.data import MaterialDataset
    from fieldbridge.model import FrameEncoder
    from fieldbridge.train import atomic_save
    cfg.data.prepared_dir = ''
    splits = [s for s in a.splits.split(',') if any(r['split'] == s for r in rows)]
    datasets = [MaterialDataset(cfg.data, s, cfg.model.field_size, cfg.model.field_mode) for s in splits]
    mine = lambda row: int(sha_t(row['sample_id']), 16) % a.num_shards == a.shard
    todo = [(ds, i) for ds in datasets for i, row in enumerate(ds.rows) if mine(row) and not (directory / (sha_t(row['sample_id']) + '.pt')).exists()]
    total = sum(mine(r) for ds in datasets for r in ds.rows); done = total - len(todo); started = time.time()
    encoder = FrameEncoder(cfg.model).cuda().eval()
    loader = torch.utils.data.DataLoader(todo, batch_size=None, num_workers=6, collate_fn=lambda it: (it[0].rows[it[1]], it[0][it[1]]))
    for row, sample in loader:
        rgb = sample['rgb']; shape = rgb.shape
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            f0 = encoder(rgb[:, :1].flatten(0, 1).cuda()).cpu().reshape(shape[0], 1, encoder.dim, shape[-2] // encoder.patch_size, shape[-1] // encoder.patch_size)
        sample['features'] = f0.expand(shape[0], shape[1], *f0.shape[2:])
        sample['rgb'] = torch.zeros((*shape[:3], 1, 1), dtype=torch.float32)
        sample['field_mask'] = sample['field_mask'].bool()
        atomic_save(sample, directory / (sha_t(row['sample_id']) + '.pt')); done += 1
        if a.status_output and (done % 20 == 0 or done == total):
            atomic_json(a.status_output, {'status': 'building', 'completed': done, 'total': total, 'sps': (done - total + len(todo)) / (time.time() - started)})
    if a.status_output:
        atomic_json(a.status_output, {'status': 'shard_complete', 'completed': done, 'total': total})


if __name__ == '__main__':
    main()
