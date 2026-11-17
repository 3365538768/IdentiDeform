"""V3p: incremental frozen V-JEPA 2 ViT-L feature cache for data/v3_main (same features as scripts/prepare_vjepa_cache.py).

Differences from prepare_vjepa_cache.py (V2 script left untouched): the cache directory is shared by several
manifests (V3 data arrives in batches: val/test first, then train 0-1999, then 2000-3999). Every per-sample file
is <sha256(sample_id)>.pt with sample_id v3main/<split>/<idx>, produced by the identical code path; existing files
are skipped. The training contract (index.json with the manifest hash) is written per manifest by --finalize,
into its own index directory whose file entries point into the shared cache.

  build : .venv/bin/python scripts/v3p/cache_v3.py --manifest v3p_cache_vt --splits val,test --shard 0 --num-shards 2 --status-output docs/v3p/cache_vt_s0.json
  index : .venv/bin/python scripts/v3p/cache_v3.py --manifest v3main_t2000 --finalize --index-dir data/prepared/v3main_t2000_vjl
"""
import argparse, hashlib, json, os, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
BASE_CONFIG = 'configs/v2main/v2main_cache.json'
CACHE = 'data/prepared/v3main_vjl'


def sha256_text(value):
    return hashlib.sha256(value.encode()).hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--manifest', required=True)
    p.add_argument('--splits', default='train,val,test'); p.add_argument('--shard', type=int, default=0); p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--status-output'); p.add_argument('--finalize', action='store_true'); p.add_argument('--index-dir'); a = p.parse_args()
    from fieldbridge.config import load_config
    from fieldbridge.runtime import atomic_json, sha256
    cfg = load_config(BASE_CONFIG)
    cfg.data.manifest = f'data/manifests/{a.manifest}.jsonl'; cfg.data.stats = f'data/manifests/{a.manifest}.specific.stats.json'
    cfg.data.prepared_dir = CACHE
    directory = Path(CACHE); directory.mkdir(parents=True, exist_ok=True)
    meta = {'schema': 1, 'manifest_sha256': sha256(cfg.data.manifest), 'frames': cfg.data.frames,
            'max_views': cfg.data.max_views, 'image_size': cfg.data.image_size,
            'field_size': cfg.model.field_size, 'field_mode': cfg.model.field_mode,
            'backbone_checkpoint_sha256': sha256(cfg.model.weights), 'hub_name': cfg.model.hub_name,
            'encoder_source_sha256': sha256('src/fieldbridge/model.py'),
            'precision': 'bf16', 'source_frames': 64, 'tubelet': 2, 'hidden_layer': -1, 'files': {},
            'v3p_shared_cache': CACHE}
    rows = [json.loads(l) for l in Path(cfg.data.manifest).read_text().splitlines() if l.strip()]
    if a.finalize:
        out = Path(a.index_dir); out.mkdir(parents=True, exist_ok=True)
        rel = os.path.relpath(directory, out)
        missing = [r['sample_id'] for r in rows if not (directory / (sha256_text(r['sample_id']) + '.pt')).exists()]
        if missing:
            raise SystemExit(f'{len(missing)} samples not cached, e.g. {missing[:5]}')
        meta['files'] = {r['sample_id']: (sha256_text(r['sample_id']) + '.pt' if rel == '.' else f'{rel}/{sha256_text(r["sample_id"])}.pt') for r in rows}
        atomic_json(out / 'index.json', meta)
        print('index', out / 'index.json', len(rows)); return
    import torch
    from transformers import AutoModel
    from fieldbridge.data import MaterialDataset
    from fieldbridge.train import atomic_save
    MEAN = torch.tensor([.485, .456, .406]).view(1, 1, 3, 1, 1); STD = torch.tensor([.229, .224, .225]).view(1, 1, 3, 1, 1)
    raw_cfg = load_config(BASE_CONFIG).data
    raw_cfg.manifest = cfg.data.manifest; raw_cfg.prepared_dir = ''; raw_cfg.frames = 64
    raw_cfg.target_space = 'native'; raw_cfg.stats = cfg.data.stats.replace('.specific.stats.json', '.stats.json')
    splits = [s for s in a.splits.split(',') if any(r['split'] == s for r in rows)]
    datasets = [MaterialDataset(raw_cfg, s, cfg.model.field_size, cfg.model.field_mode) for s in splits]
    mine = lambda row: int(sha256_text(row['sample_id']), 16) % a.num_shards == a.shard
    todo = [(ds, i) for ds in datasets for i, row in enumerate(ds.rows)
            if mine(row) and not (directory / (sha256_text(row['sample_id']) + '.pt')).exists()]
    total = sum(mine(row) for ds in datasets for row in ds.rows); done = total - len(todo)
    status = lambda st: atomic_json(a.status_output, {'status': st, 'completed': done, 'total': total, 'manifest': a.manifest,
                                                      'new_samples_per_second': (done - total + len(todo)) / max(time.time() - started, 1e-6),
                                                      'updated_unix': time.time()})
    started = time.time()
    if not todo:
        status('shard_complete'); return
    model = AutoModel.from_pretrained(Path(cfg.model.repo), dtype=torch.bfloat16).cuda().eval()
    loader = torch.utils.data.DataLoader(todo, batch_size=None, num_workers=6,
                                         collate_fn=lambda item: (item[0].rows[item[1]], item[0][item[1]]))
    for row, sample in loader:
        rgb = sample['rgb']; v, t = rgb.shape[:2]
        if t != 64:
            raise ValueError('Expected 64 source frames')
        with torch.no_grad():
            x = ((rgb[None] - MEAN) / STD)[0].cuda().to(torch.bfloat16)
            tokens = model(pixel_values_videos=x, skip_predictor=True).last_hidden_state
        h = w = cfg.data.image_size // 16
        features = tokens.reshape(v, 32, h, w, -1).permute(0, 1, 4, 2, 3).cpu()
        out = {k: (val[:, 0::2] if torch.is_tensor(val) and val.ndim >= 2 and val.shape[:2] == (v, 64) else val)
               for k, val in sample.items()}
        out['times'] = (sample['times'][:, 0::2] + sample['times'][:, 1::2]) / 2
        out['valid'] = sample['valid'][:, 0::2] & sample['valid'][:, 1::2]
        out['features'] = features
        out['rgb'] = torch.zeros((v, 32, 3, 1, 1), dtype=torch.float32)
        out['field_mask'] = out['field_mask'].bool()
        atomic_save(out, directory / (sha256_text(row['sample_id']) + '.pt')); done += 1
        if done % 10 == 0 or done == total:
            status('building')
    status('shard_complete')


if __name__ == '__main__':
    main()
