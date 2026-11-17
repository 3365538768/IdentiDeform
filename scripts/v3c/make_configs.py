"""V3c: DINOv3 ViT-L/16 configs on V2 main (same heads/budget/splits as configs/v2main, 16 frames)."""
import copy, json
from pathlib import Path

weights = json.loads(Path('docs/v3c/DINOV3_CONVERT.json').read_text())['weights']
Path('configs/v3c').mkdir(parents=True, exist_ok=True)
DINO = {'backbone': 'dinov3', 'repo': 'external/dinov3', 'weights': weights, 'hub_name': 'dinov3_vitl16', 'architecture': 'legacy'}
cache = copy.deepcopy(json.loads(Path('configs/v2main/v2main_cache.json').read_text()))
cache['model'].update(DINO)
cache['data'].update(frames=16, prepared_dir='data/prepared/v2main_dinov3l16', target_space='native', stats='data/manifests/v2main.stats.json')
cache['train']['output'] = 'runs/v3c_dino_cache'
Path('configs/v3c/v3c_dino_cache.json').write_text(json.dumps(cache, indent=2))
for group, source, static in [('B', 'B', False), ('C', 'C', False), ('S', 'B', True)]:
    for seed in (17, 29, 43):
        cfg = json.loads(Path(f'configs/v2main/v2main_{source}_seed{seed}.json').read_text())
        cfg['model'].update(DINO)
        cfg['data'].update(frames=16, prepared_dir='data/prepared/v2main_dinov3l16')
        if static:
            cfg['data']['static_first_frame'] = True
        cfg['train']['output'] = f'runs/v3c_dino_{group}_seed{seed}'
        Path(f'configs/v3c/v3c_dino_{group}_seed{seed}.json').write_text(json.dumps(cfg, indent=2))
print('ok', weights)
