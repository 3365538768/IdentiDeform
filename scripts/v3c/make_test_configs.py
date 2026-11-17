"""V3c: test-only feature-cache configs (V-JEPA 2 ViT-L for B/C/D, DINOv3 ViT-L/16 for the DINO baselines)."""
import json
from pathlib import Path
M = 'data/manifests/v2main_withtest.jsonl'
v = json.loads(Path('configs/v2main/v2main_cache.json').read_text())
v['data'].update(manifest=M, stats='data/manifests/v2main_withtest.specific.stats.json', prepared_dir='data/prepared/v2main_test_vjl')
v['train']['output'] = 'runs/v3c_test_vjl_cache'
Path('configs/v3c/v3c_test_vjl_cache.json').write_text(json.dumps(v, indent=2))
d = json.loads(Path('configs/v3c/v3c_dino_cache.json').read_text())
d['data'].update(manifest=M, stats='data/manifests/v2main_withtest.stats.json', prepared_dir='data/prepared/v2main_test_dinov3l16')
d['train']['output'] = 'runs/v3c_test_dino_cache'
Path('configs/v3c/v3c_test_dino_cache.json').write_text(json.dumps(d, indent=2))
print('ok')
