"""R1b: same B/C/D protocol as R1 but trained on video-identifiable targets (E/rho, nu, rho, sigma_y/rho)."""
import json
from pathlib import Path

jobs = []
Path('configs/r1b').mkdir(parents=True, exist_ok=True); Path('docs/r1b').mkdir(parents=True, exist_ok=True)
for seed in (17, 29, 43):
    for name in ('B', 'C', 'D'):
        c = json.loads(Path(f'configs/r1/r1_{name}_seed{seed}.json').read_text()); rid = f'r1b_{name}_seed{seed}'
        c['data'].update(target_space='specific', stats='data/manifests/deformfield_v1.specific.stats.json')
        c['train']['output'] = f'runs/{rid}'
        path = Path(f'configs/r1b/{rid}.json'); path.write_text(json.dumps(c, indent=2) + '\n')
        jobs.append({'id': rid, 'config': str(path), 'group': name, 'seed': seed, 'kind': 'train'})
Path('docs/r1b/queue.json').write_text(json.dumps({'phase': 'R1b', 'max_per_gpu': 3, 'gpus': list(range(8)), 'jobs': jobs}, indent=2) + '\n')
print(len(jobs))
