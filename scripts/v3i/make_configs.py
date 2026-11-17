"""Generate V3i ablation configs from the V2-main B/C configs (only the listed fields change).

Data scaling: train_object_fraction in {0.10, 0.25, 0.50} with subset_seed = seed (F1 mechanism; val/test untouched).
View count: C with fixed camera subsets [0] and [0, 1] (100% data). 100%/3-view = existing v2main runs.
Features are read from the /dev/shm staging copy of data/prepared/v2main_vjl (scripts/v3i/stage_cache.py).
"""
import json
from pathlib import Path
SHM = '/dev/shm/rsi-v3i/v2main_vjl'
jobs = []
for model in 'BC':
    for seed in (17, 29, 43):
        base = json.loads(Path(f'configs/v2main/v2main_{model}_seed{seed}.json').read_text())
        variants = [(f'{model}_f{int(f * 100):02d}', {'train_object_fraction': f, 'subset_seed': seed}, None) for f in (0.10, 0.25, 0.50)]
        if model == 'C':
            variants += [('C_v1', {}, [0]), ('C_v2', {}, [0, 1])]
        for group, data, views in variants:
            cfg = json.loads(json.dumps(base))
            cfg['data'].update(data, prepared_dir=SHM, view_dropout=0.0)
            cfg['train']['output'] = f'runs/v3i_{group}_seed{seed}'
            if views is not None:
                cfg['v3i'] = {'views': views}
            name = f'v3i_{group}_seed{seed}'
            Path(f'configs/v3i/{name}.json').write_text(json.dumps(cfg, indent=2) + '\n')
            jobs.append({'id': name, 'config': f'configs/v3i/{name}.json', 'group': group, 'seed': seed,
                         'fraction': data.get('train_object_fraction', 1.0), 'views': views or [0, 1, 2]})
# Small fractions first so partial curves exist early; view runs interleaved.
order = {'f10': 0, 'f25': 1, 'v1': 2, 'v2': 2, 'f50': 3}
jobs.sort(key=lambda j: (order[j['group'].split('_')[1]], j['seed'], j['group']))
Path('docs/v3i/jobs.json').write_text(json.dumps({'note': 'run by scripts/v3i/launch_a8004.py on a800-4 (not the a800-2 r_queue)', 'jobs': jobs}, indent=2) + '\n')
print(len(jobs), 'configs')
