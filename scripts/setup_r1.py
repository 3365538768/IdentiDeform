"""R1: full-data strict feedback ablation (B/C/D) with flat vs law-conditioned heads, 3 seeds.

B: no field loss, no feedback.  C: field supervision, feedback removed.  D: field supervision + feedback.
Only the feedback differs between C and D (same bottleneck, decoder, losses, stages, budget).
"""
import copy
import json
from pathlib import Path

base = json.loads(Path('configs/g1f_B_rgb_only_seed17.json').read_text())
base['model'].update(architecture='legacy', feedback='none')
base['data'].update(manifest='data/manifests/deformfield_v1.jsonl', stats='data/manifests/deformfield_v1.stats.json',
                    prepared_dir='data/prepared/r1_full_dinov3l16')
base['train'].update(epochs=30, field_epochs=0, batch_size=8, accumulation=1, workers=4, val_batch_size=8,
                     val_workers=4, early_stop_patience=5, early_stop_min_joint_epochs=8, log_interval=50,
                     contact_loss='balanced_bce', initialize_rgb_baseline='', purpose='research')
groups = {  # name: (architecture, feedback, field_weight, field_epochs, law_weight)
    'B': ('legacy', 'none', 0.0, 0, 0.1), 'C': ('legacy', 'none', 0.1, 2, 0.1), 'D': ('legacy', 'fields', 0.1, 2, 0.1),
    'Bh': ('legacy_hier', 'none', 0.0, 0, 1.0), 'Ch': ('legacy_hier', 'none', 0.1, 2, 1.0),
    'Dh': ('legacy_hier', 'fields', 0.1, 2, 1.0),
}
jobs = []
Path('configs/r1').mkdir(parents=True, exist_ok=True)
for seed in (17, 29, 43):
    for name, (arch, feedback, fw, fe, lw) in groups.items():
        c = copy.deepcopy(base); rid = f'r1_{name}_seed{seed}'
        c['model'].update(architecture=arch, feedback=feedback)
        c['train'].update(seed=seed, field_weight=fw, field_epochs=fe, law_weight=lw, output=f'runs/{rid}')
        path = Path(f'configs/r1/{rid}.json'); path.write_text(json.dumps(c, indent=2) + '\n')
        jobs.append({'id': rid, 'config': str(path), 'group': name, 'seed': seed, 'kind': 'train'})
# Flat head with the hierarchical law weight: separates "law weight" from "law-conditioned head".
c = copy.deepcopy(base); rid = 'r1_Cl_seed17'
c['train'].update(seed=17, field_weight=0.1, field_epochs=2, law_weight=1.0, output=f'runs/{rid}')
Path(f'configs/r1/{rid}.json').write_text(json.dumps(c, indent=2) + '\n')
jobs.append({'id': rid, 'config': f'configs/r1/{rid}.json', 'group': 'Cl', 'seed': 17, 'kind': 'train'})
Path('docs/r1/queue.json').write_text(json.dumps({'phase': 'R1', 'max_per_gpu': 3, 'gpus': list(range(8)),
                                                  'jobs': jobs}, indent=2) + '\n')
print(len(jobs), 'jobs')
