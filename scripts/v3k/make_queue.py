"""V3k: (re)write docs/v3k/queue.json. Usage: make_queue.py [--with-b]  (B jobs need data/v3k/hull_assets/<idx>/*.ply)."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
P = json.loads((ROOT / 'docs/v3k/PROTOCOL.json').read_text())
ZIP = 'data/external_datasets/springgaus_gdrive/mpm_synthetic.zip'
jobs = []
for arm in ('prior', 'jellymean', 'diag'):
    for s in P['A_springgaus_in_target_refinement']['scenes']:
        jobs.append({'id': f'v3k_a_{arm}_{s}', 'kind': 'cmd', 'group': f'v3k-a-{arm}', 'mem_mib': 3000, 'retries': 3,
                     'cmd': f'scripts/v3f/replay_env.sh scripts/v3k/a_sg_refine.py {ZIP} {s} --arm {arm}'})
if '--with-b' in sys.argv:
    for i in P['B_reconstructed_geometry']['samples']:
        ply = sorted((ROOT / f'data/v3k/hull_assets/{i:06d}').glob('*.ply'))
        ply = [p for p in ply if 'normalized' not in p.name]
        if len(ply) != 1:
            print('skip B', i, ply); continue
        jobs.append({'id': f'v3k_b_{i:06d}', 'kind': 'cmd', 'group': 'v3k-b-hull', 'mem_mib': 6000, 'retries': 4,
                     'cmd': f'.venv/bin/python scripts/v3k/bc_refine.py --index {i} --variant hull --asset {ply[0].relative_to(ROOT)}'})
for i in P['C_simulator_mismatch']['samples']:
    jobs.append({'id': f'v3k_c_{i:06d}', 'kind': 'cmd', 'group': 'v3k-c-mismatch', 'mem_mib': 6000, 'retries': 4,
                 'cmd': f'.venv/bin/python scripts/v3k/bc_refine.py --index {i} --variant mismatch'})
q = {'phase': 'v3k', 'max_per_gpu': 5, 'gpus': [0, 1, 2, 3, 4, 5, 6, 7], 'jobs': jobs}
tmp = ROOT / 'docs/v3k/queue.json.tmp'; tmp.write_text(json.dumps(q, indent=1)); tmp.replace(ROOT / 'docs/v3k/queue.json')
print(len(jobs), 'jobs')
