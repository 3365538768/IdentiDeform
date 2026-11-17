"""Write docs/v3h/queue.json for the shared r_queue on a800-2 (ids v3h_q_*): the prior_top1 arm on ALL 60 C1 val
samples (docs/v3b/C1_SAMPLES.json) with the identical v3d code path used on test (scripts/v3d/c1_refine.py --split val):
9-run B/C/D ensemble prediction, top-1 law, no law probes, the same compass search, budget 24, replay 0/4/12/24.
States: data/v3b_refine/v2main_C_seed29/<idx>/prior_top1/state.json (next to the other val arms).
This supersedes the deferred v3h ens_map arm (same initialization; docs/v3h/jobs_ens_map_deferred.json not run)."""
import json, os
plan = json.load(open('docs/v3b/C1_SAMPLES.json'))['samples']
base = ('.venv/bin/python scripts/v3d/c1_refine.py --run v2main_C_seed29 --split val '
        '--ensemble docs/v3b/ENSEMBLE_BCD9_VAL.json --out data/v3b_refine --index {i} '
        '--arm prior_top1 --budget 24 --replay-budgets 0,4,12,24')
jobs = [{'id': f'v3h_q_top1_{p["index"]:06d}', 'kind': 'cmd', 'group': 'v3h-top1val', 'mem_mib': 6000, 'retries': 6,
         'cmd': base.format(i=p['index'])} for p in plan]
q = {'phase': 'v3h', 'max_per_gpu': 6, 'gpus': [0, 1, 2, 4, 5, 6, 7], 'jobs': jobs}
open('docs/v3h/queue.json.tmp', 'w').write(json.dumps(q, indent=1)); os.replace('docs/v3h/queue.json.tmp', 'docs/v3h/queue.json')
print(len(jobs))
