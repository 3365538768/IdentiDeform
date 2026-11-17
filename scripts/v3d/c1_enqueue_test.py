"""Write docs/v3d/queue.json: the pre-registered C1 TEST jobs (docs/v3d/C1_TEST_PROTOCOL.json), ids v3d_*.
Per sample, in plan order: prior_top2 (budget 24, replay 0/4/12/24), lawmean_gt (24, replay 12/24),
lawmean_best3 (24, replay 12/24) -- the exact v3b command lines plus split/path plumbing;
then the addendum arm prior_top1 (24, replay 0/4/12/24) for all samples."""
import json, os, sys
MEM = int(sys.argv[1]) if len(sys.argv) > 1 else 14000
MAXG = int(sys.argv[2]) if len(sys.argv) > 2 else 4
plan = json.load(open('docs/v3d/C1_TEST_SAMPLES.json'))['samples']
base = ('.venv/bin/python scripts/v3d/c1_refine.py --run v2main_C_seed29 --split test '
        '--ensemble docs/v3d/ENSEMBLE_BCD9_TEST.json --out data/v3d_refine_test --index {i}')
def job(jid, cmd):
    return {'id': jid, 'kind': 'cmd', 'group': 'v3d-C1test', 'mem_mib': MEM, 'retries': 6, 'cmd': cmd}
jobs = []
for p in plan:
    i = p['index']
    jobs.append(job(f'v3d_C1t_{i:06d}', base.format(i=i) + ' --arm prior_top2 --budget 24 --replay-budgets 0,4,12,24'
                    + (' --gt-observed' if p['gt_observed'] else '')))
    jobs.append(job(f'v3d_C1g_{i:06d}', base.format(i=i) + ' --arm lawmean_gt --budget 24 --replay-budgets 12,24'))
    jobs.append(job(f'v3d_C1b_{i:06d}', base.format(i=i) + ' --arm lawmean_best3 --budget 24 --replay-budgets 12,24'))
# Addendum arm (lead request, registered before any test simulation): prior_top1, scheduled after all of the above.
for p in plan:
    i = p['index']
    jobs.append(job(f'v3d_C1o_{i:06d}', base.format(i=i) + ' --arm prior_top1 --budget 24 --replay-budgets 0,4,12,24'))
q = {'phase': 'v3d', 'max_per_gpu': MAXG, 'jobs': jobs}
open('docs/v3d/queue.json.tmp', 'w').write(json.dumps(q, indent=1)); os.replace('docs/v3d/queue.json.tmp', 'docs/v3d/queue.json')
print(len(jobs), MEM, MAXG)
