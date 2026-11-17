"""V5 quick-check job lists: V5a = held shear/stretch/bend (firm grip, g=2), press/drop/tap reused from V3;
V5b = same actions + v5 ranges, all actions. val 0-282 + train 0-999. Split over a800-2 (r_queue) and a800-3/4/5 (pools)."""
import json
A = ['press', 'drop', 'shear', 'stretch', 'bend', 'tap30hold']
weak = lambda i: A[(i // 3) % 6] in ('shear', 'stretch', 'bend')
VA = ['--g-all', '-2', '--weak', 'hold']; VB = VA + ['--ranges', 'v5']
jobs = []
for s, n in (('val', 283), ('train', 1000)):
    for i in range(n):
        if weak(i): jobs.append(['data/v5a_main', s, i, VA])
        jobs.append(['data/v5b_main', s, i, VB])
w = {'a2': 8, 'a4': 8, 'a3': 7, 'a5': 8}; cyc = [h for h in w for _ in range(w[h])]
H = {h: [] for h in w}
for k, j in enumerate(jobs): H[cyc[k % len(cyc)]].append(j)
for h in ('a3', 'a4', 'a5'): json.dump(H[h], open(f'docs/v5/jobs_{h}.json', 'w'))
q = {'phase': 'v5gen', 'max_per_gpu': 5, 'gpus': [0, 1, 2, 3, 4, 5, 6, 7],
     'note': 'V5 quick check (val + train 0-999): V5a = held shear/stretch/bend (firm grip, g=2) + V3 press/drop/tap; V5b = same + v5 ranges, all actions',
     'jobs': [dict(id='v5gen_%s_%s_%06d' % (j[0].split('/')[1], j[1], j[2]), kind='cmd', group='v5gen', mem_mib=7000, retries=1,
                   cmd=f'.venv/bin/python scripts/v3n/gen_v3.py --split {j[1]} --indices {j[2]} --root {j[0]} ' + ' '.join(j[3])) for j in H['a2']]}
json.dump(q, open('docs/v5/queue.json', 'w'), indent=1)
print(len(jobs), {h: len(v) for h, v in H.items()})
