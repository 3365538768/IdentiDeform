"""V5b remaining data: test 0-299 + train 1000-1999 (all actions, held weak actions, v5 ranges, g=2)."""
import json
VB = ['--g-all', '-2', '--weak', 'hold', '--ranges', 'v5']
jobs = [['data/v5b_main', 'test', i, VB] for i in range(300)] + [['data/v5b_main', 'train', i, VB] for i in range(1000, 2000)]
w = {'a2': 8, 'a4': 8, 'a3': 7, 'a5': 8}; cyc = [h for h in w for _ in range(w[h])]
H = {h: [] for h in w}
for k, j in enumerate(jobs): H[cyc[k % len(cyc)]].append(j)
for h in ('a3', 'a4', 'a5'): json.dump(H[h], open(f'docs/v5/jobs2_{h}.json', 'w'))
q = json.load(open('docs/v5/queue.json'))
q['jobs'] += [dict(id='v5gen_v5b_main_%s_%06d' % (j[1], j[2]), kind='cmd', group='v5gen', mem_mib=7000, retries=1,
                   cmd=f'.venv/bin/python scripts/v3n/gen_v3.py --split {j[1]} --indices {j[2]} --root {j[0]} ' + ' '.join(j[3])) for j in H['a2']]
q['gpus'] = [0, 1, 2, 3, 4, 5, 6, 7]; q['note'] += ' | 21:35 V5b rest: test 0-299 + train 1000-1999'
json.dump(q, open('docs/v5/queue.json', 'w'), indent=1)
print({h: len(v) for h, v in H.items()})
