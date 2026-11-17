"""Rebalance not-yet-started V3 generation jobs over a800-2 (r_queue) and the a800-3/4/5 pools."""
import json, os
st = json.load(open('docs/rq_state.json'))['jobs']
started = lambda s, i: os.path.exists(f'data/v3_main/{s}/{i:06d}')
pend, keep = [], {}
for h in ('a3', 'a4', 'a5'):
    J = json.load(open(f'docs/v3n/jobs_{h}.json'))
    keep[h] = [j for j in J if started(j[1], j[2])]
    pend += [(j[1], j[2]) for j in J if not started(j[1], j[2])]
q = json.load(open('docs/v3n/queue.json')); keep_q = []
for j in q['jobs']:
    parts = j['id'].split('_'); s, i = parts[2], int(parts[3])
    if st.get(j['id'], {}).get('status') in (None, 'pending') and not started(s, i): pend.append((s, i))
    else: keep_q.append(j)
pend = sorted(set(pend), key=lambda x: ({'val': 0, 'test': 1, 'train': 2}[x[0]], x[1]))
w = {'a2': 8, 'a4': 8, 'a3': 7, 'a5': 8}; cyc = [h for h in w for _ in range(w[h])]
assign = {h: [] for h in w}
for k, x in enumerate(pend): assign[cyc[k % len(cyc)]].append(x)
for h in ('a3', 'a4', 'a5'):
    json.dump(keep[h] + [['data/v3_main', s, i] for s, i in assign[h]], open(f'docs/v3n/jobs_{h}.json', 'w'))
q['jobs'] = keep_q + [dict(id=f'v3n_gen_{s}_{i:06d}', kind='cmd', group='v3n-gen', mem_mib=7000, retries=1,
                           cmd=f'.venv/bin/python scripts/v3n/gen_v3.py --split {s} --indices {i} --root data/v3_main') for s, i in assign['a2']]
q['gpus'] = q.pop('gpus_saved', [0, 1, 2, 3, 4, 5, 6, 7])
json.dump(q, open('docs/v3n/queue.json', 'w'), indent=1)
print('pending', len(pend), {h: len(v) for h, v in assign.items()})
