"""Redistribute not-yet-started V3 generation jobs over a800-2 (r_queue), a800-4, a800-3, a800-5 (pools)."""
import json, os
st = json.load(open('docs/rq_state.json'))['jobs']
a4 = json.load(open('docs/v3n/jobs_a4.json')); q = json.load(open('docs/v3n/queue.json'))
log = open('docs/v3n/logs/pool_a4.log').read()
def started(s, i):
    return os.path.exists(f'data/v3_main/{s}/{i:06d}') or f'"{s}", {i}]' in log
pend = [(s, i) for r, s, i in a4 if not started(s, i)]
keep_q = []
for j in q['jobs']:
    parts = j['id'].split('_'); s, i = parts[2], int(parts[3])
    if st.get(j['id'], {}).get('status') in (None, 'pending') and not started(s, i): pend.append((s, i))
    else: keep_q.append(j)
rank = {'val': 0, 'test': 1, 'train': 2}
pend.sort(key=lambda x: (rank[x[0]], x[1]))
hosts = ['a2', 'a4', 'a3', 'a5']; w = {'a2': 8, 'a4': 8, 'a3': 7, 'a5': 8}
assign = {h: [] for h in hosts}; cyc = [h for h in hosts for _ in range(w[h])]
for k, x in enumerate(pend): assign[cyc[k % len(cyc)]].append(x)
a4_running = [j for j in a4 if started(j[1], j[2])]
json.dump(a4_running + [['data/v3_main', s, i] for s, i in assign['a4']], open('docs/v3n/jobs_a4.json', 'w'))
for h in ('a3', 'a5'):
    json.dump([['data/v3_main', s, i] for s, i in assign[h]], open(f'docs/v3n/jobs_{h}.json', 'w'))
q['jobs'] = keep_q + [dict(id=f'v3n_gen_{s}_{i:06d}', kind='cmd', group='v3n-gen', mem_mib=7000, retries=1,
                           cmd=f'.venv/bin/python scripts/v3n/gen_v3.py --split {s} --indices {i} --root data/v3_main') for s, i in assign['a2']]
q['gpus'] = q.pop('gpus_saved', [0, 1, 2, 3, 4, 5, 6, 7])
q['note'] = 'V3 main data generation (a800-2 share); a800-4/a800-3/a800-5 pools use docs/v3n/jobs_a{4,3,5}.json'
json.dump(q, open('docs/v3n/queue.json', 'w'), indent=1)
print('pending', len(pend), {h: len(v) for h, v in assign.items()}, 'train<2000 pending', sum(1 for s, i in pend if s == 'train' and i < 2000))
