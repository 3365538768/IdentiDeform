"""Force-controlled press subset (V5b setup, plate force F=30, contact forces exported): the press samples of
val 0-282 and train 0-1999 with the same parameters/objects as data/v5b_main, written to data/v5f_main."""
import json
A = ['press', 'drop', 'shear', 'stretch', 'bend', 'tap30hold']
VF = ['--g-all', '-2', '--weak', 'hold', '--ranges', 'v5', '--press-force', '30']
jobs = [['data/v5f_main', s, i, VF] for s, n in (('val', 283), ('train', 2000)) for i in range(n) if A[(i // 3) % 6] == 'press']
H = {'a3': [], 'a5': []}
for k, j in enumerate(jobs): H['a3' if k % 2 == 0 else 'a5'].append(j)
for h, v in H.items(): json.dump(v, open(f'docs/v5/jobs_force_{h}.json', 'w'))
print(len(jobs), {h: len(v) for h, v in H.items()})
