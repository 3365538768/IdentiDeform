import sys, json, shutil, numpy as np
sys.path.insert(0, 'scripts/v3b')
import c1_refine as c
calls = []
def fake_sim(work, tag, params, action, obj, index, rp, keep=False):
    calls.append(tag)
    m = np.zeros((3, 8, 16, 16), bool); r = np.zeros((3, 8, 16, 16, 3), np.float32)
    if keep: np.savez_compressed(work / f'{tag}.npz', mask=np.packbits(m, -1), shape=np.array(m.shape), rgb=r.astype(np.uint8))
    return (m, r), 1.0
c.simulate = fake_sim
c.load_pack = lambda p: (np.zeros((3, 8, 16, 16), bool), np.zeros((3, 8, 16, 16, 3), np.float32))
d = 'data/v3b_refine_unittest2/v2main_C_seed29/000004'
shutil.rmtree('data/v3b_refine_unittest2', ignore_errors=True)
import os; os.makedirs(d)
shutil.copy('data/v3b_refine_test/v2main_C_seed29/000004/state.json', d + '/state.json')
sys.argv = ['x', '--run', 'v2main_C_seed29', '--index', '4', '--budget', '3', '--arm', 'prior', '--out', 'data/v3b_refine_unittest2', '--replay-budgets', '0,1,3']
c.main(); print('sims called', calls)
s = json.load(open(d + '/state.json')); print(s['replay'].keys(), [e['x'] for e in s['evals']])
