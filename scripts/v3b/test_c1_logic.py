import sys, json, numpy as np
sys.path.insert(0, 'scripts/v3b')
import c1_refine as c
rng = np.random.default_rng(0)
def fake_sim(work, tag, params, action, obj, index, rp, keep=False):
    m = np.zeros((3, 8, 16, 16), bool); r = np.zeros((3, 8, 16, 16, 3), np.float32)
    e = np.log10(params['E']); k = int(np.clip((e - 2) * 3, 0, 15)); m[:, :, :k] = True
    if params['material'] == 'metal' and e > 5.6: return None, 70.0
    if keep: np.savez_compressed(work / f'{tag}.npz', mask=np.packbits(m, -1), shape=np.array(m.shape), rgb=r.astype(np.uint8))
    return (m, r), 1.0
c.simulate = fake_sim
obs = np.zeros((3, 8, 16, 16), bool); obs[:, :, :9] = True
c.load_pack = lambda p: (obs, np.zeros((3, 8, 16, 16, 3), np.float32))
for arm, b in [('prior', 24), ('prior_sigma', 12), ('lawmean_gt', 24), ('lawmean_best3', 24)]:
    sys.argv = ['x', '--run', 'v2main_C_seed29', '--index', '4', '--budget', str(b), '--arm', arm, '--out', 'data/v3b_refine_unittest']
    c.main()
    s = json.load(open(f'data/v3b_refine_unittest/v2main_C_seed29/000004/{"" if arm=="prior" else arm+"/"}state.json'))
    print(arm, s['law'], s['initial_steps'], len(s['evals']), {k: (v['law'], v['x'], round(v['mae_z'], 3)) for k, v in s['budgets'].items()}, sorted(s['replay']))
sys.argv = ['x', '--run', 'v2main_C_seed29', '--index', '4', '--budget', '48', '--arm', 'lawmean_best3', '--out', 'data/v3b_refine_unittest', '--replay-budgets', '12,24,48']
c.main()
