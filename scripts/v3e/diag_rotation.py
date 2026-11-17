import json, glob, numpy as np, sys
def proc(src, dst):
    ms, md = src.mean(0), dst.mean(0)
    u, _, vt = np.linalg.svd((src - ms).T @ (dst - md))
    s = np.eye(3); s[2, 2] = np.sign(np.linalg.det(u @ vt)); r = (u @ s @ vt).T
    return (src - ms) @ r.T + md, r
for t in sorted(glob.glob('data/v3e/runs/*/gt/twin.json')):
    i = t.split('/')[3]; c = json.load(open(t))['cache']; base = f'data/v3e/runs/{i}/cache/{c}'
    try:
        ref = np.load(f'{base}/0.000/subset.npz')
    except Exception:
        continue
    st = json.load(open(f'{base}/state.json')); law = json.load(open(f'data/v2_main/val/{i}/gt.json'))['law']
    out = []
    for d in ['0.060', '0.140', '0.240', '0.320']:
        r = st['runs'].get(d)
        if not r or 'metrics' not in r: out.append('   --   '); continue
        s = np.load(f'{base}/{d}/subset.npz'); bc = json.load(open(f'{base}/{d}/squeeze_bc.json')); a = bc['axis']; W = bc['width']
        al, R = proc(s['x_tail'].astype(float), ref['x_tail'].astype(float))
        ang = np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))
        span = lambda x: np.subtract(*np.quantile(x[:, a], [.99, .01]))
        res = np.linalg.norm(al - ref['x_tail'], axis=1)
        out.append(f"w{r['metrics']['wloss']:.3f}/a{(span(ref['x_tail'])-span(al))/W:.3f}/p99 {np.quantile(res,.99)/W:.3f}/rot{ang:.0f}")
    print(i, law[:5], ' | '.join(out))
