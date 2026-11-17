"""One-line v3r/v3s status (for monitoring)."""
import json, glob, os
from pathlib import Path
R = Path('/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test')
st = json.loads((R / 'docs/rq_state.json').read_text())['jobs']
def q(prefix):
    c = {}
    for k, v in st.items():
        if k.startswith(prefix) and 'smoke' not in k:
            c[v['status']] = c.get(v['status'], 0) + 1
    return c
a4 = json.loads((R / 'docs/v3r/jobs_a4.json').read_text()) if (R / 'docs/v3r/jobs_a4.json').exists() else []
a4done = sum((R / 'runs/v3r_done' / j['id']).exists() for j in a4)
gtdone = len(glob.glob(str(R / 'data/v3r/runs/*/gt/state.json')))
out = [f"gt_complete={gtdone}", f"a4={a4done}/{len(a4)}", f"q_v3r={q('v3r_')}", f"q_v3s={q('v3s_')}",
       f"handoff={(R / 'docs/v3p/HANDOFF.json').exists()}", f"prior_ens={(R / 'docs/v3q/PRIOR_ENSEMBLE_val.json').exists()}",
       f"test_gen={len(glob.glob(str(R / 'data/v3_main/test/*/result.json')))//50*50}"]
v3q = [p for p in glob.glob(str(R / 'data/v3q_refine/val/*/prior_top2/state.json'))]
n12 = 0
for p in v3q:
    try:
        n12 += len(json.loads(Path(p).read_text()).get('evals', [])) >= 12
    except Exception:
        pass
out.append(f"v3q_top2_ge12={n12}")
b3 = 0
for p in glob.glob(str(R / 'data/v3q_refine/val/*/lawmean_best3/state.json')):
    try:
        b3 += len(json.loads(Path(p).read_text()).get('evals', [])) >= 12
    except Exception:
        pass
out.append(f"v3q_best3_ge12={b3}")
print(' '.join(out))
