"""V3r: append twin squeeze jobs whose parameter source is ready to the a800-4 pool list docs/v3r/jobs_a4c.json.
  ensemble_prior, lawmean_pred : docs/v3q/PRIOR_ENSEMBLE_val.json exists
  refined_top2 / refined_best3 : data/v3q_refine/val/<i>/{prior_top2,lawmean_best3}/state.json has budgets['12'] or >= 12 evals
  python3 scripts/v3r/add_ready.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV = 'export FB_GRIP_DATA=data/v3_main/val FB_GRIP_TWINS=v3 FB_SQUEEZE_CLAMP=1'
ARM = {'refined_top2': 'prior_top2', 'refined_best3': 'lawmean_best3'}


def ready(i, m):
    if m in ('ensemble_prior', 'lawmean_pred'):
        return (ROOT / 'docs/v3q/PRIOR_ENSEMBLE_val.json').exists()
    p = ROOT / f'data/v3q_refine/val/{i:06d}/{ARM[m]}/state.json'
    try:
        st = json.loads(p.read_text())
    except (OSError, ValueError):
        return False
    return '12' in st.get('budgets', {}) or len(st.get('evals', [])) >= 12


def main():
    idx = json.loads((ROOT / 'docs/v3r/SAMPLES.json').read_text())['indices']
    lists = [ROOT / f'docs/v3r/{n}' for n in ('jobs_a4.json', 'jobs_a4b.json', 'jobs_a4c.json')]
    have = {j['id'] for p in lists if p.exists() for j in json.loads(p.read_text())}
    have |= {j['id'] for j in json.loads((ROOT / 'docs/v3r/queue.json').read_text())['jobs']}
    c = ROOT / 'docs/v3r/jobs_a4c.json'; jobs = json.loads(c.read_text()) if c.exists() else []
    new = []
    for m in ('ensemble_prior', 'lawmean_pred', 'refined_best3', 'refined_top2'):
        for i in idx:
            jid = f'v3r_{m}_{i:06d}'
            if jid not in have and ready(i, m):
                new.append({'id': jid, 'cmd': f'{ENV} && .venv/bin/python scripts/v3e/squeeze_run.py --index {i} --method {m} --out data/v3r/runs'})
    c.write_text(json.dumps(jobs + new, indent=1))
    done = sum((ROOT / 'runs/v3r_done' / j['id']).exists() for j in jobs + new)
    print(f'added {len(new)}; list {len(jobs) + len(new)}; done {done}')


if __name__ == '__main__':
    main()
