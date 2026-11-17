import json
R = '/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test/'
S = json.load(open(R + 'docs/v5r/SAMPLES.json'))['indices']
C = 'end:0.03,end:0.06,end:0.10,end:0.16,end:0.25,end:0.40,centre:0.03,centre:0.06,centre:0.10,centre:0.16,centre:0.25,centre:0.40'
jobs = []
for m in ('gt', 'prior_vt', 'prior_vision', 'lawmean_pred'):
    for i in S:
        for c in ['none:0'] + C.split(','):
            jobs.append({'id': f"v5g_{m}_{i:06d}_{c.replace(':', '_')}",
                         'cmd': f'export FB_GRASP_CANDS={C} && .venv/bin/python scripts/v5g/grasp_run.py --index {i} --method {m} --cands {c} --out data/v5g/runs'})
json.dump(jobs, open(R + 'docs/v5g/jobs_main.json', 'w'), indent=1); print(len(jobs))
