"""V3e: choose evaluation samples (stratified by GT law from docs/v3b/C1_SAMPLES.json) and write docs/v3e/jobs.json.

  .venv/bin/python scripts/v3e/make_jobs.py --wave 1    (10 plasticine + 10 metal + 6 jelly, seeded random)
  .venv/bin/python scripts/v3e/make_jobs.py --wave 2    (adds all remaining C1 samples)
"""
import argparse
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
METHODS = ['gt', 'refined_top2', 'ensemble_prior', 'refined_best3', 'lawmean_pred']
QUOTA = {'plasticine': 10, 'metal': 10, 'jelly': 6}


def main():
    a = argparse.ArgumentParser(); a.add_argument('--wave', type=int, default=1); args = a.parse_args()
    c1 = json.loads((ROOT / 'docs/v3b/C1_SAMPLES.json').read_text())['samples']
    rng = random.Random(20261114); wave1 = []
    for law, k in QUOTA.items():
        pool = sorted(s['index'] for s in c1 if s['law'] == law)
        wave1 += sorted(rng.sample(pool, k))
    chosen = wave1 if args.wave == 1 else wave1 + sorted(s['index'] for s in c1 if s['index'] not in wave1)
    (ROOT / 'docs/v3e').mkdir(parents=True, exist_ok=True)
    (ROOT / 'docs/v3e/SAMPLES.json').write_text(json.dumps({'seed': 20261114, 'quota_wave1': QUOTA, 'wave1': wave1,
                                                             'wave': args.wave, 'indices': chosen}, indent=1))
    jobs = [{'id': f'v3e_{m}_{i:06d}', 'cmd': f'.venv/bin/python scripts/v3e/squeeze_run.py --index {i} --method {m}'}
            for i in chosen for m in METHODS]
    (ROOT / 'docs/v3e/jobs.json').write_text(json.dumps(jobs, indent=1))
    print(len(chosen), 'samples', len(jobs), 'jobs', chosen)


if __name__ == '__main__':
    main()
