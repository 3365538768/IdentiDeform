"""V3s: queue (docs/v3s/queue.json, shared r_queue on a800-2) for the prior-dependent transfer runs with the V3 prior.
Same commands as the v3k-A / v3l jobs (docs/v3k/queue.json, docs/v3l/queue.json); only the prior file and output dirs differ.
Prior-independent rows (Spring-Gaus / PhysTwin official, GT-E replay, static / rigid-follow, uninformed jelly-mean starts)
are NOT rerun; scripts/v3s/report.py reads them from v3k / v3l.

  python3 scripts/v3s/make_queue.py      (requires docs/v3s/PRIOR_SOURCES.json)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ZIP = 'data/external_datasets/springgaus_gdrive/mpm_synthetic.zip'
PT_CASE = 'data/external_datasets/v3g_phystwin/single_push_sloth'
PT_RGB = 'data/external_datasets/v3l_phystwin_rgb/single_push_sloth'
SCENES = ['torus', 'duck', 'apple', 'C', 'E', 'V', 'banana', 'chess', 'cream', 'cross', 'doraemon', 'droplet', 'rope', 'toothpaste']
RE = 'scripts/v3f/replay_env.sh'


def job(i, cmd, after=(), mem=3000):
    return {'id': i, 'kind': 'cmd', 'group': 'v3s', 'mem_mib': mem, 'retries': 1, 'cmd': cmd, **({'after': list(after)} if after else {})}


def main():
    assert (ROOT / 'docs/v3s/PRIOR_SOURCES.json').exists()
    J = [job('v3s_prior_sg', f'mkdir -p data/v3s && .venv/bin/python -I scripts/v3s/prior_v3.py sg {ZIP} data/v3s/prior20.json', mem=8000),
         job('v3s_prior_pt', f'mkdir -p data/v3s/pt && .venv/bin/python -I scripts/v3s/prior_v3.py pt {PT_RGB} data/v3s/pt/pt_prior.json', mem=8000)]
    for s in SCENES:
        J.append(job(f'v3s_kA_{s}', f'{RE} scripts/v3k/a_sg_refine.py {ZIP} {s} --arm prior --prior-json data/v3s/prior20.json --out data/v3s/kA', ['v3s_prior_sg']))
        J.append(job(f'v3s_ref_{s}', f'{RE} scripts/v3l/sg_metric.py {ZIP} {s} --arm prior20_refine --prior data/v3s/prior20.json --out data/v3s/sg', ['v3s_prior_sg']))
        J.append(job(f'v3s_p20_{s}', f'{RE} scripts/v3l/sg_metric.py {ZIP} {s} --arm prior20 --prior data/v3s/prior20.json --out data/v3s/sg', ['v3s_prior_sg']))
    J.append(job('v3s_pt_search_prior', f'{RE} scripts/v3l/pt_mpm.py {PT_CASE} --mode search --budget 12 --start prior --out data/v3s/pt', ['v3s_prior_pt']))
    J.append(job('v3s_pt_eval', f'bash scripts/v3s/pt_eval_run.sh {PT_CASE}', ['v3s_pt_search_prior']))
    qp = ROOT / 'docs/v3s/queue.json'
    q = json.loads(qp.read_text()) if qp.exists() else {}
    old = [j for j in q.get('jobs', []) if j['id'].startswith('v3s_smoke')]
    q.update(phase='v3s', max_per_gpu=6, gpus=[0, 1, 2, 3, 4, 5, 6, 7], note='V3s transfer with the V3 prior (owner: v3r/v3s agent)', jobs=old + J)
    qp.write_text(json.dumps(q, indent=1)); print(len(J), 'jobs')


if __name__ == '__main__':
    main()
