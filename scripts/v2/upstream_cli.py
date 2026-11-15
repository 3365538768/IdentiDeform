"""Rebuild an upstream modified_simulation.py command line from a sample's meta/run_parameters.json.

Only keys that are argparse destinations are replayed; overrides replace them (P4 Data V2 pilot).
"""
import argparse
import json
import re
from pathlib import Path

SIM = Path(__file__).resolve().parents[2] / 'upstream/DeformFieldBench/simulation/modified_simulation.py'


def argparse_dests():
    src = SIM.read_text()
    return set(re.findall(r'add_argument\(\s*"--([A-Za-z0-9_]+)"', src))


def build(run_parameters, ply, config, output, overrides=None):
    dests = argparse_dests(); params = dict(run_parameters, **(overrides or {}))
    cli = ['--ply_path', str(ply), '--config', str(config), '--output_path', str(output)]
    for k, v in params.items():
        if k not in dests or k in ('ply_path', 'config', 'output_path'):
            continue
        if isinstance(v, bool):
            cli += [f'--{k}'] if v else []
        elif v is None or isinstance(v, (list, dict)):
            continue
        else:
            cli += [f'--{k}', str(v)]
    return cli


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--run-parameters'); a = p.parse_args()
    print(' '.join(build(json.loads(Path(a.run_parameters).read_text()), 'PLY', 'CFG', 'OUT')))
