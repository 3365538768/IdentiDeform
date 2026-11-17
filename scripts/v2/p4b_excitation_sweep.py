"""P4b: does a release-dominated excitation with low damping make E/rho (or sigma_y/rho) visible?

Same cube and adjacent-value metrics as scripts/r2b_mpm_sweep.py, but with
  action 'press'   : V1 press (70 % over 3 s, release at 3 s, 4 s total)
  action 'tap'     : press 10 % over 0.5 s, release, 3.5 s free motion
and damping 'v1' (per-family dataset values; grid damping is applied every substep, i.e. x0.999^10000
per second for jelly) or 'low' (rpic 0, grid 0.99999 per substep, ~10 %/s, identical for all families).

  PYTHONPATH=.venv/lib/python3.12/site-packages .engine-envs/warp/bin/python scripts/v2/p4b_excitation_sweep.py --material jelly --sweep E --damping low
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import r2_mpm_identifiability as r2  # noqa: E402
from r2b_mpm_sweep import PX_PER_UNIT, dominant_hz, shape_signals  # noqa: E402

PRESS = r2.build_press_boundary_conditions
DAMPING = {'v1': {}, 'low': {'rpic_damping': 0.0, 'grid_v_damping_scale': 0.99999}}


def tap_boundary_conditions(x, params):
    bcs = PRESS(x, params)
    plate = next(b for b in bcs if b['type'] == 'cuboid')
    extent = float((x[:, 2].max() - x[:, 2].min()).item())
    plate['velocity'] = [0.0, 0.0, -0.1 * extent / 0.5]; plate['end_time'] = 0.5
    return bcs


def main():
    p = argparse.ArgumentParser(); p.add_argument('--material', required=True)
    p.add_argument('--sweep', choices=['E', 'yield_stress'], required=True); p.add_argument('--damping', choices=list(DAMPING), required=True)
    p.add_argument('--actions', default='press,tap'); p.add_argument('--frame-dt', type=float, default=1 / 64)
    p.add_argument('--substep-dt', type=float, default=1e-4); a = p.parse_args()
    r2.wp.init(); out = r2.ROOT / f'docs/p4/P4B_SWEEP_{a.material}_{a.sweep}_{a.damping}.json'
    rho = r2.REFERENCE[a.material]['density']
    ranges = {('jelly', 'E'): (2, 6), ('plasticine', 'E'): (2, 6), ('metal', 'E'): (3.5, 6),
              ('plasticine', 'yield_stress'): (0, 6), ('metal', 'yield_stress'): (0, 5)}
    groups = 10. ** np.arange(*ranges[(a.material, a.sweep)], .5)
    groups = np.append(groups, 10. ** ranges[(a.material, a.sweep)][1])
    report = {'material': a.material, 'sweep': a.sweep, 'damping': a.damping, 'damping_overrides': DAMPING[a.damping],
              'specific_values': groups.tolist(), 'frame_dt': a.frame_dt, 'substep_dt': a.substep_dt, 'actions': {}}
    for action in a.actions.split(','):
        r2.build_press_boundary_conditions = tap_boundary_conditions if action == 'tap' else PRESS
        runs = []
        for g in groups:
            try:
                traj = r2.simulate(a.material, {a.sweep: float(g * rho / 1000.), **DAMPING[a.damping]}, 'press', 4.0, a.frame_dt, a.substep_dt)
            except FloatingPointError:
                traj = None
            runs.append(None if traj is None else shape_signals(traj))
            print(json.dumps({'action': action, 'value': float(g), 'ok': traj is not None}), flush=True)
        r2.build_press_boundary_conditions = PRESS
        n = 1 + int(round(4.0 / a.frame_dt)); idx = {f: np.linspace(0, n - 1, int(4 * f)).round().astype(int) for f in (4, 8, 16, 32)}
        rows = []
        for i, g in enumerate(groups):
            row = {'value': float(g), 'ok': runs[i] is not None}
            if runs[i] is not None:
                row.update(dominant_height_hz=dominant_hz(runs[i][1], a.frame_dt), height_change_px=float(np.ptp(runs[i][1]) * PX_PER_UNIT))
            if i + 1 < len(runs) and runs[i] is not None and runs[i + 1] is not None:
                diff = np.sqrt(((runs[i][0] - runs[i + 1][0]) ** 2).sum(-1).mean(1)) * PX_PER_UNIT
                row.update(next_value=float(groups[i + 1]), diff_px_all=float(diff.mean()), diff_px_max=float(diff.max()),
                           **{f'diff_px_{f}fps': float(diff[k].mean()) for f, k in idx.items()})
            rows.append(row)
        report['actions'][action] = rows; out.write_text(json.dumps(report, indent=2))
    report['status'] = 'complete'; out.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
