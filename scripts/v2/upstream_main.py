"""Run upstream modified_simulation.py as __main__, optionally replacing the press action.

FB_ACTION=tap: plate compresses 10 % of the z extent over 0.5 s, then releases (3.5 s free motion).
FB_ACTION=tap30hold: 30 % over 0.5 s, plate held still until 1.0 s, then released (3 s free motion).
FB_BC_FIX=1 (V3 data): plates act by contact from outside the object and the object rests on a floor (scripts/v2/contact_bc.py).
"""
import os
import runpy
import sys
from pathlib import Path

import my_utils.sim_utils as sim_utils

def skip_intermediate_png_recompression(argv):
    # V2 main deletes these PNGs after packing. Re-encoding them changes no pixels.
    if '--output_path' not in argv or '--pack_sample_pack' not in argv or '--compile_video' in argv:
        return
    output_index = argv.index('--output_path') + 1
    if output_index >= len(argv):
        return
    dataset = Path(__file__).resolve().parents[2] / 'data/v2_main'
    if not any(Path(argv[output_index]).resolve().is_relative_to(dataset.parent / d) for d in ('v2_main', 'v3_main')):
        return
    if '--compress_render_pngs' in argv:
        argv.remove('--compress_render_pngs')
    if '--no_compress_render_pngs' not in argv:
        argv.append('--no_compress_render_pngs')



if os.environ.get('FB_BC_FIX') == '1':  # V3 contact setup, see scripts/v2/contact_bc.py
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import contact_bc
    contact_bc.install()

if os.environ.get('FB_ACTION') == 'tap' and os.environ.get('FB_BC_FIX') != '1':
    press = sim_utils.build_press_boundary_conditions

    def tap(mpm_init_pos, material_params):
        bcs = press(mpm_init_pos, material_params)
        plate = next(b for b in bcs if b['type'] == 'cuboid')
        plate['velocity'] = [0.0, 0.0, plate['velocity'][2] * (0.1 / 0.7) * (3.0 / 0.5)]
        plate['end_time'] = 0.5
        return bcs

    sim_utils.build_press_boundary_conditions = tap
elif os.environ.get('FB_ACTION') == 'tap30hold' and os.environ.get('FB_BC_FIX') != '1':  # V3: contact_bc.press handles it
    press = sim_utils.build_press_boundary_conditions

    def tap30hold(mpm_init_pos, material_params):
        bcs = press(mpm_init_pos, material_params)
        plate = next(b for b in bcs if b['type'] == 'cuboid')
        vz = plate['velocity'][2] * (0.3 / 0.7) * (3.0 / 0.5)
        plate['velocity'] = [0.0, 0.0, vz]; plate['end_time'] = 0.5
        held = dict(plate, point=[plate['point'][0], plate['point'][1], plate['point'][2] + 0.5 * vz],
                    velocity=[0.0, 0.0, 0.0], start_time=0.5, end_time=1.0)
        return bcs + [held]

    sim_utils.build_press_boundary_conditions = tap30hold
sys.argv = ['modified_simulation.py', *sys.argv[1:]]
skip_intermediate_png_recompression(sys.argv)
runpy.run_path('modified_simulation.py', run_name='__main__')
