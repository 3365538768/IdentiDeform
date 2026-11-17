"""P4: clip the densify_grids stencil to the grid (exact: out-of-grid cells were skipped anyway).

Degenerate huge-covariance gaussians otherwise make the taichi kernel loop for hours.
"""
from pathlib import Path
p = Path(__file__).resolve().parents[2] / 'upstream/DeformFieldBench/simulation/particle_filling/filling.py'
s = p.read_text()
old = """        r = ti.ceil(r / grid_dx, dtype=int)
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):"""
new = """        r = ti.ceil(r / grid_dx, dtype=int)
        # FieldBridge P4 patch: stencil clipped to the grid (scripts/v2/patch_filling.py).
        for dx in range(ti.max(-r, -i), ti.min(r, grid_density.shape[0] - 1 - i) + 1):
            for dy in range(ti.max(-r, -j), ti.min(r, grid_density.shape[1] - 1 - j) + 1):
                for dz in range(ti.max(-r, -k), ti.min(r, grid_density.shape[2] - 1 - k) + 1):"""
if 'FieldBridge P4 patch' not in s:
    assert s.count(old) == 1
    p.write_text(s.replace(old, new))
print('patched')
