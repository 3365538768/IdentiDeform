from pathlib import Path
p = Path('/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test/scripts/v3e/squeeze_sim.py'); t = p.read_text()
old = "DEPTHS = [0.06, 0.10, 0.14, 0.18, 0.24, 0.32]   # candidate closing depths (fraction of width W)\n"
assert t.count(old) == 1
new = old + ("if os.environ.get('FB_GRIP_DEPTHS'):  # V3r: extended grid (docs/v3r/PREREG_GRID_2026-11-15.md); default = V2 grid\n"
             "    DEPTHS = [float(v) for v in os.environ['FB_GRIP_DEPTHS'].split(',')]\n")
p.write_text(t.replace(old, new)); print('ok')
