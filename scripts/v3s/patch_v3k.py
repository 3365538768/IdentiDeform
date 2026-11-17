"""V3s: let scripts/v3k/a_sg_refine.py take the zero-shot prior from a prior20.py-style json (V3 prior, 30-frame
'f30_recomputed' entry = the v3f K0 setting) instead of docs/v3f/SPRINGGAUS_SYN.json. Default behaviour unchanged."""
import shutil
from pathlib import Path

R = Path('/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test')
bak = R / '.cache/keep/v3k_before_v3s'; bak.mkdir(parents=True, exist_ok=True)
if not (bak / 'a_sg_refine.py').exists():
    shutil.copy2(R / 'scripts/v3k/a_sg_refine.py', bak / 'a_sg_refine.py')
p = R / 'scripts/v3k/a_sg_refine.py'; t = p.read_text()
old = "    prior_log10E, prior_nu = (log10E_from_z(zp[0]), float(nu_from_z(zp[1]))) if zp else (None, None)\n"
new = old + ("    if a.prior_json:  # V3s: prior from scripts/v3s/prior_v3.py output (30 frames, same t0 triplet / K0 path)\n"
             "        pj = json.loads(Path(a.prior_json).read_text())['scenes'][a.scene]['f30_recomputed']\n"
             "        prior_log10E, prior_nu = float(pj['log10_E_Pa_at_rho1000']), float(pj['nu'])\n")
assert t.count(old) == 1; t = t.replace(old, new)
old = "p.add_argument('--budget', type=int, default=12); p.add_argument('--out', default=str(ROOT / 'data/v3k/sg'))"
assert t.count(old) == 1
t = t.replace(old, old + "\n    p.add_argument('--prior-json', default=None)")
p.write_text(t); print('patched')
