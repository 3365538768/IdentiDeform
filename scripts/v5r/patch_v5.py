import shutil
from pathlib import Path
R = Path('/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test')
bak = R / '.cache/keep/v3e_before_v5r'; bak.mkdir(parents=True, exist_ok=True)
for n in ('squeeze_sim', 'plan_eval'):
    if not (bak / f'{n}.py').exists(): shutil.copy2(R / f'scripts/v3e/{n}.py', bak / f'{n}.py')
def sub(p, old, new):
    p = R / p; t = p.read_text(); assert t.count(old) == 1, (p, old); p.write_text(t.replace(old, new))
sub('scripts/v3e/squeeze_sim.py', "    if TWINS == 'v3':\n        sys.path.insert(0, str(ROOT / 'scripts/v3r'))\n",
    "    if TWINS in ('v3', 'v5'):  # V3r / V5r twin modules\n        sys.path.insert(0, str(ROOT / ('scripts/v3r' if TWINS == 'v3' else 'scripts/v5r')))\n")
sub('scripts/v3e/plan_eval.py', "METHODS = ['gt', 'refined_top2', 'ensemble_prior', 'refined_best3', 'fixed_loo', 'lawmean_pred']\n",
    "METHODS = ['gt', 'refined_top2', 'ensemble_prior', 'refined_best3', 'fixed_loo', 'lawmean_pred']\n"
    "import os as _os\nif _os.environ.get('FB_GRIP_METHODS'):  # V5r: method list (default = V3e)\n    METHODS = _os.environ['FB_GRIP_METHODS'].split(',')\n"
    "PAIRED_BASE = _os.environ.get('FB_GRIP_PAIRED', 'ensemble_prior,lawmean_pred,fixed_loo,refined_best3').split(',')\n")
sub('scripts/v3e/plan_eval.py', "         'ensemble_prior': 'ensemble prior, no refinement (0 sims)',",
    "         'ensemble_prior': 'ensemble prior, no refinement (0 sims)', 'prior_vision': '(b) vision prior (9-run ensemble, 0 sims)',\n"
    "         'prior_vt': '(c) vision + touch prior (3-seed ensemble, 0 sims)', 'refined_vt': '(e) vision + touch prior + 12-sim refinement',")
sub('scripts/v3e/plan_eval.py', "        for b in ('ensemble_prior', 'lawmean_pred', 'fixed_loo', 'refined_best3'):", "        for b in PAIRED_BASE:")
print('ok')
