"""V3c: run the frozen scripts/v3c/simple_baselines.py on the test split without editing it.

Only its MANIFEST constant is re-pointed to data/manifests/v2main_withtest.jsonl (train rows are
byte-identical, so the train means / majority law are unchanged; asserted). Outputs:
runs/v3c_const_test, runs/v3c_lawmean_gt_test.
  .venv/bin/python scripts/v3c/simple_baselines_test.py
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import simple_baselines as sb

for d in ('runs/v3c_const_test', 'runs/v3c_lawmean_gt_test'):
    if Path(d).exists():
        raise SystemExit('exists: ' + d + ' (test is evaluated once)')
val_means = json.loads(Path('runs/v3c_lawmean_gt/law_means.json').read_text())
sb.MANIFEST = 'data/manifests/v2main_withtest.jsonl'
sys.argv = [sys.argv[0], '--split', 'test']
sb.main()
test_means = json.loads(Path('runs/v3c_lawmean_gt_test/law_means.json').read_text())
assert test_means == val_means, 'train law means changed'
print('train statistics identical to val run')
