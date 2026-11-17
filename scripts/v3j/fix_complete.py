"""V3j: restore state['complete'] on Exp-2 states whose search finished (budget_run set) but whose flag was dropped by
the --replay pass of the first c1j_refine.py version (bug fixed 04:05 UTC; the search results were unaffected)."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
n = 0
for f in (ROOT / 'data/v3j_refine').glob('u4_*/*/state.json'):
    s = json.loads(f.read_text())
    if 'budget_run' in s and 'budgets' in s and not s.get('complete'):
        s['complete'] = True; tmp = f.with_suffix('.fix'); tmp.write_text(json.dumps(s, indent=1)); os.replace(tmp, f); n += 1
print('restored', n)
