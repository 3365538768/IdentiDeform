"""One-off: remove intermediate PNG sequences of finished P4c samples (pack + exit 0 present)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_v2 import ROOT, prune_pngs  # noqa: E402

n = 0
for result in sorted(ROOT.glob('data/v2_pilot/p4c_*/*/*/result.json')):
    if json.loads(result.read_text()).get('exit') == 0 and list(result.parent.glob('*/sample_pack.npz')):
        prune_pngs(result.parent); n += 1
print('pruned', n)
