"""V3q one-line status: tasks done/running per phase, evaluations, failures, errors, hand-off presence."""
import json, collections, glob, os
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
T = json.loads((ROOT / 'docs/v3q/tasks.json').read_text())['tasks']
done = {t['id'] for t in T if (ROOT / 'runs/v3q_done' / t['id']).exists()}
c = collections.Counter((t['phase'], t['id'] in done) for t in T if not t.get('hold'))
ev = fail = 0
for f in glob.glob(str(ROOT / 'data/v3q_refine/*/*/*/state.json')):
    try:
        s = json.loads(Path(f).read_text())
    except ValueError:
        continue
    ev += len(s.get('evals', [])); fail += sum(bool(e.get('failed')) for e in s.get('evals', []))
cache = len(glob.glob(str(ROOT / 'data/v3q_refine/*/*/_cache/*.json')))
errs = [os.path.basename(f) for f in glob.glob(str(ROOT / 'docs/v3q/logs/v3q_*.log')) if 'Traceback' in Path(f).read_text(errors='replace')]
held = sum(1 for t in T if t.get('hold'))
print(f"done/runnable {dict(sorted(c.items()))} held {held} | evals {ev} unique_sims {cache} failed {fail} | errors {errs} | handoff {(ROOT / 'docs/v3p/HANDOFF.json').exists()}")
tr = glob.glob(str(ROOT / 'data/v3_main/train/*/result.json'))
lo = [f for f in tr if int(f.split('/')[-2]) < 2000]
print(f"train results <2000: {len(lo)} (all {len(tr)})")
