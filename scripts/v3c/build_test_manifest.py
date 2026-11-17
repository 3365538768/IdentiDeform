"""V3c: test-capable V2 main manifest without touching data/manifests/v2main.jsonl.

data/manifests/v2main_withtest.jsonl = exact bytes of v2main.jsonl + one row per checked test sample,
built with the same row logic as scripts/v2/build_v2_manifest.py (name 'v2main', so sample_id
'v2main/test/NNNNNN'). Stats siblings v2main_withtest{,.specific}.stats.json are recomputed from the
(identical) train rows; every field except manifest_sha256 is asserted equal to the frozen
v2main stats, so models trained on v2main see the same codec. Test rows use only test packs passed
by scripts/v3c/check_test_packs.py.
  .venv/bin/python scripts/v3c/build_test_manifest.py
"""
import hashlib, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'v2'))
from build_v2_manifest import stats

BASE = Path('data/manifests/v2main.jsonl'); NEW = Path('data/manifests/v2main_withtest.jsonl')
check = json.loads(Path('docs/v3c/test_pack_check.json').read_text())
ok = {r['dir'] for r in check['samples'] if r['ok']}
base_bytes = BASE.read_bytes(); base_rows = [json.loads(l) for l in base_bytes.decode().splitlines()]
rows = []
for d in sorted(Path('data/v2_main/test').glob('[0-9]*')):
    if d.name not in ok:
        continue
    packs = list(d.glob('*/sample_pack.npz')); gt = json.loads((d / 'gt.json').read_text())
    rp = json.loads((packs[0].parent / 'meta/run_parameters.json').read_text())
    idx = rp['render_frame_indices']; dt = float(rp['time_params']['frame_dt'])
    sigma = gt['regression']['yield_stress'] if gt['law'] != 'jelly' and not gt['labels']['censored_sigma'] else None
    rows.append({'sample_id': f'v2main/test/{d.name}', 'object_id': gt['object'], 'asset_group': 'deformfield:' + gt['object'],
                 'source': 'DeformFieldBench-v2main', 'pack': str(packs[0].resolve()),
                 'parameters': {'E': gt['regression']['E'], 'nu': gt['regression']['nu'], 'rho': gt['regression']['density'], 'sigma_y': sigma},
                 'law_id': gt['law_id'], 'action': gt['action'], 'field_mode': 'legacy_visual',
                 'label_provenance': 'simulator_config', 'field_warning': 'visualized flow/stress; contact mask is not force',
                 'timestamps_seconds': [(i - idx[0]) * dt for i in idx], 'time_provenance': 'render_frame_indices * frame_dt',
                 'published_split': 'test', 'split': 'test', 'sigma_censored': gt['labels']['censored_sigma'], 'damping': gt['damping']})
# key order / format must match the existing rows
assert list(rows[0]) == list(base_rows[0]), (list(rows[0]), list(base_rows[0]))
assert Path(base_rows[0]['pack']).is_absolute()
NEW.write_bytes(base_bytes + ''.join(json.dumps(r) + '\n' for r in rows).encode())
assert NEW.read_bytes()[:len(base_bytes)] == base_bytes
sha = hashlib.sha256(NEW.read_bytes()).hexdigest()
train = [r for r in base_rows if r['split'] == 'train']
report = {'base_manifest': str(BASE), 'base_sha256': hashlib.sha256(base_bytes).hexdigest(), 'manifest': str(NEW), 'sha256': sha,
          'counts': {s: sum(r['split'] == s for r in base_rows + rows) for s in ('train', 'val', 'test')}}
for suffix, specific in (('.stats.json', False), ('.specific.stats.json', True)):
    old = json.loads(Path(str(BASE).replace('.jsonl', suffix)).read_text())
    new = stats(train, specific, sha)
    diff = [k for k in set(old) | set(new) if k != 'manifest_sha256' and old.get(k) != new.get(k)]
    if diff:
        raise SystemExit('stats differ: ' + str(diff))
    out = dict(old, manifest_sha256=sha)  # keep the frozen file's exact key order/values
    path = Path(str(NEW).replace('.jsonl', suffix)); path.write_text(json.dumps(out, indent=1))
    report[suffix] = {'path': str(path), 'equal_to_frozen_except_manifest_sha256': True}
train_val_objects = {r['object_id'] for r in base_rows}; test_objects = {r['object_id'] for r in rows}
report['test_objects'] = len(test_objects); report['test_objects_overlapping_train_val'] = sorted(test_objects & train_val_objects)
report['test_law_counts'] = {str(k): sum(r['law_id'] == k for r in rows) for k in (0, 1, 2)}
Path('docs/v3c/test_manifest.json').write_text(json.dumps(report, indent=1))
print(json.dumps(report, indent=1))
