"""V3a: manifest + train statistics for the libuipc few-shot pilot (data/v3_uipc).

  .venv/bin/python scripts/v3/build_uipc_manifest.py
Writes data/manifests/v3_uipc.jsonl, .stats.json (native) and .specific.stats.json (E/(rho/1000)).
Rows follow the v2main row format (legacy_visual packs; flow/stress/force_mask are zero placeholders,
NOT field supervision). Only cases with result.json exit==0 are kept. No test split (val only).
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v2'))
from build_v2_manifest import stats  # noqa: E402

NAME = 'v3_uipc'


def main():
    root = ROOT / 'data/v3_uipc'
    rows, skipped = [], []
    for split in ('train', 'val'):
        for d in sorted((root / split).glob('[0-9]*')):
            res = d / 'result.json'
            if not res.exists() or json.loads(res.read_text()).get('exit') != 0 or not (d / 'sample_pack.npz').exists():
                skipped.append(str(d.relative_to(ROOT))); continue
            gt = json.loads((d / 'gt.json').read_text()); case = gt['case']
            rows.append({'sample_id': f'{NAME}/{split}/{d.name}', 'object_id': gt['object'], 'asset_group': 'uipc:' + gt['geometry'],
                         'source': 'FieldBench-v3-libuipc', 'pack': str(d / 'sample_pack.npz'),
                         'parameters': {'E': gt['regression']['E'], 'nu': gt['regression']['nu'], 'rho': gt['regression']['density'],
                                        'sigma_y': None},
                         'law_id': 0, 'action': gt['action'], 'field_mode': 'legacy_visual', 'geometry': gt['geometry'],
                         'label_provenance': 'simulator_config', 'engine': 'libuipc-0.0.29 StableNeoHookean BDF1',
                         'field_warning': 'flow/stress/force_mask are zero placeholders, not supervision',
                         'timestamps_seconds': gt['timestamps_seconds'], 'time_provenance': 'frame index * frame_dt',
                         'published_split': split, 'split': split, 'sigma_censored': False, 'damping': None})
    manifest = ROOT / f'data/manifests/{NAME}.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    train = [r for r in rows if r['split'] == 'train']
    native, specific = stats(train, False, sha), stats(train, True, sha)
    for s in (native, specific):  # sigma_y absent in this pilot: keep the codec well defined
        if not s['count'][3]:
            s['mean'][3], s['std'][3] = 0., 1.
        s['std'] = [max(x, 0.05) for x in s['std']]
    (ROOT / f'data/manifests/{NAME}.stats.json').write_text(json.dumps(native, indent=1))
    (ROOT / f'data/manifests/{NAME}.specific.stats.json').write_text(json.dumps(specific, indent=1))
    print(manifest, {s: sum(r['split'] == s for r in rows) for s in ('train', 'val')}, 'skipped', skipped)


if __name__ == '__main__':
    main()
