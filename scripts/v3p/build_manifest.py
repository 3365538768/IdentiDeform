"""V3p: DeformFieldBench-V3 main manifest + train statistics (same format/rules as scripts/v2/build_v2_manifest.py).

  .venv/bin/python scripts/v3p/build_manifest.py --name v3main_t2000 --train-max 2000 [--splits train,val,test]
Writes data/manifests/<name>.jsonl, .stats.json (native) and .specific.stats.json (stats fitted on the train rows
of THIS manifest). sample_id is always v3main/<split>/<idx> so the shared feature cache (data/prepared/v3main_vjl)
is reused across manifests. A sample is kept only if result.json has exit 0 and sample_pack.npz exists.
Labels follow the V3 gt.json: sigma_y is null (masked) for jelly and whenever labels.censored_sigma is true
(V3: per-sample sigma/rho > 0.5 E/rho, any law). Objects must follow data/manifests/v3_split.json.
"""
import argparse, hashlib, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v2'))
from build_v2_manifest import stats  # noqa: E402  (identical statistics code)


def main():
    a = argparse.ArgumentParser(); a.add_argument('--root', default='data/v3_main'); a.add_argument('--name', required=True)
    a.add_argument('--splits', default='train,val,test'); a.add_argument('--train-max', type=int, default=10**9,
                   help='keep train indices < this (data-scaling points)')
    a.add_argument('--min-train', type=int, default=0); a.add_argument('--prefix', default='v3main', help='sample_id/source prefix (v4main for data/v4_main)')
    a.add_argument('--alt', action='append', default=[], metavar='ACTIONS=ROOT:PREFIX',
                   help='take these actions from another root, e.g. press,drop,tap30hold=data/v3_main:v3main (V4 = V3 for those actions)')
    a.add_argument('--ranges', default='v3', help='accepted gt.json ranges tags (comma list), e.g. v5 for data/v5b_main')
    args = a.parse_args()
    alt = {}
    for spec in args.alt:
        acts, rp = spec.split('='); r, pre = rp.split(':')
        for x in acts.split(','):
            alt[x] = (r, pre)
    root = ROOT / args.root
    split_objects = json.loads((ROOT / 'data/manifests/v3_split.json').read_text())
    obj_split = {}
    for s in ('train', 'val', 'test'):
        for o in split_objects[s]:
            obj_split[o] = s
    rows = []
    for split in args.splits.split(','):
        names = sorted({d.name for d in (root / split).glob('[0-9]*')} | {d.name for r_, _ in alt.values() for d in (ROOT / r_ / split).glob('[0-9]*')})
        for name in names:
            d, prefix = root / split / name, args.prefix
            gtp = next((x / 'gt.json' for x in [d] + [ROOT / r_ / split / name for r_, _ in alt.values()] if (x / 'gt.json').exists()), None)
            if gtp is None:
                continue
            act = json.loads(gtp.read_text())['action']
            if act in alt:
                d, prefix = ROOT / alt[act][0] / split / name, alt[act][1]
            if split == 'train' and int(d.name) >= args.train_max:
                continue
            packs = list(d.glob('*/sample_pack.npz')); result = d / 'result.json'
            try:
                ok = packs and json.loads(result.read_text()).get('exit') == 0
            except (OSError, ValueError):
                ok = False
            if not ok:
                continue
            gt = json.loads((d / 'gt.json').read_text())
            if gt.get('ranges') not in args.ranges.split(',') or gt.get('objects_split', 'v3') != 'v3' or gt.get('split') != split:
                raise ValueError(f'not a V3 sample: {d}')
            if obj_split.get(gt['object']) != split:
                raise ValueError(f'object split mismatch: {d} {gt["object"]}')
            rp = json.loads((packs[0].parent / 'meta/run_parameters.json').read_text())
            idx = rp['render_frame_indices']; dt = float(rp['time_params']['frame_dt'])
            censored = bool(gt['labels']['censored_sigma'])
            sigma = gt['regression']['yield_stress'] if gt['law'] != 'jelly' and not censored else None
            rows.append({'sample_id': f'{prefix}/{split}/{d.name}', 'object_id': gt['object'], 'asset_group': 'deformfield:' + gt['object'],
                         'source': f'DeformFieldBench-{args.prefix}', 'data_root': str(d.parent.parent.relative_to(ROOT)), 'pack': str(packs[0]),
                         'parameters': {'E': gt['regression']['E'], 'nu': gt['regression']['nu'], 'rho': gt['regression']['density'],
                                        'sigma_y': sigma},
                         'law_id': gt['law_id'], 'action': gt['action'], 'field_mode': 'legacy_visual',
                         'label_provenance': 'simulator_config', 'field_warning': 'visualized flow/stress; contact mask is not force',
                         'timestamps_seconds': [(i - idx[0]) * dt for i in idx], 'time_provenance': 'render_frame_indices * frame_dt',
                         'published_split': split, 'split': split, 'sigma_censored': censored,
                         'damping': gt['damping'], 'gravity': gt.get('gravity')})
    train = [r for r in rows if r['split'] == 'train']
    if 'train' in args.splits and len(train) < args.min_train:
        raise SystemExit(f'gate: {len(train)} < {args.min_train} train samples')
    manifest = ROOT / f'data/manifests/{args.name}.jsonl'
    manifest.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    sha = hashlib.sha256(manifest.read_bytes()).hexdigest()
    fit = train if train else rows  # cache-only manifests without train rows: stats are only a loader formality
    for spec, suffix in ((False, '.stats.json'), (True, '.specific.stats.json')):
        s = stats(fit, spec, sha)
        if not train:
            s['note_v3p'] = 'NO TRAIN ROWS: stats fitted on cached rows only to satisfy the loader; never used for training/eval'
        (ROOT / f'data/manifests/{args.name}{suffix}').write_text(json.dumps(s, indent=1))
    print(manifest, {s: sum(r['split'] == s for r in rows) for s in ('train', 'val', 'test')}, 'objects', len({r['object_id'] for r in rows}))


if __name__ == '__main__':
    main()
