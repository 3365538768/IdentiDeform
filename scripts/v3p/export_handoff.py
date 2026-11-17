"""V3p hand-off: V3 prior predictions in the formats the V2 refinement code consumes.

  * per-run predictions: runs/<prefix>_{B,C,D}_seed{17,29,43}/val_epoch_XXXX/predictions.jsonl (best val epoch,
    resolved with scripts/r1_report.best_dir exactly like c1_refine.py) and runs/<run>/test_once/predictions.jsonl;
  * docs/v3p/ENSEMBLE_BCD9_{VAL,TEST}.json: format of docs/v3b/ENSEMBLE_BCD9_VAL.json / docs/v3d/ENSEMBLE_BCD9_TEST.json
    ({runs, samples: {sid: {prediction_z, z_std_across_runs, law_posterior_votes}}});
  * runs/<prefix>_ens_bcd9_{val,test}/predictions.jsonl: format of scripts/v3c/ensembles.py;
  * docs/v3p/HANDOFF.json with every path + the V3 wiring c1_refine.py needs (stats, manifest, sid format, data root).
  .venv/bin/python scripts/v3p/export_handoff.py --prefix v3main --manifest v3main_t2000
"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from r1_report import best_dir  # noqa: E402


def main():
    a = argparse.ArgumentParser(); a.add_argument('--prefix', default='v3main'); a.add_argument('--manifest', default='v3main_t2000')
    a.add_argument('--groups', default='BCD'); a.add_argument('--main-run', default=None); a.add_argument('--extra', default='{}'); a.add_argument('--docs', default='docs/v3p'); a.add_argument('--data-root', default='data/v3_main'); a.add_argument('--sid-prefix', default='v3main'); args = a.parse_args()
    runs = [f'{args.prefix}_{g}_seed{s}' for g in args.groups for s in (17, 29, 43)]
    main_run = args.main_run or (f'{args.prefix}_C_seed29' if 'C' in args.groups else f'{args.prefix}_B_seed29')
    tag = f'{args.groups.lower()}{len(runs)}'
    h = {'stage': args.docs.split('/')[-1], 'written_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
         'data_root': args.data_root, 'sample_id_format': f'{args.sid_prefix}/<split>/<idx:06d>',
         'manifest': f'data/manifests/{args.manifest}.jsonl', 'stats_specific': f'data/manifests/{args.manifest}.specific.stats.json',
         'target_space': 'specific: z = (ln1p(x) - mean) / std with x = E/(rho/1000), nu, rho, sigma_y/(rho/1000); order [E_over_rho, nu, rho, sigma_y_over_rho]',
         'law_ids': {'0': 'jelly', '1': 'metal', '2': 'plasticine'},
         'main_run': main_run, 'runs': {}, 'ensemble': {},
         'note_for_c1_refine': ('scripts/v3d/c1_refine.py is wired to V2 (STATS=data/manifests/v2main.specific.stats.json, manifest '
                                'v2main.jsonl, sid v2main/..., data/v2_main, gen_v2_main ACTIONS). For V3 re-point those to the '
                                'stats/manifest/sid/data_root above and to the V3 generator (scripts/v3n/gen_v3.py, contact setup '
                                'scripts/v2/contact_bc.py). prediction_dir: val = best val epoch dir (r1_report.best_dir), test = test_once.')}
    h.update(json.loads(args.extra))
    for split in ('val', 'test'):
        sets, used = [], []
        for r in runs:
            d = best_dir(ROOT / 'runs' / r)[0] if split == 'val' else ROOT / 'runs' / r / 'test_once'
            if not (d / 'predictions.jsonl').exists():
                continue
            h['runs'].setdefault(r, {})[split] = {'prediction_dir': str(d.relative_to(ROOT)), 'predictions': str((d / 'predictions.jsonl').relative_to(ROOT)),
                                                  'checkpoint': f'runs/{r}/best_material.pt'}
            sets.append({x['sample_id']: x for x in map(json.loads, (d / 'predictions.jsonl').read_text().splitlines())}); used.append(r)
        if len(used) != len(runs):
            h['ensemble'][split] = {'status': f'incomplete: {len(used)}/{len(runs)} runs'}; continue
        rows, ens = [], {}
        for sid in sorted(sets[0]):
            rs = [s[sid] for s in sets]; b = rs[0]
            for x in rs[1:]:
                if x['target_z'] != b['target_z'] or x['target_mask'] != b['target_mask']:
                    raise ValueError('target mismatch ' + sid)
            z = np.array([x['prediction_z'] for x in rs]); votes = np.bincount([x['predicted_law_id'] for x in rs], minlength=3)
            ens[sid] = {'prediction_z': z.mean(0).tolist(), 'z_std_across_runs': z.std(0).tolist(), 'law_posterior_votes': (votes / len(rs)).tolist()}
            rows.append({k: b[k] for k in ('sample_id', 'object_id', 'source', 'action', 'law_id', 'target_z', 'target_mask')}
                        | {'prediction_z': z.mean(0).tolist(), 'logvar': [0.] * 4, 'predicted_law_id': int(np.argmax(votes)), 'law_votes': votes.tolist()})
        js = ROOT / f'{args.docs}/ENSEMBLE_{tag.upper()}_{split.upper()}.json'
        js.write_text(json.dumps({'runs': runs, 'split': split, 'rule': 'mean z over runs; law_posterior_votes = vote fractions', 'samples': ens}))
        out = ROOT / f'runs/{args.prefix}_ens_{tag}_{split}'; out.mkdir(parents=True, exist_ok=True)
        (out / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
        (out / 'members.json').write_text(json.dumps({'runs': runs, 'rule': 'mean z; majority-vote law (ties -> lowest id)'}))
        h['ensemble'][split] = {'json': str(js.relative_to(ROOT)), 'predictions': str((out / 'predictions.jsonl').relative_to(ROOT)), 'samples': len(rows)}
    (ROOT / f'{args.docs}/HANDOFF.json').write_text(json.dumps(h, indent=1) + '\n'); print(json.dumps(h, indent=1)[:3000])


if __name__ == '__main__':
    main()
