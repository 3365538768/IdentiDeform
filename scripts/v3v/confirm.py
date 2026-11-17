"""V3v confirmation summary: docs/v3v/RESULTS.json -> docs/v3v/RESULTS_confirm.json (LOO-damping arm vs frozen comparisons).
  python3 scripts/v3v/confirm.py   (run scripts/v3v/aggregate.py first)
"""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; D = ROOT / 'data/v3v'
R = json.loads((ROOT / 'docs/v3v/RESULTS.json').read_text())
OBJS = ['bun', 'burger', 'dog', 'pig', 'potato']; CAMS = ['C0733', 'C0787', 'C0801']
ARMS = ['prior_refine_damp_loo', 'prior', 'class_mean', 'global_constant_loo', 'persist_last', 'prior_refine', 'prior_refine_damp', 'rigid_track_2cam', 'oracle']
A = 'prior_refine_damp_loo'
out = {'stage': 'v3v-confirm', 'protocol': 'docs/v3v/PROTOCOL_confirm.json', 'n_objects': 5,
       'unseen_sequences_found': False,
       'note': 'No unseen sequences exist in real_capture.zip (one drop per object). Leave-one-object-out damping instead: rpic = median of '
               'the post-hoc fitted rpic of the other 4 objects (same fold), then E-only compass search (12 sims/fold) on 2 cams x frames 0..9. '
               'Same 5 drops as the first run: this removes in-sample fitting of the damping value but is NOT a test on unseen data.',
       'summary': {a: R['summary'][a] for a in ARMS if a in R['summary']},
       'paired_differences': {k: v for k, v in R['paired_differences'].items() if k.startswith(A + ' - ')},
       'per_object': {o: {a: R['per_object'][o].get(a) for a in ARMS} for o in OBJS},
       'twins': {}}
for o in OBJS:
    f = json.loads((D / 'sims' / o / 'refine_damp_loo.json').read_text())['folds']
    out['twins'][o] = {h: {'rpic_loo': f[h]['rpic'], 'refined_log10E': f[h]['lg'], 'n_sims': f[h]['n_sims']} for h in CAMS}
(ROOT / 'docs/v3v/RESULTS_confirm.json').write_text(json.dumps(out, indent=1))
for k, v in out['paired_differences'].items():
    for b in ('heldout_camera', 'future_frames', 'future_3d'):
        if b in v:
            print(k, b, {m: (round(x['mean'], 3), [round(c, 3) for c in x['ci95_bootstrap_objects']], sum(y > 0 for y in x['per_object'].values())) for m, x in v[b].items()})
for a in ARMS:
    s = out['summary'].get(a, {})
    print(a, {b: {m: round(x['mean'], 3) for m, x in s[b].items()} for b in ('heldout_camera', 'future_frames', 'future_3d') if b in s})
print(json.dumps(out['twins']))
