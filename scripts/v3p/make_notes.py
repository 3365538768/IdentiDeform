"""V3p: derive the headline comparison + caveats (docs/v3p/notes.json) from docs/v3p/RESULTS_raw.json."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
raw = json.loads((ROOT / 'docs/v3p/RESULTS_raw.json').read_text())
pa = json.loads((ROOT / 'docs/v3p/PER_ACTION_VAL.json').read_text()) if (ROOT / 'docs/v3p/PER_ACTION_VAL.json').exists() else None
v = lambda ds, sp, m, k: raw[ds]['splits'][sp][m][k]['value']
ci = lambda ds, sp, m, k: raw[ds]['splits'][sp][m][k]['ci95']
head = {}
for ds in [d for d in ('V2', 'V3', 'V3_4k') if d in raw]:
    for sp in ('val', 'test'):
        e = 'ensemble_BCD9'
        if e not in raw[ds]['splits'][sp]:
            continue
        h = {}
        for m in ('C (3 seeds)', e):
            h[m] = {k: {'value': v(ds, sp, m, k), 'ci95': ci(ds, sp, m, k)} for k in
                    ('z_mae', 'ident_z_mae', 'law_accuracy', 'within_law_pearson_E_over_rho', 'within_law_spearman_E_over_rho',
                     'within_law_pearson_sigma_y_over_rho', 'within_law_spearman_sigma_y_over_rho', 'decade_mae_E_over_rho', 'decade_mae_sigma_y_over_rho')}
        const = {k: v(ds, sp, 'constant_train_mean', k) for k in ('decade_mae_E_over_rho', 'decade_mae_sigma_y_over_rho', 'z_mae', 'ident_z_mae')}
        h['constant_train_mean'] = const
        h['ensemble_gain_over_constant_rel'] = {k: 1 - h[e][k]['value'] / const[k] for k in const}
        head[f'{ds}/{sp}'] = h
notes = {'headline': head,
         'per_action_val_ensemble': pa,
         'differences_vs_v2': [
             'Data: V3 = contact through external rigid plates/grippers; gravity 2 (press/tap30hold/drop) else 0; narrower log10 ranges (E/rho jelly 3.5-6, plasticine 3.5-5.5, metal 4.5-6; sigma_y/rho 3.2-4.5); per-sample censoring sigma/rho > 0.5 E/rho (any law); screened objects v3_split (57/10/10).',
             'Train set: 2000 V3 samples (V2: 1955); val 283 / test 300 (V2: 283 / 300).',
             'Standardization from V3 train: std(ln E/rho) = %.3f decades (V2 %.3f), std(ln sigma_y/rho) = %.3f decades (V2 %.3f); %d/2000 train samples carry a sigma label (V2 1264/1955). z-units therefore differ; decade errors are the cross-dataset comparison.' % (
                 raw['V3']['train_std_decades']['E_over_rho'], raw['V2']['train_std_decades']['E_over_rho'],
                 raw['V3']['train_std_decades']['sigma_y_over_rho'], raw['V2']['train_std_decades']['sigma_y_over_rho'], raw['V3']['train_count'][3]),
             'Class information is much weaker in V3: privileged GT-class mean reaches z-MAE %.3f vs constant %.3f (V2: %.3f vs %.3f), because the per-law ranges overlap.' % (
                 v('V3', 'val', 'class_mean_gt_privileged', 'z_mae'), v('V3', 'val', 'constant_train_mean', 'z_mae'),
                 v('V2', 'val', 'class_mean_gt_privileged', 'z_mae'), v('V2', 'val', 'constant_train_mean', 'z_mae'))],
         'protocol_identical_to_v2': 'Same V-JEPA 2 ViT-L frozen 32-step features (64 frames, tubelet 2, 3 views, 224 px), same 2-layer transformer head and hyper-parameters (configs/v2main/* with only data paths/output changed), same loss weights (smooth-l1 + 0.05 NLL + 0.1 class CE; C/D field weight 0.1, 2 field-warmup epochs, D feedback=fields), seeds 17/29/43, best-val-epoch selection (best_material.pt), early stop patience 5 / min 8 joint epochs, test evaluated once.',
         'deviations': [
             'Feature cache built incrementally into one shared directory (scripts/v3p/cache_v3.py, same encoder code path as scripts/prepare_vjepa_cache.py); per-manifest index.json files point into it. No change to features.',
             'IO fix during training (14:31-14:45 UTC): 8 of 9 runs were paused at an epoch boundary (control.json, built-in mechanism) and resumed from last.pt with scripts/v3p/train_job.py --preload (features read once into RAM; identical math/sampling; verified on a smoke run: val MAE identical to 1e-4). C_seed29 was not paused.',
             'First-frame DINOv3 baseline uses a first-frame-only DINOv3 cache (frames 1..15 stored as a stride-0 copy of frame 0; they are masked by static_first_frame anyway), not the full 16-frame cache.',
             'Within-class correlation: V2 tables use Pearson; both Pearson and Spearman are reported here.',
             'C/D were trainable: V3 packs contain the same stress/flow/force_mask fields as V2 (contact channel is empty in both V2 and V3).'],
         'hand_off': 'docs/v3p/HANDOFF.json: main_run v3main_C_seed29 (2000-sample prior) + 9-run B/C/D ensemble (docs/v3p/ENSEMBLE_BCD9_{VAL,TEST}.json, runs/v3main_ens_bcd9_{val,test}).'}
(ROOT / 'docs/v3p/notes.json').write_text(json.dumps(notes, indent=1, ensure_ascii=False))
print(json.dumps(head, indent=0)[:3000])
