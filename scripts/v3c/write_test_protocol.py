"""V3c: freeze the V2 test protocol for the comparison baselines (docs/v3c/TEST_PROTOCOL.json)."""
import hashlib, json, time
from pathlib import Path

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

files = ['scripts/v3c/simple_baselines.py', 'scripts/v3c/vlm_v2main.py', 'scripts/v3c/baselines_report.py', 'scripts/vlm_baseline.py',
         'scripts/prepare_feature_cache.py', 'scripts/prepare_vjepa_cache.py', 'scripts/bootstrap_object_ci.py', 'scripts/audit_within_law.py',
         'data/manifests/v2main.jsonl', 'data/manifests/v2main.specific.stats.json', 'data/manifests/v2main.stats.json',
         *sorted(str(p) for p in Path('configs/v3c').glob('*.json')), *sorted(str(p) for p in Path('configs/v2main').glob('v2main_[BCD]_seed*.json'))]
protocol = {
    'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    'status': 'frozen before any test-split access; run ONCE, only if the user-facing plan says so',
    'test_split': 'data/v2_main/test (300 samples, generated separately); objects must be disjoint from train/val (fieldbridge.cli audit)',
    'statistics': 'train-split statistics of data/manifests/v2main.specific.stats.json (numbers unchanged; if the test rows are appended to a new manifest, refit stats on the identical train rows and assert equality of mean/std)',
    'metric': {'primary': 'standardized MAE over all scored (sample, parameter) entries in specific space (E/rho, nu, rho, sigma_y/rho)',
               'secondary': ['per-parameter MAE (z)', 'law accuracy (3-class)', 'within-law Pearson r for E/rho and sigma_y/rho (mean over GT laws)',
                             'paired object bootstrap vs V-JEPA 2 C (2000 draws, seed 17; seed-matched for trained methods, against each C seed for deterministic ones)']},
    'methods': {
        'constant_train_mean': {'rule': 'z = 0 (train mean) for every parameter, majority train law', 'selection': 'none', 'script': 'scripts/v3c/simple_baselines.py --split test'},
        'law_mean_gt_privileged': {'rule': 'GT law -> train-split law mean z (PRIVILEGED; not deployable)', 'selection': 'none', 'script': 'scripts/v3c/simple_baselines.py --split test'},
        'qwen3vl8b_zeroshot': {'rule': 'scripts/v3c/vlm_v2main.py --split test, 3 views x 8 frames, greedy, R5 prompt with V2 train ranges, z=0 fallback', 'selection': 'none'},
        'dinov3_static_first_frame_B': {'configs': 'configs/v3c/v3c_dino_S_seed{17,29,43}.json', 'cache': 'DINOv3 ViT-L/16 16-frame cache (test samples built with configs/v3c/v3c_dino_cache.json)', 'checkpoint': 'best_material.pt (best val standardized MAE)'},
        'dinov3_16f_B': {'configs': 'configs/v3c/v3c_dino_B_seed{17,29,43}.json', 'checkpoint': 'best_material.pt (best val standardized MAE)'},
        'dinov3_16f_C': {'configs': 'configs/v3c/v3c_dino_C_seed{17,29,43}.json', 'checkpoint': 'best_material.pt (best val standardized MAE)'},
        'vjepa2_32f_B/C/D (main)': {'configs': 'configs/v2main/v2main_{B,C,D}_seed{17,29,43}.json', 'checkpoint': 'best_material.pt (best val standardized MAE)'}},
    'evaluation_command': '.venv/bin/python -m fieldbridge.cli evaluate --config <config with test cache> --checkpoint <run>/best_material.pt --split test --output <run>/test_once',
    'reporting': 'mean +- sd over seeds 17/29/43; no re-selection, no re-training, no prompt change after test access; all methods reported, including those that lose',
    'sha256': {f: sha(f) for f in files if Path(f).exists()}}
Path('docs/v3c/TEST_PROTOCOL.json').write_text(json.dumps(protocol, indent=2) + '\n')
print('frozen', len(protocol['sha256']))
