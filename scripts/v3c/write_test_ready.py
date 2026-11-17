"""V3c: docs/v3c/TEST_READY.json — counts, hashes and the exact one-time test commands (no metric computed).

Runs scripts/v3c/eval_test.py --check-only for every trained run (guards + dataset construction only).
"""
import hashlib, json, subprocess, time
from pathlib import Path

def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while b := f.read(1 << 22): h.update(b)
    return h.hexdigest()

ENV = 'source scripts/r1_env.sh && '
caches = {'vjepa2_vitl': 'data/prepared/v2main_test_vjl', 'dinov3_vitl16': 'data/prepared/v2main_test_dinov3l16'}
runs = {f'vjepa2_32f_{g}': (f'configs/v2main/v2main_{g}_seed{{}}.json', f'runs/v2main_{g}_seed{{}}', caches['vjepa2_vitl']) for g in 'BCD'}
runs.update({name: (f'configs/v3c/v3c_dino_{g}_seed{{}}.json', f'runs/v3c_dino_{g}_seed{{}}', caches['dinov3_vitl16'])
             for name, g in [('dinov3_16f_B', 'B'), ('dinov3_16f_C', 'C'), ('dinov3_static_first_frame_B', 'S')]})
methods, checks = {}, {}
for name, (cfg, run, cache) in runs.items():
    cmds = []
    for s in (17, 29, 43):
        c, r = cfg.format(s), run.format(s)
        cmd = f'.venv/bin/python scripts/v3c/eval_test.py --config {c} --checkpoint {r}/best_material.pt --prepared-dir {cache} --output {r}/test_once'
        out = subprocess.run(['bash', '-c', ENV + 'CUDA_VISIBLE_DEVICES= ' + cmd + ' --check-only'], capture_output=True, text=True)
        ok = out.returncode == 0
        checks[f'{name}/seed{s}'] = json.loads(out.stdout.strip().splitlines()[-1]) if ok else {'error': (out.stderr or out.stdout)[-600:]}
        cmds.append({'seed': s, 'config': c, 'config_sha256': sha(c), 'checkpoint': f'{r}/best_material.pt',
                     'checkpoint_sha256': sha(f'{r}/best_material.pt'), 'guard_check_passed': ok, 'command': cmd})
    methods[name] = {'selection': 'best_material.pt = epoch with best val standardized MAE (frozen)', 'feature_cache': cache, 'runs': cmds}
methods['constant_train_mean'] = methods['law_mean_gt_privileged'] = None
methods['constant_and_law_mean_gt_privileged'] = {'command': '.venv/bin/python scripts/v3c/simple_baselines_test.py',
    'outputs': ['runs/v3c_const_test', 'runs/v3c_lawmean_gt_test'], 'gpu': False}
del methods['constant_train_mean'], methods['law_mean_gt_privileged']
methods['qwen3vl8b_zeroshot'] = {'gpu': '~26 GB each, ~5 min per shard', 'commands': [
    f'.venv/bin/python scripts/v3c/vlm_v2main.py --split test --manifest data/manifests/v2main_withtest.jsonl --shard {s} --num-shards 4 --output runs/v3c_vlm_qwen3vl8b_test' for s in range(4)]}
cache_info = {}
for k, d in caches.items():
    idx = json.loads((Path(d) / 'index.json').read_text())
    cache_info[k] = {'dir': d, 'samples': len(idx['files']), 'index_sha256': sha(Path(d) / 'index.json'),
                     'manifest_sha256': idx['manifest_sha256'], 'backbone_checkpoint_sha256': idx['backbone_checkpoint_sha256'],
                     'frames': idx['frames'], 'bytes': sum(p.stat().st_size for p in Path(d).glob('*.pt'))}
pack = json.loads(Path('docs/v3c/test_pack_check.json').read_text())
man = json.loads(Path('docs/v3c/test_manifest.json').read_text())
ready = {'written_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
         'status': 'prepared; NO test metric has been computed. Run each command exactly once (eval_test.py refuses existing outputs).',
         'frozen_protocol': {'path': 'docs/v3c/TEST_PROTOCOL.json', 'sha256': sha('docs/v3c/TEST_PROTOCOL.json')},
         'deviation_from_protocol_text': 'TEST_PROTOCOL lists `fieldbridge.cli evaluate --split test`. That path cannot be used unchanged: stats embed manifest_sha256 and the training manifest has no test rows. scripts/v3c/eval_test.py is cli.evaluate with only the manifest/stats/cache re-pointed, guarded so the codec is provably identical (manifest is a byte-extension; stats equal except the hash). Methods, checkpoints and selection rule are unchanged.',
         'test_data': {'dirs': pack['dirs'], 'result_exit0': pack['exit0'], 'valid_packs': pack['ok'], 'corrupt_quarantined': pack['quarantined'],
                       'rgb_shape': pack['rgb_shapes'], 'objects': man['test_objects'], 'object_overlap_with_train_val': man['test_objects_overlapping_train_val'],
                       'law_counts': man['test_law_counts']},
         'manifest': {**man, 'v2main.jsonl_sha256_now': sha('data/manifests/v2main.jsonl'),
                      'v2main_withtest.stats.json_sha256': sha('data/manifests/v2main_withtest.stats.json'),
                      'v2main_withtest.specific.stats.json_sha256': sha('data/manifests/v2main_withtest.specific.stats.json')},
         'feature_caches': cache_info,
         'scripts_sha256': {p: sha(p) for p in ['scripts/v3c/eval_test.py', 'scripts/v3c/simple_baselines_test.py', 'scripts/v3c/simple_baselines.py',
                            'scripts/v3c/vlm_v2main.py', 'scripts/v3c/test_report.py', 'scripts/v3c/baselines_report.py', 'scripts/v3c/build_test_manifest.py',
                            'scripts/v3c/check_test_packs.py', 'src/fieldbridge/evaluation.py', 'src/fieldbridge/model.py']},
         'run_order': ['all 18 trained-run commands (GPU, ~2 min each; source scripts/r1_env.sh first)', 'simple_baselines_test.py (CPU)',
                       'VLM 4 shards (GPU)', '.venv/bin/python scripts/v3c/test_report.py --output docs/v3c/BASELINES_TEST.json'],
         'methods': methods, 'guard_checks': checks}
Path('docs/v3c/TEST_READY.json').write_text(json.dumps(ready, indent=1) + '\n')
print(json.dumps({k: v.get('guard_check_passed') for m in methods.values() if m and 'runs' in m for k, v in [(f"{r['config']}", r) for r in m['runs']]}, indent=1))
print(json.dumps(cache_info, indent=1))
