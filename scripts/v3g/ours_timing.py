"""Wall-clock of OUR model on one PAC-NeRF scene (bgsub, triplet t0): PNG decode+preprocess, V-JEPA 2 ViT-L features,
3-seed V2-main-C head forward. Reuses scripts/v3c/pacnerf_fewshot.py functions unchanged. Run from project root with .venv python."""
import json, sys, time
from pathlib import Path
import torch
sys.path.insert(0, 'scripts/v3c'); sys.path.insert(0, 'src')
import pacnerf_fewshot as P

scene = sys.argv[1] if len(sys.argv) > 1 else 'torus'
dev = torch.device('cuda')
idx = json.loads(Path('docs/v3c/PACNERF_FEWSHOT.json').read_text())['protocol']['views'][scene]['triplets'][0]
sync = torch.cuda.synchronize
from transformers import AutoModel
t0 = time.time()
vj = AutoModel.from_pretrained('weights/hf/facebook__vjepa2-vitl-fpc64-256', dtype=torch.bfloat16).cuda().eval()
models = [P.load_model(s, dev)[0] for s in (17, 29, 43)]
sync(); t_load = time.time() - t0
rec = dict(scene=scene, views=idx, gpu=torch.cuda.get_device_name(0), model_load_s=t_load, reps=[])
for rep in range(4):
    sync(); a = time.time()
    rgb, times = P.load_rgb(scene, idx, 'bgsub')
    b = time.time()
    with torch.no_grad():
        x = ((rgb[None] - P.MEAN) / P.STD)[0].cuda().to(torch.bfloat16)
        tokens = vj(pixel_values_videos=x, skip_predictor=True).last_hidden_state
    feats = tokens.reshape(3, 32, 14, 14, -1).permute(0, 1, 4, 2, 3).cpu()
    sync(); c = time.time()
    tt = ((times[0::2] + times[1::2]) / 2).repeat(3, 1)
    s = {'features': feats, 'times': tt, 'valid': torch.ones(3, 32, dtype=torch.bool), 'rgb': torch.zeros(3, 32, 3, 1, 1),
         'prior': torch.zeros(4), 'prior_mask': torch.zeros(4, dtype=torch.bool)}
    for m in models:
        P.run(m, [s], dev)
    sync(); d = time.time()
    rec['reps'].append(dict(png_decode_preproc_s=b - a, vjepa_features_s=c - b, heads_3seeds_s=d - c, total_s=d - a))
rec['peak_mem_gb'] = torch.cuda.max_memory_allocated() / 1e9
rec['note'] = 'rep 0 includes CUDA warm-up; report median of reps 1-3'
w = sorted(r['total_s'] for r in rec['reps'][1:]); rec['median_total_s_warm'] = w[1]
Path('docs/v3g/pn_results').mkdir(parents=True, exist_ok=True)
Path(f'docs/v3g/pn_results/OURS_TIMING_{scene}.json').write_text(json.dumps(rec, indent=1))
print(json.dumps(rec, indent=1))
