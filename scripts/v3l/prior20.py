"""V3l: zero-shot prior (9-run V2-main ensemble, K0) on the Spring-Gaus synthetic scenes using ONLY frames 0-19.

  .venv/bin/python -I scripts/v3l/prior20.py <mpm_synthetic.zip> <out.json> [--scenes ...]

Same code path as scripts/v3f/springgaus_fewshot.py (build_syn + eval_syn K0), imported unchanged; the only
difference is that the first N_OBS=20 rendered frames (t = 0-0.57 s) are used (20 -> 64 by nearest repetition)
instead of all 30.  Canonical triplet t0 from docs/v3f/SPRINGGAUS_SYN.json.  As a sanity check the 30-frame
version is recomputed too and compared to the stored v3f z_t0.  Untrusted zip: only PNG/JSON decoded.
"""
import argparse, io, json, sys, time, zipfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts/v3f')]
import springgaus_fewshot as F  # noqa: E402


def main(a):
    from PIL import Image
    from fieldbridge.data import ParameterCodec
    codec = ParameterCodec(json.loads(F.STATS.read_text()))
    syn = json.loads((ROOT / 'docs/v3f/SPRINGGAUS_SYN.json').read_text())
    z = zipfile.ZipFile(a.zip)
    t0 = time.time(); vj = F.vjepa_model(); torch.cuda.synchronize(); t_vj_load = time.time() - t0
    samples, timing = {}, {}
    for scene in a.scenes:
        views = syn['protocol']['views'][scene]['triplets'][0]
        cams = json.loads(z.read(f'{F.PREFIX}{scene}/camera.json'))
        fr = json.loads(z.read(f'{F.PREFIX}{scene}/frame.json')); times_all = np.array([list(d.values())[0] for d in fr], np.float32)
        ts = time.time()
        per_view = []
        for v in views:
            fs = []
            for f in range(len(times_all)):
                im = np.asarray(Image.open(io.BytesIO(z.read(f'{F.PREFIX}{scene}/{cams[v]["camera"]}/{f:03d}.png'))).convert('RGBA'), np.float32) / 255
                rgb = im[..., :3] * im[..., 3:4]
                fs.append(np.asarray(Image.fromarray((rgb * 255).round().astype(np.uint8)).resize((224, 224), Image.BICUBIC), np.float32) / 255)
            per_view.append(fs)
        t_decode = time.time() - ts
        for n in (a.n_obs, len(times_all)):
            ts = time.time()
            pv = [f[:n] for f in per_view]
            rgb = F.frames_to_tensor(pv, n)
            idx = np.round(np.linspace(0, n - 1, 64)).astype(int)
            feats = F.encode(vj, rgb); torch.cuda.synchronize()
            samples[(scene, n)] = F.make_sample(f'springgaus/{scene}/black/t0', scene, 'Spring-Gaus-synthetic', views, feats,
                                                torch.from_numpy(times_all[:n][idx]), torch.zeros(4), torch.zeros(4, dtype=torch.bool))
            timing[f'{scene}_n{n}_encode_s'] = time.time() - ts
        timing[f'{scene}_decode_s'] = t_decode
        print(scene, views, {k: round(v, 2) for k, v in timing.items() if k.startswith(scene)}, flush=True)
    del vj; torch.cuda.empty_cache()
    keys = sorted(samples)
    mus, t_heads = [], 0.0
    for r in F.RUNS:
        model, _ = F.load_model(r, 'cuda')
        ts = time.time(); mu, logits, h = F.run(model, [samples[k] for k in keys], 'cuda'); torch.cuda.synchronize(); t_heads += time.time() - ts
        mus.append(mu); del model; torch.cuda.empty_cache()
    Z = torch.stack(mus).mean(0)
    out = {'n_obs_frames': a.n_obs, 'views': 't0 triplet (docs/v3f/SPRINGGAUS_SYN.json)', 'runs': F.RUNS,
           'timing_s': {'vjepa_load': t_vj_load, 'heads_9runs_total_all_samples': t_heads, **timing}, 'scenes': {}}
    for i, (scene, n) in enumerate(keys):
        dec = codec.decode(Z[i][None])[0]
        d = out['scenes'].setdefault(scene, {})
        tag = 'f20' if n == a.n_obs else f'f{n}_recomputed'
        d[tag] = {'z': Z[i].tolist(), 'log10_E_Pa_at_rho1000': float(np.log10(float(dec[0]))), 'nu': float(dec[1]),
                  'per_run_z_E': [float(m[i][0]) for m in mus]}
        if n != a.n_obs:
            ref = syn['ensemble9']['K0']['z_t0'].get(scene)
            d['v3f_stored_z_t0'] = ref
            d['sanity_abs_dz_E_vs_v3f'] = abs(float(Z[i][0]) - ref[0]) if ref else None
        gt = syn['protocol']['gt'].get(scene, {})
        if isinstance(gt.get('E'), (int, float)):
            d['gt_log10E'] = float(np.log10(gt['E'])); d['gt_nu'] = gt['NU']
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({s: {k: (v['log10_E_Pa_at_rho1000'] if isinstance(v, dict) and 'z' in v else v) for k, v in d.items() if k != 'v3f_stored_z_t0'} for s, d in out['scenes'].items()}, indent=1, default=str))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('zip'); p.add_argument('out')
    p.add_argument('--n-obs', type=int, default=20)
    p.add_argument('--scenes', nargs='+', default=['torus', 'duck', 'apple', 'C', 'E', 'V', 'banana', 'chess', 'cream', 'cross',
                                                  'doraemon', 'droplet', 'rope', 'toothpaste'])
    main(p.parse_args())
