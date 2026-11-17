"""V3f: Spring-Gaus (ECCV'24) external check with the V2 main B/C/D 9-run ensemble.

Data: Spring-Gaus released mpm_synthetic.zip (Google Drive, untrusted). Read DIRECTLY from the zip; only
PNG/JSON are decoded (PIL / json), nothing from the archive is executed.  Run with python -I and pass paths.
  build_syn  <mpm_synthetic.zip> <out_dir>        V-JEPA2 ViT-L features, 3 disjoint camera triplets per scene
  eval_syn   <out_dir> <report.json>               K0 / offset / ridge_last_layer / support_mean (LOSO)

Synthetic protocol (mirrors scripts/v3c/pacnerf_fewshot.py):
  10 fixed cameras -> 3 disjoint triplets chosen greedily by max-min azimuth gap around the look-at centre;
  30 rendered frames (dt 0.03 s, 0-0.87 s) -> 64 by nearest-frame repetition; 512 -> 224 px;
  RGBA alpha-composited on black (== V2 black background; the renders contain no ground/background).
  GT physical.json: all scenes 'elastic' with E, NU; density is NOT given -> rho = 1000 ASSUMED for all,
  so E/rho rankings are assumption-free up to a common density.  torus_heterogeneous has no scalar GT and is
  excluded from scoring (prediction reported).
"""
import argparse, io, itertools, json, math, sys, zipfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
STATS = ROOT / 'data/manifests/v2main.specific.stats.json'
VJ = ROOT / 'weights/hf/facebook__vjepa2-vitl-fpc64-256'
MEAN = torch.tensor([.485, .456, .406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([.229, .224, .225]).view(1, 1, 3, 1, 1)
LAW_NAMES = ['jelly', 'metal', 'plasticine']
PNAMES = ['E_over_rho', 'nu', 'rho', 'sigma_y_over_rho']
RHO = 1000.
RUNS = [f'v2main_{g}_seed{s}' for g in 'BCD' for s in (17, 29, 43)]
MODES = ('K0', 'offset', 'ridge_last_layer', 'support_mean_baseline')
PREFIX = 'mpm_synthetic/render/'


def look_at_center(c2ws):
    """Least-squares point closest to all optical axes (camera looks along -z in Blender c2w)."""
    A = np.zeros((3, 3)); b = np.zeros(3)
    for m in c2ws:
        o = m[:3, 3]; d = -m[:3, 2]; d = d / np.linalg.norm(d); P = np.eye(3) - np.outer(d, d)
        A += P; b += P @ o
    return np.linalg.solve(A, b)


def triplets(az):
    def gap(t):
        a = sorted(az[v] for v in t); d = [y - x for x, y in zip(a, a[1:])] + [360 - a[-1] + a[0]]
        return min(d)
    left, chosen = sorted(az), []
    for _ in range(3):
        best = max(itertools.combinations(left, 3), key=lambda t: (round(gap(t), 3), -sum(t)))
        chosen.append(list(best)); left = [v for v in left if v not in best]
    return chosen, [gap(t) for t in chosen]


def frames_to_tensor(frames_per_view, n_src):
    """frames_per_view: list(views) of list(n_src) float HxWx3 arrays already at 224. -> [V,64,3,224,224]."""
    idx = np.round(np.linspace(0, n_src - 1, 64)).astype(int)
    return torch.from_numpy(np.stack([np.stack([f[k] for k in idx]).transpose(0, 3, 1, 2) for f in frames_per_view]).astype(np.float32))


def vjepa_model():
    from transformers import AutoModel
    return AutoModel.from_pretrained(str(VJ), dtype=torch.bfloat16).cuda().eval()


def encode(model, rgb):
    with torch.no_grad():
        x = ((rgb[None] - MEAN) / STD)[0].cuda().to(torch.bfloat16)
        tokens = model(pixel_values_videos=x, skip_predictor=True).last_hidden_state
    return tokens.reshape(rgb.shape[0], 32, 14, 14, -1).permute(0, 1, 4, 2, 3).cpu()


def make_sample(sid, obj, source, views, feats, times, target, mask):
    tt = ((times[0::2] + times[1::2]) / 2).repeat(feats.shape[0], 1)
    return {'sample_id': sid, 'object_id': obj, 'source': source, 'action': 'drop', 'views': views,
            'features': feats, 'times': tt, 'valid': torch.ones(feats.shape[0], 32, dtype=torch.bool),
            'rgb': torch.zeros(feats.shape[0], 32, 3, 1, 1), 'prior': torch.zeros(4), 'prior_mask': torch.zeros(4, dtype=torch.bool),
            'target': target, 'target_mask': mask}


def encode_target(params):
    from fieldbridge.data import ParameterCodec, specific_parameters
    codec = ParameterCodec(json.loads(STATS.read_text()))
    if params is None:
        return torch.zeros(4), torch.zeros(4, dtype=torch.bool)
    t, m = codec.encode(specific_parameters(params))
    m[2] = False  # density assumed, never scored
    return t, m


def build_syn(a):
    from PIL import Image
    from fieldbridge.train import atomic_save
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(a.zip)
    names = set(z.namelist())
    scenes = sorted({n[len(PREFIX):].split('/')[0] for n in names if n.startswith(PREFIX) and n.count('/') >= 3})
    model = vjepa_model()
    index = {'scenes': {}, 'rho_assumed': RHO, 'files': {}}
    for scene in scenes:
        phys = json.loads(z.read(f'{PREFIX}{scene}/physical.json')); cams = json.loads(z.read(f'{PREFIX}{scene}/camera.json'))
        fr = json.loads(z.read(f'{PREFIX}{scene}/frame.json')); times_src = np.array([list(d.values())[0] for d in fr], np.float32)
        c2w = {i: np.array(c['c2w'], dtype=np.float64) for i, c in enumerate(cams)}
        ctr = look_at_center(list(c2w.values()))
        az = {i: math.degrees(math.atan2(m[2, 3] - ctr[2], m[0, 3] - ctr[0])) for i, m in c2w.items()}
        el = {i: math.degrees(math.atan2(m[1, 3] - ctr[1], math.hypot(m[0, 3] - ctr[0], m[2, 3] - ctr[2]))) for i, m in c2w.items()}
        chosen, gaps = triplets(az)
        mat = phys[0]
        homo = isinstance(mat.get('E'), (int, float))
        params = {'E': float(mat['E']), 'nu': float(mat['NU']), 'rho': RHO, 'sigma_y': None} if homo else None
        t, m = encode_target(params)
        index['scenes'][scene] = {'physical': phys, 'look_at_center': ctr.tolist(), 'azimuth_deg': az, 'elevation_deg': el,
                                  'triplets': chosen, 'min_azimuth_gap_deg': gaps, 'n_frames': len(times_src),
                                  'duration_s': float(times_src[-1]), 'homogeneous': homo}
        idx = np.round(np.linspace(0, len(times_src) - 1, 64)).astype(int)
        times = torch.from_numpy(times_src[idx])
        cache = {}
        for k, views in enumerate(chosen):
            per_view = []
            for v in views:
                if v not in cache:
                    fs = []
                    for f in range(len(times_src)):
                        im = np.asarray(Image.open(io.BytesIO(z.read(f'{PREFIX}{scene}/{cams[v]["camera"]}/{f:03d}.png'))).convert('RGBA'), np.float32) / 255
                        rgb = im[..., :3] * im[..., 3:4]
                        fs.append(np.asarray(Image.fromarray((rgb * 255).round().astype(np.uint8)).resize((224, 224), Image.BICUBIC), np.float32) / 255)
                    cache[v] = fs
                per_view.append(cache[v])
            rgb = frames_to_tensor(per_view, len(times_src))
            if k == 0:
                strip = (rgb[:, ::8].permute(0, 3, 1, 4, 2).reshape(3 * 224, -1, 3).numpy() * 255).astype(np.uint8)
                Image.fromarray(strip).save(out / f'preview_{scene}.png')
            feats = encode(model, rgb)
            sid = f'springgaus/{scene}/black/t{k}'
            name = sid.replace('/', '_') + '.pt'
            atomic_save(make_sample(sid, scene, 'Spring-Gaus-synthetic', views, feats, times, t, m), out / name)
            index['files'][sid] = name
            print(sid, views, round(gaps[k], 1), flush=True)
    (out / 'index.json').write_text(json.dumps(index, indent=2))


# ---------------------------------------------------------------- evaluation
def load_model(run, device):
    from fieldbridge.config import load_config
    from fieldbridge.local_model import build_model
    from fieldbridge.runtime import load_weights, sha256
    cfg = load_config(str(ROOT / f'configs/v2main/{run}.json'))
    model = build_model(cfg.model)
    ckpt = torch.load(ROOT / cfg.train.output / 'best_material.pt', map_location='cpu', weights_only=False)
    load_weights(model, ckpt, sha256(str(ROOT / cfg.model.weights)))
    return model.to(device).eval(), {'checkpoint_epoch': ckpt.get('epoch')}


def run(model, samples, device):
    cache = {}
    h1 = model.param_head[1].register_forward_hook(lambda m, i, o: cache.__setitem__('h', o.detach().float()))
    outs = []
    with torch.no_grad():
        for s in samples:
            batch = {k: (s[k][None].to(device) if torch.is_tensor(s[k]) else s[k]) for k in ['rgb', 'valid', 'times', 'prior', 'prior_mask', 'features']}
            batch['features'] = batch['features'].float()
            y = model(**batch)
            outs.append((y['mean'][0].float().cpu(), y['law_logits'][0].float().cpu(), cache['h'][0].cpu()))
    h1.remove()
    return [torch.stack(x) for x in zip(*outs)]


def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def pair_order(pred, gt):
    pairs = [(i, j) for i in range(len(gt)) for j in range(i + 1, len(gt)) if gt[i] != gt[j]]
    return float(np.mean([(pred[i] - pred[j]) * (gt[i] - gt[j]) > 0 for i, j in pairs])), len(pairs)


def adapt(mu, h, T, M, scene_of, trip, names, lam_rel=1.0):
    """LOSO per query scene: offset, ridge on last layer, support mean. Returns dict mode->{scene: z [n_trip,4]}."""
    res = {m: {} for m in MODES}
    for q in names:
        qi = np.where(scene_of == q)[0]; si = np.where((scene_of != q) & M[:, 0].numpy())[0]
        off = torch.zeros(4)
        for p in range(4):
            ok = M[si, p]
            if ok.any():
                off[p] = (T[si][ok, p] - mu[si][ok, p]).mean()
        ridge = mu[qi].clone(); lam = lam_rel * float((h[si] ** 2).sum(1).mean())
        for p in range(4):
            ok = M[si, p]
            if ok.sum() < 2:
                ridge[:, p] = mu[qi, p] + off[p]; continue
            H = h[si][ok]; r = T[si][ok, p] - mu[si][ok, p]
            Hm, rm = H.mean(0), r.mean(); Hc, rc = H - Hm, r - rm
            dw = torch.linalg.solve(Hc.T @ Hc + lam * torch.eye(H.shape[1]), Hc.T @ rc)
            ridge[:, p] = mu[qi, p] + h[qi] @ dw + (rm - Hm @ dw)
        sm = torch.stack([T[si][M[si, p], p].mean() if M[si, p].any() else mu[qi, p].mean() for p in range(4)])
        res['K0'][q] = mu[qi]; res['offset'][q] = mu[qi] + off; res['ridge_last_layer'][q] = ridge
        res['support_mean_baseline'][q] = sm.expand(len(qi), 4).clone()
    return res


def summarize(zt0, gt, names):
    """zt0: {scene: z(4)} canonical-triplet predictions; gt: {scene: (target, mask)}"""
    out = {}
    predE = [float(zt0[s][0]) for s in names]; gtE = [float(gt[s][0][0]) for s in names]
    predN = [float(zt0[s][1]) for s in names]; gtN = [float(gt[s][0][1]) for s in names]
    out['E_over_rho_spearman'] = spearman(predE, gtE)
    out['E_over_rho_pairwise_order_acc'], out['E_pairs'] = pair_order(predE, gtE)
    out['nu_spearman'] = spearman(predN, gtN)
    out['mae_z_E'] = float(np.mean([abs(a - b) for a, b in zip(predE, gtE)]))
    out['mae_z_nu'] = float(np.mean([abs(a - b) for a, b in zip(predN, gtN)]))
    out['mae_z_E_nu'] = (out['mae_z_E'] + out['mae_z_nu']) / 2
    return out


def eval_syn(a):
    from fieldbridge.data import ParameterCodec
    codec = ParameterCodec(json.loads(STATS.read_text()))
    d = Path(a.out); index = json.loads((d / 'index.json').read_text())
    samples = {sid: torch.load(d / f, weights_only=True) for sid, f in index['files'].items()}
    sids = sorted(samples); device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scene_of = np.array([x.split('/')[1] for x in sids]); trip = np.array([int(x[-1]) for x in sids])
    T = torch.stack([samples[x]['target'] for x in sids]); M = torch.stack([samples[x]['target_mask'] for x in sids])
    names = sorted(s for s in index['scenes'] if index['scenes'][s]['homogeneous'])
    others = sorted(s for s in index['scenes'] if not index['scenes'][s]['homogeneous'])
    gt = {s: (T[np.where((scene_of == s) & (trip == 0))[0][0]], None) for s in names}
    per_run, allz, all_law = {}, {m: [] for m in MODES}, []
    for r in RUNS:
        model, info = load_model(r, device)
        mu, logits, h = run(model, [samples[x] for x in sids], device)
        ad = adapt(mu, h, T, M, scene_of, trip, names)
        summ = {}
        for mode in MODES:
            zt0 = {s: ad[mode][s][0] for s in names}  # rows of qi are sorted by triplet (sids sorted)
            summ[mode] = summarize(zt0, gt, names)
            allz[mode].append(torch.stack([ad[mode][s] for s in names]))  # [S,3,4]
        laws = logits.argmax(-1)
        summ['law_pred_jelly_frac_all_triplets'] = float((laws[np.isin(scene_of, names)] == 0).float().mean())
        all_law.append(logits.softmax(-1))
        per_run[r] = {'model': info, 'summary': summ,
                      'K0_z': {x: mu[i].tolist() for i, x in enumerate(sids)},
                      'law_pred': {x: LAW_NAMES[int(laws[i])] for i, x in enumerate(sids)}}
        print(r, json.dumps({m: {k: round(v, 3) if isinstance(v, float) else v for k, v in summ[m].items()} for m in MODES}), flush=True)
        del model; torch.cuda.empty_cache()
    # 9-run ensemble: mean z of each mode, majority-vote law
    ens = {}
    votes = torch.stack([p.argmax(-1) for p in all_law])  # [R,N]
    maj = torch.tensor([int(np.argmax(np.bincount(votes[:, i].numpy(), minlength=3))) for i in range(votes.shape[1])])
    for mode in MODES:
        Z = torch.stack(allz[mode]).mean(0)  # [S,3,4]
        zt0 = {s: Z[i, 0] for i, s in enumerate(names)}; zall = {s: Z[i].mean(0) for i, s in enumerate(names)}
        ens[mode] = {'canonical_t0': summarize(zt0, gt, names), 'mean_over_triplets': summarize(zall, gt, names),
                     'z_t0': {s: zt0[s].tolist() for s in names}, 'z_sd_over_triplets': {s: Z[i].std(0).tolist() for i, s in enumerate(names)},
                     'decoded_t0': {s: {'E_Pa_at_rho1000': float(codec.decode(zt0[s][None])[0, 0]), 'nu': float(codec.decode(zt0[s][None])[0, 1])} for s in names}}
    hom = np.isin(scene_of, names)
    ens['law'] = {'majority_vote_jelly_frac_all_triplets': float((maj[hom] == 0).float().mean()),
                  'majority_vote_per_scene_t0': {s: LAW_NAMES[int(maj[np.where((scene_of == s) & (trip == 0))[0][0]])] for s in names + others},
                  'mean_prob_t0': {s: torch.stack([p[np.where((scene_of == s) & (trip == 0))[0][0]] for p in all_law]).mean(0).tolist() for s in names + others}}
    # heterogeneous scene(s): K0 prediction only
    het = {s: {'K0_ensemble_z_t0': torch.stack([torch.tensor(per_run[r]['K0_z'][f'springgaus/{s}/black/t0']) for r in RUNS]).mean(0).tolist()} for s in others}
    # per-run spread of K0 spearman
    k0s = [per_run[r]['summary']['K0']['E_over_rho_spearman'] for r in RUNS]
    report = {
        'protocol': {
            'source': 'Spring-Gaus (Zhong et al., ECCV 2024) released mpm_synthetic.zip (Google Drive id 1HKYOPG9gNDSmfLkhUo5W0Lw9sM8LGGbm); render/<scene>/{camera.json,frame.json,physical.json,camera_xxx/NNN.png}',
            'note_engine': 'Spring-Gaus synthetic sequences are MPM simulations (README: "generated using Material Point Method"), elastic only, rendered in Blender; Spring-Gaus itself fits a spring-mass model. Their MPM solver/constitutive model/ground friction differ from ours, so rankings and calibrated error are the target, not raw equality.',
            'gt': {s: index['scenes'][s]['physical'][0] for s in index['scenes']}, 'rho_assumed_kg_m3': RHO,
            'scored_scenes': names, 'excluded': {s: 'heterogeneous E/nu field, no scalar GT' for s in others},
            'views': {s: {k: index['scenes'][s][k] for k in ('triplets', 'min_azimuth_gap_deg', 'look_at_center')} for s in index['scenes']},
            'frames': '30 frames (dt 0.03 s, 0-0.87 s) -> 64 by nearest repetition; 512->224 px; RGBA composited on black; V-JEPA2 tubelet 2 -> 32 steps',
            'models': 'V2 main 9 runs ' + ', '.join(RUNS) + ' (best_material.pt, val-selected), no retraining',
            'ensemble': 'mean z over 9 runs (per mode: K0/offset/ridge computed per run then averaged); law = majority vote of run argmax',
            'adaptation': 'leave-one-scene-out over the scored scenes; support = other scenes x 3 triplets; offset = mean support residual; ridge_last_layer = closed-form ridge delta on the last Linear of param_head (lambda = 1.0 * mean ||h||^2, bias unpenalised); support_mean_baseline = mean support GT z, no video. Hyperparameters copied from v3c PAC-NeRF check, not tuned.',
            'law': 'all scenes elastic -> GT law = jelly; law-head refit not possible (single class), only K0 law reported',
            'metrics': 'Spearman / pairwise order of E/rho (z) vs GT over scored scenes (ties excluded from pairs); Spearman of nu; MAE in train-standardised z (V2 stats) for E/rho and nu'},
        'ensemble9': ens, 'heterogeneous': het,
        'K0_spearman_E_per_run': dict(zip(RUNS, k0s)),
        'per_run': per_run}
    Path(a.report).parent.mkdir(parents=True, exist_ok=True)
    Path(a.report).write_text(json.dumps(report, indent=2, default=float) + '\n')
    print(json.dumps({m: ens[m]['canonical_t0'] for m in MODES}, indent=1))
    print(json.dumps(ens['law'], indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='stage', required=True)
    b = sub.add_parser('build_syn'); b.add_argument('zip'); b.add_argument('out')
    e = sub.add_parser('eval_syn'); e.add_argument('out'); e.add_argument('report')
    a = p.parse_args()
    {'build_syn': build_syn, 'eval_syn': eval_syn}[a.stage](a)
