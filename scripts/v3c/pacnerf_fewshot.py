"""V3c: PAC-NeRF external 4-scene check (qualitative; torus, bird, playdoh, cat).

build: frames from the extracted copy (scripts/v3c/pacnerf_extract.py, untrusted input -> only PNG/JSON
       decoding here), 3 of 11 cameras (max-min azimuth spread; canonical + 2 disjoint extra triplets),
       14 frames -> 64 by nearest-frame repetition, 800 -> 224 px, two variants: 'raw' (rendered
       backgrounds) and 'bgsub' (object on black like V2, mask = |frame - background plate r_v_-1|).
       V-JEPA 2 ViT-L features exactly as scripts/prepare_vjepa_cache.py (64 frames, tubelet 2 -> 32 steps).
eval:  V2 main C models (3 seeds, best_material.pt); K=0 transfer + leave-one-scene-out adaptation
       (offset calibration; ridge on the last parameter layer; law-head refit), hyperparameters fixed a priori.
GT (PAC-NeRF paper Table 1): torus E=1e6 nu=.3; bird E=3e5 nu=.3; playdoh E=2e6 nu=.3 tauY=1.54e4;
cat E=1e6 nu=.3 tauY=3.85e3. Density is not published; rho=1000 is ASSUMED (equal for all scenes),
so E/rho ordering is assumption-free only up to a common density.
"""
import argparse, itertools, json, math, sys
from pathlib import Path
import numpy as np
import torch

SCENES = {'torus': dict(E=1e6, nu=.3, sigma_y=None, law='elastic'),
          'bird': dict(E=3e5, nu=.3, sigma_y=None, law='elastic'),
          'playdoh': dict(E=2e6, nu=.3, sigma_y=1.54e4, law='elastoplastic'),
          'cat': dict(E=1e6, nu=.3, sigma_y=3.85e3, law='elastoplastic')}
RHO = 1000.
SRC = Path('data/external_datasets/pacnerf_v3c_extract/data')
OUT = Path('data/prepared/v3c_pacnerf_vjl')
STATS = 'data/manifests/v2main.specific.stats.json'
MEAN = torch.tensor([.485, .456, .406]).view(1, 1, 3, 1, 1)
STD = torch.tensor([.229, .224, .225]).view(1, 1, 3, 1, 1)
LAW_NAMES = ['jelly', 'metal', 'plasticine']
MODES = ('K0', 'offset', 'ridge_last_layer', 'support_mean_baseline')  # last: GT-only reference (no video)


def cameras(scene):
    meta = json.loads((SRC / scene / 'all_data.json').read_text())
    pos = {}
    for e in meta:
        name = Path(e['file_path']).name  # r_<v>_<f>.png
        v = int(name.split('_')[1]); pos[v] = np.array(e['c2w'])[:3, 3]
    az = {v: math.degrees(math.atan2(p[2], p[0])) for v, p in pos.items()}
    el = {v: float(p[1]) for v, p in pos.items()}
    return az, el


def triplets(az):
    def gap(t):
        a = sorted(az[v] for v in t); d = [b - a_ for a_, b in zip(a, a[1:])] + [360 - a[-1] + a[0]]
        return min(d)
    left, chosen = sorted(az), []
    for _ in range(3):
        best = max(itertools.combinations(left, 3), key=lambda t: (round(gap(t), 3), -sum(t)))
        chosen.append(list(best)); left = [v for v in left if v not in best]
    return chosen, [gap(t) for t in chosen]


def load_rgb(scene, views, variant, threshold=.12):
    from PIL import Image, ImageFilter
    idx = np.round(np.linspace(0, 13, 64)).astype(int)
    out = np.zeros((len(views), 64, 3, 224, 224), np.float32)
    for i, v in enumerate(views):
        bg = np.asarray(Image.open(SRC / scene / 'data' / f'r_{v}_-1.png').convert('RGB'), np.float32) / 255
        frames = []
        for f in range(14):
            im = np.asarray(Image.open(SRC / scene / 'data' / f'r_{v}_{f}.png').convert('RGB'), np.float32) / 255
            if variant == 'bgsub':
                mask = (np.abs(im - bg).max(-1) > threshold).astype(np.uint8) * 255
                mask = np.asarray(Image.fromarray(mask).filter(ImageFilter.MedianFilter(7)), np.float32)[..., None] / 255
                im = im * mask
            frames.append(np.asarray(Image.fromarray((im * 255).round().astype(np.uint8)).resize((224, 224), Image.BICUBIC), np.float32) / 255)
        out[i] = np.stack([frames[k] for k in idx]).transpose(0, 3, 1, 2)
    times = np.linspace(0, 13 / 24, 64).astype(np.float32)
    return torch.from_numpy(out), torch.from_numpy(times)


def target(scene):
    from fieldbridge.data import ParameterCodec, specific_parameters
    codec = ParameterCodec(json.loads(Path(STATS).read_text()))
    p = dict(SCENES[scene]); p = {'E': p['E'], 'nu': p['nu'], 'rho': RHO, 'sigma_y': p['sigma_y']}
    t, m = codec.encode(specific_parameters(p))
    m[2] = False  # density is assumed, never scored
    return t, m


def build(a):
    from transformers import AutoModel
    from fieldbridge.train import atomic_save
    OUT.mkdir(parents=True, exist_ok=True)
    model = AutoModel.from_pretrained('weights/hf/facebook__vjepa2-vitl-fpc64-256', dtype=torch.bfloat16).cuda().eval()
    index = {'scenes': {}, 'rho_assumed': RHO, 'files': {}}
    for scene in SCENES:
        az, el = cameras(scene); chosen, gaps = triplets(az)
        index['scenes'][scene] = {'azimuth_deg': az, 'height': el, 'triplets': chosen, 'min_azimuth_gap_deg': gaps}
        t, m = target(scene)
        for variant in ('raw', 'bgsub'):
            for k, views in enumerate(chosen):
                rgb, times = load_rgb(scene, views, variant)
                if k == 0 and variant == 'bgsub':
                    from PIL import Image
                    strip = (rgb[:, ::16].permute(0, 3, 1, 4, 2).reshape(3 * 224, -1, 3).numpy() * 255).astype(np.uint8)
                    Image.fromarray(strip).save(OUT / f'preview_{scene}_{variant}.png')
                with torch.no_grad():
                    x = ((rgb[None] - MEAN) / STD)[0].cuda().to(torch.bfloat16)
                    tokens = model(pixel_values_videos=x, skip_predictor=True).last_hidden_state
                feats = tokens.reshape(3, 32, 14, 14, -1).permute(0, 1, 4, 2, 3).cpu()
                tt = ((times[0::2] + times[1::2]) / 2).repeat(3, 1)
                sid = f'pacnerf/{scene}/{variant}/t{k}'
                sample = {'sample_id': sid, 'object_id': scene, 'source': 'PAC-NeRF', 'action': 'drop', 'views': views,
                          'features': feats, 'times': tt, 'valid': torch.ones(3, 32, dtype=torch.bool),
                          'rgb': torch.zeros(3, 32, 3, 1, 1), 'prior': torch.zeros(4), 'prior_mask': torch.zeros(4, dtype=torch.bool),
                          'target': t, 'target_mask': m}
                name = sid.replace('/', '_') + '.pt'; atomic_save(sample, OUT / name); index['files'][sid] = name
                print(sid, views, flush=True)
    (OUT / 'index.json').write_text(json.dumps(index, indent=2))


def load_model(seed, device):
    from fieldbridge.config import load_config
    from fieldbridge.local_model import build_model
    from fieldbridge.runtime import load_weights, sha256
    cfg = load_config(f'configs/v2main/v2main_C_seed{seed}.json')
    model = build_model(cfg.model)
    ckpt = torch.load(Path(cfg.train.output) / 'best_material.pt', map_location='cpu', weights_only=False)
    load_weights(model, ckpt, sha256(cfg.model.weights))
    return model.to(device).eval(), {'checkpoint_epoch': ckpt.get('epoch'), 'best_material': ckpt.get('best_material')}


def run(model, samples, device):
    """Returns mean z [N,4], law logits [N,3], hidden h (input of last param layer) [N,d], joint [N,3d]."""
    cache = {}
    h1 = model.param_head[1].register_forward_hook(lambda m, i, o: cache.__setitem__('h', o.detach().float()))
    h2 = model.law_head.register_forward_hook(lambda m, i, o: cache.__setitem__('j', i[0].detach().float()))
    outs = []
    with torch.no_grad():
        for s in samples:
            batch = {k: (s[k][None].to(device) if torch.is_tensor(s[k]) else s[k]) for k in ['rgb', 'valid', 'times', 'prior', 'prior_mask', 'features']}
            batch['features'] = batch['features'].float()
            y = model(**batch)
            outs.append((y['mean'][0].float().cpu(), y['law_logits'][0].float().cpu(), cache['h'][0].cpu(), cache['j'][0].cpu()))
    h1.remove(); h2.remove()
    return [torch.stack(x) for x in zip(*outs)]


def spearman(x, y):
    from scipy.stats import spearmanr
    return float(spearmanr(x, y).statistic)


def pair_order(pred, gt):
    pairs = [(i, j) for i in range(len(gt)) for j in range(i + 1, len(gt)) if gt[i] != gt[j]]
    return float(np.mean([(pred[i] - pred[j]) * (gt[i] - gt[j]) > 0 for i, j in pairs])), len(pairs)


def evaluate(a):
    from fieldbridge.data import ParameterCodec
    codec = ParameterCodec(json.loads(Path(STATS).read_text()))
    index = json.loads((OUT / 'index.json').read_text())
    samples = {sid: torch.load(OUT / f, weights_only=True) for sid, f in index['files'].items()}
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    names = list(SCENES); gtE = [SCENES[s]['E'] / (RHO / 1000) for s in names]
    LAMBDA_REL, LAW_STEPS, LAW_LR, LAW_L2 = 1.0, 100, 1e-2, 1e-2  # fixed a priori, no tuning on query scenes
    report = {'protocol': {
        'scenes': SCENES, 'rho_assumed_kg_m3': RHO, 'views': index['scenes'],
        'frames': '14 rendered frames (24 fps, 0-0.54 s) -> 64 by nearest repetition; V-JEPA tubelet 2 -> 32 steps; times in seconds',
        'law_mapping': 'PAC-NeRF elastic -> jelly; PAC-NeRF plasticine (von Mises) -> elastoplastic = metal or plasticine (binary score); 3-class score uses GT=metal (von Mises without softening) as reference',
        'models': 'configs/v2main/v2main_C_seed{17,29,43}.json best_material.pt (val-selected)',
        'adaptation': {'offset': 'z += mean support residual per parameter (E, nu, sigma_y; sigma_y only from support scenes with yield)',
                       'ridge_last_layer': f'last Linear of param_head, delta on mean outputs; ridge lambda = {LAMBDA_REL} * mean ||h||^2, bias unpenalized',
                       'law_head': f'law_head refit with CE on support, {LAW_STEPS} Adam steps lr {LAW_LR}, L2-to-init {LAW_L2}',
                       'support': 'other 3 scenes, all 3 camera triplets; query = held-out scene canonical triplet t0 (also mean over t0-t2)',
                       'support_mean_baseline': 'reference without video: predict the mean GT z of the 3 support scenes (law logits = K0)'},
        'caveat': '4 scenes, 1 sequence each, free fall only; qualitative check, no statistics'}, 'variants': {}}
    for variant in ('raw', 'bgsub'):
        res = {'seeds': {}}
        ens = {}
        for seed in (17, 29, 43):
            model, info = load_model(seed, device)
            sids = [f'pacnerf/{s}/{variant}/t{k}' for s in names for k in range(3)]
            mu, logits, h, j = run(model, [samples[x] for x in sids], device)
            T = torch.stack([samples[x]['target'] for x in sids]); M = torch.stack([samples[x]['target_mask'] for x in sids])
            scene_of = np.array([x.split('/')[1] for x in sids]); trip = np.array([int(x[-1]) for x in sids])
            W0 = model.param_head[2].weight.detach().float().cpu()[:4]; b0 = model.param_head[2].bias.detach().float().cpu()[:4]
            per = {}
            for q in names:
                qi = np.where((scene_of == q))[0]; si = np.where(scene_of != q)[0]; q0 = qi[trip[qi] == 0]
                # offset
                off = torch.zeros(4)
                for p in range(4):
                    ok = M[si, p]
                    if ok.any():
                        off[p] = (T[si][ok, p] - mu[si][ok, p]).mean()
                # ridge (closed form, bias unpenalized via centering)
                ridge = mu[qi].clone()
                lam = LAMBDA_REL * float((h[si] ** 2).sum(1).mean())
                for p in range(4):
                    ok = M[si, p]
                    if ok.sum() < 2:
                        ridge[:, p] = mu[qi, p] + off[p]; continue
                    H = h[si][ok]; r = T[si][ok, p] - mu[si][ok, p]
                    Hm, rm = H.mean(0), r.mean(); Hc, rc = H - Hm, r - rm
                    dw = torch.linalg.solve(Hc.T @ Hc + lam * torch.eye(H.shape[1]), Hc.T @ rc)
                    ridge[:, p] = mu[qi, p] + h[qi] @ dw + (rm - Hm @ dw)
                # law head refit
                law_t = torch.tensor([0 if SCENES[scene_of[i]]['law'] == 'elastic' else 1 for i in si])
                lw = torch.nn.Linear(j.shape[1], 3); lw.load_state_dict({k: v.detach().float().cpu() for k, v in model.law_head.state_dict().items()})
                init = [p.detach().clone() for p in lw.parameters()]; opt = torch.optim.Adam(lw.parameters(), lr=LAW_LR)
                for _ in range(LAW_STEPS):
                    lg = lw(j[si]); lp = lg.log_softmax(-1)
                    # binary elastic vs elastoplastic likelihood (metal+plasticine pooled)
                    nll = -torch.where(law_t == 0, lp[:, 0], torch.logsumexp(lp[:, 1:], -1)).mean()
                    loss = nll + LAW_L2 * sum(((p - i) ** 2).sum() for p, i in zip(lw.parameters(), init))
                    opt.zero_grad(); loss.backward(); opt.step()
                with torch.no_grad():
                    adapted_logits = lw(j[qi])
                def pack(z, lg):
                    z0 = z[trip[qi] == 0][0]
                    prob = lg.softmax(-1)
                    raw = codec.decode(z0[None])[0]
                    return {'z_t0': z0.tolist(), 'z_mean_t012': z.mean(0).tolist(), 'z_sd_t012': z.std(0).tolist(),
                            'abs_err_z_t0': {n: (abs(float(z0[p] - T[q0[0], p])) if bool(M[q0[0], p]) else None) for p, n in enumerate(['E_over_rho', 'nu', 'rho', 'sigma_y_over_rho'])},
                            'decoded_t0': {'E_over_rho_x1000': float(raw[0]), 'nu': float(raw[1]), 'sigma_y_over_rho_x1000': float(raw[3])},
                            'law_prob_t0': dict(zip(LAW_NAMES, prob[trip[qi] == 0][0].tolist())),
                            'law_pred_t0': LAW_NAMES[int(prob[trip[qi] == 0][0].argmax())],
                            'binary_law_correct_t0': (int(prob[trip[qi] == 0][0].argmax()) == 0) == (SCENES[q]['law'] == 'elastic'),
                            'binary_law_correct_t012': float(np.mean([(int(x.argmax()) == 0) == (SCENES[q]['law'] == 'elastic') for x in prob]))}
                sm = torch.stack([T[si][M[si, p], p].mean() if M[si, p].any() else torch.tensor(0.) for p in range(4)])
                per[q] = {'target_z': T[q0[0]].tolist(), 'target_mask': M[q0[0]].tolist(),
                          'support_mean_baseline': pack(sm.expand(len(qi), 4).clone(), logits[qi]),
                          'K0': pack(mu[qi], logits[qi]), 'offset': pack(mu[qi] + off, logits[qi]),
                          'ridge_last_layer': pack(ridge, adapted_logits), 'support_offset_z': off.tolist()}
                ens.setdefault(q, []).append({k: per[q][k]['z_t0'] for k in MODES})
            summ = {}
            for mode in MODES:
                predE = [per[s][mode]['z_t0'][0] for s in names]
                summ[mode] = {'E_over_rho_spearman_vs_gt': spearman(predE, gtE),
                              'E_over_rho_pairwise_order_acc': pair_order(predE, gtE),
                              'sigma_y_order_playdoh_gt_cat': per['playdoh'][mode]['z_t0'][3] > per['cat'][mode]['z_t0'][3],
                              'mean_abs_err_z_E': float(np.mean([per[s][mode]['abs_err_z_t0']['E_over_rho'] for s in names])),
                              'mean_abs_err_z_sigma_y': float(np.mean([per[s][mode]['abs_err_z_t0']['sigma_y_over_rho'] for s in ('playdoh', 'cat')])),
                              'binary_law_acc_t0': float(np.mean([per[s][mode]['binary_law_correct_t0'] for s in names])),
                              'law_pred_t0': {s: per[s][mode]['law_pred_t0'] for s in names}}
            res['seeds'][str(seed)] = {'model': info, 'scenes': per, 'summary': summ}
        # seed-ensemble (mean z) order summary
        ens_summary = {}
        for mode in MODES:
            z = {s: np.mean([e[mode] for e in ens[s]], 0) for s in names}
            predE = [z[s][0] for s in names]
            ens_summary[mode] = {'z_t0': {s: z[s].tolist() for s in names},
                                 'E_over_rho_spearman_vs_gt': spearman(predE, gtE),
                                 'E_over_rho_pairwise_order_acc': pair_order(predE, gtE),
                                 'sigma_y_order_playdoh_gt_cat': bool(z['playdoh'][3] > z['cat'][3])}
        res['seed_ensemble'] = ens_summary
        report['variants'][variant] = res
    Path(a.output).write_text(json.dumps(report, indent=2, default=lambda o: o if not isinstance(o, (np.bool_,)) else bool(o)) + '\n')
    print(json.dumps({v: {s: r['seeds'][s]['summary'] for s in r['seeds']} for v, r in report['variants'].items()}, indent=1, default=str))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['build', 'eval']); p.add_argument('--output', default='docs/v3c/PACNERF_FEWSHOT.json')
    a = p.parse_args(); build(a) if a.stage == 'build' else evaluate(a)
