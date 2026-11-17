"""V3a: cross-engine few-shot (Warp-MPM V2 -> libuipc FEM) on frozen V-JEPA 2 ViT-L features.

Source models: runs/v2main_C_seed{17,29,43}/best_material.pt (heads trained on MPM V2 cached features,
target_space=specific, i.e. ln1p(E/(rho/1000)) standardized with the MPM train stats). uipc has rho=1000,
so the specific target equals ln1p(E); every method below predicts in that SAME standardized MPM space.

Arms, per seed s (source model seed s, K-subset drawn with seed s, K=8 subset nested in K=32):
  zero_shot      K=0: source model as is.
  affine_K       K-shot affine recalibration of the source outputs (E and nu separately, least squares).
  headft_K       K-shot fine-tune of the source param_head only (everything else frozen; equivalent to
                 training the 2-layer MLP on the frozen pooled embedding). Fixed schedule, no val selection.
  scratch_K      new light head (same architecture as the source, random init) trained on K uipc cases only.
  engcond_K      engine-conditioned head: source param_head + learned engine embedding (added to the frozen
                 pooled representation) + per-engine output affine on (lnE, nu), trained JOINTLY on N MPM jelly
                 train cases (engine 0, v2main cache) and the K uipc cases (engine 1), 50/50 batches.
  trainmean_K    constant predictor = mean of the K training targets (reference floor).
K in {4, 8, 16, 32, all(=full uipc supervision)}. Evaluation: all uipc val cases (held-out geometries).
Metrics: MAE of standardized ln(E/rho) (MPM-standardized units and raw ln units), MAE of nu,
Spearman of predicted vs true E over val (and mean within-geometry Spearman), mean signed E error.

  .venv/bin/python scripts/v3/uipc_fewshot.py --out docs/v3a/UIPC_FEWSHOT.json
"""
import argparse
import copy
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fieldbridge.config import load_config
from fieldbridge.data import read_manifest, specific_parameters
from fieldbridge.local_model import build_model
from fieldbridge.runtime import load_weights, sha256

ROOT = Path(__file__).resolve().parents[2]
DEV = torch.device('cuda')


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return float('nan')
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def load_uipc(prepared, manifest):
    rows = read_manifest(manifest)
    index = json.loads((Path(prepared) / 'index.json').read_text())
    if index['manifest_sha256'] != sha256(manifest):
        raise ValueError('uipc cache does not match the manifest')
    data = {}
    for split in ('train', 'val'):
        rs = [r for r in rows if r['split'] == split]
        feats, valid, times = [], [], []
        for r in rs:
            s = torch.load(Path(prepared) / index['files'][r['sample_id']], weights_only=True)
            if s['sample_id'] != r['sample_id']:
                raise ValueError('identity mismatch')
            feats.append(s['features'].to(torch.bfloat16)); valid.append(s['valid']); times.append(s['times'])
        data[split] = dict(rows=rs, features=torch.stack(feats).to(DEV), valid=torch.stack(valid), times=torch.stack(times),
                           E=np.array([r['parameters']['E'] for r in rs]), nu=np.array([r['parameters']['nu'] for r in rs]),
                           rho=np.array([r['parameters']['rho'] for r in rs]), geometry=[r['geometry'] for r in rs])
    return data


def targets(d, stats):
    lnE = np.log1p(d['E'] / (d['rho'] / 1000.))     # specific space, identical to the MPM source target space
    return np.stack([(lnE - stats['mean'][0]) / stats['std'][0], (d['nu'] - stats['mean'][1]) / stats['std'][1]], 1).astype(np.float32)


def run_model(model, d, idx, grad=False, batch=8):
    """Returns (mean[:, :2], joint) for samples idx. Features stay on CPU; batches move to GPU."""
    means, joints = [], []
    captured = {}
    hook = model.param_head.register_forward_hook(lambda m, i, o: captured.__setitem__('joint', i[0]))
    try:
        for start in range(0, len(idx), batch):
            j = torch.as_tensor(np.asarray(idx[start:start + batch]), dtype=torch.long)
            f = d['features'][j.to(d['features'].device)]
            b, v, t = f.shape[:3]
            with torch.autocast('cuda', torch.bfloat16), torch.set_grad_enabled(grad):
                out = model(torch.zeros(b, v, t, 3, 1, 1, device=DEV), d['valid'][j].to(DEV), d['times'][j].to(DEV),
                            torch.zeros(b, 4, device=DEV), torch.zeros(b, 4, dtype=torch.bool, device=DEV), features=f)
            means.append(out['mean'][:, :2].float()); joints.append(captured['joint'].float())
    finally:
        hook.remove()
    return torch.cat(means), torch.cat(joints)


def metrics(pred, d, stats, idx=None):
    idx = np.arange(len(d['E'])) if idx is None else idx
    y = targets(d, stats)[idx]; pred = np.asarray(pred, float)
    err = pred[:, 0] - y[:, 0]
    geoms = np.array(d['geometry'])[idx]
    within = [spearman(pred[geoms == g, 0], y[geoms == g, 0]) for g in sorted(set(geoms))]
    return dict(n=int(len(idx)), mae_lnE_std=float(np.abs(err).mean()), mae_lnE_raw=float(np.abs(err).mean() * stats['std'][0]),
                bias_lnE_std=float(err.mean()), mae_nu=float(np.abs(pred[:, 1] - y[:, 1]).mean() * stats['std'][1]),
                spearman_E=spearman(pred[:, 0], y[:, 0]), spearman_E_within_geometry_mean=float(np.nanmean(within)) if within else float('nan'),
                spearman_E_within_geometry={g: s for g, s in zip(sorted(set(geoms)), within)})


def affine_fit(p, y):
    out = np.zeros_like(p)
    coefs = []
    for c in range(2):
        A = np.stack([p[:, c], np.ones(len(p))], 1)
        w = np.linalg.lstsq(A, y[:, c], rcond=None)[0]
        coefs.append(w)
    return coefs


def affine_apply(coefs, p):
    return np.stack([p[:, c] * coefs[c][0] + coefs[c][1] for c in range(2)], 1)


def head_finetune(head, joint, y, steps, lr, seed):
    torch.manual_seed(seed)
    head = copy.deepcopy(head).float().train()
    for p in head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.05)
    y = torch.as_tensor(y, device=DEV)
    for _ in range(steps):
        batch = torch.randperm(len(y), device=DEV)[:32]
        loss = (head(joint[batch])[:, :2] - y[batch]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    return head


def scratch_train(model_cfg, d, idx, y, steps, lr, seed):
    torch.manual_seed(seed)
    model = build_model(model_cfg).to(DEV).train()
    params = [p for n, p in model.named_parameters() if p.requires_grad and not n.startswith('frame_encoder.')]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    yt = torch.as_tensor(y, device=DEV)
    g = np.random.default_rng(seed)
    for _ in range(steps):
        pick = g.choice(len(idx), size=min(8, len(idx)), replace=False)
        mean, _ = run_model(model, d, idx[pick], grad=True, batch=8)
        loss = (mean - yt[pick]).pow(2).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
    return model.eval()


class EngineHead(nn.Module):
    def __init__(self, head, engines=2):
        super().__init__()
        self.head = copy.deepcopy(head).float()
        self.embed = nn.Parameter(torch.zeros(engines, head[0].in_features))
        self.scale = nn.Parameter(torch.ones(engines, 2)); self.bias = nn.Parameter(torch.zeros(engines, 2))

    def forward(self, joint, engine):
        return self.head(joint + self.embed[engine])[:, :2] * self.scale[engine] + self.bias[engine]


def engine_cond_train(head, jm, ym, ju, yu, steps, lr, seed):
    torch.manual_seed(seed)
    m = EngineHead(head).to(DEV).train()
    for p in m.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW([{'params': m.head.parameters(), 'weight_decay': 0.05},
                             {'params': [m.embed, m.scale, m.bias], 'weight_decay': 0.}], lr=lr)
    ym, yu = torch.as_tensor(ym, device=DEV), torch.as_tensor(yu, device=DEV)
    for _ in range(steps):
        bm = torch.randperm(len(ym), device=DEV)[:32]; bu = torch.randperm(len(yu), device=DEV)[:32]
        loss = 0.5 * (m(jm[bm], 0) - ym[bm]).pow(2).mean() + 0.5 * (m(ju[bu], 1) - yu[bu]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return m.eval()


def mpm_joints(models, stats, manifest, prepared, n, seed=0):
    """Frozen pooled representations of N MPM jelly train cases under every source model (one disk pass)."""
    rows = [r for r in read_manifest(manifest) if r['split'] == 'train' and r.get('law_id') == 0]
    rng = np.random.default_rng(seed)
    rows = [rows[i] for i in sorted(rng.choice(len(rows), min(n, len(rows)), replace=False))]
    index = json.loads((Path(prepared) / 'index.json').read_text())
    if index['manifest_sha256'] != sha256(manifest):
        raise ValueError('MPM cache does not match its manifest')
    joints = {s: [] for s in models}; ys = []
    load = lambda r: torch.load(Path(prepared) / index['files'][r['sample_id']], weights_only=True)
    with ThreadPoolExecutor(8) as pool:
        for start in range(0, len(rows), 16):
            chunk = rows[start:start + 16]
            samples = list(pool.map(load, chunk))
            d = dict(features=torch.stack([x['features'].to(torch.bfloat16) for x in samples]).to(DEV),
                     valid=torch.stack([x['valid'] for x in samples]), times=torch.stack([x['times'] for x in samples]))
            for s, m in models.items():
                joints[s].append(run_model(m, d, np.arange(len(chunk)))[1])
            for r in chunk:
                p = specific_parameters(r['parameters'])
                ys.append([(math.log1p(p['E']) - stats['mean'][0]) / stats['std'][0], (p['nu'] - stats['mean'][1]) / stats['std'][1]])
            del d
    return {s: torch.cat(v) for s, v in joints.items()}, np.array(ys, np.float32), [r['sample_id'] for r in rows]


def decode(pred, d, stats):
    lnE = pred[:, 0] * stats['std'][0] + stats['mean'][0]
    return dict(E_pa=(np.expm1(lnE) * d['rho'] / 1000.).tolist(),
                nu=np.clip(pred[:, 1] * stats['std'][1] + stats['mean'][1], 0.0, 0.49).tolist())


def k_to_reach(summary, ks, factors=(1.1, 1.25, 1.5)):
    ref_arm = min((a for a in summary if a.endswith('_all') and not a.startswith(('trainmean', 'affine'))),
                  key=lambda a: summary[a]['mae_lnE_std']['mean'])
    ref = summary[ref_arm]['mae_lnE_std']['mean']; out = dict(reference_arm=ref_arm, reference_mae_lnE_std=ref, families={})
    for fam in ('affine', 'headft', 'engcond', 'scratch', 'trainmean'):
        curve = [(0, summary['zero_shot']['mae_lnE_std']['mean'])] if fam in ('affine', 'headft', 'engcond') else []
        curve += [(summary[f'{fam}_{k}']['K'], summary[f'{fam}_{k}']['mae_lnE_std']['mean']) for k in ks if f'{fam}_{k}' in summary]
        out['families'][fam] = dict(curve=curve, **{f'K_within_{f:.2f}x': next((k for k, v in curve if v <= f * ref), None) for f in factors})
    return out


def stratified_order(geoms, seed):
    """Permutation of train indices that cycles through geometries, so every prefix is geometry balanced."""
    rng = np.random.default_rng(seed)
    groups = {g: list(rng.permutation([i for i, x in enumerate(geoms) if x == g])) for g in sorted(set(geoms))}
    names = list(rng.permutation(sorted(groups)))
    order = []
    while any(groups.values()):
        for gname in names:
            if groups[gname]:
                order.append(int(groups[gname].pop()))
    return np.array(order)


def main():
    a = argparse.ArgumentParser()
    a.add_argument('--manifest', default='data/manifests/v3_uipc.jsonl')
    a.add_argument('--prepared', default='data/prepared/v3_uipc_vjl')
    a.add_argument('--source-config', default='configs/v2main/v2main_C_seed{seed}.json')
    a.add_argument('--source-ckpt', default='runs/v2main_C_seed{seed}/best_material.pt')
    a.add_argument('--seeds', default='17,29,43'); a.add_argument('--ks', default='4,8,16,32,all')
    a.add_argument('--mpm-manifest', default='data/manifests/v2main.jsonl'); a.add_argument('--mpm-prepared', default='data/prepared/v2main_vjl')
    a.add_argument('--mpm-n', type=int, default=256)
    a.add_argument('--headft-steps', type=int, default=300); a.add_argument('--headft-lr', type=float, default=3e-4)
    a.add_argument('--scratch-steps', type=int, default=600); a.add_argument('--scratch-lr', type=float, default=3e-4)
    a.add_argument('--out', default='docs/v3a/UIPC_FEWSHOT.json'); a.add_argument('--tag', default='')
    args = a.parse_args()
    started = time.time()
    data = load_uipc(ROOT / args.prepared, ROOT / args.manifest)
    tr, va = data['train'], data['val']
    results = dict(protocol=dict(
        source='Warp-MPM V2 (v2main) head on frozen V-JEPA2 ViT-L cached features, model C (aux fields, no feedback), best_material.pt',
        target='libuipc 0.0.29 StableNeoHookean FEM, BDF1, cantilever tip load ramp-hold-release, 3 views x 64 frames 224px',
        target_space='ln1p(E/(rho/1000)) standardized with MPM v2main train stats; nu standardized likewise (metrics reported raw)',
        uipc_train=len(tr['E']), uipc_val=len(va['E']), train_geometries=sorted(set(tr['geometry'])), val_geometries=sorted(set(va['geometry'])),
        E_range_pa=[float(min(tr['E'].min(), va['E'].min())), float(max(tr['E'].max(), va['E'].max()))],
        headft=dict(steps=args.headft_steps, lr=args.headft_lr, batch=32, weight_decay=0.05, loss='MSE on standardized (lnE, nu)'),
        scratch=dict(steps=args.scratch_steps, lr=args.scratch_lr, batch=8, cosine=True, loss='MSE on standardized (lnE, nu)'),
        engcond=dict(steps=args.headft_steps, lr=args.headft_lr, batch='32 MPM + 32 uipc', mpm_cases=args.mpm_n, mpm_law='jelly (law_id 0), v2main train',
                     conditioning='learned engine embedding added to frozen pooled representation + per-engine output affine (lnE, nu)'),
        model_selection='none: fixed schedules chosen a priori; val is used only for reporting', ks=args.ks, seeds=args.seeds),
        runs={}, tag=args.tag)
    seeds = [int(s) for s in args.seeds.split(',')]
    sources = {}
    for seed in seeds:
        cfg = load_config(ROOT / args.source_config.format(seed=seed))
        stats = json.loads((ROOT / cfg.data.stats).read_text())
        if stats.get('target_space') != 'specific':
            raise ValueError('source must be trained in the specific target space')
        ckpt_path = ROOT / args.source_ckpt.format(seed=seed)
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model = build_model(cfg.model)
        load_weights(model, ckpt, sha256(ROOT / cfg.model.weights))
        sources[seed] = (cfg, stats, ckpt_path, ckpt.get('epoch'), model.to(DEV).eval())
        del ckpt
    if len({json.dumps(v[1]) for v in sources.values()}) != 1:
        raise ValueError('source seeds must share the MPM statistics')
    jm_all, ym, mpm_ids = mpm_joints({s: v[4] for s, v in sources.items()}, sources[seeds[0]][1], ROOT / args.mpm_manifest,
                                     ROOT / args.mpm_prepared, args.mpm_n)
    results['protocol']['engcond']['mpm_sample_ids_sha256'] = __import__('hashlib').sha256('\n'.join(mpm_ids).encode()).hexdigest()
    print('mpm joints', len(ym), round(time.time() - started), 's', flush=True)
    for seed in seeds:
        cfg, stats, ckpt_path, epoch, model = sources[seed]
        ytr, yva = targets(tr, stats), targets(va, stats)
        ptr, jtr = run_model(model, tr, np.arange(len(ytr)))
        pva, jva = run_model(model, va, np.arange(len(yva)))
        ptr_np, pva_np = ptr.cpu().numpy(), pva.cpu().numpy()
        run = dict(source_checkpoint=str(ckpt_path.relative_to(ROOT)), source_checkpoint_sha256=sha256(ckpt_path),
                   source_epoch=epoch, arms={}, val_sample_ids=[r['sample_id'] for r in va['rows']])
        run['arms']['zero_shot'] = dict(K=0, val=metrics(pva_np, va, stats), train=metrics(ptr_np, tr, stats), val_pred=decode(pva_np, va, stats))
        order = stratified_order(tr['geometry'], seed)
        for kname in args.ks.split(','):
            k = len(order) if kname == 'all' else int(kname)
            idx = np.sort(order[:k])
            run.setdefault('subsets', {})[kname] = [tr['rows'][i]['sample_id'] for i in idx]
            coefs = affine_fit(ptr_np[idx], ytr[idx])
            run['arms'][f'affine_{kname}'] = dict(K=k, val=metrics(affine_apply(coefs, pva_np), va, stats),
                                                  coef=[list(map(float, c)) for c in coefs])
            head = head_finetune(model.param_head, jtr[idx], ytr[idx], args.headft_steps, args.headft_lr, seed)
            with torch.no_grad():
                p = head(jva)[:, :2].cpu().numpy()
            run['arms'][f'headft_{kname}'] = dict(K=k, val=metrics(p, va, stats), val_pred=decode(p, va, stats))
            eh = engine_cond_train(model.param_head, jm_all[seed], ym, jtr[idx], ytr[idx], args.headft_steps, args.headft_lr, seed)
            with torch.no_grad():
                p = eh(jva, 1).cpu().numpy()
                pm = eh(jm_all[seed], 0).cpu().numpy()
            run['arms'][f'engcond_{kname}'] = dict(K=k, val=metrics(p, va, stats), val_pred=decode(p, va, stats),
                                                   mpm_train_fit_mae_lnE_std=float(np.abs(pm[:, 0] - ym[:, 0]).mean()),
                                                   engine_affine=dict(scale=eh.scale.detach().cpu().tolist(), bias=eh.bias.detach().cpu().tolist()))
            scratch = scratch_train(cfg.model, tr, idx, ytr[idx], args.scratch_steps, args.scratch_lr, seed)
            p, _ = run_model(scratch, va, np.arange(len(yva)))
            p = p.detach().cpu().numpy()
            run['arms'][f'scratch_{kname}'] = dict(K=k, val=metrics(p, va, stats), val_pred=decode(p, va, stats))
            del scratch; torch.cuda.empty_cache()
            const = np.repeat(ytr[idx].mean(0, keepdims=True), len(yva), 0)
            run['arms'][f'trainmean_{kname}'] = dict(K=k, val=metrics(const, va, stats))
            print(seed, kname, {n: round(v['val']['mae_lnE_std'], 3) for n, v in run['arms'].items()}, flush=True)
        results['runs'][str(seed)] = run
        del jm_all[seed]; torch.cuda.empty_cache()
    summary = {}
    arms = list(next(iter(results['runs'].values()))['arms'])
    for arm in arms:
        vals = [r['arms'][arm]['val'] for r in results['runs'].values()]
        summary[arm] = dict(K=next(iter(results['runs'].values()))['arms'][arm]['K'],
                            **{m: dict(mean=float(np.nanmean([v[m] for v in vals])), std=float(np.nanstd([v[m] for v in vals])))
                               for m in ('mae_lnE_std', 'mae_lnE_raw', 'bias_lnE_std', 'mae_nu', 'spearman_E', 'spearman_E_within_geometry_mean')})
    results['summary'] = summary
    if any(a.endswith('_all') for a in summary):
        results['k_to_reach_full_supervision'] = k_to_reach(summary, args.ks.split(','))
    results['wall_seconds'] = time.time() - started
    out = ROOT / args.out; out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix('.tmp'); tmp.write_text(json.dumps(results, indent=1)); tmp.replace(out)
    for arm, s in summary.items():
        print(f"{arm:16s} K={s['K']:4d}  MAE lnE(std) {s['mae_lnE_std']['mean']:.3f}+-{s['mae_lnE_std']['std']:.3f}  "
              f"nu {s['mae_nu']['mean']:.3f}  rho_S {s['spearman_E']['mean']:.3f}  within {s['spearman_E_within_geometry_mean']['mean']:.3f}")


if __name__ == '__main__':
    main()
