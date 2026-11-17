"""V5m multimodal prior: vision + touch + action -> material (B variant of the V2 prior recipe, re-implemented on
pre-pooled features so that all modalities share one token sequence).

Tokens (d=256, + modality embedding):
  vision : Linear(1024->d) on spatially averaged V-JEPA features (== the 1x1 conv + spatial mean of the B head) + the
           same sinusoidal time encoding (seconds); one token per (view, time step), padding-masked.
  touch  : per body, channels sign*log1p(|F_xyz|/F0) and log1p(|F|/F0) over 400 x 10 ms -> Conv1d(k=25, s=12) -> 32 tokens,
           + time encoding + that body's action embedding; plate: (sign*log1p(|F|/F0p), plate gap) -> Conv1d -> 32 tokens.
  action : MLP(12->d) per body, plate vector, drop impulse, scene vector (bbox size, gravity, action type).
Encoder: the V2 2-layer pre-norm transformer (width 256, 8 heads, ff 1024, GELU); masked mean pooling over all tokens;
heads: param MLP -> (mu, logvar) for 5 targets [E/rho, nu, rho, sigma_y/rho, E] (train-standardized ln1p), law Linear.
Loss/optimizer = V2 recipe: smooth-l1 + 0.05 Gaussian NLL + 0.1 law CE (masked); AdamW lr 3e-4, wd 0.05, batch 8,
bf16 autocast, grad clip 1.0, <=30 epochs, early stop on val z-MAE (all 5 targets) patience 5 / min 8 epochs / rel 0.002;
checkpoint = best val epoch. Arms choose the modalities:
  vision | vision_action | vision_touch_action | touch_action ; --mix-v5f adds the 40 V5f force-press train samples.
  .venv/bin/python scripts/v5m/train_mm.py --arm vision_touch_action --seed 29 --out runs/v5m_vta_seed29
"""
import argparse, json, math, random, time
from pathlib import Path
import numpy as np, torch
from torch import nn
import torch.nn.functional as Fn
ROOT = Path(__file__).resolve().parents[2]
ARMS = {'vision': ('vision',), 'vision_action': ('vision', 'action'), 'vision_touch_action': ('vision', 'touch', 'action'), 'touch_action': ('touch', 'action')}


def tenc(t, d):
    freq = torch.exp(torch.arange(0, d, 2, device=t.device).float() * (-math.log(10000.) / d))
    ph = t[..., None] * freq
    return torch.stack((ph.sin(), ph.cos()), -1).flatten(-2)


class MM(nn.Module):
    def __init__(self, mods, d=256, heads=8, layers=2, K=5):
        super().__init__()
        self.mods, self.d = set(mods), d
        self.vis = nn.Linear(1024, d)
        self.fconv = nn.Conv1d(4, d, 25, 12, padding=12); self.pconv = nn.Conv1d(2, d, 25, 12, padding=12)
        mlp = lambda: nn.Sequential(nn.Linear(12, d), nn.GELU(), nn.Linear(d, d))
        self.body, self.platev, self.imp, self.scene = mlp(), mlp(), mlp(), mlp()
        self.mtype = nn.Embedding(6, d)  # vision, body-force, plate-force, body-action, other-action, scene
        layer = nn.TransformerEncoderLayer(d, heads, d * 4, dropout=0., batch_first=True, norm_first=True, activation='gelu')
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.param_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2 * K)); self.law_head = nn.Linear(d, 3)

    def forward(self, b, F0, F0p):
        toks, masks = [], []
        B = b['y'].shape[0]; dev = b['y'].device
        if 'vision' in self.mods:
            v = self.vis(b['vis'].float()).flatten(1, 2) + tenc(b['times'].flatten(1), self.d) + self.mtype.weight[0]
            toks.append(v); masks.append(b['valid'].flatten(1))
        if 'action' in self.mods or 'touch' in self.mods:
            be = self.body(b['bvec'])  # [B, NB, d] (identifies the body for its force tokens)
        if 'touch' in self.mods:
            F = b['F'].permute(0, 2, 3, 1)  # B, NB, 3, T
            ch = torch.cat([F.sign() * torch.log1p(F.abs() / F0), torch.log1p(F.norm(dim=2, keepdim=True) / F0)], 2)
            nb = ch.shape[1]
            ft = self.fconv(ch.flatten(0, 1)).transpose(1, 2).reshape(B, nb, -1, self.d)  # B, NB, S, d
            S = ft.shape[2]; ts = (torch.arange(S, device=dev).float() * 12 + 6) * 0.01
            ft = ft + tenc(ts, self.d) + be[:, :, None] + self.mtype.weight[1]
            toks.append(ft.flatten(1, 2)); masks.append(b['bmask'][:, :, None].expand(B, nb, S).flatten(1))
            P = b['plate'].transpose(1, 2)
            pch = torch.stack([P[:, 0].sign() * torch.log1p(P[:, 0].abs() / F0p), P[:, 1]], 1)
            pt = self.pconv(pch).transpose(1, 2) + tenc(ts, self.d) + self.platev(b['pvec'])[:, None] + self.mtype.weight[2]
            toks.append(pt); masks.append(b['pmask'][:, None].expand(B, S))
        if 'action' in self.mods:
            toks += [be + self.mtype.weight[3], (self.platev(b['pvec']) + self.mtype.weight[4])[:, None], (self.imp(b['ivec']) + self.mtype.weight[4])[:, None],
                     (self.scene(b['scene']) + self.mtype.weight[5])[:, None]]
            masks += [b['bmask'], b['pmask'][:, None], b['imask'][:, None], torch.ones(B, 1, dtype=torch.bool, device=dev)]
        x = torch.cat(toks, 1); m = torch.cat(masks, 1)
        e = self.enc(x, src_key_padding_mask=~m)
        w = m[..., None].to(e.dtype); pooled = (e * w).sum(1) / w.sum(1).clamp_min(1)
        mu, lv = self.param_head(pooled).chunk(2, -1)
        return mu, lv.clamp(-6, 4), self.law_head(pooled)


def masked_mean(v, m):
    m = m.to(v.dtype); return (v * m).sum() / m.sum().clamp_min(1)


def main():
    a = argparse.ArgumentParser(); a.add_argument('--arm', choices=list(ARMS), required=True); a.add_argument('--seed', type=int, required=True)
    a.add_argument('--out', required=True); a.add_argument('--mix-v5f', action='store_true'); a.add_argument('--data', default='data/prepared/v5m/data.pt')
    a.add_argument('--epochs', type=int, default=30); args = a.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device('cuda'); out = ROOT / args.out; out.mkdir(parents=True, exist_ok=True)
    D = torch.load(ROOT / args.data, weights_only=False); st = D['stats']; meta = D['meta']
    keys = [k for k in D if k not in ('meta', 'stats')]
    G = {k: D[k].to(dev) for k in keys}
    G['y'] = torch.where(G['ym'], (G['y'] - st['y_mean'].to(dev)) / st['y_std'].to(dev), torch.zeros_like(G['y']))
    split = np.array([m['split'] for m in meta])
    tr = np.where((split == 'train') | ((split == 'ftrain') if args.mix_v5f else False))[0]
    evals = {s: np.where(split == s)[0] for s in ('val', 'test', 'fval', 'ftrain')}
    model = MM(ARMS[args.arm]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)
    gen = torch.Generator().manual_seed(args.seed)
    F0, F0p = st['F0'], st['F0_plate']
    batch = lambda idx: {k: G[k][idx] for k in keys}

    def run_eval(idx):
        model.eval(); P, LV, L = [], [], []
        with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
            for i in range(0, len(idx), 64):
                mu, lv, lg = model(batch(torch.as_tensor(idx[i:i + 64], device=dev)), F0, F0p)
                P.append(mu.float()); LV.append(lv.float()); L.append(lg.float())
        model.train(); return torch.cat(P), torch.cat(LV), torch.cat(L)

    best, bad, obs, log = None, 0, 0, []
    for ep in range(args.epochs):
        t0 = time.time(); perm = tr[torch.randperm(len(tr), generator=gen).numpy()]
        for i in range(0, len(perm) - len(perm) % 8, 8):
            b = batch(torch.as_tensor(perm[i:i + 8], device=dev))
            with torch.autocast('cuda', torch.bfloat16):
                mu, lv, lg = model(b, F0, F0p)
            mu, lv, y, m = mu.float(), lv.float(), b['y'], b['ym']
            loss = masked_mean(Fn.smooth_l1_loss(mu, y, reduction='none'), m) + 0.05 * masked_mean(.5 * ((mu - y) ** 2 * (-lv).exp() + lv), m) \
                + 0.1 * Fn.cross_entropy(lg.float(), b['law'])
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        mu, _, _ = run_eval(evals['val']); y, m = G['y'][evals['val']], G['ym'][evals['val']]
        score = float((mu - y).abs()[m].mean()); obs += 1
        improved = best is None or score < best - max(abs(best) * .002, 1e-8)
        if improved:
            best, bad = score, 0; torch.save({'model': model.state_dict(), 'epoch': ep + 1, 'arm': args.arm, 'stats': st, 'mix_v5f': args.mix_v5f}, out / 'best.pt')
        else:
            bad += 1
        log.append({'epoch': ep + 1, 'val_z_mae5': score, 'best': best, 'seconds': time.time() - t0}); print(json.dumps(log[-1]), flush=True)
        if bad >= 5 and obs >= 8:
            break
    model.load_state_dict(torch.load(out / 'best.pt', weights_only=False)['model'])
    for s, idx in evals.items():
        if not len(idx):
            continue
        mu, lv, lg = run_eval(idx)
        with (out / f'pred_{s}.jsonl').open('w') as f:
            for j, i in enumerate(idx):
                mm = meta[i]
                f.write(json.dumps({'sample_id': mm['sid'], 'object_id': mm['obj'], 'action': mm['action'], 'law_id': int(G['law'][i]),
                                    'predicted_law_id': int(lg[j].argmax()), 'prediction_z': mu[j].tolist(), 'logvar': lv[j].tolist(),
                                    'target_z': G['y'][i].tolist(), 'target_mask': G['ym'][i].tolist()}) + '\n')
    (out / 'status.json').write_text(json.dumps({'status': 'done', 'arm': args.arm, 'seed': args.seed, 'mix_v5f': args.mix_v5f, 'best_val_z_mae5': best, 'log': log}, indent=1))


if __name__ == '__main__':
    main()
