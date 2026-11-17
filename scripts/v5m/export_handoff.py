"""V5m hand-off: 3-seed ensemble predictions of the v5m arms (default: vision, vision_touch_action) on V5b val/test, in the
format of docs/v5p/ENSEMBLE_BCD9_{VAL,TEST}.json (read by the gripper twin / refinement code), plus physical values.

Per sample: prediction_z / z_std_across_runs in the V5b prior's space (data/manifests/v5bmain_t2000.specific.stats.json,
order [E_over_rho, nu, rho, sigma_y_over_rho]; v5m outputs are decoded with the v5m train stats and re-encoded, so
existing decoders work unchanged); law_posterior_votes = mean softmax law probability over the 3 seeds (also law_top2);
physical = {E_over_rho, nu, rho, sigma_y_over_rho, E} decoded (E_over_rho / sigma_y_over_rho in the specific units
x/(rho/1000); rho [kg/m^3], E [Pa]); rho and E are only meaningful for the touch arm (vision-only rho/E are reported but
are not identifiable from video). Writes docs/v5m/ENSEMBLE_<ARM>_{VAL,TEST}.json, runs/v5m_ens_<arm>_<split>/predictions.jsonl,
docs/v5m/HANDOFF.json.
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v5m'))
from train_mm import MM, ARMS  # noqa: E402


def main():
    arms = sys.argv[1:] or ['vision', 'vision_touch_action']
    D = torch.load(ROOT / 'data/prepared/v5m/data.pt', weights_only=False); st = D['stats']; meta = D['meta']
    dev = torch.device('cuda'); keys = [k for k in D if k not in ('meta', 'stats')]
    G = {k: D[k].to(dev) for k in keys}
    mean, std = st['y_mean'].numpy(), st['y_std'].numpy()
    ps = json.loads((ROOT / 'data/manifests/v5bmain_t2000.specific.stats.json').read_text()); pm, pstd = np.array(ps['mean']), np.array(ps['std'])
    split = np.array([m['split'] for m in meta])
    h = {'stage': 'v5m', 'written_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'data_root': 'data/v5b_main',
         'sample_id_format': 'v5bmain/<split>/<idx:06d>', 'stats_specific': 'data/manifests/v5bmain_t2000.specific.stats.json',
         'target_space': 'prediction_z: same space/order as docs/v5p/HANDOFF.json [E_over_rho, nu, rho, sigma_y_over_rho]; physical: decoded values incl. E [Pa], rho [kg/m^3]',
         'law_ids': {'0': 'jelly', '1': 'metal', '2': 'plasticine'}, 'doc': __doc__, 'arms': {}}
    for arm in arms:
        runs = [f'runs/v5m_{arm}_seed{s}' for s in (17, 29, 43)]
        models = []
        for r in runs:
            m = MM(ARMS[arm]).to(dev); m.load_state_dict(torch.load(ROOT / r / 'best.pt', weights_only=False)['model']); m.eval(); models.append(m)
        h['arms'][arm] = {'runs': runs, 'modalities': list(ARMS[arm])}
        for sp in ('val', 'test'):
            idx = np.where(split == sp)[0]; Z, P = [], []
            with torch.no_grad(), torch.autocast('cuda', torch.bfloat16):
                for m in models:
                    zz, pp = [], []
                    for i in range(0, len(idx), 64):
                        b = {k: G[k][torch.as_tensor(idx[i:i + 64], device=dev)] for k in keys}
                        mu, lv, lg = m(b, st['F0'], st['F0_plate']); zz.append(mu.float()); pp.append(lg.float().softmax(-1))
                    Z.append(torch.cat(zz).cpu().numpy()); P.append(torch.cat(pp).cpu().numpy())
            Z, P = np.stack(Z), np.stack(P)  # [3, N, 5], [3, N, 3]
            ln = Z * std + mean  # ln1p space (nu linear)
            z4 = (ln[..., :4] - pm) / pstd  # re-encoded in the V5b prior's stats
            phys = np.where(np.arange(5) == 1, ln, np.expm1(np.clip(ln, 0, 40)))
            samples, rows = {}, []
            for j, i in enumerate(idx):
                sid = meta[i]['sid']; pr = P[:, j].mean(0); top = np.argsort(-pr)[:2]
                ph = phys[:, j].mean(0)  # mean of decoded values would mix scales -> use decoded mean-z instead
                ph = np.where(np.arange(5) == 1, ln[:, j].mean(0), np.expm1(np.clip(ln[:, j].mean(0), 0, 40)))
                samples[sid] = {'prediction_z': z4[:, j].mean(0).tolist(), 'z_std_across_runs': z4[:, j].std(0).tolist(), 'law_posterior_votes': pr.tolist(),
                                'law': int(top[0]), 'law_top2': [[int(t), float(pr[t])] for t in top],
                                'physical': dict(zip(['E_over_rho', 'nu', 'rho', 'sigma_y_over_rho', 'E'], map(float, ph))),
                                'ln1p_std_across_runs': dict(zip(['E_over_rho', 'nu', 'rho', 'sigma_y_over_rho', 'E'], map(float, ln[:, j].std(0))))}
                rows.append({'sample_id': sid, 'object_id': meta[i]['obj'], 'action': meta[i]['action'], 'law_id': int(D['law'][i]), 'predicted_law_id': int(top[0]),
                             'prediction_z': samples[sid]['prediction_z'], 'law_probs': pr.tolist(), 'physical': samples[sid]['physical']})
            js = ROOT / f'docs/v5m/ENSEMBLE_{arm.upper()}_{sp.upper()}.json'
            js.write_text(json.dumps({'runs': runs, 'split': sp, 'rule': 'mean z over 3 seeds (re-encoded to v5bmain_t2000 stats); law_posterior_votes = mean softmax', 'samples': samples}))
            out = ROOT / f'runs/v5m_ens_{arm}_{sp}'; out.mkdir(parents=True, exist_ok=True)
            (out / 'predictions.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
            h['arms'][arm][sp] = {'json': str(js.relative_to(ROOT)), 'predictions': str((out / 'predictions.jsonl').relative_to(ROOT)), 'samples': len(rows)}
    (ROOT / 'docs/v5m/HANDOFF.json').write_text(json.dumps(h, indent=1) + '\n'); print(json.dumps({k: v for k, v in h.items() if k != 'doc'}, indent=1))


if __name__ == '__main__':
    main()
