"""V5m: multimodal (vision + touch + action) tensors for V5b (train 0-1999 / val / test) and V5f (force-press train / val).

Per sample (saved to data/prepared/v5m/data.pt as stacked tensors):
  vision   : frozen V-JEPA 2 ViT-L cached features (the V5b prior cache), spatially averaged -> [3 views, 32, 1024].
             Exact for the B head: its first op is a 1x1 conv followed by a spatial mean, and the two commute.
  touch    : contact_forces.npz F [400 bins x <=8 bodies x 3] (force each rigid body exerts on the object);
             plate channel [400 x 2] = (plate vertical force, plate gap to the object top / bbox height).
             V5b press/tap: F_plate + commanded plate path from contact_bc press_plate.profile;
             V5f force-controlled press: reaction R and measured plate z from press_force_log.json (0.1 s, interpolated).
  action   : one vector per body (12): centre rel. bbox [-1,1]^3, half-size rel. half-bbox (3), velocity direction (3),
             speed / max bbox extent per s, start/4 s, end/4 s; a plate vector (12) from the press_plate profile
             (displacement mode) or the force set-point (force mode); a drop-impulse vector; a scene vector
             (log bbox extent (3), |gravity|/9.8, action one-hot (6)).
  targets  : ln1p of [E/(rho/1000), nu, rho, sigma_y/(rho/1000), E] (sigma_y masked for jelly / censored), law id.
  .venv/bin/python scripts/v5m/build_data.py
"""
import hashlib, json
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
ACTIONS = ['press', 'drop', 'shear', 'stretch', 'bend', 'tap30hold']; NB = 8; T = 400
CACHE = ROOT / 'data/prepared/v3main_vjl'  # shared cache dir (now V5b/V5f files only)
BIN_T = (np.arange(T) + .5) * 0.01


def plate_path(pp, t):
    z = np.full_like(t, pp['z0'])
    for s, e, v in pp.get('profile') or []:
        z += v * np.clip(t - s, 0, e - s)
    return z


def sample(row, split):
    d = Path(row['pack']).parent.parent
    gt = json.loads((d / 'gt.json').read_text()); bc = json.loads((d / 'contact_bc.json').read_text())
    z = np.load(d / 'contact_forces.npz')
    lo, hi = np.array(bc['particles_lo']), np.array(bc['particles_hi']); ext = hi - lo; c = (lo + hi) / 2
    bodies = json.loads(str(z['bodies'])) if str(z['bodies']) else []
    F = np.zeros((T, NB, 3), np.float32); bmask = np.zeros(NB, bool); bvec = np.zeros((NB, 12), np.float32)
    for i, b in enumerate(bodies[:NB]):
        F[:, i] = z['F'][:, i]; bmask[i] = True
        v = np.array(b['velocity']); sp = np.linalg.norm(v)
        bvec[i] = np.concatenate([(np.array(b['point']) - c) / (ext / 2), np.array(b['size']) / (ext / 2), v / sp if sp > 0 else np.zeros(3),
                                  [sp / ext.max(), b['start_time'] / 4, min(b['end_time'], 8) / 4]])
    plate = np.zeros((T, 2), np.float32); pmask = False; pvec = np.zeros(12, np.float32)
    pp = bc.get('press_plate')
    if pp:
        pmask = True
        if pp.get('mode') == 'force':
            log = np.array(json.loads((d / 'press_force_log.json').read_text())['log_t_z_R'])
            plate[:, 0] = np.interp(BIN_T, log[:, 0], log[:, 2]); zt = np.interp(BIN_T, log[:, 0], log[:, 1])
            pvec[:9] = [(pp['z0'] - c[2]) / (ext[2] / 2), 0, 0, 0, 0, 0, 0, 1, np.log1p(pp['force'])]
        else:
            plate[:, 0] = z['F_plate'] if z['F_plate'].size else 0; zt = plate_path(pp, BIN_T)
            prof = (pp.get('profile') or []) + [[0, 0, 0]] * 2
            pvec[:9] = [(pp['z0'] - c[2]) / (ext[2] / 2), prof[0][2] / ext.max(), prof[0][0] / 4, prof[0][1] / 4,
                        prof[1][2] / ext.max(), prof[1][0] / 4, prof[1][1] / 4, 0, 0]
        plate[:, 1] = (zt - hi[2]) / ext[2]
    imp = [x for x in bc['bcs'] if x['type'] == 'particle_impulse']
    ivec = np.zeros(12, np.float32); imask = bool(imp)
    if imp:
        ivec[:5] = [*imp[0]['force'], imp[0]['num_dt'], imp[0]['start_time']]
    scene = np.zeros(12, np.float32); scene[:3] = np.log(ext); scene[3] = abs(gt.get('gravity', 0)) / 9.8; scene[4 + ACTIONS.index(gt['action'])] = 1
    p = row['parameters']; s = p['rho'] / 1000.
    y = [p['E'] / s, p['nu'], p['rho'], (p['sigma_y'] / s) if p['sigma_y'] is not None else None, p['E']]
    ym = np.array([v is not None for v in y]); yv = np.array([np.log1p(v) if (v is not None and j != 1) else (v if v is not None else 0.) for j, v in enumerate(y)], np.float32)
    feat = torch.load(CACHE / (hashlib.sha256(row['sample_id'].encode()).hexdigest() + '.pt'), weights_only=True)
    vis = feat['features'].float().mean((-1, -2)).to(torch.float16)  # [V, 32, 1024]
    return dict(vis=vis, valid=feat['valid'], times=feat['times'].float(), F=torch.from_numpy(F), bmask=torch.from_numpy(bmask), bvec=torch.from_numpy(bvec),
                plate=torch.from_numpy(plate), pmask=torch.tensor(pmask), pvec=torch.from_numpy(pvec), ivec=torch.from_numpy(ivec), imask=torch.tensor(imask),
                scene=torch.from_numpy(scene), y=torch.from_numpy(yv), ym=torch.from_numpy(ym), law=torch.tensor(gt['law_id']),
                meta=dict(sid=row['sample_id'], obj=row['object_id'], action=gt['action'], split=split, set=row['sample_id'].split('/')[0]))


def main():
    rows = [(r, r['split']) for r in map(json.loads, open(ROOT / 'data/manifests/v5bmain_t2000.jsonl'))]
    rows += [(r, 'f' + r['split']) for r in map(json.loads, open(ROOT / 'data/manifests/v5fmain_all.jsonl'))]
    out = {}
    for i, (r, sp) in enumerate(rows):
        s = sample(r, sp)
        for k, v in s.items():
            out.setdefault(k, []).append(v)
        if i % 500 == 0:
            print(i, len(rows), flush=True)
    data = {k: (torch.stack(v) if k != 'meta' else v) for k, v in out.items()}
    tr = [i for i, m in enumerate(data['meta']) if m['split'] == 'train']
    Fb = data['F'][tr].norm(dim=-1); Fb = Fb[Fb > 1e-6]
    Fp = data['plate'][tr, :, 0].abs(); Fp = Fp[Fp > 1e-6]
    y, ym = data['y'][tr], data['ym'][tr]
    data['stats'] = dict(F0=float(Fb.median()), F0_plate=float(Fp.median()),
                         y_mean=torch.stack([y[ym[:, j], j].mean() for j in range(5)]), y_std=torch.stack([y[ym[:, j], j].std().clamp_min(.05) for j in range(5)]),
                         targets=['E_over_rho', 'nu', 'rho', 'sigma_y_over_rho', 'E'], transform='ln1p (nu linear); specific = divided by rho/1000; stats from V5b train 0-1999')
    (ROOT / 'data/prepared/v5m').mkdir(parents=True, exist_ok=True)
    torch.save(data, ROOT / 'data/prepared/v5m/data.pt')
    print({k: (tuple(v.shape) if torch.is_tensor(v) else len(v)) for k, v in data.items() if k != 'stats'}, data['stats'])


if __name__ == '__main__':
    main()
