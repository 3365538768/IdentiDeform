"""V3j: reviewer-requested C1 controls (validation only). See docs/v3j/PROTOCOL.json (pre-registered).

Same objective, observed video, simulator call, failure rule and nested best-of-first-k budget rule as
scripts/v3d/c1_refine.py (functions imported verbatim). Arms:
  Exp 1 (stronger uninformed baselines; search over log10(E/rho) [+ log10(sigma_y/rho) unless law = jelly],
         nu and hardening/softening fixed as in C1):
    cma_gtmean  PRIVILEGED GT law, start = GT-law train mean, CMA-ES sigma0 = 1 decade
    cma_best3   3 law-mean probes (counted), CMA-ES from the best probe (its law, its law mean), sigma0 = 1 decade
    cma_prior   9-run V-JEPA ensemble prediction, top-1 law (= prior_top1 start), CMA-ES sigma0 = 0.5 decade
    CMA-ES: numpy (mu/mu_w, lambda) CMA-ES (Hansen 2016 tutorial defaults), lambda = 4, box = train range
    (log10 E/rho [2, 6], log10 sigma_y/rho [0, 4.5]); sampled points are clipped to the box (repair) and the
    clipped points enter the update. The start point is evaluated first (sim 1; for cma_best3 sims 1-3 are
    the probes), then generations of 4 run in parallel; a generation cut by the budget is resumed on extension.
  Exp 2 (unfrozen nuisance parameters; 4-D compass search, else identical to C1):
    u4_prior    prior_top1 start (ensemble prediction, top-1 law, ensemble nu, hardening 0.25)
    u4_gtmean   PRIVILEGED GT law, GT-law train mean (law-mean nu, hardening 0.25)
    dims: log10 E/rho, log10 sigma_y/rho, hardening in [0.05, 0.45], nu in [0.10, 0.46] (jelly: E, nu);
    initial steps 0.5 / 0.5 decade / 0.1 / 0.08, all halved after an unsuccessful poll, stop when the
    E step < 0.02 (same 5 halvings as C1). Plasticine softening stays 0.25.
Simulations identical to an evaluation already stored in data/v3b_refine (same law and same candidate
parameters, e.g. the prior_top1 / lawmean_gt start point, the 3 law-mean probes) are reused (flag 'reused_from')
and count toward the budget.

  .venv/bin/python scripts/v3j/c1j_refine.py --index 210 --arm cma_prior --budget 12
  .venv/bin/python scripts/v3j/c1j_refine.py --index 210 --arm u4_prior --replay   (unseen-action replay, Exp 2)
State: data/v3j_refine/<arm>/<index>/state.json (resumable; simulation dirs deleted after scoring).
"""
import argparse
import glob
import json
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v3d'))
import c1_refine as C  # noqa: E402
from c1_refine import LAWS, ACTIONS, BOUNDS, FAILED_LOSS, QUICK_FAIL_SECONDS  # noqa: E402,F401

OLD = ROOT / 'data/v3b_refine/v2main_C_seed29'
OUT = ROOT / 'data/v3j_refine'
TRAIN = {'E': (2.0, 6.0), 'sigma_y': (0.0, 4.5), 'hardening': (0.05, 0.45), 'nu': (0.10, 0.46)}
U4_BOUNDS = {'E': BOUNDS['E'], 'sigma_y': BOUNDS['sigma_y'], 'hardening': TRAIN['hardening'], 'nu': TRAIN['nu']}
U4_STEP = {'E': 0.5, 'sigma_y': 0.5, 'hardening': 0.1, 'nu': 0.08}
H0 = 0.25
ARMS = ('cma_gtmean', 'cma_best3', 'cma_prior', 'u4_prior', 'u4_gtmean')
ARM_ID = {a: i for i, a in enumerate(ARMS)}
SIGMA0 = {'cma_gtmean': 1.0, 'cma_best3': 1.0, 'cma_prior': 0.5}
LAMBDA = 4
REUSE_FROM = {'cma_gtmean': ['lawmean_gt'], 'cma_best3': ['lawmean_best3', 'lawmean_gt'], 'cma_prior': ['prior_top1'],
              'u4_prior': ['prior_top1'], 'u4_gtmean': ['lawmean_gt']}


class CMAES:
    """(mu/mu_w, lambda)-CMA-ES, default strategy parameters of Hansen (2016), 'The CMA Evolution Strategy: A Tutorial'."""

    def __init__(self, m, sigma, lam, rng):
        n = len(m); self.n, self.m, self.sigma, self.lam, self.rng = n, np.array(m, float), float(sigma), lam, rng
        mu = lam // 2; w = np.log(mu + .5) - np.log(np.arange(1, mu + 1)); self.w = w / w.sum(); self.mu = mu
        self.mueff = 1 / (self.w ** 2).sum()
        self.cc = (4 + self.mueff / n) / (n + 4 + 2 * self.mueff / n)
        self.cs = (self.mueff + 2) / (n + self.mueff + 5)
        self.c1 = 2 / ((n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0., math.sqrt((self.mueff - 1) / (n + 1)) - 1) + self.cs
        self.pc, self.ps, self.C = np.zeros(n), np.zeros(n), np.eye(n)
        self.chiN = math.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n)); self.g = 0

    def ask(self):
        vals, B = np.linalg.eigh(self.C); D = np.sqrt(np.maximum(vals, 1e-20)); self.B, self.D = B, D
        z = self.rng.standard_normal((self.lam, self.n))
        return self.m + self.sigma * (z * D) @ B.T

    def tell(self, X, f):
        X = np.asarray(X, float); idx = np.argsort(np.asarray(f), kind='stable')[:self.mu]
        y = (X[idx] - self.m) / self.sigma; yw = self.w @ y
        self.m = self.m + self.sigma * yw
        Cis = self.B @ np.diag(1 / self.D) @ self.B.T
        self.ps = (1 - self.cs) * self.ps + math.sqrt(self.cs * (2 - self.cs) * self.mueff) * (Cis @ yw)
        self.g += 1
        hs = float(np.linalg.norm(self.ps) / math.sqrt(1 - (1 - self.cs) ** (2 * self.g)) / self.chiN < 1.4 + 2 / (self.n + 1))
        self.pc = (1 - self.cc) * self.pc + hs * math.sqrt(self.cc * (2 - self.cc) * self.mueff) * yw
        rank_mu = sum(wi * np.outer(yi, yi) for wi, yi in zip(self.w, y))
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * (np.outer(self.pc, self.pc) + (1 - hs) * self.cc * (2 - self.cc) * self.C)
                  + self.cmu * rank_mu)
        self.C = (self.C + self.C.T) / 2
        self.sigma *= math.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))


def full_params(law, xf):
    """xf = [log10 E/rho, log10 sigma_y/rho, hardening, nu] -> simulator material dict (C.candidate + hardening)."""
    p = C.candidate(law, xf[0], xf[1], xf[2 + 1])
    if law != 'jelly':
        p['hardening'] = float(xf[2])
    return p


def main():
    a = argparse.ArgumentParser(); a.add_argument('--index', type=int, required=True)
    a.add_argument('--arm', choices=ARMS, required=True); a.add_argument('--budget', type=int, default=24)
    a.add_argument('--replay', action='store_true'); a.add_argument('--replay-budgets', default='0,4,12,24')
    args = a.parse_args()
    sid = f'v2main/val/{args.index:06d}'
    old = {arm: json.loads((OLD / f'{args.index:06d}' / arm / 'state.json').read_text()) for arm in ('prior_top1', 'lawmean_gt', 'lawmean_best3')}
    ref = old['prior_top1']  # target_z / target_mask (identical across runs)
    gt = json.loads((ROOT / f'data/v2_main/val/{args.index:06d}/gt.json').read_text())
    gcfg = json.loads((ROOT / f'data/v2_main/val/{args.index:06d}/config.json').read_text())
    obj, action = gt['object'], gt['action']
    replay_action = ACTIONS[(ACTIONS.index(action) + 3) % len(ACTIONS)]
    rp = json.loads(C.V1_RUN_PARAMETERS.read_text())
    work = OUT / args.arm / f'{args.index:06d}'; work.mkdir(parents=True, exist_ok=True)
    state_path = work / 'state.json'
    state = json.loads(state_path.read_text()) if state_path.exists() else {'evals': [], 'replay': {}}
    bad = next((i for i, e in enumerate(state['evals']) if e.get('failed') and e['seconds'] < QUICK_FAIL_SECONDS), None)
    if bad is not None:
        state['evals'] = state['evals'][:bad]
    if not args.replay:
        state.pop('complete', None)

    def save():
        tmp = state_path.with_suffix('.tmp'); tmp.write_text(json.dumps(state, indent=1)); os.replace(tmp, state_path)

    means = C.law_means()
    ens = json.loads((ROOT / 'docs/v3b/ENSEMBLE_BCD9_VAL.json').read_text())['samples'][sid]
    cma = args.arm.startswith('cma_')
    box = TRAIN if cma else U4_BOUNDS

    def start(law, z):
        e, nu_, _, sg = C.decode(z)
        x = [math.log10(max(e, 1e-3)), math.log10(max(sg, 1e-3))]
        return [round(float(np.clip(x[d], *box[n])), 6) for d, n in enumerate(('E', 'sigma_y'))], nu_

    def xfull(x2, nu_, h=H0):
        return [round(float(v), 6) for v in (x2[0], x2[1], h, nu_)]

    if args.arm in ('cma_prior', 'u4_prior'):
        votes = ens['law_posterior_votes']; base_z = list(ens['prediction_z'])
        order = sorted(range(3), key=lambda k: (-votes[k], abs(means[LAWS[k]][0] - base_z[0])))
        law = LAWS[order[0]]; x2, nu = start(law, base_z)
        assert law == ref['law'], (law, ref['law'])
    elif args.arm in ('cma_gtmean', 'u4_gtmean'):
        law = gt['law']; base_z = list(means[law]); x2, nu = start(law, base_z)
    else:
        law = None; base_z = None; x2 = nu = None
    if not cma:
        nu = float(np.clip(nu, *TRAIN['nu']))
    names = None
    state.update(sample_id=sid, arm=args.arm, object=obj, observed_action=action, replay_action=replay_action, gt_law=gt['law'],
                 target_z=ref['target_z'], target_mask=ref['target_mask'],
                 gt_nuisance={'nu': gcfg['nu'], 'hardening': gcfg.get('hardening'), 'softening': gcfg.get('softening')})

    # ---- evaluation with cache, parallel batches, reuse of identical stored C1 simulations -----------------------
    observed = None

    def key_of(lw, xf):
        return [lw, [round(float(v), 6) for v in xf]]

    def find(lw, xf):
        k = key_of(lw, xf)
        for e in state['evals']:
            if [e['law'], e['xf']] == k:
                return e
        return None

    def reusable(lw, xf):
        p = full_params(lw, xf)
        for arm in REUSE_FROM[args.arm]:
            s = old[arm]
            for e in s['evals']:
                if e.get('law', s['law']) != lw or e.get('failed'):
                    continue
                nu_old = s['nu'] if lw == s['law'] else C.decode(means[lw])[1]
                if C.candidate(lw, e['x'][0], e['x'][1], round(float(nu_old), 6)) == p:  # nu stored to 6 decimals here
                    return arm, e
        return None

    def run_batch(points):
        """points: list of (law, xf). Returns list of evals (None if the budget is exhausted before that point)."""
        nonlocal observed
        out, todo = [], []
        for lw, xf in points:
            xf = [round(float(v), 6) for v in xf]
            e = find(lw, xf)
            if e is None:
                for t in todo:
                    if key_of(lw, xf) == key_of(*t[:2]):
                        e = t
                        break
            if e is None:
                if len(state['evals']) + len(todo) >= args.budget:
                    out.append(None); continue
                r = reusable(lw, xf)
                if r is not None:
                    arm, ee = r
                    rec = {'n': len(state['evals']) + 1, 'law': lw, 'xf': xf, 'x': xf[:2], 'seconds': ee['seconds'],
                           **{k: ee[k] for k in ('iou', 'iou_last', 'rgb_mae_union', 'loss')}, 'reused_from': f'{arm}#{ee["n"]}'}
                    state['evals'].append(rec); save(); print('eval (reused)', rec, flush=True); out.append(rec); continue
                t = (lw, xf, len(state['evals']) + len(todo)); todo.append(t); out.append(t)
            else:
                out.append(e)
        if todo:
            if observed is None:
                observed = C.load_pack(glob.glob(str(ROOT / f'data/v2_main/val/{args.index:06d}/*/sample_pack.npz'))[0])

            def sim(t):
                lw, xf, n = t
                return C.simulate(work, f'obs_{n:02d}', full_params(lw, xf), action, obj, args.index, rp)
            with ThreadPoolExecutor(len(todo)) as ex:
                res = list(ex.map(sim, todo))
            done = {}
            for (lw, xf, n), (frames, seconds) in zip(todo, res):
                fit = C.compare(observed, frames) if frames is not None else {'iou': None, 'iou_last': None, 'rgb_mae_union': None,
                                                                                  'loss': FAILED_LOSS, 'failed': True}
                rec = {'n': len(state['evals']) + 1, 'law': lw, 'xf': xf, 'x': xf[:2], 'seconds': seconds, **fit}
                state['evals'].append(rec); done[n] = rec; print('eval', rec, flush=True)
            save()
            out = [done[o[2]] if isinstance(o, tuple) else o for o in out]
        return out

    if args.replay:
        return replay(args, state, save, work, gcfg, obj, replay_action, rp)

    # ---- search ---------------------------------------------------------------------------------------------------
    if args.arm == 'cma_best3':
        probes = []
        for lw in LAWS:
            xl, nul = start(lw, means[lw])
            probes.append((lw, xfull(xl, nul)))
        res = run_batch(probes)
        cand = [(e, lw, xf) for e, (lw, xf) in zip(res, probes) if e is not None]
        e_best, law, xf0 = min(cand, key=lambda c: c[0]['loss'])
        base_z = list(means[law]); x2, nu = xf0[:2], xf0[3]
    dims = ['E'] if law == 'jelly' else ['E', 'sigma_y']
    if not cma:
        dims = ['E', 'nu'] if law == 'jelly' else ['E', 'sigma_y', 'hardening', 'nu']
    x0f = xfull(x2, nu)
    state.update(law=law, x0=x2, x0f=x0f, nu=nu, base_z=base_z, dims=dims, box=box,
                 optimizer=('cma-es' if cma else 'compass-4d'), **({'sigma0': SIGMA0[args.arm], 'lambda': LAMBDA} if cma else {'initial_steps': U4_STEP}))
    save()
    pos = {'E': 0, 'sigma_y': 1, 'hardening': 2, 'nu': 3}
    ix = [pos[d] for d in dims]

    def embed(v):
        xf = list(x0f)
        for j, d in zip(ix, dims):
            xf[j] = float(np.clip(v[dims.index(d)], *box[d]))
        return xf

    best = run_batch([(law, x0f)])[0]
    if cma:
        rng = np.random.default_rng([20261115, args.index, ARM_ID[args.arm]])
        es = CMAES([x0f[j] for j in ix], SIGMA0[args.arm], LAMBDA, rng)
        gens = 0
        while best is not None and len(state['evals']) < args.budget and gens < 200 and es.sigma > 1e-3:
            X = [embed(v) for v in es.ask()]
            res = run_batch([(law, xf) for xf in X])
            gens += 1
            if any(r is None for r in res):
                break
            es.tell([[xf[j] for j in ix] for xf in X], [r['loss'] for r in res])
        state['cma_generations_told'] = es.g
    else:
        step = [U4_STEP[d] for d in dims]; last = None
        directions = [(d, s) for d in range(len(dims)) for s in (+1, -1)]
        while best is not None and len(state['evals']) < args.budget and step[0] >= 0.02:
            improved = False
            for d, s in ([last] if last else []) + [v for v in directions if v != last]:
                y = list(best['xf']); j = ix[d]
                y[j] = round(float(np.clip(y[j] + s * step[d], *box[dims[d]])), 6)
                if y == best['xf']:
                    continue
                e = run_batch([(law, y)])[0]
                if e is None:
                    break
                if e['loss'] < best['loss']:
                    best, last, improved = e, (d, s), True
                    break
            if not improved:
                step = [v / 2 for v in step]; last = None

    # ---- nested budgets (best of the first k simulations; k = 0 is the start point) ---------------------------------
    m = np.array(ref['target_mask'], bool); budgets = {}
    for k in range(args.budget + 1):
        n = min(k, len(state['evals']))
        if args.arm == 'cma_best3' and n == 0:
            continue
        if n == 0:
            lk, xf = law, x0f
        else:
            e = min(state['evals'][:n], key=lambda e: e['loss']); lk, xf = e['law'], e['xf']
        z = list(base_z) if lk == law else list(means[lk])
        z[0] = C.encode(0, 10 ** xf[0])
        if lk != 'jelly':
            z[3] = C.encode(3, 10 ** xf[1])
        elif args.arm not in ('cma_prior', 'u4_prior'):
            z[3] = 0.0
        err = np.abs(np.array(z) - np.array(ref['target_z']))
        fit = None
        if n:
            fe = next(e for e in state['evals'][:n] if e['law'] == lk and e['xf'] == xf)
            fit = dict(iou=fe['iou'], rgb_mae_union=fe['rgb_mae_union'], loss=fe['loss'], failed=fe.get('failed', False))
        budgets[str(k)] = {'law': lk, 'x': xf[:2], 'xf': xf, 'sims': n, 'wall_seconds': float(sum(e['seconds'] for e in state['evals'][:n])),
                           'prediction_z': z, 'abs_err_z': [float(v) if mm else None for v, mm in zip(err, m)],
                           'mae_ident': float(np.mean([err[j] for j in (0, 3) if m[j]])),
                           'hardening': xf[2] if lk != 'jelly' else None, 'nu': xf[3], 'observed_fit': fit}
    state['budgets'] = budgets; state['complete'] = True; state['budget_run'] = args.budget; save()
    print('done', args.index, args.arm, {k: round(v['mae_ident'], 3) for k, v in budgets.items() if int(k) in (0, 4, 12, 24)}, flush=True)


def replay(args, state, save, work, gcfg, obj, replay_action, rp):
    """Unseen-action replay of the budget-k estimates vs the GT-parameter replay (C.gt_replay, shared per sample)."""
    gt_params = {k: gcfg[k] for k in ('material', 'E', 'nu', 'density', 'yield_stress', 'hardening', 'softening') if k in gcfg}
    root = OUT / 'gt_replay' / f'{args.index:06d}'; root.mkdir(parents=True, exist_ok=True)
    wanted = [k for k in args.replay_budgets.split(',') if k in state.get('budgets', {})]
    tags = {}
    for k in wanted:
        b = state['budgets'][k]; tag = json.dumps([b['law'], b['xf']])
        if k in state['replay'] and state['replay'][k].get('tag') == tag and 'iou' in state['replay'][k]:
            continue
        tags.setdefault(tag, []).append(k)
    if not tags:
        print('replay: nothing to do'); return
    if (OLD / f'{args.index:06d}' / 'prior_top1' / 'state.json').exists() and json.loads((OLD / f'{args.index:06d}' / 'prior_top1' / 'state.json').read_text())['replay'].get('gt_failed'):
        state['replay']['gt_failed'] = True; state['replay']['skipped'] = 'C1 GT replay failed for this sample'; save(); print('replay skipped'); return

    def run(item):
        if item == 'gt':
            return C.gt_replay(root, replay_action, obj, args.index, rp, gt_params)
        lw, xf = json.loads(item)
        return C.simulate(work, f'replay_{abs(hash(item)) % 10 ** 8:08d}', full_params(lw, xf), replay_action, obj, args.index, rp)
    with ThreadPoolExecutor(len(tags) + 1) as ex:
        res = dict(zip(['gt', *tags], ex.map(run, ['gt', *tags])))
    ref = res.pop('gt'); state['replay']['gt_failed'] = ref is None
    if ref is not None:
        for tag, (frames, sec) in res.items():
            r = {**(C.compare(ref, frames) if frames is not None else {'failed': True}), 'seconds': sec, 'tag': tag}
            for k in tags[tag]:
                state['replay'][k] = r
            print('replay', tags[tag], r, flush=True)
    save()


if __name__ == '__main__':
    main()
