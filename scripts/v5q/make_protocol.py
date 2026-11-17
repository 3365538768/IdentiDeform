"""V5q: pre-register the V5b C1 (same rule as scripts/v3q/make_protocol.py; only data/setup/bounds/law means change) refinement experiment -> docs/v5q/PROTOCOL.json (refuses to overwrite a different one).

Sample rule = the V2 rule applied to the V5b indices:
  val : scripts/v3b/c1_select.py   (seed 20261113; per (GT law x observed action) cell 3 random samples (54) + 6 from a
        shuffled pool of 4th picks -> 60)
  test: scripts/v3d/c1_select_test.py (seed 20261113; 1 per cell (18) + 12 from the shuffled pool of 2nd picks -> 30)
Cells use only the generator labels (law = LAWS[i % 3], action = ACTIONS[(i // 3) % 6] as scripts/v3n/gen_v3.py; checked
against gt.json where it exists), over the planned indices val 0-282 / test 0-299. Objects are held out by construction
(data/manifests/v3_split.json: val/test objects disjoint from train). If a selected sample's generation fails (no
exit-0 result.json once its split is finished), it is replaced by the next unselected index of the same cell's
shuffled pool (stored below). gt_observed noise floor: first selected sample (index order) of each action.

  .venv/bin/python scripts/v5q/make_protocol.py
"""
import hashlib, json, random, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/v2'))
LAWS = ['jelly', 'metal', 'plasticine']
ACTIONS = ['press', 'drop', 'shear', 'stretch', 'bend', 'tap30hold']
SEED = 20261113


def select(indices, per_cell, total):
    cells = {}
    for i in indices:
        cells.setdefault((LAWS[i % 3], ACTIONS[(i // 3) % 6]), []).append(i)
    rng = random.Random(SEED)
    chosen, extra, pools = [], [], {}
    for key in sorted(cells):
        pool = sorted(cells[key]); rng.shuffle(pool); pools[f'{key[0]}/{key[1]}'] = pool
        chosen += [(key, i) for i in pool[:per_cell]]; extra += [(key, i) for i in pool[per_cell:per_cell + 1]]
    rng.shuffle(extra)
    chosen += extra[:total - len(chosen)]
    chosen.sort(key=lambda c: c[1])
    seen, gt_obs = set(), []
    for (law, act), i in chosen:
        if act not in seen:
            seen.add(act); gt_obs.append(i)
    return {'seed': SEED, 'per_cell': per_cell, 'total': total, 'cells': {k: len(v) for k, v in pools.items()},
            'samples': [{'index': i, 'law': l, 'action': a, 'gt_observed': i in gt_obs} for (l, a), i in chosen],
            'replacement_pools': pools}


def check_labels(split, sel):
    n = 0
    for s in sel['samples']:
        f = ROOT / f'data/v5b_main/{split}/{s["index"]:06d}/gt.json'
        if f.exists():
            g = json.loads(f.read_text()); assert (g['law'], g['action']) == (s['law'], s['action']), (split, s); n += 1
    return n


def main():
    out = ROOT / 'docs/v5q/PROTOCOL.json'
    val = select(range(283), 3, 60); test = select(range(300), 1, 30)
    split = json.loads((ROOT / 'data/manifests/v3_split.json').read_text())
    assert not (set(split['train']) & (set(split['val']) | set(split['test'])))
    sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
    files = ['scripts/v5q/c1_refine.py', 'scripts/v5q/law_means.py', 'scripts/v5q/make_protocol.py', 'scripts/v3b/c1_refine.py', 'scripts/v3q/c1_refine.py',
             'scripts/v3n/gen_v3.py', 'scripts/v2/contact_bc.py', 'scripts/v2/upstream_main.py', 'scripts/v2/run_upstream_v2.sh',
             'docs/v5q/LAW_MEANS.json']
    P = {
     'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
     'status': 'pre-registered BEFORE any V5q search simulation (only GT re-simulation smoke tests of the code path may precede it)',
     'question': 'Does the prior-guided few-simulation refinement (C1) replicate on the V5b data with the V5b setup, and how does it compare with V2 (and the aborted V3 run)?',
     'data': 'data/v5b_main (scripts/v3n/gen_v3.py --g-all -2 --weak hold --ranges v5: V5 class-separating ranges, screened objects (v3 split), FB_BC_FIX=1 contact setup, gravity -2 on every action, shear/stretch/bend = gripper-held FB_WEAK=hold, press/tap30hold one-sided plate)',
     'samples': {
        'val': {**{k: v for k, v in val.items() if k != 'replacement_pools'}, 'rule': 'scripts/v3b/c1_select.py logic on V5b val indices 0-282',
                'labels_checked_against_existing_gt_json': check_labels('val', val)},
        'test': {**{k: v for k, v in test.items() if k != 'replacement_pools'}, 'rule': 'scripts/v3d/c1_select_test.py logic on V5b test indices 0-299',
                 'labels_checked_against_existing_gt_json': check_labels('test', test)},
        'objects': 'held out: val/test objects (data/manifests/v3_split.json) are disjoint from train objects',
        'failed_generation_replacement': 'next unselected index of the same cell in replacement_pools (shuffled order); none otherwise',
        'replacement_pools': {'val': val['replacement_pools'], 'test': test['replacement_pools']}},
     'simulation_setup': 'scripts/v5q/c1_refine.py simulate(): same config (TEMPLATE + damping 0 / 0.99999, dt 1e-4, frame_dt 0.01, 400 frames, '
                         'g = [0,0,-2] for every action), same CLI (gen_v3 overrides, 3 views, view seed = index, camera 0.65) '
                         'and same env (FB_BC_FIX=1, FB_WEAK=hold, FB_PRESS_MODE=gsmall, FB_G_ON_AT 1.0/3.0, FB_DROP_GAP=0.25, FB_ACTION=tap30hold) as the '
                         'observation; only material keys differ (density 1000, E = E/rho, sigma_y = sigma_y/rho, nu fixed, hardening/softening 0.25). '
                         'Checked per candidate: non-material config == observed config.json, contact_bc.json bcs/press plate == observed. '
                         'Extra render passes (stress/flow/force) and packing off, as V2 C1.',
     'refinement': {'objective': '(1 - mean object-mask IoU) + RGB-MAE(union)/255 vs observed 3-view video (sample_pack.npz), unchanged',
                    'optimizer': 'deterministic opportunistic compass search, unchanged (step 0.5 decade, halved after unsuccessful poll, stop < 0.02)',
                    'bounds_log10_specific': {'E_over_rho': {'jelly': [3.0, 5.5], 'plasticine': [3.0, 4.5], 'metal': [4.5, 6.0]},
                                              'sigma_y_over_rho': {'plasticine': [2.5, 4.0], 'metal': [3.0, 4.5]}},
                    'law_means': 'docs/v5q/LAW_MEANS.json: V5b train means over the generated train gt.json, indices 0-1999 (all 2000 exit 0); censored sigma excluded (as V2/V3)',
                    'budgets': 'k = 0..24 nested (best observed loss among the first k simulations; k = 0 start point; all sims incl. probes counted)',
                    'failure_rule': 'quick crash (<60 s) retried once after 75 s; still failing or blow-up = failed candidate, loss 9, counted, never selected; no sample dropped',
                    'cache': 'identical candidate (law, E, sigma_y, nu) of the same sample simulated once and shared across arms (same config => same setup)',
                    'replay': 'not run (accuracy only, as v3h)'},
     'arms': {'prior_top1': 'V5b main prior ensemble (docs/v5p/HANDOFF.json prediction dirs; mean prediction_z decoded with the prior stats; law votes = '
                            'fraction of members; tie -> law whose train-mean E/rho is closest), top-1 law, no probes, budget 24',
              'prior_top2': 'same ensemble prediction simulated under the top-2 laws (2 probes counted), search continues under the better, budget 24',
              'lawmean_gt': 'PRIVILEGED: GT law, V5b train mean of that law, budget 24',
              'lawmean_best3': 'non-privileged: the 3 V5b law means simulated first (counted), search continues from the best; first point k = 3; budget 24',
              'gt_observed': 'GT parameters re-simulated through the candidate code path (noise floor and setup check), 1 per action per split'},
     'schedule': ['P1 (no prior needed, start when val/test generation is finished): gt_observed val+test; val lawmean_gt 24; val lawmean_best3 to 12; '
                  'test lawmean_gt 24; test lawmean_best3 to 12',
                  'P2 (only when docs/v5p/HANDOFF.json exists AND the lead confirms the V5 gate passed (docs/v5/GATE.json); if the gate fails, P1 is kept as a supplementary result): val prior_top1 24; test prior_top1 24; val prior_top2 24',
                  'P3: val lawmean_best3 12 -> 24; test lawmean_best3 12 -> 24',
                  'P4: test prior_top2 24'],
     'cut_order_if_time_short': ['1. drop test prior_top2 (P4)', '2. drop the lawmean_best3 12 -> 24 extension (P3; best3 reported to k = 12)',
                                 '3. any further cut: report the samples complete at the deadline (sample order, no other selection), labelled partial'],
     'time_budget': 'set by the lead',
     'metrics': {'primary': 'identifiable z-MAE: |z_hat - z_gt| on E/rho and sigma_y/rho (ln1p specific standardised with the V5b prior stats; '
                            'sigma_y only where defined and not censored, or the prior target_mask), pooled over (sample, target) elements; '
                            'if the chosen law is jelly the sigma_y estimate is the prior prediction (prior arms) or the all-law train mean (law-mean arms), as V2',
                 'secondary': 'same error in V2 z units (data/manifests/v2main.specific.stats.json) and in log10 decades, for the V2 comparison '
                              '(V2 val/test numbers are in V2 z units; z units depend on the train ranges)',
                 'ci': 'object-clustered bootstrap, 2000 draws, seed 17 (scripts/v3h/report.py resampler) at every k',
                 'sims_saved': 'uninformed (lawmean_gt) simulations needed to match prior_top1 at k (k = 0, 4, 12): log-budget interpolation as '
                               'scripts/v3b/c1_report.speedup (primary, as F1) and linear (v3h); censored if > 24; object bootstrap',
                 'post_vs_initial': 'per-sample regression of error at k = 4/12/24 on error at k = 0 for the no-probe arms (prior_top1, lawmean_gt): '
                                    'slope, intercept, Pearson r (pooled and per arm); arm-level points as V2 F2 reported descriptively; the same '
                                    'per-sample fit is computed on the V2 val60 states for a like-for-like comparison',
                 'failures': 'failed simulations per arm / split (total, at start/probe)',
                 'v2_reference': {'val_prior_top1': [0.344, 0.327, 0.283, 0.236], 'val_lawmean_gt': [0.674, 0.577, 0.339, 0.257],
                                  'val_lawmean_best3': [None, 0.771, 0.536, 0.482], 'test_prior_top1': [0.496, 0.573, 0.430, 0.298],
                                  'budgets': [0, 4, 12, 24]}},
     'deliverables': ['docs/v5q/RESULTS.json', 'docs/v5q/RESULTS_ZH.md', 'docs/paper_figs/data/f1_bands_v5b.json'],
     'no_selection_on_test': 'no hyperparameter, sample, arm or metric choice depends on any test outcome; nothing tuned on val either (V2 settings)',
     'sha256': {f: sha(f) for f in files},
    }
    if out.exists():
        old = json.loads(out.read_text())
        if old['samples'] != P['samples']:
            raise SystemExit('protocol already registered with different samples; refusing to overwrite')
        print('protocol unchanged'); return
    out.write_text(json.dumps(P, indent=1) + '\n')
    print('val', [s['index'] for s in val['samples']]); print('test', [s['index'] for s in test['samples']])


if __name__ == '__main__':
    main()
