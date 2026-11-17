"""Write the frozen pre-registration docs/v3d/C1_TEST_PROTOCOL.json (refuses to overwrite)."""
import hashlib, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
out = ROOT / 'docs/v3d/C1_TEST_PROTOCOL.json'
assert not out.exists(), 'protocol already frozen'
assert not (ROOT / 'data/v3d_refine_test').exists(), 'test simulations already exist'
sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
files = ['scripts/v3d/c1_refine.py', 'scripts/v3b/c1_refine.py', 'scripts/v3d/c1_select_test.py', 'scripts/v3d/c1_make_ensemble_test.py',
         'scripts/v3d/c1_enqueue_test.py', 'scripts/v3d/c1_test_report.py', 'scripts/v3b/c1_report.py', 'docs/v3d/C1_TEST_SAMPLES.json',
         'docs/v3d/ENSEMBLE_BCD9_TEST.json', 'runs/v3c_ens_bcd9_test/predictions.jsonl', 'runs/v2main_C_seed29/test_once/predictions.jsonl',
         'data/manifests/v2main.jsonl', 'data/manifests/v2main.specific.stats.json', 'data/manifests/v2main_withtest.jsonl',
         'docs/v3b/C1_REFINE_RESULTS.json']
samples = json.loads((ROOT / 'docs/v3d/C1_TEST_SAMPLES.json').read_text())['samples']
P = {
 'frozen_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
 'status': 'frozen BEFORE any test-split simulation; run once; not changed afterwards',
 'purpose': 'test-split replication of the validation C1 claim (docs/v3b/V3B_RESULTS_ZH.md, docs/v3b/C1_REFINE_RESULTS.json)',
 'code': 'scripts/v3d/c1_refine.py = scripts/v3b/c1_refine.py with only split/path plumbing (--split test, --ensemble) and the '
         'addendum arm prior_top1; objective (1-IoU)+RGB_MAE_union/255 vs observed 3-view 128-frame video, deterministic compass '
         'search over log10(E/rho) [+log10(sigma_y/rho) unless jelly], step 0.5 decade halved after unsuccessful poll, bounds, '
         'nu/rho fixed, hardening/softening 0.25, nested budgets, unseen-action replay ACTIONS[(i+3)%6] vs GT replay: all unchanged. '
         'No hyperparameter changes.',
 'failure_rule': 'unchanged from val: quick crash (<60 s) retried once after 75 s; still failing or mid-simulation blow-up = failed '
                 'candidate with loss 9, counted in the budget, never selected; no sample dropped; GT-replay failures only drop the '
                 'sample from replay IoU.',
 'samples': {'n': len(samples), 'rule': 'scripts/v3d/c1_select_test.py: logic of scripts/v3b/c1_select.py, seed 20261113, '
             'stratified by GT law x observed action over the 300 test samples, 1 per cell (18) + 12 second picks from a shuffled '
             'extra pool; selection uses only gt.json law/action', 'indices': [s['index'] for s in samples],
             'gt_observed_noise_floor': [s['index'] for s in samples if s['gt_observed']]},
 'start_model': {'prior_top2': 'docs/v3d/ENSEMBLE_BCD9_TEST.json = saved 9-run V-JEPA B/C/D test predictions (runs/<run>/test_once, '
                 'mean z, law posterior = vote fractions), identical to runs/v3c_ens_bcd9_test; top-2 laws by vote (tie: train-mean E/rho '
                 'closest to prediction), each simulated once, search continues under the better',
                 'targets': 'target_z/target_mask from runs/v2main_C_seed29/test_once (train-split statistics)'},
 'arms': {'prior_top2': 'budget 24, report 0/2/3/4/8/12/16/24, replay at 0/4/12/24',
          'lawmean_gt': 'PRIVILEGED (GT law, train mean), budget 24, replay at 12/24',
          'lawmean_best3': 'non-privileged (3 law means simulated, best continued), budget 24, replay at 12/24',
          'prior_top1 (ADDENDUM, requested by lead before freezing, before any test simulation)':
              '9-run ensemble test prediction with its top-1 law (= prior_top2 budget-0 point), no law probes, same compass search; '
              'budget 24, replay 0/4/12/24; motivated by the val finding that video-fit law probing lowered law accuracy '
              '(0.733 -> 0.617); scheduled after the three primary arms; reported in the same tables, does not affect the primary analysis'},
 'metrics': 'identical to val: pooled identifiable z-MAE over E/rho and sigma_y/rho (train-standardized ln1p specific, sigma_y only '
            'where defined), per-parameter MAE, law accuracy, observed-fit IoU / RGB-MAE of the selected candidate, unseen-action '
            'replay IoU, paired object-clustered bootstrap (2000 draws, seed 17) of per-sample differences at equal budgets, speedup = '
            'uninformed sims to match prior_top2+k (log-budget interpolation, object bootstrap 1000 draws, censored above 24), both budget '
            'accountings (all sims incl. probes; search-only = k minus probes), failure counts.',
 'hypotheses_replication_criteria': {
     'H1': 'prior_top2 beats non-privileged lawmean_best3 at every budget: paired CI upper bound < 0 at k=0 (vs best3 at 3), 4, 12, 24',
     'H2': 'privileged lawmean_gt needs >= 6 sims to match prior_top2+0 (val: 11.7; censored >24 also counts as replicated)',
     'H3': 'prior_top2 - lawmean_gt paired CI upper bound < 0 at k=0 and k=4 (val: yes); at 12/24 val was n.s., reported descriptively'},
 'execution': 'docs/v3d/queue.json (ids v3d_*), scripts/r_queue.py, outputs data/v3d_refine_test/v2main_C_seed29/<idx>/<arm>/state.json; '
              'per-sample order prior_top2, lawmean_gt, lawmean_best3, then all prior_top1 jobs.',
 'completion_rule': 'Analysis set = samples complete in all three primary arms at budget 24. If not all 30 are complete by '
                    '2026-11-14 18:30 UTC, the report is produced on the complete subset (in pre-registered sample order, no other '
                    'selection) and labelled partial; jobs keep running and a final update on all 30 replaces it, labelled with its time. '
                    'prior_top1 is reported on the analysed samples where it is complete.',
 'no_selection_on_test': 'no hyperparameter, sample, arm or metric choice depends on any test outcome',
 'sha256': {f: sha(f) for f in files},
}
out.write_text(json.dumps(P, indent=1) + '\n')
print(P['frozen_utc'])
