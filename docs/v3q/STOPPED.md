# V3q stopped (2026-11-15 ~15:55 UTC): switch to V4 data

Stopped on the lead/user decision (V4 = V3 contact setup + gravity -2 on every action; the V3 prior is much weaker).
Nothing new is launched: docs/v3q/STOP exists, docs/v3q/queue.json has "gpus": [], the a800-4 pool is stopped, all
v3q searches were killed on both hosts (states are resumable but must NOT be resumed for V3).

## What finished (V3 data, data/v3_main)
- Protocol pre-registered before any search: docs/v3q/PROTOCOL.json (+ clarification: prior = V3 main prior from v3p hand-off).
- Setup check, GT re-simulation through the candidate code path, 1 per action per split (12/12 done):
  IoU 0.99998-1.0000, loss < 5e-5 on press/drop/shear/stretch/bend/tap30hold, val and test
  (V2 C1 noise floor was IoU 0.977/0.958), so the candidate setup reproduces the V3 observation.
- Searches complete: val lawmean_gt 20/60 (budget 24); val lawmean_best3 9 at 24 + 3 at 12; test lawmean_gt 2/30.
  Partial (resumable) states: val lawmean_gt 40, val best3 29, val prior_top1 57, val prior_top2 30, test lawmean_gt 13,
  test best3 15, test prior_top1 15. 1548 evaluations / 1191 unique simulations, 0 failed simulations.
- Prior-arm (P2) searches had only just started (val 15:17, test 15:25 UTC), so there is no prior-arm result.
- Interim aggregate of the complete searches only (NOT a deliverable, n = 10 val): docs/v3q/RESULTS_V3_PARTIAL.json.

## Where things are
- Code: scripts/v3q/ (c1_refine.py = V3 simulate path + V2 C1 search; make_protocol.py, law_means.py, make_prior.py,
  enqueue.py, worker.py, pool.py, throttle.py, report.py, write_zh.py, status.py, plan_update.py).
- States: data/v3q_refine/<split>/<idx>/<arm>/state.json; cross-arm simulation cache data/v3q_refine/<split>/<idx>/_cache.
- Task list / host partition: docs/v3q/tasks.json, docs/v3q/SAMPLES_RESOLVED.json; done markers runs/v3q_done/.
- Prior ensembles built from docs/v3p/HANDOFF.json: docs/v3q/PRIOR_ENSEMBLE_{val,test}.json (9-run B/C/D).
- Law means: docs/v3q/LAW_MEANS.json. Execution log: docs/v3q/DEVIATIONS.md.
- Pre-protocol smoke: data/v3q_smoke (2 GT re-simulations).

## For V4 (when re-pointed)
c1_refine.py builds config/env with the gen_v3.py rule (gravity() / v3_env()); for V4 these must follow
gen_v3.py --g-all -2 (gravity -2 on every action) and read data/v4_main; Sample() asserts gt.json ranges/gravity,
and every candidate is checked against the observed config.json and contact_bc.json, so a setup mismatch aborts.
Sample selection rule, arms, budgets, failure rule and report carry over unchanged.
