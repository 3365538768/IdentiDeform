# V5q stopped (2026-11-16 03:45 UTC): machines ending (~06:40 UTC)

This was the V5b C1 refinement, run as a supplementary experiment (gate decision: the main line stays on V2). It was stopped on the lead instruction because the servers are only available until about 06:40 UTC and the run could not finish.

Nothing will relaunch:
- docs/v5q/STOP exists.
- docs/v5q/queue.json has "gpus": [].
- The a800-4 pool and finisher are killed.
- All v5q simulations were killed on both hosts.

## What finished (data/v5b_main; gen_v3.py --g-all -2 --weak hold --ranges v5)

### Protocol
Pre-registered before any search in docs/v5q/PROTOCOL.json. It uses the same rules as V3q, with these V5b inputs:
- V5 bounds.
- Law means from V5b train 0-1999: docs/v5q/LAW_MEANS.json.
- Prior: the V5b main 9-run ensemble from docs/v5p/HANDOFF.json, built into docs/v5q/PRIOR_ENSEMBLE_{val,test}.json.

### Setup check
GT parameters were re-simulated through the candidate path for all 6 actions on val and test (12/12, including the held shear/stretch/bend). IoU was 0.99983-1.0000 and loss <= 3e-4. The candidate simulations therefore reproduce the V5b observations.

### Searches
Complete:
- val lawmean_gt: 11/60 (k = 24)
- val lawmean_best3: 3 (k = 12)
- val prior_top1: 3/60 (k = 24)
- test: nothing complete

Partial (resumable) states:
- val lawmean_gt: 49
- val best3: 3
- val prior_top1: 21

That is 813 evaluations, 776 unique simulations and 0 failed simulations.

### Partial aggregate
docs/v5q/RESULTS_PARTIAL.json covers complete searches only, with n, and is clearly labelled PARTIAL. It is too small for any claim. Errors at k = 0/4/12/24, in V5b prior z units:
- val lawmean_gt (n = 11): 0.578 / 0.306 / 0.192 / 0.192
- val prior_top1 (n = 3): 0.274 / 0.321 / 0.246 / 0.246

No paired or sims-saved analysis is possible at these n. The full deliverables (docs/v5q/RESULTS.json, RESULTS_ZH.md and docs/paper_figs/data/f1_bands_v5b.json) were NOT produced.

## Where things are
- Code: scripts/v5q/.
  - c1_refine.py holds the V5b setup.
  - The other scripts are enqueue, worker, pool, report, write_zh, status, plan_update, throttle, make_prior, law_means, make_protocol and finisher.sh.
- States: data/v5q_refine/<split>/<idx>/<arm>/state.json, plus _cache.
- Tasks: docs/v5q/tasks.json; done markers in runs/v5q_done/.
- Execution log: docs/v5q/DEVIATIONS.md.
