# V3r pre-registration: extended candidate-depth grid (2026-11-15 13:35 UTC)

Written BEFORE any extended-depth simulation and BEFORE any twin result has been looked at.

## Why
GT-only check of the V2 grid on the 60 V3 val samples (docs/v3r/RUN_LOG.md, 11:25 / 12:35 UTC): the GT oracle depth is the largest
candidate (0.32) for 47/60 samples, and every sample is GT-safe at 0.06. The stiffer V3 materials (log10 sigma_y/rho 3.2-4.5,
log10 E/rho >= 3.5 / 4.5) push the V2 task to its ceiling: success cannot discriminate between twins (fixed 0.06 = 100 %).
Decision (coordinator request, 13:20 UTC): extend the candidate grid upward so that the GT-oracle depth is spread out again.

## Unchanged
tau = 3 % permanent crush, Procrustes-aligned crush metric, d_min = 0.06, squeeze action (own sticky jaws, close 1.0 s, hold to
1.5 s, release, relax to 4.0 s, g = 0), planning rule (largest twin-feasible candidate, else abstain), success / regret / utility
definitions, fixed-depth leave-one-out heuristic over the (final) grid + abstain, object bootstrap. V2 numbers stay on the V2 grid.

## Extension
- Extra candidate depths simulated for GT on all 60 samples: 0.40, 0.48, 0.56, 0.64 (plus a re-simulated d = 0 reference, since
  the reference dumps were deleted after measuring; Warp-MPM re-simulation agrees to < 4e-6, docs/v3e/DETERMINISM.json).
- Geometry cap: a depth is admissible for a sample only if the jaw gap stays >= 3 grid cells, (1 - d) W >= 3 dx
  (dx = 2/150). Smallest W on the 60 samples is 9.56 dx -> d <= 0.686 for every sample, so all candidates up to 0.64 are admissible
  for all 60 (no sample-specific capping needed).
- Numerical failures: a depth whose GT simulation fails (MPM blow-up, after the existing quick-retry rule) counts as GT-infeasible,
  exactly as in V2; failures per depth are reported, nothing is dropped silently. If >= 10 % of samples fail at a depth, that depth
  is excluded from the grid and this is reported.

## Final-grid rule (fixed now)
Final grid = V2 grid + extra depths up to D_top, with D_top = the SMALLEST value in {0.40, 0.48, 0.56, 0.64} for which at most
1/3 of the 60 samples have GT oracle depth = D_top (top-censored). If no value satisfies it, D_top = 0.64 and the residual ceiling
is reported. The choice uses GT outcomes only (no twin results), and is recorded in docs/v3r/GRID_FINAL.json.

## Twin simulations already started under the V2 grid
7 refined_best3 twin jobs (samples 2, 27, 31, 40, 60, 135, 145; V2 grid) were launched 12:56-13:12 UTC before this note;
their outputs have not been opened. They are extended to the final grid like every other twin (resumable state; the existing
depths are kept, only the new depths are simulated).
