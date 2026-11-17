# Paper figure sources

Paths are relative to the project root (`/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test`).
Regenerate with:

    .venv/bin/python scripts/paper_figs/compute_f1_bands.py         # F1 bands -> docs/paper_figs/data/f1_bands.json
    .engine-envs/plot/bin/python scripts/paper_figs/make_figs.py    # all figures (PDF + 300-dpi PNG)

Layout: each PDF/PNG is exactly 6.875 in (full width) or 3.25 in (single column) wide, with symmetric left and
right whitespace. `save()` in make_figs.py rescales the axes until the tight content box spans the width minus
0.03 in on each side, then saves a box of exactly that width centred on the content, cropped tight vertically.
Panel letters are small bold "(a)", "(b)", ... at the top-left of each panel; there are no titles.

Style: STIX (Times-like) serif, all text ≥ 7 pt at print size, Okabe–Ito colour-blind-safe palette. Each method keeps its
colour in every figure, with marker and line style as a second cue. Plots have no titles. Widths are 3.25 in (single
column) and 6.875 in (full width). "z-MAE" means MAE in train-standardized target units. Unless noted, CIs are 95%
object (cluster) bootstrap intervals with 2000 draws.

## F1_teaser (full width)
- **(a) Val, 60 samples / 10 objects.** Curves at every k = 0..24 come from `docs/v3h/PRIOR_QUALITY.json`
  `prior_top1_val60.result.arms.{prior_top1,prior_top2,lawmean_gt,lawmean_best3}.curve`. The same values (for
  prior_top2, lawmean_gt and lawmean_best3, at k ∈ {0,3,4,8,12,16,24}) also appear in
  `docs/v3b/C1_REFINE_RESULTS.json` `prior_top2.curves` and `curves_common`. lawmean_best3 is drawn from k = 3,
  after its 3 law probes; prior_top2's budget includes its 2 probes.
  - Bands are recomputed at every k by `scripts/paper_figs/compute_f1_bands.py`. It uses the
    `scripts/v3h/report.py` loaders on the stored refinement states
    (`data/v3b_refine/v2main_C_seed29/<idx>/<arm>/state.json`) with the same bootstrap (seed 17, 2000 draws). The
    script asserts that every curve value matches PRIOR_QUALITY.json to 1e-9 and that the stored CIs at
    k ∈ {0,4,12,24} (`...arms.<arm>.mae.<k>.ci95`) are reproduced exactly. Output:
    `docs/paper_figs/data/f1_bands.json` `val60`. No prior_top2 band is drawn, because it overlaps prior_top1.
  - Annotation "11.7 sims saved" is `docs/v3b/C1_REFINE_RESULTS.json`
    `prior_top2.speedup.lawmean_gt.by_prior_budget.0.uninformed_sims_to_match` = 11.71 (bootstrap CI
    `[6.53, censored]`, censored fraction 0.067). This value is for prior_top2 at k = 0. prior_top1 at k = 0 is
    identical: `PRIOR_QUALITY.json prior_top1_val60.paired.top1_minus_prior_top2_k0` = 0.0. The arrow is drawn at
    y = prior_top1(k = 0) = 0.344.
- **(b) Test, 30 samples / 9 objects.** Curves are `docs/v3d/C1_TEST_RESULTS.json` `curves.<arm>.<k>.identifiable_mae_pooled`.
  prior_top1 comes from `prior_top1_addendum.curves.prior_top1`. The test file only has budgets
  k ∈ {0,2,3,4,8,12,16,24}, so the test lines are piecewise-linear between them.
  - Bands are an object bootstrap of the per-sample `samples.<arm>[].ident_err.<k>` (E/ρ and σ_y/ρ errors; None
    entries dropped), recomputed by `compute_f1_bands.py` (seed 17, 2000 draws). The pooled means are asserted
    equal to the stored curves. The test file does not store CIs for the pooled MAE itself, only for paired
    differences, so these bands are our recomputation.
  - Annotation "9.7 sims saved" is `prior_top1_addendum.speedup.lawmean_gt.by_prior_budget.0.uninformed_sims_to_match`
    = 9.70 (CI [4.66, 18.66]). The same value is
    `preregistered_replication.H2_privileged_sims_to_match_top2_budget0`.
- **(c) Gripper success**, n = 55 (short labels: GT oracle, Prior+12, Prior, Best3+12, Fixed, Law mean): `docs/v3e/GRIPPER_TASK.json` `summary.<twin>.all.success` and
  `success_ci95_object_bootstrap`. "Prior + 12 sims" = `refined_top2` (prior_top2 + 12-sim refinement);
  "Prior, 0 sims" = `ensemble_prior`. No prior_top1 + refinement twin exists in the gripper task.

## F2_init_vs_refined (single column)
- Source: `docs/v3h/PRIOR_QUALITY.json` `main.arms.<arm>.mae.<k>.{mean,ci95}`. This is the **30-sample, 9-object
  validation subset**, not the 60-sample set used in F1.
- x is the error at `init_budget` (0; 3 for lawmean_best3). y is the error after 4 sims (a) or 12 sims (b). Error
  bars in x and y are the stored CIs.
- Arms: prior_top1 (V-JEPA 9-ens, top-1), prior (V-JEPA single), dino_c (DINOv3 C ens.), gtnoise_{0.25,0.5,1},
  lawmean_gt, and lawmean_best3. lawmean_best3 is drawn hollow because it is a probe arm and is not in the fit.
  prior_top2 is left out for readability; its values are in the same JSON.
- Linear fit: `scaling_fit_no_probe_arms.{4,12}.{slope,intercept,r}`, fitted on the 7 filled arms listed in
  `...arms`. The dashed diagonal is y = x.
- **Refinement floor band.** The JSON has no "floor" key; the text in PRIOR_QUALITY_ZH.md says "≈0.2–0.3". The
  band here comes from the data: the min–max of `main.arms.<a>.curve.12` over the fit arms with initial error
  ≤ 0.35 (prior, prior_top1, dino_c, gtnoise_0.25, gtnoise_0.5), which gives **0.202–0.282**. The same band is
  drawn in both panels.

## F3_identifiability (full width) and F3s_identifiability_dinov3 (single column, supplementary)
Both figures use the same bars: the bar is the 3-seed mean, dots are the individual seeds, V1 / baseline condition
is grey and V2 / improved condition is blue. Each r is the within-law Pearson r averaged over laws.

- **F3 (a) V1 vs V2 data, V-JEPA 2 family, model C.**
  - V1: `docs/p1c/P1C_RESULTS.json` `groups.Cvg.within_law_r.{E,sigma_y}`. This is ViT-g. P1c used the
    identifiable targets (`docs/P1_RESULTS_2026-11-10_ZH.md` §2), so key `E` is r(E/ρ) and `sigma_y` is r(σ_y/ρ).
  - V2: `docs/v2main/V2MAIN_RESULTS.json` `groups.C.within_law_r.{E,sigma_y}`. This is ViT-L. The values equal
    `docs/v3c/BASELINES_VAL.json` `methods."vjepa2_32f_C (reference)".within_law_r.*.mean_over_laws`.
  - **Gap:** no V1 ViT-L run on the identifiable targets reports within-law r. `docs/p1/P1_RESULTS.json` (ViT-L)
    is in the raw E/ν/ρ/σ_y space and is not used. The V1 and V2 backbone sizes therefore differ, as stated in
    the F3s axis label; the caption should say so too.
  - **Gap:** V1 has σ_y/ρ labels only for metal (R_LINE doc: plasticine "无标签", no labels). V2 averages over
    metal and plasticine.
- **F3s supplementary (same layout as F3a).**
  - (a) DINOv3 C. V1: `docs/r1b/R1B_RESULTS.json` `groups.C.within_law_r.{E,sigma_y}`. R1b used the identifiable
    targets (R_LINE doc §2). V2: `docs/v3c/BASELINES_VAL.json`
    `methods.dinov3_16f_C.within_law_r.{E_over_rho,sigma_y_over_rho}.values`.
  - (b) V-JEPA 2: same data as F3a.
- **F3 (b) P4d damping pair**: `docs/p4d/P4D_RESULTS.json` `arms.{v1,low}.runs[].within_law_r`. Per run, r(E/ρ)
  is the mean of `E_over_rho/{jelly,metal,plasticine}/all` and r(σ_y/ρ) is the mean of
  `sigma_over_rho/{metal,plasticine}/all`. The arm-level `arms.<arm>.within_law_r` dict is not used because its
  aggregation is undocumented.
- **F3 (c) R2 trajectory equivalence, press only**: `docs/r2/MPM_IDENTIFIABILITY_{jelly,plasticine,metal}.json`
  `results[action=press].relative_rms` (all-particle trajectory RMS / reference motion RMS), log scale.
  - Bars:
    - E×4, jelly (`E_x4`) = 0.473.
    - E×4, metal = 0.0104.
    - (E, ρ, σ_y)×4 (`E_rho_sigma_x4`). This bar is the maximum over the three materials, 9.5e-5 (metal). Dots
      mark the three per-material values: jelly 1.2e-6, plasticine 7.6e-5, metal 9.5e-5.
  - **"Numerical floor" line (gap):** R2 has no repeat-run or precision measurement. The dashed line is drawn
    at the largest joint-scaling residual (9.5e-5): the joint ×4 change is pure solver noise by construction
    (the trajectory is analytically invariant), so this marks the noise floor and is not an independent
    measurement. The only re-simulation agreement check in the repo, `docs/v3e/DETERMINISM.json` (<4e-6 absolute,
    gripper metrics), uses different metrics and is not used here.
  - The drop action, the `rho_x4` / `sigma_x4` cases and the ×0.25 cases are in the same files and are not
    plotted. For drop, the joint ×4 residual is 1.7e-3 to 7.8e-3.

## F4_scaling_views (full width)
- Source: `docs/v3i/ABLATIONS.json`.
- (a) `scaling.{B,C}_{f10,f25,f50,f100}.val_mae.{mean,sd}`.
- (b) `scaling.C_*.within_law_r_{E_over_rho,sigma_y_over_rho}.{mean,sd}`.
- (c) `views.C_v{1,2,3}.val_mae`.
- (d) `views.C_v*.within_law_r_*`.
- Error bars are ±1 sd over 3 seeds.
- The object counts 9/22/44/87 are taken from the `protocol` string ("87 train objects -> 9/22/44/87 objects").
  There is no numeric key for them, and `runs[].train_samples` is null.
- Paired object-bootstrap tests are in `pairs[]` and are not plotted.

## F5_baselines (single column)
- Val values are from `docs/v3c/BASELINES_VAL.json` `methods.<m>.standardized_mae`. Test values are from
  `docs/v3c/BASELINES_TEST.json` `methods.<m>.standardized_mae`.
- PIXIE is from `docs/v3f/PIXIE_BASELINE.json` `{val,test}.{filled,filled_train_calibrated}.standardized_mae`.
  `filled` is the primary variant; `shell` is not plotted.
- V-JEPA 2 ensemble (9): the val value is `BASELINES_TEST.json methods.vjepa2_ensemble_BCD9.val.standardized_mae`.
- Rows are ordered by test MAE. The x axis is broken between 0.9 and 1.3 to fit Qwen3-VL and released PIXIE.
- **Error bars** are ±1 sd over 3 seeds, with tick marks at the seed values, and appear only for the trained
  3-seed models (`standardized_mae.{mean,sd,values}`).
- **Gap:** deterministic or single-run methods (constant, GT-law mean, Qwen3-VL zero-shot, PIXIE, the ensemble)
  have no stored uncertainty for their absolute MAE. The files contain only paired object-bootstrap CIs of the
  difference against V-JEPA C (`vs_vjepa_C`), so these methods are drawn without error bars.

## F6_gripper (single column)
- Source: `docs/v3e/GRIPPER_TASK.json` `summary.<twin>.all`.
- (a) `success` and `success_ci95_object_bootstrap`.
- (b) `regret_mean` and `regret_ci95_object_bootstrap`.
- n = 55; 5 ice_telescope samples are excluded (`excluded_samples`).
- Paired differences (`paired_success_differences`) are not plotted. For example, refined_top2 minus
  ensemble_prior success = 0.0 [−0.153, 0.133].
- Candidate qualitative panel: `assets/v3e_outcomes.png`, copied from `docs/v3e/figs/outcomes.png`.

## F7_breakdown (full width)
- Source: `docs/v3h/PRIOR_QUALITY.json` `breakdown_60`. These are the 60 C1 val samples.
- (a) `breakdown_60.{action,gt_law}.<group>.{prior_top1,lawmean_gt}.{0,24}`.
- (b) `...diff.prior_top1_minus_lawmean_gt_k{0,24}.{mean,ci95}`.
- Group sizes are `...n`.
- `drop_vs_driven` is in the file and is not plotted.

## Copied assets (`assets/`)
| file | source |
|---|---|
| `v3f_springgaus_real_replay.png` | `docs/v3f/springgaus_real_replay.png` |
| `v3h_prior_quality.png` | `docs/v3h/prior_quality.png` (earlier internal version of F2) |
| `v3e_*.png` | `docs/v3e/figs/*.png` (outcomes, sheet_029/182/293, pilot_particles_g0/gravity) |
| `v3a_check_val_1_80000.png` | `docs/v3a/check_sheets/v3a_check_val_1_80000.png` (one of 5 check sheets) |

## Not produced
- **Qualitative strip** (observed 3-view frames vs prior replay vs refined replay) was not made. The refinement
  states (`data/v3b_refine/v2main_C_seed29/<idx>/<arm>/state.json`) store only parameters, losses and scalar
  replay IoU/RGB-MAE. They store no rendered frames or particle trajectories. Making the strip would require new
  simulations and renders, which this task excluded.
