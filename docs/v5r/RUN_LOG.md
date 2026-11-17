# v5r run log (UTC, 2026-11-16)
- 02:55 pre-registration docs/v5r/PREREG_2026-11-16.md; samples docs/v5r/SAMPLES.json (= docs/v5q/SAMPLES_RESOLVED.json val 60, = V3r indices).
- 02:57 GT (full grid 0.06-0.64 + d=0) and twins (a) lawmean_pred, (b) prior_vision, (c) prior_vt, supplementary prior_vis3 (V2 grid)
  launched: a800-3 / a800-5 pools (2/GPU) + 12 samples on the a800-2 r_queue (phase v5r). Sample -> host fixed (scripts/v5r/make_jobs.py).
- 03:15 second a800-3 / a800-5 pools for the not-yet-started jobs (2/GPU each). Sims ~9-11 min each on the shared GPUs.
- 03:45 coordinator: hard deadline 05:55 for RESULTS; v5q stopped, a800-2 / a800-4 free. 03:50 not-started twin jobs of 48 samples (whole
  samples only, no collision with started jobs) moved to an a800-4 pool (5/GPU), switched to the FULL depth grid (V2 depths are simulated
  first, so the V2-grid result is available early; extended depths beyond the final D_top are simply unused). a800-2 queue cap 16/GPU.
  Priority order in the a800-4 list: prior_vision, prior_vt, lawmean_pred, prior_vis3.
- 04:02 a800-5 withdrawn (user): a800-5 v5r pools and sims killed (nothing left there); the 24 a800-5 samples (54 unfinished jobs: GT + twins, full grid, GT first) moved to an a800-4 pool docs/v5r/jobs_a4c.json (4/GPU, next to jobs_a4.json at 5/GPU); finished depths kept, in-flight sims rerun. Lists renamed *.WITHDRAWN.json.
- 04:20 deadline: not-started twin jobs back to the V2 grid (scripts/v5r/trim.py); the 88 not-started a800-4 twin jobs moved to the a800-2 queue (16/GPU), only those whose parameter set is unique within the sample (no cross-host cache dir; scripts/v5r/to_a2b.py). Order: vision, vt, class mean, vis3 last.
- 05:10 deadline cut (coordinator priority order; scripts/v5r/cut.py, list in docs/v5r/CUT.json): all 40 unfinished prior_vis3 jobs and the GT extended-depth tails of 29 samples (<= 2 of 4 extended depths done) stopped and removed from the lists/queue, to give the GPU time to the 3 main twins on the V2 grid (sims had slowed to ~19 min at 16/GPU on a800-2). Extended-grid analysis therefore covers only the GT-complete subset.
- 05:22 second cut: class-mean (a) jobs with <= 2 sims done (would not finish by 06:15; ~20-30 min per sim on a800-2 at 16/GPU) stopped so the vision+touch twins (key contrast (c)-(b)) can finish; (a) is reported on the samples where it completed.
