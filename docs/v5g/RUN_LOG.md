# v5g run log (UTC, 2026-11-16)
- 06:20 v5r phase removed from the a800-2 r_queue (gpus [], jobs []); no v5r pools/sims left on a800-4 or a800-2.
- 06:25-06:40 pilot 1 (GT, 6 samples x 2 positions x 5 levels): (i) a TypeError (fixed), (ii) upstream press forces a 4 s clip,
  (iii) upstream jaw cuboids (6 phases x 2 jaws, each a full-grid kernel per substep) ran 4 s/frame. Rewritten: sim_type stretch
  (clip length from the config, 2.6 s), jaws as one small kernel per jaw over its swept region (also measures the grip force): 25 s/sim.
- 06:47-06:55 pilot 1b: levels 0.04-0.20 identical (the 3-cell depth floor of _gripper dominated) -> floor 1 cell, levels 0.03-0.40.
- 06:58-07:05 pilot 2 (levels 0.03-0.25): both failure modes occur for GT (drop at low levels / end grips / soft objects; damage at
  high levels on plastic objects); thresholds set to left > 0.5 (drop) and |crush| > 5 % (damage); level 0.40 added.
- 07:07 pre-registration written; main run launched on a800-4 (docs/v5g/jobs_main.json, 3120 one-simulation jobs, 9/GPU, GT first).
- 07:20 throughput ~7 jobs/min on a800-4 alone (3120 jobs would take ~7 h): samples with no started job split off; 24 samples (1248 jobs) to an a800-3 nohup pool (GPUs 0,1,3-7; GPU 2 belongs to another user; 9/GPU), 36 samples stay on a800-4. Each sample on one host. Job order: GT, then the prior twins, class mean last.
- 07:32 coordinator: a800-3 split (v5g: GPUs 0,1,3,4 (+2 after v3j); 5,6,7 to another agent). The first a800-3 pool keeps only its 135 started jobs (27 were running on GPUs 5/6/7, single simulations, finish within ~10 min); the 1113 unstarted jobs go to a new pool docs/v5g/jobs_a3b.json on GPUs 0,1,3,4.
- 08:20 coordinator: throughput ~16 jobs/min, 3120 jobs do not fit: unstarted jobs reordered sample-major (scripts/v5g/reorder.py; fixed sample order of docs/v5r/SAMPLES.json, all arms of a sample together) so whole samples complete; analysis on paired-complete samples only (scripts/v5g/analyze.py). Nothing running was cancelled.
- 08:41 a800-3 GPU 2 released by v3j: a third a800-3 pool (docs/v5g/jobs_a3c.json, --gpus 2, 9/GPU) takes the last fifth of the remaining a800-3 samples (whole samples). PREREG_ADDENDUM.md (08:18) made per-candidate outcome prediction the primary metric before any paired-complete twin result existed.
- 08:45 DEVIATION (coordinator: shrink so that complete paired results exist): grip levels 0.06 and 0.16 dropped from all not-started jobs -> 8 candidates {end, centre} x {0.03, 0.10, 0.25, 0.40}; the analysis uses these 8 candidates for EVERY sample (earlier samples have all 12, the extra 4 are ignored). Per-method completeness is now read from the cache state (scripts/v5g/plan_eval.load_state).
- 10:20 FINAL CUT: analysis on the samples paired-complete at this time (pools keep running; nothing later is used).
- 10:25 final: RESULTS.json / RESULTS_ZH.md / PREDICTION.json / F6b_grasp.pdf written from the 10:20 cut (n = 56 paired-complete). v5g pools on a800-3 / a800-4 stopped by PID after the cut (no v5g process left). data/v5g kept (results source); nothing deleted.
