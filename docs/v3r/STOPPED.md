# v3r / v3s stopped — switch to V4 data (2026-11-15 15:53 UTC)

Coordinator instruction (user decision): V4 data = V3 contact setup + gravity 2 on every action, because the V3 prior is much weaker.
All v3r/v3s work stopped: docs/v3r/queue.json and docs/v3s/queue.json have gpus = []; a800-4 v3r pools killed (POOL_STOP files
in docs/v3r, docs/v3s); running simulations are killed by main. Nothing will be relaunched without new instructions.

## State kept (valid for reuse)
- Code: scripts/v3e/* parameterised (FB_GRIP_DATA / FB_GRIP_TWINS / FB_GRIP_DEPTHS / FB_SQUEEZE_CLAMP; V2 defaults unchanged;
  originals in .cache/keep/v3e_before_v3r/), clamp port in scripts/v3e/squeeze_main.py, scripts/v3r/* (select, twins, make_jobs,
  add_ready, ensure_pool, report, status, plan_update), scripts/v3s/* (prior_v3, make_queue, pt_eval_run.sh, report),
  scripts/v3k/a_sg_refine.py --prior-json (original in .cache/keep/v3k_before_v3s/).
- Pre-registration: docs/v3r/PREREG_GRID_2026-11-15.md (extended grid 0.40/0.48/0.56/0.64, final-grid rule: smallest D_top with
  <= 1/3 of samples GT-oracle at D_top). Final grid NOT yet chosen (GT extension incomplete).
- GT squeezes (g = 0, independent of the action data; valid wherever V4 uses the same val indices/objects/GT parameters):
  data/v3r/runs/<idx>/gt — V2 grid + d=0 complete for all 60 samples (0 failures); extended depths complete for 51/60;
  missing 164, 176, 183, 195, 210, 228, 253, 261, 278 (killed mid-run; resumable: same squeeze_run.py command with
  FB_GRIP_DEPTHS=0.06,...,0.64 continues from state.json). GT-only V2-grid summary: .cache/v3r_gt_only.json
  (oracle 0.32 for 47/60, fixed 0.06 = 100 %).
- NOTE for V4: V4 changes the generator gravity, not the GT parameters, if it keeps gen_v2.sample draws + V3 ranges + v3 objects;
  then the GT squeezes above are reusable as-is (check data/v4_*/val/<i>/config.json material keys == data/v3_main/val/<i>).
- Twins: 7 refined_best3 V2-grid twin runs (samples 2, 27, 31, 40, 60, 135, 145; V3q lawmean_best3 budget 12) exist, not inspected;
  obsolete for V4 (refinement on V3 videos). No prior twin was simulated.
- v3s: data/v3s/prior20.json and data/v3s/pt/pt_prior.json = V3 9-run prior on Spring-Gaus (14 scenes) / PhysTwin (not analysed);
  completed v3s jobs: kA apple/duck, p20 C/E/apple/duck/torus, ref E/apple; 35 pending jobs disabled. Wrapper verified to reproduce
  the V2 prior exactly (data/v3s/smoke/prior20_v2.json). For V4: write docs/v3s/PRIOR_SOURCES.json from the V4 hand-off and rerun
  scripts/v3s/make_queue.py (outputs data/v3s should then be moved aside).
- Disk: data/v3r 2.8 GB (subset trajectories), data/v3s 31 MB; nothing deleted yet (GT runs are reusable).
- Run log: docs/v3r/RUN_LOG.md.
