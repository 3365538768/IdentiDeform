# V5g pre-registration addendum: grasp-outcome PREDICTION as the primary metric

Written 2026-11-16 08:18 UTC, BEFORE any twin result has been looked at: no sample has all four twins complete (paired n = 0); the
only twin outputs that exist are 6 vision-prior samples, and the only aggregate shown so far, in a partial test print of the report
script, covers those 6 (success 4/6) — no per-candidate twin predictions have been inspected.

Reason (coordinator / user, 08:20 UTC): under the pre-registered planning metric only ~15/56 samples have any GT-feasible grasp,
so "always abstain" wins on success. The user's claim is that a twin lets one EVALUATE grasp positions and forces before execution:
grasp-outcome prediction (cf. DefGraspSim).

## Primary (new): per-candidate outcome prediction
- Unit: (sample, candidate grasp) pair; 12 candidates per sample. Each twin predicts the class {success, drop, damage} from its own
  simulation (same outcome rule and thresholds as the pre-registration: left > 0.5 -> drop, else |crush| > 5 % -> damage, else
  success); GT gives the true class.
- Metrics over all pairs of the paired-complete samples: 3-class macro-F1, accuracy, per-class recall; per-sample feasibility
  accuracy (twin says "some candidate succeeds" vs GT); safe-grasp precision (among pairs predicted success, fraction GT success)
  and recall (among GT successes, fraction predicted success).
- Object (cluster) bootstrap over object identities, 2000 draws, 95 % percentile CIs; paired contrasts vt - vision and
  vision - class-mean (and every twin vs the baselines), bootstrapped on the same draws.
## Non-twin baselines
- (i) majority class over all pairs (of the other objects, leave-one-object-out);
- (ii) per-candidate LOO majority: for each candidate, the most common GT class on the OTHER objects (knows position/force effects,
  not the material);
- (iii) per-candidate x vision-predicted-law LOO majority: most common GT class among the other objects whose vision-prior law equals
  this sample's vision-prior law (falls back to (ii) if none): knows the material class, not the parameters.
## Secondary
- The pre-registered planning metrics (success with abstain, drop, damage, missed, level / force regret, fixed-grasp LOO), reported
  as is, stating that always-abstain wins on raw success.
- Planning success on the GT-feasible subset, labelled descriptive (selected on GT outcomes).
## Analysis set
- Paired-complete samples only (GT + vision + vision+touch + class mean all complete), jobs ordered sample-major in the fixed sample
  order of docs/v5r/SAMPLES.json; final cut ~10:20 UTC, n reported.
