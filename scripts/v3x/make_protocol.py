"""V3x: write docs/v3x/jobs_a2.json, jobs_a3.json and the pre-registration docs/v3x/PROTOCOL.json."""
import hashlib, json, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
S = json.loads((ROOT / "docs/v3x/SAMPLES.json").read_text()); order = S["new_order"]
base = ".venv/bin/python scripts/v3d/c1_refine.py --run v2main_C_seed29 --split test --ensemble docs/v3d/ENSEMBLE_BCD9_TEST.json"
def job(i, arm, out):
    return {"id": f"v3x_{arm}_{i:06d}", "cmd": f"{base} --out {out} --index {i} --arm {arm} --budget 24 --replay-budgets 24"}
a2 = [job(i, arm, "data/v3x_refine_test") for i in order for arm in ("prior_top1", "lawmean_best3")]
a2 += [job(i, "lawmean_gt", "data/v3x_refine_test") for i in order]
a3 = [job(i, "prior_top2", "data/v3x_refine_test_a3") for i in order]
(ROOT / "docs/v3x/jobs_a2.json").write_text(json.dumps(a2, indent=1)); (ROOT / "docs/v3x/jobs_a3.json").write_text(json.dumps(a3, indent=1))
sha = lambda p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest()
P = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
 "status": "pre-registration, written BEFORE any v3x simulation; not changed afterwards (deviations go to docs/v3x/RUN_LOG.md)",
 "purpose": "enlarge the held-out TEST refinement of docs/v3d (30 samples / 9 objects) by 30 new test samples (all 10 test objects) -> 60-sample analysis + new-30 replication",
 "samples": {"n_new": 30, "indices_in_run_order": order, "original_30": S["original_30"], "file": "docs/v3x/SAMPLES.json",
   "rule": S["rule"], "script": "scripts/v3x/select_more.py", "objects_missing_in_original_30": S["objects_missing_in_original"]},
 "refinement": "IDENTICAL code scripts/v3d/c1_refine.py (unchanged, sha below), same prior (docs/v3d/ENSEMBLE_BCD9_TEST.json 9-run B/C/D ensemble, targets from runs/v2main_C_seed29/test_once), compass search, budget 24, nested budgets, same failure rule (quick crash <60 s retried once; failed sim loss 9, counted, never selected; no sample dropped).",
 "only_change": "--replay-budgets 24 for every arm (v3d: top2/top1 0,4,12,24; gt/best3 12,24) and no --gt-observed noise-floor sims, to save simulation time; replay runs after the search and cannot affect the refinement. Output dirs data/v3x_refine_test (a800-2) and data/v3x_refine_test_a3 (a800-3, prior_top2 only, own GT replay; no file shared across hosts).",
 "arms_priority": ["prior_top1", "lawmean_best3", "lawmean_gt", "prior_top2 (only on a800-3 once free, if time)"],
 "execution": "a800-2: setsid nohup scripts/v3x/pool.py docs/v3x/jobs_a2.json (top1+best3 interleaved per sample, then lawmean_gt); a800-3 after docs/v3j/DONE: docs/v3x/jobs_a3.json",
 "metrics": "identical to docs/v3d: pooled identifiable z-MAE over E/rho and sigma_y/rho at k=0/4/12/24 (best3 k=0 = n/a, first point at k=3); paired object-clustered bootstrap 95% CI (2000 draws, seed 17) of per-sample differences; sims for lawmean_gt to reach prior k=0 error (log-budget interpolation, object bootstrap, censored >24); unseen-action replay IoU at k=24.",
 "analysis_sets": {"combined_60": "original 30 (docs/v3d data) + new samples paired-complete in the compared arms", "replication_new": "new samples alone"},
 "completion_rule": "Hard stop 10:30 UTC 2026-11-16. Each comparison is reported only on samples complete (budget 24 reached) in BOTH arms, n stated. Arms incomplete at k=24 may be reported at lower k on samples whose search has reached that k (evals are nested), labelled partial. No other selection.",
 "no_selection_on_test": "no hyperparameter, sample, arm or metric choice depends on any test outcome",
 "sha256": {p: sha(p) for p in ["scripts/v3d/c1_refine.py", "scripts/v3b/c1_refine.py", "scripts/v3x/select_more.py", "docs/v3x/SAMPLES.json",
   "docs/v3d/ENSEMBLE_BCD9_TEST.json", "runs/v2main_C_seed29/test_once/predictions.jsonl", "data/manifests/v2main_withtest.jsonl", "scripts/v3x/pool.py"]}}
assert P["sha256"]["scripts/v3d/c1_refine.py"] == "951cb201e8ad855195ebba92fb8146543ce3507cf3fb21d10da2898ecf737e80"
(ROOT / "docs/v3x/PROTOCOL.json").write_text(json.dumps(P, indent=1)); print(len(a2), len(a3))
