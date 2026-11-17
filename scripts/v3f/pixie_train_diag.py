"""V3f PIXIE: diagnostic on the 87 TRAIN objects (PIXIE never saw them; nothing is fitted here except the reported
correlations): sample-level law accuracy, Pearson/Spearman of PIXIE z vs target z for E/rho, nu, rho, within-law r."""
import json, sys
from pathlib import Path
sys.path.insert(0, "scripts/v3f")
import numpy as np
from scipy.stats import pearsonr, spearmanr
from fieldbridge.data import ParameterCodec, specific_parameters
import pixie_score as ps
codec = ParameterCodec(json.loads(Path(ps.STATS).read_text()))
d, per = ps.predictions("train", "filled", codec)
rows = [json.loads(l) for l in open(d / "predictions.jsonl")]
P = np.array([r["prediction_z"] for r in rows]); T = np.array([r["target_z"] for r in rows]); M = np.array([r["target_mask"] for r in rows], bool)
L = np.array([r["law_id"] for r in rows]); PL = np.array([r["predicted_law_id"] for r in rows])
out = {"samples": len(rows), "objects": len(per), "law_accuracy": float((L == PL).mean()),
       "majority_law_rate": float(np.bincount(L).max() / len(L)), "confusion_gt_rows_pred_cols": [[int(((L == i) & (PL == j)).sum()) for j in range(3)] for i in range(3)]}
for j, n in enumerate(["E_over_rho", "nu", "rho"]):
    ok = M[:, j]; w = []
    for law in range(3):
        k = ok & (L == law)
        if k.sum() > 5 and P[k, j].std() > 0: w.append(float(pearsonr(P[k, j], T[k, j])[0]))
    out[n] = {"pearson": float(pearsonr(P[ok, j], T[ok, j])[0]), "spearman": float(spearmanr(P[ok, j], T[ok, j])[0]), "within_law_r_mean": float(np.mean(w))}
rep = json.loads(ps.OUT.read_text()); rep["train_diagnostic_filled"] = out; ps.OUT.write_text(json.dumps(rep, indent=2) + "\n")
print(json.dumps(out, indent=1))
