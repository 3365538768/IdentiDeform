"""Within-constitutive-law audit of saved validation predictions (reviewer a1nG).

Reports, per parameter, global and within-law Pearson r, the prediction/target spread ratio, and
the MAE of a law-mean oracle (GT law known, predict that law's val mean). The oracle uses val
statistics and GT law, so it is a diagnostic reference, not a deployable baseline.

  python scripts/audit_within_law.py runs/RUN/val_epoch_0016 [...] --output docs/WITHIN_LAW_AUDIT.json
"""
import argparse
import json
from pathlib import Path

import numpy as np

PARAMS = ["E", "nu", "rho", "sigma_y"]


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 and a.std() > 0 and b.std() > 0 else None


def audit(pred_dir):
    rows = [json.loads(line) for line in open(Path(pred_dir) / "predictions.jsonl")]
    pz = np.array([r["prediction_z"] for r in rows])
    tz = np.array([r["target_z"] for r in rows])
    mask = np.array([r["target_mask"] for r in rows])
    law = np.array([r["law_id"] for r in rows])
    out = {"predictions": str(pred_dir), "samples": len(rows),
           "law_accuracy": float(np.mean([r["predicted_law_id"] == r["law_id"] for r in rows])),
           "parameters": {}}
    for j, name in enumerate(PARAMS):
        ok = mask[:, j]
        if ok.sum() < 6:
            continue
        oracle_err = np.zeros(len(rows))
        per_law = {}
        for value in sorted(set(law[ok].tolist())):
            k = ok & (law == value)
            oracle_err[k] = np.abs(tz[k, j] - tz[k, j].mean())
            per_law[str(value)] = {"count": int(k.sum()), "pearson": pearson(pz[k, j], tz[k, j]),
                                   "spread_ratio": float(pz[k, j].std() / max(tz[k, j].std(), 1e-12))}
        out["parameters"][name] = {
            "count": int(ok.sum()), "pearson": pearson(pz[ok, j], tz[ok, j]),
            "model_mae": float(np.abs(pz[ok, j] - tz[ok, j]).mean()),
            "constant_mae": float(np.abs(tz[ok, j] - tz[ok, j].mean()).mean()),
            "law_mean_oracle_mae": float(oracle_err[ok].mean()), "within_law": per_law}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("prediction_dirs", nargs="+")
    parser.add_argument("--output")
    args = parser.parse_args()
    result = [audit(d) for d in args.prediction_dirs]
    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
