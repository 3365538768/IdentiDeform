"""Helpers shared by the refinement scripts."""
import json


def best_dir(run):
    """Validation-prediction directory of the best joint-training epoch of a fieldbridge run (by val MAE)."""
    timing = [json.loads(l) for l in (run / 'timing.jsonl').read_text().splitlines() if l.strip()]
    joint = [t for t in timing if t.get('phase') == 'joint_training' and t.get('val_mae') is not None]
    best = min(joint, key=lambda t: t['val_mae'])
    return run / f"val_epoch_{best['epoch'] - 1:04d}", best
