"""V3l Part 2: PhysTwin's own evaluation (evaluate_chamfer.py / evaluate_track.py @81c71879), functions copied verbatim,
applied to a vertices array [68, N, 3] whose first rows follow PhysTwin's layout (object_points[0], surface_points, ...).

  .engine-envs/v3g_phystwin/bin/python -I scripts/v3l/pt_eval.py <case_dir> <vertices .npy|.pkl> [--out json]

Untrusted case data: pickles loaded with a restricted unpickler (numpy reconstruct/ndarray/dtype/scalar only).
"""
import argparse, json, pickle, sys
from pathlib import Path
import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
from scipy.spatial import KDTree

OK = {('numpy.core.multiarray', '_reconstruct'), ('numpy', 'ndarray'), ('numpy', 'dtype'), ('numpy.core.multiarray', 'scalar'),
      ('numpy._core.multiarray', '_reconstruct'), ('numpy._core.multiarray', 'scalar')}


class R(pickle.Unpickler):
    def find_class(self, m, n):
        if (m, n) in OK:
            return super().find_class(m, n)
        raise pickle.UnpicklingError(f'blocked {m}.{n}')


def rload(p):
    with open(p, 'rb') as f:
        return R(f).load()


# ---- verbatim from evaluate_chamfer.py
def evaluate_prediction_chamfer(start_frame, end_frame, vertices, object_points, object_visibilities, object_motions_valid,
                                num_original_points, num_surface_points):
    chamfer_errors = []
    if not isinstance(vertices, torch.Tensor):
        vertices = torch.tensor(vertices, dtype=torch.float32)
    if not isinstance(object_points, torch.Tensor):
        object_points = torch.tensor(object_points, dtype=torch.float32)
    if not isinstance(object_visibilities, torch.Tensor):
        object_visibilities = torch.tensor(object_visibilities, dtype=torch.bool)
    if not isinstance(object_motions_valid, torch.Tensor):
        object_motions_valid = torch.tensor(object_motions_valid, dtype=torch.bool)
    for frame_idx in range(start_frame, end_frame):
        x = vertices[frame_idx]
        current_object_points = object_points[frame_idx]
        current_object_visibilities = object_visibilities[frame_idx]
        current_object_motions_valid = object_motions_valid[frame_idx - 1]  # noqa: F841
        chamfer_object_points = current_object_points[current_object_visibilities]
        chamfer_x = x[:num_surface_points]
        chamfer_error = chamfer_distance(chamfer_object_points.unsqueeze(0), chamfer_x.unsqueeze(0), single_directional=True, norm=1)[0]
        chamfer_errors.append(chamfer_error.item())
    chamfer_errors = np.array(chamfer_errors)
    return {'frame_len': len(chamfer_errors), 'chamfer_error': np.mean(chamfer_errors), 'per_frame': chamfer_errors.tolist()}


# ---- verbatim from evaluate_track.py
def evaluate_prediction_track(start_frame, end_frame, vertices, gt_track_3d, idx, mask):
    track_errors = []
    for frame_idx in range(start_frame, end_frame):
        new_mask = ~np.isnan(gt_track_3d[frame_idx][mask]).any(axis=1)
        gt_track_points = gt_track_3d[frame_idx][mask][new_mask]
        pred_x = vertices[frame_idx][idx][new_mask]
        if len(pred_x) == 0:
            track_error = 0
        else:
            track_error = np.mean(np.linalg.norm(pred_x - gt_track_points, axis=1))
        track_errors.append(track_error)
    return np.mean(track_errors)
# ----


def evaluate(case_dir, vertices):
    case_dir = Path(case_dir)
    data = rload(case_dir / 'final_data.pkl')
    split = json.loads((case_dir / 'split.json').read_text())
    gt_track_3d = rload(case_dir / 'gt_track_3d.pkl')
    object_points = data['object_points']; num_original_points = object_points.shape[1]
    num_surface_points = num_original_points + data['surface_points'].shape[0]
    train_frame, test_frame = split['train'][1], split['test'][1]
    assert test_frame == vertices.shape[0], (test_frame, vertices.shape)
    a = [vertices, object_points, data['object_visibilities'], data['object_motions_valid'], num_original_points, num_surface_points]
    ctr = evaluate_prediction_chamfer(1, train_frame, *a); cte = evaluate_prediction_chamfer(train_frame, test_frame, *a)
    mask = ~np.isnan(gt_track_3d[0]).any(axis=1)
    _, idx = KDTree(vertices[0]).query(gt_track_3d[0][mask])
    ttr = evaluate_prediction_track(1, train_frame, vertices, gt_track_3d, idx, mask)
    tte = evaluate_prediction_track(train_frame, test_frame, vertices, gt_track_3d, idx, mask)
    return {'chamfer_train': float(ctr['chamfer_error']), 'chamfer_train_frames': ctr['frame_len'],
            'chamfer_test': float(cte['chamfer_error']), 'chamfer_test_frames': cte['frame_len'],
            'track_train': float(ttr), 'track_test': float(tte),
            'chamfer_per_frame': ctr['per_frame'] + cte['per_frame']}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('case'); p.add_argument('vertices'); p.add_argument('--out')
    a = p.parse_args()
    v = np.load(a.vertices) if a.vertices.endswith('.npy') else rload(a.vertices)
    r = evaluate(a.case, np.asarray(v))
    print(json.dumps({k: v for k, v in r.items() if k != 'chamfer_per_frame'}))
    if a.out:
        Path(a.out).write_text(json.dumps(r, indent=1))
