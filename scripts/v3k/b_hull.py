"""V3k-B: reconstructed geometry for V2-main val samples — 3-view visual hull from the FRAME-0 object masks.

  .venv/bin/python scripts/v3k/b_hull.py --indices 21 52 ... [--split val] [--vox 0.006] [--dilate 1]

Per sample (only the observed sample_pack frame 0 is used for the hull; the GT asset is used ONLY for (a) the
camera check and (b) the GT normalisation that defines where the observed cameras sit in mpm space, which is
known to the renderer — not shape information leaked into the hull):
  cameras   replicate upstream modified_simulation.py synthetic orbit cameras: azimuths = sorted
            RandomState(seed=index).uniform(0,360) draws with min gap 20 deg (3 views); elevation 13, radius
            init_radius / camera_distance_scale (world units), centre = undo(mpm (1,1,1)), up = mpm +z,
            observant basis = generate_local_coord, FoV 60 deg, 800x800 pinhole (OpenCV axes: x right, y down,
            z forward); pack views are ordered by view-dir NAME (lexicographic 'az{round}_el13').
            Pixel mapping 800 -> 224: u224 = (u800 + 0.5) * 224/800 - 0.5.
  check     GT asset gaussians (x90 rotation, sigmoid(opacity) > 0.02) projected (2-px dilated splat at 224)
            vs observed frame-0 mask -> IoU per view.
  hull      voxel grid (vox mpm units) over [0.5, 1.5]^3 mpm space (GT normalisation); keep voxels whose centre
            projects inside all three frame-0 masks (masks dilated by --dilate px); largest 26-connected component; only the
            surface shell (voxels with an empty 26-neighbour) is written as gaussians (--no-shell: solid) so that
            the pipeline's fill_particles fills the interior exactly as for any 3DGS asset.
  colour    per view z-buffer (voxel visible if depth <= nearest depth at its 224-px pixel + 2 vox); mean frame-0
            RGB over visible views; fallback mean object colour.
  asset     hull voxels -> world (undo GT normalisation) -> raw asset frame (undo the pipeline's +90 deg x
            rotation) -> 3DGS PLY in the same layout as the assets (x,y,z,f_dc,opacity,scale,rot,rgb; sh0),
            isotropic scale = 0.6*vox in world units, opacity 0.99. The pipeline renormalises it with its own
            quantile box (identical asset routine).
"""
import argparse, glob, json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / 'data/geometry/kaggle_v2/3DGS_model/3DGS_model'
PI_UP = 3.1415926  # upstream generate_rotation_matrix uses this constant
SIZE, RENDER = 224, 800

PLY_PROPS = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'), ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'), ('opacity', 'f4'),
             ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'), ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'),
             ('rot_3', 'f4'), ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]


def read_ply(path):
    b = Path(path).read_bytes(); i = b.index(b'end_header\n') + len('end_header\n')
    head = b[:i].decode().splitlines(); n = int(next(l for l in head if l.startswith('element vertex')).split()[-1])
    tmap = {'float': 'f4', 'uchar': 'u1', 'double': 'f8', 'int': 'i4'}
    props = [(l.split()[2], '<' + tmap[l.split()[1]]) for l in head if l.startswith('property')]
    return np.frombuffer(b[i:], dtype=np.dtype(props), count=n)


def write_ply(path, rec):
    head = 'ply\nformat binary_little_endian 1.0\nelement vertex %d\n' % len(rec['x'])
    head += ''.join(f"property {'float' if t == 'f4' else 'uchar'} {k}\n" for k, t in PLY_PROPS) + 'end_header\n'
    arr = np.zeros(len(rec['x']), dtype=np.dtype([(k, '<' + t) for k, t in PLY_PROPS]))
    for k, _ in PLY_PROPS:
        arr[k] = rec[k]
    Path(path).write_bytes(head.encode() + arr.tobytes())


def rot_x90():
    c, s = math.cos(90 / 180 * PI_UP), math.sin(90 / 180 * PI_UP)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], np.float64)


def robust_norm(pos, scale=0.6):
    low = np.quantile(pos, 0.01, axis=0); high = np.quantile(pos, 0.99, axis=0)
    s = scale / max(float((high - low).max()), 1e-12)
    return s, (low + high) / 2, high - low


def azimuths(seed, n=3, min_gap=20.0):
    rng = np.random.RandomState(int(seed)); picked = []; tries = 0
    while len(picked) < n and tries < 5000:
        tries += 1; c = float(rng.uniform(0.0, 360.0))
        if all(min(abs(c - a), 360 - abs(c - a)) >= min_gap for a in picked):
            picked.append(c)
    return sorted(a % 360.0 for a in picked)


def local_coord(v):
    v = v / np.linalg.norm(v); h1 = np.array([1., 1., 1.])
    if abs(h1 @ v) < 0.01:
        h1 = np.array([0.72, 0.37, -0.67])
    h1 = h1 - (h1 @ v) * v; h1 /= np.linalg.norm(h1); h2 = np.cross(h1, v)
    return np.column_stack((h1, h2, v))


def cameras(index, cfg, cds, mean, s):
    """World-space cameras as in get_camera_view (synthetic). Returns list of (name, w2c 4x4)."""
    center = mean + (np.array(cfg['mpm_space_viewpoint_center'], float) - 1.0) / s
    up_w = np.array(cfg['mpm_space_vertical_upward_axis'], float) / s
    obs = local_coord(up_w); r = float(cfg['init_radius']) / cds; el = float(cfg['init_elevation'])
    out = []
    for az in azimuths(index):
        can = np.array([math.cos(math.radians(az)) * math.cos(math.radians(el)), math.sin(math.radians(az)) * math.cos(math.radians(el)),
                        math.sin(math.radians(el))]) * r
        # upstream uses az/180*np.pi (np.pi here, not 3.1415926)
        pos = center + obs @ can
        fwd = center - pos; fwd /= np.linalg.norm(fwd); down = -obs[:, 2]
        y = down - (down @ fwd) * fwd; y /= np.linalg.norm(y); x = np.cross(y, fwd)
        c2w = np.eye(4); c2w[:3, :3] = np.column_stack((x, y, fwd)); c2w[:3, 3] = pos
        out.append((f'az{int(round(az))}_el{int(round(el))}', np.linalg.inv(c2w)))
    return out


def project(X, w2c):
    Xc = X @ w2c[:3, :3].T + w2c[:3, 3]; f = RENDER / (2 * math.tan(math.radians(30)))
    u = f * Xc[:, 0] / Xc[:, 2] + RENDER / 2; v = f * Xc[:, 1] / Xc[:, 2] + RENDER / 2
    k = SIZE / RENDER
    return (u + .5) * k - .5, (v + .5) * k - .5, Xc[:, 2]


def splat(u, v, dil):
    import cv2
    m = np.zeros((SIZE, SIZE), np.uint8); ui, vi = np.round(u).astype(int), np.round(v).astype(int)
    ok = (ui >= 0) & (ui < SIZE) & (vi >= 0) & (vi < SIZE); m[vi[ok], ui[ok]] = 1
    if dil:
        m = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dil + 1, 2 * dil + 1)))
    return m.astype(bool)


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def run(index, a):
    import cv2
    from scipy import ndimage
    d = ROOT / f'data/v2_main/{a.split}/{index:06d}'
    gt = json.loads((d / 'gt.json').read_text()); cfg = json.loads((d / 'config.json').read_text()); obj = gt['object']
    pack_path = glob.glob(str(d / '*/sample_pack.npz'))[0]; pk = np.load(pack_path)
    views = json.loads(bytes(pk['meta_json']).decode())['views']
    masks = (pk['object_mask'][:, 0, 0] > 127).copy(); rgb0 = pk['rgb'][:, 0].transpose(0, 2, 3, 1).astype(np.float32) / 255.  # V,H,W,3
    meta_fallback = []
    for vi in range(len(masks)):
        if masks[vi].mean() > 0.5:  # degenerate released object mask (e.g. all-ones, whole frame covered by a bright floater): background subtraction
            im = rgb0[vi]; border = np.concatenate([im[0], im[-1], im[:, 0], im[:, -1]]); bg = np.median(border, 0)
            masks[vi] = np.abs(im - bg).max(-1) > 0.12; meta_fallback.append(vi)  # |RGB - border-median background| > 0.12
    cds = float(gt.get('camera_distance_scale', 0.65))
    # GT asset -> pipeline world (x90, opacity filter) -> normalisation
    g = read_ply(ASSETS / f'{obj}.ply')
    raw = np.stack([g['x'], g['y'], g['z']], 1).astype(np.float64)
    keep = 1 / (1 + np.exp(-g['opacity'].astype(np.float64))) > cfg.get('opacity_threshold', 0.02)
    Rx = rot_x90(); W = raw[keep] @ Rx.T
    s, mean, ext = robust_norm(W, cfg.get('scale', 0.6))
    cams = cameras(index, cfg, cds, mean, s)
    names = [c[0] for c in cams]
    if sorted(names) != list(views):
        print('WARNING view names differ', names, views, flush=True)
    order = {n: i for i, n in enumerate(views)}
    cam_of = {order.get(n, i): w for i, (n, w) in enumerate(sorted(cams, key=lambda c: c[0]))}
    meta = {'index': index, 'object': obj, 'action': gt['action'], 'law': gt['law'], 'views': views, 'camera_names': names,
            'gt_scale_origin': s, 'gt_mean_world': mean.tolist(), 'gt_quantile_extent_world': ext.tolist(), 'vox_mpm': a.vox, 'mask_dilate_px': a.dilate, 'mask_fallback_rgb_views': meta_fallback}
    # (1) camera check: GT gaussian centres
    # rendered GT set = surface gaussians + interior particles of the pipeline's own filling cache (mpm space,
    # includes filling artefacts that are rendered and simulated); fall back to gaussian centres only
    meta['camera_check_iou_gt_asset'] = []; meta['camera_check_precision_gt_asset'] = []
    Wc = W; cache = ASSETS / 'filled' / obj / 'mpm_init_pos.pt'
    if cache.exists():
        import torch
        P = torch.load(cache, map_location='cpu', weights_only=False)['mpm_init_pos'].numpy().astype(np.float64)
        Wc = mean + (P - 1.0) / s; meta['camera_check_points'] = f'filled cache ({len(P)} pts)'
    else:
        meta['camera_check_points'] = 'gaussian centres only'
    for vi in range(len(views)):
        u, v, _ = project(Wc, cam_of[vi]); sp = splat(u, v, 0)
        meta['camera_check_iou_gt_asset'].append(iou(sp, masks[vi]))
        dm1 = cv2.dilate(masks[vi].astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        ui, vv = np.round(u).astype(int), np.round(v).astype(int); ok = (ui >= 0) & (ui < SIZE) & (vv >= 0) & (vv < SIZE)
        meta['camera_check_precision_gt_asset'].append(float(dm1[vv[ok], ui[ok]].mean() * ok.mean()))
    # (2) hull carving in GT-normalised mpm space
    ax = np.arange(0.5, 1.5 + 1e-9, a.vox)
    M = np.stack(np.meshgrid(ax, ax, ax, indexing='ij'), -1).reshape(-1, 3)
    Xw = mean + (M - 1.0) / s
    dm = [cv2.dilate(m.astype(np.uint8), np.ones((2 * a.dilate + 1,) * 2, np.uint8)).astype(bool) if a.dilate else m for m in masks]
    inside = np.ones(len(M), bool)
    for vi in range(len(views)):
        u, v, z = project(Xw, cam_of[vi]); ui, vv = np.round(u).astype(int), np.round(v).astype(int)
        ok = (ui >= 0) & (ui < SIZE) & (vv >= 0) & (vv < SIZE) & (z > 0)
        ins = np.zeros(len(M), bool); ins[ok] = dm[vi][vv[ok], ui[ok]]; inside &= ins
    n = len(ax); occ = inside.reshape(n, n, n)
    lab, nl = ndimage.label(occ, structure=np.ones((3, 3, 3)))
    if nl == 0:
        raise RuntimeError('empty hull')
    sizes = ndimage.sum(occ, lab, range(1, nl + 1)); occ = lab == (1 + int(np.argmax(sizes)))
    solid = occ
    if a.shell:  # surface voxels only; the pipeline's fill_particles fills the interior as for any 3DGS asset
        occ = occ & ~ndimage.binary_erosion(occ, np.ones((3, 3, 3)))
    meta['solid_hull_voxels'] = int(solid.sum()); meta['shell_only'] = bool(a.shell)
    Hm = M[occ.reshape(-1)]; Hw = mean + (Hm - 1.0) / s
    meta['hull_voxels'] = int(len(Hm)); meta['hull_components'] = int(nl)
    # hull re-projection IoU (splat radius ~ voxel footprint)
    meta['hull_reproj_iou'] = []
    for vi in range(len(views)):
        u, v, _ = project(Hw, cam_of[vi]); hp = ndimage.binary_closing(splat(u, v, 0), np.ones((3, 3))) | splat(u, v, 0)
        meta['hull_reproj_iou'].append(iou(hp, masks[vi])); meta.setdefault('hull_reproj_recall', []).append(float((hp & masks[vi]).sum() / max(masks[vi].sum(), 1)))
        meta.setdefault('hull_reproj_precision', []).append(float((hp & masks[vi]).sum() / max(hp.sum(), 1)))
    # GT volume: voxelise normalised GT gaussian centres on the same grid, fill holes (solid volume)
    Gm = (Wc - mean) * s + 1.0  # filled cache when available (solid), else gaussian centres + hole filling
    gi = np.clip(np.round((Gm - 0.5) / a.vox).astype(int), 0, n - 1)
    gocc = np.zeros((n, n, n), bool); gocc[gi[:, 0], gi[:, 1], gi[:, 2]] = True
    gocc = ndimage.binary_fill_holes(ndimage.binary_closing(gocc, np.ones((3, 3, 3)), iterations=2))
    meta['gt_voxels_filled'] = int(gocc.sum()); meta['volume_ratio_hull_over_gt'] = float(solid.sum() / max(gocc.sum(), 1))
    meta['voxel_iou_hull_vs_gt'] = float((solid & gocc).sum() / max((solid | gocc).sum(), 1))
    s_h, mean_h, ext_h = robust_norm(Hw @ np.eye(3), cfg.get('scale', 0.6))
    meta['hull_quantile_extent_mpm_gt_norm'] = (ext_h * s).tolist(); meta['gt_quantile_extent_mpm'] = (ext * s).tolist()
    meta['hull_quantile_centre_mpm_gt_norm'] = ((mean_h - mean) * s + 1.0).tolist()
    meta['extent_ratio_hull_over_gt_max'] = float(ext_h.max() / ext.max())
    meta['scale_ratio_s_hull_over_s_gt'] = float(s_h / s)
    # (4) colours via per-view z-buffer
    col = np.zeros((len(Hw), 3)); cnt = np.zeros(len(Hw))
    for vi in range(len(views)):
        u, v, z = project(Hw, cam_of[vi]); ui, vv = np.clip(np.round(u).astype(int), 0, SIZE - 1), np.clip(np.round(v).astype(int), 0, SIZE - 1)
        pix = vv * SIZE + ui; zb = np.full(SIZE * SIZE, np.inf); np.minimum.at(zb, pix, z)
        vis = (z <= zb[pix] + 2 * a.vox / s) & masks[vi][vv, ui]
        col[vis] += rgb0[vi][vv[vis], ui[vis]]; cnt[vis] += 1
    fallback = np.mean([rgb0[vi][masks[vi]].mean(0) for vi in range(len(views))], 0)
    col = np.where(cnt[:, None] > 0, col / np.maximum(cnt, 1)[:, None], fallback)
    meta['coloured_fraction'] = float((cnt > 0).mean())
    # (3) asset in raw frame (undo x90: W = raw @ Rx.T  ->  raw = W @ Rx)
    Hr = Hw @ Rx
    sc = math.log(0.6 * a.vox / s)
    rec = {'x': Hr[:, 0], 'y': Hr[:, 1], 'z': Hr[:, 2], 'opacity': np.full(len(Hr), math.log(0.99 / 0.01)),
           'scale_0': np.full(len(Hr), sc), 'scale_1': np.full(len(Hr), sc), 'scale_2': np.full(len(Hr), sc),
           'rot_0': np.ones(len(Hr)), 'rot_1': np.zeros(len(Hr)), 'rot_2': np.zeros(len(Hr)), 'rot_3': np.zeros(len(Hr))}
    for c in range(3):
        rec[f'f_dc_{c}'] = (col[:, c] - 0.5) / 0.28209479177387814
    rgbu = np.clip(np.round(col * 255), 0, 255).astype(np.uint8)
    rec['red'], rec['green'], rec['blue'] = rgbu[:, 0], rgbu[:, 1], rgbu[:, 2]
    out = ROOT / f'data/v3k/hull_assets/{index:06d}'; out.mkdir(parents=True, exist_ok=True)
    write_ply(out / f'{obj}.ply', rec)
    # re-read check: the pipeline's own normalisation of the written asset
    rr = read_ply(out / f'{obj}.ply'); Wr = np.stack([rr['x'], rr['y'], rr['z']], 1).astype(np.float64) @ Rx.T
    meta['roundtrip_max_err_world'] = float(np.abs(Wr - Hw).max())
    meta['asset_path'] = str((out / f'{obj}.ply').relative_to(ROOT))
    (out / 'hull_meta.json').write_text(json.dumps(meta, indent=1))
    print(index, obj, gt['action'], 'cam', np.round(meta['camera_check_iou_gt_asset'], 3).tolist(), 'prec', np.round(meta['camera_check_precision_gt_asset'], 3).tolist(), 'hull', np.round(meta['hull_reproj_iou'], 3).tolist(),
          'vol', round(meta['volume_ratio_hull_over_gt'], 2), 'ext', round(meta['extent_ratio_hull_over_gt_max'], 3), 'n', meta['hull_voxels'], flush=True)
    return meta


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--indices', type=int, nargs='+', required=True)
    p.add_argument('--split', default='val'); p.add_argument('--vox', type=float, default=0.006); p.add_argument('--dilate', type=int, default=1)
    p.add_argument('--no-shell', dest='shell', action='store_false')
    a = p.parse_args(); summary = {}
    for i in a.indices:
        try:
            summary[i] = run(i, a)
        except Exception as e:  # noqa
            import traceback; traceback.print_exc(); summary[i] = {'failed': repr(e)}
    Path(ROOT / 'data/v3k/hull_assets').mkdir(parents=True, exist_ok=True)
    prev = ROOT / 'data/v3k/hull_assets/summary.json'
    old = json.loads(prev.read_text()) if prev.exists() else {}
    old.update({str(k): v for k, v in summary.items()}); prev.write_text(json.dumps(old, indent=1))
