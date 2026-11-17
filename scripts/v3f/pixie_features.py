"""V3f PIXIE baseline, stage 1+2: our 3DGS asset -> PIXIE-style 64^3 CLIP voxel grid -> PIXIE released UNets.

Faithful to PIXIE (github.com/vlongle/pixie, read before use):
  * object normalisation as pixie/blender/generate_blendernerf_data.py normalize_scene(): scale 1/max bbox extent,
    bbox centre at origin, Blender world frame (Z up; our assets are Y up -> (x, y, z) -> (x, -z, y), glTF import rule);
  * cameras as BlenderNeRF COS settings in that script: 200 views, radius U(0.6, 0.9) (= camera_dist/2), upper
    hemisphere, looking at origin, focal 20 mm on 36 mm sensor (FOV 84 deg), 512x512, transparent (black) background;
  * CLIP dense features exactly as f3rm/features/clip_extract.py: OpenAI ViT-L/14@336px, MaskCLIP patch encodings
    (model.get_patch_encodings), CLIP preprocess without centre crop (vendored f3rm clip package, imported, not copied);
  * voxel grid as pixie/voxel/voxelize.py: dense_voxel_grid(-0.5..0.5, voxel 1/64, 'ij'), features = alpha * feature;
  * UNets: SegmentationUNet / RegressionUNet wrappers re-declared verbatim from Wavelet-Generation trainer/*.py with
    the released config (config/training/default.yaml), checkpoints loaded strictly; unscaling as
    pixie/voxel/map_pred_to_coords.py unscale_prediction with normalization_stats/normalization_ranges.yaml.
APPROXIMATED (documented deviation, f3rm/nerfstudio/tinycudann not installed in the time box):
  * images are rendered from our 3DGS (DC colour, diff-gaussian-rasterization depth/alpha build) instead of Blender
    Cycles renders of a mesh;
  * the f3rm distilled NeRF feature field is replaced by multi-view back-projection: each occupied voxel gets the mean
    bilinear-sampled patch feature over views where it is visible (depth test); occluded/interior voxels copy the
    nearest visible voxel's feature; alpha = 1 inside the occupancy mask and 0 outside (f3rm: 1-exp(-0.01*density));
  * occupancy = voxels hit by Gaussian centres with opacity > 0.1 (+ 1-voxel dilation), interior filled
    (variant 'filled', primary) or not ('shell', sensitivity).
  usage: python -I pixie_features.py --objects a b ... --out DIR [--views 200]
"""
import argparse, importlib.util, json, math, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

R = Path('/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test')
PIXIE = R / 'external/pixie'
GEOM = R / 'data/geometry/kaggle_v2/3DGS_model/3DGS_model'
CKPT_SEG = R / 'external/pixie_hf/checkpoints_discrete/20250505_013424/epoch_510.pth'
CKPT_CONT = R / 'external/pixie_hf/checkpoints_continuous_mse/20250805_040343/epoch_250.pth'
CLIP_PT = R / 'external/clip_vitl14_336/ViT-L-14-336px.pt'
C0 = 0.28209479177387814
PIXIE_NAMES = {0: 'jelly', 1: 'metal', 2: 'sand', 3: 'visplas', 4: 'fluid', 5: 'snow', 6: 'stationary', 7: 'background'}


def load_ply(path):
    d = Path(path).read_bytes(); i = d.index(b'end_header\n') + 11
    head = d[:i].decode().split('\n'); props = [l.split()[-1] for l in head if l.startswith('property')]
    types = [l.split()[1] for l in head if l.startswith('property')]
    dt = np.dtype([(n, '<f4' if t == 'float' else 'u1') for n, t in zip(props, types)])
    a = np.frombuffer(d[i:], dt)
    xyz = np.stack([a['x'], a['y'], a['z']], 1).astype(np.float32)
    dc = np.stack([a['f_dc_0'], a['f_dc_1'], a['f_dc_2']], 1).astype(np.float32)
    op = 1 / (1 + np.exp(-a['opacity'].astype(np.float32)))
    sc = np.exp(np.stack([a['scale_0'], a['scale_1'], a['scale_2']], 1).astype(np.float32))
    q = np.stack([a['rot_0'], a['rot_1'], a['rot_2'], a['rot_3']], 1).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    return xyz, dc, op, sc, q


def qmul(a, b):
    w1, x1, y1, z1 = a.T; w2, x2, y2, z2 = b.T
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], 1)


def normalise(xyz, sc, q, op):
    # Y-up asset -> Blender Z-up: (x, y, z) -> (x, -z, y) = rotation +90 deg about X
    Rm = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], np.float32)
    xyz = xyz @ Rm.T
    qr = np.array([[math.cos(math.pi / 4), math.sin(math.pi / 4), 0, 0]], np.float32)
    q = qmul(np.repeat(qr, len(q), 0), q)
    solid = op > 0.1
    lo, hi = xyz[solid].min(0), xyz[solid].max(0)
    s = 1.0 / float((hi - lo).max())
    xyz = (xyz - (lo + hi) / 2) * s
    return xyz, sc * s, q, {'scale': s, 'bbox_lo': lo.tolist(), 'bbox_hi': hi.tolist()}


def cameras(n, seed=0):
    rng = np.random.default_rng(seed); out = []
    for _ in range(n):
        r = rng.uniform(0.6, 0.9)
        v = rng.normal(size=3); v[2] = abs(v[2]); v /= np.linalg.norm(v)
        c = v * r
        f = -c / np.linalg.norm(c)                       # camera +z (forward, COLMAP convention)
        up = np.array([0, 0, 1.0]); x = np.cross(f, up)
        if np.linalg.norm(x) < 1e-6:
            x = np.array([1.0, 0, 0])
        x /= np.linalg.norm(x); y = np.cross(f, x)      # y down
        Rwc = np.stack([x, y, f], 1)                      # columns = camera axes in world
        w2c = np.eye(4); w2c[:3, :3] = Rwc.T; w2c[:3, 3] = -Rwc.T @ c
        out.append((w2c.astype(np.float32), c.astype(np.float32)))
    return out


def proj_matrix(tanfov, znear=0.01, zfar=10.):
    P = torch.zeros(4, 4)
    P[0, 0] = 1 / tanfov; P[1, 1] = 1 / tanfov; P[3, 2] = 1.0
    P[2, 2] = zfar / (zfar - znear); P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def render_views(g, cams, H, tanfov, dev):
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    xyz, col, op, sc, q = g
    imgs, depths, alphas = [], [], []
    P = proj_matrix(tanfov).to(dev)
    for w2c, c in cams:
        V = torch.from_numpy(w2c).to(dev)
        view = V.T.contiguous(); full = (P @ V).T.contiguous()
        st = GaussianRasterizationSettings(image_height=H, image_width=H, tanfovx=tanfov, tanfovy=tanfov,
                                           bg=torch.zeros(3, device=dev), scale_modifier=1.0, viewmatrix=view,
                                           projmatrix=full, sh_degree=0, campos=torch.from_numpy(c).to(dev),
                                           prefiltered=False, debug=False)
        ras = GaussianRasterizer(st)
        means2d = torch.zeros_like(xyz)
        with torch.no_grad():
            out = ras(means3D=xyz, means2D=means2d, opacities=op[:, None], colors_precomp=col, scales=sc, rotations=q)
        img, _, depth, alpha = out
        imgs.append(img.clamp(0, 1).cpu()); depths.append(depth.reshape(H, H).cpu()); alphas.append(alpha.reshape(H, H).cpu())
    return torch.stack(imgs), torch.stack(depths), torch.stack(alphas)


def load_clip(dev):
    sys.path.insert(0, str(PIXIE / 'third_party/f3rm'))
    from f3rm.features.clip import clip
    from f3rm.features.clip_extract import CLIPArgs
    from torchvision.transforms import CenterCrop, Compose
    model, pre = clip.load(str(CLIP_PT), device=dev)
    assert CLIPArgs.model_name == 'ViT-L/14@336px' and CLIPArgs.skip_center_crop
    pre = Compose([t for t in pre.transforms if not isinstance(t, CenterCrop)])
    return model.eval(), pre


@torch.no_grad()
def clip_patches(model, pre, imgs, dev):
    from PIL import Image
    x = torch.stack([pre(Image.fromarray((im.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8))) for im in imgs])
    feats = []
    for i in range(0, len(x), 32):
        feats.append(model.get_patch_encodings(x[i:i + 32].to(dev)).float())
    f = torch.cat(feats)
    h = x.shape[-2] // model.visual.patch_size
    return f.reshape(len(x), h, h, -1)  # (n, 24, 24, 768)


def occupancy(xyz, op, D=64):
    from scipy import ndimage
    keep = op > 0.1
    idx = np.floor((xyz[keep] + 0.5) * D).astype(int)
    idx = idx[((idx >= 0) & (idx < D)).all(1)]
    shell = np.zeros((D, D, D), bool); shell[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    shell = ndimage.binary_dilation(shell, iterations=1)
    filled = ndimage.binary_fill_holes(shell)
    return shell, filled


def backproject(feats, cams, depths, alphas, tanfov, H, occ, dev, D=64, tol=2.0 / 64):
    grid = torch.stack(torch.meshgrid(*[torch.arange(-0.5, 0.5, 1 / D)] * 3, indexing='ij'), -1).float()  # dense_voxel_grid
    pts = grid[torch.from_numpy(occ)].to(dev)
    acc = torch.zeros(len(pts), feats.shape[-1], device=dev); cnt = torch.zeros(len(pts), device=dev)
    for k, (w2c, _) in enumerate(cams):
        V = torch.from_numpy(w2c).to(dev)
        pc = pts @ V[:3, :3].T + V[:3, 3]
        z = pc[:, 2]
        u = pc[:, 0] / z / tanfov; v = pc[:, 1] / z / tanfov          # NDC in [-1, 1]
        inside = (z > 0.01) & (u.abs() < 1) & (v.abs() < 1)
        px = ((u + 1) / 2 * H - 0.5).round().long().clamp(0, H - 1); py = ((v + 1) / 2 * H - 0.5).round().long().clamp(0, H - 1)
        dz = depths[k].to(dev)[py, px]; al = alphas[k].to(dev)[py, px]
        dexp = dz / al.clamp(min=1e-6)                                 # rasteriser depth is alpha-weighted
        vis = inside & (al > 0.5) & (z <= dexp + tol)
        if vis.any():
            f = feats[k].permute(2, 0, 1)[None].to(dev)                  # (1, C, h, w)
            g = torch.stack([u[vis], v[vis]], -1)[None, None]
            s = torch.nn.functional.grid_sample(f, g, mode='bilinear', align_corners=False)[0, :, 0].T
            acc[vis] += s; cnt[vis] += 1
    seen = cnt > 0
    out = torch.zeros(len(pts), feats.shape[-1], device=dev)
    out[seen] = acc[seen] / cnt[seen, None]
    if (~seen).any() and seen.any():  # nearest visible voxel (stands in for the NeRF field's interpolation)
        ps, pu = pts[seen], pts[~seen]
        nn_idx = torch.cat([torch.cdist(pu[i:i + 4096], ps).argmin(1) for i in range(0, len(pu), 4096)])
        out[~seen] = out[seen][nn_idx]
    vol = torch.zeros(D, D, D, feats.shape[-1])
    vol[torch.from_numpy(occ)] = out.cpu()
    return vol, {'occupied_voxels': int(occ.sum()), 'directly_seen_fraction': float(seen.float().mean()),
                 'mean_views_per_seen_voxel': float(cnt[seen].mean()) if seen.any() else 0.}


def load_unets(dev):
    import yaml
    sys.path.insert(0, str(PIXIE / 'third_party/Wavelet-Generation'))
    from models.module.diffusion_network import MyUNetModel, FeatureProjector
    cfg = yaml.safe_load((PIXIE / 'config/training/default.yaml').read_text())
    tr = cfg['training']

    class Wrap(nn.Module):  # == SegmentationUNet / RegressionUNet (trainer/training_{discrete,continuous_mse}.py)
        def __init__(self, out_ch):
            super().__init__()
            self.projector = FeatureProjector(768, out_channels=32, hidden_channels=128)
            self.unet = MyUNetModel(in_channels=32, model_channels=tr['unet_model_channels'], out_channels=out_ch,
                                    num_res_blocks=tr['unet_num_res_blocks'], channel_mult=tuple(tr['unet_channel_mult']),
                                    attention_resolutions=tuple(tr['attention_resolutions']), spatial_size=64, dims=3,
                                    activation=nn.LeakyReLU(0.02))

        def forward(self, x):
            return self.unet(self.projector(x))

    nets, info = [], {}
    for path, out_ch in ((CKPT_SEG, cfg['num_material_classes']), (CKPT_CONT, 3)):
        m = Wrap(out_ch)
        ck = torch.load(path, map_location='cpu', weights_only=True)
        sd = next((ck[k] for k in ('model_state_dict', 'state_dict') if isinstance(ck, dict) and k in ck), ck)
        sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
        missing, unexpected = m.load_state_dict(sd, strict=False)
        assert not missing and not unexpected, (path, missing[:5], unexpected[:5])
        info[path.parent.parent.name] = {'epoch': ck.get('epoch') if isinstance(ck, dict) else None, 'strict_load': True}
        nets.append(m.to(dev).eval())
    rng = yaml.safe_load((PIXIE / 'normalization_stats/normalization_ranges.yaml').read_text())
    return nets, rng, info


def unscale(c, rng):
    c = np.clip(c, -1, 1)
    lin = lambda x, lo, hi: (x + 1) * (hi - lo) / 2 + lo
    return (10 ** lin(c[0], rng['density_p1'], rng['density_p99']), 10 ** lin(c[1], rng['E_p1'], rng['E_p99']),
            lin(c[2], rng['nu_p1'], rng['nu_p99']))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--objects', nargs='+', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--views', type=int, default=200); ap.add_argument('--res', type=int, default=512)
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'
    tanfov = 18.0 / 20.0  # sensor 36 mm / 2 over focal 20 mm
    clip_model, pre = load_clip(dev)
    (seg, cont), rng, ckinfo = load_unets(dev)
    cams = cameras(a.views)
    for obj in a.objects:
        res_path = out / f'{obj}.json'
        if res_path.exists():
            print('skip', obj); continue
        t0 = time.time()
        xyz, dc, op, sc, q = load_ply(GEOM / f'{obj}.ply')
        xyz, sc, q, norm = normalise(xyz, sc, q, op)
        col = np.clip(0.5 + C0 * dc, 0, 1)
        g = [torch.from_numpy(np.ascontiguousarray(x)).float().to(dev) for x in (xyz, col, op, sc, q)]
        imgs, depths, alphas = render_views(g, cams, a.res, tanfov, dev)
        from PIL import Image
        Image.fromarray((torch.cat(list(imgs[:6]), 2).permute(1, 2, 0).numpy() * 255).astype(np.uint8)).resize((6 * 128, 128)).save(out / f'{obj}_views.png')
        feats = clip_patches(clip_model, pre, imgs, dev)
        shell, filled = occupancy(xyz, op)
        result = {'object_id': obj, 'normalisation': norm, 'views': a.views, 'resolution': a.res, 'checkpoints': ckinfo,
                  'normalization_ranges': rng, 'variants': {}}
        for variant, occ in (('filled', filled), ('shell', shell)):
            vol, bp = backproject(feats, cams, depths, alphas, tanfov, a.res, occ, dev)
            x = vol.permute(3, 0, 1, 2)[None].to(dev)
            with torch.no_grad():
                logits = seg(x)[0]; c = cont(x)[0]
            cls = logits.argmax(0).cpu().numpy(); m = occ
            c = c.cpu().numpy(); rho, E, nu = unscale(c, rng)
            hist = np.bincount(cls[m], minlength=8)
            fg = hist[:7]
            maj = int(fg.argmax()) if fg.sum() else 7
            prob = torch.softmax(logits, 0)[:, torch.from_numpy(m).to(dev)].mean(1).cpu().numpy()
            result['variants'][variant] = {
                **bp, 'class_hist_mask_voxels': {PIXIE_NAMES[i]: int(h) for i, h in enumerate(hist)},
                'mean_softmax_mask_voxels': {PIXIE_NAMES[i]: float(p) for i, p in enumerate(prob)},
                'majority_class_excl_background': PIXIE_NAMES[maj],
                'median_E_Pa': float(np.median(E[m])), 'median_nu': float(np.median(nu[m])), 'median_rho': float(np.median(rho[m])),
                'median_log10_E': float(np.median(np.log10(E[m]))), 'iqr_log10_E': float(np.subtract(*np.percentile(np.log10(E[m]), [75, 25])))}
            print(obj, variant, result['variants'][variant]['majority_class_excl_background'],
                  '%.3g' % result['variants'][variant]['median_E_Pa'], '%.3f' % result['variants'][variant]['median_nu'],
                  '%.0f' % result['variants'][variant]['median_rho'], bp, flush=True)
            del x, vol
        result['seconds'] = time.time() - t0
        res_path.write_text(json.dumps(result, indent=1))
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
