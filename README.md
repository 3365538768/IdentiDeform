# IdentiDeform

**What Can a Video Tell About Material? Identifiable Video Priors for Few-Simulation System Identification**

Project page: https://3365538768.github.io/IdentiDeform/  ·  Paper: coming soon  ·  Model: [HandsomeHusky/IdentiDeform-prior](https://huggingface.co/HandsomeHusky/IdentiDeform-prior)  ·  Dataset: [HandsomeHusky/IdentiDeform](https://huggingface.co/datasets/HandsomeHusky/IdentiDeform)

Multi-view video of a deformable object identifies its constitutive class and the density-normalized quantities
E/ρ and σ<sub>y</sub>/ρ. It does not identify density itself. IdentiDeform is a simulated benchmark built around this
identifiable subspace. This repository contains:

1. **Data generation** for the benchmark: MPM simulation of 3D-Gaussian objects under six actions (press, drop, shear,
   stretch, bend, press–hold–release), rendered from three cameras.
2. **The video prior.** A frozen V-JEPA 2 video encoder with a light head predicts the material class, E/ρ, ν, ρ and
   σ<sub>y</sub>/ρ from three videos.
3. **Few-simulation refinement.** Gradient-free simulation-in-the-loop fitting starts from the prior and stops after
   0–24 simulations.

> Trained prior heads: https://huggingface.co/HandsomeHusky/IdentiDeform-prior. Dataset (simulations, manifests,
> cached V-JEPA 2 features, object assets): https://huggingface.co/datasets/HandsomeHusky/IdentiDeform.
> This repository contains code and configurations only.

## Repository layout

```
src/fieldbridge/          model, data loading, training and evaluation (Python package)
configs/v2main/           training configs: models B/C/D x seeds 17/29/43, and the feature-cache config
configs/sim/              renderer/camera settings and the object-level train/val/test split
scripts/v2/               benchmark data generation (calls the DeformFieldBench simulator)
scripts/prepare_vjepa_cache.py   frozen V-JEPA 2 feature cache
refine/                   prior ensemble + simulation-in-the-loop refinement
docs/                     project page (GitHub Pages)
```

Model variants (same backbone and head size; see `configs/v2main/`):
**B** predicts parameters only. **C** adds auxiliary physical-field targets (`field_weight` 0.1). **D** also feeds the
predicted fields back into the parameter head. The prior used for refinement is the 9-run B/C/D ensemble.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[video,dinov3]"         # torch 2.7/2.8, transformers, opencv
```

The simulator and renderer come from **DeformFieldBench** (MPM + 3D Gaussian Splatting). It is *not* included
here. Clone it to `upstream/DeformFieldBench` and build its environment by following its README:

```bash
git clone https://github.com/3365538768/DeformFieldBench upstream/DeformFieldBench
export GSIM_PYTHON=/path/to/deformfieldbench/env/bin/python   # interpreter with warp, taichi, gaussian rasterizer
```

Place the 3D-Gaussian object assets (`<object>.ply`) under `data/geometry/kaggle_v2/3DGS_model/3DGS_model/`. The
assets are in the dataset under `data/geometry/kaggle_v2/`. Download the V-JEPA 2 ViT-L weights
(`facebook/vjepa2-vitl-fpc64-256`) to `weights/hf/facebook__vjepa2-vitl-fpc64-256/`.

All paths in the configs are relative to the repository root.

## 0. Download the data

```bash
huggingface-cli download HandsomeHusky/IdentiDeform --repo-type dataset --local-dir .
```

This puts `data/v2_main/` (2538 simulations), `data/manifests/` (train/val and train/val/test manifests with
statistics), `data/prepared/v2main_vjl/` (frozen V-JEPA 2 features for train and val, ~120 GB) and
`data/geometry/kaggle_v2/` (object assets) at the repository root, where the configs expect them. With the cached
features you can skip sections 1 and the feature-cache step of section 2.

The full research working tree (all experiment lines, logs and scripts beyond this release) is archived on the
branch [`archive/rsi_real_test-20261117`](https://github.com/3365538768/IdentiDeform/tree/archive/rsi_real_test-20261117),
with its data in [HandsomeHusky/IdentiDeform-extra](https://huggingface.co/datasets/HandsomeHusky/IdentiDeform-extra).

## 1. Generate the benchmark

```bash
# one shard per GPU; each sample = 4 s, 128 frames x 3 views at 224 px, packed into sample_pack.npz
python scripts/v2/gen_v2_main.py --split train --count 2000 --shard 0 --num-shards 8
python scripts/v2/gen_v2_main.py --split val   --count 300
python scripts/v2/gen_v2_main.py --split test  --count 300
# manifest + train statistics (data/manifests/v2main.jsonl, *.stats.json, *.specific.stats.json)
python scripts/v2/build_v2_manifest.py --root data/v2_main --name v2main
```

Parameters are sampled in identifiable form. log<sub>10</sub>(E/(ρ/1000)) and log<sub>10</sub>(σ<sub>y</sub>/(ρ/1000)) are
log-uniform per class, and ρ is drawn independently (`scripts/v2/gen_v2.py`). Objects are split by identity
(`configs/sim/object_splits.json`).

## 2. Train the video prior

```bash
# frozen V-JEPA 2 features (run sharded on several GPUs, then once unsharded to merge)
python scripts/prepare_vjepa_cache.py --config configs/v2main/v2main_cache.json --splits train,val,test \
       --shard 0 --num-shards 4 --status-output outputs/cache_status_0.json
python scripts/prepare_vjepa_cache.py --config configs/v2main/v2main_cache.json --splits train,val,test \
       --status-output outputs/cache_status.json
# heads: B / C / D, seeds 17 / 29 / 43
python -m fieldbridge.train --config configs/v2main/v2main_C_seed29.json
```

## 3. Evaluate

```bash
python -m fieldbridge.cli evaluate --config configs/v2main/v2main_C_seed29.json \
       --checkpoint runs/v2main_C_seed29/best_material.pt --split test --output runs/v2main_C_seed29/test_once
```

This writes `predictions.jsonl` and metrics (standardized MAE over the four targets, class accuracy, within-class
correlation).

## 4. Prior + few simulations (refinement)

```bash
python refine/make_ensemble.py        # 9-run B/C/D ensemble on val -> outputs/ensemble_val.json
python refine/c1_refine.py --run v2main_C_seed29 --split val --index 12 --arm prior_top1 \
       --budget 24 --replay-budgets 0,4,12,24 --ensemble outputs/ensemble_val.json
```

Arms: `prior_top1` (ensemble prediction, top-1 class; ours), `prior_top2` (also probes the second class),
`lawmean_gt` (privileged: train-set mean of the true class), `lawmean_best3` (simulate all three class means and keep
the best). The objective is (1 − mean object-mask IoU) + RGB-MAE/255 against the observed video. The search is a
deterministic compass search over log<sub>10</sub>(E/ρ) and log<sub>10</sub>(σ<sub>y</sub>/ρ). Each run writes a
resumable `state.json` that contains every evaluated simulation and the estimates after 0/4/12/24 simulations,
including unseen-action replays.

## License

MIT (see `LICENSE`). The simulator (DeformFieldBench), V-JEPA 2 and the object assets keep their own licenses.

## Citation

```bibtex
@inproceedings{identideform2026,
  title     = {What Can a Video Tell About Material? Identifiable Video Priors
               for Few-Simulation System Identification},
  author    = {Anonymous},
  booktitle = {Under review},
  year      = {2026}
}
```

## GitHub Pages

The project page is the static site in `docs/`. To publish it, go to **Settings → Pages → Build and deployment →
Source: Deploy from a branch → Branch: `main`, folder: `/docs`**.
