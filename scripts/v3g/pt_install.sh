#!/bin/bash

cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
source scripts/v3g/pt_env.sh; source ../env.sh
PY=$PT_VENV/bin/python
for i in 1 2 3; do $UV pip install --python $PY torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121 && break; sleep 20; done
for i in 1 2 3; do $UV pip install --python $PY "numpy==1.26.4" "warp-lang==1.7.0" usd-core matplotlib "pyglet<2" open3d trimesh rtree pyrender stannum termcolor fvcore wandb moviepy imageio opencv-python-headless cma scipy scikit-learn plyfile kornia tqdm pyyaml "gsplat==1.4.0" && break; sleep 20; done
for i in 1 2 3; do $UV pip install --python $PY --no-index pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html && break; sleep 20; done
$PY -c "import torch,warp,open3d,pytorch3d;print('OK',torch.__version__,warp.__version__,torch.cuda.is_available())"
echo INSTALL_DONE
