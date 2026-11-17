#!/bin/bash
cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
source ../env.sh >/dev/null 2>&1; source scripts/v3g/sg_env.sh
P=$SG_VENV/bin/python
for i in 1 2 3 4; do
  $UV pip install -p $P torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121 && break; sleep 20; done
for i in 1 2 3 4; do
  $UV pip install -p $P opencv-python==4.8.1.78 open3d==0.17.0 imageio==2.26.0 termcolor==2.1.0 gitpython==3.1.41 pyyaml==6.0 yacs==0.1.8 yapf==0.32.0 yarl==1.8.2 tqdm==4.66.1 plyfile==0.8.1 trimesh==4.0.5 "numpy<2" scipy imageio-ffmpeg setuptools wheel ninja fvcore iopath && break; sleep 20; done
for i in 1 2 3 4; do
  $UV pip install -p $P pytorch3d --no-index -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt241/download.html && break; sleep 20; done
$P -c "import torch,pytorch3d,open3d;print(torch.__version__,pytorch3d.__version__,open3d.__version__)"
echo INSTALL_DONE
