#!/bin/bash
cd /kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
source ../env.sh >/dev/null 2>&1; source scripts/v3g/sg_env.sh
export TORCH_CUDA_ARCH_LIST="8.0" MAX_JOBS=8
P=$SG_VENV/bin/python
for m in simple-knn; do
  $UV pip install -p $P --no-build-isolation external/v3g_springgaus/submodules/$m || echo BUILD_FAIL $m
done
cd external/v3g_springgaus && $P -c "import diff_gaussian_rasterization, simple_knn._C, pytorch3d.ops, open3d; print(\"imports ok\")"
echo BUILD_DONE
