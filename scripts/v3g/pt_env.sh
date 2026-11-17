# v3g PhysTwin env: everything stays inside rsi_real_test; CUDA 12.6 compat libcuda first (driver 535).
R=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export FB_CACHE="$R/.cache"
export XDG_CACHE_HOME="$FB_CACHE" HF_HOME="$FB_CACHE/huggingface" TRITON_CACHE_DIR="$FB_CACHE/triton"
export WARP_CACHE_PATH="$FB_CACHE/warp_v3g_pt" CUDA_CACHE_PATH="$FB_CACHE/nv" TORCH_HOME="$FB_CACHE/torch_v3g_pt"
export TORCH_EXTENSIONS_DIR="$FB_CACHE/torch_extensions" MPLCONFIGDIR="$FB_CACHE/matplotlib" UV_CACHE_DIR="$FB_CACHE/uv"
export TMPDIR="$FB_CACHE/tmp" WANDB_MODE=disabled WANDB_DIR="$FB_CACHE/wandb_v3g_pt" WANDB_SILENT=true
mkdir -p "$TMPDIR" "$WANDB_DIR"
export PT_VENV="$R/.engine-envs/v3g_phystwin"
export UV_PYTHON_BIN_DIR="$R/.engine-envs/python/bin" UV_PYTHON_INSTALL_DIR="$R/.engine-envs/python" UV=/kairos-yangsen/kairis-vepfs-6t/linrui/tools/uv
