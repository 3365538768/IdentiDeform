# v3g PAC-NeRF env. Usage: source scripts/v3g/pn_env.sh (from rsi_real_test root)
source /kairos-yangsen/kairis-vepfs-6t/linrui/env.sh >/dev/null 2>&1
R=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
unset VIRTUAL_ENV PYTHONPATH
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export FB_CACHE="$R/.cache"
export XDG_CACHE_HOME="$FB_CACHE" HF_HOME="$FB_CACHE/huggingface" TRITON_CACHE_DIR="$FB_CACHE/triton"
export TI_OFFLINE_CACHE_FILE_PATH="$FB_CACHE/taichi" CUDA_CACHE_PATH="$FB_CACHE/nv" MPLCONFIGDIR="$FB_CACHE/matplotlib"
export TORCH_EXTENSIONS_DIR="$FB_CACHE/v3g_torch_ext" UV_CACHE_DIR="$FB_CACHE/uv" TORCH_HOME="$R/weights/torch_cache"
export UV_PYTHON_INSTALL_DIR="$R/.engine-envs/python" TMPDIR="$FB_CACHE/tmp"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH
export TI_ENABLE_PRINT_BACKTRACE=0
mkdir -p "$TMPDIR" "$TORCH_EXTENSIONS_DIR"
PNV=$R/.engine-envs/pacnerf
export UV_PYTHON_BIN_DIR="$R/.engine-envs/python/bin" UV_NO_MODIFY_PATH=1
export PATH="$PNV/bin:$PATH"
