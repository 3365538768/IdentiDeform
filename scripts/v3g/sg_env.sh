# v3g Spring-Gaus official baseline env (source from rsi_real_test root).
R=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export FB_CACHE="$R/.cache"
export XDG_CACHE_HOME="$FB_CACHE" HF_HOME="$FB_CACHE/huggingface" TORCH_HOME="$R/weights/torch_cache"
export CUDA_CACHE_PATH="$FB_CACHE/nv" TORCH_EXTENSIONS_DIR="$FB_CACHE/torch_extensions" MPLCONFIGDIR="$FB_CACHE/matplotlib"
export UV_CACHE_DIR="$FB_CACHE/uv" TMPDIR="$FB_CACHE/tmp" UV_PYTHON_INSTALL_DIR="$R/.engine-envs/v3g_pythons"
export CUDA_HOME=/usr/local/cuda-12.6 PATH=/usr/local/cuda-12.6/bin:$PATH
export SG_VENV="$R/.engine-envs/v3g_springgaus"
mkdir -p "$TMPDIR"
export UV=/kairos-yangsen/kairis-vepfs-6t/linrui/tools/uv
export UV_PYTHON_BIN_DIR=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test/.engine-envs/python/bin UV_PYTHON_INSTALL_DIR=/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test/.engine-envs/python
