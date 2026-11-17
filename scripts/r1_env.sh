# Source before R-line jobs on A800-2 (driver 535 needs the CUDA 12.6 compat libcuda first).
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/compat${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export TORCH_HOME="$PWD/weights/torch_cache"
export FIELDBRIDGE_PATH_MAP="$PWD/configs/runtime/path-map.json"
# All caches/temp files stay inside the project (user rule 2026-11-11: nothing under /root or outside rsi_real_test).
export FB_CACHE="$PWD/.cache"
export XDG_CACHE_HOME="$FB_CACHE" HF_HOME="$FB_CACHE/huggingface" TRITON_CACHE_DIR="$FB_CACHE/triton"
export WARP_CACHE_PATH="$FB_CACHE/warp" TI_OFFLINE_CACHE_FILE_PATH="$FB_CACHE/taichi" CUDA_CACHE_PATH="$FB_CACHE/nv"
export TORCH_EXTENSIONS_DIR="$FB_CACHE/torch_extensions" MPLCONFIGDIR="$FB_CACHE/matplotlib" UV_CACHE_DIR="$FB_CACHE/uv"
export TMPDIR="$FB_CACHE/tmp"
