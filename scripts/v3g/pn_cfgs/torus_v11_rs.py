# v3g (dynamic-only rerun) override of the official PAC-NeRF config (only paths, camera count, taichi memory pool changed)
_base_ = "/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test/external/pacnerf/configs/torus.py"
cfg = dict(
    n_cameras=11,
    data_dir="/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test/runs/v3g/pacnerf/torus_v11/data_in",
    base_dir="/kairos-yangsen/kairis-vepfs-6t/linrui/rsi_real_test/runs/v3g/pacnerf/torus_v11_rs/ckpt",
    taichi_cuda_memory=0.12,
)
