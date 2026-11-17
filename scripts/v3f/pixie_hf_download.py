"""V3f PIXIE: fetch only the released UNet checkpoints from HF dataset vlongle/pixie into a new dir (argv[1])."""
import sys
from huggingface_hub import snapshot_download
snapshot_download("vlongle/pixie", repo_type="dataset", local_dir=sys.argv[1],
                  allow_patterns=["checkpoints_continuous_mse/*", "checkpoints_discrete/*"], max_workers=4)
print("hf ok")
