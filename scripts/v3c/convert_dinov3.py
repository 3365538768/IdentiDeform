"""V3c: convert the downloaded HF DINOv3 ViT-L/16 weights to the upstream hub format (CPU, fp32 parity).

Reuses restore_original() from scripts/prepare_dinov3_hf.py (its post-download step had failed only
because it was launched with system python). No network access.
  .venv/bin/python scripts/v3c/convert_dinov3.py --status docs/v3c/DINOV3_CONVERT.json
"""
import argparse, hashlib, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from safetensors.torch import load_file
from transformers import AutoModel
from prepare_dinov3_hf import restore_original, UPSTREAM, REVISION


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def main():
    p = argparse.ArgumentParser(); p.add_argument('--status', required=True); a = p.parse_args()
    started = time.time(); torch.set_float32_matmul_precision('highest')
    import subprocess
    commit = subprocess.check_output(['git', '-C', 'external/dinov3', 'rev-parse', 'HEAD'], text=True).strip()
    directory = Path('weights/dinov3_hf')
    original = torch.hub.load('external/dinov3', 'dinov3_vitl16', source='local', pretrained=False).eval()
    original.load_state_dict(restore_original(load_file(str(directory / 'model.safetensors')), original.state_dict()), strict=True)
    hf = AutoModel.from_pretrained(directory, local_files_only=True, attn_implementation='eager').eval()
    torch.manual_seed(17); x = torch.randn(2, 3, 224, 224)
    with torch.inference_mode():
        a_ = original.forward_features(x)['x_norm_patchtokens']; b_ = hf(x).last_hidden_state[:, 5:]
    maximum = float((a_ - b_).abs().max()); mean = float((a_ - b_).abs().mean())
    torch.testing.assert_close(a_, b_, atol=2e-4, rtol=2e-4)
    tmp = Path('weights/dinov3_vitl16_import.part'); torch.save(original.state_dict(), tmp)
    checksum = digest(tmp); target = Path('weights/dinov3_vitl16_pretrain_lvd1689m-' + checksum[:8] + '.pth'); tmp.replace(target)
    out = dict(status='complete', revision=REVISION, upstream_commit=commit, upstream_expected=UPSTREAM,
               source_sha256=digest(directory / 'model.safetensors'), weights=str(target), weights_sha256=checksum,
               strict_load=True, device='cpu_fp32', fp32_patch_max_delta=maximum, fp32_patch_mean_delta=mean,
               parity_images=2, feature_shape=list(a_.shape), elapsed_seconds=time.time() - started)
    Path(a.status).parent.mkdir(parents=True, exist_ok=True); Path(a.status).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
