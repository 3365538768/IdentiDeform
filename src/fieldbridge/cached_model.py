"""Field-bridge model on externally cached video-backbone features (e.g. V-JEPA 2).

The backbone runs only in the cache builder (scripts/prepare_vjepa_cache.py); training sees
[B, V, T, dim, h, w] tokens exactly like the DINO caches, so heads, losses and budget are shared
with the DINO runs. model.py stays untouched (feature-cache contract).
"""
from dataclasses import replace
import torch
import torch.utils.checkpoint
from torch import nn
from .model import MaterialFieldModel

CACHED_BACKBONES = {'vjepa2_vitl': (1024, 16), 'vjepa2_vitg': (1408, 16), 'vjepa2_vitl_tail4': (1024, 16)}


class CachedFeatureEncoder(nn.Module):
    def __init__(self, dim, patch_size):
        super().__init__()
        self.dim, self.patch_size = dim, patch_size

    def forward(self, x):
        raise RuntimeError('This backbone is only available through a prepared feature cache')


class CachedFieldModel(MaterialFieldModel):
    def __init__(self, cfg):
        if cfg.backbone not in CACHED_BACKBONES:
            raise ValueError('Unknown cached backbone')
        super().__init__(replace(cfg, backbone='tiny', freeze_backbone=True))
        self.cfg = cfg
        dim, patch = CACHED_BACKBONES[cfg.backbone]
        self.frame_encoder = CachedFeatureEncoder(dim, patch)
        self.project = nn.Conv2d(dim, cfg.width, 1)


class TailFinetuneFieldModel(CachedFieldModel):
    """P3: cache holds the input of V-JEPA block (N-k); the last k blocks + final norm are trained.

    Equivalent to finetuning the last k blocks from RGB (no augmentation), at a fraction of the cost.
    Tail parameters are named tail.* and get train.backbone_lr.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        from transformers import AutoModel
        k = int(cfg.backbone.rsplit('_tail', 1)[1])
        full = AutoModel.from_pretrained(cfg.repo)
        layers = full.encoder.layer
        self.tail = nn.ModuleList(list(layers[len(layers) - k:]))
        self.tail_norm = full.encoder.layernorm
        self.tail_start = len(layers) - k
        del full

    def forward(self, rgb, valid, times, prior, prior_mask, features=None, diagnostic_fields=None):
        if features is None:
            raise ValueError('Tail finetuning needs a block-input feature cache')
        b, v, t, d, h, w = features.shape
        x = features.permute(0, 1, 2, 4, 5, 3).reshape(b * v, t * h * w, d).to(self.tail_norm.weight.dtype)
        for layer in self.tail:
            if self.training:
                x = torch.utils.checkpoint.checkpoint(lambda y, m=layer: m(y, None, None, False)[0], x, use_reentrant=False)
            else:
                x = layer(x, None, None, False)[0]
        x = self.tail_norm(x).reshape(b, v, t, h, w, d).permute(0, 1, 2, 5, 3, 4)
        return super().forward(rgb, valid, times, prior, prior_mask, features=x, diagnostic_fields=diagnostic_fields)
