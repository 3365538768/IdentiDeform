"""Law-conditioned (hierarchical) parameter head on the legacy field-bridge model.

The legacy head regresses one parameter vector for all constitutive laws, so most of its E/nu
skill comes from separating law means (docs/WITHIN_LAW_AUDIT.json). Here the head emits one
mean/log-variance vector per law. Training supervises the ground-truth law's component; the
deployable prediction uses the argmax predicted law. Feedback controls are unchanged, so
B/C/D comparisons stay matched. model.py is untouched (feature-cache contract).
"""
import torch
from torch import nn
from .model import MaterialFieldModel

LAWS = 3


class LawConditionedHead(nn.Module):
    def __init__(self, d, law_head):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, LAWS * 8))
        self.law_head = [law_head]  # shared reference, not a second registered module
        self.last = None

    def forward(self, joint):
        components = self.trunk(joint).reshape(len(joint), LAWS, 8)
        self.last = components
        chosen = self.law_head[0](joint).detach().argmax(-1)
        return components[torch.arange(len(joint), device=joint.device), chosen]


class HierarchicalFieldModel(MaterialFieldModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.use_prior:
            raise ValueError('Law-conditioned head does not support static priors yet')
        self.param_head = LawConditionedHead(cfg.width, self.law_head)

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        mean, logvar = self.param_head.last.chunk(2, -1)
        output['law_means'], output['law_logvars'] = mean, logvar.clamp(-6, 4)
        return output
