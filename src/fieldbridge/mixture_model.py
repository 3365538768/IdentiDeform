"""C2 (v3b): soft constitutive-law mixture head on cached video features (opt-in architecture).

Selected only by model.architecture == 'legacy_cached_mixture'; all older configs are unchanged and
model.py is untouched (feature-cache contract). The trunk, field heads and feedback route are those
of CachedFieldModel, so B/C/D style controls stay matched; only the parameter head differs.

Head: per law k (jelly/metal/plasticine) a mean and log-variance over the 4 'specific' targets;
the existing law_head gives the posterior p(law | video). Deployable point prediction is the
posterior-weighted mixture mean (no GT law needed). The MAP-law component and the per-law components
are exported (mixture_logits/means/logvars) so the MAP variant can be scored post hoc.
In contrast to the hard hierarchical head (hier_model.py), a wrong argmax law does not select a
wrong regressor outright, and the regression loss back-propagates into the posterior.

Training terms (mixture_material_terms), weighted exactly like the legacy head:
  parameter = SmoothL1(mixture mean, y) + SmoothL1(GT-law component, y)   (specialization)
  nll       = exact diagonal-mixture NLL, normalised per observed parameter
  law       = cross-entropy of p(law)
"""
import torch
from torch import nn
from torch.nn import functional as F
from .cached_model import CachedFieldModel
from .direct_video import mixture_log_prob
from .losses import masked_mean

LAWS = 3


class LawMixtureHead(nn.Module):
    def __init__(self, d, law_head):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, LAWS * 8))
        self.law_head = [law_head]  # shared reference; registered once on the parent model
        self.last = None

    def forward(self, joint):
        components = self.trunk(joint).reshape(len(joint), LAWS, 8).float()
        means, logvars = components.chunk(2, -1)
        logvars = logvars.clamp(-6, 4)
        weights = self.law_head[0](joint).float().softmax(-1)[..., None]
        mean = (weights * means).sum(1)
        variance = (weights * (logvars.exp() + (means - mean[:, None]).square())).sum(1)
        self.last = (means, logvars)
        # The parent forward applies .chunk(2) and clamps the log-variance.
        return torch.cat((mean, variance.clamp_min(1e-6).log()), -1)


class LawMixtureFieldModel(CachedFieldModel):
    def __init__(self, cfg):
        super().__init__(cfg)
        if cfg.use_prior:
            raise ValueError('Law-mixture head does not support static priors')
        self.param_head = LawMixtureHead(cfg.width, self.law_head)

    def forward(self, *args, **kwargs):
        output = super().forward(*args, **kwargs)
        means, logvars = self.param_head.last
        self.param_head.last = None
        output['mixture_logits'] = output['law_logits']
        output['mixture_means'], output['mixture_logvars'] = means, logvars
        return output


def mixture_material_terms(output, batch, cfg):
    y, m = batch['target'], batch['target_mask']
    mean = output['mean'].float()
    means, logvars = output['mixture_means'].float(), output['mixture_logvars'].float()
    logits = output['law_logits'].float()
    law_index = batch['law']
    known = law_index >= 0
    rows = torch.arange(len(y), device=y.device)
    component = means[rows, law_index.clamp_min(0)]
    parameter = masked_mean(F.smooth_l1_loss(mean, y, reduction='none'), m)
    parameter = parameter + masked_mean(F.smooth_l1_loss(component, y, reduction='none'), m & known[:, None])
    count = m.sum(-1).clamp_min(1)
    nll = masked_mean(-mixture_log_prob(logits, means, logvars, y, m) / count, m.any(-1))
    law = masked_mean(F.cross_entropy(logits, law_index.clamp_min(0), reduction='none'), known)
    return parameter, nll, law
