"""Additive, opt-in wiring of architecture='legacy_cached_mixture' (v3b C2)."""
from pathlib import Path
import shutil
R = Path('src/fieldbridge')
B = Path('docs/v3b/source_backup'); B.mkdir(parents=True, exist_ok=True)
edits = {
 'config.py': [('"legacy", "legacy_hier", "legacy_cached", ', '"legacy", "legacy_hier", "legacy_cached", "legacy_cached_mixture", ')],
 'local_model.py': [("    if cfg.architecture == 'legacy_hier':\n",
   "    if cfg.architecture == 'legacy_cached_mixture':  # v3b C2, opt-in\n        from .mixture_model import LawMixtureFieldModel\n        return LawMixtureFieldModel(cfg)\n    if cfg.architecture == 'legacy_hier':\n")],
 'losses.py': [("    # No empirical stress-speed/contact correlation is treated as a physical law.\n",
   "    if cfg.model.architecture == 'legacy_cached_mixture':  # v3b C2 soft law mixture, opt-in\n        from .mixture_model import mixture_material_terms\n        parameter, nll, law = mixture_material_terms(output, batch, cfg)\n    # No empirical stress-speed/contact correlation is treated as a physical law.\n")],
}
for name, reps in edits.items():
    p = R / name; text = p.read_text()
    if 'legacy_cached_mixture' in text:
        print('already', name); continue
    shutil.copyfile(p, B / (name + '.before-v3b'))
    for old, new in reps:
        assert text.count(old) == 1, (name, old)
        text = text.replace(old, new)
    tmp = p.with_suffix('.py.v3btmp'); tmp.write_text(text); compile(text, str(p), 'exec'); tmp.replace(p)
    print('patched', name)
