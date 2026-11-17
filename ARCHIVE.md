# rsi_real_test archive (2026-11-17)

Full working tree of the FieldBench / IdentiDeform project on a800-2, archived before the directory was deleted.
The curated release is the `main` branch.

- Data: HF datasets `HandsomeHusky/IdentiDeform` (v2main release) and `HandsomeHusky/IdentiDeform-extra`
  (v3/v4/v5 lines, `runs/`, `external/`, other `data/`).
- Prior heads: `HandsomeHusky/IdentiDeform-prior`.
- `upstream/DeformFieldBench` = github 3365538768/DeformFieldBench @131c0b0 + `upstream_patches/DeformFieldBench-131c0b0.diff`.
- Not archived (rebuildable): `.venv`, `.engine-envs`, caches, V-JEPA 2 weights (`facebook/vjepa2-vitl-fpc64-256`).
- Not archived on purpose: paper PDF, reviews, `Codex_README.md`.

Rebuild: check out this branch, download both HF datasets into the repo root (they hold `data/`, `runs/`, `external/`),
clone `upstream/DeformFieldBench` at 131c0b0 and apply the patch, then build the environments.
