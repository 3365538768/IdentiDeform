"""V3x: 30 MORE test samples by continuing scripts/v3d/c1_select_test.py (seed 20261113, same RNG stream).
Reproduces the original draws exactly (asserted against docs/v3d/C1_TEST_SAMPLES.json), then continues:
continuation order = remaining 2nd picks extra[12:] (original shuffled order), then round r=3,4,... where each
round takes pool[r-1] of every (law, action) cell and shuffles them with the same rng. Coverage rule: every
one of the 10 test objects must appear in the NEW 30 (so the new 30 is a self-contained replication and the 60
cover all objects): objects absent from the natural first 30 of the continuation (or from the original 30) get
their first sample in continuation order first; the rest is filled in continuation order. Uses only gt.json law/action/object labels."""
import json, random
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
PER_CELL, TOTAL, SEED, NEW = 1, 30, 20261113, 30
rows = [json.loads(l) for l in (ROOT / "data/manifests/v2main_withtest.jsonl").read_text().splitlines() if l.strip()]
test = sorted(r["sample_id"] for r in rows if r["split"] == "test")
assert len(test) == 300
cells, objof = {}, {}
for sid in test:
    i = int(sid.rsplit("/", 1)[1]); g = json.loads((ROOT / f"data/v2_main/test/{i:06d}/gt.json").read_text())
    cells.setdefault((g["law"], g["action"]), []).append(i); objof[i] = g["object"]
rng = random.Random(SEED)
chosen, extra, pools = [], [], {}
for key in sorted(cells):
    pool = sorted(cells[key]); rng.shuffle(pool); pools[key] = pool
    chosen += [(key, i) for i in pool[:PER_CELL]]; extra += [(key, i) for i in pool[PER_CELL:PER_CELL + 1]]
rng.shuffle(extra)
chosen += extra[:TOTAL - len(chosen)]
orig = sorted(i for _, i in chosen)
ref = [s["index"] for s in json.loads((ROOT / "docs/v3d/C1_TEST_SAMPLES.json").read_text())["samples"]]
assert orig == sorted(ref), (orig, ref)
cont = list(extra[TOTAL - 18:])
r = 2
while any(len(p) > r for p in pools.values()):
    rnd = [(k, pools[k][r]) for k in sorted(pools) if len(pools[k]) > r]; rng.shuffle(rnd); cont += rnd; r += 1
used = set(orig); cont = [c for c in cont if c[1] not in used]
objs_orig = {objof[i] for i in orig}; nat = {objof[i] for _, i in cont[:NEW]}
missing = sorted(set(objof.values()) - (objs_orig & nat))
new = []
for o in missing:
    new.append(next(c for c in cont if objof[c[1]] == o))
for c in cont:
    if len(new) >= NEW: break
    if c not in new: new.append(c)
order = [i for _, i in new]
out = {"split": "test", "seed": SEED, "rule": __doc__, "original_30": orig, "objects_original": sorted(objs_orig),
       "objects_missing_in_original": missing, "objects_all": sorted(set(objof.values())),
       "new_order": order,
       "samples": [{"index": i, "law": l, "action": a, "object": objof[i]} for (l, a), i in new]}
(ROOT / "docs/v3x/SAMPLES.json").write_text(json.dumps(out, indent=1))
import collections
print(order); print("missing", missing)
print(collections.Counter(objof[i] for i in order)); print(collections.Counter(l for (l,a),i in new)); print(collections.Counter(objof[i] for i in orig))
