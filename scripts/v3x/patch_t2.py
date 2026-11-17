from pathlib import Path
p = Path("scripts/v3x/report.py"); s = p.read_text()
old = """        R["paired"] = P
"""
new = """        R["paired"] = P
        # Table-2 style rows: top1/best3 on the paired-complete set; GT / top2 on their own complete sets
        Pset = [i for i in idx if done(i, "prior_top1") and done(i, "lawmean_best3")]
        T2 = {"paired_set": {"n": len(Pset), "objects": len({obj[i] for i in Pset})}}
        for a, sel in (("prior_top1", Pset), ("lawmean_best3", Pset),
                       ("lawmean_gt", [i for i in idx if done(i, "lawmean_gt")]), ("prior_top2", [i for i in idx if done(i, "prior_top2")])):
            if not sel: continue
            T2[a] = {"n": len(sel), "objects": len({obj[i] for i in sel}),
                     **{f"k{k}": float(np.mean([v for i in sel for v in E(i, a, k)])) for k in KS + [3] if not (a == "lawmean_best3" and k == 0)}}
        R["table2"] = T2
"""
assert old in s; s = s.replace(old, new); p.write_text(s)
