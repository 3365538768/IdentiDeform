from pathlib import Path
p = Path("scripts/v3x/write_zh.py"); s = p.read_text()
s = s.replace("""        if not row["n_paired_complete"]: continue""", """        if p.endswith('_reached_k') or not row["n_paired_complete"]: continue""")
old = """    sp = S.get("lawmean_gt_sims_to_reach_prior_k0")
    if sp:"""
new = """    L.append("\\n### 配对差异（部分结果：两臂都已达到预算 k 的样本，含未跑完 24 次的搜索；每格 n）\\n\\n| 比较 | k=0 | k=4 | k=12 | k=24 |\\n|---|---|---|---|---|")
    for p, row in S['paired'].items():
        if not p.endswith('_reached_k') or not row: continue
        a, b = p[:-len('_reached_k')].split('_minus_')
        L.append(f"| {NAME[a]} − {NAME[b]} | " + " | ".join((f"{row[f'k{k}']['mean']:+.3f} [{row[f'k{k}']['ci95'][0]:+.3f}, {row[f'k{k}']['ci95'][1]:+.3f}] (n={row[f'k{k}']['n']})" if f'k{k}' in row else "–") for k in (0, 4, 12, 24)) + " |")
    for key, lab in (('lawmean_gt_sims_to_reach_prior_k0', '24 次内'), ('lawmean_gt_sims_to_reach_prior_k0_within12', '只用到 12 次（含部分完成的搜索）')):
      sp = S.get(key)
      if sp:
        hi = f(sp['ci95'][1]) if sp['ci95'][1] else '>%d' % sp['kmax']
        mid = f(sp['sims']) if sp['sims'] else '>%d（删失）' % sp['kmax']
        L.append(f"\\n**{lab}，特权 law-mean-GT 追平先验 k=0（{sp['prior_k0_mae']:.3f}）所需仿真次数：** {mid} [{f(sp['ci95'][0])}, {hi}]，n = {sp['n']}，{sp['objects']} 个物体，bootstrap 删失比例 {sp['frac_censored']:.2f}。")
    if False:"""
assert old in s; s = s.replace(old, new)
p.write_text(s)
