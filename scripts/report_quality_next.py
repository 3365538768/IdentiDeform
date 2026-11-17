import pathlib,json,time,numpy as np
R=pathlib.Path(__file__).resolve().parents[1]
rows=[]
for name in ['25us','12us','6us']:
 p=R/f'docs/quality_v2/drop_{name}.json'
 if p.exists():
  d=json.loads(p.read_text());rows.append({'step':name,**d['actions']['drop'][0]})
comparisons=[]
for a,b in zip(rows,rows[1:]):comparisons.append({'steps':[a['step'],b['step']],'height_relative_difference':abs(a['height_change_px']-b['height_change_px'])/max(abs(b['height_change_px']),1e-8)})
cameras={}
for base in sorted((R/'data/v2_quality').glob('camera_*')):
 samples=[]
 for p in sorted(base.glob('*/*/*/sample_pack.npz')):
  with np.load(p) as z:m=z['object_mask'][:,:,0]>127
  area=m.sum((-1,-2));border=m[...,0,:].any(-1)|m[...,-1,:].any(-1)|m[...,:,0].any(-1)|m[...,:,-1].any(-1)
  samples.append({'source':str(p.relative_to(R)),'empty_fraction':float((area==0).mean()),'border_fraction':float(border.mean()),'p10_area_fraction':float(np.quantile(area,.1)/(m.shape[-1]*m.shape[-2]))})
 cameras[base.name]=samples
report={'updated_unix':time.time(),'numerical':rows,'comparisons':comparisons,'camera_samples':cameras,'scope':'Highest-stiffness metal cube and limited camera pilot; not full dataset certification.','bulk_generation_allowed':False,'remaining_gates':['full-trajectory convergence and grid refinement','all material families and actions','view coverage with sufficient pixel footprint','visible nonrigid motion','force/field alignment','object-level held-out audit']}
p=R/'docs/quality_v2/QUALITY_REPORT.json';p.write_text(json.dumps(report,indent=2))
notes=R/'docs/notes_r.json';n=json.loads(notes.read_text());n.append('质量报告已更新：docs/quality_v2/QUALITY_REPORT.json。完成数值摘要及配对相机指标，仍禁止直接扩量，需全轨迹/网格/多材质验收。');notes.write_text(json.dumps(n,ensure_ascii=False,indent=2))
print(json.dumps(report),flush=True)
