import pathlib,json,time
R=pathlib.Path(__file__).resolve().parents[1]
a=json.loads((R/'docs/r2b/metal_probe_dt25us.json').read_text());b=json.loads((R/'docs/r2b/metal_probe_dt12_5us.json').read_text())
rows=[]
for action in a['actions']:
 x,y=a['actions'][action][0],b['actions'][action][0]
 rows.append({'action':action,'height_change_px_dt25':x['height_change_px'],'height_change_px_dt12_5':y['height_change_px'],'relative_height_change_difference':abs(x['height_change_px']-y['height_change_px'])/max(abs(y['height_change_px']),1e-8),'frequency_dt25':x['dominant_height_hz'],'frequency_dt12_5':y['dominant_height_hz']})
report={'updated_unix':time.time(),'status':'complete','scope':'Highest-stiffness cube; summary response comparison only, not full trajectory or physical validation. Frame integration uses rounded substeps; nominal sampled times differ slightly.','rows':rows,'requires_review':True}
(R/'docs/r2b/CONVERGENCE_SUMMARY.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
