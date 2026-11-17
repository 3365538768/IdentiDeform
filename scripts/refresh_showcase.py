import pathlib,json,time,os
R=pathlib.Path(__file__).resolve().parents[1];D=R/'docs/rebuttal'
def read(p,default):
    try:return json.loads((R/p).read_text())
    except (OSError,ValueError):return default
def update():
    s=read('docs/rq_state.json',{'jobs':{}})
    d={'updated_unix':time.time(),'queue_updated_unix':s.get('updated_unix'),'paused':s.get('paused'),'jobs':s.get('jobs',{}),'results':read('docs/p2/P2_TEST_RESULTS.json',{}),'audit':read('docs/b1/B1_AUDITED_RESULTS.json',{}),'media':read('docs/rebuttal/media_manifest.json',{}),'convergence':read('docs/r2b/CONVERGENCE_SUMMARY.json',{}),'plan':[
    {'stage':'Rebuttal evidence','state':'in_progress','detail':'Main test table available; field gains inconclusive. Still requires regularizer controls, replay validation and real-world evidence.'},
    {'stage':'Numerical repair','state':s.get('jobs',{}).get('r2b_metal_E_dt25us',{}).get('status','pending'),'detail':'25 us probe passed; full sweep followed by 12.5 us highest-stiffness convergence probe.'},
    {'stage':'Data V2 quality','state':'blocked_quality_gate','detail':'Stretch: 8.1% empty frames, 45.0% touching border. Camera coverage and object-level uncertainty must be audited before scaling.'},
    {'stage':'Cross-engine / real video','state':'planned','detail':'V2 research track; no validated transfer claim yet.'}]}
    t=D/'live.json.tmp';t.write_text(json.dumps(d));os.replace(t,D/'live.json')
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');a=p.parse_args()
    while True:
        update()
        if not a.watch:break
        time.sleep(60)
