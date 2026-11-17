import pathlib,json,subprocess,time,numpy as np
R=pathlib.Path(__file__).resolve().parents[1]
D=R/'docs/rebuttal'; M=D/'media'; M.mkdir(parents=True,exist_ok=True)
items=[]
for action in ['press','tap30hold','stretch']:
    candidates=sorted((R/'data/v2_pilot/b1'/action).glob('*_2.75_a'))
    if not candidates:continue
    a=candidates[0]; b=a.with_name(a.name[:-1]+'b')
    for member,folder in [('a',a),('b',b)]:
        packs=sorted(folder.glob('*/sample_pack.npz'))
        if not packs:continue
        with np.load(packs[0]) as pack:
            for field in ['rgb','stress','flow']:
                for view in range(3):
                    target=M/f'{action}_{member}_{field}_{view}.mp4'
                    if not target.exists():
                        frames=np.ascontiguousarray(pack[field][view].transpose(0,2,3,1))
                        subprocess.run(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s','224x224','-r','32','-i','pipe:0','-an','-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-movflags','+faststart',str(target)],input=frames.tobytes(),check=True)
        items.append({'action':action,'member':member,'gt':json.loads((folder/'gt.json').read_text()),'source':str(packs[0].relative_to(R))})
    print('exported',action,flush=True)
(D/'media_manifest.json').write_text(json.dumps({'selection':'Lexicographically first object per action, base 2.75; exploratory train examples, not held-out evaluation. RGB and projected fields are simulator ground truth, not network predictions.','items':items},indent=2))
print('media complete',len(items),flush=True)
