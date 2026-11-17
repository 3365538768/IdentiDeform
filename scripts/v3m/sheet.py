import sys, numpy as np, cv2
sys.path.insert(0,'scripts/v3m'); from common import load, cropped_view
idxs=[int(x) for x in sys.argv[1].split(',')]; out=sys.argv[2]
rows=[]
for i in idxs:
    z,g=load(i)
    if z is None: continue
    rgb=z['rgb']; m=z['object_mask']
    tiles=[]
    for v in (0,1,2):
        fr,_=cropped_view(rgb[v],m[v],out=96)
        tiles+= [fr[t] for t in (0,32,64,127)]
    row=np.concatenate(tiles,1).copy()
    cv2.putText(row,f"{i} {g['action']} {g['law']}",(2,12),cv2.FONT_HERSHEY_SIMPLEX,0.4,(255,0,0),1)
    rows.append(row)
cv2.imwrite(out, cv2.cvtColor(np.concatenate(rows,0),cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,85])
