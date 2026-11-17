import numpy as np, cv2, glob, json
def load(i, split='val'):
    p=glob.glob(f'data/v2_main/{split}/{i:06d}/*/sample_pack.npz')
    if not p: return None, None
    z=np.load(p[0]); g=json.load(open(f'data/v2_main/{split}/{i:06d}/gt.json'))
    return z, g
def crop_box(mask, margin=0.12, minside=48):
    # mask: T,H,W bool -> square box (y0,y1,x0,x1) covering the union
    u=mask.any(0); ys,xs=np.nonzero(u)
    H,W=u.shape
    if len(ys)==0: return 0,H,0,W
    cy,cx=(ys.min()+ys.max())/2,(xs.min()+xs.max())/2
    s=max(ys.max()-ys.min(),xs.max()-xs.min(),minside)*(1+2*margin)
    s=min(s,H); y0=int(round(min(max(cy-s/2,0),H-s))); x0=int(round(min(max(cx-s/2,0),W-s)))
    return y0,y0+int(s),x0,x0+int(s)
def cropped_view(rgb_tchw, mask_tchw, out=256, box=None):
    frames=rgb_tchw.transpose(0,2,3,1)
    if box is None: box=crop_box(mask_tchw[:,0]>127)
    y0,y1,x0,x1=box
    return np.stack([cv2.resize(f[y0:y1,x0:x1],(out,out),interpolation=cv2.INTER_CUBIC if (y1-y0)<out else cv2.INTER_AREA) for f in frames]), box
