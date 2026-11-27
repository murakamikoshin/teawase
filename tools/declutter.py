"""切り出した絵から、となりのキャラの はみ出し(枠のふちに触れている 別のかたまり)を消す。 python3 declutter.py 画像..."""
import sys, os
import numpy as np
from PIL import Image
from scipy import ndimage
for f in sys.argv[1:]:
    im=Image.open(f).convert('RGB'); a=np.asarray(im).astype(int)
    ink=~((a.min(axis=2)>235))
    ink=ndimage.binary_dilation(ink,iterations=3)
    lab,n=ndimage.label(ink)
    if n<2: continue
    sizes=ndimage.sum(ink,lab,range(1,n+1)); big=1+int(np.argmax(sizes))
    ys,xs=np.nonzero(ink); y0,y1,x0,x1=ys.min(),ys.max(),xs.min(),xs.max()
    edge=set(np.unique(np.concatenate([lab[y0],lab[y1],lab[:,x0],lab[:,x1]])))-{0,big}
    edge={e for e in edge if sizes[e-1]<0.10*sizes[big-1]}
    if os.environ.get('DECL_ALL'): edge|={i+1 for i in range(n) if i+1!=big and sizes[i]<0.06*sizes[big-1]}     # 動きの線や 地面の影など、本体から離れた小さいものは全部消す     # 絵のいちばん外側にある 小さな別のかたまり=となりのキャラの はみ出し
    if not edge: continue
    out=np.asarray(im).copy(); out[np.isin(lab,list(edge))]=255
    Image.fromarray(out).save(f); print(f,'removed',len(edge))
