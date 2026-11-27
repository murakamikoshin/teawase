"""キャラクター表を マス目で切り分ける。白い背景から 連結した かたまりを見つけ、マスごとにまとめる。
使い方: python3 slice.py 表.png 列 行 出力フォルダ id1,id2,..."""
import sys
from PIL import Image
import numpy as np
from scipy import ndimage
def run(f,cols,rows,out,ids):
    im=Image.open(f).convert('RGBA'); bg=Image.new('RGBA',im.size,(255,255,255,255)); bg.alpha_composite(im); im=bg.convert('RGB')
    a=np.asarray(im).astype(int); ink=(a.min(axis=2)<235)
    ink=ndimage.binary_dilation(ink,iterations=max(2,im.width//300))
    lab,n=ndimage.label(ink); H,W=ink.shape; boxes={}
    for i,sl in enumerate(ndimage.find_objects(lab),1):
        ys,xs=sl; area=(lab[sl]==i).sum()
        if area<(W*H)/(cols*rows)/400: continue
        cx=(xs.start+xs.stop)/2; cy=(ys.start+ys.stop)/2
        k=(min(rows-1,int(cy/H*rows)),min(cols-1,int(cx/W*cols)))
        b=boxes.get(k); bb=[xs.start,ys.start,xs.stop,ys.stop]
        boxes[k]=bb if not b else [min(b[0],bb[0]),min(b[1],bb[1]),max(b[2],bb[2]),max(b[3],bb[3])]
    res=[]
    for r in range(rows):
        for c in range(cols):
            idx=r*cols+c
            if idx>=len(ids) or not ids[idx]: continue
            b=boxes.get((r,c))
            if not b: print('MISSING cell',r,c); continue
            p=max(8,im.width//120); b=[max(0,b[0]-p),max(0,b[1]-p),min(W,b[2]+p),min(H,b[3]+p)]
            cell=im.crop(b); s=max(cell.size)+2*p; sq=Image.new('RGB',(s,s),(255,255,255)); sq.paste(cell,((s-cell.width)//2,(s-cell.height)//2))
            sq.save('%s/%s.png'%(out,ids[idx])); res.append((ids[idx],cell.size))
    print(res)
if __name__=='__main__':
    run(sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),sys.argv[4],sys.argv[5].split(','))
