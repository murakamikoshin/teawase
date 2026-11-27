import sys,os,glob,shutil,subprocess
src=sys.argv[1]; work=sys.argv[2]; pipe=sys.argv[3]
ids=[15,10,11,12,18,13,14,19,20,31,32,33,1,2,3,4,5,16,6,7,8,9,17,101,102,103,104,105,116,106,107,108,109,117]
files=sorted(glob.glob(os.path.join(src,'mcp-Canva-blob-*.png')))
assert len(files)==len(ids)
ok=0
for i,f in zip(ids,files):
    bid='m%d'%i; d=os.path.join(work,bid,'source'); os.makedirs(d,exist_ok=True)
    shutil.copy(f,os.path.join(d,bid+'_source.png'))
    r=subprocess.run([sys.executable,os.path.join(pipe,'make_still.py'),os.path.join(work,bid)],capture_output=True,text=True)
    if r.returncode: print(bid,'FAIL',r.stderr[-300:])
    else: ok+=1
print('ok',ok)
from PIL import Image
sheet=Image.new('RGB',(9*150,4*150),(20,16,54))
for k,i in enumerate(ids):
    p=os.path.join(work,'m%d'%i,'source','m%d_still.png'%i)
    if not os.path.exists(p): continue
    t=Image.open(p).convert('RGBA')
    if k==0: print('still size',t.size)
    s=140/max(t.size); t=t.resize((int(t.width*s),int(t.height*s)),Image.NEAREST)
    sheet.paste(t,((k%9)*150+5,(k//9)*150+5),t)
sheet.save(os.path.join(work,'sheet.png'))
