import sys,os,subprocess
work=sys.argv[1]; pipe=sys.argv[2]; steps=sys.argv[3]; H=sys.argv[4]
ids=[15,10,11,12,18,13,14,19,20,31,32,33,1,2,3,4,5,16,6,7,8,9,17,101,102,103,104,105,116,106,107,108,109,117]
from PIL import Image
sheet=Image.new('RGB',(9*150,4*150),(20,16,54))
for k,i in enumerate(ids):
    src=os.path.join(work,'m%d'%i,'source','m%d_source.png'%i); out=os.path.join(work,'m%d'%i,'t_%s.png'%steps)
    subprocess.run([sys.executable,os.path.join(pipe,'pixel4.py'),src,out,H,'24'],check=True,stdout=subprocess.DEVNULL)
    if 'm' in steps: subprocess.run([sys.executable,os.path.join(pipe,'material.py'),out,out,'3','1','2'],check=True,stdout=subprocess.DEVNULL)
    t=Image.open(out).convert('RGBA')
    s=140/max(t.size); t=t.resize((max(1,int(t.width*s)),max(1,int(t.height*s))),Image.NEAREST)
    sheet.paste(t,((k%9)*150+5,(k//9)*150+5),t)
sheet.save(os.path.join(work,'sheet_%s.png'%steps))
