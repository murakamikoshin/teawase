"""ドット絵を ゲームの ARTL に足す/差しかえる。使い方: python3 addart.py id1,id2,..."""
import re,base64,sys
from PIL import Image
F='/home/claude/teawase/index.html'
s=open(F).read()
a=s.index('var ARTL={'); e=s.index('};',a)
body=s[a+len('var ARTL={'):e]
ents=dict((int(m.group(1)),m.group(0)) for m in re.finditer(r'(\d+):\[\d+,\d+,"[^"]*"\]',body))
n0=len(ents)
for i in [int(x) for x in sys.argv[1].split(',')]:
    p='/home/claude/dots5/%d.png'%i; im=Image.open(p)
    ents[i]='%d:[%d,%d,"data:image/png;base64,%s"]'%(i,im.width,im.height,base64.b64encode(open(p,'rb').read()).decode())
s=s[:a]+'var ARTL={'+','.join(ents[k] for k in ents)+s[e:]
open(F,'w').write(s); print('art',n0,'->',len(ents))
