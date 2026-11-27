"""ムービー集(hsgal/index.html)に 動画を足す/名前を直す。使い方: python3 galadd.py id,id,..."""
import sys,re,json,subprocess,base64; sys.path.insert(0,'/home/claude/tools')
import hs_board
ids=[int(x) for x in sys.argv[1].split(',')]
g=open('/home/claude/teawase/index.html').read(); p='/home/claude/hsgal/index.html'; h=open(p).read()
m=re.search(r'var D=(\[.*?\]),RK=',h,re.S); D=json.loads(m.group(1)); have={d['id']:d for d in D}
for i in ids:
    q=re.search(r'\[%d,"([^"]+)",(\d),"'%i,g); name,r=q.group(1),int(q.group(2))
    subprocess.check_call(['ffmpeg','-loglevel','error','-y','-ss','4.2','-i','/home/claude/hs2/%d.mp4'%i,'-frames:v','1','-vf','scale=180:320','-q:v','9','/home/claude/hsgal/p/%d.jpg'%i])
    d={'id':i,'n':name,'r':r,'m':hs_board.BOARD[i]['name'] if i in hs_board.BOARD else '','p':base64.b64encode(open('/home/claude/hsgal/p/%d.jpg'%i,'rb').read()).decode()}
    if i in have: have[i].update(d)
    else: D.append(d)
D.sort(key=lambda x:(-x['r'],x['id']))
open(p,'w').write(h[:m.start(1)]+json.dumps(D,ensure_ascii=False)+h[m.end(1):]); print(len(D),'videos')
