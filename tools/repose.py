"""立ち絵の描き直し(ポーズ変更)。 python3 repose.py ref P02 → repose/P02_ref.png と 頼む文を出す / python3 repose.py ingest P02 画像 → 古い絵を oldart に退避して入れかえ"""
import sys,json,re,base64,io,os,subprocess,shutil
from PIL import Image
PLAN=json.load(open('/home/claude/repose/plan.json'))
cmd,key=sys.argv[1],sys.argv[2]; sh=PLAN[key]; ids=sh["ids"]; C=sh["cols"]; R=sh["rows"]
if cmd=="ref":
    s=open('/home/claude/teawase/index.html').read(); a=s.index('var ARTH={'); e=s.index('};',a)
    d={int(i):b for i,b in re.findall(r'(\d+):\[\d+,\d+,"data:image/webp;base64,([^"]+)"',s[a:e])}
    W=420 if C==3 else 400; im=Image.new('RGB',(C*W,R*W),(255,255,255))
    for k,i in enumerate(ids):
        t=Image.open(io.BytesIO(base64.b64decode(d[i]))).convert('RGBA'); t.thumbnail((W-70,W-70),Image.LANCZOS)
        im.paste(t,((k%C)*W+(W-t.width)//2,(k//C)*W+(W-t.height)//2),t)
    im.save('/home/claude/repose/%s_ref.png'%key)
    kind="モンスター" if key[0]=="Q" else "ちびキャラ"
    print("添付のキャラクター表と同じ%d体を、同じデザイン・同じ色・同じ絵柄・同じ並び順(横%d×縦%d)のまま、ポーズだけ変えて描き直してください。顔・体の形・もよう・服・持ち物・色は添付と同じにする。%s、全身、1体ずつ離して配置し重ならない。濃い輪郭線、1段のベタ影、グラデーションなし、発光なし。背景は真っ白の無地。文字・番号・枠線は入れない。ポーズ(左上から右へ順に)：%s。"%(len(ids),C,R,kind,"。".join("%d＝%s"%(j+1,p) for j,p in enumerate(sh["poses"]))))
else:
    os.makedirs('/home/claude/oldart',exist_ok=True)
    for i in ids:
        for d in ('hires','lowres','newsrc','flat2'):
            f='/home/claude/%s/%d.png'%(d,i)
            if os.path.exists(f): shutil.move(f,'/home/claude/oldart/%s_%d.png'%(d,i))
        shutil.copy('/home/claude/dots5/%d.png'%i,'/home/claude/oldart/dot_%d.png'%i)
    L=','.join(map(str,ids))
    subprocess.check_call(['python3','/home/claude/tools/slice.py',sys.argv[3],str(C),str(R),'/home/claude/newsrc',L])
    subprocess.check_call(['python3','/home/claude/tools/declutter.py']+['/home/claude/newsrc/%d.png'%i for i in ids])
    for i in ids: subprocess.check_call(['python3','/home/claude/tools/dot5.py','/home/claude/newsrc/%d.png'%i,'/home/claude/dots5/%d.png'%i,'108','20'])
    subprocess.check_call(['python3','/home/claude/tools/addart.py',L]); subprocess.check_call(['python3','/home/claude/tools/hiart.py'])
