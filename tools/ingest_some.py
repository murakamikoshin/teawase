"""シート画像から えらんだ id だけ 差しかえる(目の 描き直し用。ほかの 子は さわらない)。
使い方: python3 ingest_some.py シート名 画像ファイル id,id,…   ★前面で 最後まで 待つ(index.html を 書きかえる)"""
import sys,json,subprocess,os,shutil
sh=json.load(open('/home/claude/artplan/sheets.json'))[sys.argv[1]]
ids=[str(i) for i in sh["ids"]]; want=[i for i in sys.argv[3].split(',') if i in ids]
tmp='/tmp/ingsome_'+sys.argv[1]; shutil.rmtree(tmp,ignore_errors=True); os.makedirs(tmp)
subprocess.check_call(['python3','/home/claude/tools/slice.py',sys.argv[2],str(sh["cols"]),str(sh["rows"]),tmp,','.join(ids)])
os.makedirs('/home/claude/newsrc_old',exist_ok=True); ok=[]
for i in want:
    src='%s/%s.png'%(tmp,i)
    if not os.path.exists(src): continue
    dst='/home/claude/newsrc/%s.png'%i
    if os.path.exists(dst) and not os.path.exists('/home/claude/newsrc_old/%s.png'%i): shutil.copy(dst,'/home/claude/newsrc_old/%s.png'%i)   # 前の 絵は とっておく
    shutil.copy(src,dst)
    subprocess.check_call(['python3','/home/claude/tools/dot5.py',dst,'/home/claude/dots5/%s.png'%i,'108','20']); ok.append(i)
subprocess.check_call(['python3','/home/claude/tools/addart.py',','.join(ok)])
subprocess.check_call(['python3','/home/claude/tools/hiart.py'])
print('replaced',ok)
