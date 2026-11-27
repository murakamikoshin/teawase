"""貼ってもらったシート画像を 切り分け → ドット化 → ゲームに追加。
使い方: python3 ingest.py シート名(H01など) 画像ファイル"""
import sys,json,subprocess,os
sh=json.load(open('/home/claude/artplan/sheets.json'))[sys.argv[1]]
ids=[str(i) for i in sh["ids"]]
os.makedirs('/home/claude/newsrc',exist_ok=True)
subprocess.check_call(['python3','/home/claude/tools/slice.py',sys.argv[2],str(sh["cols"]),str(sh["rows"]),'/home/claude/newsrc',','.join(ids)])
ok=[]
for i in ids:
    if os.path.exists('/home/claude/newsrc/%s.png'%i):
        subprocess.check_call(['python3','/home/claude/tools/dot5.py','/home/claude/newsrc/%s.png'%i,'/home/claude/dots5/%s.png'%i,'108','20']); ok.append(i)
subprocess.check_call(['python3','/home/claude/tools/addart.py',','.join(ok)])
subprocess.check_call(['python3','/home/claude/tools/hiart.py'])   # 元の絵(イラスト表示用)も入れなおす
print('added',ok)
