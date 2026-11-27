"""hs3 に増えた動画を hs2 と アプリに配って、publish 用の一覧を出す"""
import os,json,shutil
done=set(open('/home/claude/hs3/published.txt').read().split()) if os.path.exists('/home/claude/hs3/published.txt') else set()
new=sorted(int(f[:-4]) for f in os.listdir('/home/claude/hs3') if f.endswith('.mp4') and f[:-4] not in done)
for i in new:
    old='/home/claude/hs2_v1/%d.mp4'%i
    if not os.path.exists(old) and os.path.exists('/home/claude/hs2/%d.mp4'%i): shutil.copy('/home/claude/hs2/%d.mp4'%i,old)
    shutil.copy('/home/claude/hs3/%d.mp4'%i,'/home/claude/hs2/%d.mp4'%i); shutil.copy('/home/claude/hs3/%d.mp4'%i,'/home/claude/TeawaseApp/Teawase/Resources/v/%d.mp4'%i)
print(len(new),'new; total',len(os.listdir('/home/claude/hs3'))-2)
print(json.dumps([{'as':'v/%d.mp4'%i,'path':'/home/claude/hs2/%d.mp4'%i} for i in new]))
open('/home/claude/hs3/pending.txt','w').write(' '.join(map(str,new)))
