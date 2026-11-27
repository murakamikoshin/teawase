"""描き直しが崩れた子を 前の絵に戻す。 python3 restore.py id,id,..."""
import sys,os,shutil,subprocess
ids=[int(x) for x in sys.argv[1].split(',')]
for i in ids:
    got=[]
    for d in ('hires','lowres','flat2','newsrc'):
        f='/home/claude/oldart/%s_%d.png'%(d,i)
        if os.path.exists(f): got.append(d)
    if not got: print('no old art for',i); continue
    if 'newsrc' not in got and os.path.exists('/home/claude/newsrc/%d.png'%i): os.replace('/home/claude/newsrc/%d.png'%i,'/home/claude/oldart/rejected_%d.png'%i)
    for d in got: shutil.copy('/home/claude/oldart/%s_%d.png'%(d,i),'/home/claude/%s/%d.png'%(d,i))
    if os.path.exists('/home/claude/oldart/dot_%d.png'%i): shutil.copy('/home/claude/oldart/dot_%d.png'%i,'/home/claude/dots5/%d.png'%i)
    print('restored',i,got)
subprocess.check_call(['python3','/home/claude/tools/addart.py',','.join(map(str,ids))])
subprocess.check_call(['python3','/home/claude/tools/hiart.py'])
