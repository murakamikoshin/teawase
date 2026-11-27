# -*- coding: utf-8 -*-
"""tools/skilldef.py の 表を index.html に 入れる(/*SKD*/ ... /*SKD_END*/ を 作りかえ)。"""
import json,re,sys
sys.path.insert(0,'/home/claude/tools'); import skilldef
P='/home/claude/teawase/index.html'; h=open(P,encoding='utf-8').read()
C={c['id']:c for c in json.load(open('/home/claude/artplan/chars.json',encoding='utf-8'))}
TR_B=[0.03,0.04,0.05,0.06,0.08,0.10,0.13,0.16]
TR_K={"pw":1,"hp":1,"slow":1.2,"zone":1.5,"perf":1.6,"first":2,"last":2,"weak":1.5,"adv":1.5,"vs":1.6,"hand":1.6}
def f(v): return ("%g"%round(v,2))+"倍"
def pct(v): return "%d%%"%round(v*100)
def sdesc(x):
    d=[]
    if 'turn' in x: d.append("%dターン目に出すと %s"%(x['turn'],f(x['tk'])))
    if 'last' in x: d.append("さいごのターンに出すと "+f(x['last']))
    if 'full' in x: d.append("相手の体力が満タンなら "+f(x['full']))
    if 'lowHp' in x: d.append("相手の体力が 半分以下なら "+f(x['lowHp']))
    if 'perfK' in x: d.append("PERFECTなら さらに "+f(x['perfK']))
    if 'vsHigh' in x: d.append("SSR以上の相手に "+f(x['vsHigh']))
    if 'zone' in x: d.append("当たりの広さが "+f(x['zone']))
    if 'slow' in x: d.append("ゲージが %d%% ゆっくり"%round((1-x['slow'])*100))
    if 'floor' in x: d.append("はずしても %d割の力は出る"%round(x['floor']*10))
    if 'neutral' in x: d.append("不利な手でも 等倍で通る")
    if 'reverse' in x: d.append("不利な手のとき 逆に "+f(x['reverse']))
    if 'peek' in x: d.append("次のターンの 相手の手が見える")
    if 'perf2' in x: d.append("PERFECT 1回が 2回ぶんに数えられる")
    if 'any' in x: d.append(("いつ出しても " if not d else "いつでも ")+f(x['any']))
    return "。".join(d)
def trtext(k,v):
    T={"pw":"全員のパワー +%s","hp":"相手の体力 −%s","slow":"全員のゲージが %s ゆっくり","zone":"全員の当たりが %s 広い","perf":"PERFECTのとき さらに +%s",
       "first":"1ターン目の こうげき +%s","last":"さいごのターンの こうげき +%s","vs":"SSR以上の相手に +%s","h_rock":"ゴウの なかまのパワー +%s","h_scissors":"ザンの なかまのパワー +%s","h_paper":"ジュツの なかまのパワー +%s"}
    if k=="weak": return "不利な手の倍率 +%.2f（0.7 → %.2f）"%(v,min(1,0.7+v))
    if k=="adv": return "有利な手の倍率 +%.2f（1.5 → %.2f）"%(v,1.5+v)
    if k=="shard": return "かけらが出るとき +%dこ"%v
    return T[k]%pct(v)
SKD={};TRH={};names={}
for (i,sn,x,tn,te,why) in skilldef.D:
    c=C[i]; r=c['r']
    assert i!=84 and sn!="星葬" and "星葬" not in tn, i
    if sn:
        d=sdesc(x)+("。進化した 子なので さらに いつでも 1.2倍" if c.get('from') else "")
        SKD[i]={"n":[sn,d],"x":x}; names[sn]=names.get(sn,0)+1
    e=[]
    for k,m in te:
        if k=="shard": e.append([k,int(m)*(2 if r>=5 else 1)])
        else: e.append([k,round(TR_B[r-1]*TR_K["hand" if k.startswith("h_") else k]*m,3)])
    TRH[i]={"n":tn,"e":e,"d":"・".join(trtext(k,v) for k,v in e)}
dup=[k for k,v in names.items() if v>2]; assert not dup,dup
blk="/*SKD*/var SKD=%s;for(var kd in SKD){SK[kd]=SKD[kd].n;SKX[kd]=SKD[kd].x;}var TRH=%s;/*SKD_END*/\n"%(json.dumps(SKD,ensure_ascii=False),json.dumps(TRH,ensure_ascii=False))
if "/*SKD*/" in h: h=re.sub(r'/\*SKD\*/.*?/\*SKD_END\*/\n',lambda m:blk,h,flags=re.S)
else: h=h.replace("var TRC={};",blk+"var TRC={};",1)
if "TRH[m.id]" not in h:
    h=h.replace("  if(TRC[m.id]){return TRC[m.id];}\n","  if(TRC[m.id]){return TRC[m.id];}\n  if(typeof TRH!==\"undefined\"&&TRH[m.id]&&m.id!==84){TRC[m.id]=TRH[m.id];return TRH[m.id];}\n",1)
assert "TRH[m.id]" in h
open(P,'w',encoding='utf-8').write(h)
print("skills",len(SKD),"traits",len(TRH))
