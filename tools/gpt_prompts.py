# -*- coding: utf-8 -*-
"""ChatGPT で 全キャラを 描き直す ための 発注文(透過 PNG)を 作る。エイオン(83)・オルレア(84)・ニヒル(70)は のぞく。
出力: artplan/ChatGPT発注.md と artplan/sheets.json の G01〜(切り分け用。ingest.py G05 画像 で 取りこめる)"""
import json,sys
sys.path.insert(0,'/home/claude/tools')
src=open('/home/claude/tools/artplan.py',encoding='utf-8').read()
ns={}; exec(src[:src.index('humans=[')],ns)
names,LOOK,MLOOK,EVOLOOK,TITLES=ns['names'],ns['LOOK'],ns['MLOOK'],ns['EVOLOOK'],ns['TITLES']
chars=json.load(open('/home/claude/artplan/chars.json',encoding='utf-8'))
C={c['id']:c for c in chars}
SKIP={70,83,84}
XLOOK={
 10:"赤い つんつん髪、赤と金の 軽い よろい、炎の 剣を ふりかぶる 少年",11:"長い 水色の髪、青い ローブに 波もよう、先に 丸い 水晶の ついた 杖の 少女",
 12:"緑の フードと マント、茶色い 革の 服、短い 剣を 持つ 森の 少年",13:"赤と黒の 戦国の よろいと 大きな 金の 角の かぶと、口元を 面頬で かくした 武将、刀",
 14:"白い 狼の 頭の かぶと(顔は 見えない)、白と 水色の よろいと 毛皮、氷の 剣の 騎士",15:"金髪、小さな 金の 冠、白と 金の よろい、青い マント、聖剣を 高く かかげる 騎士",
 18:"長い 金髪の ツインテール、紫と 金の 服、いなずま形の 剣を 持つ 少女",19:"長い 黒髪、紫の とんがり帽子と ローブ、三日月の 杖の 魔女",
 20:"黒い つんつん髪、炎の ような 金の 冠、赤と 金の よろい、赤い マントの 覇王",31:"とがった 角の ある 黒い フルフェイスの かぶと、黒と 赤の よろい、黒い 大剣の 騎士(顔は 見えない)",
 32:"白い 長い ひげ、緑と 金の ローブと とんがり帽子、砂時計の ついた 杖の 老人",33:"緑の 竜の うろこの よろい、竜の 頭の かぶと、赤い マント、大剣の 英雄",
 55:"白い 髪、緑の 長い マフラー、紺の 忍び装束、くないを 持つ 少年",56:"長い 黒髪、赤と 白の 巫女服、神楽鈴を 持つ 少女",
 57:"そり頭、オレンジの 修行僧の 服、手に さらし、拳を かまえる 少年",58:"ピンクの 長い髪、花の 冠、緑の マント、花の 咲いた 杖の 少女",
 59:"白い ターバン、紫の ベスト、日焼けした 肌、曲刀を 持つ 盗賊の 少年",60:"竜の 角の ついた 青い かぶと、青い よろいと マント、長い 槍の 見習い騎士",
 61:"白い 狼の 耳と しっぽ、紺と 白の 着物、刀を 腰に さした 侍",62:"金髪、紺の とんがり帽子に 星、紺の ローブ、星の ついた 杖の 占い師の 少女",
 63:"茶色い ひげ、銀の よろい、赤い マント、大きな 四角い 盾の 将軍",
 81:"長い 白金の 髪、星の 冠、白と 金の 神々しい ドレス、背に 光の 翼(★7 なので 描きこみ 多め)",
 82:"長い 銀髪、黒と 紫と 金の 王の ローブ、黒い 星の 冠、背に 黒い 結晶の 飾り(★7 なので 描きこみ 多め)",
 1:"からを 頭まで かぶった 黄色い ひよこ、頭の 上に 小さな 火",2:"緑の まるい あまがえる、頭に 水の しずく",3:"紫の いもむし、背中に 緑の 葉っぱ",
 4:"茶色い ねずみ、ほっぺが ふくらんで どんぐりを 抱える",5:"水色の まるい ふぐ",16:"こげ茶の まるい もぐら、頭に 小さな 芽",
 6:"しっぽが 炎の 柴犬の 子",7:"緑の こうらの かめ(こうらは 盾の ように かたい)",8:"紫の かさの きのこ、かさに 葉っぱ、小さな 手足",
 9:"紫の うさぎ、ひたいに 金の 一本角",17:"青緑の いたち、しっぽの まわりに 小さな つむじ風",
 101:"体が 炎に つつまれた 大きな にわとり(プチボーの 進化)",102:"王冠を かぶった 大きな かえる、背中から 水が ふき出す(ケロシズクの 進化)",
 103:"大きな 葉っぱの 羽の ちょう(ハッパムシの 進化)",104:"どんぐりの かぶとと よろいの ねずみ戦士(コロネズの 進化)",
 105:"とげが いっぱいの 大きな ふぐ(ウキブクの 進化)",116:"鉄の よろいと 大きな つめの もぐら(ポテモグの 進化)",
 106:"全身が 炎の 大きな 狐犬(モエイヌの 進化)",107:"こうらの 上に 小さな 城が のった 大きな かめ(コウランの 進化)",
 108:"かさに こけの 生えた 大きな きのこの 番人(モリタケの 進化)",109:"白い 一角うさぎ、三日月の もよう(ツノウサの 進化)",117:"両手が たつまきの いたち(カゼイタチの 進化)"}
ids=[c['id'] for c in chars if c['id'] not in SKIP]
def look(i):
    c=C[i]
    if c.get('look'): return c['look'].replace('、すすのついた顔','')
    if i in XLOOK: return XLOOK[i]
    if i in LOOK: return LOOK[i].replace('すすのついた頬、','').replace('、すすのついた顔','')
    if i in EVOLOOK: return EVOLOOK[i]
    if i in MLOOK: return MLOOK[i]
    if c.get('lore'): return c['lore'][0]+"。"+c['lore'][1]
    n=names.get(i); return n[1] if n else ""
HUM_X={10,11,12,13,14,15,18,19,20,31,32,33,55,56,57,58,59,60,61,62,63,81,82}
def is_human(i): return i in LOOK or i in HUM_X
HAND={"rock":"ゴウ(力)","scissors":"ザン(刃)","paper":"ジュツ(術)"}
EL={"fire":"炎","water":"水","leaf":"草木","earth":"大地","wind":"風","thunder":"雷","ice":"氷","dark":"闇","light":"光","star":"星","petal":"花","poison":"毒"}
TIER0={1:"【★1 素朴】まるい シンプルな 形・色 2〜3色",2:"【★2 素朴】シンプル・色 2〜3色、ちょっと だけ 特徴を 足す",3:"【★3 素朴】服は 無地寄り・小物 1つ・色 2〜3色。かわいく シンプルに",
 4:"【★4 しっかり】縁取りや 紋章など 装飾 少し・小物 2つ・色 3〜4色",
 5:"【★5 豪華】金の ふち・宝石・なびく マントや 布。かっこいい 決めポーズ",
 6:"【★6 超豪華】金銀の 細かい 装飾、大きな マント・翼・武器。服や 武器の 作りこみを ★5より 一段 上に。堂々とした 主役級の ポーズと 表情",
 7:"【★7 最上級】★6より さらに 上。冠・翼・衣装の 作りこみで ひと目で 別格と わかる"}
TIER={k:(v+("")) for k,v in TIER0.items()}
def line(k,i,tier=True):
    c=C[i]; t=c['t'] or (TITLES.get(i) if isinstance(TITLES,dict) else "") or ""
    return "%d. %s%s(★%d)— %s%s"%(k,c['n'],"「"+t+"」" if t else "",c['r'],look(i),("。"+TIER.get(c['r'],"")) if tier else "")
def mkbody(ch):
    rs={C[i]['r'] for i in ch}
    if len(rs)==1:
        r=rs.pop(); return "\n".join(line(j+1,i,False) for j,i in enumerate(ch))+"\n全員 ★%d: %s。"%(r,TIER.get(r,"").split("】")[-1])
    return "\n".join(line(j+1,i) for j,i in enumerate(ch))
mon=[i for i in ids if not is_human(i)]
hum=sorted([i for i in ids if is_human(i)],key=lambda i:(C[i]['r'],i))
# モンスター: 進化前と 進化後を 同じ 列(上と 下)に
pairs=[]; used=set()
for i in mon:
    if i in used: continue
    e=i+100 if (i+100) in C and (i+100) not in SKIP and C[i+100].get('from')==i else None
    if e: pairs.append((i,e)); used|={i,e}
singles=[i for i in mon if i not in used]
STYLE0=("日本の スマホRPGの キャラクター表。2〜2.5頭身の ちびキャラ。太めの 濃い 輪郭線、セル塗り(影は 1〜2段)、色は あざやかで 少なめ、★3〜4は 細部を 描きこみすぎない。"
 "目は キャラごとに 色と 形を 変える(つり目・たれ目・丸い目・半目など 性格に 合わせる)。瞳は 髪や 服に 合う 色(赤・青・緑・金・紫など。黒目に しない)で はっきり、小さな 白い ハイライトを 1つ。両目は 同じ 形・同じ 色・まつ毛も 左右 同じ。"
 "顔に 汚れ・しみ・すす・傷・ほくろを 描かない。ポーズは 1体ずつ はっきり 変える(ふりかぶる・走る・とぶ・しゃがむ・ふり返る・両手を 広げる など)。体の 向きも まぜる: 正面・左向き・右向き・ななめを バラバラに(右向きは 1まいに 2体まで)。全身を 入れて、どの 子も 切れない。人の キャラは 腕・手・足が 必ず 2本ずつ、指は 親指を 入れて 5本(開いた 手は 指を 数えて 描く。余分な 手足や 指を 描かない)。出す 前に 1体ずつ「腕 2・手 2・足 2・指 5」を 数えて 確かめ、ちがったら 直してから 出す。"
 "**レア度(★)が 高いほど 衣装・装飾・武器・ポーズを 豪華に**。同じ シートの 中でも ★の ちがいが ひと目で わかる ように。**体から はなれた エフェクト(オーラ・光の 粒・炎・後光・光の 輪)は 描かない**(ゲーム側で つける)。")
TECH=("**背景は 完全に 透明(透過 PNG)**。影の 楕円・床・背景色・枠・文字・名前・番号は 入れない。キャラ どうしは 十分に はなして、重ならない・はみ出さない。")
STYLE=STYLE0.replace("**","").replace("(ゲーム側で つける)",""); TECH=TECH.replace("**","")
md=["# テアワセ！ ChatGPT 発注文(全キャラ 描き直し・透過素材)\n",
"エイオン・オルレア・ニヒルは のぞく。全部で **%d体 / %dまい**。\n"%(len(ids),0),
"## つかい方\n- コードブロックの 中を そのまま 貼る。\n- **G01 だけ ニヒルの 絵を 添付**(絵柄の 参考。添付なしで 試す ときは G01 の 1段落目を 消す)。\n- G01 が 決まったら 次は **G18(★6〜7)** で ★の 差を 確かめる → よければ 残りを 順に。\n- 2まい目以降は **気に入った G01 の 絵を 毎回 添付**(絵柄が そろう)。\n"
"- 出力は **透過 PNG**。背景が 白や 市松模様で 出たら「背景を 透明に して 出し直して」と 1回 言う。\n"
"- できた 画像は シート名(G01 など)を そえて 貼って くれれば、こちらで 切り分けて ゲームに 入れる。並び順が くずれたら その シートだけ 作り直し。\n"
"- 1まいに 入れる 数が 多くて 崩れる なら、言って くれれば 4体ずつに 分け直す。\n"]
sheets=json.load(open('/home/claude/artplan/sheets.json',encoding='utf-8'))
for k in [k for k in sheets if k.startswith('G')]: del sheets[k]
n=0
def chunks(a,k): return [a[j:j+k] for j in range(0,len(a),k)]
# 人: 6体(横3×縦2)
for ch in chunks(hum,6):
    n+=1; key="G%02d"%n; sheets[key]={"cols":3,"rows":2,"ids":ch}
    body=mkbody(ch)
    md.append("\n## %s(人・%d体・横3×縦2)\n```\n%s%s\n横長(3:2)の 画像に、%d体を 横3×縦2の マス目に 等間隔で 並べる。左上から 右へ、上の 段 → 下の 段の 順:\n%s\n%s\n```\n中身: %s\n"%(
        key,len(ch),"添付画像は 線の太さ・塗り方・頭身だけ 参考に して。目・表情・服・色・装飾の 量・ポーズ・オーラは まねない(添付の キャラは 特別な 目なので 目は 絶対に 似せない)。目は 下の 指定どおり 1体ずつ ちがう 色と 形で、明るく 個性的に。今回は ★3 なので 添付画像より ずっと 素朴に。添付画像の キャラ自体は 描かない。\n" if n==1 else "添付画像と 同じ 絵柄で 描いて ください。添付画像の キャラ自体は 描かない。\n",STYLE,len(ch),body,TECH," / ".join(C[i]['n'] for i in ch)))
# モンスター: 進化 ペアは 上下(横4×縦2)
pp=pairs[:]; ss=singles[:]
while pp or ss:
    top=[];bot=[]
    while pp and len(top)<4:
        a,b=pp.pop(0); top.append(a); bot.append(b)
    while len(top)<4 and ss: top.append(ss.pop(0))
    while len(bot)<4 and ss: bot.append(ss.pop(0))
    ch=top+bot; n+=1; key="G%02d"%n; sheets[key]={"cols":4,"rows":2,"ids":ch}
    note="上の 段の 子の 真下は その子が 進化した 姿(同じ 生きもの・同じ 色合いで、大きく 強そうに)。" if any(C[i].get('from') for i in bot) else ""
    body=mkbody(ch)
    md.append("\n## %s(モンスター・%d体・横4×縦2)\n```\n添付画像と 同じ 絵柄で 描いて ください。添付画像の キャラ自体は 描かない。人間では なく モンスター(指定した モチーフの 形に する)。\n%s\n横長(3:2)の 画像に、%d体を 横4×縦2の マス目に 等間隔で 並べる。%s左上から 右へ、上の 段 → 下の 段の 順:\n%s\n%s\n```\n中身: %s\n"%(
        key,len(ch),STYLE,len(ch),note,body,TECH," / ".join(C[i]['n'] for i in ch)))
# ===== 追加分: まだ 絵の ない 子(ガチャ限定 442〜500・モンスター 45〜54/201〜231 と 進化・ハジマリ) =====
import re
h=open('/home/claude/teawase/index.html',encoding='utf-8').read()
MR={int(a):(b,int(r),hd) for a,b,r,hd in re.findall(r'\[(\d+),"([^"]+)",(\d),"(rock|scissors|paper)"\]',h)}
EVN={int(a):b for a,b in re.findall(r'(\d+):"([^"]+)"',h[h.find('var EVO_NAMES'):h.find('}',h.find('var EVO_NAMES'))])}
import gacha80
X=[]
for g in gacha80.G:
    if g[0]>=442: C[g[0]]={'id':g[0],'n':g[1],'r':g[2],'h':g[3],'fx':g[4],'t':g[5],'look':g[6]}; X.append(g[0])
xh=sorted(X,key=lambda i:(C[i]['r'],i))
xm=[i for i in list(range(45,55))+list(range(201,232))+[79]]
for i in xm:
    nm,r,hd=MR[i]; lk=MLOOK.get(i) or names[i][1]
    if i==79: lk="さいしょの ひとしずく。きらきら 光る 水の しずくの 精(★6 レジェンド)"
    else: lk=lk+"(人間では なく この モチーフの 生きもの)"
    C[i]={'id':i,'n':nm,'r':r,'h':hd,'fx':None,'t':'','look':lk}
xpairs=[]
for i in xm:
    if i in EVN and (i+100) in names:
        e=i+100; b=C[i]; C[e]={'id':e,'n':EVN[i],'r':min(b['r']+1,3),'h':b['h'],'fx':None,'t':'','from':i,
          'look':EVOLOOK.get(e) or ("%sが 成長して 大きく 強そうに なった 姿(同じ 色合い)"%b['n'])}
        xpairs.append((i,e))
xs=[i for i in xm if i not in {a for a,b in xpairs}]
md.append("\n---\n# 追加分(まだ 絵が ない 子 %d体)\nここからも 同じ やり方。G01 の 絵を 添付して。\n"%(len(xh)+len(xm)+len(xpairs)))
first=n
for ch in chunks(xh,6):
    n+=1; key="G%02d"%n; sheets[key]={"cols":3,"rows":2,"ids":ch}
    body=mkbody(ch)
    md.append("\n## %s(人・%d体・横3×縦2)\n```\n添付画像と 同じ 絵柄で 描いて ください。添付画像の キャラ自体は 描かない。\n%s\n横長(3:2)の 画像に、%d体を 横3×縦2の マス目に 等間隔で 並べる。左上から 右へ、上の 段 → 下の 段の 順:\n%s\n%s\n```\n中身: %s\n"%(
        key,len(ch),STYLE,len(ch),body,TECH," / ".join(C[i]['n'] for i in ch)))
pp=xpairs[:]; ss=xs[:]
while pp or ss:
    top=[];bot=[]
    while pp and len(top)<4:
        a,b=pp.pop(0); top.append(a); bot.append(b)
    if not bot:
        top=ss[:4]; bot=ss[4:8]; ss=ss[8:]
    ch=top+bot; n+=1; key="G%02d"%n
    cw,rw=(4,2) if len(ch)>4 else (len(ch),1)
    sheets[key]={"cols":cw,"rows":rw,"ids":ch}
    note="上の 段の 子の 真下は その子が 進化した 姿(同じ 生きもの・同じ 色合いで、大きく 強そうに)。" if any(C[i].get('from') for i in bot) else ""
    body=mkbody(ch)
    md.append("\n## %s(モンスター・%d体・横%d×縦%d)\n```\n添付画像と 同じ 絵柄で 描いて ください。添付画像の キャラ自体は 描かない。人間では なく モンスター(指定した モチーフの 形に する)。\n%s\n横長の 画像に、%d体を 横%d×縦%dの マス目に 等間隔で 並べる。%s左から 右へ%s:\n%s\n%s\n```\n中身: %s\n"%(
        key,len(ch),cw,rw,STYLE,len(ch),cw,rw,note,"、上の 段 → 下の 段の 順" if rw>1 else "",body,TECH," / ".join(C[i]['n'] for i in ch)))
# ===== ポーズ分(★5以上: 1体 4ポーズ。モーション・ムービー用) =====
POSE=["待機(少し 構えた 立ち姿)","攻撃(武器や 手を 大きく ふりかぶる)","必殺(いちばんの 決めポーズ・全力)","勝利(よろこぶ・余裕の 笑み)"]
hi=sorted([i for i in C if C[i]['r']>=5 and i not in SKIP and (i in ids or i in X)],key=lambda i:(C[i]['r'],i))
md.append("\n---\n# ポーズ分(★5以上 %d体・1体 4ポーズ)\nバトルの 動き・必殺ムービー用。**その 2体の 通常の 絵(G シートで 作った もの。切りぬいて なくても OK)を 添付**する。1まいに 2体(上の 段が 1体目、下の 段が 2体目)。\n"%len(hi))
pn=0
for ch in chunks(hi,2):
    pn+=1; key="P%02d"%pn; sheets[key]={"cols":4,"rows":len(ch),"ids":ch,"poses":["idle","atk","sp","win"]}
    body="\n".join("%s段: %s"%("上" if j==0 else "下",line(j+1,i)) for j,i in enumerate(ch))
    md.append("\n## %s(ポーズ・%d体×4)\n```\n添付画像の キャラたちと 同じ 絵柄・同じ 服・同じ 色で、ポーズだけ 変えて 描いて ください。\n%s\n横長の 画像に 横4×縦%dの マス目。1段に 1体、左から %s の 4ポーズ。顔の 向きと 体の 大きさは 4つで そろえる。\n%s\n%s\n```\n中身: %s\n"%(
        key,len(ch),STYLE,len(ch)," / ".join(POSE),body,TECH," / ".join(C[i]['n'] for i in ch)))
xn=len(xh)+len(xm)+len(xpairs)
md[1]="エイオン・オルレア・ニヒルは のぞく。全部で **%d体 / %dまい**(G01〜G%02d: 今 絵が ある %d体 / G%02d〜G%02d: まだ 絵が ない %d体)+ ポーズ分 P01〜P%02d(★5以上 %d体)。\n"%(len(ids)+xn,n,first,len(ids),first+1,n,xn,pn,len(hi))
open('/home/claude/artplan/ChatGPT発注.md','w',encoding='utf-8').write("".join(md))
json.dump(sheets,open('/home/claude/artplan/sheets.json','w',encoding='utf-8'),ensure_ascii=False)
print(len(ids),n,len(hum),len(mon),len(pairs))
