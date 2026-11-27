# -*- coding: utf-8 -*-
"""151体の名簿と、7エリア・レジェンド枠を ゲームに入れる"""
import json
F='/home/claude/teawase/index.html'
s=open(F).read()
def rep(o,n,c=1):
    global s
    assert s.count(o)==c,(s.count(o),o[:90]); s=s.replace(o,n)

# ---- 追加の野生(54体): id,名前,レア,手,エリア,モチーフ,ひとこと ----
WILD=[
 (201,"ミノムシン",1,"rock","forest","みのむし × 葉っぱのコート","コートを ぬがされると すごく はずかしがる。"),
 (202,"フクロン",1,"paper","forest","ふくろうの子 × お月見","首が ぐるっと 回る。夜ふかしが とくい。"),
 (203,"コガネン",1,"scissors","forest","こがね虫 × どんぐりのコイン","ぴかぴかの物を 見つけると 巣に もちかえる。"),
 (204,"クモイト",1,"scissors","forest","子ぐも × あやとり","糸で もようを つくるのが すき。ほめると のびる。"),
 (205,"カニパン",1,"rock","sea","かに × あんぱん","こうらが ふかふか。横にしか 歩けないのを 気にしている。"),
 (206,"ヒトデン",1,"paper","sea","ひとで × 星のクッキー","夜に なると 海の底で ぼんやり 光る。"),
 (207,"マグマル",1,"rock","volcano","溶岩の しずく","さわると あつい。ひえると 石になって ねむる。"),
 (208,"ケムリン",1,"paper","volcano","けむり × えんとつ","もくもく ふくらむ。くしゃみで ちぢむ。"),
 (209,"ユキダマ",1,"rock","snow","雪玉 × 坂道","ころがるほど 大きくなる。止まりかたは 知らない。"),
 (210,"ペンコロ",1,"scissors","snow","ペンギンの子 × 氷のそり","おなかで すべる。坂を 見ると がまん できない。"),
 (211,"ツララン",1,"scissors","snow","つらら × 朝日","朝日を あびると きらきら 光る。昼は 元気がない。"),
 (212,"シモバシラ",1,"paper","snow","霜柱の精 × 冬の土","ふまれると さくっと 鳴く。それが じまん。"),
 (213,"スナネコ",1,"scissors","desert","砂ねこ × 砂時計","足あとを のこさずに 歩く。昼は 砂に もぐる。"),
 (214,"ラクダマ",1,"rock","desert","らくだの子 × こぶ","こぶに おやつを ためている。中身は ひみつ。"),
 (215,"サソリン",1,"scissors","desert","さそり × はさみ","はさみを ふって あいさつする。こわがられて かなしい。"),
 (216,"タンブル",1,"paper","desert","ころがる草 × 風","風まかせで 旅をする。行き先は 自分でも 知らない。"),
 (217,"ホシクズ",1,"paper","tower","星くずの子","流れ星から こぼれた。ねがいごとを 聞くのが すき。"),
 (218,"ミツバチョ",2,"scissors","field","みつばち × はちみつの壺","壺を かかえて 飛ぶ。なめると すごく おこる。"),
 (219,"カブトムサ",2,"rock","forest","かぶとむし × 武者","つので 一騎打ちを いどむ。負けると 木の かげで なく。"),
 (220,"イカスミ",2,"scissors","sea","いか × 墨の忍者","すみを はいて すがたを けす。自分も 見えなくなる。"),
 (221,"サンゴロウ",2,"paper","sea","さんご × 職人","百年かけて 枝を のばす。せっかちが きらい。"),
 (222,"ヒバナトカゲ",2,"scissors","volcano","とかげ × 火花","しっぽを ふると 火花が ちる。夜は 花火みたい。"),
 (223,"ガンセキ",2,"rock","volcano","岩の かめ","火山の岩に まぎれている。動くのは 10年に 1回。"),
 (224,"フイゴン",2,"paper","volcano","ふいご × 鍛冶場","息を ふきこむと 火が 大きくなる。おだてに 弱い。"),
 (225,"ヨウガンヘビ",2,"scissors","volcano","へび × 溶岩の川","溶岩の川を およぐ。ぬけがらは 石になる。"),
 (226,"シロクマル",2,"rock","snow","白くま × かまくら","かまくらを 作って 客を まつ。だれも 来ないと すねる。"),
 (227,"オコジョン",2,"scissors","snow","おこじょ × 吹雪","吹雪の中を 走りぬける。夏は 茶色くなる。"),
 (228,"コオリバナ",2,"paper","snow","氷の花","いちばん 寒い朝にだけ さく。さわると とける。"),
 (229,"ツボヘビ",2,"paper","desert","つぼのコブラ × 笛","笛の音で おどる。へたな笛だと 出てこない。"),
 (230,"スカラベン",2,"rock","desert","ふんころがし × 太陽","まるい物を ころがして 太陽を はこぶ つもり。"),
 (231,"ハネウマ",2,"paper","tower","つばさの 子馬","まだ うまく 飛べない。雲の上で れんしゅう中。"),
 (232,"ポルカ",3,"paper","field","旅の吟遊詩人 × リュート","歌で 相手の 次の一手を 聞きだす。"),
 (233,"マリン",3,"scissors","sea","海賊の 少女船長","宝の地図は ぜんぶ 頭の中。船は まだ ない。"),
 (234,"ウシオ",3,"rock","sea","素もぐりの 銛使い","息を 5分 止められる。魚より 先に およぐ。"),
 (235,"テッサ",3,"rock","volcano","鍛冶の少女 × 大づち","火山の火で 剣を うつ。失敗作は ひとつも ない。"),
 (236,"グレン",3,"paper","volcano","火消しの 若頭 × まとい","どんな火も こわくない。火事と きけば 走る。"),
 (237,"ユキノ",3,"paper","snow","雪国の かんじき娘","ふぶきの中でも 道に まよわない。"),
 (238,"トウヤ",3,"scissors","snow","氷の 彫刻家 × のみ","ひとふりで 氷を 花に かえる。"),
 (239,"ニコラ",3,"rock","snow","そり引きの 少年","だれよりも 先に 雪原を かけぬける。"),
 (240,"ナディア",3,"scissors","desert","砂漠の 踊り子 × 双剣","おどりながら 斬る。見とれると 負ける。"),
 (241,"ラシード",3,"rock","desert","隊商の 護衛 × 槍","砂嵐の中でも 荷を 守りぬく。"),
 (242,"イカロ",3,"paper","tower","空とぶ 発明家 × つばさ","自作の つばさで 塔を めざす。落ちても こりない。"),
 (243,"スピカ",3,"scissors","tower","星の 見習い天使","輪っかが まだ 小さい。おつかいで 地上に 来た。"),
 (244,"レオン",4,"rock","field","獅子心の 騎士団長","最初の ひとたちで 勝負を きめる。"),
 (245,"エルナ",4,"paper","forest","千年樹の 精霊使い","木々の こえを 聞いて 先を 読む。"),
 (246,"ドレーク",4,"rock","sea","大海の 提督","七つの海を わたった。さいごに 笑うのは いつも この人。"),
 (247,"イグニス",4,"paper","volcano","火山の 錬金術師","溶岩から 金を つくる。まだ 成功は 3回だけ。"),
 (248,"アヌ",4,"rock","desert","王墓の 守護者","千年 ねむらずに 墓を 守っている。"),
 (249,"シェラ",4,"paper","desert","千夜の 語り部","話を きいていると 時間が ゆっくりになる。"),
 (250,"ゼフィ",4,"scissors","tower","西風の 剣聖","風より はやく ぬいて 風より しずかに おさめる。"),
 (251,"ティタニア",5,"paper","forest","妖精の女王 × 真夏の夜","森の 花は みんな この人に おじぎする。"),
 (252,"ワダツミ",5,"scissors","sea","海の神ワダツミ × 竜宮","潮の 満ち引きを 指一本で かえる。"),
 (253,"フリムニル",5,"rock","snow","北欧の 霜の巨人","一歩で 山を こえる。足あとが 湖になった。"),
 (254,"ソラリス",5,"rock","tower","天空の 機神","塔の てっぺんで 千年 止まっていた。いま 動きだす。"),
]
EVO_NEW={201:"オオミノガ",205:"ヨロイガニ",207:"マグマッチョ",209:"ナダレオウ",213:"スナジシ"}
LIM_NEW=[(36,"ヤマタ",5,"rock","日本神話の 八つ首の大蛇","八つの首が それぞれ ちがう ことを 考えている。"),
         (37,"カグヤ",5,"paper","竹取物語 × 月の姫","月から 来て、月へ 帰る。その間だけ 会える。"),
         (38,"フェンリス",5,"scissors","北欧神話の 鎖の狼","どんな 鎖も かみちぎる。まだ 本気では ない。")]
PAID_NEW=[(68,"スザク",5,"rock","四神の朱雀 × 南の空","南の空を まもる 炎の鳥。羽は もえても へらない。"),
          (69,"ビャッコ",5,"scissors","四神の白虎 × 西の空","西の空を まもる 白い虎。ひとこえで 風が 止まる。")]
LEG=[(71,"ゼノン",6,"paper","はじまりの王","この世界で 最初に なかまを つくった と 言われる。"),
     (72,"アマツ",6,"rock","天を裂く 雷帝","ひとたび 剣を ぬけば 空が 二つに われる。"),
     (73,"ノア",6,"paper","方舟の 守り人","すべての 生きものの 名前を おぼえている。"),
     (74,"クロガネ",6,"scissors","千の刃の 鬼神","千本の 刀を 同時に あやつる。本人は 一本も 持たない。"),
     (75,"ユグド",6,"paper","世界樹の 化身","根は 地の底、枝は 星まで とどく。"),
     (76,"バハル",6,"rock","海を呑む 竜","背中に 島が ひとつ のっている。"),
     (77,"エクリプス",6,"scissors","日食の 魔王","あらわれると 昼が 夜になる。"),
     (78,"オーロラ",6,"paper","極光の 女神","夜空の 光の カーテンは この人の 衣の すそ。"),
     (79,"ハジマリ",6,"rock","さいしょの ひとしずく","すべての 子の はじまり。おく5より 先にしか いない。"),
     (80,"ミライ",6,"scissors","まだ見ぬ者","ほかの 全員と なかまになった者の 前にだけ あらわれる。")]
TITLES={232:"旅の吟遊詩人",233:"海賊の少女船長",234:"素もぐりの銛使い",235:"鍛冶の少女",236:"火消しの若頭",237:"雪国のかんじき娘",238:"氷の彫刻家",239:"そり引きの少年",
 240:"砂漠の踊り子",241:"隊商の護衛",242:"空とぶ発明家",243:"見習い天使",244:"獅子心の騎士団長",245:"千年樹の精霊使い",246:"大海の提督",247:"火山の錬金術師",
 248:"王墓の守護者",249:"千夜の語り部",250:"西風の剣聖",251:"妖精の女王",252:"竜宮の王",253:"氷河の巨人王",254:"天空の機神",
 36:"八つ首の大蛇",37:"月の姫",38:"鎖を噛む狼",68:"紅蓮の朱雀",69:"白金の白虎",
 71:"はじまりの王",72:"天を裂く雷帝",73:"方舟の守り人",74:"千の刃の鬼神",75:"世界樹の化身",76:"海を呑む竜",77:"日食の魔王",78:"極光の女神",79:"さいしょのひとしずく",80:"まだ見ぬ者"}
VOICE={232:"一曲、どうだい？",233:"宝は ぜんぶ あたしのもの！",234:"もぐるぞ。ついてこい。",235:"うちの剣、ためしてみる？",236:"火事は どこでい！",237:"道なら まかせて。",238:"うごかないで。いま ほるから。",239:"先に 行ってるよ！",
 240:"目を はなさないで。",241:"荷は わたさん。",242:"こんどこそ 飛ぶ！",243:"おつかいの とちゅう なの。",244:"一番槍は、わたしだ。",245:"木々が、教えてくれたわ。",246:"さいごに 笑うのは わしだ。",247:"金に なれ…なれ…！",
 248:"千年、ここに いる。",249:"むかしむかし…",250:"風が、通っただけだ。",251:"花よ、ひらきなさい。",252:"潮よ、したがえ。",253:"…小さいな。",254:"システム、再起動。",
 36:"どの首と 話したい？",37:"月が きれいですね。",38:"鎖は、もう ない。",68:"南の空は、わたしが まもる。",69:"西の空は、オレが まもる。",
 71:"はじめまして、だね。",72:"天よ、ひらけ。",73:"きみの 名前も おぼえたよ。",74:"千本、ぬいた。",75:"…根は、すべてに つながる。",76:"海が、足りぬ。",77:"昼は、おわりだ。",78:"夜空を、見上げて。",79:"…ぽたり。",80:"やっと、会えたね。"}
SKN={232:("先読みの歌","次のターンの 相手の手が見える",{"peek":1}),233:("荒波","さいごのターンに出すと 1.6倍",{"last":1.6}),234:("ひと突き","相手の体力が 半分以下なら 1.5倍",{"lowHp":1.5}),
 235:("たたき上げ","はずしても 6割の力は出る",{"floor":0.6}),236:("火消し","不利な手でも 等倍で通る",{"neutral":1}),237:("雪道","ゲージが ゆっくりになる",{"slow":0.6}),
 238:("ひとふり","PERFECTなら さらに1.5倍",{"perfK":1.5}),239:("一番のり","1ターン目に出すと 1.5倍",{"turn":1,"tk":1.5}),240:("舞い","当たりの広さが 1.5倍",{"zone":1.5}),
 241:("護衛","SSR以上の相手に 1.3倍",{"vsHigh":1.3}),242:("発明","いつ出しても 1.2倍",{"any":1.2}),243:("おいのり","PERFECT 1回が 2回ぶんに数えられる",{"perf2":1}),
 244:("一番槍","相手の体力が満タンなら 1.5倍",{"full":1.5}),245:("木々の声","次の手が見える。いつでも 1.15倍",{"peek":1,"any":1.15}),246:("大海原","さいごのターンに出すと 1.8倍",{"last":1.8}),
 247:("錬金","PERFECTなら さらに1.6倍",{"perfK":1.6}),248:("不眠","はずしても 7割の力は出る",{"floor":0.7}),249:("千夜","ゲージが とても ゆっくり",{"slow":0.5}),
 250:("西風","当たり1.6倍。いつでも 1.1倍",{"zone":1.6,"any":1.1}),251:("真夏の夜","いつ出しても 1.35倍",{"any":1.35}),252:("満ち引き","不利な手のとき 逆に1.5倍",{"reverse":1.5}),
 253:("霜の一歩","相手の体力が満タンなら 1.8倍",{"full":1.8}),254:("再起動","PERFECTなら さらに1.8倍",{"perfK":1.8}),
 36:("八つ首","SSR以上の相手に 1.5倍",{"vsHigh":1.5}),37:("月あかり","次の手が見える。PERFECTで1.4倍",{"peek":1,"perfK":1.4}),38:("鎖ちぎり","不利な手のとき 逆に1.6倍",{"reverse":1.6}),
 68:("朱雀","いつでも1.3倍。さいごのターンは さらに1.3倍",{"any":1.3,"last":1.3}),69:("白虎","いつでも1.3倍。PERFECTで さらに1.3倍",{"any":1.3,"perfK":1.3}),
 71:("はじまり","いつ出しても 1.5倍",{"any":1.5}),72:("雷帝","PERFECTなら 2倍",{"perfK":2}),73:("方舟","はずしても 8割。いつでも1.3倍",{"floor":0.8,"any":1.3}),
 74:("千刃","当たり2倍。いつでも1.3倍",{"zone":2,"any":1.3}),75:("世界樹","次の手が見える。いつでも1.4倍",{"peek":1,"any":1.4}),76:("大渦","相手の体力が満タンなら 2倍",{"full":2}),
 77:("日食","不利な手のとき 逆に2倍",{"reverse":2}),78:("極光","ゲージが とてもゆっくり。いつでも1.3倍",{"slow":0.45,"any":1.3}),79:("ひとしずく","さいごのターンに出すと 2.2倍",{"last":2.2}),
 80:("未来","いつ出しても 1.6倍。PERFECT 1回が 2回ぶん",{"any":1.6,"perf2":1})}
# 既存の野生の エリア
AREA_OLD={1:"field",4:"field",16:"field",41:"field",43:"field",47:"field",9:"field",6:"field",48:"field",57:"field",60:"field",63:"field",15:"field",
 3:"forest",8:"forest",17:"forest",50:"forest",12:"forest",55:"forest",58:"forest",19:"forest",
 2:"sea",5:"sea",46:"sea",7:"sea",51:"sea",11:"sea",64:"sea",
 42:"volcano",45:"volcano",10:"volcano",13:"volcano",20:"volcano",
 49:"snow",14:"snow",61:"snow",53:"desert",59:"desert",65:"desert",
 44:"tower",52:"tower",54:"tower",18:"tower",56:"tower",62:"tower",66:"tower"}
def js(o): return json.dumps(o,ensure_ascii=False,separators=(',',':'))
area_of=dict(AREA_OLD); area_of.update({w[0]:w[4] for w in WILD})

# ---- 名簿 ----
rep(''' [65,"ヒミカ",5,"paper"],[66,"ラグナ",5,"rock"]
];''',''' [65,"ヒミカ",5,"paper"],[66,"ラグナ",5,"rock"]
].concat('''+js([[w[0],w[1],w[2],w[3]] for w in WILD])+''');
var AREA_OF='''+js({str(k):v for k,v in area_of.items()})+''';   // どのエリアに すんでいるか''')
rep('var LIMS_PLUS=[[34,"ロキ",5,"scissors"],[35,"ヴァルキリア",5,"paper"]];','var LIMS_PLUS=[[34,"ロキ",5,"scissors"],[35,"ヴァルキリア",5,"paper"]].concat('+js([[l[0],l[1],l[2],l[3]] for l in LIM_NEW])+');')
rep('var PAID_PLUS=[[67,"セイリュウ",5,"scissors"]];','var PAID_PLUS=[[67,"セイリュウ",5,"scissors"]].concat('+js([[l[0],l[1],l[2],l[3]] for l in PAID_NEW])+');\nvar LEG_PLUS='+js([[l[0],l[1],l[2],l[3]] for l in LEG])+';     // レジェンド(★6)。ふつうの抽選とは別に ごくまれに出る\nvar LEG_RATE=0.002;   // 1回の出会いごとに 0.2%(500回に1回)')
rep('var EVO_NAMES={41:"カガミモチオウ",','var EVO_NAMES={'+','.join('%d:"%s"'%(k,v) for k,v in EVO_NEW.items())+',41:"カガミモチオウ",')
rep('''var ALL=M.concat(EVOS,LIMS,PAID);''','''var LEGS=LEG_PLUS.filter(function(a){return hasArt(a[0]);}).map(function(a){return {id:a[0],name:a[1],r:6,hand:a[3],leg:true};});
var ALL=M.concat(EVOS,LIMS,PAID,LEGS);''')
rep('''  var all=M.concat(EVOS,LIMS,PAID), per=''','''  var all=M.concat(EVOS,LIMS,PAID,LEGS), per=''')
rep('''  var T={67:"蒼天の竜王",''','''  var T='''+js({str(k):v for k,v in TITLES.items()})+''', T2={67:"蒼天の竜王",''')
rep('''  for(var k in T){HERO[k]={t:T[k],w:"sword",hair:"#888",armor:"#666",trim:"#f2b632",cape:"#333",blade:"#fff"};}''','''  for(var k2 in T2){T[k2]=T2[k2];}
  for(var k in T){HERO[k]={t:T[k],w:"sword",hair:"#888",armor:"#666",trim:"#f2b632",cape:"#333",blade:"#fff"};}''')
lore={str(w[0]):[w[5],w[6]] for w in WILD}
lore.update({str(l[0]):[l[4],l[5]] for l in LIM_NEW+PAID_NEW+LEG})
rep(''' 67:["四神の青龍 × 東の空",''',''' '''+js(lore)[1:-1]+''',
 67:["四神の青龍 × 東の空",''')
rep('''var SKX={67:{any:1.3,zone:1.5},''','''var SKX2='''+js({str(k):v[2] for k,v in SKN.items()})+''';
var SKX={67:{any:1.3,zone:1.5},''')
rep('''  for(var k in N){SK[k]=N[k];}''','''  var N2='''+js({str(k):[v[0],v[1]] for k,v in SKN.items()})+''';
  for(var k in N){SK[k]=N[k];}
  for(var k3 in N2){SK[k3]=N2[k3];SKX[k3]=SKX2[k3];}''')
rep('''  if(x.any){k*=x.any;}''','''  if(x.any){k*=x.any;}
  if(x.full&&E.dmg===0){k*=x.full;}''')
rep(''' 67:"東の空は、わたしが まもる。",''',''' '''+js({str(k):v for k,v in VOICE.items()})[1:-1]+''',
 67:"東の空は、わたしが まもる。",''')

# ---- ★6(レジェンド) ----
rep('--r4:#c66bff; --r5:#ffcf4a;','--r4:#c66bff; --r5:#ffcf4a; --r6:#ff4fd8;')
rep('.rkl{--rc:#ff5a7a;','.rk6{--rc:var(--r6);--rg:linear-gradient(135deg,#ffffff,#ff4fd8 22%,#6ae3ff 44%,#fff36a 66%,#ff6a3a 84%,#b24dff)}\n.rkl{--rc:#ff5a7a;')
rep('var HP=[60,115,200,310,460];','var HP=[60,115,200,310,460,640];')
rep('var PW=[20,28,38,50,65];','var PW=[20,28,38,50,65,84];')
rep('var BR=[0.92,0.72,0.50,0.27,0.11];','var BR=[0.92,0.72,0.50,0.27,0.11,0.05];')
rep('function stars(r){return "★★★★★".slice(0,r);}','function stars(r){return "★★★★★★".slice(0,r);}')
rep('var RK=["N","R","SR","SSR","UR"];','var RK=["N","R","SR","SSR","UR","LEGEND"];')
rep('''col=["#9aa3b5","#3fbf6b","#3f8fe8","#a35cf0","#f2b632"][m.r-1];''','''col=["#9aa3b5","#3fbf6b","#3f8fe8","#a35cf0","#f2b632","#ff4fd8"][m.r-1];''')

# ---- 7エリア ----
a=s.index('var AREAS=['); b=s.index('];',a)+2
s=s[:a]+'''var AREAS=[
 {id:"field",name:"はらっぱ",f:0,w:[50,28,15,5.5,1.5]},
 {id:"forest",name:"よるの森",f:0.08,w:[38,32,20,8,2]},
 {id:"sea",name:"うみべ",f:0.16,w:[32,33,23,9.5,2.5]},
 {id:"volcano",name:"かざん",f:0.26,w:[24,33,27,12.5,3.5]},
 {id:"snow",name:"ゆきやま",f:0.38,w:[16,32,31,16,5]},
 {id:"desert",name:"さばく",f:0.50,w:[8,30,35,20,7]},
 {id:"tower",name:"てんくうの塔",f:0.62,w:[0,26,40,25,9]}
];'''+s[b:]
rep('var ALL=M.concat(EVOS,LIMS,PAID,LEGS);','var ALL=M.concat(EVOS,LIMS,PAID,LEGS);\nAREAS.forEach(function(a){a.need=Math.round(ALL.length*a.f);});   // 解放に必要な種類は 全体の数に合わせる')
rep('var AREA_HP={field:1,forest:1.35,tower:1.8};','var AREA_HP={field:1,forest:1.25,sea:1.4,volcano:1.5,snow:1.6,desert:1.7,tower:1.8};')
# 出会いの抽選: そのエリアに すむ子から(いなければ 同じレア度の だれか)。レジェンドは別枠
rep('''  var pool=M.filter(function(m){return m.r===r;});
  return pool[Math.floor(Math.random()*pool.length)];
}''','''  if(!force&&LEGS.length){
    var lg=LEGS.filter(function(m){
      if(m.id===79){return run.depth>=4;}                                            // ハジマリ: おく5より先
      if(m.id===80){return ALL.every(function(q){return q.id===80||count(q.id)>0;});} // ミライ: ほかの全員が なかま
      return true;
    });
    if(lg.length&&Math.random()<LEG_RATE*(area().id==="tower"?2:1)){return lg[Math.floor(Math.random()*lg.length)];}
  }
  var aid=area().id;
  var pool=M.filter(function(m){return m.r===r&&AREA_OF[m.id]===aid;});
  if(!pool.length){pool=M.filter(function(m){return m.r===r;});}
  return pool[Math.floor(Math.random()*pool.length)];
}''')
# 乱入は「いまの相手より 上」だけ
rep('''  var pool=Math.random()<0.5?LIMS:M.filter(function(m){return m.r>=4;});''','''  var cur=byId(E.mid);
  if(cur.lim||cur.leg){return;}                                  // 限定やレジェンドの上は いない
  var hi=M.filter(function(m){return m.r>=4&&m.r>cur.r;});        // いまの相手より レア度が上の子だけ
  var pool=(cur.r>=5||!hi.length||Math.random()<0.5)?LIMS:hi;''')
# エリアえらび: 1つを大きく見せて 左右で切りかえ(7つあっても スクロールなし)
a=s.index('''  h+='<div class="areas">';'''); b=s.index('''  h+='<div class="pline">''')
s=s[:a]+'''  var opens=AREAS.filter(function(x){return kinds()>=x.need;}), ai=opens.indexOf(area()), nextA=AREAS[opens.length];
  h+='<div class="areasel"><button data-act="areamv" data-id="-1" id="area-prev"'+(opens.length<2?" disabled":"")+'>‹</button>'+
     '<div class="acard" style="background-image:url('+areaBg(area().id)+')"><b>'+area().name+'</b><span>'+(ai+1)+' / '+AREAS.length+'・UR '+area().w[4]+'%'+
     (nextA?'　つぎ：'+nextA.name+'（なかま'+nextA.need+'種）':'')+'</span></div>'+
     '<button data-act="areamv" data-id="1" id="area-next"'+(opens.length<2?" disabled":"")+'>›</button></div>';
'''+s[b:]
rep('''  else if(act==="area"){''','''  else if(act==="areamv"){
    var ops=AREAS.filter(function(x){return kinds()>=x.need;}), ix=ops.indexOf(area());
    S.area=ops[(ix+Number(b.getAttribute("data-id"))+ops.length)%ops.length].id; save(); beep(700,0.06); renderHome();
  }
  else if(act==="area"){''')
rep('#enc{overflow:hidden}','''.areasel{display:grid;grid-template-columns:44px 1fr 44px;gap:7px;align-items:stretch}
.areasel>button{border-radius:12px;background:linear-gradient(180deg,#4a3bb8,#1c1558);border:1px solid var(--gold2);color:#fff;font-size:22px;line-height:1}
.areasel>button:disabled{opacity:.3}
.acard{height:46px;border-radius:12px;border:2px solid var(--gold2);background-size:cover;background-position:center 62%;image-rendering:pixelated;display:flex;flex-direction:column;justify-content:center;padding:0 12px;box-shadow:inset 0 0 0 100px rgba(8,6,32,.42),0 0 14px -4px var(--gold2);overflow:hidden}
.acard b{font-family:var(--display);font-weight:400;font-size:17px;line-height:1.15;text-shadow:0 2px 0 rgba(0,0,0,.7)}
.acard span{font-size:10px;font-weight:900;color:#e8e5ff;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;text-shadow:0 1px 2px #000}
#enc{overflow:hidden}''')
# 背景: 新しい4エリア
rep('''  }else{
    band(["#0a0420","#23105c","#5a2fc4","#b06ae0","#ffd08a"],0,92);''','''  }else if(id==="sea"){
    band(["#1b5fc4","#3f9ae8","#8fd0f5","#ffe9b8"],0,62);
    disc(92,20,8,"#fff7d0"); cloud(14,16,26,"#ffffff"); cloud(58,30,20,"#f2fbff");
    band(["#2f8fd6","#1f6fc0","#17509a"],62,90);
    dots(40,64,88,["#bfe9ff","#ffffff","#8fd0f5"]);
    band(["#f2d9a0","#e8c684","#d9b06a"],90,H);
    dots(26,92,H,["#fff3d0","#c99a58","#ff9a8a"]);
  }else if(id==="volcano"){
    band(["#1a0608","#4a0f12","#a8281b","#f07a2a"],0,70);
    dots(30,0,40,["#ffb347","#ff6a2a","#ffe14a"]);
    hill(74,3,23,"#2a1014",0.5);
    c.fillStyle="#2a1014"; for(var vy=0;vy<46;vy++){c.fillRect(60-Math.floor(vy*0.75),28+vy,Math.floor(vy*1.5)+6,1);}
    c.fillStyle="#ffb347"; c.fillRect(60,24,6,6); c.fillStyle="#ff5a2a"; c.fillRect(62,30,2,26); c.fillRect(58,44,2,14); c.fillRect(66,40,2,20);
    c.fillStyle="#170a0c"; c.fillRect(0,84,W,H-84);
    dots(34,86,H,["#ff6a2a","#ffb347","#4a1a12"]);
  }else if(id==="snow"){
    band(["#0f2a5a","#2f5fa8","#8fbfe8","#e8f6ff"],0,66);
    dots(70,0,70,["#ffffff","#e8f6ff"]);
    hill(62,7,29,"#b8d4ee",1.2); hill(72,5,19,"#dcecfa",0.3);
    for(var si=0;si<7;si++){tree(8+si*17+Math.floor(r()*6),84,18+Math.floor(r()*8),"#2a5a6a");}
    c.fillStyle="#f4faff"; c.fillRect(0,84,W,H-84);
    dots(30,86,H,["#cfe4f6","#ffffff","#a8c8e6"]);
  }else if(id==="desert"){
    band(["#f07a2a","#f2a84a","#ffd98a","#fff1c9"],0,60);
    disc(30,24,10,"#fff7d0"); disc(30,24,8,"#ffffff");
    c.fillStyle="#c98a3a"; for(var py=0;py<26;py++){c.fillRect(84-py,36+py,py*2+1,1);} 
    c.fillStyle="#b0742c"; for(py=0;py<16;py++){c.fillRect(58-py,46+py,py*2+1,1);}
    hill(66,4,31,"#e8b45a",0.8); hill(78,5,21,"#d99a44",2.1);
    c.fillStyle="#c9883a"; c.fillRect(0,92,W,H-92);
    dots(40,80,H,["#f2c878","#b0742c","#ffe0a0"]);
  }else{
    band(["#0a0420","#23105c","#5a2fc4","#b06ae0","#ffd08a"],0,92);''')
# ガイド: レジェンドの確率を正直に書く
rep('''<br>予告：画面がふるえたら''','''<br>レジェンド（★6）：どのエリアでも <b>'+(LEG_RATE*100)+'%</b>（塔は2倍）。天井の対象外<br>予告：画面がふるえたら''')
rep('''  for(var r=1;r<=5;r++){rates+=''','''  for(var r=1;r<=5;r++){rates+=''')
open(F,'w').write(s)
tot=46+len(WILD)
print('wild',tot,'evo',25+len(EVO_NEW),'lim',5+len(LIM_NEW),'paid',1+len(PAID_NEW),'leg',len(LEG),'TOTAL',tot+25+len(EVO_NEW)+5+len(LIM_NEW)+1+len(PAID_NEW)+len(LEG))
