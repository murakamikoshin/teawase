"""必殺技の絵コンテ: キャラごとに 登場・カットイン・技(1〜4つ)・しめ の組み合わせと技名。83(エイオン)は hs_83.py、70(ニヒル)は Koshin 作の動画。"""
T = {  # id: (属性, 技名, 英語, 登場, カットイン, [技...], しめ, レア度)
 1:("fire","ヒヨコ火弾","Hiyoko Flare","drop","stamp",["meteor"],"white",1), 2:("water","雨ノ輪唱","Rain Chorus","rise","panels",["wave"],"iris",1),
 3:("leaf","若葉ノ舞","Leaf Dance","spin","band",["bloom"],"white",1), 4:("leaf","ドングリ連弾","Acorn Barrage","dash","stamp",["meteor"],"cut",1),
 5:("water","泡沫ノ渦","Bubble Vortex","drop","eye",["vortex"],"iris",1), 6:("fire","火走リ","Fire Dash","dash","band",["rush"],"white",2),
 7:("earth","岩甲落トシ","Shell Crush","rise","stamp",["crush"],"shatter",2), 8:("poison","胞子ノ霧","Spore Mist","spin","stamp",["vortex"],"iris",2),
 9:("star","一角流星","Horn Comet","spin","panels",["stars"],"white",2), 10:("fire","紅蓮一閃","Crimson Flash","dash","split",["slash"],"cut",3),
 11:("water","満チ潮","Rising Tide","rise","eye",["wave"],"white",3), 12:("leaf","森ノ影矢","Shadow Arrows","shadow","panels",["blades"],"iris",3),
 13:("petal","桜花居合","Blossom Draw","flash","split",["slash"],"shatter",4), 14:("ice","氷狼ノ牙","Frost Fang","dash","eye",["rush"],"shatter",4),
 15:("light","聖剣ノ誓","Oath of the Holy Sword","rise","band",["pillars","beam"],"white",5), 16:("earth","土ボコリ","Dust Burst","rise","panels",["pillars"],"cut",1),
 17:("wind","鎌風","Sickle Wind","spin","split",["slash"],"white",2), 18:("star","雷光双刃","Twin Bolt Blades","flash","stamp",["rush"],"cut",3),
 19:("dark","月読ノ帳","Veil of the Moon","shadow","eye",["orb"],"iris",4), 20:("fire","夜明ケノ王蹴","Dawnbreaker","drop","stamp",["slash","orb"],"white",5),
 31:("dark","叛逆ノ黒剣","Traitor's Edge","flash","band",["rush","crush"],"shatter",5), 32:("star","時止メノ杖","Chrono Halt","spin","eye",["stars","rings"],"iris",5),
 33:("wind","竜鱗ノ剣","Dragonscale Blade","dash","panels",["slash","beam"],"cut",5), 34:("dark","道化ノ千変","Trickster's Masque","spin","split",["blades","vortex"],"shatter",5),
 35:("wind","戦乙女ノ槍","Valkyrie Lance","drop","band",["pillars","rush"],"white",5), 36:("poison","八岐ノ顎","Eight-Headed Maw","rise","stamp",["meteor","vortex"],"iris",5),
 37:("light","月光ノ舞","Moonlight Waltz","drop","eye",["bloom","orb"],"white",5), 38:("ice","鎖断チノ咆哮","Chainbreaker Howl","flash","panels",["rings","crush"],"shatter",5),
 41:("light","モチモチ落トシ","Mochi Drop","drop","band",["crush"],"white",1), 42:("fire","小サナ灯","Little Flame","shadow","stamp",["orb"],"white",1),
 43:("leaf","木ノ実嵐","Nut Storm","spin","panels",["meteor"],"iris",1), 44:("wind","綿雲ノ渦","Cotton Vortex","rise","split",["vortex"],"white",1),
 55:("wind","疾風影分身","Gale Phantoms","dash","eye",["rush"],"cut",3), 56:("petal","鈴鳴リ神楽","Bell Kagura","drop","split",["rings"],"white",3),
 57:("earth","鉄拳連打","Iron Fist Rush","dash","stamp",["crush"],"shatter",3), 58:("petal","百花繚乱","Hundred Blossoms","spin","band",["bloom"],"iris",3),
 59:("earth","砂塵ノ刃","Sandstorm Edge","shadow","split",["blades"],"cut",3), 60:("ice","氷槍突キ","Ice Lance","rise","stamp",["beam"],"shatter",3),
 61:("ice","白狼雪月花","White Wolf Trinity","flash","eye",["slash"],"white",4), 62:("star","星読ミノ詩","Astral Verse","shadow","panels",["stars"],"white",4),
 63:("earth","不動ノ大盾","Immovable Aegis","drop","band",["rings"],"shatter",4), 64:("ice","水竜ノ舞","Water Dragon Dance","rise","split",["wave"],"iris",4),
 65:("light","日輪ノ裁キ","Solar Judgment","rise","stamp",["orb","pillars"],"white",5), 66:("dark","終焉ノ竜剣","Ragnarok Blade","drop","split",["beam","crush"],"shatter",5),
 67:("thunder","蒼天雷鳴","Azure Thunder","spin","band",["storm","beam"],"white",5), 68:("fire","紅蓮ノ天翔","Crimson Ascension","rise","eye",["bloom","meteor"],"white",5),
 69:("wind","白虎烈爪","White Tiger Claw","dash","panels",["slash","rings"],"cut",5),
 71:("light","創国ノ王笏","Kingdom's Genesis","drop","stamp",["pillars","stars","beam"],"white",6), 72:("thunder","天裂ク雷帝","Heavensplitter","flash","stamp",["storm","slash","crush"],"shatter",6),
 73:("earth","方舟ノ大洪水","Ark Deluge","rise","panels",["wave","pillars","orb"],"iris",6), 74:("dark","千刃地獄","Thousand Blade Hell","flash","split",["blades","rush","slash"],"cut",6),
 75:("leaf","世界樹ノ芽吹キ","Yggdrasil Bloom","rise","eye",["pillars","bloom","orb"],"white",6), 76:("water","海呑ミ","Ocean Devourer","spin","band",["wave","vortex","crush"],"iris",6),
 77:("dark","日食ノ王冠","Eclipse Crown","shadow","eye",["orb","stars","rings"],"shatter",6),
 78:("light","極光ノ天幕","Aurora Veil","rise","eye",["wave","stars","pillars"],"white",6), 80:("star","未ダ見ヌ明日","Unseen Tomorrow","shadow","split",["rings","orb","stars"],"shatter",6),
 232:("wind","旅ノ追イ風","Tailwind Ballad","spin","band",["rings"],"iris",3), 233:("water","荒波斬リ","Breaker Cutlass","flash","band",["slash"],"cut",3),
 234:("water","一本銛","Lone Harpoon","drop","stamp",["beam"],"white",3), 235:("fire","火花ノ大槌","Spark Hammer","flash","panels",["crush"],"shatter",3),
 236:("fire","纏振リ","Matoi Whirl","spin","stamp",["rings"],"white",3), 237:("ice","雪踏ミ","Snow Tread","drop","band",["wave"],"iris",3),
 238:("ice","氷像ノ牙","Ice Statue Fang","rise","panels",["blades"],"shatter",3), 239:("ice","ソリ突撃","Sled Charge","dash","stamp",["rush"],"white",3),
 240:("wind","砂塵ノ双舞","Dust Twin Dance","flash","eye",["slash"],"cut",3), 241:("earth","砂丘ノ槍","Dune Lance","shadow","split",["pillars"],"shatter",3),
 242:("wind","手作リ翼ノ急降下","Homemade Dive","drop","panels",["rush"],"cut",3), 243:("light","見習イノ祝福","Novice Blessing","rise","stamp",["stars"],"white",3),
 244:("light","獅子心ノ突撃","Lionheart Charge","dash","eye",["beam","pillars"],"shatter",4), 245:("leaf","千年樹ノ目覚メ","Elder Tree Wake","spin","band",["pillars","bloom"],"iris",4),
 246:("water","大艦隊一斉射","Grand Fleet Salvo","flash","split",["meteor","wave"],"shatter",4), 247:("fire","溶岩ノ調合","Magma Mixture","shadow","panels",["orb","meteor"],"white",4),
 248:("dark","冥界ノ門","Underworld Gate","shadow","eye",["vortex","orb"],"iris",4), 249:("star","千夜ノ灯","Thousand Nights Lamp","rise","band",["stars","rings"],"white",4),
 250:("wind","疾風ノ影斬リ","Gale Shadow Cut","dash","split",["slash","rush"],"cut",4), 251:("petal","花園ノ女王舞","Garden Queen Waltz","spin","panels",["bloom","rings"],"white",5),
 252:("water","大海原ノ三叉","Trident of the Deep","rise","split",["wave","beam"],"shatter",5), 253:("ice","氷河砕キ","Glacier Breaker","drop","stamp",["crush","blades"],"shatter",5),
 254:("light","天輪ノ裁キ","Halo Judgment","flash","band",["beam","rings"],"iris",5),
 81:("light","星冠ノ創世","Genesis Corona","rise","eye",["stars","bloom","pillars","orb"],"white",7), 82:("dark","終夜ノ葬列","Night's Requiem","shadow","split",["vortex","blades","storm","crush"],"shatter",7),
 101:("fire","炎冠ノ雄叫ビ","Blaze Crow","flash","band",["rings"],"white",3), 102:("water","大雨乞イ","Monsoon Call","drop","stamp",["storm"],"iris",3),
 103:("leaf","鱗粉ノ舞","Scale Waltz","spin","eye",["bloom"],"white",3), 104:("leaf","鉄ドングリ砲","Iron Acorn Cannon","dash","panels",["beam"],"cut",3),
 105:("water","千本針","Thousand Needles","rise","split",["blades"],"shatter",3), 106:("fire","狐火乱舞","Foxfire Frenzy","spin","stamp",["rush"],"white",3),
 107:("earth","城塞崩シ","Fortress Fall","drop","panels",["crush"],"shatter",3), 108:("poison","大胞子雲","Great Spore Cloud","shadow","band",["vortex"],"iris",3),
 109:("star","月角ノ祈リ","Moonhorn Prayer","rise","eye",["stars"],"white",3), 116:("earth","地割レ","Earthsplitter","rise","stamp",["pillars"],"shatter",3),
 117:("wind","双竜巻","Twin Cyclone","dash","split",["vortex"],"cut",3), 141:("light","鏡開キ","Kagami Break","drop","eye",["crush"],"white",3),
 142:("fire","篝火ノ輪","Bonfire Ring","shadow","panels",["rings"],"white",3), 143:("leaf","樫ノ大嵐","Oak Tempest","flash","band",["meteor"],"iris",3),
 144:("thunder","雷雲メェ雷","Thunder Fleece","spin","split",["storm"],"shatter",3),
}
# ガチャ限定: 技名は gacha80.py、流れは id から決める(同じ流れが かぶらないように ずらす)
import gacha80
_I=["drop","dash","rise","spin","flash","shadow"]; _C=["band","eye","split","stamp","panels"]; _F=["white","shatter","iris","cut"]
_A={"fire":["meteor","beam","slash","rings","orb"],"water":["wave","vortex","beam","orb","rush"],"leaf":["bloom","blades","pillars","orb","rush"],"earth":["crush","pillars","rings","rush","meteor"],
    "wind":["slash","vortex","rush","blades","rings"],"thunder":["storm","beam","slash","rush","crush"],"ice":["blades","beam","wave","stars","slash"],"dark":["vortex","blades","orb","crush","storm"],
    "light":["pillars","stars","beam","bloom","orb"],"star":["stars","rings","orb","meteor","beam"],"petal":["bloom","slash","rings","blades","rush"],"poison":["vortex","meteor","blades","orb","storm"]}
for _k,_g in enumerate(gacha80.G):
    _n={3:1,4:1,5:2,6:3}[_g[2]]; _a=_A[_g[4]]
    T[_g[0]]=(_g[4],_g[7],_g[8],_I[(_k*5+_g[0])%6],_C[(_k*3+_k//5)%5],[_a[(_k+j*2)%5] for j in range(_n)],_F[(_k+_k//4)%4],_g[2])
for _i,_new in ((425,"shadow"),(433,"rise"),(403,"drop")):
    _t=list(T[_i]); _t[3]=_new; T[_i]=tuple(_t)
DUR = {1:(1.4,1.9,2.0,1.1), 2:(1.4,1.9,2.0,1.1), 3:(1.5,2.0,2.3,1.2), 4:(1.7,2.1,2.8,1.3), 5:(1.8,2.2,2.2,1.4), 6:(2.0,2.3,2.1,1.5), 7:(2.2,2.4,2.1,1.6)}
BOARD = {}
for i, (el, nm, en, intro, cut, atks, fin, r) in T.items():
    d = DUR[r]
    BOARD[i] = {"elem": el, "name": nm, "en": en, "r": r, "seq": [("i_" + intro, d[0]), ("c_" + cut, d[1])] + [("a_" + a, d[2]) for a in atks] + [("f_" + fin, d[3])]}
if __name__ == "__main__":
    seen = {}
    for i, b in BOARD.items():
        k = tuple(n for n, _ in b["seq"]); seen.setdefault(k, []).append(i)
    print(len(BOARD), "boards;", "same flow:", [v for v in seen.values() if len(v) > 1])
