# -*- coding: utf-8 -*-
"""ChatGPT発注.md から「1回 貼って、あとは『次』で 進める」一括プロンプトを 作る → artplan/ChatGPT一括.md"""
import re,json
md=open('/home/claude/artplan/ChatGPT発注.md',encoding='utf-8').read()
blocks=re.findall(r'## (G\d\d|P\d\d)\(([^)]*)\)\n```\n(.*?)\n```',md,re.S)
B={k:(h,b) for k,h,b in blocks}
g01=B['G01'][1].split('\n')
STYLE=[l for l in g01 if l.startswith('日本の')][0]
TECH=[l for l in g01 if l.startswith('背景は')][0]
def sheet(k):
    h,b=B[k]; L=[l for l in b.split('\n') if not (l.startswith('日本の') or l.startswith('背景は') or l.startswith('添付画像') or l.startswith('人間では'))]
    mon='モンスター' in h
    return "■ %s(%s)%s\n%s"%(k,h,"  ※人間では なく モンスター(指定した モチーフの 生きもの)" if mon else "","\n".join(L))
def batch(keys,title,attach):
    head=("これから キャラクター表を %dまい 続けて 作って もらいます。\n"
      "%s\n"
      "【ぜんぶの シートに 共通の ルール】\n%s\n%s\n"
      "【進め方】1回の 返事で 作る 画像は 1まいだけ。画像の 前に シート名(%s など)を 1行 書く。私が「次」と 言ったら リストの 次の シートを 作る。ルールは 毎回 守る(特に 背景透明・腕と手と足は 2本・指 5本・向きを まぜる・浮いた エフェクトなし)。毎回 出す 前に 全員の 腕・手・足・指の 数を 数えて 確かめる。\n"
      "まずは %s を 作って ください。\n\n【シート一覧】\n")%(len(keys),attach,STYLE,TECH,keys[0],keys[0])
    return "## %s\n```\n%s%s\n```\n"%(title,head,"\n\n".join(sheet(k) for k in keys))
ks=sorted(B)
G=[k for k in ks if k.startswith('G')]; P=[k for k in ks if k.startswith('P')]
out=["# ChatGPT 一括プロンプト\n",
"新しい 会話で 1回 貼る → あとは「次」と 打つだけ。**気に入った G01 の 絵を 1まい 添付**して 貼る。途中で 絵柄が ずれたら「最初の 添付画像の 絵柄に もどして」と 言う。\n",
"できた 画像は シート名と いっしょに こっちへ 貼って くれれば 取りこむ(多少 変でも キャラとして おかしく なければ OK)。\n"]
out.append(open("/home/claude/artplan/fix1.md",encoding="utf-8").read())
out.append(batch([k for k in G if 'G09'<=k<='G22'],"一括A: G09〜G22(今 絵が ある 子の のこり)","添付画像(G01)と 同じ 絵柄で 描いて ください。添付画像の キャラ自体は 描かない。"))
out.append(batch([k for k in G if k>='G23'],"一括B: G23〜G40(まだ 絵が ない 子)","添付画像(G01)と 同じ 絵柄で 描いて ください。添付画像の キャラ自体は 描かない。"))
open('/home/claude/artplan/ChatGPT一括.md','w',encoding='utf-8').write("\n".join(out))
print(len(G),len(P))
