"""手合わせの バランス確認: むかし(3体・PERFECT 2連で 必殺)と いま(5体から えらぶ・つぎの手が 見える・必殺ゲージ)で、同じ レア度の 相手を どれだけ けずれるか。
q = PERFECT を 出せる 割合。HP の 倍率を 変えて、むかしと 同じ くらいに なる 所を さがす(2026-11-27: x1.35 を 採用 = index.html の HP5)"""
import random
H=['r','s','p']; beats={'r':'s','s':'p','p':'r'}
def mu(a,b): return 1 if a==b else (1.5 if beats[a]==b else 0.7)
def rollh(fh): return [fh if random.random()<0.6 else random.choice(H) for _ in range(3)]
def acc(q): return 1.0 if random.random()<q else random.uniform(0.45,0.92)
def old(q,P=50,HP=310):
    hs=rollh(random.choice(H)); party=[random.choice(H) for _ in range(3)]; used=[]; d=0; combo=0
    for t in range(3):
        c=max([i for i in range(3) if i not in used], key=lambda i:mu(party[i],hs[t])); used.append(c)
        a=acc(q); d+=P*(0.5+a)*mu(party[c],hs[t])*(2 if combo>=2 else 1); combo=combo+1 if a>=1 else 0
        if d>=HP: break
    return min(d/HP,1)
def new(q,P=50,HP=310,n=5):
    hs=rollh(random.choice(H)); party=[random.choice(H) for _ in range(n)]; used=[]; d=0; g=0
    for t in range(3):
        av=[i for i in range(n) if i not in used]
        def score(i):
            s=mu(party[i],hs[t])
            if t<2 and mu(party[i],hs[t+1])>1 and sum(1 for j in av if j!=i and mu(party[j],hs[t+1])>1)==0: s-=0.3
            return s
        c=max(av,key=score); used.append(c); m=mu(party[c],hs[t]); a=acc(q)
        use=g>=2 and (t==2 or m>1 or not any(mu(party[j],hs[t+1])>1 for j in av if j!=c))
        d+=P*(0.5+a)*m*(2 if use else 1)
        g=0 if use else min(2,g+(1 if a>=1 else 0)+(1 if m>1 else 0))
        if d>=HP: break
    return min(d/HP,1)
if __name__=="__main__":
    N=40000
    for q in (0.3,0.5,0.7):
        o=[old(q) for _ in range(N)]
        for k in (1.0,1.35):
            n=[new(q,HP=310*k) for _ in range(N)]
            print(f"PERFECT {q:.0%}: むかし けずり {sum(o)/N:.2f} 倒しきり {sum(x>=1 for x in o)/N:.0%} | いま HP x{k}: けずり {sum(n)/N:.2f} 倒しきり {sum(x>=1 for x in n)/N:.0%}")
