"""きょうの挑戦・ともだちバトルの バランス確認(W12): 3体を 順に 出して、相手 3体と 1対1 で 力くらべ。2勝で 勝ち。
むかし = PERFECT 2連で 必殺(×2) / いま = 必殺ゲージ(PERFECT +1・有利 +1、2で 必殺)。相手の 強さの 倍率 K を 変えて、むかしと 同じ 勝率に なる 所を さがす。
自分と 相手は 同じ パワー(P)、手は ばらばら。自分は 相手 3体の 手を 見て 並べ方を えらぶ(いちばん 勝ちが 多い 並び)。"""
import random,itertools
H=['r','s','p']; beats={'r':'s','s':'p','p':'r'}
def mu(a,b): return 1 if a==b else (1.5 if beats[a]==b else 0.7)
def acc(q): return 1.0 if random.random()<q else random.uniform(0.45,0.92)
def play(q,K,mode,P=50):
    me=[random.choice(H) for _ in range(3)]; foe=[random.choice(H) for _ in range(3)]
    best=max(itertools.permutations(range(3)),key=lambda o:sum(mu(me[o[t]],foe[t]) for t in range(3)))
    wins=0; g=0; combo=0
    for t in range(3):
        h=me[best[t]]; m=mu(h,foe[t]); a=acc(q)
        if mode=='old': sp=2 if combo>=2 else 1; combo=combo+1 if a>=1 else 0
        else:
            sp=2 if g>=2 else 1; g=0 if sp==2 else min(2,g+(1 if a>=1 else 0)+(1 if m>1 else 0))
        d=P*(0.5+a)*m*sp; opp=P*mu(foe[t],h)*K
        wins+=d>=opp
    return wins>=2
if __name__=="__main__":
    N=60000; random.seed(1)
    for q in (0.3,0.5,0.7):
        o=sum(play(q,1.2,'old') for _ in range(N))/N
        row=[f"いま K{K}: {sum(play(q,K,'new') for _ in range(N))/N:.0%}" for K in (1.2,1.3,1.4,1.5)]
        print(f"PERFECT {q:.0%}: むかし(K1.2) {o:.0%} | "+" / ".join(row))
