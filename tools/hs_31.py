"""モルドレッド(★5 叛逆の騎士) 専用の必殺技「叛逆ノ黒剣」— 高品質版(hs_fx)・通し直し。敵は さいしょから 画面の こちら側(じゅうたんの 手前に 立つ者)
金色の 玉座の間(丸柱・旗の ひだ・天窓の 光と ほこり)。ひざまずく 黒い騎士(忠誠の姿) → かぶとの目が 赤く灯ると、金の旗が 1枚ずつ 斬り落とされ、広間が 赤く そまる(前ふり: 叛逆) →
かぶとの すき間のカットイン → 玉座の 前に 立ち、床に 立てた 黒剣を ひねる。赤い ひびが じゅうたんを こちらへ 走り(うしろでは 玉座が 二つに 割れる)、
ひびから 黒曜石の 刃が 列ごとに 突き出て、1列ずつ こちらへ 近づく → さいごの 列が 目の前を ふさぐ → 赤白 → 暗転
使い方: python3 hs_31.py 出力.mp4 / python3 hs_31.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old31", "/home/claude/tools/pre_fin/hs_31_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = 8.7
RED, GOLD, STONE, BAN = old.RED, old.GOLD, old.STONE, old.BAN
T_EYE, T_CUT, T_TWIST = old.T_EYE, old.T_CUT, old.T_TWIST
T_ROW0, RSTEP, NROW = 6.2, 0.24, 7


class Mordred(FX, old.Mordred):
    def __init__(s, W=540):
        old.Mordred.__init__(s, W); s.fx_init(31); H, k = s.H, s.k
        tex = 0.70 + 0.6 * s.fbm(0.004, 2, 6, 4, 0, 1)                                    # 石の かべ
        for sg in (-1, 1):
            for j in range(3):                                                       # 丸柱の まるみ
                x = W / 2 + sg * W * (0.47 - j * 0.11); w = (12 - j * 3) * k; q = (s.XX - x) / (w / 2 + 1); ins = (np.abs(q) < 1.0) & (s.YY > H * (0.02 + j * 0.06)) & (s.YY < H * (0.98 - j * 0.14))
                tex = np.where(ins, 0.45 + 1.5 * np.sqrt(np.clip(1 - q * q, 0, 1)) * (0.8 - 0.3 * q * sg), tex)
        for (sg, dx, sc) in BAN:                                                         # 旗の ひだ
            x = W / 2 + sg * W * (0.50 - dx); bw = W * 0.10 * sc; top = H * (0.06 + (1 - sc) * 0.16); bh = H * 0.30 * sc
            ins = (np.abs(s.XX - x) < bw / 2) & (s.YY > top) & (s.YY < top + bh * 1.14)
            tex = np.where(ins, 0.75 + 0.45 * np.sin((s.XX - x) / bw * 9.5 + 0.6) * (0.5 + 0.5 * (s.YY - top) / bh) + 0.25 * s._tiles[2][(s.YY.astype(np.int32) // 3) % 256, (s.XX.astype(np.int32) // 2) % 256], tex)
        flo = s.YY > H * 0.46; dep = np.clip((s.YY - H * 0.44) / (H * 0.56), 0, 1)
        g2 = cv2.remap(s._tiles[1], ((s.XX - W / 2) / (0.25 + dep) * 0.35 % 256).astype(np.float32), ((dep * 500 / (0.3 + dep)) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        tex = np.where(flo, (0.65 + 0.7 * g2) * (0.8 + 0.5 * dep), tex)
        s.tex = tex.astype(np.float32)
        L = s.L0()
        for i in range(4):
            x = W * (0.2 + 0.2 * i); cv2.fillPoly(L, [np.array([(x - 14 * k, 0), (x + 14 * k, 0), (x + 60 * k, H * 0.7), (x - 20 * k, H * 0.7)], np.int32)], 1.0, cv2.LINE_AA)
        s.shafts = cv2.GaussianBlur(L, (0, 0), 7 * k) * sstep(0.75, 0.3, s.yn)
        rg = np.random.RandomState(311); s.SW = (W / 2, H * 0.615)                         # 引きの 絵での 剣先(ひびの 出どころ)
        s.crk2 = []
        for i in range(7):                                                           # こちらへ 走る ひび(剣先 → 画面の 下)
            a = (i - 3) * 0.20; pts = [s.SW]; v = 0.0
            while v < 1: v += 0.06 + rg.rand() * 0.05; d = min(v, 1) ** 1.5 * H * 0.5; pts.append((s.SW[0] + math.sin(a) * d * 1.9 + (rg.rand() - 0.5) * 30 * k * (0.3 + v), s.SW[1] + d))
            s.crk2.append(pts)
        s.shd = []                                                                   # 刃の 列: (列, x, 高さの 倍, 幅の 倍, かたむき)
        for r in range(NROW):
            n = 5 + (r % 2)
            for j in range(n): s.shd.append((r, (j - (n - 1) / 2) / (n - 1) * 1.0 + (rg.rand() - 0.5) * 0.10, 0.65 + 0.6 * rg.rand(), 0.8 + 0.5 * rg.rand(), (rg.rand() - 0.5) * 0.30))

    def hall(s, red, cuts, t, split=0.0, crown=True):
        k = s.k; light = GOLD * (1 - red) + RED * red; t_ = getattr(s, "_t", 0.0)
        f = old.Mordred.hall(s, red, cuts, t, split, crown) * s.tex[:, :, None]
        f += s.nebula(t_, [(0.10, 0.07, 0.04), (0.25, 0.15, 0.06), (0.6, 0.4, 0.2)], 0.0009, (3.0, -2.0), 0.3) * (light * 0.8 + 0.2)[None, None, :]
        hz_ = 0.5 + 0.9 * s.fbm(0.002, t_ * 8, t_ * 12, 3, 0, 3); d = np.zeros_like(f); s.starfield(d, t_, 1400, (1, 1, 1), 3, 1.3, (4.0, 14.0))
        f += (s.shafts * (hz_ * 0.16 + d[:, :, 0] * 1.5))[:, :, None] * (light * 0.9 + 0.1)[None, None, :]                             # 天窓の 光と ほこり
        return f

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        s.draw_lit(f, img, cx, cy, hfrac, rim=getattr(s, "_rimc", GOLD), rim_dir=(0.0, -1.0), rim_v=0.8 * alpha, amb=(1.0, 0.8, 0.7), amb_v=0.10, alpha=alpha, shade=0.95)

    def add(s, f, L, col, gain=1.0, b1=3, b2=12):
        s.glowmask(f, np.clip(L, 0, 1), col, gain * 0.8, 0.8, 1.0, float(b2))

    def eyes(s, f, v, streak=1.0):
        if v <= 0: return
        sc = s._pl[2]
        for sg, e in ((-1, s.eyeL), (1, s.eyeR)):
            p = s.at(*e); L = s.L0(); cv2.ellipse(L, (int(p[0]), int(p[1])), (max(1, int(26 * sc)), max(1, int(9 * sc))), sg * 18, 0, 360, 1.0, -1, cv2.LINE_AA)
            s.glowmask(f, L, RED, 1.0 * v, 1.0, 1.0, 9.0); s.flare(f, p, max(2.0, 9 * sc), RED * 0.8 + 0.2, 0.6 * v, 1.0 + 1.5 * streak, 0.0, False)

    def shard(s, f, x, base, h, w, lean, glow=1.0):
        """床から つき出る 黒曜石の 刃(面の 明暗・赤い うつりこみ・ふち光)"""
        W, H, k = s.W, s.H, s.k
        if h < 3: return
        tip = (x + lean * h, base - h); sy = base - h * 0.80
        O = [(x - w / 2, base), (x - w / 2 + lean * h * 0.8, sy), tip, (x + w / 2 + lean * h * 0.8, sy), (x + w / 2, base)]; mid = (x + lean * h * 0.4, base)
        x0 = int(max(0, min(p[0] for p in O) - 8 * k)); x1 = int(min(W, max(p[0] for p in O) + 8 * k)); y0 = int(max(0, tip[1] - 8 * k)); y1 = int(min(H, base + 4 * k))
        if x1 - x0 < 3 or y1 - y0 < 3: return
        sh = lambda P: np.array([(p[0] - x0, p[1] - y0) for p in P], np.int32)
        M = np.zeros((y1 - y0, x1 - x0), np.float32); cv2.fillPoly(M, [sh(O)], 1.0, cv2.LINE_AA); ML = np.zeros_like(M); cv2.fillPoly(ML, [sh([O[0], O[1], tip, mid])], 1.0, cv2.LINE_AA)
        Y = s.YY[y0:y1, x0:x1]; X = s.XX[y0:y1, x0:x1]; v = np.clip((Y - tip[1]) / max(1.0, h), 0, 1)
        st = cv2.remap(s._tiles[0], ((X * 3.0 + x) % 256).astype(np.float32), ((Y * 0.15) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        left = np.array([0.020, 0.018, 0.028], np.float32)[None, None, :] * (0.6 + 0.8 * st)[:, :, None] + (v ** 2.2 * 0.45 * glow)[:, :, None] * RED[None, None, :]
        right = np.array([0.105, 0.10, 0.14], np.float32)[None, None, :] * (0.5 + 1.2 * st * (1.2 - v))[:, :, None] + (v ** 3 * 0.25 * glow)[:, :, None] * RED[None, None, :]
        right += (sstep(0.62, 0.9, st) * (1 - v) * 0.22)[:, :, None]                                                                 # てり
        col = left * ML[:, :, None] + right * (1 - ML)[:, :, None]
        F = f[y0:y1, x0:x1]; F[:] = F * (1 - M[:, :, None]) + col * M[:, :, None]
        E = np.zeros_like(M); cv2.polylines(E, [sh(O)], False, 1.0, max(1, int(1.5 * k + w * 0.012)), cv2.LINE_AA); cv2.line(E, tuple(int(q) for q in sh([mid])[0]), tuple(int(q) for q in sh([tip])[0]), 0.5, max(1, int(w * 0.02)), cv2.LINE_AA)
        F += ((E + cv2.GaussianBlur(E, (0, 0), 4 * k) * 1.2) * 0.9 * glow)[:, :, None] * (RED * 0.9 + 0.1)[None, None, :]

    def frame(s, t):
        s._t = t; o = {}; s._rimc = GOLD
        if t < T_CUT: f, sh = s.sc_fealty(t)
        elif t < T_TWIST: s._rimc = RED; f, sh = s.sc_cut(t - T_CUT); s.starfield(f, t, 300, RED * 0.7 + 0.3, 5, 0.5, (0.0, -20.0)); o = dict(vig=0.4, thr=0.9)
        else: s._rimc = RED; f, sh = s.sc_strike(t)
        f, sh = s.finale(f, sh, t, DUR, s.tip, (1, 0.2, 0.2), (1, 0.7, 0.3), -1)
        return s.film(f, t, sh * s.k, 0.6, o.get("thr", 0.82), (1.0, 0.75, 0.7), 1.0 + 0.2 * sh, 0.04, o.get("vig", 0.6), 1.15, 1.05)

    # 3) 玉座の 前で 剣を ひねる → ひびが こちらへ → 黒曜石の 刃が 列ごとに 近づく
    def sc_strike(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_TWIST; split = ease((u - 0.35) / 0.5) if u > 0.35 else 0.0; g = ease2(u / (T_ROW0 - T_TWIST))
        f = s.hall(1.0, [9] * 6, t, split, u < 0.35)
        if split > 0:
            s.glowline(f, [(s.TH[0], s.TH[1] - H * 0.12), (s.TH[0], s.TH[1] + H * 0.10)], RED, 2.0 + 3 * split, 0.9, 1.2)
        lit = np.exp(-((s.XX - W / 2) / (W * (0.10 + 0.5 * np.clip((s.YY - s.SW[1]) / (H * 0.4), 0, 1)))) ** 2) * sstep(s.SW[1], s.SW[1] + 20 * k, s.YY) * np.clip(1.6 * g - (s.YY - s.SW[1]) / (H * 0.4), 0, 1)
        f += (lit * 0.28)[:, :, None] * RED[None, None, :]                                                                            # ひびの 赤が 床を そめる
        for pts in s.crk2:
            n = int(g * (len(pts) - 1))
            if n >= 1:
                for j in range(n): s.glowline(f, [pts[j], pts[j + 1]], RED * 0.9 + 0.1, 1.2 + 3.5 * (j / len(pts)) ** 1.3, 0.9, 1.0)
                if n < len(pts) - 1: s.flare(f, pts[n], 5 * k, RED * 0.7 + 0.3, 0.9, 0.6, 0.0, False)
        cy = H * 0.50; kk = math.exp(-u * 8)
        img = s.pose(t, squash=0.07 * kk * math.cos(u * 20), jit=1.0 * k * g * (t < T_ROW0)); s.draw(f, img, W / 2, cy, 0.26); s.eyes(f, 1.0, 0.7)
        s.flare(f, s.SW, (8 + 16 * kk) * k, RED * 0.7 + 0.3, 0.9, 1.0, u * 3, False)
        r_ = s.rnd[:200]; pv = (u * (0.8 + r_[:, 2]) - r_[:, 0] * 1.0) % 1.0                                                           # はがれる 床の かけら・火の粉
        s.splat(f, W / 2 + (r_[:, 1] - 0.5) * W * (0.2 + 0.9 * r_[:, 0]), s.SW[1] + r_[:, 0] * H * 0.4 - pv * H * 0.12, (1 - pv) * g * 1.5, RED * 0.8 + 0.2, 1.0, 0.5)
        sh = 6 * kk + 2.5 * g * (t < T_ROW0) + (8 * math.exp(-(u - 0.35) * 6) if u > 0.35 else 0)
        VPy = H * 0.30
        for (r, xn, hm, wm, lean) in s.shd:                                           # 遠い列から 近い列へ
            v = t - (T_ROW0 + r * RSTEP + abs(xn) * 0.06)
            if v <= 0: continue
            base = H * (0.66 + 0.075 * r ** 1.25); ps = (base - VPy) / (H - VPy); gr = back(v / 0.13, 1.6)
            s.shard(f, W / 2 + xn * W * 0.78 * ps, base, H * (0.34 + 0.035 * r) * ps * hm * gr, W * 0.13 * ps * wm, lean, 1.0)
            if v < 0.25: s.flare(f, (W / 2 + xn * W * 0.78 * ps, min(H - 2, base)), 10 * k * ps * (1 - v / 0.25), RED * 0.7 + 0.3, 0.8, 0.8, 0.0, False)
        for r in range(NROW):
            v = t - (T_ROW0 + r * RSTEP)
            if 0 < v < 0.3: sh = max(sh, (5 + 2 * r) * math.exp(-v * 9)); f += (1 - v / 0.3) ** 2 * 0.10 * RED[None, None, :] * (1 + 0.3 * r)
        e = clamp((t - (T_ROW0 + NROW * RSTEP)) / 0.25)                                    # 目の前の 1本
        if e > 0: s.shard(f, W * 0.5, H * 1.6, H * 1.75 * back(e, 1.2), W * 1.5, 0.02, 1.4); sh = max(sh, 14 * e); f += e ** 2 * 0.5 * (RED * 0.6 + 0.4)[None, None, :]
        if u < 0.1: f = f * (u / 0.1) + (1 - u / 0.1) * 0.4 * RED[None, None, :]
        return f, sh

    def audio(s, path):
        a = Snd(DUR, 31); C = 130.81; maj = [0, 2, 4, 5, 7, 9, 11]; loc = [0, 1, 3, 5, 6, 8, 10]
        def fq(d, o=0, sc=maj): return C * 2 ** ((sc[d % 7] + 12 * (d // 7 + o)) / 12.0)
        def brass(d, t0, dur, g=0.06, o=1, sc=maj): a.note(fq(d, o, sc), t0, dur, g, "saw", 0.03); a.note(fq(d, o, sc) * 1.004, t0, dur, g * 0.6, "saw", 0.03)
        for i, d in enumerate([0, 0, 4, 7]): brass(d, 0.15 + i * 0.3, 0.4, 0.06)                                              # 王の ふし
        for d in (0, 2, 4): brass(d, 0.15, 1.3, 0.025, 0)
        a.boom(T_EYE, 42, 1.1, 0.6); a.note(fq(6, 2, loc), T_EYE, 0.8, 0.07, "sine")
        for i in range(6):                                                                                                    # 旗が 落ちるたび ふしが ゆがむ
            tk = T_EYE + 0.25 + i * 0.18; a.whoosh(tk - 0.05, tk + 0.03, 0.5, 0.6); brass([7, 6, 4, 3, 1, 0][i], tk, 0.3, 0.07, 1, loc); a.boom(tk + 0.12, 70, 0.3, 0.1)
        a.boom(T_CUT, 38, 1.2)
        for i in range(5): a.boom(T_CUT + 0.5 + i * 0.17 + 0.12, 55, 0.7, 0.15); brass([0, 1, 0, 3, 0][i], T_CUT + 0.62 + i * 0.17, 0.2, 0.08, 0, loc)
        a.boom(T_TWIST, 44, 1.2, 0.4); a.a += (a.nz - a.lp(a.nz, 0.4)) * a.env(T_TWIST, 0.001, 0.12) * 0.4                              # ひねる
        tk = T_TWIST; i = 0
        while tk < T_ROW0:                                                                                                    # ひびが こちらへ: 低い きざみが つまる
            a.note(fq([0, 0, 1, 0, 3, 0, 1, 0][i % 8], -1, loc), tk, 0.09, 0.12, "saw", 0.003); a.hat(tk, 0.04)
            if i % 3 == 0: a.a += (a.nz - a.lp(a.nz, 0.5)) * a.env(tk, 0.001, 0.04) * 0.2
            tk += max(0.05, 0.11 - i * 0.004); i += 1
        a.rise(T_TWIST, T_ROW0, 60, 260, 0.16)
        a.boom(T_TWIST + 0.35, 36, 1.2, 0.6); a.a += a.lp(a.nz, 0.25) * a.env(T_TWIST + 0.35, 0.01, 0.4) * 0.5                          # うしろで 玉座が 割れる
        for j in range(NROW):                                                                                                 # 列ごとに つき出る: だんだん 低く 重く(近づく)
            t0 = T_ROW0 + j * RSTEP; a.rise(t0 - 0.06, t0 + 0.04, 900 - 70 * j, 2600 - 200 * j, 0.14); a.boom(t0, 62 - 3.5 * j, 0.7 + 0.14 * j, 0.25); a.hat(t0, 0.3, 0.04)
            brass([0, 1, 3, 4, 5, 6, 7][j], t0, 0.22, 0.05 + 0.006 * j, 0, loc)
        for j, (d, o) in enumerate(((0, -1), (0, 0), (2, 0), (4, 0), (0, 1))): brass(d, T_ROW0 + j * 0.03, DUR - T_ROW0, 0.045, o, loc)
        te = T_ROW0 + NROW * RSTEP; a.boom(te, 34, 1.7, 0.6); a.a += a.lp(a.nz, 0.3) * a.env(te, 0.004, 0.35) * 0.6
        s.fin_snd(a, DUR, 31); a.save(path)


if __name__ == "__main__":
    r = Mordred()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
