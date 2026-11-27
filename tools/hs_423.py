"""クラーケ(★5 大だこ使いの魔女) 専用の必殺技「深海ノ腕」— 高品質版(hs_fx)。深海の 闇から 八本の 大だこの 腕が 画面の ふちを まわりこんで 巻きつき、視界を とじる
光の とどかない 深海(マリンスノー・青い 発光)。魔女が 浮かび、水晶玉が 灯る。奥の 闇に 大きな 腕の 影が ゆれる(前ふり) → 目の カットイン+技名 →
水晶玉を かかげる: 玉から 音の 輪 → 画面の ふちから 八本の 腕(吸盤・てり)が 1本ずつ のびて、うずを 巻きながら こちらを つつむ →
すき間が せばまり、さいごの すき間から 水晶玉の 光が さしこむ → 白 → 暗転
使い方: python3 hs_423.py 出力.mp4 / python3 hs_423.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.9
T_CUT, T_ARMS, T_HIT = 2.8, 4.8, 10.2
ABY = np.array([0.008, 0.03, 0.07], np.float32); BIO = np.array([0.30, 0.90, 1.0], np.float32); PUR = np.array([0.62, 0.42, 0.95], np.float32); ORBC = np.array([0.72, 0.66, 1.0], np.float32); PINK = np.array([1.0, 0.72, 0.86], np.float32)
EYE = (330, 292); ORB = (150, 385)


class Kraken(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 423, (0.6, 0.5, 1.0), "深海ノ腕", "Abyssal Arms", W); s.fx_init(423)
        s._vt = [s.text_img(c, hz.SERIF, int(76 * s.k), 3) for c in "深海ノ腕"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; H = s.H; rg = np.random.RandomState(4230); s._sn = rg.rand(500, 5).astype(np.float32); s.C = (W * 0.5, H * 0.50)
        s._arms = []
        for i in range(8):                                                           # ふちの 出どころ(下が 多め)と まわる 向き
            a = [1.57, 0.55, 2.6, 1.05, 2.1, -0.35, 3.5, -1.57][i] + (rg.rand() - 0.5) * 0.2; R = max(W, H) * 0.86
            s._arms.append((W * 0.5 + math.cos(a) * W * 0.80, H * 0.5 + math.sin(a) * H * 0.66, 1 if i % 2 == 0 else -1, rg.rand() * 6.28, 0.85 + 0.3 * rg.rand()))

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 深海 ----
    def abyss(s, t, lum=1.0, glow=0.0):
        W, H, k = s.W, s.H, s.k
        f = ABY[None, None, :] * (2.6 - 2.0 * s.yn)[:, :, None] + np.array([0.02, 0.07, 0.12], np.float32)[None, None, :] * np.exp(-((s.yn - 0.1) / 0.35) ** 2)[:, :, None]
        ray = s.tex(1, s.XX * 0.35 + s.YY * 0.10 + t * 3, 5.0) ** 2 * np.exp(-s.yn / 0.55); f += (ray * 0.10)[:, :, None] * BIO[None, None, :]                      # 上からの かすかな 光
        hz_ = s.fbm(0.0012, t * 6, t * 3, 3, 40 * k, 1); f += (sstep(0.4, 0.9, hz_) * 0.05)[:, :, None] * np.array([0.3, 0.5, 0.8], np.float32)[None, None, :]
        f += (np.exp(-np.hypot(s.xn - 0.5, (s.yn - 0.52) * 0.8) / 0.33) * (0.05 + 0.20 * glow))[:, :, None] * ORBC[None, None, :]
        sn = s._sn; x = (sn[:, 0] * W + np.sin(t * (0.5 + sn[:, 2]) + sn[:, 3] * 20) * 12 * k) % W; y = (sn[:, 1] * H + t * H * (0.02 + 0.05 * sn[:, 2])) % H
        s.splat(f, x, y, (0.25 + 0.7 * sn[:, 4] ** 2) * 0.7, (0.7, 0.85, 1.0), 1.0, 0.3)                                                                           # マリンスノー
        bi = sn[:60]; tw = np.sin(t * (1 + 2 * bi[:, 3]) + bi[:, 2] * 30) ** 2; s.splat(f, (bi[:, 1] * W + t * 6 * k) % W, (bi[:, 0] * H - t * 9 * k) % H, tw * 1.0, BIO, 1.6, 0.9)      # 発光する いきもの
        return (f * lum).astype(np.float32)

    # ---- 大だこの 腕(式): 先細りの 管 + てり + 吸盤 ----
    def armpts(s, i, grow, t, scale=1.0, n=46):
        ex, ey, dr, ph, wk = s._arms[i]; C = s.C; out = []
        m = max(2, int(n * grow))
        for j in range(m + 1):
            q = j / n; rot = dr * (2.2 * q ** 1.2) + 0.10 * math.sin(t * 1.3 + ph + q * 5) * q; rad = 1 - 0.84 * q
            vx = (ex - C[0]) * rad; vy = (ey - C[1]) * rad; x = C[0] + vx * math.cos(rot) - vy * math.sin(rot); y = C[1] + vx * math.sin(rot) + vy * math.cos(rot)
            out.append((x, y, s.W * 0.115 * wk * (1 - 0.86 * q) * scale + 3 * s.k, q))
        return out

    def arms(s, f, t, grows, scale=1.0, shadow=False, v=1.0):
        W, H, k = s.W, s.H, s.k; M = s.L0(); S = s.L0(); K = s.L0()
        for i in range(7, -1, -1):
            g = grows[i]
            if g <= 0.02: continue
            P = s.armpts(i, g, t, scale)
            for (x, y, w, q) in P: cv2.circle(M, (int(x), int(y)), int(w + 3 * k), 1.0, -1, cv2.LINE_AA); cv2.circle(S, (int(x), int(y)), int(w + 3 * k), 0.04, -1, cv2.LINE_AA)       # ふちどり
            for (x, y, w, q) in P: cv2.circle(S, (int(x), int(y)), int(w), 0.34, -1, cv2.LINE_AA)
            if shadow: continue
            for (x, y, w, q) in P: cv2.circle(S, (int(x - w * 0.34), int(y - w * 0.36)), max(1, int(w * 0.36)), 0.78, -1, cv2.LINE_AA)                                    # てり
            for (x, y, w, q) in P[::2]: cv2.circle(S, (int(x - w * 0.42), int(y - w * 0.46)), max(1, int(w * 0.13)), 1.0, -1, cv2.LINE_AA)
            dr = s._arms[i][2]
            for j in range(2, len(P) - 1, 3):                                         # 吸盤(うずの 内がわに ならぶ)
                x, y, w, q = P[j]; nx = P[j + 1][0] - P[j - 1][0]; ny = P[j + 1][1] - P[j - 1][1]; nl = math.hypot(nx, ny) + 1e-6; ox = -ny / nl * dr * w * 0.42; oy = nx / nl * dr * w * 0.42
                r = max(2, int(w * 0.30)); cv2.circle(K, (int(x + ox), int(y + oy)), r, 1.0, -1, cv2.LINE_AA); cv2.circle(K, (int(x + ox), int(y + oy)), max(1, int(r * 0.45)), 0.25, -1, cv2.LINE_AA)
        M = cv2.GaussianBlur(M, (0, 0), 0.8 * k) * v
        if shadow:
            f *= (1 - M * 0.55)[:, :, None]; return
        S = cv2.GaussianBlur(S, (0, 0), 2.0 * k) * (0.82 + 0.36 * s.tex(2, s.XX * 1.4, s.YY * 1.4)); K = cv2.GaussianBlur(K, (0, 0), 0.9 * k)
        dk = np.array([0.02, 0.008, 0.05], np.float32); md = np.array([0.22, 0.11, 0.40], np.float32); hi = np.array([0.95, 0.80, 1.0], np.float32)
        a1 = sstep(0.05, 0.42, S)[:, :, None]; a2 = sstep(0.5, 1.0, S)[:, :, None]; col = dk + (md - dk) * a1 + (hi - md) * a2
        col = col * (1 - K * 0.75)[:, :, None] + (PINK * 0.75)[None, None, :] * (K * sstep(0.1, 0.4, S))[:, :, None]
        d = np.hypot(s.XX - s.C[0], s.YY - s.C[1]) / W; col = col * (0.55 + 0.9 * np.exp(-d / 0.5))[:, :, None] + (ORBC * 0.10)[None, None, :] * np.exp(-d / 0.3)[:, :, None]      # 玉の 光が あたる
        f *= (1 - M)[:, :, None]; f += col * M[:, :, None]

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        g = getattr(s, '_og', 0.3)
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(ORBC * 0.7 + BIO * 0.3), rim_dir=(-0.8, 0.2), rim_v=(0.35 + 0.5 * g) * alpha, amb=(0.6, 0.7, 1.0), amb_v=0.25, aura=ORBC, aura_v=0.08 + 0.14 * g, alpha=alpha, shade=0.60 + 0.3 * g)

    def frame(s, t):
        o = {}; s._og = 0.3
        if t < T_CUT: f, sh = s.sc_deep(t)
        elif t < T_ARMS: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.92)
        else: f, sh = s.sc_arms(t)
        f, sh = s.finale(f, sh, t, DUR, None, (0.7, 0.65, 1), (0.3, 0.9, 1.0), -1)
        return s.film(f, t, sh * s.k, 0.6, o.get("thr", 0.80), (0.88, 0.9, 1.0), 1.0 + 0.15 * min(sh, 8), 0.045, o.get("vig", 0.62), 1.15, 1.15)

    def witch(s, f, t, alpha=1.0, size=0.42, **kw):
        W, H, k = s.W, s.H, s.k; img = s.pose(t, **kw); s.draw(f, img, W / 2, H * 0.55 + 7 * k * math.sin(t * 1.7), size, alpha)
        op = s.at(*s.A(*ORB)); s.flare(f, op, (8 + 3 * math.sin(t * 4) ** 2 + 14 * s._og) * k * size / 0.42, ORBC, (0.5 + 0.5 * s._og) * alpha, 0.5, t * 0.4, False)
        return op

    # 1) 深海に 浮かぶ。水晶玉が 灯り、奥の 闇に 腕の 影
    def sc_deep(s, t):
        W, H, k = s.W, s.H, s.k; g = ease((t - 0.7) / 1.2); s._og = 0.15 + 0.45 * g
        f = s.abyss(t, 1.0, g)
        s.arms(f, t * 0.6, [0.55 + 0.25 * math.sin(t * 0.7 + i) for i in range(8)], 0.8, True, 0.55 * clamp((t - 1.2) / 1.2))      # 影
        s.witch(f, t, clamp(t / 0.7), sway=0.5, breath=0.5)
        return f * clamp(t / 0.6), 0

    # 2) 目の カットイン(波うつ ふちの 帯)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT; s._og = 0.8
        f = s.abyss(t, 0.35, 0.5)
        a = ease(u / 0.28); cy = H * 0.33; hh = H * 0.11
        band = np.array([0.02, 0.02, 0.09], np.float32)[None, None, :] + (PUR * 0.45)[None, None, :] * (0.3 + 0.9 * s.tex(0, s.XX * 0.3 + t * 25, s.YY * 0.5 + t * 8))[:, :, None]
        ex, ey = s.A(*EYE); zf = 1.5; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(band, img, W / 2 - (ex - s.aw / 2) * sc + (1 - a) * W * 0.9 - u * 5 * k, cy - (ey - s.ah / 2) * sc, zf)
        top = cy - hh * a + 13 * k * np.sin(s.XX * 0.030 + t * 3.5); bot = cy + hh * a + 13 * k * np.sin(s.XX * 0.026 - t * 3.0 + 2)                                        # たこの 腕の ように うねる ふち
        m = (sstep(top - 1, top + 1, s.YY) * sstep(bot + 1, bot - 1, s.YY))[:, :, None] * (a > 0.02); f = f * (1 - m) + band * m
        s.glowmask(f, (np.exp(-((s.YY - top) / (2.0 * k)) ** 2) + np.exp(-((s.YY - bot) / (2.0 * k)) ** 2)) * a, ORBC * 0.6 + BIO * 0.4, 0.9, 1.0, 0.8, 8.0)
        n = len(s._vt); step = W * 0.20
        for i, T in enumerate(s._vt):                                                # 技名: 深みから 浮かび上がる
            p = (u - 0.5 - i * 0.15) / 0.5
            if p > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.68 + (1 - ease(p)) * 34 * k + 3 * k * math.sin(t * 3 + i), clamp(p) ** 1.3, 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.35 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.68 - 76 * k, 1.0, 1.0)
        if u > 1.4:
            g = ease((u - 1.4) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.68 + 58 * k), (W / 2 + W * 0.42 * g, H * 0.68 + 58 * k)], ORBC, 1.0, 0.8, 0.8)
            s.text(f, s.txt_en, W / 2, H * 0.68 + 84 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.6
        return f, 3 * math.exp(-u * 6)

    # 3) 玉を かかげる → 八本の 腕が 巻きつく → すき間から 玉の 光
    def sc_arms(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_ARMS; e = clamp((u - 0.7) / (T_HIT - T_ARMS - 0.7)); sh = 0.0
        s._og = 0.6 + 0.4 * ease(u / 0.6)
        f = s.abyss(t, 1.0 - 0.45 * e, 1.0)
        size = 0.42 * (1 - 0.22 * ease2(e)); a_ = 1 - clamp((e - 0.62) / 0.25)
        op = s.witch(f, t, max(a_, 0.0), size, sway=0.5, breath=0.4, squash=-0.03 * ease(u / 0.5), lean=4 * k * ease(u / 0.5)) if a_ > 0.02 else s.C
        d = np.hypot(s.XX - op[0], s.YY - op[1])
        for i in range(3):                                                           # 玉からの 音の 輪
            q = (u - i * 0.22) / 1.1
            if 0 < q < 1: f += (np.exp(-((d - q * W * 1.1) / ((3 + 16 * q) * k)) ** 2) * (1 - q) ** 1.5 * 0.7)[:, :, None] * (ORBC * 0.7 + BIO * 0.3)[None, None, :]
        if u > 0.5:
            grows = [ease(clamp((e - i * 0.075) / 0.50)) for i in range(8)]; s.arms(f, t, grows, 1.0 + 1.5 * e ** 2.0)
            sh += 1.2 * sum(1 for g in grows if 0.02 < g < 0.98) * 0.5 + 6 * e ** 2
            for i in range(8):                                                       # のびる しゅんかんの あわ
                q = e - i * 0.075
                if 0 < q < 0.25: ex, ey, _, _, _ = s._arms[i]; rg = s.rnd[i * 20:i * 20 + 20]; s.splat(f, ex + (s.C[0] - ex) * (0.25 + 0.5 * rg[:, 0]) + (rg[:, 1] - 0.5) * 120 * k, ey + (s.C[1] - ey) * (0.25 + 0.5 * rg[:, 0]) - q * H * 0.5 * rg[:, 2], rg[:, 3] * (1 - q / 0.25), BIO * 0.5 + 0.5, 1.4, 0.5)
        C = s.C; gl = e ** 2.2
        if e > 0.25:                                                                 # すき間から さす 玉の 光
            s.flare(f, C, (10 + 60 * gl) * k, ORBC * 0.6 + 0.4, 0.5 + 0.7 * gl, 0.6, t * 0.3, False)
            f = s.godrays(f, C, 0.25 + 0.7 * gl, 0.85, 10, 0.88, 0.6)
        if u < 0.12: f = f + (1 - u / 0.12) * 0.4
        if e > 0.88: f = f + ((e - 0.88) / 0.12) ** 2 * 1.5
        return f, sh

    # ---- この子だけの曲: 深い 持続音と クジラの ような 声 → 水の 中の 鈴 → 玉の ソナー音 → 腕 8本ぶんの 低い うねりが かさなる → 高い 一音が ふくらむ ----
    def audio(s, path):
        a = Snd(DUR, 423); D = 73.42 * 2; ph = [0, 1, 3, 5, 7, 8, 10]
        def fq(d, o=0): return D * 2 ** ((ph[d % 7] + 12 * (d // 7 + o)) / 12.0)
        a.a += (a.tone(D / 4) + 0.6 * a.tone(D / 4 * 1.5) + 0.3 * a.tone(D / 2 * 1.01)) * 0.10 * np.clip(a.t / 1.5, 0, 1) * a.gate(0, T_HIT, 0.3)
        a.a += a.lp(a.lp(a.nz, 0.02), 0.02) * 2.0 * a.gate(0, T_HIT, 0.5)                                                  # 水の 圧
        def whale(t0, f0, f1, ln, g=0.06):
            u_ = np.clip((a.t - t0) / ln, 0, 1); fr = f0 + (f1 - f0) * np.sin(u_ * 3.1416) ; phs = 2 * np.pi * np.cumsum(fr) / a.sr; a.a += np.sin(phs + 1.5 * np.sin(2 * np.pi * 4 * a.t)) * a.env(t0, ln * 0.4, ln * 0.4) * g
        whale(0.4, 180, 260, 1.4); whale(1.7, 150, 210, 1.0, 0.05)
        rg = np.random.RandomState(423)
        for i in range(22): a.note(700 + rg.rand() * 1600, rg.rand() * T_HIT, 0.06, 0.02, "sine", 0.01)                          # あわ
        a.note(fq(11, 1), 0.7, 1.2, 0.09, "bell")
        a.boom(T_CUT, 44, 1.1, 0.5); a.whoosh(T_CUT - 0.3, T_CUT, 0.3, 0.1)
        for i in range(4): a.note(fq([0, 2, 4, 7][i], 1), T_CUT + 0.5 + i * 0.15, 0.9, 0.08, "bell")
        for d in (0, 2, 4): a.note(fq(d, 0), T_CUT, 1.9, 0.04, "pad", 0.1)
        for i in range(3): a.note(fq(14, 1), T_ARMS + i * 0.22, 0.5, 0.10 - 0.02 * i, "sine", 0.002); a.note(fq(14, 2), T_ARMS + i * 0.22, 0.3, 0.04, "sine", 0.002)      # ソナー
        T0 = T_ARMS + 0.7; L = T_HIT - T0
        for i in range(8):                                                                                              # 腕 1本ずつ: 低い うねり
            tk = T0 + i * 0.075 * L; whale(tk, 60 + 8 * i, 95 + 12 * i, 1.6, 0.07); a.boom(tk, 46 + 3 * i, 0.9, 0.3); a.a += a.lp(a.nz, 0.08) * a.env(tk, 0.05, 0.3) * 0.25
            a.note(fq(i, -1), tk, L * (1 - i * 0.075), 0.035, "saw", 0.5)
        a.rise(T0 + L * 0.3, T_HIT, 60, 220, 0.12); a.rise(T0 + L * 0.5, T_HIT, 900, 2600, 0.035)
        for j, d in enumerate((0, 4, 7, 9, 11)): a.note(fq(d, 0), T_HIT - 2.0 + j * 0.1, 2.2, 0.045, "pad", 0.8)
        a.boom(T_HIT, 36, 1.7, 0.9)
        for j, d in enumerate((0, 4, 7, 11)): a.note(fq(d, 1), T_HIT + 0.02, 0.6, 0.06, "bell")
        s.fin_snd(a, DUR, 423); a.save(path)


if __name__ == "__main__":
    r = Kraken()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
