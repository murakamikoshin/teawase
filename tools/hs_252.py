"""ワダツミ(★5 大海原の神) 専用の必殺技「大海原ノ三叉」— 高品質版(hs_fx)。海に 三叉を 突き立て、海の 中の こちらへ 3本の 水の きりが 降りてくる
夜明けの 海の 上に 立つ → 三叉の 3つの 先に 光が 1つずつ 灯る(前ふり) → 目の カットイン+技名 →
三叉を 海に 突き立てる(水しぶき) → 水面が せり上がって 画面が 海の 中へ → 見上げると 水面の 丸い 窓(ゆれる 光・光の すじ・あわ)。
窓から 3本の うずまく 水の きりが 左・右・まん中の 順に こちらへ 降りてくる。さいごの 1本の あわの うずに のまれる → 白 → 暗転
使い方: python3 hs_252.py 出力.mp4 / python3 hs_252.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.6
T_CUT, T_STRIKE, T_DIVE, T_HIT = 2.7, 4.7, 5.55, 9.9
T_DR = (6.35, 6.95, 7.55)                                                              # 3本の きりが 出る 時刻
TEAL = np.array([0.25, 0.90, 0.85], np.float32); DEEP = np.array([0.02, 0.09, 0.20], np.float32); FOAM = np.array([0.85, 1.0, 1.0], np.float32); PEACH = np.array([1.0, 0.74, 0.52], np.float32); GOLD = np.array([1.0, 0.85, 0.45], np.float32)
EYE = (367, 245); TIPS = [(20, 205), (54, 136), (129, 141)]; HEAD = (107, 258)


class Wadatsumi(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 252, (0.3, 0.85, 0.9), "大海原ノ三叉", "Trident of the Deep", W); s.fx_init(252)
        s._vt = [s.text_img(c, hz.SERIF, int(60 * s.k), 3) for c in "大海原ノ三叉"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; s._uw = 0.0

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 夜明けの 海(水面の 上) ----
    def dawn(s, t, lum=1.0):
        W, H, k = s.W, s.H, s.k; hy = 0.60
        sky = np.array([0.05, 0.13, 0.26], np.float32)[None, None, :] * (1 - s.yn / hy).clip(0, 1)[:, :, None] ** 0.8 * 1.6 + (PEACH * 1.25)[None, None, :] * np.exp(-((s.yn - hy) / 0.15) ** 2)[:, :, None] + np.array([0.10, 0.30, 0.38], np.float32)[None, None, :] * np.exp(-((s.yn - hy + 0.16) / 0.16) ** 2)[:, :, None]
        cl = s.fbm(0.0016, t * 6, 0, 4, 40 * k, 1) * np.exp(-((s.yn - 0.36) / 0.20) ** 2); sky += (sstep(0.45, 0.8, cl) * 0.35)[:, :, None] * (PEACH * 0.7 + 0.3)[None, None, :]
        sky += (np.exp(-np.hypot(s.xn - 0.5, (s.yn - hy) * 1.5) / 0.13) * 1.5)[:, :, None] * np.array([1.0, 0.9, 0.7], np.float32)[None, None, :]      # のぼる 日の 光
        d = np.clip((s.yn - hy) / (1 - hy), 0, 1); z = 1.0 / (d + 0.035); u = (s.xn - 0.5) * z * 70; v = z * 46 + t * 7
        n = s.tex(0, u, v) * 0.6 + s.tex(2, u * 2.3 + 40, v * 2.3 + t * 5) * 0.4
        sea = (DEEP * 1.3)[None, None, :] * (0.35 + 1.2 * n)[:, :, None] * (0.6 + 0.9 * (1 - d))[:, :, None] + (TEAL * 0.22)[None, None, :] * (n * (1 - d))[:, :, None]
        sea += (np.exp(-((s.xn - 0.5) / (0.05 + 0.22 * d)) ** 2) * sstep(0.50, 0.75, n) * (1.8 - d))[:, :, None] * (PEACH * 0.8 + 0.2)[None, None, :]                # 日の 道
        m = sstep(hy - 0.002, hy + 0.002, s.yn)[:, :, None]
        return ((sky * (1 - m) + sea * m) * lum).astype(np.float32)

    # ---- 海の 中から 見上げる: 水面の 丸い 窓・光の すじ・あわ ----
    def under(s, t, lum=1.0):
        W, H, k = s.W, s.H, s.k; C = (W / 2, H * 0.20); R = W * 0.56
        dx = s.XX - C[0]; dy = s.YY - C[1]; r = np.hypot(dx, dy) / R + 1e-4; th = np.arctan2(dy, dx)
        rip = s.tex(0, s.XX * 0.55 + t * 9, s.YY * 0.55 + t * 5) + s.tex(2, s.XX * 1.2 - t * 13, s.YY * 1.2 + t * 4) - 1.0
        rw = r + rip * 0.07; win = sstep(1.0, 0.90, rw)
        base = DEEP[None, None, :] * (1.9 - 1.5 * s.yn)[:, :, None] + (TEAL * 0.10)[None, None, :] * (1 - s.yn)[:, :, None]
        ca = s.tex(1, s.XX * 0.38 + rip * 22 + t * 6, s.YY * 0.38 - t * 4); ca = sstep(0.50, 0.95, ca) * 0.6                                              # ゆれる 光の あみ
        sky = (np.array([0.55, 0.85, 0.95], np.float32)[None, None, :] * (0.75 + 0.5 * ca)[:, :, None] + (PEACH * 0.9)[None, None, :] * np.exp(-(rw / 0.35) ** 2)[:, :, None])
        out = base + (TEAL * 0.16)[None, None, :] * (ca * np.exp(-np.clip(r - 1, 0, None) / 0.6))[:, :, None]
        f = out * (1 - win)[:, :, None] + sky * win[:, :, None]
        f += (np.exp(-((rw - 0.95) / 0.05) ** 2) * 0.8)[:, :, None] * FOAM[None, None, :]                                                         # 窓の ふち
        ray = s.tex(3, th / 6.2832 * 1024 + t * 3, t * 5) ** 2 * np.exp(-np.clip(r - 0.6, 0, None) / 1.3) * sstep(0.5, 1.1, r) * (dy > -R * 0.2)
        f += (ray * 0.55)[:, :, None] * (TEAL * 0.6 + 0.4)[None, None, :]
        bb = np.zeros_like(f); s.starfield(bb, t, 240, (0.8, 1.0, 1.0), 6, 1.5, (4.0, -55.0)); s.starfield(bb, t, 400, (0.5, 0.8, 0.9), 2, 0.5, (6.0, 9.0)); f += bb
        return (f * lum).astype(np.float32)

    def ripple(s, f, c, fronts):
        dx = s.XX - c[0]; dy = s.YY - c[1]; d = np.hypot(dx, dy) + 1e-3; disp = np.zeros_like(d); edge = np.zeros_like(d)
        for rf, amp, wd in fronts:
            disp += amp * np.exp(-((d - rf) / wd) ** 2); edge += np.exp(-((d - rf - wd * 0.4) / (wd * 0.20)) ** 2) * min(1.0, amp / (22 * s.k))
        f = cv2.remap(f, (s.XX - dx / d * disp).astype(np.float32), (s.YY - dy / d * disp).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return f + (edge * 0.5)[:, :, None] * FOAM[None, None, :]

    # ---- 水の きり(手前を 向いた うず): しん・あわの らせん・ふちの 輪 ----
    def drill(s, f, c, R, t, v=1.0, tail=None):
        if v <= 0.01: return
        W, H, k = s.W, s.H, s.k
        if tail is not None: s.glowline(f, [tail, c], FOAM * 0.7 + TEAL * 0.3, max(1.0, min(R * 0.10, 9.0) / k), 0.45 * v, 0.9)
        dx = s.XX - c[0]; dy = s.YY - c[1]; r = np.hypot(dx, dy) / R + 1e-4; th = np.arctan2(dy, dx); lr = np.log(r + 0.03)
        ph = th + t * 9 - 3.2 * lr
        n = s.tex(1, ph / 6.2832 * 512, lr * 60 - t * 150) ; n2 = s.tex(3, ph / 6.2832 * 1024 + 30, lr * 150 - t * 330)
        arm = 0.5 + 0.5 * np.cos(3 * ph)                                                                                                  # 3すじ(三叉)
        ins = sstep(1.0, 0.90, r)
        I = ins * (0.20 + 0.9 * arm ** 2 * (0.4 + 0.9 * n) + 0.7 * n2 ** 3) * (0.35 + 0.8 * np.clip(r, 0, 1))
        f *= (1 - ins * 0.75 * v)[:, :, None]
        f += (I * v)[:, :, None] * (FOAM * 0.75 + TEAL * 0.25)[None, None, :] + (ins * 0.20 * v * (1 - r).clip(0, 1))[:, :, None] * (DEEP * 2.0 + TEAL * 0.3)[None, None, :]
        f += (np.exp(-(r / 0.16) ** 2) * 1.5 * v)[:, :, None] * FOAM[None, None, :]
        f += (np.exp(-((r - 0.96) / 0.045) ** 2) * (0.6 + 0.8 * n) * v)[:, :, None] * FOAM[None, None, :]
        f += (np.exp(-np.clip(r - 1, 0, None) / 0.25) * (r > 1) * 0.30 * v)[:, :, None] * TEAL[None, None, :]

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        if s._uw > 0: return s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(FOAM), rim_dir=(0.0, -1.0), rim_v=0.6 * alpha, amb=(0.4, 0.8, 1.0), amb_v=0.45, aura=TEAL, aura_v=0.25, alpha=alpha, shade=0.62)
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(PEACH * 0.7 + 0.3), rim_dir=(0.0, 1.0), rim_v=0.55 * alpha, amb=(0.7, 0.9, 1.0), amb_v=0.14, aura=TEAL, aura_v=0.10, alpha=alpha, shade=0.88)

    def frame(s, t):
        o = {}; s._uw = 0.0
        if t < T_CUT: f, sh = s.sc_dawn(t)
        elif t < T_STRIKE: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.92)
        else: f, sh = s.sc_deep(t)
        f, sh = s.finale(f, sh, t, DUR, None, (0.6, 1, 1), (0.3, 0.6, 1.0), -1)
        return s.film(f, t, sh * s.k, 0.55, o.get("thr", 0.84), (0.85, 0.97, 1.0), 1.0 + 0.15 * min(sh, 8), 0.035, o.get("vig", 0.55), 1.12, 1.08)

    def stand(s, f, t, cy=None, size=0.40, **kw):
        W, H, k = s.W, s.H, s.k; cy = cy or H * 0.565; fy = cy + size * H * 0.47
        L = s.L0()
        for i in range(3):                                                           # 足もとの 波の 輪
            q = ((t * 0.5 + i / 3.0) % 1.0); cv2.ellipse(L, (int(W / 2), int(fy)), (int(W * (0.06 + 0.30 * q)), int(H * (0.006 + 0.028 * q))), 0, 0, 360, float((1 - q) ** 1.5 * 0.7), max(1, int(2 * k)), cv2.LINE_AA)
        s.glowmask(f, L, FOAM * 0.6 + PEACH * 0.4, 0.5, 1.0, 0.6, 6.0)
        sh_ = s.L0(); cv2.ellipse(sh_, (int(W / 2), int(fy)), (int(W * 0.13), int(H * 0.012)), 0, 0, 360, 0.5, -1, cv2.LINE_AA); f *= (1 - cv2.GaussianBlur(sh_, (0, 0), 4 * k))[:, :, None]
        img = s.pose(t, **kw); s.draw(f, img, W / 2, cy, size)

    # 1) 夜明けの 海に 立つ。三叉の 3つの 先に 光が 灯る
    def sc_dawn(s, t):
        W, H, k = s.W, s.H, s.k
        f = s.dawn(t); s.stand(f, t, sway=0.3, breath=0.5)
        for i, tp in enumerate(TIPS):
            q = (t - 0.9 - i * 0.5)
            if q > 0: s.flare(f, s.at(*s.A(*tp)), (7 + 9 * math.exp(-q * 5) + 1.5 * math.sin(t * 9 + i)) * k, FOAM * 0.6 + TEAL * 0.4, clamp(q * 6), 0.5, 0.0, False)
        return f * clamp(t / 0.5), 0

    # 2) 目の カットイン(波の 形の 帯)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.dawn(t, 0.32)
        a = ease(u / 0.28); cy = H * 0.33; hh = H * 0.115
        ca = sstep(0.5, 0.85, s.tex(1, s.XX * 0.9 + t * 20, s.YY * 0.9 - t * 9))
        band = (DEEP * 2.2)[None, None, :] + (TEAL * 0.55)[None, None, :] * (0.35 + 0.8 * ca)[:, :, None]
        ex, ey = s.A(*EYE); zf = 1.5; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(band, img, W / 2 - (ex - s.aw / 2) * sc + (1 - a) * W * 0.9 - u * 5 * k, cy - (ey - s.ah / 2) * sc, zf)
        top = cy - hh * a + 9 * k * np.sin(s.XX * 0.022 + t * 5); bot = cy + hh * a + 9 * k * np.sin(s.XX * 0.019 - t * 4 + 2)
        m = (sstep(top - 1, top + 1, s.YY) * sstep(bot + 1, bot - 1, s.YY))[:, :, None] * (a > 0.02)
        f = f * (1 - m) + band * m
        s.glowmask(f, (np.exp(-((s.YY - top) / (1.6 * k)) ** 2) + np.exp(-((s.YY - bot) / (1.6 * k)) ** 2)) * a, FOAM, 0.9, 1.0, 0.8, 8.0)
        n = len(s._vt); step = W * 0.148
        for i, T in enumerate(s._vt):                                                # 技名: 波に のって 1文字ずつ
            q = (u - 0.5 - i * 0.11) / 0.2
            if q > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.66 + (1 - ease(q)) * 20 * k + 4 * k * math.sin(t * 4 + i), clamp(q * 1.5), 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.35 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.66 - 66 * k, 1.0, 1.0)
        if u > 1.3:
            g = ease((u - 1.3) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.66 + 50 * k), (W / 2 + W * 0.42 * g, H * 0.66 + 50 * k)], FOAM * 0.6 + TEAL * 0.4, 1.0, 0.8, 0.8)
            s.text(f, s.txt_en, W / 2, H * 0.66 + 76 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.7
        return f, 5 * math.exp(-u * 6)

    # 3) 突き立てる → 海の 中へ → 3本の 水の きりが こちらへ
    def sc_deep(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_STRIKE; ud = t - T_DIVE; sh = 0.0; A = None
        wq = ease2(clamp((ud + 0.25) / 0.55))                                        # 水面が せり上がる(0→1)
        if wq < 1:
            up = ease(u / 0.42); hit = u - 0.5
            A = s.dawn(t, 1.0 + 0.3 * up * (hit < 0))
            cy = H * (0.565 - 0.05 * up) if hit < 0 else H * (0.515 + 0.075 * ease(hit / 0.07))
            s.stand(A, t, cy, 0.40, sway=0.3, breath=0.3, lean=(-7 * k * up if hit < 0 else 6 * k), squash=(0.0 if hit < 0 else 0.07 * math.exp(-hit * 9)))
            for i, tp in enumerate(TIPS): s.flare(A, s.at(*s.A(*tp)), (8 + 6 * up) * k, FOAM * 0.6 + TEAL * 0.4, 1.0, 0.6, 0.0, False)
            if hit > 0:                                                              # 水しぶきと 輪
                fy = H * 0.565 + 0.40 * H * 0.47; L = s.L0()
                for i in range(3):
                    q = (hit - i * 0.07) / 0.5
                    if 0 < q < 1.4: cv2.ellipse(L, (int(W / 2), int(fy)), (int(W * 0.9 * q), int(H * 0.09 * q)), 0, 0, 360, float(max(0, 1 - q / 1.4) ** 1.3), max(1, int(3 * k)), cv2.LINE_AA)
                s.glowmask(A, L, FOAM, 1.0, 1.0, 1.0, 9.0)
                rg = s.rnd[:260]; ang = -1.5708 + (rg[:, 0] - 0.5) * 2.2; v0 = (0.35 + 0.9 * rg[:, 1]) * H * 1.3; x = W / 2 + np.cos(ang) * v0 * hit * 0.6; y = fy + np.sin(ang) * v0 * hit + 0.5 * H * 3.2 * hit * hit
                s.splat(A, x, y, (0.5 + rg[:, 2]) * max(0.0, 1 - hit / 0.7), FOAM, 1.4, 0.5)
                if hit < 0.1: A = A + (1 - hit / 0.1) * 0.8
                sh = 12 * math.exp(-hit * 5)
        if wq <= 0: return A, sh
        s._uw = 1.0
        TL = T_DR[2]; pl = clamp((t - TL) / (T_HIT - TL)); big = pl ** 2.5
        B = s.under(t, 1.0 - 0.35 * big); Cw = (W / 2, H * 0.20)
        img = s.pose(t, sway=0.9, breath=0.4); s.draw(B, img, Cw[0] + 3 * k * math.sin(t * 5), Cw[1] - 6 * k + 3 * k * math.cos(t * 4), 0.17, 0.9)        # 水面の 上の 影
        fr = []
        for j, (tg, dur) in enumerate((((W * 0.20, H * 0.66), 1.25), ((W * 0.80, H * 0.66), 1.25))):                                             # 左・右の きり
            p = (t - T_DR[j]) / dur
            if 0 < p < 1:
                pp = p ** 2.2; c = (Cw[0] + (j * 2 - 1) * W * 0.05 + (tg[0] - Cw[0]) * pp, Cw[1] + (tg[1] - Cw[1]) * pp)
                s.drill(B, c, k * (5 + 250 * p ** 2.7), t + j, min(1.0, p * 8) * (1 - clamp((p - 0.9) / 0.1)), Cw)
            q = t - T_DR[j] - dur
            if q > -0.05: fr.append(((q + 0.05) ** 1.2 * W * 1.5 + 60 * k, 26 * k * math.exp(-max(q, 0) * 2.5), (30 + 40 * max(q, 0)) * k)); sh += 11 * math.exp(-abs(q) * 5)
            if 0 <= q < 0.1: B = B + (1 - q / 0.1) * 0.6
        for j, (tg, dur) in enumerate((((W * 0.20, H * 0.66), 1.25), ((W * 0.80, H * 0.66), 1.25))):
            q = t - T_DR[j] - dur
            if q > -0.05 and q < 1.2: B = s.ripple(B, tg, [fr[min(j, len(fr) - 1)]])
        if pl > 0:                                                                   # まん中の 1本: 目の前まで 来て のみこむ
            c = (W / 2 + 6 * k * math.sin(t * 7) * (1 - pl), Cw[1] + H * 0.30 * ease2(pl)); R = k * (5 + 60 * pl) + W * 1.7 * big
            s.drill(B, c, R, t + 5, min(1.0, pl * 8), Cw if pl < 0.6 else None)
            s.rushdust(B, t, c, 1200, 0.6 + 2.2 * pl, FOAM, 0.2 + 0.6 * pl, 4, 1.0)
            B = s.godrays(B, c, 0.3 * pl, 1.0, 8, 0.88, 0.4); sh += 2 + 9 * big
            if pl > 0.84: B = B + ((pl - 0.84) / 0.16) ** 2 * 1.3
        if wq >= 1: return B, sh
        line = H * (1.08 - 1.2 * wq) + 12 * k * np.sin(s.XX * 0.02 + t * 9) + 6 * k * np.sin(s.XX * 0.047 - t * 13)
        m = sstep(line - 2, line + 2, s.YY)[:, :, None]; f = A * (1 - m) + B * m
        f += (np.exp(-((s.YY - line) / (5 * k)) ** 2) * 1.2)[:, :, None] * FOAM[None, None, :]
        return f, sh + 4

    # ---- この子だけの曲: 低い 海鳴りの 上に 3つの 鈴 → 突き立てる 一打と しぶき → 水の 中の こもった 和音と あわ → 3本ぶんの うなりと 3つの 衝撃 ----
    def audio(s, path):
        a = Snd(DUR, 252); E = 164.81; mn = [0, 2, 3, 5, 7, 8, 10]
        def fq(d, o=0): return E * 2 ** ((mn[d % 7] + 12 * (d // 7 + o)) / 12.0)
        sea = a.lp(a.nz, 0.04) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.35 * a.t)); a.a += sea * np.clip(a.t / 1.0, 0, 1) * a.gate(0, T_STRIKE + 0.5, 0.2) * 0.9
        a.a += a.tone(E / 2) * 0.07 * np.clip(a.t / 1.5, 0, 1) * a.gate(0, T_STRIKE + 0.5, 0.2)
        for i in range(3): a.note(fq([4, 7, 9][i], 1), 0.9 + i * 0.5, 1.0, 0.14, "bell"); a.note(fq([4, 7, 9][i], 0), 0.9 + i * 0.5, 0.6, 0.05, "sine")
        a.boom(T_CUT, 50, 1.0, 0.5); a.whoosh(T_CUT - 0.3, T_CUT, 0.3)
        for d in (0, 2, 4, 7): a.note(fq(d, 0), T_CUT, 1.9, 0.045, "pad", 0.05)
        for i in range(6): a.note(fq([0, 2, 4, 7, 4, 9][i], 1), T_CUT + 0.5 + i * 0.11, 0.4, 0.10, "bell")
        a.rise(T_STRIKE, T_STRIKE + 0.5, 150, 500, 0.10); a.whoosh(T_STRIKE + 0.3, T_STRIKE + 0.5, 0.5, 0.5)
        th = T_STRIKE + 0.5; a.boom(th, 40, 1.7, 0.7); a.a += (a.nz - a.lp(a.nz, 0.5)) * a.env(th, 0.01, 0.35) * 0.5 + a.lp(a.nz, 0.2) * a.env(th, 0.02, 0.6) * 0.6
        m = (a.t >= T_DIVE)                                                                                         # 水の 中: こもった 音
        a.a += a.lp(a.lp(a.nz, 0.02), 0.02) * m * 2.2 * np.clip((a.t - T_DIVE) / 0.3, 0, 1) * (a.t < T_HIT)
        for j, d in enumerate((0, 4, 7, 9)): a.note(fq(d, -1), T_DIVE + j * 0.08, T_HIT - T_DIVE, 0.05, "pad", 0.4)
        rg = np.random.RandomState(252)
        for i in range(46): tb = T_DIVE + rg.rand() * (T_HIT - T_DIVE); a.note(500 + rg.rand() * 1400, tb, 0.05, 0.03, "sine", 0.01)   # あわ
        for j in range(3):
            dur = 1.25 if j < 2 else T_HIT - T_DR[2]
            a.rise(T_DR[j], T_DR[j] + dur, 70 + 15 * j, 240 + 60 * j, 0.11 + 0.03 * j); a.whoosh(T_DR[j], T_DR[j] + dur, 0.45 + 0.2 * j, 0.25)
            a.note(fq([4, 7, 11][j], 1), T_DR[j], 0.6, 0.10, "bell")
            if j < 2: a.boom(T_DR[j] + dur, 42, 1.4, 0.6); a.a += a.lp(a.nz, 0.15) * a.env(T_DR[j] + dur, 0.01, 0.4) * 0.5
        for i in range(10): a.note(fq(i + 4, 0), T_DR[2] + 0.3 + i * 0.19, 0.4, 0.07, "bell")
        a.boom(T_HIT, 36, 1.7, 0.9)
        for j, d in enumerate((0, 4, 7, 11)): a.note(fq(d, 0), T_HIT + 0.02, 0.6, 0.06, "bell")
        s.fin_snd(a, DUR, 252); a.save(path)


if __name__ == "__main__":
    r = Wadatsumi()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
