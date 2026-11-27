"""アヌ(★4 冥界の番人) 専用の必殺技「冥界ノ門」— 高品質版(hs_fx)。砂漠に 冥界の 門が 立ち、とびらが ひらいて 魂の 火が あふれ、門に のみこまれる
夜の 砂漠(砂丘・ピラミッド) → うしろに 石の 門が 砂から せり上がり、きざまれた 文字が 下から 灯る(前ふり) → 目の カットイン+技名 →
杖を かかげる: とびらが 左右に ひらき、青い 魂の 火が 流れ出る → 門が こちらへ 迫り、入口が 画面を こえて 魂の 火の 海に のまれる → 白 → 暗転
使い方: python3 hs_248.py 出力.mp4 / python3 hs_248.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.6
T_CUT, T_OPEN, T_PULL, T_HIT = 2.7, 4.7, 7.0, 9.9
GOLD = np.array([1.0, 0.78, 0.30], np.float32); SOUL = np.array([0.30, 0.70, 1.0], np.float32); PALE = np.array([0.75, 0.95, 1.0], np.float32); NIGHT = np.array([0.03, 0.05, 0.14], np.float32); SAND = np.array([0.30, 0.24, 0.20], np.float32)
EYE = (340, 275); ORB = (114, 172)
SOULR = [(0.0, (0.0, 0.0, 0.0)), (0.30, (0.04, 0.14, 0.50)), (0.60, (0.18, 0.58, 1.0)), (0.85, (0.60, 0.95, 1.0)), (1.0, (1.0, 1.0, 1.0))]


class Anu(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 248, (0.35, 0.7, 1.0), "冥界ノ門", "Underworld Gate", W); s.fx_init(248)
        s._vt = [s.text_img(c, hz.SERIF, int(74 * s.k), 3) for c in "冥界ノ門"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; s.hy = 0.66

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 夜の 砂漠 ----
    def desert(s, t, lum=1.0, glow=0.0):
        W, H, k = s.W, s.H, s.k; hy = s.hy
        f = NIGHT[None, None, :] * (0.5 + 1.6 * (s.yn / hy).clip(0, 1) ** 1.5)[:, :, None] + np.array([0.06, 0.16, 0.22], np.float32)[None, None, :] * np.exp(-((s.yn - hy) / 0.12) ** 2)[:, :, None]
        f = f + s.nebula(t, [(0.04, 0.06, 0.20), (0.10, 0.08, 0.26), (0.30, 0.40, 0.60)], 0.0008, (3.0, 0.5), 0.5, 0)
        s.starfield(f, t, 900, (0.9, 0.92, 1.0), 2, 0.9, (0.0, 0.0))
        for x0, yt, sl, v in ((0.16, 0.50, 0.55, 0.9), (0.90, 0.56, 0.62, 0.7)):                                      # 遠くの ピラミッド
            dxp = (s.xn - x0) * W / H; m = sstep(-0.003, 0.003, (s.yn - yt) * sl - np.abs(dxp)) * (s.yn < hy + 0.02)
            col = np.array([0.045, 0.05, 0.085], np.float32)[None, None, :] * (0.7 + 0.9 * (dxp < 0))[:, :, None] * (0.8 + 0.4 * s.tex(2, s.XX * 0.3, s.YY * 3.0))[:, :, None] * v
            f = f * (1 - m)[:, :, None] + col * m[:, :, None]
        dune = hy + 0.020 * np.sin(s.xn * 4.2 + 0.5) + 0.010 * np.sin(s.xn * 9.5 + 2.0)
        m = sstep(dune - 0.003, dune + 0.003, s.yn); d = np.clip((s.yn - dune) / (1 - hy), 0, 1)
        rip = s.tex(1, s.XX * 0.35 + s.YY * 0.5, s.YY * 2.6 / (0.35 + d)) * 0.6 + s.tex(3, s.XX * 0.9, s.YY * 0.9) * 0.4                    # 風紋
        col = SAND[None, None, :] * (0.22 + 0.5 * rip)[:, :, None] * (0.55 + 0.7 * d)[:, :, None] * np.array([0.55, 0.7, 1.1], np.float32)[None, None, :]
        col += (np.exp(-d / 0.05) * 0.10)[:, :, None] * PALE[None, None, :]
        if glow > 0: col += (np.exp(-np.hypot((s.xn - 0.5) * 1.6, (s.yn - hy) * 2.6) / 0.30) * glow * (0.35 + 0.9 * rip))[:, :, None] * (SOUL * 0.9)[None, None, :]
        f = f * (1 - m)[:, :, None] + col * m[:, :, None]
        return (f * lum).astype(np.float32)

    # ---- 冥界の 門: 2本の 塔・かもい・とびら・きざまれた 文字・中の 魂の 火 ----
    def gate(s, f, c, S, t, opn=0.0, lit=1.0, v=1.0, clip_y=None):
        W, H, k = s.W, s.H, s.k; x = (s.XX - c[0]) / S; y = (s.YY - c[1]) / S; ax = np.abs(x)
        vis = (s.YY < clip_y).astype(np.float32) if clip_y is not None else 1.0
        tw = 0.36 + 0.05 * (y + 1) * 0.5                                             # 塔は 下ほど 太い
        tower = sstep(0.42 - 0.004, 0.42 + 0.004, ax) * sstep(0.42 + tw + 0.004, 0.42 + tw - 0.004, ax) * (y > -1.0) * (y < 1.02)
        lint = (ax < 0.90) * (y > -1.0) * (y < -0.76); corn = (ax < 0.98 - 0.05 * (y + 1.12) / 0.12) * (y > -1.12) * (y <= -1.0)
        door = (ax < 0.42) * (y >= -0.76) * (y < 1.02)
        stone = np.clip(tower + lint + corn, 0, 1) * vis * v
        n = s.tex(2, x * 90, y * 90) * 0.5 + s.tex(0, x * 30, y * 30) * 0.5; blk = 1 - 0.5 * sstep(0.92, 0.99, np.abs(np.cos(y * 22.0))) - 0.35 * sstep(0.95, 0.99, np.abs(np.cos(x * 30.0 + np.floor(y * 7.0) * 1.3)))
        shade = (0.30 + 0.5 * n) * blk * (0.75 + 0.5 * sstep(0.42, 0.50, ax) * (ax < 0.6)) * (1 + 0.5 * lint)
        scol = np.array([0.20, 0.19, 0.26], np.float32)[None, None, :] * shade[:, :, None]
        gl = sstep(0.60, 0.80, s.tex(3, np.floor(x * 26) * 37.0 + np.floor(y * 16) * 11.0, np.floor(y * 16) * 53.0)) * (np.abs(np.cos(x * 26 * 3.1416)) < 0.75) * (np.abs(np.cos(y * 16 * 3.1416)) < 0.70)     # 文字(ます目ごとに ある/ない)
        gl = gl * (tower * (ax > 0.50) * (ax < 0.72) + lint * (ax < 0.8) * (y > -0.95) * (y < -0.81)) * sstep(1.0 - 2.1 * lit + 0.1, 1.0 - 2.1 * lit - 0.1, y) * vis * v
        f *= (1 - stone)[:, :, None]; f += scol * stone[:, :, None]
        f += (gl * (0.9 + 0.3 * math.sin(t * 3)))[:, :, None] * (SOUL * 0.6 + GOLD * 0.4)[None, None, :] + (cv2.GaussianBlur(gl, (0, 0), 5 * k) * 0.8)[:, :, None] * SOUL[None, None, :]
        dm = door * vis * v; gap = sstep(0.42 * opn + 0.004, 0.42 * opn - 0.004, ax) * (opn > 0.001)                    # ひらいた すき間
        dn = s.tex(1, (ax - 0.42 * opn) * 60, y * 40) * 0.6 + n * 0.4; dcol = np.array([0.16, 0.13, 0.12], np.float32)[None, None, :] * ((0.35 + 0.55 * dn) * (1 - 0.5 * sstep(0.9, 0.99, np.abs(np.cos(y * 9.0)))))[:, :, None]
        dcol += (sstep(0.03, 0.0, np.abs(ax - 0.42 * opn - 0.02)) * 0.5)[:, :, None] * GOLD[None, None, :] * (opn < 0.98)
        rI = np.hypot(x, (y - 0.15) * 0.75) + 1e-3; thI = np.arctan2((y - 0.15) * 0.75, x); lg = np.log(rI + 0.03)
        st1 = s.tex(1, thI / 6.2832 * 768, lg * 50 - t * 170); st2 = s.tex(3, thI / 6.2832 * 1536 + 20, lg * 120 - t * 380)
        inner = s.ramp(np.clip(0.22 + 0.80 * np.exp(-(rI / 0.40) ** 2) + (0.50 * st1 ** 1.5 + 0.35 * st2 ** 3) * sstep(0.04, 0.35, rI), 0, 1), SOULR) * 1.35
        f *= (1 - dm)[:, :, None]; f += (dcol * (1 - gap)[:, :, None] + inner * gap[:, :, None]) * dm[:, :, None]
        if opn > 0:                                                                  # もれる 光
            f += (np.exp(-(ax / (0.42 * opn + 0.25)) ** 2) * np.exp(-np.clip(y - 1.0, 0, None) / 0.5) * (y > 0.9) * 0.55 * opn * v)[:, :, None] * SOUL[None, None, :]
            f += (np.exp(-(np.hypot(x, (y - 0.1) * 0.6) / (0.5 + 0.5 * opn)) ** 2) * 0.22 * opn * v)[:, :, None] * SOUL[None, None, :]

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        w = getattr(s, '_gl', 0.0)
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(PALE * (1 - 0.3 * w) + SOUL * 0.3 * w), rim_dir=(0.0, -1.0), rim_v=(0.35 + 0.55 * w) * alpha, amb=(0.6, 0.8, 1.0), amb_v=0.18, aura=SOUL, aura_v=0.12 * w, alpha=alpha, shade=0.80 - 0.12 * w)

    def frame(s, t):
        o = {}; s._gl = 0.0
        if t < T_CUT: f, sh = s.sc_rise(t)
        elif t < T_OPEN: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.92)
        else: f, sh = s.sc_open(t)
        f, sh = s.finale(f, sh, t, DUR, None, (0.5, 0.85, 1), (0.2, 0.5, 1.0), -1)
        return s.film(f, t, sh * s.k, 0.6, o.get("thr", 0.80), (0.8, 0.92, 1.0), 1.0 + 0.15 * min(sh, 8), 0.04, o.get("vig", 0.58), 1.15, 1.10)

    def keeper(s, f, t, alpha=1.0, **kw):
        W, H, k = s.W, s.H, s.k; cy = H * 0.70; fy = cy + 0.40 * H * 0.47
        sh_ = s.L0(); cv2.ellipse(sh_, (int(W / 2), int(fy)), (int(W * 0.17), int(H * 0.012)), 0, 0, 360, 0.6 * alpha, -1, cv2.LINE_AA); f *= (1 - cv2.GaussianBlur(sh_, (0, 0), 4 * k))[:, :, None]
        img = s.pose(t, **kw); s.draw(f, img, W / 2, cy, 0.40, alpha)
        s.flare(f, s.at(*s.A(*ORB)), (7 + 3 * math.sin(t * 6) ** 2 + 10 * s._gl) * k, SOUL * 0.6 + PALE * 0.4, (0.6 + 0.4 * s._gl) * alpha, 0.6, t, False)

    GC = (0.5, 0.385); GS = 0.265

    # 1) 夜の 砂漠。うしろに 門が せり上がり、文字が 下から 灯る
    def sc_rise(s, t):
        W, H, k = s.W, s.H, s.k; r = ease2((t - 0.5) / 1.5); lit = clamp((t - 1.5) / 1.1)
        f = s.desert(t, 1.0, 0.15 * lit); S = H * s.GS; c = (W * s.GC[0], H * s.GC[1] + (1 - r) * S * 2.1)
        if r > 0:
            s.gate(f, c, S, t, 0.0, lit, 1.0, H * (s.hy + 0.012))
            if r < 1:
                rg = s.rnd[:200]; x = W * (0.5 + (rg[:, 0] - 0.5) * 0.85); y = H * s.hy - rg[:, 1] * H * 0.05 * (1 + 2 * rg[:, 2]) * math.sin(r * 3.14); s.splat(f, x, y, rg[:, 3] * 0.5 * math.sin(r * 3.14), SAND * 2.5, 2.2, 0.8)   # 砂けむり
        s._gl = 0.2 * lit; s.keeper(f, t, 1.0, sway=0.15, breath=0.5)
        sh = 2.5 * math.sin(r * 3.14) * (0 < r < 1)
        return f * clamp(t / 0.5), sh

    # 2) 目の カットイン(だ円の わく)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.desert(t, 0.30, 0.15); s.gate(f, (W * s.GC[0], H * s.GC[1]), H * s.GS, t, 0.0, 1.0, 0.35, H * (s.hy + 0.012))
        a = ease(u / 0.28); cy = H * 0.33; hh = H * 0.105
        band = np.array([0.03, 0.06, 0.16], np.float32)[None, None, :] + s.ramp(np.clip(s.flame(t, np.ones((H, W), np.float32), 300.0, 0.002, 0.9), 0, 1), SOULR) * 0.55
        ex, ey = s.A(*EYE); zf = 1.5; sc = zf * H / s.ah; s._gl = 0.7
        img = s.pose(t, breath=0.2); s.draw(band, img, W / 2 - (ex - s.aw / 2) * sc + (1 - a) * W * 0.9 - u * 5 * k, cy - (ey - s.ah / 2) * sc, zf)
        q = (np.abs(s.XX - W / 2) / (W * 0.47)) ** 4 + (np.abs(s.YY - cy) / (hh * max(a, 0.02))) ** 4
        m = sstep(1.0, 0.96, q)[:, :, None]; f = f * (1 - m) + band * m
        s.glowmask(f, np.exp(-((q - 1.0) / 0.05) ** 2) * (a > 0.05), GOLD, 0.9 * a, 1.0, 0.8, 8.0)
        n = len(s._vt); step = W * 0.20
        for i, T in enumerate(s._vt):                                                # 技名: 下から せり上がる(門と 同じ)
            p = (u - 0.5 - i * 0.14) / 0.3
            if p > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.66 + (1 - ease(p)) * 40 * k, clamp(p * 1.5), 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.35 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.66 - 76 * k, 1.0, 1.0)
        if u > 1.3:
            g = ease((u - 1.3) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.66 + 58 * k), (W / 2 + W * 0.42 * g, H * 0.66 + 58 * k)], GOLD, 1.0, 0.8, 0.8)
            s.text(f, s.txt_en, W / 2, H * 0.66 + 84 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.7
        return f, 5 * math.exp(-u * 6)

    # 3) とびらが ひらく → 魂の 火が 流れ出る → 門が 迫って のみこまれる
    def sc_open(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_OPEN; opn = ease2((u - 0.5) / 1.7); e = clamp((t - T_PULL) / (T_HIT - T_PULL)); ee = e ** 2.4
        s._gl = clamp(u / 0.5)
        f = s.desert(t, 1.0 - 0.3 * e, 0.15 + 0.85 * opn)
        S = H * s.GS * (1 + 8.5 * ee); c = (W * 0.5, H * (s.GC[1] + 0.10 * ease2(e)) - (S - H * s.GS) * 0.12)
        s.gate(f, c, S, t, opn, 1.0, 1.0, H * (s.hy + 0.012) + (H * 0.6) * clamp(e * 4))
        if opn > 0.05:                                                               # 魂の 火: 入口から こちらへ
            src = (c[0], c[1] + S * 0.25)
            s.rushdust(f, t, src, 1300, 0.45 + 2.3 * e, SOUL * 0.7 + PALE * 0.3, (0.25 + 0.6 * e) * opn, 6, 1.0)
            rg = s.rnd[:26]; p = (u * (0.28 + 0.25 * rg[:, 1]) + rg[:, 0]) % 1.0; a = rg[:, 2] * 6.2832; rr = p ** 2.0 * W * (0.5 + 0.9 * rg[:, 3])
            x = src[0] + np.cos(a) * rr + np.sin(t * 3 + rg[:, 4] * 9) * 12 * k; y = src[1] + np.sin(a) * rr * 1.2 - p * H * 0.08
            B = np.zeros((H, W), np.float32)
            for i in range(26):
                if p[i] < 0.04: continue
                cv2.circle(B, (int(x[i]), int(y[i])), max(1, int((3 + 20 * p[i] ** 1.5) * k)), float(min(1.0, p[i] * 5) * (1 - p[i]) * 1.2 * opn), -1, cv2.LINE_AA)
            B = cv2.GaussianBlur(B, (0, 0), 4 * k); f += s.ramp(np.clip(B, 0, 1), SOULR) * 1.2
        a_ = 1 - clamp(e / 0.28)
        if a_ > 0.02: s.keeper(f, t, a_, sway=0.15 + 0.5 * opn, breath=0.4, lean=-5 * k * ease(u / 0.5), squash=-0.03 * ease(u / 0.4))
        sh = 2.5 * (0.5 < u < 2.2) * abs(math.sin(u * 23)) + 2 * e + 9 * ee
        f = s.godrays(f, (c[0], c[1] + S * 0.1), 0.25 * opn + 0.35 * e, 0.95, 8, 0.88, 0.45)
        if u < 0.12: f = f + (1 - u / 0.12) * 0.4
        if e > 0.84: f = f + ((e - 0.84) / 0.16) ** 2 * 1.4
        return f, sh

    # ---- この子だけの曲: 低い 持続と 砂の 風 → 石の ずり上がる 音 → とびらの 石ずれ → 魂の 声(ゆれる 高い 音)が ふえて ふくらむ ----
    def audio(s, path):
        a = Snd(DUR, 248); D = 146.83; ph = [0, 1, 4, 5, 7, 8, 10]                                                    # 中東ふうの 音階
        def fq(d, o=0): return D * 2 ** ((ph[d % 7] + 12 * (d // 7 + o)) / 12.0)
        a.a += (a.tone(D / 2) + 0.5 * a.tone(D / 2 * 1.5)) * 0.08 * np.clip(a.t / 1.5, 0, 1) * a.gate(0, T_HIT, 0.3)
        a.a += (a.lp(a.nz, 0.10) - a.lp(a.nz, 0.03)) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.3 * a.t)) * a.gate(0, T_OPEN, 0.3) * 0.6
        a.a += a.lp(a.nz, 0.03) * a.gate(0.5, 2.0, 0.2) * (0.6 + 0.4 * np.sin(2 * np.pi * 17 * a.t)) * 1.3; a.boom(2.0, 42, 1.2, 0.5)     # 門が せり上がる
        for i in range(6): a.note(fq(i, 1), 1.5 + i * 0.18, 0.5, 0.08, "bell")
        a.boom(T_CUT, 46, 1.1, 0.5); a.whoosh(T_CUT - 0.3, T_CUT, 0.3)
        for d in (0, 2, 4): a.note(fq(d, 0), T_CUT, 1.9, 0.05, "saw", 0.05)
        for i in range(4): a.note(fq([0, 2, 4, 7][i], 1), T_CUT + 0.5 + i * 0.14, 0.5, 0.11, "bell"); a.boom(T_CUT + 0.5 + i * 0.14, 70, 0.4, 0.1)
        mel = [7, 8, 7, 5, 4, 5, 7, 4]
        for i, d in enumerate(mel): a.note(fq(d, 0), T_CUT + 1.0 + i * 0.12, 0.2, 0.05, "fm")
        a.note(fq(11, 1), T_OPEN, 0.8, 0.12, "bell"); a.boom(T_OPEN + 0.5, 40, 1.0, 0.4)
        a.a += a.lp(a.nz, 0.025) * a.gate(T_OPEN + 0.5, T_OPEN + 2.2, 0.15) * (0.55 + 0.45 * np.sin(2 * np.pi * 23 * a.t)) * 1.6                 # 石の とびら
        a.boom(T_OPEN + 2.2, 44, 1.4, 0.6)
        rg = np.random.RandomState(248)
        for i in range(30):                                                                                           # 魂の 声
            tk = T_OPEN + 0.8 + rg.rand() * (T_HIT - T_OPEN - 1.0); f0 = fq(int(rg.rand() * 7) + 7, 0)
            a.a += np.sin(2 * np.pi * f0 * a.t + 3 * np.sin(2 * np.pi * 5.5 * a.t)) * a.env(tk, 0.25, 0.5) * 0.022 * (1 + (tk - T_OPEN) / 3)
        for j, d in enumerate((0, 4, 7, 8)): a.note(fq(d, -1), T_OPEN + 0.5 + j * 0.2, T_HIT - T_OPEN, 0.05, "pad", 0.5)
        a.rise(T_PULL, T_HIT, 60, 250, 0.15); a.whoosh(T_PULL, T_HIT, 0.9, 0.3)
        a.a += a.lp(a.nz, 0.06) * np.clip((a.t - T_PULL) / (T_HIT - T_PULL), 0, 1) ** 2 * (a.t < T_HIT) * (a.t > T_PULL) * 0.8
        for i in range(10): a.note(fq(i + 4, 0), T_PULL + 0.2 + i * 0.26, 0.5, 0.08, "bell")
        a.boom(T_HIT, 36, 1.7, 0.9)
        for j, d in enumerate((0, 4, 7, 11)): a.note(fq(d, 0), T_HIT + 0.02, 0.6, 0.06, "bell")
        s.fin_snd(a, DUR, 248); a.save(path)


if __name__ == "__main__":
    r = Anu()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
