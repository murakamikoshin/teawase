"""フリムニル(★5 霜の巨人の王) 専用の必殺技「氷河砕キ」— 高品質版(hs_fx)。こちらが 氷に 閉じこめられ、氷の 斧で 画面ごと 砕かれる
ふぶきの 氷河に 立つ(画面の すみに 霜が つきはじめる=前ふり) → 目の カットイン+技名 →
こちらへ 歩いて くる。霜が ふちから まん中へ のびて 画面が 凍る(氷ごしに 青く にじむ) → 斧を ふり上げ、ふり下ろす:
画面に ひびが 走り、氷の かけらに 割れて こちらへ 飛び散る。すき間から 白い 光 → 白 → 暗転
使い方: python3 hs_253.py 出力.mp4 / python3 hs_253.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.5
T_CUT, T_FREEZE, T_RAISE, T_STRIKE, T_BREAK, T_HIT = 2.7, 4.7, 6.9, 7.5, 7.85, 9.8
ICE = np.array([0.55, 0.82, 1.0], np.float32); FROST = np.array([0.86, 0.95, 1.0], np.float32); DEEPB = np.array([0.04, 0.08, 0.18], np.float32); SNOW = np.array([0.62, 0.72, 0.88], np.float32)
EYE = (337, 246); BLADE = (720, 268)


class Hrimnir(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 253, (0.6, 0.85, 1.0), "氷河砕キ", "Glacier Breaker", W); s.fx_init(253)
        s._vt = [s.text_img(c, hz.SERIF, int(74 * s.k), 3) for c in "氷河砕キ"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; H = s.H; k = s.k
        rg = np.random.RandomState(2530); s._sn = rg.rand(700, 4).astype(np.float32)
        s.P = (W * 0.5, H * 0.45)                                                    # 斧が 当たる 点
        pts = [(s.P[0], s.P[1])]
        for j, (rr, n) in enumerate(((0.11, 5), (0.26, 8), (0.46, 10), (0.74, 12), (1.10, 13))):   # かけらの たね(輪に ならべて ずらす)
            for i in range(n):
                a = (i + rg.rand() * 0.7) * 6.2832 / n + j * 0.5; r = rr * W * (0.8 + 0.4 * rg.rand()); pts.append((s.P[0] + math.cos(a) * r, s.P[1] + math.sin(a) * r * 1.25))
        s._seed = np.array(pts, np.float32); K = len(pts)
        d = (s.XX[:, :, None] - s._seed[None, None, :, 0]) ** 2 + (s.YY[:, :, None] - s._seed[None, None, :, 1]) ** 2
        s._lab = np.argmin(d, 2).astype(np.float32); d.sort(2); s._edge = np.exp(-((np.sqrt(d[:, :, 1]) - np.sqrt(d[:, :, 0])) / (1.6 * k)) ** 2).astype(np.float32)   # ひびの 線(2番目に 近い たねとの 差)
        s._edge2 = np.exp(-((np.sqrt(d[:, :, 1]) - np.sqrt(d[:, :, 0])) / (7 * k)) ** 2).astype(np.float32)
        v = s._seed - np.array(s.P, np.float32)[None, :]; n_ = np.hypot(v[:, 0], v[:, 1]) + 1e-3; s._dir = v / n_[:, None]; s._sr = rg.rand(K, 3).astype(np.float32)
        s._dP = np.hypot(s.XX - s.P[0], s.YY - s.P[1])
        # 霜の もと(動かない 模様)
        dx = s.xn - 0.5; dy = s.yn - 0.46; s._rr = np.hypot(dx * 1.15, dy * 0.95) * 2; th = np.arctan2(dy, dx)
        s._fd = (s.tex(1, th / 6.2832 * 2048, s._rr * 22) * 0.75 + s.tex(2, s.XX * 0.7, s.YY * 0.7) * 0.25).astype(np.float32)
        rd = 1 - np.abs(2 * s.tex(3, s.XX * 6.5, s.YY * 6.5) - 1); st_ = s.tex(0, th / 6.2832 * 3072, s._rr * 30) ; s._fc = (0.34 + 0.22 * rd ** 3 + 0.75 * st_ ** 1.6 + 0.25 * s.tex(2, th / 6.2832 * 1024, s._rr * 120) ** 2).astype(np.float32)
        s._wall = (0.40 + 0.07 * s.tex(1, s.XX * 0.35, 7.0) + 0.03 * s.tex(3, s.XX * 1.4, 90.0)).astype(np.float32)
        s._wtex = (s.tex(2, s.XX * 2.2, s.YY * 0.22) * 0.6 + s.tex(0, s.XX * 0.9, s.YY * 0.5) * 0.4).astype(np.float32)

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- ふぶきの 氷河 ----
    def field(s, t, lum=1.0):
        W, H, k = s.W, s.H, s.k; hy = 0.63
        f = DEEPB[None, None, :] * (0.7 + 2.6 * (s.yn / hy).clip(0, 1) ** 1.6)[:, :, None] + np.array([0.20, 0.30, 0.44], np.float32)[None, None, :] * np.exp(-((s.yn - hy) / 0.16) ** 2)[:, :, None]
        cl = s.fbm(0.0015, t * 22, t * 3, 4, 50 * k, 1); f += (sstep(0.40, 0.85, cl) * 0.22 * (1 - s.yn))[:, :, None] * np.array([0.5, 0.62, 0.8], np.float32)[None, None, :]
        wm = sstep(s._wall - 0.003, s._wall + 0.003, s.yn) * (s.yn < hy + 0.01); dw = np.clip((s.yn - s._wall) / 0.22, 0, 1)                        # 氷河の 壁
        wc = (ICE * 0.55)[None, None, :] * (0.30 + 0.9 * s._wtex)[:, :, None] * (1.15 - 0.6 * dw)[:, :, None] + (FROST * 0.55)[None, None, :] * np.exp(-dw / 0.06)[:, :, None]
        wc *= (1 - 0.6 * sstep(0.78, 0.9, s.tex(1, s.XX * 3.1, s.YY * 0.12)))[:, :, None]                                                         # たての 割れ目
        f = f * (1 - wm)[:, :, None] + wc * wm[:, :, None]
        gm = sstep(hy - 0.003, hy + 0.003, s.yn); d = np.clip((s.yn - hy) / (1 - hy), 0, 1); z = 1.0 / (d + 0.05)
        gn = s.tex(0, (s.xn - 0.5) * z * 50 + t * 12, z * 30) * 0.6 + s.tex(2, s.XX * 0.6, s.YY * 0.6) * 0.4
        gc = SNOW[None, None, :] * (0.38 + 0.5 * gn)[:, :, None] * (0.55 + 0.6 * d)[:, :, None] + (ICE * 0.25)[None, None, :] * np.exp(-d / 0.08)[:, :, None]
        f = f * (1 - gm)[:, :, None] + gc * gm[:, :, None]
        sn = s._sn; x = (sn[:, 0] * W * 1.3 + t * W * (0.9 + 0.9 * sn[:, 2])) % (W * 1.3) - W * 0.15; y = (sn[:, 1] * H + t * H * (0.18 + 0.2 * sn[:, 3]) + np.sin(t * 3 + sn[:, 2] * 30) * 14 * k) % H
        for q in (0.0, 0.012, 0.024): s.splat(f, x - q * W, y - q * H * 0.22, (0.25 + 0.75 * sn[:, 3]) * (0.9 - q * 20), FROST, 1.0, 0.25)       # ふぶき(尾を ひく)
        return (f * lum).astype(np.float32)

    # ---- 霜: ふちから まん中へ のびる ----
    def frost(s, f, amt, v=1.0):
        if amt <= 0.005: return f
        thr = 1.75 - amt * 1.10; m = sstep(0.0, 0.10, s._rr + s._fd * 0.36 - thr) * v
        deep = sstep(0.0, 0.5, s._rr + s._fd * 0.36 - thr)
        col = FROST[None, None, :] * (s._fc * (0.55 + 0.45 * deep))[:, :, None] + (ICE * 0.25)[None, None, :]
        f = f * (1 - m * (0.55 + 0.4 * deep))[:, :, None] + col * (m * 0.9)[:, :, None]
        ice = clamp((amt - 0.45) / 0.5) * v                                          # 凍ると 氷ごしの 青い にじみ
        if ice > 0:
            g = f.mean(2, keepdims=True); f = f * (1 - 0.30 * ice) + (g * (ICE * 1.25)[None, None, :]) * 0.30 * ice + (ICE * 0.05 * ice)[None, None, :]
        return f

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(FROST), rim_dir=(-0.6, -0.8), rim_v=0.6 * alpha, amb=(0.6, 0.8, 1.0), amb_v=0.20, aura=ICE, aura_v=0.10, alpha=alpha, shade=0.86)

    def frame(s, t):
        o = {}
        if t < T_CUT: f, sh = s.sc_field(t)
        elif t < T_FREEZE: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.92)
        else: f, sh = s.sc_break(t)
        f, sh = s.finale(f, sh, t, DUR, None, (0.7, 0.9, 1), (0.4, 0.7, 1.0), -1)
        return s.film(f, t, sh * s.k, 0.5, o.get("thr", 0.86), (0.85, 0.95, 1.0), 1.0 + 0.15 * min(sh, 8), 0.035, o.get("vig", 0.55), 1.10, 1.08)

    def giant(s, f, t, size, cy, **kw):
        W, H, k = s.W, s.H, s.k; fy = cy + size * H * 0.44
        sh_ = s.L0(); cv2.ellipse(sh_, (int(W / 2), int(fy)), (int(W * 0.42 * size / 0.42), int(H * 0.016 * size / 0.42)), 0, 0, 360, 0.6, -1, cv2.LINE_AA); f *= (1 - cv2.GaussianBlur(sh_, (0, 0), 5 * k))[:, :, None]
        img = s.pose(t, **kw); s.draw(f, img, W / 2, cy, size)

    # 1) ふぶきの 氷河。画面の すみに 霜が つきはじめる
    def sc_field(s, t):
        W, H, k = s.W, s.H, s.k
        f = s.field(t); s.giant(f, t, 0.40, H * 0.60, sway=0.12, breath=0.7)
        if t > 1.2: s.flare(f, s.at(*s.A(*BLADE)), (8 + 10 * math.exp(-(t - 1.2) * 4)) * k, FROST, clamp((t - 1.2) * 5), 0.8, 0.4, False)
        f = s.frost(f, 0.16 * ease((t - 0.8) / 1.8))
        return f * clamp(t / 0.5), 0

    # 2) 目の カットイン(氷の かけらの 形の 窓)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.frost(s.field(t, 0.32), 0.16)
        a = ease(u / 0.26); cy = H * 0.33; hh = H * 0.115
        band = (DEEPB * 2.0)[None, None, :] + (ICE * 0.5)[None, None, :] * (0.3 + 0.8 * s._wtex)[:, :, None]
        ex, ey = s.A(*EYE); zf = 1.55; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(band, img, W / 2 - (ex - s.aw / 2) * sc - (1 - a) * W * 0.9 + u * 5 * k, cy - (ey - s.ah / 2) * sc, zf)
        edge = hh * a * np.clip((W * 0.58 - np.abs(s.XX - W / 2 - (s.YY - cy) * 0.25)) / (W * 0.17), 0, 1); dd = np.abs(s.YY - cy)
        m = (sstep(edge, edge - 2, dd) * (edge > 1))[:, :, None]
        f = f * (1 - m) + band * m
        s.glowmask(f, np.exp(-((dd - edge) / (1.6 * k)) ** 2) * (edge > 1) * (dd < hh * 1.2), FROST, 0.9 * a, 1.0, 0.8, 8.0)
        n = len(s._vt); step = W * 0.20
        for i, T in enumerate(s._vt):                                                # 技名: 1文字ずつ 落ちて きて 止まる(重い)
            q = (u - 0.5 - i * 0.16) / 0.14
            if q > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.66 - (1 - ease(q)) * 60 * k, clamp(q * 2), 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.35 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.66 - 76 * k, 1.0, 1.0)
        if u > 1.3:
            g = ease((u - 1.3) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.66 + 58 * k), (W / 2 + W * 0.42 * g, H * 0.66 + 58 * k)], FROST, 1.0, 0.8, 0.8)
            s.text(f, s.txt_en, W / 2, H * 0.66 + 84 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.7
        sh = 5 * math.exp(-u * 6) + sum(6 * math.exp(-max(0, u - 0.64 - i * 0.16) * 14) * (u > 0.64 + i * 0.16) for i in range(4))
        return f, sh

    # 3) 歩いて くる → 画面が 凍る → 斧で 砕く
    def sc_break(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_FREEZE; ur = t - T_RAISE; us = t - T_STRIKE; ub = t - T_BREAK
        ap = ease2(u / 2.2); size = 0.40 + 0.22 * ap; cy = H * (0.60 - 0.03 * ap); step = math.sin(u * 5.2) * (1 - ap)
        f = s.field(t, 1.0 - 0.25 * ap)
        if ur < 0: s.giant(f, t, size, cy + abs(step) * 6 * k, sway=0.12, breath=0.5, lean=4 * k * step)
        elif us < 0: g = ease(ur / 0.5); s.giant(f, t, size + 0.02 * g, cy - 0.02 * H * g, sway=0.1, lean=-7 * k * g, squash=-0.07 * g)          # ふり上げる
        else: s.giant(f, t, size + 0.03, cy + 0.012 * H, lean=9 * k, squash=0.09 * math.exp(-us * 7), jit=2 * k * math.exp(-us * 5))              # ふり下ろす
        bp = s.at(*s.A(*BLADE)); s.flare(f, bp, (9 + 4 * math.sin(t * 9) ** 2 + (16 * ease(ur / 0.5) if ur > 0 else 0)) * k, FROST, 0.9, 0.8, 0.4, False)
        f = s.frost(f, 0.16 + 0.84 * ease(u / 2.0))
        sh = 1.5 * abs(step) * (ur < 0)
        if us < 0: return f, sh
        # ひび
        cg = clamp(us / 0.22); rv = sstep(cg * W * 1.6 + 30 * k, cg * W * 1.6 - 30 * k, s._dP)
        f = f * (1 - 0.55 * s._edge * rv)[:, :, None] + ((s._edge * 1.3 + s._edge2 * 0.35) * rv)[:, :, None] * FROST[None, None, :]
        f += (np.exp(-(s._dP / (W * (0.10 + 0.25 * cg))) ** 2) * 1.6 * math.exp(-us * 5))[:, :, None] * FROST[None, None, :]
        if us < 0.08: f = f + (1 - us / 0.08) * 1.0
        sh = 16 * math.exp(-us * 5)
        if ub < 0: return f, sh
        # 割れて こちらへ 飛び散る
        g = clamp(ub / (T_HIT - T_BREAK)); z = 1 + 2.6 * g ** 2.0
        sx = s.P[0] + (s.XX - s.P[0]) / z; sy = s.P[1] + (s.YY - s.P[1]) / z
        lab0 = cv2.remap(s._lab, sx, sy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE).astype(np.int32)
        amp = (6 + 300 * g ** 1.5) * k * (0.45 + 0.9 * s._sr[:, 0]); ox = (s._dir[:, 0] * amp)[lab0] + (np.sin(t * 3 + s._sr[:, 1] * 9) * 10 * k * g)[lab0]; oy = (s._dir[:, 1] * amp * 1.1)[lab0] + (220 * k * g ** 2 * (s._sr[:, 2] - 0.3))[lab0]
        qx = (sx - ox / z).astype(np.float32); qy = (sy - oy / z).astype(np.float32)
        lab1 = cv2.remap(s._lab, qx, qy, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=-1).astype(np.int32)
        ok = (lab1 == lab0).astype(np.float32); ok = cv2.GaussianBlur(ok, (0, 0), 0.8 * k)
        out = cv2.remap(f, qx, qy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        glint = (0.80 + 0.55 * np.sin(t * 7 + s._sr[:, 1] * 12) ** 2)[lab0]                                                                    # かけらごとに 光り方が ちがう
        light = (FROST * (1.25 + 0.9 * g))[None, None, :] * (0.75 + 0.5 * np.exp(-(s._dP / (W * 0.8)) ** 2))[:, :, None]
        rim = np.clip(cv2.GaussianBlur(ok, (0, 0), 3 * k) - ok * 0.0, 0, 1) * (1 - ok)
        f = out * (glint * ok)[:, :, None] + light * (1 - ok)[:, :, None]
        f += (np.clip(ok - cv2.erode(ok, np.ones((3, 3), np.uint8)), 0, 1) * 1.5)[:, :, None] * FROST[None, None, :]
        s.rushdust(f, t, s.P, 1300, 0.8 + 2.0 * g, FROST, 0.3 + 0.6 * g, 4, 1.0)
        f = s.godrays(f, s.P, 0.30 + 0.4 * g, 1.0, 8, 0.88, 0.45)
        if ub < 0.1: f = f + (1 - ub / 0.1) * 0.8
        if g > 0.80: f = f + ((g - 0.80) / 0.20) ** 2 * 1.4
        return f, 10 * math.exp(-ub * 3) + 2 + 7 * g ** 2

    # ---- この子だけの曲: 風の 音と 低い 角笛 → 霜の こまかい 鈴(だんだん ふえる)→ 無音の 一拍 → 一撃と 氷の 割れる 音 → ガラスの 雨 ----
    def audio(s, path):
        a = Snd(DUR, 253); C = 130.81; mn = [0, 2, 3, 5, 7, 8, 10]
        def fq(d, o=0): return C * 2 ** ((mn[d % 7] + 12 * (d // 7 + o)) / 12.0)
        wind = (a.lp(a.nz, 0.12) - a.lp(a.nz, 0.02)) * (0.55 + 0.45 * np.sin(2 * np.pi * 0.23 * a.t + 1.0)) * (0.8 + 0.2 * np.sin(2 * np.pi * 1.7 * a.t))
        a.a += wind * np.clip(a.t / 0.8, 0, 1) * a.gate(0, T_STRIKE, 0.1) * 1.1
        a.note(fq(0, -1), 0.4, 2.2, 0.10, "saw", 0.4); a.note(fq(4, -1), 1.4, 1.4, 0.08, "saw", 0.3)                         # 角笛
        a.note(fq(11, 1), 1.2, 0.6, 0.10, "bell")
        a.boom(T_CUT, 44, 1.2, 0.5); a.whoosh(T_CUT - 0.3, T_CUT, 0.3)
        for d in (0, 2, 4): a.note(fq(d, -1), T_CUT, 1.9, 0.07, "saw", 0.05)
        for i in range(4): a.boom(T_CUT + 0.64 + i * 0.16, 62 - i * 5, 0.9, 0.18); a.note(fq([0, 4, 7, 11][i], 0), T_CUT + 0.64 + i * 0.16, 0.5, 0.09, "bell")
        rg = np.random.RandomState(253); n = 70
        for i in range(n):                                                                                                   # 霜が のびる: 高い 細かい 音が ふえていく
            tk = T_FREEZE + 0.2 + (T_RAISE - T_FREEZE) * (i / n) ** 0.6; a.note(1500 + rg.rand() * 3500, tk, 0.05, 0.02 + 0.03 * i / n, "sine", 0.002)
        for i in range(5): a.boom(T_FREEZE + 0.3 + i * 0.6 * (1 - i * 0.05), 55, 0.5, 0.15)                                    # 足音
        a.rise(T_FREEZE, T_RAISE, 60, 95, 0.10)
        for j, d in enumerate((0, 4, 7)): a.note(fq(d, -1), T_FREEZE + j * 0.4, T_RAISE - T_FREEZE, 0.05, "pad", 0.5)
        a.rise(T_RAISE, T_STRIKE, 200, 900, 0.10); a.whoosh(T_STRIKE - 0.18, T_STRIKE, 0.6, 0.5)
        a.boom(T_STRIKE, 36, 2.0, 0.9); a.boom(T_STRIKE, 90, 0.9, 0.2)
        for i in range(6): tk = T_STRIKE + i * 0.045; a.a += (a.nz - a.lp(a.nz, 0.7)) * a.env(tk, 0.001, 0.03) * (0.7 - i * 0.08)   # ぴしぴし
        a.a += (a.nz - a.lp(a.nz, 0.5)) * a.env(T_BREAK, 0.005, 0.5) * 0.6 + a.lp(a.nz, 0.15) * a.env(T_BREAK, 0.01, 0.7) * 0.6; a.boom(T_BREAK, 48, 1.3, 0.5)
        for i in range(60): tk = T_BREAK + 0.05 + rg.rand() * (T_HIT - T_BREAK); a.note(1200 + rg.rand() * 4200, tk, 0.07, 0.035, "bell", 0.001)   # ガラスの 雨
        for j, d in enumerate((0, 4, 7, 9, 11, 14)): a.note(fq(d, 0), T_BREAK + j * 0.05, T_HIT - T_BREAK, 0.045, "pad", 0.3)
        a.rise(T_BREAK, T_HIT, 90, 320, 0.12); a.whoosh(T_BREAK + 0.3, T_HIT, 0.8, 0.3)
        a.boom(T_HIT, 36, 1.6, 0.9)
        for j, d in enumerate((0, 4, 7, 11)): a.note(fq(d, 1), T_HIT + 0.02, 0.6, 0.06, "bell")
        s.fin_snd(a, DUR, 253); a.save(path)


if __name__ == "__main__":
    r = Hrimnir()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
