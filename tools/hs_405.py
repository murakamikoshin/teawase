"""ヴィルゴ(★4 乙女座の弓使い) 専用の必殺技「麦穂ノ矢」— 高品質版(hs_fx)。空へ 放った 光の 矢が はじけ、金の 矢の 雨に なって こちらへ ふりそそぐ
夕方の 麦畑(風に ゆれる 麦・入道雲)。矢の 先に 麦の 光の つぶが 集まる(前ふり) → 目の カットイン+技名 →
引きしぼって 放つ: 光の 矢が 空へ → 視点が 空を 見上げる → 高い 所で はじける →
金の 矢が 空の 一点から こちらへ ふりそそぐ(だんだん 多く 大きく)→ さいごの 1本の 矢じりが 目の前に → 白 → 暗転
使い方: python3 hs_405.py 出力.mp4 / python3 hs_405.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.1
T_CUT, T_DRAW, T_SHOT, T_TILT, T_BURST, T_HIT = 2.7, 4.7, 5.25, 5.45, 6.55, 9.4
GOLD = np.array([1.0, 0.80, 0.32], np.float32); WHEAT = np.array([0.95, 0.72, 0.26], np.float32); CREAM = np.array([1.0, 0.95, 0.80], np.float32); SKYB = np.array([0.30, 0.50, 0.85], np.float32)
EYE = (388, 268); TIPA = (790, 300); NOCK = (300, 350)


class Virgo(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 405, (1.0, 0.85, 0.4), "麦穂ノ矢", "Harvest Arrow", W); s.fx_init(405)
        s._vt = [s.text_img(c, hz.SERIF, int(76 * s.k), 3) for c in "麦穂ノ矢"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; s.hy = 0.62
        rg = np.random.RandomState(4050); s._stk = rg.rand(46, 5).astype(np.float32); s._ar = rg.rand(150, 5).astype(np.float32); s.VP = (W * 0.52, s.H * 0.36)

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    def sky(s, t, hy, zen=0.0):
        """夕方の 空と 入道雲(zen: 見上げた ときの 高い 空)"""
        W, H, k = s.W, s.H, s.k; v = (s.yn / hy).clip(0, 1)
        f = (SKYB * np.array([0.75, 0.9, 1.15], np.float32))[None, None, :] * (1.15 - 0.45 * v)[:, :, None] * (1 - v ** 2.2 * (1 - zen))[:, :, None] + np.array([1.2, 0.86, 0.50], np.float32)[None, None, :] * (v ** 2.2 * (1 - zen))[:, :, None]
        n = s.fbm(0.00055, t * 5 + zen * 300, zen * 500, 3, 90 * k, 1); n = cv2.GaussianBlur(n, (0, 0), 5 * k) + (s.tex(2, s.XX * 0.5, s.YY * 0.5) - 0.5) * 0.05; yb = 0.50 - 0.30 * zen
        cu = sstep(0.52, 0.62, n + 0.30 * np.exp(-((s.yn - yb) / 0.16) ** 2) - 0.08); top = np.clip(cu - cv2.warpAffine(cu, np.float32([[1, 0, 6 * k], [0, 1, 10 * k]]), (W, H)), 0, 1)
        dt = s.fbm(0.0030, t * 9 + zen * 200, zen * 300, 4, 40 * k, 2); cc = CREAM[None, None, :] * (0.78 + 0.42 * dt - 0.25 * sstep(0.62, 0.9, n))[:, :, None] * np.array([1.05, 0.98, 0.95], np.float32)[None, None, :] + (GOLD * 0.8)[None, None, :] * top[:, :, None]
        f = f * (1 - cu * 0.92)[:, :, None] + cc * (cu * 0.92)[:, :, None]
        return f

    # ---- 夕方の 麦畑 ----
    def field(s, t, lum=1.0):
        W, H, k = s.W, s.H, s.k; hy = s.hy
        f = s.sky(t, hy); f += (np.exp(-np.hypot(s.xn - 0.78, (s.yn - 0.50) * 1.2) / 0.16) * 0.9)[:, :, None] * np.array([1.0, 0.85, 0.55], np.float32)[None, None, :]
        d = np.clip((s.yn - hy) / (1 - hy), 0, 1); z = 1.0 / (d + 0.09); wv = np.sin(t * 2.0 + s.xn * 6 - d * 5) * 4 + np.sin(t * 5.3 + s.xn * 19) * 1.2
        u = (s.xn - 0.5) * z * 250 + wv * z * 1.3; b = s.tex(1, u, z * 5.0) * 0.6 + s.tex(3, u * 2.3, z * 10.0) * 0.4; gust = 0.75 + 0.5 * s.tex(0, s.XX * 0.2 - t * 45, s.YY * 0.6)
        g = np.array([0.20, 0.12, 0.03], np.float32)[None, None, :] + WHEAT[None, None, :] * (b ** 1.3 * (0.5 + 0.7 * d) * gust)[:, :, None] + CREAM[None, None, :] * (sstep(0.72, 0.95, b) * gust * 0.5)[:, :, None]
        gm = sstep(hy - 0.004, hy + 0.004, s.yn + (b - 0.5) * 0.012)[:, :, None]
        return ((f * (1 - gm) + g * gm) * lum).astype(np.float32)

    def frontwheat(s, f, t, v=1.0):
        """手前の 麦(くきと 穂)"""
        W, H, k = s.W, s.H, s.k; L = s.L0(); E = s.L0()
        for i in range(len(s._stk)):
            p = s._stk[i]; x0 = (p[0] * 1.1 - 0.05) * W; hh = H * (0.14 + 0.13 * p[1]); sw = (math.sin(t * 2.0 + p[2] * 9) * 16 + math.sin(t * 5.3 + p[3] * 20) * 5) * k * (0.6 + p[1])
            top = (x0 + sw, H * 1.01 - hh); cv2.line(L, (int(x0), H), (int(top[0]), int(top[1])), 0.8, max(1, int(2 * k)), cv2.LINE_AA)
            an = math.degrees(math.atan2(-hh, sw)) + 90; cv2.ellipse(E, (int(top[0] + sw * 0.12), int(top[1] - 14 * k)), (max(2, int(5 * k)), max(4, int(19 * k))), an * 0.4, 0, 360, 1.0, -1, cv2.LINE_AA)
            for j in range(4):
                for sg in (-1, 1): cv2.line(E, (int(top[0] + sw * 0.1), int(top[1] - (2 + j * 8) * k)), (int(top[0] + sw * 0.1 + sg * 9 * k), int(top[1] - (14 + j * 8) * k)), 0.7, 1, cv2.LINE_AA)
        m = np.clip(L + E, 0, 1) * v; col = (WHEAT * 0.55)[None, None, :] * L[:, :, None] + (GOLD * 1.05)[None, None, :] * (E * (0.75 + 0.3 * s.tex(2, s.XX * 2, s.YY * 2)))[:, :, None]
        f *= (1 - m)[:, :, None]; f += col * v

    def arrow(s, f, a, b, w, v=1.0):
        """金の 矢(光の すじ+先の 光)。a: うしろ、b: 先"""
        if v <= 0.01: return
        L = s.L0(); cv2.line(L, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 1.0, max(1, int(w)), cv2.LINE_AA)
        s.glowmask(f, cv2.GaussianBlur(L, (0, 0), 0.7 * s.k), GOLD * 0.7 + CREAM * 0.3, 1.1 * v, 1.0, 0.9, 6.0)

    def head(s, f, c, S, t, v=1.0):
        """正面から 見た 矢じり: 4つの 面の 金の 星形 + 軸の 輪"""
        x = (s.XX - c[0]) / S; y = (s.YY - c[1]) / S; ax = np.abs(x); ay = np.abs(y); d4 = ax + ay + 2.2 * np.minimum(ax, ay); r = np.hypot(x, y)
        ins = sstep(1.0, 1.0 - max(0.01, 2.5 / S), d4) * v; face = (x > 0).astype(np.float32) * 0.5 + (y > 0).astype(np.float32) * 0.25 + ((ax > ay) * 0.2)
        shade = 0.45 + 0.75 * face + 0.5 * np.exp(-(np.minimum(ax, ay) / 0.03) ** 2) - 0.35 * d4
        col = (GOLD[None, None, :] * shade[:, :, None] + CREAM[None, None, :] * (np.exp(-(r / 0.10) ** 2) * 1.2)[:, :, None]) * (0.85 + 0.25 * s.tex(1, x * 40, y * 40))[:, :, None]
        ring = np.exp(-((r - 1.25) / 0.05) ** 2) * 0.8 * v
        f *= (1 - ins)[:, :, None]; f += col * ins[:, :, None] + ring[:, :, None] * (GOLD * 0.8)[None, None, :] + (np.exp(-(r / 2.0) ** 2) * 0.25 * v)[:, :, None] * GOLD[None, None, :]

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(CREAM), rim_dir=(0.8, -0.5), rim_v=0.6 * alpha, amb=(1.0, 0.88, 0.7), amb_v=0.14, aura=GOLD, aura_v=0.08 + 0.14 * getattr(s, '_ch', 0.0), alpha=alpha, shade=0.92)

    def frame(s, t):
        o = {}; s._ch = 0.0
        if t < T_CUT: f, sh = s.sc_field(t)
        elif t < T_DRAW: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.95)
        else: f, sh = s.sc_rain(t)
        f, sh = s.finale(f, sh, t, DUR, None, (1, 0.9, 0.5), (1, 0.75, 0.3), -1)
        return s.film(f, t, sh * s.k, 0.5, o.get("thr", 0.90), (1.0, 0.92, 0.75), 1.0 + 0.15 * min(sh, 8), 0.035, o.get("vig", 0.50), 1.22, 0.98)

    def archer(s, f, t, **kw):
        W, H, k = s.W, s.H, s.k; img = s.pose(t, **kw); s.draw(f, img, W * 0.46, H * 0.63, 0.42)

    # 1) 夕方の 麦畑。矢の 先に 光の つぶが 集まる
    def sc_field(s, t):
        W, H, k = s.W, s.H, s.k; g = ease((t - 0.8) / 1.7); s._ch = g
        f = s.field(t); s.archer(f, t, sway=0.15, breath=0.5); tp = s.at(*s.A(*TIPA))
        p = s._ar[:60]; q = np.clip((t - 0.6 - p[:, 0] * 1.4) / 0.9, 0, 1); sx = p[:, 1] * W; sy = H * (0.66 + 0.3 * p[:, 2])
        s.splat(f, sx + (tp[0] - sx) * q ** 2, sy + (tp[1] - sy) * q ** 2 - np.sin(q * 3.14) * 40 * k, (0.4 + p[:, 3]) * (q > 0) * (q < 1), GOLD * 0.8 + 0.2, 1.2, 0.5)
        s.flare(f, tp, (5 + 9 * g + 2 * math.sin(t * 9) ** 2) * k, GOLD * 0.7 + CREAM * 0.3, 0.3 + 0.7 * g, 0.6, t, False)
        s.frontwheat(f, t)
        return f * clamp(t / 0.5), 0

    # 2) 目の カットイン(矢の 形の 帯)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.field(t, 0.30)
        a = ease(u / 0.24); cy = H * 0.33; hh = H * 0.10
        band = (WHEAT * 0.75)[None, None, :] * (0.55 + 0.6 * s.tex(1, s.XX * 0.08 - t * 150, s.YY * 2.0))[:, :, None] + (CREAM * 0.2)[None, None, :]
        ex, ey = s.A(*EYE); zf = 1.5; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(band, img, W / 2 - (ex - s.aw / 2) * sc - (1 - a) * W * 0.9 + u * 6 * k, cy - (ey - s.ah / 2) * sc, zf)
        tipx = W * (1.02 * a); edge = hh * np.clip((tipx - s.XX) / (W * 0.16), 0, 1) ** 0.8; dd = np.abs(s.YY - cy)                                # 右が とがった 矢の 形
        m = (sstep(edge, edge - 2, dd) * (edge > 1))[:, :, None]; f = f * (1 - m) + band * m
        s.glowmask(f, np.exp(-((dd - edge) / (1.6 * k)) ** 2) * (edge > 1), CREAM, 0.8 * a, 1.0, 0.7, 7.0)
        n = len(s._vt); step = W * 0.20
        for i, T in enumerate(s._vt):                                                # 技名: 左から 矢の ように 入る
            p = (u - 0.45 - i * 0.10) / 0.14
            if p > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step - (1 - ease(p)) * 160 * k, H * 0.68, clamp(p * 2), 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.3 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.68 - 76 * k, 1.0, 1.0)
        if u > 1.2:
            g = ease((u - 1.2) / 0.35); s.glowline(f, [(W * 0.08, H * 0.68 + 58 * k), (W * (0.08 + 0.84 * g), H * 0.68 + 58 * k)], GOLD, 1.0, 0.8, 0.8)
            s.text(f, s.txt_en, W / 2, H * 0.68 + 84 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.7
        return f, 4 * math.exp(-u * 6)

    # 3) 放つ → 見上げる → はじける → 矢の 雨が こちらへ
    def sc_rain(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_DRAW; us = t - T_SHOT; ub = t - T_BURST; sh = 0.0
        tq = ease2(clamp((t - T_TILT) / 0.85)); ty = H * 1.1 * tq; VP = s.VP
        if tq < 1:
            s._ch = 1.0; A = s.field(t)
            g = ease(u / 0.5) if us < 0 else 0.0
            s.archer(A, t, sway=0.1, lean=(-5 * k * g if us < 0 else 6 * k * math.exp(-us * 6)), squash=(-0.03 * g if us < 0 else 0.04 * math.exp(-us * 8)))
            tp = s.at(*s.A(*TIPA)); nk = s.at(*s.A(*NOCK)); dv = (tp[0] - nk[0], tp[1] - nk[1]); dn = math.hypot(*dv); dv = (dv[0] / dn, dv[1] / dn)
            if us < 0: s.flare(A, tp, (14 + 14 * g + 3 * math.sin(t * 20) ** 2) * k, GOLD * 0.6 + CREAM * 0.4, 1.0, 0.8, t * 2, False)
            else:
                q = us / 0.22; a = (tp[0] + dv[0] * W * 2.2 * max(0.0, q - 0.5), tp[1] + dv[1] * W * 2.2 * max(0.0, q - 0.5) - 0); b = (tp[0] + dv[0] * W * 2.2 * q, tp[1] + dv[1] * W * 2.2 * q - W * 0.9 * q * q)
                s.arrow(A, a, b, 6 * k, max(0.0, 1.6 - q));
                if us < 0.1: A = A + (1 - us / 0.1) * 0.8
                sh += 8 * math.exp(-us * 6)
            s.frontwheat(A, t)
        if tq > 0:
            U = s.sky(t, 1.6, 1.0)
            if tq < 1:
                M = np.float32([[1, 0, 0], [0, 1, ty]]); Ash = cv2.warpAffine(A, M, (W, H), borderMode=cv2.BORDER_REPLICATE); m = sstep(ty + 40 * k, ty - 40 * k, s.YY)[:, :, None]; f = Ash * (1 - m) + U * m
            else: f = U
        else: f = A
        if tq > 0.2 and ub < 0:                                                      # のぼっていく 光の 矢
            q = clamp((t - T_TILT - 0.15) / (T_BURST - T_TILT - 0.15)); c = (W * 0.75 + (VP[0] - W * 0.75) * ease(q), H * 1.05 + (VP[1] - H * 1.05) * ease(q))
            s.arrow(f, (c[0] + 30 * k * (1 - q), c[1] + (140 - 110 * q) * k), c, (6 - 4 * q) * k, 1.0); s.flare(f, c, (12 - 7 * q) * k, CREAM, 1.0, 0.6, t * 3, False)
        if ub > 0:
            e = clamp(ub / (T_HIT - T_BURST)); d = np.hypot(s.XX - VP[0], s.YY - VP[1])
            for i in range(3):                                                       # はじける 輪
                qq = (ub - i * 0.08) / 0.8
                if 0 < qq < 1: f += (np.exp(-((d - qq * W * 0.9) / ((4 + 22 * qq) * k)) ** 2) * (1 - qq) ** 1.5 * 0.9)[:, :, None] * (GOLD * 0.7 + CREAM * 0.3)[None, None, :]
            s.flare(f, VP, (22 * math.exp(-ub * 3) + 6) * k, CREAM, 0.9, 0.8, t, True)
            L = s.L0(); Hd = s.L0(); ar = s._ar                                      # 矢の 雨: 空の 一点から 放射で 落ちて くる
            for i in range(len(ar)):
                p = ar[i]; tb = 0.10 + (p[0] ** 0.6) * (T_HIT - T_BURST - 1.1); q = (ub - tb) / (1.0 + 0.6 * p[1])
                if q <= 0 or q >= 1.15: continue
                a = p[2] * 6.2832; rr = q ** 2.3 * W * (0.9 + 1.2 * p[3]); r0 = rr * (0.80 - 0.25 * q) ; ca, sa = math.cos(a), math.sin(a)
                cv2.line(L, (int(VP[0] + ca * r0), int(VP[1] + sa * r0 * 1.2)), (int(VP[0] + ca * rr), int(VP[1] + sa * rr * 1.2)), float(min(1.0, q * 4)), max(1, int((1.5 + 9 * q ** 2) * k)), cv2.LINE_AA)
                cv2.circle(Hd, (int(VP[0] + ca * rr), int(VP[1] + sa * rr * 1.2)), max(1, int((1.5 + 8 * q ** 2) * k)), float(min(1.0, q * 4)), -1, cv2.LINE_AA)
            Lb = cv2.GaussianBlur(L, (0, 0), 0.8 * k); f = f * (1 - 0.5 * np.clip(Lb, 0, 1))[:, :, None]
            f += (Lb * 1.1 + cv2.GaussianBlur(L, (0, 0), 6 * k) * 0.6)[:, :, None] * GOLD[None, None, :] + (cv2.GaussianBlur(Hd, (0, 0), 1.0 * k) * 1.3)[:, :, None] * CREAM[None, None, :]
            sh += 6 * math.exp(-ub * 4) + 1 + 4 * e
            pl = clamp((ub - (T_HIT - T_BURST - 1.15)) / 1.15)                       # さいごの 1本
            if pl > 0:
                C = (VP[0] + (W * 0.5 - VP[0]) * pl, VP[1] + (H * 0.46 - VP[1]) * pl); S = k * (3 + 26 * pl) + W * 1.25 * pl ** 3.2
                s.head(f, C, S, t, min(1.0, pl * 5)); f = s.godrays(f, C, 0.25 + 0.4 * pl, 1.0, 8, 0.88, 0.45); sh += 9 * pl ** 2
                if pl > 0.84: f = f + ((pl - 0.84) / 0.16) ** 2 * 1.5
            if ub < 0.1: f = f + (1 - ub / 0.1) * 0.7
        if u < 0.12: f = f + (1 - u / 0.12) * 0.3
        return f, sh

    # ---- この子だけの曲: たて琴の 分散和音(明るい 調べ)→ 引きしぼる きしみ → 放つ 弦の 音 → 高く のぼる 一音 → 鈴の 雨が ふえて こちらへ ----
    def audio(s, path):
        a = Snd(DUR, 405); C4 = 261.63; mj = [0, 2, 4, 7, 9]
        def fq(d, o=0): return C4 * 2 ** ((mj[d % 5] + 12 * (d // 5 + o)) / 12.0)
        a.a += (a.lp(a.nz, 0.10) - a.lp(a.nz, 0.03)) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * a.t)) * a.gate(0, T_TILT + 0.5, 0.3) * 0.5              # 麦畑の 風
        for d in (0, 2, 4): a.note(fq(d, -1), 0.1, 2.6, 0.045, "pad", 0.6)
        for i in range(12): a.note(fq(i % 5 + (i // 5) * 5, 0), 0.5 + i * 0.17, 0.5, 0.07, "bell")
        a.boom(T_CUT, 55, 0.8, 0.4); a.whoosh(T_CUT - 0.25, T_CUT, 0.4, 0.5)
        for i in range(4): a.note(fq([2, 3, 4, 5][i], 1), T_CUT + 0.45 + i * 0.10, 0.5, 0.10, "bell")
        for d in (0, 2, 4): a.note(fq(d, 0), T_CUT, 1.9, 0.04, "pad", 0.05)
        a.rise(T_DRAW, T_SHOT, 180, 520, 0.08); a.a += (a.nz - a.lp(a.nz, 0.5)) * a.gate(T_DRAW + 0.1, T_SHOT, 0.05) * 0.06 * (0.5 + 0.5 * np.sin(2 * np.pi * 37 * a.t))    # きしみ
        a.note(fq(0, -1), T_SHOT, 0.5, 0.20, "fm", 0.001); a.a += (a.nz - a.lp(a.nz, 0.6)) * a.env(T_SHOT, 0.001, 0.05) * 0.6; a.whoosh(T_SHOT, T_SHOT + 0.3, 0.6, 0.6)      # 弦
        a.rise(T_TILT, T_BURST, 600, 2400, 0.06); a.note(fq(9, 2), T_BURST, 1.2, 0.14, "bell"); a.boom(T_BURST, 70, 0.7, 0.3)
        for i in range(8): a.note(fq(9 - i, 2), T_BURST + 0.05 + i * 0.03, 0.3, 0.05, "bell")
        rg = np.random.RandomState(405); n = 80
        for i in range(n):                                                                                              # 矢の 雨: 風切りと 鈴が ふえる
            tk = T_BURST + 0.3 + (T_HIT - T_BURST - 0.4) * (i / n) ** 0.7; g = 0.4 + 0.6 * i / n
            a.note(fq(int(rg.rand() * 10) + 5, 1), tk, 0.12, 0.035 * g, "bell", 0.001)
            if i % 3 == 0: a.a += (a.nz - a.lp(a.nz, 0.6)) * np.exp(-((a.t - tk) / 0.05) ** 2) * 0.10 * g
        for j, d in enumerate((0, 2, 4, 5, 7, 9)): a.note(fq(d, 0), T_BURST + 0.4 + j * 0.06, T_HIT - T_BURST, 0.045, "pad", 0.5)
        a.rise(T_HIT - 1.15, T_HIT, 90, 380, 0.13); a.whoosh(T_HIT - 1.0, T_HIT, 0.8, 0.3)
        a.boom(T_HIT, 38, 1.6, 0.8)
        for j, d in enumerate((0, 2, 4, 7)): a.note(fq(d, 1), T_HIT + 0.02, 0.6, 0.06, "bell")
        s.fin_snd(a, DUR, 405); a.save(path)


if __name__ == "__main__":
    r = Virgo()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
