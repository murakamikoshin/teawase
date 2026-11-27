"""アンカ(★4 錨の女戦士) 専用の必殺技「大錨落トシ」— 高品質版(hs_fx)。鎖つきの 錨を 空へ 投げ、まわりながら 手前の 海へ 落ちて、立ち上がった 水の 壁が こちらへ かぶさる
晴れた 海の 桟橋に 立つ(しぶき)→ 真ちゅうの 丸窓の カットイン+技名 → ふりかぶって 投げる(鎖が のびる)→
錨が 弧を えがいて 大きく なり 手前の 海へ 落ちる → 水柱 → 水の 壁が せり上がって 空を かくす(画面に 水の つぶ)→ 波がしらが くずれて 白い 泡が かぶさる → 白 → 暗転
使い方: python3 hs_418.py 出力.mp4 / python3 hs_418.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.3
T_CUT, T_ACT, T_THROW, T_SPL, T_HIT = 2.7, 4.7, 5.1, 6.9, 9.4
SEA0 = np.array([0.03, 0.22, 0.48], np.float32); SEA1 = np.array([0.10, 0.52, 0.72], np.float32); SKYT = np.array([0.16, 0.42, 0.86], np.float32); SKYH = np.array([0.72, 0.90, 1.0], np.float32); FOAM = np.array([0.95, 0.99, 1.0], np.float32); WOOD = np.array([0.46, 0.29, 0.16], np.float32); BRASS = np.array([0.95, 0.72, 0.30], np.float32)
EYE = (330, 265); HAND = (175, 330); ANCH = (630, 250)


class Anka(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 418, (0.5, 0.8, 1.0), "大錨落トシ", "Anchor Drop", W); s.fx_init(418)
        s._vt = [s.text_img(c, hz.SERIF, int(70 * s.k), 3) for c in "大錨落トシ"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; s.hy = 0.52; rg = np.random.RandomState(4180); s._bd = rg.rand(26, 5).astype(np.float32)

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 晴れた 海と 桟橋 ----
    def sea(s, t, lum=1.0):
        W, H, k = s.W, s.H, s.k; hy = s.hy; v = (s.yn / hy).clip(0, 1)
        f = SKYT[None, None, :] * (1 - v ** 1.4)[:, :, None] + SKYH[None, None, :] * (v ** 1.4)[:, :, None]
        n = s.fbm(0.0009, t * 5, 0, 3, 80 * k, 1); n = cv2.GaussianBlur(n, (0, 0), 3 * k); cu = sstep(0.50, 0.64, n + 0.10 * np.exp(-((s.yn - 0.33) / 0.12) ** 2) - 0.03)
        shd = np.clip(0.55 + 1.2 * (s.fbm(0.003, t * 8, 3, 3, 0, 2) - 0.4) + 0.5 * (0.35 - s.yn), 0.86, 1.25)
        f = f * (1 - cu * 0.92)[:, :, None] + (cu * 0.92 * shd)[:, :, None] * FOAM[None, None, :]
        f += (np.exp(-np.hypot(s.xn - 0.16, (s.yn - 0.10) * H / W) / 0.10) * 0.8)[:, :, None] * np.array([1.0, 0.97, 0.85], np.float32)[None, None, :]            # 日
        d = np.clip((s.yn - hy) / (1 - hy), 0, 1); pu = (s.XX - W / 2) / (d + 0.08) * 0.16; pv = 46.0 / (d + 0.08)
        w1 = s.tex(2, pu + t * 14, pv + t * 22); w2 = s.tex(0, pu * 2.1 - t * 20, pv * 2.1 + t * 30); wv = w1 * 0.6 + w2 * 0.4
        sw = SEA0[None, None, :] * (1 - d * 0.5)[:, :, None] + (SEA1 - SEA0 * 0.5)[None, None, :] * (d * 0.9 * (0.6 + 0.6 * wv))[:, :, None] + (sstep(0.70, 0.9, wv) * (0.25 + 0.5 * d))[:, :, None] * FOAM[None, None, :]
        sw += (np.exp(-np.abs(s.xn - 0.16) / 0.10) * np.exp(-d / 0.35) * sstep(0.5, 0.8, w2) * 0.6)[:, :, None]                                                   # 日の 道
        m = sstep(hy - 0.002, hy + 0.003, s.yn)[:, :, None]; f = f * (1 - m) + sw * m
        f += (np.exp(-((s.yn - hy) / 0.012) ** 2) * 0.25)[:, :, None]
        y0, y1 = 0.745, 0.868; hw = 0.21 + (s.yn - y0) * 1.5                                                                # 桟橋
        pm = sstep(-0.004, 0.0, s.yn - y0) * sstep(0.004, 0.0, s.yn - y1) * sstep(0.004, -0.004, np.abs(s.xn - 0.5) - hw)
        pl = np.clip((s.yn - y0) / (y1 - y0), 0, 1.2) ** 0.8 * 7.0; ln = np.exp(-((np.abs((pl % 1.0) - 0.5) - 0.5) / 0.05) ** 2)
        wc = WOOD[None, None, :] * ((0.65 + 0.7 * s.tex(1, s.XX * 0.25, np.floor(pl) * 37 + s.YY * 0.05)) * (1 - 0.6 * ln))[:, :, None]
        f = f * (1 - pm)[:, :, None] + wc * pm[:, :, None]
        fm = sstep(-0.002, 0.002, s.yn - y1) * sstep(0.002, -0.002, s.yn - y1 - 0.022) * sstep(0.004, -0.004, np.abs(s.xn - 0.5) - (0.21 + (y1 - y0) * 1.5))
        f = f * (1 - fm)[:, :, None] + (WOOD * 0.38)[None, None, :] * ((0.6 + 0.6 * s.tex(1, s.XX * 0.3, s.YY * 1.5)) * fm)[:, :, None]
        return (f * lum).astype(np.float32)

    def spray(s, f, t, v=1.0):
        W, H = s.W, s.H; r = s.rnd[:90]
        x = ((r[:, 0] + t * (0.03 + 0.05 * r[:, 4])) % 1.0) * W; y = (0.52 + 0.45 * ((r[:, 1] - t * (0.04 + 0.05 * r[:, 2])) % 1.0)) * H
        s.splat(f, x, y, (0.3 + 0.6 * r[:, 2]) * v, FOAM, 1.1, 0.3)

    # ---- 錨(式): 輪・さお・横木・弓なりの 腕・つめ。太い 線で 形、ずらした 細い 線で てり ----
    def anchor(s, f, c, L, ang, v=1.0, soft=0.0):
        if v <= 0.01 or L < 6: return (int(c[0]), int(c[1]))
        M = s.L0(); S = s.L0(); ca, sa = math.cos(ang), math.sin(ang)
        def P(x, y, o=0.0): return (int(c[0] + (x * ca - y * sa) * L - o * L), int(c[1] + (x * sa + y * ca) * L - o * L))
        T = lambda w: max(1, int(w * L))
        arc = [(0.36 * math.cos(a), 0.12 + 0.36 * math.sin(a)) for a in np.linspace(0, math.pi, 18)]
        for o, tgt, val, kth in ((0.0, M, 1.0, 1.0), (0.0, S, 0.22, 1.0), (0.012, S, 1.0, 0.30)):
            cv2.circle(tgt, P(0, -0.5, o), T(0.09), val, T(0.036 * kth), cv2.LINE_AA)
            cv2.line(tgt, P(0, -0.42, o), P(0, 0.46, o), val, T(0.075 * kth), cv2.LINE_AA)
            cv2.line(tgt, P(-0.26, -0.31, o), P(0.26, -0.31, o), val, T(0.052 * kth), cv2.LINE_AA)
            cv2.polylines(tgt, [np.array([P(x, y, o) for x, y in arc], np.int32)], False, val, T(0.08 * kth), cv2.LINE_AA)
            if kth == 1.0:
                for sg in (-1, 1): cv2.fillPoly(tgt, [np.array([P(sg * 0.36, -0.10, o), P(sg * 0.50, 0.17, o), P(sg * 0.25, 0.19, o)], np.int32)], val, cv2.LINE_AA)
            else:
                for sg in (-1, 1): cv2.line(tgt, P(sg * 0.36, -0.07, o), P(sg * 0.30, 0.14, o), val, T(0.02), cv2.LINE_AA)
        M = cv2.GaussianBlur(M, (0, 0), max(0.7, L * (0.004 + soft))) * v; S = cv2.GaussianBlur(S, (0, 0), max(0.7, L * 0.008)) * (0.8 + 0.4 * s.tex(1, s.XX * 0.5, s.YY * 0.5))
        dk = np.array([0.05, 0.06, 0.09], np.float32); md = np.array([0.12, 0.14, 0.20], np.float32); hi = np.array([0.80, 0.90, 1.0], np.float32)
        a1 = sstep(0.05, 0.45, S)[:, :, None]; a2 = sstep(0.55, 1.05, S)[:, :, None]; colr = dk + (md - dk) * a1 + (hi - md) * a2
        f[:] = f * (1 - M)[:, :, None] + colr * M[:, :, None]
        return P(0, -0.5)

    def chain(s, f, p0, p1, sag, size, v=1.0):
        n = max(6, int(math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / (size * 1.5))); L = s.L0(); pa = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
        for i in range(n + 1):
            u = i / n; x = p0[0] + (p1[0] - p0[0]) * u; y = p0[1] + (p1[1] - p0[1]) * u + sag * 4 * u * (1 - u); sz = size * (0.8 + 0.4 * u)
            cv2.ellipse(L, (int(x), int(y)), (int(sz), max(1, int(sz * (0.55 if i % 2 else 0.22)))), math.degrees(pa), 0, 360, 1.0, max(1, int(sz * 0.3)), cv2.LINE_AA)
        L = cv2.GaussianBlur(L, (0, 0), 0.7 * s.k) * v
        f[:] = f * (1 - L * 0.95)[:, :, None] + (L * (0.6 + 0.8 * s.tex(0, s.XX * 2, s.YY * 2)))[:, :, None] * np.array([0.24, 0.26, 0.32], np.float32)[None, None, :]

    def beads(s, f, age):
        """画面に ついた 水の つぶ(うしろを さかさに うつす)"""
        W, H, k = s.W, s.H, s.k; src = f.copy()
        for b in s._bd:
            r = int((7 + 32 * b[2] ** 2) * k); x = int((0.06 + 0.88 * b[0]) * W); y = int((0.06 + 0.8 * b[1]) * H + age * (10 + 60 * b[3]) * k * (r / (20 * k)))
            a = clamp((age - b[4] * 0.5) * 8)
            if a <= 0.01 or y - 2 * r < 0 or y + 2 * r >= H or x - 2 * r < 0 or x + 2 * r >= W: continue
            patch = cv2.resize(src[y - 2 * r:y + 2 * r, x - 2 * r:x + 2 * r], (2 * r, 2 * r), interpolation=cv2.INTER_AREA)[::-1, ::-1]
            yy, xx = np.mgrid[-r:r, -r:r].astype(np.float32) / r; d = np.hypot(xx, yy); m = sstep(1.0, 0.88, d) * a
            colr = patch * (0.85 + 0.25 * d ** 2)[:, :, None] + (np.exp(-(np.hypot(xx + 0.35, yy + 0.4) / 0.2) ** 2) * 1.2)[:, :, None] - (np.exp(-((d - 0.92) / 0.08) ** 2) * 0.25 * (yy > 0))[:, :, None]
            f[y - r:y + r, x - r:x + r] = f[y - r:y + r, x - r:x + r] * (1 - m)[:, :, None] + colr * m[:, :, None]

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        s.draw_lit(f, img, cx, cy, hfrac, rim=(1.0, 0.97, 0.88), rim_dir=(-0.7, -0.7), rim_v=0.5 * alpha, amb=(0.85, 0.95, 1.0), amb_v=0.10, aura=SEA1, aura_v=0.05, alpha=alpha, shade=0.97)

    def frame(s, t):
        o = {}
        if t < T_CUT: f, sh = s.sc_pier(t)
        elif t < T_ACT: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.97)
        else: f, sh = s.sc_drop(t)
        f, sh = s.finale(f, sh, t, DUR, None, (0.85, 0.95, 1), (0.4, 0.75, 1.0), -1)
        return s.film(f, t, sh * s.k, 0.4, o.get("thr", 0.95), (0.93, 0.98, 1.0), 1.0 + 0.15 * min(sh, 8), 0.03, o.get("vig", 0.4), 1.2, 0.96)

    def gal(s, f, t, alpha=1.0, **kw):
        W, H, k = s.W, s.H, s.k; cy = H * 0.60; fy = cy + 0.42 * H * 0.47
        sd = np.exp(-(((s.XX - W * 0.53) / (W * 0.20)) ** 2 + ((s.YY - fy) / (H * 0.012)) ** 2)); f *= (1 - 0.5 * alpha * sd)[:, :, None]
        img = s.pose(t, **kw); s.draw(f, img, W / 2, cy, 0.42, alpha)
        return s.at(*s.A(*HAND)), s.at(*s.A(*ANCH))

    # 1) 桟橋に 立つ
    def sc_pier(s, t):
        f = s.sea(t); s.gal(f, t, clamp(t / 0.6), sway=0.3, breath=0.5); s.spray(f, t, 0.6)
        return f * clamp(t / 0.5), 0

    # 2) 目の カットイン(真ちゅうの 丸窓)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.sea(t, 0.34)
        a = back(clamp(u / 0.35)); cy = H * 0.32; R = H * 0.165 * a
        band = (SEA0 * 0.9)[None, None, :] + (SEA1 * 0.8)[None, None, :] * (0.3 + 0.8 * sstep(0.45, 0.9, s.tex(2, s.XX * 0.5 + t * 30, s.YY * 0.5 + t * 12)))[:, :, None]
        ex, ey = s.A(*EYE); zf = 1.5; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(band, img, W / 2 - (ex - s.aw / 2) * sc, cy - (ey - s.ah / 2) * sc, zf)
        d = np.hypot(s.XX - W / 2, s.YY - cy); th = np.arctan2(s.YY - cy, s.XX - W / 2)
        if R > 6:
            m = sstep(R, R - 2, d)[:, :, None]; f = f * (1 - m) + band * m
            rw = 15 * k; rim = sstep(rw, rw - 2, np.abs(d - R - rw * 0.5) * 2)                                                     # 真ちゅうの ふち
            shn = 0.45 + 0.6 * np.clip(-np.cos(th - 2.3), 0, 1) ** 2 + 0.35 * np.clip(np.cos(th - 2.3), 0, 1) ** 6 + 0.25 * s.tex(1, th * 80, d * 2)
            f = f * (1 - rim)[:, :, None] + BRASS[None, None, :] * (shn * rim)[:, :, None]
            for i in range(8):
                b = i * math.pi / 4 + 0.39; bx, by = W / 2 + math.cos(b) * (R + rw * 0.5), cy + math.sin(b) * (R + rw * 0.5); bd = np.hypot(s.XX - bx, s.YY - by)
                f = f * (1 - sstep(4.5 * k, 3.5 * k, bd) * 0.7)[:, :, None] + (np.exp(-(np.hypot(s.XX - bx + 1.5 * k, s.YY - by + 1.5 * k) / (2.0 * k)) ** 2) * 0.9)[:, :, None] * BRASS[None, None, :]
            f += (np.exp(-(np.hypot(s.XX - W / 2 + R * 0.45, s.YY - cy + R * 0.5) / (R * 0.28)) ** 2) * 0.22 * (d < R))[:, :, None]                                  # ガラスの てり
        n = len(s._vt); step = W * 0.172
        for i, T in enumerate(s._vt):                                                # 技名: 波に ゆられて 浮かぶ
            p = (u - 0.5 - i * 0.13) / 0.35
            if p > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.70 + (1 - ease(p)) * 30 * k + 4 * k * math.sin(t * 3.0 + i * 0.9), clamp(p) ** 1.2, 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.35 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.70 - 74 * k, 1.0, 1.0)
        if u > 1.4:
            g = ease((u - 1.4) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.70 + 58 * k), (W / 2 + W * 0.42 * g, H * 0.70 + 58 * k)], FOAM, 1.0, 0.7, 0.7)
            s.text(f, s.txt_en, W / 2, H * 0.70 + 84 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.5
        return f, 3 * math.exp(-u * 6)

    # 3) 投げる → 錨が 手前の 海へ → 水の 壁が かぶさる
    def sc_drop(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_ACT; sh = 0.0; q = t - T_SPL
        f = s.sea(t)
        th_ = t - T_THROW
        hp, ap = s.gal(f, t, 1.0, sway=0.2, breath=0.4, squash=(0.06 * ease(u / (T_THROW - T_ACT)) if th_ < 0 else -0.06 * math.exp(-th_ * 4)))
        s.spray(f, t, 0.6)
        imp = (W * 0.5, H * 0.93)
        if 0 <= th_ and q < 0:                                                       # 錨が 弧を えがく(鎖が 手から のびる)
            p = th_ / (T_SPL - T_THROW); x = ap[0] + (imp[0] - ap[0]) * p ** 0.9; y = ap[1] + (imp[1] - ap[1]) * p ** 2 - H * 0.95 * p * (1 - p) ** 1.3
            L = k * (70 + 560 * p ** 2.4); rp = s.anchor(f, (x, y), L, -0.5 + 7.0 * p - 2.2 * p * p, min(1.0, p * 12), 0.006 * p)
            s.chain(f, hp, rp, H * 0.10 * math.sin(p * 3.14), k * (4 + 16 * p ** 2), min(1.0, p * 12))
            if th_ < 0.12: f = f + (1 - th_ / 0.12) * 0.3
            sh += 5 * math.exp(-th_ * 5) + 5 * p ** 3
        if q >= 0:
            s.chain(f, hp, (imp[0], H * 1.05), H * 0.02, k * 14, 1.0 - clamp(q / 0.8))
            rise = ease2(clamp(q / 1.25)); e = clamp((t - (T_SPL + 1.25)) / (T_HIT - T_SPL - 1.25))
            ed = s.tex(2, s.XX * 0.35 + t * 25, 3 + t * 10) * 0.10 + 0.05 * np.sin(s.xn * 9 + t * 5)                           # 壁の 上の ふち(でこぼこ)
            top = 1.02 - rise * 1.20 + (ed - 0.07) * (0.4 + rise) + 0.10 * np.abs(s.xn - 0.5) * 2 * (1 - rise) ** 0.5
            wm = sstep(-0.004, 0.012, s.yn - top); dep = np.clip((s.yn - top) / 0.9, 0, 1)
            fl1 = s.tex(1, s.XX * 0.55 + 20 * np.sin(s.YY * 0.01 + t * 3), s.YY * 0.16 + t * 520); fl2 = s.tex(0, s.XX * 1.3, s.YY * 0.30 + t * 800)
            wall = SEA0[None, None, :] * (0.5 + 0.8 * dep)[:, :, None] + (SEA1 * 1.15)[None, None, :] * ((1 - dep) ** 1.5 * (0.5 + 0.7 * fl1))[:, :, None] + (sstep(0.55, 0.92, fl1 * 0.6 + fl2 * 0.4) * (0.75 - 0.45 * dep))[:, :, None] * FOAM[None, None, :]
            crest = np.exp(-np.clip(s.yn - top, 0, 9) / (0.035 + 0.05 * rise)) * wm
            wall = wall + (crest * (0.9 + 0.5 * fl2))[:, :, None] * FOAM[None, None, :]
            f = f * (1 - wm * 0.96)[:, :, None] + wall * (wm * 0.96)[:, :, None]
            n = 150; r = s.rnd[200:200 + n]; ag = q * (0.6 + 0.8 * r[:, 0]); sx = imp[0] + (r[:, 1] - 0.5) * W * (0.5 + 2.2 * ag); sy = imp[1] - H * (2.0 * ag - 1.3 * ag * ag) * (0.4 + r[:, 2])
            s.splat(f, sx, sy, (0.4 + 0.6 * r[:, 3]) * clamp(1.6 - q) * 1.5, FOAM, 2.0, 0.4)
            if q < 0.14: f = f + (1 - q / 0.14) * 0.8
            sh += 11 * math.exp(-q * 2.5) + 2 + 7 * e ** 2
            if e > 0:                                                                # 波がしらが くずれて 泡が 上から かぶさる
                fo = s.tex(2, s.XX * 0.6 + t * 40, s.YY * 0.6 - t * 260) * 0.6 + s.tex(0, s.XX * 1.4, s.YY * 1.4 - t * 420) * 0.4
                fm = sstep(0.0, 0.25, e * 1.5 - s.yn + (fo - 0.5) * 0.5) * clamp(e * 3)
                f = f * (1 - fm * 0.9)[:, :, None] + (fm * (1.0 + 0.3 * fo))[:, :, None] * FOAM[None, None, :]
                if e > 0.8: f = f + ((e - 0.8) / 0.2) ** 2 * 1.5
            if q > 0.25: s.beads(f, q - 0.25)
        if u < 0.12: f = f + (1 - u / 0.12) * 0.25
        return f, sh

    # ---- この子だけの曲: 船の 鐘と 手風琴の ような 和音 → かけ声の ような 上がる 音 → 鎖の じゃらじゃら → どぼん → 水の 壁の ごう音 ----
    def audio(s, path):
        a = Snd(DUR, 418); G = 196.0; pn = [0, 2, 4, 7, 9]
        def fq(d, o=0): return G * 2 ** ((pn[d % 5] + 12 * (d // 5 + o)) / 12.0)
        for d in (0, 2, 3): a.note(fq(d, -1), 0.1, 2.6, 0.04, "pad", 0.6)
        for i, tk in enumerate((0.5, 1.5)): a.note(fq(7, 1), tk, 1.2, 0.09, "bell"); a.note(fq(7, 1) * 2.76, tk, 0.8, 0.03, "bell")          # 船の 鐘
        for i, d in enumerate((0, 2, 4, 5, 4, 2)): a.note(fq(d + 5, 0), 0.7 + i * 0.3, 0.32, 0.06, "fm")
        a.boom(T_CUT, 58, 0.6, 0.4); a.whoosh(T_CUT - 0.3, T_CUT, 0.2, 0.1)
        for i in range(5): a.note(fq([5, 7, 9, 10, 12][i], 0), T_CUT + 0.5 + i * 0.13, 0.7, 0.08, "bell")
        for d in (0, 2, 4): a.note(fq(d, 0), T_CUT, 1.9, 0.04, "pad", 0.05)
        a.rise(T_ACT, T_THROW, 150, 420, 0.08); a.boom(T_THROW, 60, 0.6, 0.5); a.whoosh(T_THROW, T_THROW + 0.6, 0.9, 0.25)
        rg = np.random.RandomState(418)
        for i in range(34): tk = T_THROW + 0.05 + i * (T_SPL - T_THROW - 0.1) / 34; a.note(1700 + rg.rand() * 2200, tk, 0.04, 0.035, "bell", 0.002)   # 鎖
        a.rise(T_THROW + 0.8, T_SPL, 700, 120, 0.09)
        a.boom(T_SPL, 38, 1.8, 1.0); a.a += (a.lp(a.nz, 0.25) - a.lp(a.nz, 0.04)) * a.env(T_SPL, 0.005, 0.6) * 1.8 + (a.nz - a.lp(a.nz, 0.6)) * a.env(T_SPL, 0.002, 0.25) * 0.5
        wn = a.lp(a.nz, 0.18) - a.lp(a.nz, 0.02); hi = a.nz - a.lp(a.nz, 0.45)
        g = np.clip((a.t - T_SPL) / (T_HIT - T_SPL), 0, 1) ** 1.3 * (a.t < T_HIT + 0.15) * (a.t > T_SPL)
        a.a += wn * g * 2.0 + hi * g * 0.35
        a.rise(T_SPL + 0.3, T_HIT, 70, 240, 0.11)
        for j, d in enumerate((0, 2, 4, 7)): a.note(fq(d, 0), T_SPL + 0.4 + j * 0.1, T_HIT - T_SPL - 0.3, 0.045, "pad", 0.6)
        a.boom(T_HIT, 40, 1.5, 0.8)
        for j, d in enumerate((0, 2, 4, 7)): a.note(fq(d, 1), T_HIT + 0.02 + j * 0.03, 0.8, 0.05, "bell")
        s.fin_snd(a, DUR, 418); a.save(path)


if __name__ == "__main__":
    r = Anka()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
