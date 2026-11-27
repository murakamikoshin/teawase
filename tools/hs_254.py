"""ソラリス(★5 天輪の機神) 専用の必殺技「天輪ノ裁キ」— 高品質版(hs_fx)。3つの 天輪が こちらに ねらいを 定め、空から 光の 柱が 落ちてくる
朝の 雲海の 上に 浮かぶ → 背の 輪から 小さな 天輪が 3つ 生まれて まわる(前ふり) → 照準の 丸窓の カットイン+技名 →
手のひらを 向ける: 天輪が 1つずつ こちらへ 飛んできて、画面の まわりに カチリと はまる(照準) → 視点が 空を 見上げる →
輪の まん中の 空に 光が 灯り、光の 柱が 落ちてくる。奥の 輪から 順に はじけ、光に のまれる → 白 → 暗転
使い方: python3 hs_254.py 出力.mp4 / python3 hs_254.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.7
T_CUT, T_LOCK, T_TILT, T_FALL, T_HIT = 2.7, 4.7, 6.75, 7.4, 10.0
GOLD = np.array([1.0, 0.80, 0.34], np.float32); WARM = np.array([1.0, 0.93, 0.76], np.float32); SKYB = np.array([0.26, 0.46, 0.86], np.float32); DEEP = np.array([0.02, 0.04, 0.13], np.float32)
EYE = (456, 329); PALM = (231, 384); HALO = (418, 222)
RR = (0.47, 0.335, 0.215)                                                              # はまった時の 輪の 半径(手前・中・奥)


class Solaris(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 254, (1.0, 0.85, 0.45), "天輪ノ裁キ", "Halo Judgment", W); s.fx_init(254)
        s._vt = [s.text_img(c, hz.SERIF, int(66 * s.k), 3) for c in "天輪ノ裁キ"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; s.C = (W * 0.5, s.H * 0.43)

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 朝の 雲海 ----
    def heaven(s, t, lum=1.0):
        W, H, k = s.W, s.H, s.k; hy = 0.70
        v = (s.yn / hy).clip(0, 1)
        f = (SKYB * np.array([0.8, 0.95, 1.15], np.float32))[None, None, :] * (0.55 + 0.5 * (1 - v))[:, :, None] * (1 - v ** 1.8)[:, :, None] + (np.array([1.25, 0.95, 0.55], np.float32))[None, None, :] * (v ** 1.8)[:, :, None]
        f += (np.exp(-np.hypot(s.xn - 0.5, (s.yn - 0.44) * 0.8) / 0.22) * 0.75)[:, :, None] * np.array([1.0, 0.85, 0.55], np.float32)[None, None, :]
        ci = s.fbm(0.0016, t * 10, 0, 4, 60 * k, 1) * np.exp(-((s.yn - 0.22) / 0.2) ** 2); f += (sstep(0.42, 0.8, s.tex(2, s.XX * 0.25 + t * 4, s.YY * 1.1) * 0.5 + ci) * 0.30)[:, :, None] * WARM[None, None, :]     # すじ雲
        d = np.clip((s.yn - hy) / (1 - hy), 0, 1); z = 1.0 / (d + 0.06); n = s.tex(0, (s.xn - 0.5) * z * 34 + t * 2, z * 22 - t * 3) * 0.55 + s.tex(3, (s.xn - 0.5) * z * 80, z * 55 - t * 6) * 0.25 + s.fbm(0.003, t * 8, 0, 3, 0, 2) * 0.3
        cl = (WARM * 1.05)[None, None, :] * sstep(0.35, 0.85, n)[:, :, None] + np.array([0.50, 0.58, 0.82], np.float32)[None, None, :] * (1 - sstep(0.35, 0.85, n))[:, :, None] * (0.75 + 0.3 * d)[:, :, None]
        cl = cl * (1 - 0.35 * np.exp(-d / 0.10))[:, :, None] + (WARM * 0.45)[None, None, :] * np.exp(-d / 0.05)[:, :, None]
        m = sstep(hy - 0.012, hy + 0.012, s.yn + (n - 0.5) * 0.03)[:, :, None]
        return ((f * (1 - m) + cl * m) * lum).astype(np.float32)

    def upper(s, t, top):
        """見上げた 上の 空: 青 → こい 藍 + 星"""
        f = DEEP[None, None, :] * np.ones((s.H, s.W, 1), np.float32) * 1.4 + s.nebula(t, [(0.04, 0.08, 0.26), (0.10, 0.12, 0.40), (0.4, 0.5, 0.8)], 0.0008, (3.0, 1.0), 0.5, 0)
        s.starfield(f, t, 800, (0.9, 0.95, 1.0), 4, 0.9, (0.0, 0.0)); return f

    # ---- 天輪: 金の 輪 + 目もり + 4つの つめ ----
    def halo(s, f, c, R, rot, v=1.0, w=1.0, hot=0.0):
        if v <= 0.01 or R < 3: return
        k = s.k; L = s.L0(); ip = lambda x, y: (int(round(c[0] + x)), int(round(c[1] + y))); th = max(1, int(3.2 * k * w))
        cv2.circle(L, ip(0, 0), int(R), 1.0, th, cv2.LINE_AA); cv2.circle(L, ip(0, 0), int(R * 0.93), 0.55, max(1, int(1.2 * k * w)), cv2.LINE_AA)
        n = 36
        for i in range(n):
            a = rot + i * 6.2832 / n; big = i % 9 == 0; r0 = R * (0.80 if big else 0.885)
            cv2.line(L, ip(math.cos(a) * r0, math.sin(a) * r0), ip(math.cos(a) * R * 0.93, math.sin(a) * R * 0.93), 1.0 if big else 0.55, max(1, int((2.4 if big else 1.0) * k * w)), cv2.LINE_AA)
            if big:
                d = R * 0.085; q = (math.cos(a) * R * 1.02, math.sin(a) * R * 1.02)
                cv2.fillPoly(L, [np.array([ip(q[0] + math.cos(a) * d, q[1] + math.sin(a) * d), ip(q[0] - math.sin(a) * d * 0.4, q[1] + math.cos(a) * d * 0.4), ip(q[0] - math.cos(a) * d * 0.3, q[1] - math.sin(a) * d * 0.3), ip(q[0] + math.sin(a) * d * 0.4, q[1] - math.cos(a) * d * 0.4)], np.int32)], 1.0, cv2.LINE_AA)
        s.glowmask(f, L, GOLD * (1 - 0.6 * hot) + WARM * 0.6 * hot + 0.1, (0.85 + 1.6 * hot) * v, 1.0, 1.0 + hot, 9.0)

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(WARM), rim_dir=(0.0, -1.0), rim_v=0.5 * alpha, amb=(1.0, 0.9, 0.75), amb_v=0.10, aura=GOLD, aura_v=0.16, alpha=alpha, shade=0.93)

    def frame(s, t):
        o = {}
        if t < T_CUT: f, sh = s.sc_sky(t)
        elif t < T_LOCK: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.95)
        else: f, sh = s.sc_judge(t)
        f, sh = s.finale(f, sh, t, DUR, None, (1, 0.9, 0.6), (1, 0.75, 0.3), -1)
        return s.film(f, t, sh * s.k, 0.5, o.get("thr", 0.90), (1.0, 0.93, 0.8), 1.0 + 0.15 * min(sh, 8), 0.035, o.get("vig", 0.50), 1.28, 0.98)

    def angel(s, f, t, size=0.44, cy=None, alpha=1.0, **kw):
        W, H, k = s.W, s.H, s.k; cy = (cy or H * 0.50) + 6 * k * math.sin(t * 2.2)
        img = s.pose(t, **kw); s.draw(f, img, W / 2, cy, size, alpha)

    # 1) 雲海の 上。背の 輪から 小さな 天輪が 3つ 生まれて まわる
    def sc_sky(s, t):
        W, H, k = s.W, s.H, s.k
        f = s.heaven(t)
        orb = []
        for i in range(3):
            q = t - 0.8 - i * 0.5
            if q > 0: a = t * 1.7 + i * 2.094; orb.append((math.sin(a), a, clamp(q * 3), q))
        hc = None
        def ring(o):
            dp, a, v, q = o; c = (hc[0] + math.cos(a) * W * 0.30 * ease(q / 0.5), hc[1] + H * 0.10 + math.sin(a) * H * 0.035 * ease(q / 0.5) + H * 0.02)
            s.halo(f, c, (24 + 7 * dp) * k * (1 + 1.2 * math.exp(-q * 6)), t * 2 + a, v, 0.7, math.exp(-q * 5))
        s.angel(f, t, 0.44, None, clamp(t / 0.6), sway=0.2, breath=0.5); hc = s.at(*s.A(*HALO))
        for o in orb: ring(o)
        s.flare(f, hc, 9 * k, WARM, 0.5, 0.6, t * 0.5, False)
        f = s.godrays(f, (W / 2, H * 0.44), 0.25, 1.05, 8, 0.88, 0.4)
        return f * clamp(t / 0.5), 0

    # 2) 目の カットイン(照準の 丸窓)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.heaven(t, 0.30)
        a = back(clamp(u / 0.32), 1.4); c = (W / 2, H * 0.33); R = W * 0.31 * a
        band = (WARM * 0.85)[None, None, :] * (0.55 + 0.45 * s.tex(1, np.arctan2(s.YY - c[1], s.XX - c[0]) / 6.2832 * 768 + t * 30, 9.0))[:, :, None] + (GOLD * 0.25)[None, None, :]
        ex, ey = s.A(*EYE); zf = 1.5; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(band, img, c[0] - (ex - s.aw / 2) * sc, c[1] - (ey - s.ah / 2) * sc, zf)
        d = np.hypot(s.XX - c[0], s.YY - c[1]); m = sstep(R, R - 2, d)[:, :, None] * (R > 4)
        f = f * (1 - m) + band * m
        s.halo(f, c, R * 1.04, t * 0.8, min(1.0, a), 1.1, math.exp(-u * 5)); s.halo(f, c, R * 1.22, -t * 0.5, 0.5 * min(1.0, a), 0.7)
        n = len(s._vt); step = W * 0.165
        for i, T in enumerate(s._vt):                                                # 技名: 光の中から 1文字ずつ 浮かぶ
            q = (u - 0.5 - i * 0.12) / 0.25
            if q > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.68, clamp(q), 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.35 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.68 - 70 * k, 1.0, 1.0)
        if u > 1.3:
            g = ease((u - 1.3) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.68 + 52 * k), (W / 2 + W * 0.42 * g, H * 0.68 + 52 * k)], GOLD, 1.0, 0.8, 0.8)
            s.text(f, s.txt_en, W / 2, H * 0.68 + 78 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.7
        return f, 4 * math.exp(-u * 6)

    # 3) 天輪が こちらに はまる → 見上げる → 光の 柱が 落ちてくる
    def sc_judge(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_LOCK; C = s.C; sh = 0.0
        tq = ease2(clamp((t - T_TILT) / (T_FALL - T_TILT))); ty = H * 1.05 * tq                                # 見上げる(景色が 下へ)
        e = clamp((t - T_FALL - 0.25) / (T_HIT - T_FALL - 0.25)); rb = W * (0.012 + 1.25 * e ** 2.3) if t > T_FALL else 0.0
        if tq < 1:
            A = s.heaven(t, 1.0 - 0.25 * ease(u / 1.5))
            s.angel(A, t, 0.44 + 0.03 * ease(u / 0.4), None, 1.0, sway=0.2, breath=0.4, squash=-0.03 * math.exp(-u * 6))
            pp = s.at(*s.A(*PALM)); s.flare(A, pp, (10 + 5 * math.sin(t * 11) ** 2) * k, WARM, 1.0, 0.7, t, False)
        else: A = None; pp = (W * 0.36, H * 0.5)
        if tq > 0:
            U = s.upper(t, None)
            if A is not None:
                M = np.float32([[1, 0, 0], [0, 1, ty]]); Ash = cv2.warpAffine(A, M, (W, H), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
                top = (SKYB * 1.0)[None, None, :] * np.exp(-np.clip(ty - s.YY, 0, None) / (H * 0.45))[:, :, None]
                m = sstep(ty + 1, ty - 1, s.YY)[:, :, None]; f = Ash * (1 - m) + (U + top) * m
            else: f = U
        else: f = A
        f *= (0.62 + 0.38 * np.exp(-(np.hypot(s.XX - C[0], s.YY - C[1]) / (W * 0.75)) ** 2 * clamp(u / 1.0)))[:, :, None] if u < 99 else 1.0
        # 光の 柱(まん中から ひろがる)
        if rb > 0:
            dx = s.XX - C[0]; dy = s.YY - C[1]; r = np.hypot(dx, dy) / rb + 1e-4; th = np.arctan2(dy, dx)
            n = s.tex(1, th / 6.2832 * 768, np.log(r + 0.05) * 50 - t * 260) ; n2 = s.tex(3, th / 6.2832 * 1536, np.log(r + 0.05) * 120 - t * 520)
            ins = sstep(1.0, 0.86, r)
            f = f * (1 - ins * 0.9)[:, :, None] + (ins * (0.9 + 0.9 * n ** 1.5 * r + 0.5 * n2 ** 3 * r))[:, :, None] * (WARM * (1 - 0.35 * r.clip(0, 1))[:, :, None] + GOLD[None, None, :] * (0.35 * r.clip(0, 1))[:, :, None])
            f += (np.exp(-(r / 0.35) ** 2) * (0.8 + 1.6 * e))[:, :, None] + (np.exp(-((r - 0.97) / 0.05) ** 2) * 1.6)[:, :, None] * WARM[None, None, :]
            f += (np.exp(-np.clip(r - 1, 0, None) / 0.45) * (r > 1) * (0.25 + 0.9 * s.tex(2, th / 6.2832 * 512, t * 40) ** 2) * 0.8)[:, :, None] * GOLD[None, None, :]
            s.rushdust(f, t, C, 1100, 0.8 + 2.2 * e, WARM, 0.25 + 0.6 * e, 4, 1.0); sh += 2 + 10 * e ** 2
        elif t > T_FALL - 0.45:
            q = clamp((t - (T_FALL - 0.45)) / 0.7); s.flare(f, C, (3 + 12 * q) * k, WARM, q, 1.0, t, True)       # 空に 光が 灯る
        # 3つの 天輪
        for i in range(3):
            tb = T_LOCK + 0.25 + i * 0.62; q = (t - tb) / 0.36
            if q <= 0: continue
            R1 = RR[i] * W; snap = t - tb - 0.36
            if q < 1: g = back(q, 2.2); c = (pp[0] + (C[0] - pp[0]) * ease(q), pp[1] + (C[1] - pp[1]) * ease(q)); R = 8 * k + (R1 - 8 * k) * g; hot = 0.5
            else: c = C; R = R1 * (1 + 0.012 * math.sin(t * 5 + i)); hot = math.exp(-snap * 7)
            v = 1.0; br = 0.0
            if rb > 0:                                                               # 柱の ふちが 通ると はじける
                br = (rb - R1) / (W * 0.10)
                if br > 0: v = max(0.0, 1 - br * 0.9); R = R * (1 + 0.5 * br); hot = max(hot, 1.0 if br < 1 else 0.0); sh += 9 * math.exp(-br * 3)
                elif br > -0.6: hot = max(hot, 0.5 * (1 + br / 0.6))
            s.halo(f, c, R, (1 if i % 2 == 0 else -1) * (t * 0.45 + i) + (0 if q >= 1 else (1 - q) * 5), v, 1.25 - 0.25 * i, hot)
            if 0 <= snap < 0.12: f = f + (1 - snap / 0.12) * 0.35; sh += 6 * (1 - snap / 0.12)
            if 0 < br < 1.2:
                a_ = np.linspace(0, 6.2832, 220, endpoint=False) + i; rr_ = R1 * (1 + 0.9 * br * (0.6 + 0.8 * s.rnd[:220, i])); s.splat(f, C[0] + np.cos(a_) * rr_, C[1] + np.sin(a_) * rr_, (1.2 - br) * (0.5 + s.rnd[:220, 3]), GOLD * 0.8 + 0.2, 1.3, 0.6)
        if rb > 0: f = s.godrays(f, C, 0.25 + 0.4 * e, 1.1, 8, 0.88, 0.45)
        if 0 <= t - T_FALL - 0.25 < 0.1: f = f + (1 - (t - T_FALL - 0.25) / 0.1) * 0.6
        if e > 0.84: f = f + ((e - 0.84) / 0.16) ** 2 * 1.4
        return f, sh

    # ---- この子だけの曲: 明るい 合唱の 和音 → 天輪 3つの 鈴 → カチリ×3(だんだん 高い)→ 見上げる 風 → 無音の 一拍 → 光の 柱の うなりと 輪の はじける 音×3 ----
    def audio(s, path):
        a = Snd(DUR, 254); C4 = 261.63; lyd = [0, 2, 4, 6, 7, 9, 11]
        def fq(d, o=0): return C4 * 2 ** ((lyd[d % 7] + 12 * (d // 7 + o)) / 12.0)
        for j, d in enumerate((0, 4, 7, 9)): a.note(fq(d, -1), 0.05 + j * 0.05, 2.6, 0.05, "pad", 0.5)
        for i in range(3): a.note(fq([7, 9, 11][i], 1), 0.8 + i * 0.5, 0.9, 0.13, "bell"); a.note(fq([7, 9, 11][i], 2), 0.8 + i * 0.5 + 0.03, 0.3, 0.04, "sine")
        a.boom(T_CUT, 55, 0.8, 0.4); a.whoosh(T_CUT - 0.3, T_CUT, 0.25)
        for d in (0, 2, 4, 6, 7): a.note(fq(d, 0), T_CUT, 1.9, 0.04, "pad", 0.05)
        for i in range(5): a.note(fq([4, 6, 7, 9, 11][i], 1), T_CUT + 0.5 + i * 0.12, 0.45, 0.10, "bell")
        for j, d in enumerate((0, 4)): a.note(fq(d, -2), T_LOCK, T_FALL - T_LOCK, 0.07, "saw", 0.3)
        for i in range(3):                                                                                           # 飛んでくる → カチリ
            tb = T_LOCK + 0.25 + i * 0.62; a.rise(tb, tb + 0.36, 300 + 150 * i, 900 + 300 * i, 0.08); ts = tb + 0.36
            a.a += (a.nz - a.lp(a.nz, 0.6)) * a.env(ts, 0.001, 0.025) * 0.7; a.boom(ts, 90, 0.6, 0.08); a.note(fq([4, 7, 11][i], 1), ts, 0.7, 0.13, "bell"); a.note(fq([4, 7, 11][i], 0), ts, 0.3, 0.07, "fm")
        a.whoosh(T_TILT, T_FALL, 0.35, 0.15); a.note(fq(14, 2), T_FALL - 0.4, 0.6, 0.08, "sine")
        tf = T_FALL + 0.25; a.rise(tf, T_HIT, 60, 240, 0.16); a.whoosh(tf, T_HIT, 0.9, 0.3)
        a.a += a.lp(a.nz, 0.06) * np.clip((a.t - tf) / (T_HIT - tf), 0, 1) ** 1.6 * (a.t < T_HIT) * (a.t > tf) * 0.9
        for j, (d, o) in enumerate(((0, -1), (4, -1), (0, 0), (2, 0), (4, 0), (6, 0), (7, 0), (11, 0))): a.note(fq(d, o), tf + j * 0.05, T_HIT - tf + 0.3, 0.045, "pad", 0.4)
        for i in range(3):
            eb = ((RR[2 - i] - 0.012) / 1.25) ** (1 / 2.3); tk = tf + eb * (T_HIT - tf)                              # 奥の 輪から はじける
            a.boom(tk, 50 - 4 * i, 1.0 + 0.2 * i, 0.4); a.a += (a.nz - a.lp(a.nz, 0.5)) * a.env(tk, 0.003, 0.25) * 0.4
            for j in range(6): a.note(fq(11 + j * 2 - i * 2, 1), tk + j * 0.03, 0.25, 0.06, "bell")
        a.boom(T_HIT, 36, 1.7, 0.9)
        for j, d in enumerate((0, 4, 7, 11, 14)): a.note(fq(d, 0), T_HIT + 0.02, 0.6, 0.06, "bell")
        s.fin_snd(a, DUR, 254); a.save(path)


if __name__ == "__main__":
    r = Solaris()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
