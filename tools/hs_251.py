"""ティタニア(★5 花園の女王) 専用の必殺技「花園ノ女王舞」— 高品質版(hs_fx)。花が 咲き、花びらが 舞い、蝶に なって こちらへ 飛んでくる
夜の 花園に 妖精の 女王が 浮かぶ → 足もとに 花が 1輪ずつ 輪に なって 咲く(前ふり) → 目の カットイン+技名 →
ワルツ: 花びらが らせんに 舞い上がる → 花びらが 光る 蝶に 変わり、次々に こちらへ 飛んでくる →
さいごの 1匹(女王と 同じ 色の 羽)が 目の前に とまり、羽が 画面を おおう → 白 → 暗転
使い方: python3 hs_251.py 出力.mp4 / python3 hs_251.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.8
T_CUT, T_DANCE, T_FLY, T_HIT = 2.7, 4.7, 7.1, 10.1
PINK = np.array([1.0, 0.62, 0.80], np.float32); VIO = np.array([0.72, 0.52, 1.0], np.float32); SKYB = np.array([0.45, 0.62, 1.0], np.float32); CREAM = np.array([1.0, 0.95, 0.85], np.float32); LEAF = np.array([0.25, 0.75, 0.45], np.float32)
EYE = (364, 254); WAND = (118, 229)
PAL = [((1.0, 0.86, 0.92), (0.90, 0.55, 1.0), (0.42, 0.58, 1.0)), ((1.0, 0.97, 0.75), (1.0, 0.60, 0.75), (0.95, 0.30, 0.65)), ((0.85, 1.0, 0.92), (0.45, 0.90, 0.95), (0.35, 0.50, 1.0))]


class Titania(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 251, (1.0, 0.6, 0.8), "花園ノ女王舞", "Garden Queen Waltz", W); s.fx_init(251)
        s._vt = [s.text_img(c, hz.SERIF, int(60 * s.k), 3) for c in "花園ノ女王舞"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H
        rg = np.random.RandomState(2511); s._pt = rg.rand(90, 5).astype(np.float32); s._bf = rg.rand(40, 6).astype(np.float32)

    def tex(s, i, u, v):
        return cv2.remap(s._tiles[i], (u % 256).astype(np.float32), (v % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    def box(s, c, rad):
        x0 = max(0, int(c[0] - rad)); x1 = min(s.W, int(c[0] + rad) + 1); y0 = max(0, int(c[1] - rad)); y1 = min(s.H, int(c[1] + rad) + 1)
        return (slice(y0, y1), slice(x0, x1)) if x1 > x0 + 1 and y1 > y0 + 1 else None

    # ---- 夜の 花園 ----
    def garden(s, t, lum=1.0, glow=0.0):
        W, H, k = s.W, s.H, s.k
        f = np.array([0.04, 0.07, 0.16], np.float32)[None, None, :] * (0.5 + 1.5 * s.yn ** 1.4)[:, :, None] + np.array([0.10, 0.16, 0.22], np.float32)[None, None, :] * np.exp(-((s.yn - 0.78) / 0.16) ** 2)[:, :, None]
        f = f + s.nebula(t, [(0.06, 0.12, 0.26), (0.20, 0.12, 0.34), (0.50, 0.40, 0.70)], 0.0008, (4.0, 1.0), 0.55, 0)
        s.starfield(f, t, 700, (0.85, 0.92, 1.0), 3, 0.8, (0.0, 0.0))
        f += (np.exp(-np.hypot(s.xn - 0.5, (s.yn - 0.55) * 0.7) / 0.40) * (0.10 + 0.30 * glow))[:, :, None] * (PINK * 0.6 + VIO * 0.4)[None, None, :]
        hy = 0.80 + 0.015 * np.sin(s.xn * 4.0 + 0.6) + 0.006 * np.sin(s.xn * 15.0)
        m = sstep(hy - 0.004, hy + 0.004, s.yn); d = np.clip((s.yn - hy) / 0.2, 0, 1)
        gr = s.tex(2, s.XX * 1.3 + np.sin(t * 1.1 + s.YY * 0.03) * 3 * k, s.YY * 1.6) * 0.5 + s.tex(0, s.XX * 0.4, s.YY * 0.7) * 0.6
        col = np.array([0.012, 0.045, 0.04], np.float32)[None, None, :] * (0.4 + 1.5 * gr)[:, :, None] * (1.5 - 1.0 * d)[:, :, None]
        col += (np.exp(-np.hypot(s.xn - 0.5, (s.yn - 0.88) * 2.2) / 0.30) * (0.08 + 0.30 * glow) * (0.5 + gr))[:, :, None] * (LEAF * 0.5 + PINK * 0.5)[None, None, :]
        f = f * (1 - m)[:, :, None] + col * m[:, :, None]
        fl = np.zeros_like(f); s.starfield(fl, t, 70, (0.8, 1.0, 0.5), 5, 2.2, (9.0, -6.0)); f += fl * (s.yn > 0.45)[:, :, None]        # ほたる
        return (f * lum).astype(np.float32)

    # ---- 花(式): 5枚の 花びら・しん・うすい 光 ----
    def flower(s, f, c, R, op, sq=0.5, ph=0.0, v=1.0):
        if op <= 0.01 or v <= 0.01: return
        sl = s.box(c, R * 1.7)
        if sl is None: return
        x = s.XX[sl] - c[0]; y = (s.YY[sl] - c[1]) / sq; r = np.hypot(x, y) + 1e-4; th = np.arctan2(y, x)
        f[sl] += (np.exp(-(r / (R * 1.1)) ** 2) * 0.28 * op * v)[:, :, None] * PINK[None, None, :]
        for lay, (rs, pp, cb) in enumerate(((1.0, 0.0, 0.80), (0.68, 0.63, 1.0))):                         # 外の 5枚 + 内の 5枚
            lobe = np.abs(np.cos(2.5 * (th + ph + pp))) ** 0.6; rp = R * rs * op * (0.38 + 0.62 * lobe)
            q = r / rp; ins = sstep(1.0, 0.90, q) * v
            col = (CREAM[None, None, :] * (1 - sstep(0.2, 1.0, q))[:, :, None] + PINK[None, None, :] * sstep(0.2, 1.0, q)[:, :, None]) * (cb * (0.55 + 0.45 * lobe ** 0.5) * (1 - 0.35 * sstep(0.85, 1.0, q)))[:, :, None]
            f[sl] = f[sl] * (1 - ins)[:, :, None] + col * 1.15 * ins[:, :, None]
        core = sstep(R * 0.17 * op, R * 0.11 * op, r) * v
        f[sl] = f[sl] * (1 - core)[:, :, None] + np.array([1.3, 1.0, 0.35], np.float32)[None, None, :] * core[:, :, None]

    def ringpos(s, i, n=9):
        a = -1.5708 + (i + 0.5) * 6.2832 / n; return (s.W / 2 + math.cos(a) * s.W * 0.40, s.H * 0.885 + math.sin(a) * s.H * 0.050, math.sin(a))

    def ringflowers(s, f, t, n_open, front, v=1.0):
        for i in range(9):
            x, y, dp = s.ringpos(i)
            if (dp > 0) != front: continue
            op = back(clamp((n_open - i) / 1.6), 1.3) if n_open > i else 0.0
            s.flower(f, (x, y), (29 + 13 * dp) * s.k * (1 + 0.05 * math.sin(t * 3 + i)), op, 0.55, i * 0.7 + t * 0.15, v)

    # ---- 花びら(らせんに 舞う) ----
    def petals(s, f, t, u, front, v=1.0):
        if v <= 0.01: return
        W, H, k = s.W, s.H, s.k; L = s.L0(); L2 = s.L0()
        for i in range(90):
            p = s._pt[i]; q = (u * (0.22 + 0.16 * p[1]) + p[0]) % 1.0
            if u * (0.22 + 0.16 * p[1]) + p[0] < 1.0 and q > u * 0.6 + 0.05: continue                      # 下から 順に 立ちのぼる
            a = p[2] * 6.2832 + t * (1.6 + 1.2 * p[3]) + q * 5; dp = math.sin(a)
            if (dp > 0) != front: continue
            rad = W * (0.40 - 0.17 * q) * (0.8 + 0.4 * p[4]); x = W / 2 + math.cos(a) * rad; y = H * (0.87 - 0.72 * q) + math.sin(a) * H * 0.03
            sz = (10.5 + 5.0 * dp) * k * (0.7 + 0.6 * p[4]); fade = min(1.0, q * 8) * min(1.0, (1 - q) * 5)
            cv2.ellipse(L if i % 3 else L2, (int(x), int(y)), (max(1, int(sz)), max(1, int(sz * (0.25 + 0.3 * abs(math.sin(t * 4 + i)))))), math.degrees(a * 1.7 + i), 0, 360, float(fade * (0.75 + 0.25 * dp)), -1, cv2.LINE_AA)
        for LL, col in ((L, PINK * 0.9 + 0.1), (L2, CREAM)):
            m = np.clip(LL, 0, 1) * v; f *= (1 - m * 0.8)[:, :, None]; f += (m * 1.05 + cv2.GaussianBlur(m, (0, 0), 6 * k) * 0.35)[:, :, None] * col[None, None, :]

    # ---- 蝶(式): 上下の 羽・すじ・ふち・胴。flap で 羽ばたく ----
    def bfly(s, f, c, S, rot, flap, v=1.0, pal=0, fmin=0.14):
        if v <= 0.01 or S < 2: return
        sl = s.box(c, S * 1.3)
        if sl is None: return
        x = s.XX[sl] - c[0]; y = s.YY[sl] - c[1]; cr, sr = math.cos(rot), math.sin(rot); xr = x * cr + y * sr; yr = -x * sr + y * cr
        fx = max(fmin, abs(math.cos(flap))); ax = np.abs(xr) / fx / S; ay = -yr / S; r = np.hypot(ax, ay) + 1e-4; th = np.arctan2(ay, ax)
        ru = np.where(np.abs(th - 0.62) < 1.25, 0.60 + 0.40 * np.cos((th - 0.62) * 2.1), 0.0); rl = np.where((th < 0.05) & (th > -1.52), 0.70 * (0.52 + 0.48 * np.cos((th + 0.80) * 2.5)), 0.0)
        up = ru >= rl; R = np.maximum(ru, rl) + 1e-4; q = r / R; soft = min(0.12, 2.5 / S); ins = sstep(1.0, 1.0 - soft, q) * v
        ci, cm, co = [np.asarray(c_, np.float32) for c_ in PAL[pal % 3]]
        w1 = sstep(0.10, 0.50, q)[:, :, None]; w2 = sstep(0.50, 0.92, q)[:, :, None]; col = ci[None, None, :] * (1 - w1) + cm[None, None, :] * w1 * (1 - w2) + co[None, None, :] * w2
        lum = 1 - 0.85 * sstep(0.84, 0.97, q)
        if S > 16 * s.k:
            vein = np.maximum(sstep(0.93, 0.99, np.abs(np.cos(th * 8.0 + up * 1.3))) * sstep(0.12, 0.3, q), sstep(0.95, 0.995, np.abs(np.cos(q * 7.0 + np.sin(th * 8.0 + up * 1.3) * 0.8))))
            lum = lum * (1 - 0.80 * vein * min(1.0, (S - 16 * s.k) / (20 * s.k)))
            if S > 90 * s.k: lum = lum * (0.82 + 0.36 * s.tex(1, xr * 0.5 + 60, yr * 0.5))              # 大きい 羽は 光が すける むら
            spot = np.exp(-((q - 0.72) / 0.07) ** 2) * sstep(0.6, 0.9, np.cos(th * 8.0 + up * 1.3 + 3.14)); col = col + spot[:, :, None] * 0.6
        f[sl] += (np.exp(-(np.hypot(xr, yr) / (S * 0.9)) ** 2) * 0.16 * v)[:, :, None] * cm[None, None, :]
        f[sl] = f[sl] * (1 - ins * 0.92)[:, :, None] + col * (lum * ins * 1.25)[:, :, None]
        body = np.exp(-((xr / (0.045 * S + 0.6)) ** 2 + ((yr - 0.02 * S) / (0.36 * S)) ** 2) ** 1.5) * v
        f[sl] *= (1 - 0.92 * np.clip(body * 1.6, 0, 1))[:, :, None]

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        w = getattr(s, '_glow', 0.0)
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(CREAM * 0.6 + PINK * 0.4), rim_dir=(0.0, 1.0), rim_v=(0.25 + 0.5 * w) * alpha, amb=(0.75, 0.80, 1.0), amb_v=0.14 * (1 - w), aura=PINK * 0.6 + VIO * 0.4, aura_v=0.10 + 0.14 * w, alpha=alpha, shade=0.84 + 0.12 * w)

    def frame(s, t):
        o = {}; s._glow = 0.0
        if t < T_CUT: f, sh = s.sc_bloom(t)
        elif t < T_DANCE: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.92)
        else: f, sh = s.sc_waltz(t)
        f, sh = s.finale(f, sh, t, DUR, None, (1, 0.7, 0.85), (0.7, 0.5, 1.0), -1)
        return s.film(f, t, sh * s.k, 0.55, o.get("thr", 0.82), (1.0, 0.88, 0.95), 1.0 + 0.15 * min(sh, 8), 0.035, o.get("vig", 0.55), 1.15, 1.08)

    def hover(s, t): return s.H * (0.575 + 0.012 * math.sin(t * 2.1))

    # 1) 夜の 花園。足もとに 花が 1輪ずつ 咲く
    def sc_bloom(s, t):
        W, H, k = s.W, s.H, s.k; n = (t - 0.7) / 0.19; g = clamp(n / 9); s._glow = 0.5 * g
        f = s.garden(t, 1.0, g)
        s.ringflowers(f, t, n, False)
        dn = ease(t / 0.9); img = s.pose(t, sway=0.35, breath=0.6); s.draw(f, img, W / 2, s.hover(t) - (1 - dn) * H * 0.10, 0.42, clamp(t / 0.6))
        wp = s.at(*s.A(*WAND)); s.flare(f, wp, (6 + 5 * g + 2 * math.sin(t * 8) ** 2) * k, PINK * 0.5 + CREAM * 0.5, 0.5 + 0.5 * g, 0.5, t, False)
        s.ringflowers(f, t, n, True)
        return f * clamp(t / 0.5), 0

    # 2) 目の カットイン(花びらの 形の 窓)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT; s._glow = 0.6
        f = s.garden(t, 0.35, 1.0)
        a = ease(u / 0.28); cy = H * 0.33; hh = H * 0.135
        band = (PINK * 0.55)[None, None, :] * (0.6 + 0.6 * s.fbm(0.002, t * 30, t * 12, 3, 30 * k, 1))[:, :, None] + (VIO * 0.25)[None, None, :] * s.yn[:, :, None]
        ex, ey = s.A(*EYE); zf = 1.45; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(band, img, W / 2 - (ex - s.aw / 2) * sc - (1 - a) * W * 0.9 + u * 5 * k, cy - (ey - s.ah / 2) * sc, zf)
        edge = hh * a * np.clip(1 - ((s.XX - W / 2) / (W * 0.60)) ** 2, 0, 1) ** 0.7; dd = np.abs(s.YY - cy)
        m = sstep(edge, edge - 2, dd)[:, :, None] * (edge > 1)[:, :, None]
        f = f * (1 - m) + band * m
        s.glowmask(f, np.exp(-((dd - edge) / (1.6 * k)) ** 2) * (edge > 1), CREAM * 0.6 + PINK * 0.4, 0.9 * a, 1.0, 0.8, 8.0)
        n = len(s._vt); step = W * 0.148
        for i, T in enumerate(s._vt):                                                # 技名: ふわりと 1文字ずつ(ワルツの 3拍子)
            q = (u - 0.5 - i * 0.12) / 0.22
            if q > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.66 + (1 - ease(q)) * 18 * k, clamp(q * 1.5), 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.35 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.66 - 66 * k, 1.0, 1.0)
        if u > 1.3:
            g = ease((u - 1.3) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.66 + 50 * k), (W / 2 + W * 0.42 * g, H * 0.66 + 50 * k)], PINK * 0.6 + CREAM * 0.4, 1.0, 0.8, 0.8)
            s.text(f, s.txt_en, W / 2, H * 0.66 + 76 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.7
        return f, 4 * math.exp(-u * 6)

    # 3) ワルツ: 花びらが らせんに 舞う → 蝶に なって こちらへ。さいごの 1匹の 羽が 画面を おおう
    def sc_waltz(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_DANCE; uf = t - T_FLY; s._glow = 1.0
        TL = T_HIT - 1.75; pl = clamp((t - TL) / (T_HIT - TL)); big = pl ** 2.6                             # さいごの 1匹
        f = s.garden(t, 1.0 - 0.45 * big, 1.0)
        pv = 1 - ease(uf / 0.8) if uf > 0 else 1.0
        s.ringflowers(f, t, 99, False); s.petals(f, t, u, False, pv)
        bl = []
        if uf > 0:
            for i in range(34):                                                      # 蝶: 花びらの らせんから 生まれ、1匹ずつ こちらへ
                b = s._bf[i]; tb = i * 0.062; p = (uf - tb) / (1.15 + 0.5 * b[3])
                if p <= 0 or p >= 1: continue
                a0 = b[0] * 6.2832; st = (W / 2 + math.cos(a0) * W * 0.26, H * (0.50 - 0.30 * b[1])); da = b[2] * 6.2832; dr = W * (0.35 + 1.0 * b[4])
                pp = p ** 2.4; x = st[0] + math.cos(da) * dr * pp + math.sin(t * 5 + i) * 14 * k * (1 - p); y = st[1] + math.sin(da) * dr * pp * 1.3 + math.cos(t * 4 + i * 2) * 10 * k * (1 - p) - 30 * k * math.sin(p * 3.1)
                S = k * (11 + 170 * p ** 2.3) * (0.7 + 0.6 * b[5]); al = min(1.0, p * 6) * (1 - clamp((p - 0.86) / 0.14))
                bl.append((p, (x, y), S, math.sin(t * 3 + i) * 0.35 + (b[2] - 0.5) * 0.8, t * (13 + 5 * b[5]) + i, al, i % 3))
        bl.sort(key=lambda z: z[0])
        for z in bl:
            if z[0] < 0.45: s.bfly(f, z[1], z[2], z[3], z[4], z[5], z[6])
        a_ = 1 - clamp((pl - 0.55) / 0.25)
        if a_ > 0.02:
            beat = u * 2 * math.pi / 0.78                                            # 3拍子で ゆれる
            img = s.pose(t, sway=0.6 + 0.3 * math.sin(beat / 3), breath=0.7, lean=9 * k * math.sin(beat / 3), squash=-0.025 * abs(math.sin(beat / 2)))
            s.draw(f, img, W / 2 + 16 * k * math.sin(beat / 3), s.hover(t) - 0.02 * H * abs(math.sin(beat / 2)), 0.42, a_)
            s.flare(f, s.at(*s.A(*WAND)), (10 + 4 * math.sin(t * 9) ** 2) * k, PINK * 0.5 + CREAM * 0.5, 0.9 * a_, 0.6, t * 1.5, False)
        s.petals(f, t, u, True, pv); s.ringflowers(f, t, 99, True, 1 - 0.7 * big)
        for z in bl:
            if z[0] >= 0.45: s.bfly(f, z[1], z[2], z[3], z[4], z[5], z[6])
        sh = 0.0
        if pl > 0:
            c = (W / 2 + W * 0.16 * (1 - pl) * math.sin(t * 3), H * (0.20 + 0.30 * ease2(pl)) + 8 * k * math.sin(t * 6) * (1 - pl))
            S = k * (9 + 46 * pl) + W * 1.9 * big; slow = 1 - ease2(clamp((pl - 0.55) / 0.3))
            flap = math.sin(t * 13) * 1.1 * slow                                       # とまると 羽ばたきが ゆっくりに → ひろげる
            s.bfly(f, c, S, 0.12 * (1 - pl) * math.sin(t * 2), flap, min(1.0, pl * 6), 0, 0.42)
            f = s.godrays(f, c, 0.35 * big, 1.0, 8, 0.88, 0.4); sh = 5 * big
            if pl > 0.84: f = f + ((pl - 0.84) / 0.16) ** 2 * 1.3
        if 0 <= uf < 0.15: f = f + (1 - uf / 0.15) * 0.5
        if u < 0.12: f = f + (1 - u / 0.12) * 0.4
        return f, sh

    # ---- この子だけの曲: オルゴールの ワルツ(3拍子・F の 明るい 調べ)。花 9輪 = 9つの 音 → ズン・チャッ・チャ → 蝶が 来るほど 高く 細かく ----
    def audio(s, path):
        a = Snd(DUR, 251); F = 174.61; maj = [0, 2, 4, 5, 7, 9, 11]
        def fq(d, o=0): return F * 2 ** ((maj[d % 7] + 12 * (d // 7 + o)) / 12.0)
        for d in (0, 4): a.note(fq(d, 0), 0.0, 2.6, 0.035, "pad", 0.6)
        for i in range(9): a.note(fq([0, 2, 4, 7, 4, 7, 9, 11, 14][i], 1), 0.7 + i * 0.19, 0.6, 0.11, "bell"); a.note(fq([0, 2, 4, 7, 4, 7, 9, 11, 14][i], 2), 0.7 + i * 0.19 + 0.02, 0.2, 0.03, "sine")
        a.boom(T_CUT, 60, 0.6, 0.4); a.whoosh(T_CUT - 0.3, T_CUT, 0.2)
        for d in (0, 2, 4, 6): a.note(fq(d, 0), T_CUT, 1.9, 0.04, "pad", 0.05)
        for i in range(6): a.note(fq([4, 5, 7, 9, 7, 11][i], 1), T_CUT + 0.5 + i * 0.12, 0.4, 0.11, "bell")
        bt = 0.26; ch = [(0, 2, 4), (3, 5, 7), (4, 6, 8), (0, 2, 4)]; mel = [7, 9, 11, 9, 7, 4, 5, 7, 9, 7, 5, 2, 4, 6, 8, 11, 9, 8, 7, 9, 11, 14, 11, 9]
        nb = int((T_FLY - T_DANCE) / bt)
        for j in range(nb + 12):                                                                                      # ズン・チャッ・チャ
            tk = T_DANCE + j * bt
            if tk > T_HIT - 0.2: break
            c = ch[(j // 3) % 4]; g = 1.0 if tk < T_FLY else 1.0 + (tk - T_FLY) / 3
            if j % 3 == 0: a.note(fq(c[0], -1), tk, 0.5, 0.15, "saw"); a.boom(tk, 70, 0.25, 0.1)
            else: a.note(fq(c[1], 0), tk, 0.15, 0.05, "fm"); a.note(fq(c[2], 0), tk, 0.15, 0.05, "fm")
            a.note(fq(mel[j % len(mel)], 1), tk, 0.45, 0.10 * g, "bell")
            if tk > T_FLY: a.note(fq(mel[j % len(mel)] + 2, 2), tk + bt / 2, 0.2, 0.05, "bell"); a.hat(tk + bt / 2, 0.04)
        for i in range(34): a.note(fq(i % 9 + 7, 2), T_FLY + i * 0.062, 0.08, 0.022, "sine")                           # 蝶の 羽音
        a.whoosh(T_FLY, T_FLY + 0.4, 0.3, 0.4); a.rise(T_HIT - 1.75, T_HIT, 120, 420, 0.09)
        for j, d in enumerate((0, 2, 4, 6, 7, 9)): a.note(fq(d, 0), T_HIT - 1.75 + j * 0.05, 2.0, 0.045, "pad", 0.5)
        a.a += a.lp(a.nz, 0.2) * np.clip((a.t - (T_HIT - 1.2)) / 1.2, 0, 1) ** 2 * (a.t < T_HIT) * 0.35
        a.boom(T_HIT, 44, 1.2, 0.8)
        for j, d in enumerate((0, 4, 7, 11, 14)): a.note(fq(d, 1), T_HIT + 0.02 + j * 0.03, 0.7, 0.07, "bell")
        s.fin_snd(a, DUR, 251); a.save(path)


if __name__ == "__main__":
    r = Titania()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
