"""エクリプス(★6 黒い太陽の王) 専用の必殺技「日食ノ王冠」— 高品質版(hs_fx)・通し直し。王の 影が こちらへ のびて 立ち上がり、目の前を おおう
まひるの 野。ひざまずく 王と、こちらへ のびる 影(前ふり) → 金の 目が ひらく(左は 明るい地に 黒い目、右は 闇に 金の目の 二つ割りカットイン) →
杖の「黒い太陽」が 空へ のぼり、日を くう。空が 暗く なり、コロナが ひらき、地平に 夕やけの 輪。影は 足もとから こちらへ のびつづける →
影が 目の前で 立ち上がり、巨大な「影の王」に なって 視界を おおう。闇の 中に 金の 目が ふたつ ひらく → 頭の 上に 日食の 輪(王冠)。その 一点が 光って(ダイヤモンドリング)→ 白 → 暗転
使い方: python3 hs_77.py 出力.mp4 / python3 hs_77.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old77", "/home/claude/tools/pre_fin/hs_77_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = 10.3
GOLD, SKY, SAND, INK = old.GOLD, old.SKY, old.SAND, old.INK
T_OPEN, T_CUT, T_RISE = old.T_OPEN, old.T_CUT, old.T_RISE
T_TOT, T_KING, T_DARK, T_RING, T_HIT = 6.2, 6.5, 8.45, 9.0, 9.6
PEARL = np.array([1.0, 0.95, 0.86], np.float32)


class Eclipse(FX, old.Eclipse):
    def __init__(s, W=540):
        old.Eclipse.__init__(s, W); s.fx_init(77); H = s.H; k = s.k; s.yn = s.yn.astype(np.float32); s.SC = (W * 0.5, H * 0.15); s.SR = W * 0.085; s._ecl = 0.0
        g0 = s.gnd; dz = (H - g0) / (np.clip(s.YY - g0, 0, None) + 5 * k)
        s._gt = cv2.remap(s._tiles[2], (((s.XX - W / 2) * dz * 0.42 / k + 128) % 256).astype(np.float32), ((dz * 95) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        hx = np.interp((np.arange(W) * 0.3 / k) % 255, np.arange(256), s._tiles[1][11]); s._ridge = sstep(0, 3 * k, s.YY - (g0 - H * (0.01 + 0.035 * hx))[None, :]); s._gm = sstep(g0 - 1.5, g0 + 1.5, s.YY)

    # ---- まひるの 野 → 日食の 闇 ----
    def noon(s, t, lum=1.0, ecl=0.0):
        W, H, k = s.W, s.H, s.k; n = s.fbm(0.0009, t * 10, 0, 4, 60 * k, 0); hz_ = np.exp(-np.clip(s.gnd - s.YY, 0, None) / (H * 0.10))
        day = SKY[None, None, :] * (1.15 - 0.5 * s.yn)[:, :, None] + (0.25 * s.yn ** 2 + sstep(0.52, 0.85, n) * 0.16 * sstep(0.75, 0.3, s.yn))[:, :, None] + (hz_ * 0.18)[:, :, None]
        ngt = np.array([0.018, 0.022, 0.065], np.float32)[None, None, :] * (0.6 + 0.9 * n)[:, :, None] + (hz_ * 0.60)[:, :, None] * np.array([1.0, 0.42, 0.16], np.float32)[None, None, :] + (np.exp(-np.clip(s.gnd - s.YY, 0, None) / (H * 0.30)) * 0.10)[:, :, None] * np.array([0.5, 0.25, 0.6], np.float32)[None, None, :]
        f = day * (1 - ecl) + ngt * ecl
        if ecl > 0.3: s.starfield(f, t, 500, (1.0, 0.97, 0.9), 2, (ecl - 0.3) * 1.2 * 1.0, (0.0, 0.0))
        sand = SAND[None, None, :] * ((0.62 + 0.55 * s._gt) * (1.05 - 0.3 * np.clip((s.YY - s.gnd) / (H - s.gnd), 0, 1)))[:, :, None]
        gdk = sand * 0.09 + (np.exp(-np.clip(s.YY - s.gnd, 0, None) / (H * 0.05)) * 0.20 * (0.5 + 0.8 * s._gt))[:, :, None] * np.array([1.0, 0.45, 0.2], np.float32)[None, None, :]
        rd = (SAND * 0.78 + SKY * 0.18)[None, None, :] * (0.8 + 0.3 * s._gt)[:, :, None]; rdk = np.array([0.03, 0.02, 0.035], np.float32)[None, None, :] * np.ones((H, W, 1), np.float32)
        f = f * (1 - s._ridge[:, :, None]) + (rd * (1 - ecl) + rdk * ecl) * s._ridge[:, :, None]; f = f * (1 - s._gm[:, :, None]) + (sand * (1 - ecl) + gdk * ecl) * s._gm[:, :, None]
        return (f * lum).astype(np.float32)

    def eclipse(s, f, t, cov, mc=None, mr=None, v=1.0):
        """日と、それを くう 黒い太陽。cov 0..1。皆既で コロナ・紅炎"""
        W, H, k = s.W, s.H, s.k; Sc = s.SC; R = s.SR; mc = Sc if mc is None else mc; mr = R * 1.03 if mr is None else mr
        d = np.hypot(s.XX - Sc[0], s.YY - Sc[1]); dm = np.hypot(s.XX - mc[0], s.YY - mc[1]); vis = clamp(math.hypot(mc[0] - Sc[0], mc[1] - Sc[1]) / (2 * R)) if cov < 1 else 0.0; tot = float(sstep(0.86, 1.0, cov))
        f += ((np.exp(-(d / R) ** 6) * 2.2 + np.exp(-d / (2.2 * R)) * 0.7 + np.exp(-d / (9 * R)) * 0.22) * (0.15 + 0.85 * vis) * (1 - tot) * v)[:, :, None] * (PEARL * 0.8 + GOLD * 0.2)[None, None, :]
        if tot > 0.01:
            an = np.arctan2(s.YY - mc[1], s.XX - mc[0]); lr = np.log(dm / mr + 1e-3)
            st = cv2.remap(s._tiles[0], ((an / 6.2832 * 768 + t * 3) % 256).astype(np.float32), ((lr * 40 - t * 4) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            cor = np.exp(-np.clip(dm - mr, 0, None) / (mr * 0.75)) * (0.30 + 0.9 * (0.5 + 0.5 * np.cos(an * 12 + t * 0.3)) ** 3 + 0.7 * st) * (dm > mr * 0.98) + np.exp(-((dm - mr) / (mr * 0.05)) ** 2) * 1.6
            f += (cor * tot * v)[:, :, None] * (PEARL * 0.75 + GOLD * 0.25)[None, None, :]
            pr = cv2.remap(s._tiles[2], ((an / 6.2832 * 256) % 256).astype(np.float32), np.full(an.shape, 9.0, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            f += (np.exp(-((dm - mr * 1.05) / (mr * 0.05)) ** 2) * sstep(0.68, 0.85, pr) * tot * v * 1.2)[:, :, None] * np.array([1.0, 0.25, 0.32], np.float32)[None, None, :]
        m = sstep(mr + 1.2, mr - 1.2, dm) * v; f *= (1 - m)[:, :, None]; f += (m * 0.02)[:, :, None] * np.array([0.6, 0.4, 1.0], np.float32)[None, None, :]
        if cov < 0.9:                                                                # のぼる間は 杖と 同じ 金の とげ
            L = s.L0()
            for i in range(12):
                a = i * 6.2832 / 12 + t * 0.3; w = 0.16; cv2.polylines(L, [np.array([(mc[0] + math.cos(a - w) * mr * 1.04, mc[1] + math.sin(a - w) * mr * 1.04), (mc[0] + math.cos(a) * mr * (1.55 + 0.2 * (i % 2)), mc[1] + math.sin(a) * mr * (1.55 + 0.2 * (i % 2))), (mc[0] + math.cos(a + w) * mr * 1.04, mc[1] + math.sin(a + w) * mr * 1.04)], np.int32)], False, 1.0, 1, cv2.LINE_AA)
            cv2.circle(L, (int(mc[0]), int(mc[1])), int(mr * 1.04), 1.0, 1, cv2.LINE_AA); s.glowmask(f, L, GOLD, 0.8 * (1 - cov / 0.9) * v, 1.0, 0.8, 6.0)

    def gshadow(s, f, cx, feet, hfrac, ln, alpha):
        """足もとから こちらへ のびる 影(手前ほど 広く・ぼける)"""
        W, H, k = s.W, s.H, s.k; sc = hfrac * H / s.ah; w = s.pw * sc; h = s.ph * sc * ln; sp = 1 + 0.9 * ln; src = np.ascontiguousarray(s.sil[::-1, :, 3])
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [s.pw, 0], [s.pw, s.ph], [0, s.ph]]), np.float32([[cx - w / 2, feet - 6 * k], [cx + w / 2, feet - 6 * k], [cx + w / 2 * sp, feet + h], [cx - w / 2 * sp, feet + h]]))
        im = cv2.warpPerspective(src, M, (W, H), flags=cv2.INTER_LINEAR); im = cv2.GaussianBlur(im, (0, 0), 1.5 * k) * 0.6 + cv2.GaussianBlur(im, (0, 0), 6 * k) * 0.4
        a = (np.clip(im, 0, 1) * alpha)[:, :, None]; f[:] = f * (1 - a) + f * a * np.array([0.10, 0.10, 0.22], np.float32)[None, None, :]

    def sking(s, f, t, g, ev):
        """目の前で 立ち上がる 影の王(闇の けむり・ふちの 金の 光・金の 目)"""
        W, H, k = s.W, s.H, s.k; hk = 0.7 + 1.2 * g; cy = H * (1.50 - 0.83 * ease2(g) ** 0.8); sc = hk * H / s.ah
        big = cv2.resize(s.sil[:, :, 3], None, fx=sc, fy=sc, interpolation=cv2.INTER_LINEAR); bh, bw = big.shape; x0 = int(W / 2 - bw / 2); y0 = int(cy - bh / 2); A = s.L0()
        xa, ya = max(0, x0), max(0, y0); xb, yb = min(W, x0 + bw), min(H, y0 + bh)
        if xb > xa and yb > ya: A[ya:yb, xa:xb] = big[ya - y0:yb - y0, xa - x0:xb - x0]
        n = s.fbm(0.0030, t * 30, -t * 70, 3, 0, 2); n2 = s.fbm(0.0030, 50 - t * 20, -t * 50, 3, 0, 1)
        A = cv2.remap(A, (s.XX + (n - 0.5) * 20 * k).astype(np.float32), (s.YY + (n2 - 0.5) * 20 * k).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); A = np.clip(cv2.GaussianBlur(A, (0, 0), 2 * k) * 1.15, 0, 1)
        col = np.array([0.010, 0.007, 0.028], np.float32)[None, None, :] + (n ** 2 * 0.07)[:, :, None] * np.array([0.7, 0.35, 1.0], np.float32)[None, None, :]
        f[:] = f * (1 - A[:, :, None] * 0.98) + col * A[:, :, None]
        Ab = cv2.GaussianBlur(A, (0, 0), 3 * k); rim = np.clip(Ab - np.roll(Ab, int(6 * k), axis=0), 0, 1); f += (rim * 0.40 * (1 - 0.5 * g))[:, :, None] * GOLD[None, None, :]
        f += (np.clip(cv2.GaussianBlur(A, (0, 0), 14 * k) - A, 0, 1) * 0.25)[:, :, None] * np.array([0.35, 0.15, 0.6], np.float32)[None, None, :]
        s._pl = (W / 2, cy, sc); s._pose = dict(t=t, sway=0, breath=0, lean=0, squash=0, stretch=None); return [s.at(*s.eyeB), s.at(*s.eyeG)], sc

    def keyes(s, f, pts, sc, v, t):
        if v <= 0.01: return
        for p in pts:
            L = s.L0(); cv2.ellipse(L, (int(p[0]), int(p[1])), (max(1, int(15 * sc)), max(1, int(21 * sc * min(1.0, v)))), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
            sl = s.L0(); cv2.ellipse(sl, (int(p[0]), int(p[1])), (max(1, int(3.5 * sc)), max(1, int(17 * sc * min(1.0, v)))), 0, 0, 360, 1.0, -1, cv2.LINE_AA); L = np.clip(cv2.GaussianBlur(L, (0, 0), 0.6 * s.k) - cv2.GaussianBlur(sl, (0, 0), 0.6 * s.k) * 0.92, 0, 1)
            s.glowmask(f, L, GOLD, 1.1 * min(1.2, v), 1.0, 1.0, max(6.0, 22 * sc)); s.flare(f, p, 10 * sc, GOLD, 0.5 * min(1.2, v), 1.4, 0.0, False)

    def gold_eye(s, f, v, streak=1.0):
        if v <= 0.01: return
        old.Eclipse.gold_eye(s, f, min(1.0, v), streak); s.flare(f, s.at(*s.eyeG), 6 * s._pl[2] * 2, GOLD, 0.45 * v, 0.8, 0.0, False)

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        e = s._ecl
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(PEARL * (1 - e) + GOLD * e), rim_dir=(0.0, -1.0), rim_v=(0.6 + 0.4 * e) * alpha, amb=(0.6, 0.5, 0.9) if e > 0.5 else (1.0, 0.97, 0.9), amb_v=0.15, alpha=alpha, shade=1.0 - 0.55 * e)

    def add(s, f, L, col, gain=1.0, b1=3, b2=12):
        s.glowmask(f, np.clip(L, 0, 1), col, gain * 0.8, 1.0, 1.0, float(b2))

    def frame(s, t):
        o = {}; s._ecl = 0.0
        if t < T_CUT: f, sh = s.sc_noon(t); f = f * clamp(t / 0.4); o = dict(thr=0.95, vig=0.4)
        elif t < T_RISE: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.4, thr=0.92)
        else: f, sh = s.sc_ecl(t)
        f, sh = s.finale(f, sh, t, DUR, None, (1, 0.82, 0.3), (0.5, 0.3, 0.9), -1)
        thr = o.get("thr", 0.95 - 0.15 * s._ecl)
        return s.film(f, t, sh * s.k, 0.5, thr, (1.0, 0.95, 0.85), 1.0 + 0.15 * sh, 0.04, o.get("vig", 0.4 + 0.25 * s._ecl), 1.10, 1.0 + 0.12 * s._ecl)

    # 1) まひる
    def sc_noon(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_OPEN
        f = s.noon(t, 1.0); s.eclipse(f, t, 0.0, (W * 3, -H), 1.0); f = s.godrays(f, s.SC, 0.25, 1.2, 8, 0.86, 0.5)
        cy = s.gnd - H * 0.215; s.gshadow(f, W / 2, s.gnd, 0.44, 0.22 + 0.16 * ease(t / 2.0), 0.6)
        img = s.pose(t); s.draw(f, img, W / 2, cy, 0.44)
        if u > 0:                                                                    # 金の 目が ひらく
            s.gold_eye(f, clamp(1.6 - u), 1.0)
            if u < 0.1: f = f * (0.4 + 0.6 * u / 0.1)
            p = s.at(*s.eyeG); L = s.L0()
            for i in range(2):
                v = (u - i * 0.1) / 0.6
                if 0 < v < 1: s.circ(L, p, W * 1.1 * v, 1 - v, 3)
            f *= (1 - np.clip(cv2.GaussianBlur(L, (0, 0), 2 * k), 0, 1) * 0.5)[:, :, None]
        return f, (5 * math.exp(-u * 8) if u > 0 else 0)

    # 2) 二つ割り: 明るい地に 黒い目 / 闇に 金の目
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT; img = s.pose(t); zf = 1.55; sc = zf * H / s.ah; cy = H * 0.36
        ex = (s.eyeB[0] + s.eyeG[0]) / 2; ey = (s.eyeB[1] + s.eyeG[1]) / 2; px = W * 0.5 - (ex - s.aw / 2) * sc; py = cy - (ey - s.ah / 2) * sc
        n = s.fbm(0.0014, t * 12, 0, 3, 40 * k, 0)
        A = (PEARL * 0.80)[None, None, :] * (0.85 + 0.25 * n)[:, :, None] + (np.exp(-np.hypot(s.XX - W * 0.15, s.YY - cy + H * 0.1) / (W * 0.4)) * 0.3)[:, :, None]
        B = INK[None, None, :] * (0.6 + 1.2 * n)[:, :, None] + s.nebula(t, [(0.05, 0.02, 0.12), (0.14, 0.06, 0.26), (0.9, 0.7, 0.3)], 0.0016, (10.0, 3.0), 0.6, 1)
        s._ecl = 0.0; s.draw(A, img, px, py, zf); s.place(B, s.rim, px, py, zf, 1.0, GOLD)                 # 右半分は 金の ふちどりだけ
        B += cv2.GaussianBlur(np.clip(B - 0.5, 0, 1), (0, 0), 6 * k) * 0.8
        s._pl = (px, py, sc); s.gold_eye(B, 0.8 + 0.2 * math.sin(u * 10), 0.6)
        split = W * 0.5 + (1 - ease(u / 0.25)) * W * 0.6; m = sstep(split + 1, split - 1, s.XX)[:, :, None]; f = A * m + B * (1 - m)
        bh = H * 0.16 * ease(u / 0.3); band = sstep(bh + 1, bh - 1, np.abs(s.YY - cy))[:, :, None]
        bg = INK[None, None, :] * (0.4 + 0.9 * n)[:, :, None]; f = f * band + bg * (1 - band)
        s.glowline(f, [(split, cy - bh), (split, cy + bh)], GOLD, 1.5, 0.9, 0.8)
        for yy in (cy - bh, cy + bh): s.glowline(f, [(0, yy), (W, yy)], GOLD, 1.5, 0.7, 0.8)
        n_ = len(s.vt); step = W * 0.17
        for i, T in enumerate(s.vt):                                                 # 技名: 影が 先に のびて、字が あとから 追いつく
            a = (u - 0.55 - i * 0.13) / 0.3
            if a <= 0: continue
            x = W / 2 + (i - (n_ - 1) / 2) * step; y = H * 0.70
            L2 = s.L0(); s.line(L2, (x, y + 38 * k), (x + 70 * k * ease(a), y + 38 * k + 26 * k * ease(a)), 0.5, int(8 * k)); f += cv2.GaussianBlur(L2, (0, 0), 3 * k)[:, :, None] * (GOLD * 0.35)[None, None, :]
            if a > 0.4: s.text(f, T, x, y, clamp((a - 0.4) * 3), 1.0)
        for i, T in enumerate(s.vo):
            if u > 0.45 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.585, 1.0, 1.0)
        if u > 1.4: s.text(f, s.txt_en, W / 2, H * 0.70 + 84 * k, ease((u - 1.4) / 0.3), 1.0)
        if u < 0.08: f = f * (u / 0.08) + (1 - u / 0.08)
        return f, 0

    # 3) 日を くう → 影が こちらへ のびて 立つ → 闇に 金の 目 → 日食の 輪の 一点が 光る
    def sc_ecl(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_RISE; kg = t - T_KING; dk = t - T_DARK; dr = t - T_RING
        cov = ease2(clamp(u / (T_TOT - T_RISE))); ecl = float(sstep(0.55, 1.0, cov)); s._ecl = ecl; cy = s.gnd - H * 0.215
        f = s.noon(t, 1.0, ecl)
        img = s.pose(t, lean=-6 * k * ease(u / 0.4)); s.draw(np.zeros_like(f), img, W / 2, cy, 0.44); o = s.at(*s.orb)
        mc = (o[0] + (s.SC[0] - o[0]) * cov, o[1] + (s.SC[1] - o[1]) * cov); s.eclipse(f, t, cov, mc, (26 * k) + (s.SR * 1.03 - 26 * k) * cov)
        if cov < 0.95: f = s.godrays(f, s.SC, 0.25 * (1 - cov), 1.2, 8, 0.86, 0.5)
        s.gshadow(f, W / 2, s.gnd, 0.44, 0.38 + 1.5 * cov + (1.2 * clamp(kg / 0.6) if kg > 0 else 0), 0.6 + 0.35 * cov)
        s.draw(f, img, W / 2, cy, 0.44); s.gold_eye(f, 0.6 + 0.4 * ecl)
        sh = 6 * math.exp(-abs(t - T_TOT) * 6)
        if abs(t - T_TOT) < 0.06: f = f + 0.25
        if kg > 0:
            g = ease2(clamp(kg / (T_DARK - T_KING))); f *= 1 - 0.4 * g; pts, sc = s.sking(f, t, g, 0)
            if dk > 0: f *= 1 - ease(clamp(dk / 0.3)) * 0.97                           # 視界は 闇
            s.keyes(f, pts, sc, clamp((kg - 1.25) / 0.35) * (1 + 0.15 * math.sin(t * 9)), t); sh = max(sh, 2 + 5 * g)
            if dr > -0.45:                                                           # 頭の 上の 日食の 輪(王冠)と、その 一点の 光
                C = ((pts[0][0] + pts[1][0]) / 2, min(pts[0][1], pts[1][1]) - H * 0.20); R = W * 0.40; a_ = ease(clamp((dr + 0.45) / 0.45))
                ring = [(C[0] + math.cos(i * 0.1309) * R, C[1] + math.sin(i * 0.1309) * R * 0.30) for i in range(48)]; s.glowline(f, ring, PEARL * 0.7 + GOLD * 0.3, 1.5, 0.8 * a_, 1.2, True)
                dd = np.abs(np.hypot((s.XX - C[0]) / R, (s.YY - C[1]) / (R * 0.30)) - 1); f += (np.exp(-dd / 0.22) * sstep(C[1] + R * 0.5, C[1] - R * 0.2, s.YY) * 0.30 * a_)[:, :, None] * (PEARL * 0.6 + GOLD * 0.4)[None, None, :]
                if dr > 0:
                    e = clamp(dr / (T_HIT - T_RING)); p = (C[0] + R * 0.72, C[1] - R * 0.30 * 0.69); s.flare(f, p, (10 + 70 * e) * k, PEARL, 0.8 + 1.0 * e, 1.2, 0.0, False); f += (np.exp(-np.hypot(s.XX - p[0], s.YY - p[1]) / ((12 + 420 * e ** 1.8) * k)) * (0.5 + 2.2 * e))[:, :, None] * PEARL[None, None, :]
                    f = s.godrays(f, p, 0.5 * e, 0.7, 10, 0.88, 0.6); sh = max(sh, 3 + 9 * e)
                    if e > 0.75: f = f + ((e - 0.75) / 0.25) ** 2 * 0.9
        if u < 0.12: f = f * (u / 0.12) + (1 - u / 0.12)
        return f, sh

    # ---- この子だけの曲: まひるは 明るい 長調の たてごと。目が ひらくと 同じ 旋律が 短調に うら返る。日が かげり、影の王は 低い 金管 → 闇に 目 → 一点の 光で 長調へ ----
    def audio(s, path):
        a = Snd(DUR, 77); C4 = 261.63; maj = [0, 2, 4, 5, 7, 9, 11]; mn = [0, 1, 3, 5, 7, 8, 10]
        def fq(d, o=0, sc=maj): return C4 * 2 ** ((sc[d % 7] + 12 * (d // 7 + o)) / 12.0)
        tune = [4, 2, 0, 2, 4, 7, 4, 2]
        for i, d in enumerate(tune): a.note(fq(d), 0.2 + i * 0.21, 0.5, 0.09, "bell")                               # まひる
        for d in (0, 4): a.note(fq(d, -1), 0.2, 1.6, 0.05, "pad", 0.2)
        a.a += a.lp(a.nz, 0.02) * 0.25 * a.gate(0, T_OPEN, 0.3)
        a.boom(T_OPEN, 50, 0.9, 0.5); a.note(fq(7, 1, mn), T_OPEN, 0.9, 0.12, "bell")
        a.boom(T_CUT, 46, 1.0)
        for i in range(5): a.note(fq(tune[i], 0, mn), T_CUT + 0.55 + i * 0.13, 0.4, 0.11, "bell"); a.note(fq(tune[i], -1, mn), T_CUT + 0.55 + i * 0.13, 0.4, 0.06, "saw", 0.01)
        a.rise(T_RISE, T_TOT, 400, 90, 0.16); a.a += a.tone(55) * 0.12 * a.gate(T_RISE, T_DARK, 0.4)                 # 日が かげる
        for i, d in enumerate(tune[:7]): a.note(fq(d, 0, mn) * (1 - 0.01 * i), T_RISE + i * 0.19, 0.5, 0.07 * (1 - i / 10.0), "bell")
        a.boom(T_TOT, 36, 1.4, 0.8); a.a += (a.nz - a.lp(a.nz, 0.5)) * a.env(T_TOT, 0.001, 0.05) * 0.4; a.note(fq(14, 2), T_TOT, 0.8, 0.05, "sine", 0.05)   # 皆既
        for j, (d, o) in enumerate(((0, -2), (4, -2), (0, -1), (2, -1), (4, -1))): a.note(fq(d, o, mn), T_KING + j * 0.32, T_DARK - T_KING - j * 0.32 + 0.3, 0.10, "saw", 0.15)   # 影の王が 立つ
        a.rise(T_KING, T_DARK, 50, 120, 0.20); a.a += a.lp(a.nz, 0.04) * np.clip((a.t - T_KING) / (T_DARK - T_KING), 0, 1) ** 2 * (a.t < T_DARK) * 0.8
        tk = T_KING + 0.2; i = 0
        while tk < T_DARK - 0.05: a.boom(tk, 46, 0.35 + 0.5 * (tk - T_KING) / (T_DARK - T_KING), 0.15); tk += max(0.12, 0.34 - i * 0.03); i += 1
        a.note(fq(7, 1, mn), T_KING + 1.25, 0.8, 0.10, "bell"); a.note(fq(8, 1, mn), T_KING + 1.25, 0.8, 0.06, "bell")   # 金の 目
        a.boom(T_DARK, 34, 1.5, 0.7); a.a *= 1 - 0.8 * np.clip((a.t - T_DARK - 0.1) / 0.2, 0, 1) * (a.t < T_RING - 0.02)   # 闇: しずまる
        a.note(fq(14, 2), T_RING, T_HIT - T_RING + 0.2, 0.16, "sine", 0.02); a.rise(T_RING, T_HIT, 600, 2400, 0.10); a.whoosh(T_RING, T_HIT, 0.8, 0.3)   # 一点の 光
        a.boom(T_HIT, 44, 1.5, 0.6)
        for j, (d, o) in enumerate(((0, -1), (4, -1), (0, 0), (2, 0), (4, 0))): a.note(fq(d, o), T_HIT + j * 0.03, 0.6, 0.05, "pad", 0.1)
        for i, d in enumerate([4, 7, 9, 11]): a.note(fq(d), T_HIT + 0.02 + i * 0.08, 0.5, 0.08, "bell")
        s.fin_snd(a, DUR, 77); a.save(path)


if __name__ == "__main__":
    r = Eclipse()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
