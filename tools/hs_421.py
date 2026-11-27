"""キャノ(★4 砲手の少女) 専用の必殺技「大砲三連発」— 高品質版(hs_fx)。1発目は 左を、2発目は 右を かすめ、3発目は 砲口が こちらを 向いて 大写しに なり、奥の 火が ふき出す
夕ぐれの 甲板に 立つ(火の粉)→ 砲口の 二重丸の 窓の カットイン+技名 → 1発(左を かすめる)→ 2発(右を かすめる)→
砲口が こちらへ 向いて 大きく なる(らせんの みぞ・奥で 火が ともる)→ 火が ふき出す → 白 → 暗転
使い方: python3 hs_421.py 出力.mp4 / python3 hs_421.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.3
T_CUT, T_ACT, T1, T2, T3, T_FIRE, T_HIT = 2.7, 4.7, 5.0, 5.75, 6.5, 8.55, 9.4
SUN = np.array([1.0, 0.60, 0.26], np.float32); TEAL = np.array([0.08, 0.22, 0.34], np.float32); SEA = np.array([0.06, 0.16, 0.26], np.float32); WOOD = np.array([0.42, 0.25, 0.14], np.float32); IRON = np.array([0.17, 0.17, 0.20], np.float32); BRASS = np.array([0.95, 0.70, 0.28], np.float32); HOT = np.array([1.0, 0.88, 0.55], np.float32); SMK = np.array([0.90, 0.86, 0.82], np.float32)
EYE = (320, 235); MUZ = (525, 355)
PASS = [(T1, (-0.25, 0.30)), (T2, (1.25, 0.78))]


class Cano(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 421, (1.0, 0.7, 0.35), "大砲三連発", "Triple Cannon", W); s.fx_init(421)
        s._vt = [s.text_img(c, hz.SERIF, int(70 * s.k), 3) for c in "大砲三連発"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; s.hy = 0.60

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 夕ぐれの 甲板(空・海・手すり・板) ----
    def deck(s, t, lum=1.0):
        W, H, k = s.W, s.H, s.k; hy = s.hy; v = (s.yn / hy).clip(0, 1)
        f = TEAL[None, None, :] * (1 - v ** 1.3)[:, :, None] + (SUN * 1.05)[None, None, :] * (v ** 2.6)[:, :, None] + np.array([0.45, 0.25, 0.35], np.float32)[None, None, :] * (v * (1 - v) * 1.6)[:, :, None]
        n = s.fbm(0.0009, t * 5, 0, 3, 80 * k, 1); n = cv2.GaussianBlur(n, (0, 0), 4 * k); cu = sstep(0.50, 0.68, n + 0.12 * np.exp(-((s.yn - 0.30) / 0.14) ** 2))
        lit = np.clip(0.5 + 1.5 * (s.fbm(0.003, t * 8, 3, 3, 0, 2) - 0.4), 0.2, 1.3)
        f = f * (1 - cu * 0.8)[:, :, None] + (cu * 0.8)[:, :, None] * (np.array([0.30, 0.20, 0.32], np.float32)[None, None, :] + (np.array([1.0, 0.62, 0.42], np.float32) * 0.9)[None, None, :] * (lit * (0.3 + 0.7 * v))[:, :, None])
        f += (np.exp(-np.hypot(s.xn - 0.30, (s.yn - hy) * 1.5) / 0.15) * 0.8)[:, :, None] * np.array([1.0, 0.85, 0.55], np.float32)[None, None, :]
        d0 = np.clip((s.yn - hy) / 0.10, 0, 1); wv = s.tex(2, s.XX * 0.5 / (d0 + 0.2) + t * 12, 200.0 / (d0 + 0.2) + t * 10)                                    # 海(手すりの 向こう)
        sw = SEA[None, None, :] * (0.8 + 0.6 * wv)[:, :, None] + (np.exp(-np.abs(s.xn - 0.30) / 0.10) * sstep(0.45, 0.8, wv) * 0.9)[:, :, None] * SUN[None, None, :]
        m = sstep(hy - 0.002, hy + 0.003, s.yn)[:, :, None]; f = f * (1 - m) + sw * m
        fy0 = 0.70; d = np.clip((s.yn - fy0) / (1 - fy0), 0, 1); pu = (s.XX - W / 2) / (d * 0.9 + 0.35)                                                          # 板(奥へ のびる)
        pi = pu / (52 * k); ln = np.exp(-((np.abs((pi % 1.0) - 0.5) - 0.5) / 0.035) ** 2)
        wc = WOOD[None, None, :] * ((0.55 + 0.8 * s.tex(1, np.floor(pi) * 53 + pu * 0.05, 80.0 / (d + 0.15))) * (1 - 0.65 * ln) * (0.55 + 0.6 * d))[:, :, None]
        wc += (np.exp(-np.abs(s.xn - 0.32) / 0.2) * (1 - d) * 0.22)[:, :, None] * SUN[None, None, :]
        m = sstep(fy0 - 0.002, fy0 + 0.003, s.yn)[:, :, None]; f = f * (1 - m) + wc * m
        rl = sstep(0.0, 0.003, s.yn - 0.615) * sstep(0.003, 0.0, s.yn - 0.640); px = ((s.xn + 0.07) % 0.143) / 0.143; po = sstep(0.0, 0.003, s.yn - 0.635) * sstep(0.003, 0.0, s.yn - 0.705) * sstep(0.10, 0.08, np.abs(px - 0.5))     # 手すり
        rm = np.maximum(rl, po); rc = (WOOD * 0.55)[None, None, :] * (0.6 + 0.7 * s.tex(1, s.XX * 0.9, s.YY * 0.3))[:, :, None] + (SUN * 0.35)[None, None, :] * (rl * sstep(0.004, 0.0, s.yn - 0.619))[:, :, None]
        f = f * (1 - rm)[:, :, None] + rc * rm[:, :, None]
        return (f * lum).astype(np.float32)

    def embr(s, f, t, v=1.0):
        W, H = s.W, s.H; r = s.rnd[:90]
        x = ((r[:, 0] + 0.03 * np.sin(t * (1 + r[:, 2]) + r[:, 3] * 6.28) + t * 0.02) % 1.0) * W; y = ((r[:, 1] - t * (0.04 + 0.07 * r[:, 4])) % 1.0) * H
        s.splat(f, x, y, (0.3 + 0.6 * r[:, 2]) * v * (0.6 + 0.4 * np.sin(t * 8 + r[:, 5] * 20)) * 1.3, SUN, 1.1, 0.3)

    def puff(s, f, c, R, age, v=1.0):
        """砲煙(もくもく した かたまり)"""
        if v <= 0.01: return
        d = np.hypot(s.XX - c[0], (s.YY - c[1]) * 1.15) / R; n = s.tex(2, s.XX * 0.9 + c[0], s.YY * 0.9 - age * 60) * 0.6 + s.tex(0, s.XX * 2.0, s.YY * 2.0 - age * 90) * 0.4
        m = sstep(1.0, 0.35, d + (n - 0.5) * 0.7) * v
        f[:] = f * (1 - m * 0.85)[:, :, None] + (m * 0.85 * (0.62 + 0.5 * n))[:, :, None] * SMK[None, None, :] + (m * np.exp(-age * 9) * 1.2)[:, :, None] * SUN[None, None, :]

    def shot(s, f, c, R, v=1.0):
        """鉄の 弾(球の かげ・夕日の ふち)"""
        if v <= 0.01 or R < 2: return
        dx = (s.XX - c[0]) / R; dy = (s.YY - c[1]) / R; d = np.hypot(dx, dy); z = np.sqrt(np.clip(1 - d * d, 0, 1)); m = sstep(1.0, 0.95, d) * v
        lam = np.clip(0.25 + 0.8 * (-dx * 0.6 + dy * 0.3) + 0.5 * z, 0, 1.5)
        colr = IRON[None, None, :] * (lam * (0.5 + 0.9 * s.tex(1, dx * 80, dy * 80)))[:, :, None] + (SUN * 0.9)[None, None, :] * (np.clip(-dx * 0.8 + 0.2, 0, 1) ** 2 * (1 - z) ** 1.3)[:, :, None] + (np.exp(-(np.hypot(dx + 0.4, dy + 0.4) / 0.2) ** 2) * 0.35)[:, :, None]
        f[:] = f * (1 - m)[:, :, None] + colr * m[:, :, None]

    # ---- 砲口(式): 鉄の 輪・真ちゅうの たが・らせんの みぞの 穴・奥の 火 ----
    def muzzle(s, f, c, R, glow, t, v=1.0):
        dx = (s.XX - c[0]) / R; dy = (s.YY - c[1]) / R; d = np.hypot(dx, dy); th = np.arctan2(dy, dx)
        sheen = 0.35 + 0.9 * np.clip(np.cos(th + 2.2), 0, 1) ** 3 + 0.5 * np.clip(np.cos(th - 0.9), 0, 1) ** 8
        tx = s.tex(1, th / 6.2832 * 512, d * 60)
        barrel = IRON[None, None, :] * (sheen * (0.55 + 0.8 * tx) * (0.7 + 0.5 * np.clip((d - 0.74) / 0.26, 0, 1)))[:, :, None] + (SUN * 0.4)[None, None, :] * (np.clip(np.cos(th + 2.2), 0, 1) ** 6 * 0.6)[:, :, None]
        band = sstep(1.0, 1.02, d) * sstep(1.20, 1.18, d); bc = BRASS[None, None, :] * ((0.35 + 1.0 * np.clip(np.cos(th + 2.2), 0, 1) ** 2 + 0.2 * tx) * (1 - 0.5 * np.abs((d - 1.10) / 0.10) ** 2))[:, :, None]
        outer = sstep(1.20, 1.22, d) * sstep(1.75, 1.72, d); oc = IRON[None, None, :] * (sheen * 0.6 * (0.5 + 0.8 * tx) * np.clip(1.9 - d, 0, 1))[:, :, None]                # うしろへ つづく 砲身
        lg = np.log(d + 0.05); rif = 0.5 + 0.5 * np.sin(th * 8 + lg * 9.0 + t * 0.6)                                           # らせんの みぞ
        bore = (IRON * 0.5)[None, None, :] * ((0.25 + 0.6 * rif) * np.clip(d / 0.74, 0, 1) ** 1.6)[:, :, None]
        heat = np.exp(-(d / (0.20 + 0.45 * glow)) ** 2) * glow; flick = 0.75 + 0.5 * s.tex(2, th / 6.2832 * 256 + t * 90, lg * 50 - t * 200)
        bore = bore + (SUN * 1.3)[None, None, :] * (heat * flick)[:, :, None] + HOT[None, None, :] * (np.exp(-(d / (0.10 + 0.22 * glow)) ** 2) * glow * 1.2)[:, :, None] + (SUN * 0.5)[None, None, :] * (rif * glow * np.clip(d / 0.74, 0, 1) * 0.5)[:, :, None]
        mb = sstep(0.74, 0.72, d); mr = sstep(1.0, 0.99, d) * (1 - mb)
        lip = np.exp(-((d - 0.75) / 0.02) ** 2) * (0.5 + 0.8 * glow)
        out = barrel * mr[:, :, None] + bore * mb[:, :, None] + bc * band[:, :, None] + oc * outer[:, :, None]
        A = np.clip(mr + mb + band + outer, 0, 1) * v
        f[:] = f * (1 - A)[:, :, None] + out * v + (lip * v)[:, :, None] * (BRASS * 0.5 + SUN * 0.5)[None, None, :] * (d < 1.0)[:, :, None]

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        s.draw_lit(f, img, cx, cy, hfrac, rim=(1.0, 0.7, 0.4), rim_dir=(-0.8, -0.3), rim_v=0.55 * alpha, amb=(1.0, 0.85, 0.75), amb_v=0.12, aura=SUN, aura_v=0.06, alpha=alpha, shade=0.95)

    def frame(s, t):
        o = {}
        if t < T_CUT: f, sh = s.sc_deck(t)
        elif t < T_ACT: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.5, thr=0.95)
        else: f, sh = s.sc_fire(t)
        f, sh = s.finale(f, sh, t, DUR, None, (1.0, 0.92, 0.8), (1.0, 0.6, 0.3), -1)
        return s.film(f, t, sh * s.k, 0.45, o.get("thr", 0.92), (1.0, 0.95, 0.9), 1.0 + 0.15 * min(sh, 8), 0.03, o.get("vig", 0.45), 1.2, 0.96)

    def girl(s, f, t, alpha=1.0, kick=0.0, **kw):
        W, H, k = s.W, s.H, s.k; cx = W * 0.47 - kick * 16 * k; cy = H * 0.60; fy = cy + 0.44 * H * 0.47
        sd = np.exp(-(((s.XX - cx - W * 0.06) / (W * 0.20)) ** 2 + ((s.YY - fy) / (H * 0.014)) ** 2)); f *= (1 - 0.55 * alpha * sd)[:, :, None]
        img = s.pose(t, **kw); s.draw(f, img, cx, cy, 0.44, alpha); return s.at(*s.A(*MUZ))

    # 1) 甲板に 立つ
    def sc_deck(s, t):
        f = s.deck(t); s.girl(f, t, clamp(t / 0.6), sway=0.3, breath=0.6); s.embr(f, t, 0.7)
        return f * clamp(t / 0.5), 0

    # 2) 目の カットイン(砲口の ような 二重丸の 窓)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.deck(t, 0.34)
        a = back(clamp(u / 0.35)); cy = H * 0.32; R = H * 0.15 * a
        bandc = (IRON * 0.9)[None, None, :] + (SUN * 0.7)[None, None, :] * (0.25 + 0.9 * sstep(0.45, 0.9, s.tex(0, s.XX * 0.4 - t * 60, s.YY * 0.4 + t * 20)))[:, :, None]
        ex, ey = s.A(*EYE); zf = 1.45; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(bandc, img, W / 2 - (ex - s.aw / 2) * sc, cy - (ey - s.ah / 2) * sc, zf)
        if R > 6:
            c = (W / 2, cy); d = np.hypot(s.XX - c[0], s.YY - c[1]); th = np.arctan2(s.YY - c[1], s.XX - c[0])
            m = sstep(R, R - 2, d)[:, :, None]; f = f * (1 - m) + bandc * m
            for r0, r1, colr in ((R, R + 13 * k, IRON * 1.4), (R + 13 * k, R + 20 * k, BRASS), (R + 20 * k, R + 26 * k, IRON * 0.9)):
                rm = sstep(r0 - 1, r0 + 1, d) * sstep(r1 + 1, r1 - 1, d); shn = 0.4 + 0.9 * np.clip(np.cos(th + 2.2), 0, 1) ** 3 + 0.3 * s.tex(1, th * 80, d * 2)
                f = f * (1 - rm)[:, :, None] + colr[None, None, :] * (shn * rm)[:, :, None]
        n = len(s._vt); step = W * 0.172
        for i, T in enumerate(s._vt):                                                # 技名: どん と 1文字ずつ(大きい → ちぢむ)
            p = (u - 0.5 - i * 0.13) / 0.18
            if p > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.69, clamp(p * 2), 1.0 + 0.9 * (1 - min(1.0, p)) ** 2)
        for i, T in enumerate(s._vo):
            if u > 0.35 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.69 - 74 * k, 1.0, 1.0)
        if u > 1.4:
            g = ease((u - 1.4) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.69 + 56 * k), (W / 2 + W * 0.42 * g, H * 0.69 + 56 * k)], HOT, 1.0, 0.7, 0.7)
            s.text(f, s.txt_en, W / 2, H * 0.69 + 82 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.5
        sh = 3 * math.exp(-u * 6)
        for i in range(n):
            q = u - 0.5 - i * 0.13
            if 0 < q < 0.15: sh += 3 * (1 - q / 0.15)
        return f, sh

    # 3) 1発・2発が かすめる → 砲口が こちらを 向く → 火
    def sc_fire(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_ACT; sh = 0.0
        e = clamp((t - T3) / (T_FIRE - T3)); ef = clamp((t - T_FIRE) / (T_HIT - T_FIRE))
        f = s.deck(t, 1.0 - 0.45 * e)
        kick = sum(math.exp(-(t - t0) * 7) for t0, _ in PASS if t >= t0)
        mz = s.girl(f, t, 1.0, kick, sway=0.2, breath=0.4, squash=-0.03 * kick)
        s.embr(f, t, 0.7)
        for j, (t0, to) in enumerate(PASS):
            q = t - t0
            if q < 0: continue
            s.puff(f, (mz[0] + 26 * k + q * 30 * k, mz[1] - 6 * k - q * 40 * k), k * (26 + 120 * min(q, 1.2) ** 0.6), q, clamp(1.6 - q * 0.9))
            if q < 0.10: s.flare(f, mz[0] + 10 * k, mz[1], HOT, 1.6 * (1 - q / 0.10), 60 * k) if hasattr(s, "flare_ok") else s.glowmask(f, np.exp(-(np.hypot(s.XX - mz[0] - 10 * k, s.YY - mz[1]) / (46 * k)) ** 2), HOT, 2.2 * (1 - q / 0.10), 1.0, 0.8, 9.0); sh += 10 * (1 - q / 0.10)
            p = q / 0.55
            if p < 1.25:                                                             # 弾が こちらへ 来て わきを かすめる
                g = p ** 1.6; c = (mz[0] + (to[0] * W - mz[0]) * g, mz[1] + (to[1] * H - mz[1]) * g - H * 0.10 * math.sin(min(p, 1.0) * 3.14)); R = k * (7 + 240 * p ** 2.6)
                L = s.L0(); cv2.line(L, (int(mz[0]), int(mz[1])), (int(c[0]), int(c[1])), float(0.5 * (1 - min(1.0, p))), max(1, int(R * 0.5)), cv2.LINE_AA)
                f[:] = f * (1 - cv2.GaussianBlur(L, (0, 0), 4 * k) * 0.0)[:, :, None] + (cv2.GaussianBlur(L, (0, 0), 6 * k) * 0.5)[:, :, None] * SMK[None, None, :]
                s.shot(f, c, R); sh += 9 * p ** 3 * (p < 1.1)
            if 0.50 < q < 0.75: f[:] = f * (1 - 0.5 * (1 - abs((q - 0.62) / 0.13)))                                           # かすめた 影
        if e > 0:
            ee = e ** 2.2; g = ease2(clamp(e * 1.8)); c = (mz[0] + (W * 0.5 - mz[0]) * g, mz[1] + (H * 0.45 - mz[1]) * g)
            R = k * 30 * (1 - g) + k * 34 * g + W * 0.72 * ee; glow = clamp((e - 0.35) / 0.65) ** 1.5 * 0.55 + ef
            s.muzzle(f, c, R, min(1.6, glow), t, min(1.0, e * 7)); sh += 1 + 3 * e + 6 * glow * (0.5 + 0.5 * math.sin(t * 50)) * 0.3
            if e > 0.3:                                                              # 火なわの 火花(ふちで はねる)
                n = 60; r = s.rnd[240:240 + n]; a = r[:, 0] * 6.283 + t * 2; qq = (r[:, 1] + t * (1.5 + r[:, 2])) % 1.0; rr = R * (0.2 + 0.7 * qq)
                s.splat(f, c[0] + np.cos(a) * rr, c[1] + np.sin(a) * rr, (1 - qq) * 1.5 * glow, HOT, 1.3, 0.4)
            if ef > 0:                                                               # 火が ふき出す
                dxp = s.XX - c[0]; dyp = s.YY - c[1]; dist = np.hypot(dxp, dyp) + 1e-3; th = np.arctan2(dyp, dxp); lg = np.log(dist / k + 5.0)
                tur = s.tex(2, th / 6.2832 * 512 + lg * 30, lg * 46 - t * 420) * 0.6 + s.tex(0, th / 6.2832 * 768, lg * 80 - t * 700) * 0.4
                Rf = k * 30 + W * 1.5 * ef ** 1.3; heat = np.clip(1.25 - dist / Rf + (tur - 0.5) * 0.9, 0, 1.3) * min(1.0, ef * 8)
                fire = np.array([0.55, 0.06, 0.0], np.float32)[None, None, :] * sstep(0.0, 0.3, heat)[:, :, None] + np.array([0.45, 0.38, 0.05], np.float32)[None, None, :] * sstep(0.25, 0.6, heat)[:, :, None] + np.array([0.0, 0.42, 0.30], np.float32)[None, None, :] * sstep(0.55, 0.9, heat)[:, :, None] + np.array([0.0, 0.14, 0.6], np.float32)[None, None, :] * sstep(0.85, 1.2, heat)[:, :, None]
                am = sstep(0.0, 0.25, heat)[:, :, None]; f[:] = f * (1 - am) + fire * 1.15
                sh += 10 * (1 - ef * 0.5)
                if ef < 0.1: f[:] = f + (1 - ef / 0.1) * 0.7
                if ef > 0.72: f[:] = f + ((ef - 0.72) / 0.28) ** 2 * 1.6
        if u < 0.12: f = f + (1 - u / 0.12) * 0.25
        return f, sh

    # ---- この子だけの曲: 小太鼓の 行進 → どん(1)・どん(2)→ 火なわの じりじり+上がる 音 → 大きな どん ----
    def audio(s, path):
        a = Snd(DUR, 421); G = 146.8; pn = [0, 2, 4, 7, 9]
        def fq(d, o=0): return G * 2 ** ((pn[d % 5] + 12 * (d // 5 + o)) / 12.0)
        for d in (0, 2, 3): a.note(fq(d, -1), 0.1, 2.6, 0.04, "pad", 0.6)
        snr = a.nz - a.lp(a.nz, 0.5)
        for i in range(14):
            tk = 0.4 + i * 0.16
            if i % 4 != 3: a.a += snr * a.env(tk, 0.002, 0.05) * (0.35 if i % 2 == 0 else 0.2)
        for i, d in enumerate((5, 5, 7, 9, 7, 5)): a.note(fq(d, 0), 0.5 + i * 0.33, 0.3, 0.07, "fm")
        a.boom(T_CUT, 58, 0.6, 0.4); a.whoosh(T_CUT - 0.3, T_CUT, 0.2, 0.1)
        for i in range(5): a.boom(T_CUT + 0.5 + i * 0.13, 85 - i * 5, 0.2, 0.3); a.note(fq(5 + i, 0), T_CUT + 0.5 + i * 0.13, 0.5, 0.06, "bell")
        for d in (0, 2, 4): a.note(fq(d, 0), T_CUT, 1.9, 0.04, "pad", 0.05)
        for t0, _ in PASS:
            a.boom(t0, 46, 1.0, 0.9); a.a += (a.lp(a.nz, 0.3) - a.lp(a.nz, 0.05)) * a.env(t0, 0.003, 0.3) * 1.6 + snr * a.env(t0, 0.001, 0.08) * 0.6
            a.whoosh(t0 + 0.25, t0 + 0.65, 1.2, 0.5); a.rise(t0 + 0.15, t0 + 0.62, 500, 180, 0.08)
        siz = a.nz - a.lp(a.nz, 0.7); g = np.clip((a.t - T3) / (T_FIRE - T3), 0, 1) * (a.t < T_FIRE)
        a.a += siz * g * 0.35 * (0.6 + 0.4 * np.sin(a.t * 61.0) * np.sin(a.t * 17.0))                                         # 火なわ
        a.rise(T3, T_FIRE, 90, 640, 0.10)
        nb = 10
        for i in range(nb): tk = T3 + (T_FIRE - T3) * (1 - (1 - i / nb) ** 1.7); a.a += snr * a.env(tk, 0.002, 0.05) * (0.2 + 0.03 * i)
        for j, d in enumerate((0, 2, 4)): a.note(fq(d, 0), T3 + j * 0.06, T_FIRE - T3, 0.045, "pad", 0.6)
        a.boom(T_FIRE, 34, 2.0, 1.1); a.boom(T_FIRE + 0.03, 62, 0.7, 0.6); wn = a.lp(a.nz, 0.2) - a.lp(a.nz, 0.02)
        a.a += wn * a.env(T_FIRE, 0.004, 0.9) * 2.4 + snr * a.env(T_FIRE, 0.001, 0.2) * 0.7
        a.boom(T_HIT, 40, 1.4, 0.6)
        for j, d in enumerate((0, 2, 4, 7)): a.note(fq(d, 1), T_HIT + 0.02 + j * 0.03, 0.8, 0.05, "bell")
        s.fin_snd(a, DUR, 421); a.save(path)


if __name__ == "__main__":
    r = Cano()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
