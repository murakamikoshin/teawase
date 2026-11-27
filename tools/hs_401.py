"""アリエス(★4 牡羊座の盾騎士) 専用の必殺技「金羊ノ守リ」— 高品質版(hs_fx)。金の 羊毛を まとい、角の 紋の 盾ごと こちらへ 体当たり
夕やけの 高原(山なみ・草) → 金の 羊毛の わたが 盾に 集まる(前ふり) → 目の カットイン+技名 →
盾を かまえる: 金の 羊毛が うずを 巻いて 大きな 輪に なる → 土けむりを 上げて こちらへ 突進(盾が ぐんぐん 大きく)→
盾が 画面に ぶつかる: 金の ふち・びょう・うずまく 2本の 角の 紋。衝撃の 輪と 羊毛の わたが はじけ、盾に おしこまれる → 白 → 暗転
使い方: python3 hs_401.py 出力.mp4 / python3 hs_401.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.0
T_CUT, T_GUARD, T_CHARGE, T_BASH, T_HIT = 2.7, 4.7, 6.3, 7.55, 9.3
GOLD = np.array([1.0, 0.78, 0.28], np.float32); CREAM = np.array([1.0, 0.96, 0.86], np.float32); DUSK = np.array([0.20, 0.10, 0.26], np.float32); SUN = np.array([1.0, 0.62, 0.30], np.float32); BRZ = np.array([0.42, 0.26, 0.08], np.float32)
EYE = (460, 280); SHIELD = (645, 500); SR = 150.0


class Aries(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 401, (1.0, 0.8, 0.35), "金羊ノ守リ", "Golden Fleece Guard", W); s.fx_init(401)
        s._vt = [s.text_img(c, hz.SERIF, int(66 * s.k), 3) for c in "金羊ノ守リ"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; s.hy = 0.66
        rg = np.random.RandomState(4010); s._pf = rg.rand(120, 6).astype(np.float32)
        s._rdg = [(0.40 + 0.05 * i + 0.045 * s.tex(i % 4, s.XX * (0.25 + 0.1 * i), 11.0 * (i + 1)) + 0.02 * s.tex((i + 1) % 4, s.XX * 0.9, 5.0 * (i + 2))).astype(np.float32) for i in range(3)]

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 夕やけの 高原 ----
    def highland(s, t, lum=1.0):
        W, H, k = s.W, s.H, s.k; hy = s.hy; v = (s.yn / hy).clip(0, 1)
        f = DUSK[None, None, :] * (1.2 - 0.6 * v)[:, :, None] * (1 - v ** 1.6)[:, :, None] + (SUN * 1.15)[None, None, :] * (v ** 1.6)[:, :, None]
        f += (np.exp(-np.hypot(s.xn - 0.30, (s.yn - 0.50) * 1.3) / 0.16) * 1.0)[:, :, None] * np.array([1.0, 0.85, 0.55], np.float32)[None, None, :]           # しずむ 日
        cl = s.fbm(0.0016, t * 8, 0, 4, 50 * k, 1); f += (sstep(0.45, 0.8, cl) * np.exp(-((s.yn - 0.24) / 0.16) ** 2) * 0.45)[:, :, None] * (SUN * 0.9 + 0.1)[None, None, :]
        for i, r in enumerate(s._rdg):                                                # 山なみ(3かさね: 奥ほど かすむ)
            m = sstep(r - 0.003, r + 0.003, s.yn)[:, :, None]; hz_ = 0.65 - 0.25 * i
            col = (DUSK * (0.9 - 0.25 * i))[None, None, :] * (1 - hz_) + (SUN * 0.75)[None, None, :] * hz_ * (0.6 + 0.4 * np.exp(-((s.xn - 0.3) / 0.35) ** 2))[:, :, None]
            col = col * (0.85 + 0.3 * s.tex(2, s.XX * 0.8, s.YY * 0.8))[:, :, None]; f = f * (1 - m) + col * m
        d = np.clip((s.yn - hy) / (1 - hy), 0, 1); z = 1.0 / (d + 0.09); u = (s.xn - 0.5) * z * 240 + np.sin(t * 1.8 + s.xn * 5) * 3 * z
        b = s.tex(1, u, z * 5.0) * 0.6 + s.tex(3, u * 2.3, z * 10.0) * 0.4
        g = np.array([0.05, 0.05, 0.03], np.float32)[None, None, :] + np.array([0.30, 0.26, 0.08], np.float32)[None, None, :] * (b ** 1.5 * (0.4 + 0.9 * d))[:, :, None] + (SUN * 0.5)[None, None, :] * (sstep(0.7, 0.95, b) * (1 - d) * 0.8)[:, :, None]
        gm = sstep(hy - 0.004, hy + 0.004, s.yn + (b - 0.5) * 0.012)[:, :, None]
        return ((f * (1 - gm) + g * gm) * lum).astype(np.float32)

    # ---- 金の 羊毛(わた)----
    def wool(s, f, xs, ys, rs, v=1.0):
        W, H, k = s.W, s.H, s.k; L = s.L0()
        for x, y, r in zip(xs, ys, rs):
            if r > 0.6: cv2.circle(L, (int(x), int(y)), max(1, int(r)), 1.0, -1, cv2.LINE_AA)
        m = np.clip(cv2.GaussianBlur(L, (0, 0), 2.2 * k) * 1.25, 0, 1) * v; top = np.clip(m - cv2.warpAffine(m, np.float32([[1, 0, 2 * k], [0, 1, 4 * k]]), (W, H)), 0, 1)
        col = (GOLD * 0.75 + CREAM * 0.25)[None, None, :] * (0.62 + 0.3 * s.tex(2, s.XX * 1.6, s.YY * 1.6))[:, :, None] + CREAM[None, None, :] * (top * 0.9)[:, :, None]
        f *= (1 - m * 0.92)[:, :, None]; f += col * m[:, :, None]; f += (cv2.GaussianBlur(m, (0, 0), 9 * k) * 0.30)[:, :, None] * GOLD[None, None, :]

    # ---- 盾(式): 金の ふち・びょう・白い 面・うずまく 2本の 角の 紋 ----
    def shield(s, f, c, R, v=1.0, glow=0.0):
        if v <= 0.01: return
        x = (s.XX - c[0]) / R; y = (s.YY - c[1]) / R; r = np.hypot(x, y) + 1e-4; th = np.arctan2(y, x)
        ins = sstep(1.0, 1.0 - max(0.004, 2.0 / R), r) * v; rimm = sstep(0.845, 0.86, r)
        spec = 0.5 + 0.5 * np.cos(th + 2.2); br = s.tex(1, th / 6.2832 * 1024, r * 14)
        rim = (BRZ[None, None, :] * 0.6 + GOLD[None, None, :] * (0.35 + 0.9 * spec ** 2 + 0.25 * br)[:, :, None]) * (0.55 + 0.75 * np.exp(-((r - 0.93) / 0.045) ** 2))[:, :, None] + CREAM[None, None, :] * (np.exp(-((r - 0.875) / 0.010) ** 2) * 0.6 + np.exp(-((r - 0.985) / 0.008) ** 2) * 0.4)[:, :, None]
        ar = ((th / 6.2832 * 16 + 0.5) % 1.0 - 0.5) * 6.2832 / 16 * 0.93; rv = np.exp(-(((r - 0.93) / 0.022) ** 2 + (ar / 0.022) ** 2)); rv2 = np.exp(-(((r - 0.925) / 0.010) ** 2 + ((ar + 0.006) / 0.010) ** 2))
        rim = rim * (1 - 0.6 * rv)[:, :, None] + CREAM[None, None, :] * (rv2 * 0.9)[:, :, None]                                                      # びょう
        face = CREAM[None, None, :] * (0.70 + 0.16 * br + 0.22 * np.exp(-(np.hypot(x + 0.3, y + 0.35) / 0.5) ** 2) - 0.22 * r ** 2)[:, :, None] * np.array([1.0, 0.97, 0.9], np.float32)[None, None, :]
        emb = np.zeros_like(r)
        for sg in (-1, 1):                                                           # 角: 対数らせんの 帯
            hx = x - sg * 0.30; hy_ = y + 0.06; rho = np.hypot(hx, hy_) + 1e-4; phi = np.arctan2(hy_, hx * sg)
            band = np.abs(((phi + 3.3 * np.log(rho / 0.045) + 1.2) % 6.2832) - 3.1416); wdt = 1.25
            emb = np.maximum(emb, sstep(wdt, wdt - 0.35, band) * sstep(0.335, 0.32, rho) * sstep(0.035, 0.05, rho))
        dia = sstep(1.0, 0.92, np.abs(x) / 0.085 + np.abs(y - 0.30) / 0.26); emb = np.maximum(emb, dia)                                                # まん中の ひし形
        rid = 0.75 + 0.35 * np.cos(np.hypot(x * 1.0, y) * 60.0)                                                                                       # 角の きざみ
        ecol = (GOLD * 0.95)[None, None, :] * (0.55 + 0.5 * spec + 0.2 * rid * (1 - dia))[:, :, None] + CREAM[None, None, :] * glow * 1.2
        face = face * (1 - emb)[:, :, None] + ecol * emb[:, :, None]; face *= (1 - 0.35 * np.exp(-((r - 0.845) / 0.03) ** 2))[:, :, None]
        col = face * (1 - rimm)[:, :, None] + rim * rimm[:, :, None]
        f *= (1 - ins)[:, :, None]; f += col * ins[:, :, None]
        if glow > 0: f += (emb * ins * glow * 0.9)[:, :, None] * GOLD[None, None, :] + (np.exp(-(r / 1.3) ** 2) * 0.25 * glow * v)[:, :, None] * GOLD[None, None, :]

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(SUN * 0.6 + CREAM * 0.4), rim_dir=(-0.7, -0.6), rim_v=0.6 * alpha, amb=(1.0, 0.8, 0.65), amb_v=0.16, aura=GOLD, aura_v=0.10 + 0.2 * getattr(s, '_gw', 0.0), alpha=alpha, shade=0.86)

    def frame(s, t):
        o = {}; s._gw = 0.0
        if t < T_CUT: f, sh = s.sc_field(t)
        elif t < T_GUARD: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.92)
        else: f, sh = s.sc_bash(t)
        f, sh = s.finale(f, sh, t, DUR, None, (1, 0.85, 0.4), (1, 0.6, 0.2), -1)
        return s.film(f, t, sh * s.k, 0.5, o.get("thr", 0.88), (1.0, 0.88, 0.7), 1.0 + 0.15 * min(sh, 8), 0.035, o.get("vig", 0.52), 1.15, 1.0)

    def knight(s, f, t, size=0.42, cx=None, cy=None, **kw):
        W, H, k = s.W, s.H, s.k; cx = cx if cx is not None else W / 2; cy = cy if cy is not None else H * 0.67; fy = cy + size * H * 0.45
        if size < 0.6:
            sh_ = s.L0(); cv2.ellipse(sh_, (int(cx), int(fy)), (int(W * 0.42 * size), int(H * 0.03 * size)), 0, 0, 360, 0.5, -1, cv2.LINE_AA); f *= (1 - cv2.GaussianBlur(sh_, (0, 0), 4 * k))[:, :, None]
        img = s.pose(t, **kw); s.draw(f, img, cx, cy, size)

    def tufts(s, t, u, n, gather):
        W, H, k = s.W, s.H, s.k; p = s._pf[:n]; sp = None
        a = p[:, 0] * 6.2832 + t * (0.5 + p[:, 2]); r0 = W * (0.75 - 0.45 * gather) * (0.5 + p[:, 1]);
        return a, r0, p

    # 1) 夕やけの 高原。金の 羊毛の わたが 盾に 集まる
    def sc_field(s, t):
        W, H, k = s.W, s.H, s.k
        f = s.highland(t); s.knight(f, t, sway=0.15, breath=0.5); sc_ = s.at(*s.A(*SHIELD))
        g = ease((t - 0.8) / 1.8); p = s._pf[:26]; q = np.clip((t - 0.6 - p[:, 3] * 1.2) / 1.0, 0, 1)
        sx = W * (p[:, 0] * 1.4 - 0.2); sy = H * (0.15 + 0.5 * p[:, 1]); x = sx + (sc_[0] - sx) * q ** 2 + np.sin(t * 2 + p[:, 2] * 9) * 12 * k * (1 - q); y = sy + (sc_[1] - sy) * q ** 2
        s.wool(f, x, y, (5 + 9 * p[:, 4]) * k * np.abs(np.sin(np.clip(q, 0.02, 1) * 3.1416)) ** 0.5 * (q > 0) * (q < 1), 1.0)
        s.flare(f, sc_, (6 + 10 * g + 2 * math.sin(t * 8) ** 2) * k, GOLD * 0.7 + CREAM * 0.3, 0.3 + 0.6 * g, 0.5, t, False)
        return f * clamp(t / 0.5), 0

    # 2) 目の カットイン(盾の 丸い 窓)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.highland(t, 0.30)
        a = back(clamp(u / 0.30), 1.4); c = (W / 2, H * 0.33); R = W * 0.33 * a
        if R > 6: s.shield(f, c, R * 1.18, min(1.0, a), 0.0)
        band = (SUN * 0.7)[None, None, :] * (0.6 + 0.5 * s.tex(1, s.XX * 0.6 - t * 30, s.YY * 0.6))[:, :, None] + (DUSK * 0.4)[None, None, :]
        ex, ey = s.A(*EYE); zf = 1.5; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(band, img, c[0] - (ex - s.aw / 2) * sc, c[1] - (ey - s.ah / 2) * sc, zf)
        d = np.hypot(s.XX - c[0], s.YY - c[1]); m = (sstep(R, R - 2, d) * (R > 6))[:, :, None]; f = f * (1 - m) + band * m
        s.glowmask(f, np.exp(-((d - R) / (2.0 * k)) ** 2) * (R > 6), CREAM, 0.7 * min(1.0, a), 1.0, 0.6, 6.0)
        n = len(s._vt); step = W * 0.165
        for i, T in enumerate(s._vt):                                                # 技名: どしん、と 1文字ずつ
            p = (u - 0.55 - i * 0.13) / 0.14
            if p > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.70 - (1 - ease(p)) * 30 * k, clamp(p * 2), 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.35 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.70 - 70 * k, 1.0, 1.0)
        if u > 1.35:
            g = ease((u - 1.35) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.70 + 52 * k), (W / 2 + W * 0.42 * g, H * 0.70 + 52 * k)], GOLD, 1.0, 0.8, 0.8)
            s.text(f, s.txt_en, W / 2, H * 0.70 + 78 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.7
        return f, 5 * math.exp(-u * 6) + sum(4 * math.exp(-max(0, u - 0.69 - i * 0.13) * 16) * (u > 0.69 + i * 0.13) for i in range(5))

    # 3) 羊毛の 輪 → 突進 → 盾が ぶつかる
    def sc_bash(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_GUARD; uc = t - T_CHARGE; ub = t - T_BASH; sh = 0.0
        ring = ease(u / 1.4); s._gw = ring
        f = s.highland(t, 1.0 - 0.3 * clamp(uc / 1.0))
        P = (W * 0.5, H * 0.46)                                                      # 盾が ぶつかる 所
        # 突進: キャラごと 大きく(盾の 点が P へ)
        z = ease2(uc / (T_BASH - T_CHARGE - 0.25)) if uc > 0 else 0.0; size = 0.42 * (1 + 2.3 * z ** 2.2)
        sc = size * H / s.ah; sxo = (s.A(*SHIELD)[0] - s.aw / 2) * sc; syo = (s.A(*SHIELD)[1] - s.ah / 2) * sc
        cx0 = W / 2; cy0 = H * 0.67; cx = cx0 + (P[0] - sxo - cx0) * z; cy = cy0 + (P[1] - syo - cy0) * z
        pf = s._pf; a = pf[:, 0] * 6.2832 + t * (1.2 + 0.8 * pf[:, 2]) * (1 if True else 0); rr = (W * (0.9 - 0.52 * ring) * (0.75 + 0.5 * pf[:, 1])) * (1 + 1.5 * z)
        hc = (cx, cy - size * H * 0.05); wx = hc[0] + np.cos(a) * rr; wy = hc[1] + np.sin(a) * rr * 0.62; back_ = np.sin(a) < 0; wr = (9 + 14 * pf[:, 4]) * k * ring * (1 + 1.2 * z)
        wv = 1.0 if ub < 0 else max(0.0, 1 - ub * 3)
        if wv > 0: s.wool(f, wx[back_], wy[back_], wr[back_], wv)
        if ub < 0.02:
            s.knight(f, t, size, cx, cy, sway=0.15 * (1 - z), breath=0.3 * (1 - z), lean=(4 * k * ease(u / 0.4)) * (1 - z), squash=(0.05 * math.exp(-uc * 6) if uc > 0 else -0.02 * ring))
            if uc > 0 and z < 0.8:                                                   # 土けむり
                dq = s._pf[:60]; dx = cx + (dq[:, 0] - 0.5) * W * 0.9 * (0.4 + z); dy = cy + size * H * 0.45 - dq[:, 1] * H * 0.10 * (0.3 + z) * (1 + uc * 2)
                s.splat(f, dx, dy, dq[:, 3] * 0.5 * (1 - z), SUN * 0.7 + 0.2, 3.0, 0.9)
            sp = (cx + sxo, cy + syo); s.flare(f, sp, (10 + 30 * z) * k, GOLD * 0.6 + CREAM * 0.4, 0.5 + 0.5 * ring, 0.6, t, False)
        if wv > 0: s.wool(f, wx[~back_], wy[~back_], wr[~back_], wv)
        if uc > 0: s.rushdust(f, t, P, 700, 1.0 + 1.5 * z, CREAM, 0.25 * z, 3, 1.0); sh += 2 + 5 * z
        # 式の 盾: キャラの 盾と 入れかわって 画面を おおう
        R_art = SR * s.ah / 760.0 * sc
        if z > 0.55 or ub > 0:
            e = clamp(ub / (T_HIT - T_BASH)) if ub > 0 else 0.0
            R = R_art * 1.02 if ub < 0 else W * (0.60 + 0.9 * e ** 2.0) * (1 + 0.06 * math.exp(-ub * 9) * math.cos(ub * 40))
            R = max(R, 8.0); s.shield(f, P if ub > 0 else (cx + sxo, cy + syo), R, clamp((z - 0.55) / 0.2) if ub < 0 else 1.0, clamp(ub * 3) * (0.3 + 0.9 * e) if ub > 0 else 0.0)
            if ub > 0:
                d = np.hypot(s.XX - P[0], s.YY - P[1])
                for i in range(3):                                                   # 衝撃の 輪
                    q = (ub - i * 0.09) / 0.7
                    if 0 < q < 1: f += (np.exp(-((d - (R + q * W * 1.2)) / ((6 + 30 * q) * k)) ** 2) * (1 - q) ** 1.5 * 1.3)[:, :, None] * (CREAM * 0.6 + GOLD * 0.4)[None, None, :]
                bp = s._pf; ba = bp[:, 0] * 6.2832; br_ = R * 0.95 + ub * W * (0.5 + 1.4 * bp[:, 1]); s.wool(f, P[0] + np.cos(ba) * br_, P[1] + np.sin(ba) * br_ + ub * ub * H * 0.25, (10 + 22 * bp[:, 4]) * k * max(0.0, 1 - ub * 0.9), 1.0)   # はじける 羊毛
                if ub < 0.10: f = f + (1 - ub / 0.10) * 1.1
                sh += 18 * math.exp(-ub * 4) + 2 + 7 * e
                f = s.godrays(f, P, 0.25 + 0.4 * e, 1.0, 8, 0.88, 0.45)
                if e > 0.78: f = f + ((e - 0.78) / 0.22) ** 2 * 1.4
        if u < 0.12: f = f + (1 - u / 0.12) * 0.35
        return f, sh

    # ---- この子だけの曲: 角笛と 牧歌ふうの 調べ → 太鼓が はやまる 突進 → ごおん、と 盾の 鐘 → 金管の 和音 ----
    def audio(s, path):
        a = Snd(DUR, 401); G = 196.0; mj = [0, 2, 4, 7, 9]
        def fq(d, o=0): return G * 2 ** ((mj[d % 5] + 12 * (d // 5 + o)) / 12.0)
        for j, d in enumerate((0, 2)): a.note(fq(d, -1), 0.1, 2.6, 0.05, "pad", 0.6)
        for (d, t0, ln) in ((2, 0.5, 0.7), (3, 1.2, 0.4), (4, 1.6, 0.9)): a.note(fq(d, 0), t0, ln, 0.10, "saw", 0.08)      # 角笛
        for i in range(8): a.note(fq(i % 5 + 5, 1), 0.8 + i * 0.2, 0.3, 0.05, "bell")
        a.boom(T_CUT, 50, 1.0, 0.45); a.whoosh(T_CUT - 0.3, T_CUT, 0.3)
        for d in (0, 2, 4): a.note(fq(d, 0), T_CUT, 1.9, 0.045, "saw", 0.05)
        for i in range(5): a.boom(T_CUT + 0.69 + i * 0.13, 66 - i * 3, 0.8, 0.15); a.note(fq([0, 2, 3, 4, 5][i], 1), T_CUT + 0.69 + i * 0.13, 0.4, 0.09, "bell")
        for i in range(12): a.note(fq(i % 5 + (i // 5) * 5, 1), T_GUARD + 0.1 + i * 0.12, 0.3, 0.06, "bell")              # 羊毛が うずを 巻く
        for j, d in enumerate((0, 2, 4)): a.note(fq(d, 0), T_GUARD + j * 0.3, T_CHARGE - T_GUARD, 0.05, "saw", 0.3)
        a.whoosh(T_GUARD, T_CHARGE, 0.3, 0.2)
        n = 13
        for i in range(n):                                                                                              # ひづめの 音: だんだん はやく
            tk = T_CHARGE + (T_BASH - T_CHARGE - 0.1) * (1 - (1 - i / n) ** 1.5); a.boom(tk, 80 - i, 0.35 + 0.05 * i, 0.07); a.hat(tk + 0.03, 0.05)
        a.rise(T_CHARGE, T_BASH, 90, 420, 0.13); a.whoosh(T_CHARGE + 0.4, T_BASH, 0.8, 0.4)
        a.boom(T_BASH, 36, 2.0, 0.9); a.boom(T_BASH, 95, 1.0, 0.25)
        for h_, g in ((1.0, 0.16), (2.76, 0.09), (5.4, 0.05), (8.9, 0.03)): a.note(G * 0.75 * h_, T_BASH, 1.6, g, "sine", 0.002)   # 盾の 鐘
        a.a += (a.nz - a.lp(a.nz, 0.5)) * a.env(T_BASH, 0.002, 0.12) * 0.6
        for j, d in enumerate((0, 2, 4, 5, 7, 9)): a.note(fq(d, 0), T_BASH + 0.1 + j * 0.04, T_HIT - T_BASH + 0.2, 0.05, "saw", 0.2)
        a.rise(T_BASH + 0.3, T_HIT, 80, 280, 0.12)
        a.boom(T_HIT, 38, 1.6, 0.8)
        for j, d in enumerate((0, 2, 4, 7)): a.note(fq(d, 1), T_HIT + 0.02, 0.6, 0.06, "bell")
        s.fin_snd(a, DUR, 401); a.save(path)


if __name__ == "__main__":
    r = Aries()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
