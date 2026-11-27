"""ピスケ(★4 魚座の水術師) 専用の必殺技「双魚ノ環」— 高品質版(hs_fx)。水の 魚が 二匹、弧を えがいて こちらへ とびこみ(画面に 水の つぶ)、さいごに 二匹が 追いかけ合う 水の 輪が こちらを くぐる
朝もやの 水鏡(空を うつす 水面)に 立つ。両手の 水の 輪が まわる(前ふり) → 目の カットイン+技名 →
左の 輪から 青い 魚が はねて 弧を えがき、こちらへ とびこむ(しぶき・画面に 水の つぶ)→ 右の 輪から 白い 魚も →
二匹が まん中で 追いかけ合って 水の 輪に なり、輪が ひろがって こちらを くぐる → 白 → 暗転
使い方: python3 hs_408.py 出力.mp4 / python3 hs_408.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.4
T_CUT, T_ACT, T_RING, T_HIT = 2.7, 4.7, 7.45, 9.7
LEAP = [(5.2, 1.05, -1), (6.05, 1.05, 1)]                                              # (はじまり, 長さ, 左右)
AQUA = np.array([0.35, 0.78, 1.0], np.float32); DEEP = np.array([0.10, 0.34, 0.75], np.float32); FOAM = np.array([0.92, 0.98, 1.0], np.float32); ROSE = np.array([1.0, 0.76, 0.80], np.float32)
EYE = (455, 262); RINGS = [(130, 300), (745, 310)]


class Pisce(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 408, (0.5, 0.85, 1.0), "双魚ノ環", "Twin Fish Ring", W); s.fx_init(408)
        s._vt = [s.text_img(c, hz.SERIF, int(76 * s.k), 3) for c in "双魚ノ環"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; s.hy = 0.60; rg = np.random.RandomState(4080); s._bd = rg.rand(2, 22, 5).astype(np.float32)

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 朝もやの 水鏡 ----
    def mirror(s, t, lum=1.0):
        W, H, k = s.W, s.H, s.k; hy = s.hy; v = (s.yn / hy).clip(0, 1)
        f = np.array([0.42, 0.62, 0.92], np.float32)[None, None, :] * (1 - v ** 1.6)[:, :, None] + (ROSE * 1.02)[None, None, :] * (v ** 1.6)[:, :, None]
        n = s.fbm(0.0009, t * 5, 0, 3, 80 * k, 1); n = cv2.GaussianBlur(n, (0, 0), 4 * k); cu = sstep(0.50, 0.66, n + 0.22 * np.exp(-((s.yn - 0.30) / 0.2) ** 2) - 0.06)
        f = f * (1 - cu * 0.8)[:, :, None] + (FOAM * 0.98)[None, None, :] * (cu * 0.8 * (0.85 + 0.3 * s.fbm(0.003, t * 8, 3, 3, 0, 2)))[:, :, None]
        f += (np.exp(-np.hypot(s.xn - 0.5, (s.yn - hy) * 1.4) / 0.20) * 0.45)[:, :, None] * np.array([1.0, 0.95, 0.85], np.float32)[None, None, :]
        d = np.clip((s.yn - hy) / (1 - hy), 0, 1)
        mx = s.XX + (np.sin(s.YY * 0.30 + t * 2.2) * 3.0 + np.sin(s.YY * 0.8 - t * 3.6) * 1.2) * k * (0.3 + 2.5 * d); rf = cv2.remap(f, mx.astype(np.float32), (2 * hy * H - s.YY - 2).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        rf = rf * np.array([0.72, 0.84, 0.98], np.float32)[None, None, :] * (0.98 - 0.30 * d)[:, :, None] * (0.9 + 0.2 * s.tex(0, s.XX * 0.2 + t * 5, s.YY * 2.6))[:, :, None]
        m = sstep(hy - 0.002, hy + 0.002, s.yn)[:, :, None]; f = f * (1 - m) + rf * m
        f += (np.exp(-((s.yn - hy) / 0.012) ** 2) * 0.35)[:, :, None]                                                    # 地平の もや
        return (f * lum).astype(np.float32)

    def ripples(s, f, c, t, n=3, v=0.6, R=0.34):
        W, H, k = s.W, s.H, s.k; L = s.L0()
        for i in range(n):
            q = (t * 0.45 + i / n) % 1.0; cv2.ellipse(L, (int(c[0]), int(c[1])), (int(W * (0.04 + R * q)), int(H * (0.004 + R * 0.10 * q))), 0, 0, 360, float((1 - q) ** 1.5), max(1, int(2 * k)), cv2.LINE_AA)
        s.glowmask(f, L, FOAM, v, 1.0, 0.5, 5.0)

    # ---- 水の 魚(式): せなかの 曲線に そった 管 + 尾びれ + てり ----
    def fish(s, f, c, ang, L, bend, white=False, v=1.0):
        if v <= 0.01 or L < 6: return
        k = s.k; M = s.L0(); S = s.L0(); ca, sa = math.cos(ang), math.sin(ang); n = 16; P = []
        for j in range(n + 1):
            u = j / n; lx = (0.5 - u) * L; ly = bend * math.sin(u * 3.4 - 0.4) * L * 0.16 * u; w = L * 0.17 * (math.sin(math.pi * (0.12 + 0.80 * u)) ** 0.7) * (1 - 0.35 * u) + L * 0.02
            P.append((c[0] + lx * ca - ly * sa, c[1] + lx * sa + ly * ca, w))
        tx, ty, tw = P[-1]; dx = P[-1][0] - P[-3][0]; dy = P[-1][1] - P[-3][1]; dl = math.hypot(dx, dy) + 1e-6; dx /= dl; dy /= dl                              # 尾びれ
        tail = np.array([(tx - dx * L * 0.02, ty - dy * L * 0.02), (tx + dx * L * 0.26 - dy * L * 0.20, ty + dy * L * 0.26 + dx * L * 0.20), (tx + dx * L * 0.14, ty + dy * L * 0.14), (tx + dx * L * 0.26 + dy * L * 0.20, ty + dy * L * 0.26 - dx * L * 0.20)], np.int32)
        cv2.fillPoly(M, [tail], 1.0, cv2.LINE_AA); cv2.fillPoly(S, [tail], 0.40, cv2.LINE_AA)
        for x, y, w in P: cv2.circle(M, (int(x), int(y)), int(w + 2 * k), 1.0, -1, cv2.LINE_AA); cv2.circle(S, (int(x), int(y)), int(w + 2 * k), 0.10, -1, cv2.LINE_AA)
        for x, y, w in P: cv2.circle(S, (int(x), int(y)), int(w), 0.45, -1, cv2.LINE_AA)
        for x, y, w in P: cv2.circle(S, (int(x - w * 0.30), int(y - w * 0.38)), max(1, int(w * 0.38)), 0.85, -1, cv2.LINE_AA)
        hx, hy_, hw = P[1]; cv2.circle(S, (int(hx - hw * 0.35), int(hy_ - hw * 0.45)), max(1, int(hw * 0.18)), 1.0, -1, cv2.LINE_AA)
        ex, ey, ew = P[2]; cv2.circle(S, (int(ex + sa * ew * 0.35), int(ey - ca * ew * 0.35)), max(1, int(ew * 0.20)), 0.0, -1, cv2.LINE_AA)                               # 目
        M = cv2.GaussianBlur(M, (0, 0), 0.8 * k) * v; S = cv2.GaussianBlur(S, (0, 0), max(0.8, L * 0.008)) * (0.93 + 0.14 * s.tex(1, s.XX * 0.3, s.YY * 0.3))
        dk = (np.array([0.20, 0.36, 0.62], np.float32) if white else np.array([0.04, 0.16, 0.50], np.float32)); md = (np.array([0.80, 0.90, 1.0], np.float32) if white else AQUA * 0.9); hi = np.array([1.15, 1.15, 1.15], np.float32)
        a1 = sstep(0.05, 0.50, S)[:, :, None]; a2 = sstep(0.55, 1.0, S)[:, :, None]; col = dk + (md - dk) * a1 + (hi - md) * a2
        f[:] = f * (1 - M * 0.88)[:, :, None] + col * (M * 0.92)[:, :, None]                                             # 少し すける

    def beads(s, f, j, age):
        """画面に ついた 水の つぶ(うしろの 景色を さかさに うつす)"""
        W, H, k = s.W, s.H, s.k; src = f.copy()
        for b in s._bd[j]:
            r = int((7 + 30 * b[2] ** 2) * k); x = int((0.08 + 0.84 * b[0]) * W); y = int((0.10 + 0.75 * b[1]) * H + age * (10 + 60 * b[3]) * k * (r / (20 * k)))
            a = clamp(age * 8) * clamp(1.6 - age * 0.55)
            if a <= 0.01 or y - 2 * r < 0 or y + 2 * r >= H or x - 2 * r < 0 or x + 2 * r >= W: continue
            patch = cv2.resize(src[y - 2 * r:y + 2 * r, x - 2 * r:x + 2 * r], (2 * r, 2 * r), interpolation=cv2.INTER_AREA)[::-1, ::-1]
            yy, xx = np.mgrid[-r:r, -r:r].astype(np.float32) / r; d = np.hypot(xx, yy); m = sstep(1.0, 0.88, d) * a
            col = patch * (0.85 + 0.25 * d ** 2)[:, :, None] + (np.exp(-(np.hypot(xx + 0.35, yy + 0.4) / 0.2) ** 2) * 1.2)[:, :, None] - (np.exp(-((d - 0.92) / 0.08) ** 2) * 0.25 * (yy > 0))[:, :, None]
            f[y - r:y + r, x - r:x + r] = f[y - r:y + r, x - r:x + r] * (1 - m)[:, :, None] + col * m[:, :, None]

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(FOAM), rim_dir=(0.0, -1.0), rim_v=0.5 * alpha, amb=(1.0, 0.92, 0.95), amb_v=0.10, aura=AQUA, aura_v=0.10, alpha=alpha, shade=0.95)

    def frame(s, t):
        o = {}
        if t < T_CUT: f, sh = s.sc_pond(t)
        elif t < T_ACT: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.95)
        else: f, sh = s.sc_fish(t)
        f, sh = s.finale(f, sh, t, DUR, None, (0.7, 0.9, 1), (0.4, 0.8, 1.0), -1)
        return s.film(f, t, sh * s.k, 0.45, o.get("thr", 0.92), (0.92, 0.97, 1.0), 1.0 + 0.15 * min(sh, 8), 0.03, o.get("vig", 0.45), 1.22, 0.96)

    def mage(s, f, t, alpha=1.0, spin=1.0, **kw):
        W, H, k = s.W, s.H, s.k; cy = H * 0.585; fy = cy + 0.42 * H * 0.47
        s.ripples(f, (W / 2, fy), t, 3, 0.5 * alpha)
        img = s.pose(t, **kw); s.draw(f, img, W / 2, cy, 0.42, alpha); out = []
        for i, rp in enumerate(RINGS):                                               # 手の 水の 輪: まわる 光
            p = s.at(*s.A(*rp)); out.append(p); L = s.L0(); a0 = t * 4 * spin * (1 if i == 0 else -1); R = 37 * k
            for j in range(10): a = a0 - j * 0.22; cv2.circle(L, (int(p[0] + math.cos(a) * R), int(p[1] + math.sin(a) * R)), max(1, int((5 - j * 0.4) * k)), float(1 - j / 10), -1, cv2.LINE_AA)
            s.glowmask(f, cv2.GaussianBlur(L, (0, 0), 1.0 * k), FOAM, 0.8 * alpha, 1.0, 0.7, 6.0)
        return out

    # 1) 朝もやの 水鏡に 立つ
    def sc_pond(s, t):
        f = s.mirror(t); s.mage(f, t, clamp(t / 0.6), 0.5 + 1.0 * clamp((t - 1.0) / 1.5), sway=0.3, breath=0.5)
        return f * clamp(t / 0.5), 0

    # 2) 目の カットイン(水の 輪の 丸窓 2つが つながる)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.mirror(t, 0.36)
        a = ease(u / 0.28); cy = H * 0.33; R = H * 0.125 * a
        band = (DEEP * 0.9)[None, None, :] + (AQUA * 0.6)[None, None, :] * (0.4 + 0.8 * sstep(0.45, 0.9, s.tex(1, s.XX * 0.5 + t * 18, s.YY * 0.5 - t * 9)))[:, :, None]
        ex, ey = s.A(*EYE); zf = 1.5; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(band, img, W / 2 - (ex - s.aw / 2) * sc, cy - (ey - s.ah / 2) * sc, zf)
        d = np.minimum(np.hypot(s.XX - W * (0.5 - 0.20 * a), s.YY - cy), np.hypot(s.XX - W * (0.5 + 0.20 * a), s.YY - cy))                                   # 2つの 円が 重なった 形
        m = (sstep(R, R - 2, d) * (R > 4))[:, :, None]; f = f * (1 - m) + band * m
        s.glowmask(f, np.exp(-((d - R) / (2.2 * k)) ** 2) * (R > 4), FOAM, 0.9 * a, 1.0, 0.8, 7.0)
        n = len(s._vt); step = W * 0.20
        for i, T in enumerate(s._vt):                                                # 技名: 水から 浮かぶ
            p = (u - 0.5 - i * 0.14) / 0.35
            if p > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.68 + (1 - ease(p)) * 26 * k + 3 * k * math.sin(t * 3.5 + i), clamp(p) ** 1.2, 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.35 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.68 - 76 * k, 1.0, 1.0)
        if u > 1.4:
            g = ease((u - 1.4) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.68 + 58 * k), (W / 2 + W * 0.42 * g, H * 0.68 + 58 * k)], FOAM, 1.0, 0.7, 0.7)
            s.text(f, s.txt_en, W / 2, H * 0.68 + 84 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.5
        return f, 3 * math.exp(-u * 6)

    # 3) 魚が 1匹ずつ とびこむ → 二匹の 輪が こちらを くぐる
    def sc_fish(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_ACT; ur = t - T_RING; sh = 0.0
        e = clamp(ur / (T_HIT - T_RING)) if ur > 0 else 0.0; ee = e ** 2.3
        f = s.mirror(t, 1.0 - 0.15 * e)
        a_ = 1 - clamp((e - 0.45) / 0.25)
        hp = s.mage(f, t, a_, 1.5, sway=0.3, breath=0.4, squash=-0.02 * ease(u / 0.5)) if a_ > 0.02 else [(W * 0.2, H * 0.5), (W * 0.8, H * 0.5)]
        for j, (t0, dur, sg) in enumerate(LEAP):
            p = (t - t0) / dur
            if 0 < p < 1:                                                            # はねて 弧を えがき、こちらへ
                st = hp[0 if sg < 0 else 1]; x = st[0] + (W * (0.5 + sg * 0.10) - st[0]) * p + sg * W * 0.10 * math.sin(p * 3.14); y = st[1] - H * 0.42 * math.sin(p * 2.6) + H * 0.30 * p ** 3
                dxp = (W * (0.5 + sg * 0.10) - st[0]) + sg * W * 0.10 * 3.14 * math.cos(p * 3.14); dyp = -H * 0.42 * 2.6 * math.cos(p * 2.6) + H * 0.90 * p * p
                L = k * (46 + 620 * p ** 2.6); s.fish(f, (x, y), math.atan2(dyp, dxp), L, math.sin(t * 16) * (1 - p * 0.5), sg > 0, min(1.0, p * 6))
                rg = s.rnd[j * 40:j * 40 + 40]; s.splat(f, x - dxp * 0.12 * rg[:, 0] + (rg[:, 1] - 0.5) * L * 0.5, y - dyp * 0.12 * rg[:, 0] + (rg[:, 2] - 0.5) * L * 0.5, rg[:, 3] * 0.9 * (0.4 + p), FOAM, 1.0 + 2 * p, 0.4)
                sh += 4 * p ** 2
            q = t - t0 - dur
            if q > 0:
                if q < 0.14: f = f + (1 - q / 0.14) * 0.75; sh += 10 * (1 - q / 0.14)
                s.beads(f, j, q)
        if ur > 0:
            C = (W * 0.5, H * (0.40 + 0.08 * ease2(e))); R = k * (26 + 60 * e) + W * 0.95 * ee; rot = ur * 3.2 + 2.0 * ee
            d = np.hypot(s.XX - C[0], s.YY - C[1]) / R; th = np.arctan2(s.YY - C[1], s.XX - C[0]); wv = s.tex(1, th / 6.2832 * 512 - rot * 60, d * 20) ; band = np.exp(-((d - 1.0) / 0.20) ** 2)       # 水の 輪
            f = f * (1 - band * 0.6)[:, :, None] + (band * (0.45 + 0.7 * wv))[:, :, None] * (AQUA * 0.8 + FOAM * 0.2)[None, None, :] + (np.exp(-((d - 0.84) / 0.03) ** 2) * 0.8)[:, :, None] * FOAM[None, None, :]
            L = R * 1.25
            for i, wh in ((0, False), (1, True)):
                a = rot + i * math.pi; s.fish(f, (C[0] + math.cos(a) * R, C[1] + math.sin(a) * R), a + math.pi / 2 + 0.25, L, 0.9, wh, min(1.0, ur * 4))
            n = 60; aa = rot - np.linspace(0, 6.28, n); rr = R * (1.05 + 0.25 * s.rnd[:n, 0]); s.splat(f, C[0] + np.cos(aa) * rr, C[1] + np.sin(aa) * rr, s.rnd[:n, 1] * 0.8, FOAM, 1.0 + 3 * e, 0.4)
            f = s.godrays(f, C, 0.15 + 0.4 * e, 1.05, 8, 0.88, 0.4); sh += 1 + 9 * ee
            if ur < 0.1: f = f + (1 - ur / 0.1) * 0.4
            if e > 0.84: f = f + ((e - 0.84) / 0.16) ** 2 * 1.5
        if u < 0.12: f = f + (1 - u / 0.12) * 0.25
        return f, sh

    # ---- この子だけの曲: 水琴窟の ような 鈴と たて琴 → 魚が はねる 2回(水音)→ 2つの 旋律が 追いかけ合う カノン → 水の 幕 ----
    def audio(s, path):
        a = Snd(DUR, 408); G = 196.0; pn = [0, 2, 4, 7, 9]
        def fq(d, o=0): return G * 2 ** ((pn[d % 5] + 12 * (d // 5 + o)) / 12.0)
        for d in (0, 2): a.note(fq(d, -1), 0.1, 2.6, 0.04, "pad", 0.7)
        rg = np.random.RandomState(408)
        for i in range(12): a.note(fq(int(rg.rand() * 8) + 5, 1), 0.4 + i * 0.19 + rg.rand() * 0.05, 0.7, 0.06, "bell")
        a.boom(T_CUT, 60, 0.6, 0.35); a.whoosh(T_CUT - 0.3, T_CUT, 0.2, 0.1)
        for i in range(4): a.note(fq([2, 4, 5, 7][i], 1), T_CUT + 0.5 + i * 0.14, 0.8, 0.09, "bell")
        for d in (0, 2, 4): a.note(fq(d, 0), T_CUT, 1.9, 0.04, "pad", 0.05)
        for j, (t0, dur, sg) in enumerate(LEAP):                                                                        # はねる → とびこむ
            a.note(fq(5 + j * 2, 1), t0, 0.3, 0.09, "bell"); a.rise(t0, t0 + dur, 300 + 80 * j, 1200, 0.05); a.whoosh(t0 + 0.4, t0 + dur, 0.5, 0.4)
            ts = t0 + dur; a.boom(ts, 52, 1.2, 0.4); a.a += (a.lp(a.nz, 0.25) - a.lp(a.nz, 0.04)) * a.env(ts, 0.005, 0.35) * 1.4 + (a.nz - a.lp(a.nz, 0.6)) * a.env(ts, 0.002, 0.12) * 0.4
            for i in range(8): a.note(900 + rg.rand() * 1800, ts + 0.05 + rg.rand() * 0.5, 0.05, 0.03, "sine", 0.002)
        mel = [5, 7, 9, 7, 5, 9, 10, 12, 10, 9, 12, 14]; st = (T_HIT - T_RING - 0.2) / 14
        for i in range(14):                                                                                             # 追いかけ合う 2つの 声
            tk = T_RING + 0.1 + i * st * (1 - 0.02 * i); a.note(fq(mel[i % 12], 1), tk, 0.3, 0.08, "bell"); a.note(fq(mel[i % 12], 0), tk + st * 0.5, 0.3, 0.07, "fm")
        for j, d in enumerate((0, 2, 4, 5, 7)): a.note(fq(d, 0), T_RING + j * 0.08, T_HIT - T_RING + 0.2, 0.045, "pad", 0.5)
        a.rise(T_RING, T_HIT, 100, 360, 0.11); a.a += (a.lp(a.nz, 0.2) - a.lp(a.nz, 0.03)) * np.clip((a.t - T_RING) / (T_HIT - T_RING), 0, 1) ** 2 * (a.t < T_HIT) * (a.t > T_RING) * 1.4
        a.boom(T_HIT, 40, 1.5, 0.8)
        for j, d in enumerate((0, 2, 4, 7)): a.note(fq(d, 1), T_HIT + 0.02, 0.6, 0.06, "bell")
        s.fin_snd(a, DUR, 408); a.save(path)


if __name__ == "__main__":
    r = Pisce()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
