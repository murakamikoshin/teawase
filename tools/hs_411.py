"""テング(★4 烏天狗) 専用の必殺技「天狗風」— 高品質版(hs_fx)。羽うちわの ひとふりで 雲海に 竜巻が 立ち、ゆれながら こちらへ 近づいて のみこみ、うずの 中から 目(空)を 見上げる。黒い 羽が 画面を かすめる
夕やけの 雲海の 上に 浮かぶ(紅葉が ながれる)→ 扇形の 窓の カットイン+技名 → うちわを ふる(横なぐりの 風)→
とおくの 雲海に 竜巻 → ゆれながら 近づき 太る(本人は 風に のって 上へ)→ のまれる → うずの 中: 黒い 羽が 奥から 飛んで くる → 目の 光が ひらく → 白 → 暗転
使い方: python3 hs_411.py 出力.mp4 / python3 hs_411.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.6
T_CUT, T_ACT, T_SWING, T_TOR, T_IN, T_HIT = 2.7, 4.7, 5.0, 5.35, 8.0, 9.75
SUN = np.array([1.0, 0.62, 0.30], np.float32); DUSK = np.array([0.34, 0.20, 0.46], np.float32); TOP = np.array([0.06, 0.07, 0.20], np.float32); CLOUD = np.array([1.0, 0.74, 0.64], np.float32); INK = np.array([0.05, 0.05, 0.10], np.float32); PALE = np.array([0.95, 0.97, 1.0], np.float32)
EYE = (385, 270); FAN = (160, 320)


class Tengu(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 411, (0.75, 0.95, 0.8), "天狗風", "Tengu Gale", W); s.fx_init(411)
        s._vt = [s.text_img(c, hz.SERIF, int(88 * s.k), 3) for c in "天狗風"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; s.hy = 0.60

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 夕やけの 雲海 ----
    def dusk(s, t, lum=1.0):
        W, H, k = s.W, s.H, s.k; hy = s.hy; v = (s.yn / hy).clip(0, 1)
        f = TOP[None, None, :] * (1 - v ** 1.2)[:, :, None] + DUSK[None, None, :] * (v ** 1.2 * (1 - v ** 4))[:, :, None] + (SUN * 1.05)[None, None, :] * (v ** 4)[:, :, None]
        n = s.fbm(0.0009, t * 6, 0, 3, 80 * k, 1); n = cv2.GaussianBlur(n, (0, 0), 4 * k); cu = sstep(0.50, 0.70, n + 0.15 * np.exp(-((s.yn - 0.25) / 0.15) ** 2))
        lit = np.clip(0.4 + 1.6 * (s.fbm(0.003, t * 9, 3, 3, 0, 2) - 0.35), 0, 1.3)
        f = f * (1 - cu * 0.75)[:, :, None] + (cu * 0.75)[:, :, None] * (DUSK[None, None, :] * 0.9 + (CLOUD - DUSK * 0.6)[None, None, :] * (lit * (0.3 + 0.7 * v))[:, :, None])
        f += (np.exp(-np.hypot(s.xn - 0.5, (s.yn - hy) * 1.6) / 0.17) * 0.75)[:, :, None] * np.array([1.0, 0.82, 0.55], np.float32)[None, None, :]              # しずむ 日
        rd = hy + 0.01 - 0.10 * s.tex(2, s.XX * 0.16, 11) ** 2 - 0.05 * np.exp(-((s.xn - 0.86) / 0.12) ** 2) - 0.04 * np.exp(-((s.xn - 0.1) / 0.10) ** 2)      # 雲から 出た 峰
        hm = sstep(-0.003, 0.003, s.yn - rd); hc = (INK * 1.6)[None, None, :] + (DUSK * 0.5)[None, None, :] * (0.3 + 0.9 * s.tex(1, s.XX * 0.7, s.YY * 0.7) * np.clip((s.yn - rd) / 0.08, 0, 1))[:, :, None]
        f = f * (1 - hm)[:, :, None] + hc * hm[:, :, None]
        d = np.clip((s.yn - hy) / (1 - hy), 0, 1)
        pu = (s.XX - W / 2) / (d + 0.10) * 0.22; pv = 60.0 / (d + 0.10)
        c1 = s.tex(2, pu + t * 10, pv + t * 3); c2 = s.tex(0, pu * 2.3 - t * 16, pv * 2.3); cs = sstep(0.25, 0.85, c1 * 0.7 + c2 * 0.3)
        sea = (DUSK * 0.55)[None, None, :] + (CLOUD * 1.02 - DUSK * 0.55)[None, None, :] * (cs * (1.0 - 0.45 * d))[:, :, None]
        sea += (np.exp(-np.abs(s.xn - 0.5) / 0.12) * np.exp(-d / 0.5) * 0.35 * cs)[:, :, None] * SUN[None, None, :]
        m = sstep(hy - 0.004, hy + 0.012, s.yn)[:, :, None]; f = f * (1 - m) + sea * m
        f += (np.exp(-((s.yn - hy) / 0.018) ** 2) * 0.25)[:, :, None] * CLOUD[None, None, :]
        return (f * lum).astype(np.float32)

    def leaves(s, f, t, v=1.0, wind=0.0):
        W, H = s.W, s.H; r = s.rnd[:120]
        x = ((r[:, 0] + t * (0.05 + 0.08 * r[:, 4]) * (1 + 14 * wind)) % 1.0) * W; y = ((r[:, 1] + 0.03 * np.sin(t * (1 + r[:, 2]) + r[:, 3] * 6.28) + t * 0.02) % 1.0) * H
        br = (0.4 + 0.6 * r[:, 2]) * v; red = r[:, 5] > 0.45
        s.splat(f, x[red], y[red], br[red] * 1.5, (1.0, 0.25, 0.10), 1.6, 0.15); s.splat(f, x[~red], y[~red], br[~red] * 1.5, (1.0, 0.65, 0.15), 1.4, 0.15)

    def gust(s, f, t, v):
        if v <= 0.01: return
        g = sstep(0.62, 0.95, s.tex(0, s.XX * 0.10 - t * 900, s.YY * 1.3 + 30 * np.sin(s.XX * 0.004 + t * 3))) + 0.6 * sstep(0.7, 0.98, s.tex(1, s.XX * 0.06 - t * 1400, s.YY * 2.2))
        f += (g * v * 0.6)[:, :, None] * PALE[None, None, :]

    # ---- 黒い 羽(式): 細長い 葉の 形 + 羽じく + 羽枝の すじ ----
    def feather(s, f, c, L, ang, v=1.0, soft=0.0):
        if v <= 0.01 or L < 4: return
        M = s.L0(); Q = s.L0(); ca, sa = math.cos(ang), math.sin(ang); n = 14; up = []; dn = []
        for j in range(n + 1):
            u = j / n; w = L * 0.17 * math.sin(math.pi * u ** 0.75) * (1 - 0.3 * u); px = c[0] + (u - 0.5) * L * ca + 0.06 * L * math.sin(u * 3.1) * -sa; py = c[1] + (u - 0.5) * L * sa + 0.06 * L * math.sin(u * 3.1) * ca
            up.append((px - sa * w, py + ca * w)); dn.append((px + sa * w * 0.8, py - ca * w * 0.8)); 
            if j: cv2.line(Q, (int(lx), int(ly)), (int(px), int(py)), 1.0, max(1, int(L * 0.012)), cv2.LINE_AA)
            lx, ly = px, py
        cv2.fillPoly(M, [np.array(up + dn[::-1], np.int32)], 1.0, cv2.LINE_AA)
        M = cv2.GaussianBlur(M, (0, 0), max(0.7, L * (0.006 + soft))) * v; Q = cv2.GaussianBlur(Q, (0, 0), max(0.6, L * 0.004))
        lx_ = (s.XX - c[0]) * ca + (s.YY - c[1]) * sa; ly_ = -(s.XX - c[0]) * sa + (s.YY - c[1]) * ca
        barb = 0.5 + 0.5 * np.sin((lx_ + np.abs(ly_) * 1.6) / max(1.5, L * 0.012))                                    # 羽枝
        sheen = np.clip(0.25 + 0.5 * barb * (0.4 + 0.6 * s.tex(1, lx_ * 0.4, ly_ * 0.4)) + 0.5 * np.clip(-ly_ / (L * 0.15), 0, 1), 0, 1.4)
        body = INK[None, None, :] * 0.8 + np.array([0.16, 0.17, 0.30], np.float32)[None, None, :] * sheen[:, :, None] + (Q * 0.5)[:, :, None] * np.array([0.8, 0.8, 0.9], np.float32)[None, None, :]
        f[:] = f * (1 - M)[:, :, None] + body * M[:, :, None]

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        s.draw_lit(f, img, cx, cy, hfrac, rim=(1.0, 0.72, 0.42), rim_dir=(0.0, 1.0), rim_v=0.6 * alpha, amb=(1.0, 0.8, 0.75), amb_v=0.16, aura=SUN, aura_v=0.08, alpha=alpha, shade=0.92)

    def frame(s, t):
        o = {}
        if t < T_CUT: f, sh = s.sc_sky(t)
        elif t < T_ACT: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.5, thr=0.95)
        else: f, sh = s.sc_gale(t)
        f, sh = s.finale(f, sh, t, DUR, None, (1.0, 0.95, 0.85), (0.7, 0.9, 0.8), -1)
        return s.film(f, t, sh * s.k, 0.45, o.get("thr", 0.9), (1.0, 0.95, 0.9), 1.0 + 0.15 * min(sh, 8), 0.03, o.get("vig", 0.5), 1.2, 0.96)

    def boy(s, f, t, alpha=1.0, dy=0.0, **kw):
        W, H, k = s.W, s.H, s.k; cy = H * (0.50 + 0.012 * math.sin(t * 2.4)) + dy
        img = s.pose(t, **kw); s.draw(f, img, W / 2, cy, 0.40, alpha)
        return s.at(*s.A(*FAN))

    # 1) 夕やけの 雲海の 上に 浮かぶ
    def sc_sky(s, t):
        f = s.dusk(t); s.boy(f, t, clamp(t / 0.6), sway=0.3, breath=0.5); s.leaves(f, t, 0.8)
        return f * clamp(t / 0.5), 0

    # 2) 目の カットイン(うちわの 扇形の 窓)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.dusk(t, 0.34); s.leaves(f, t, 0.4, 0.3)
        a = ease(u / 0.3); cy = H * 0.33
        band = (INK * 2.0 + DUSK * 0.5)[None, None, :] + (SUN * 0.55)[None, None, :] * (0.3 + 0.9 * sstep(0.5, 0.92, s.tex(0, s.XX * 0.12 - t * 320, s.YY * 0.9)))[:, :, None]
        ex, ey = s.A(*EYE); zf = 1.5; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(band, img, W / 2 - (ex - s.aw / 2) * sc, cy - (ey - s.ah / 2) * sc, zf)
        px, py = W / 2, cy + H * 0.36; rr = np.hypot(s.XX - px, s.YY - py); th = np.arctan2(s.XX - px, py - s.YY)                # かなめ(下)からの 扇形
        r1, r2 = H * 0.25, H * 0.25 + H * 0.24 * a; am = 0.86 * a
        d = np.maximum(np.maximum(r1 - rr, rr - r2), (np.abs(th) - am) * rr)                                                  # 中が 負
        m = (sstep(0, -2, d) * (a > 0.05))[:, :, None]; f = f * (1 - m) + band * m
        s.glowmask(f, np.exp(-(d / (2.2 * k)) ** 2) * (a > 0.05), PALE, 0.9 * a, 1.0, 0.8, 7.0)
        for i in range(-3, 4):                                                       # うちわの 骨
            b = am * i / 3.5; s.glowline(f, [(px + math.sin(b) * (r1 - 26 * k), py - math.cos(b) * (r1 - 26 * k)), (px + math.sin(b) * (r1 - 4 * k), py - math.cos(b) * (r1 - 4 * k))], PALE, 0.6 * a, 0.6, 0.4)
        n = len(s._vt); step = W * 0.25
        for i, T in enumerate(s._vt):                                                # 技名: 風に のって 横から
            p = (u - 0.5 - i * 0.14) / 0.3
            if p > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step + (1 - ease(p)) * 90 * k, H * 0.72, clamp(p) ** 1.2, 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.35 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.72 - 84 * k, 1.0, 1.0)
        if u > 1.4:
            g = ease((u - 1.4) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.72 + 64 * k), (W / 2 + W * 0.42 * g, H * 0.72 + 64 * k)], PALE, 1.0, 0.7, 0.7)
            s.text(f, s.txt_en, W / 2, H * 0.72 + 90 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.5
        return f, 3 * math.exp(-u * 6)

    # ---- 竜巻(外から): ゆれる ろうと。筒に 巻いた もよう ----
    def funnel(s, f, t, g):
        W, H, k = s.W, s.H, s.k; S = 0.11 * math.exp(g * 3.5); yb = H * (s.hy + 0.02 + 0.55 * g ** 2.2); X0 = W * (0.5 + 0.13 * math.sin(t * 1.9) * (1 - g) ** 1.5)
        up = np.clip((yb - s.YY) / H, 0, 2)
        cxl = X0 + S * W * 0.35 * np.sin(up * 6.0 / (S + 0.3) + t * 2.6) * up ** 0.8 / (up + 0.25)
        w = S * (W * 0.085 + W * 1.1 * up ** 1.6) * (0.92 + 0.08 * np.sin(up * 40 / (S + 0.5) - t * 9)) + 1.0
        un = (s.XX - cxl) / w; ins = sstep(1.0, 0.86, np.abs(un)) * sstep(0.0, 0.008, (yb - s.YY) / H) * min(1.0, g * 6)
        cyl = np.arcsin(np.clip(un, -1, 1)) / math.pi
        tw = s.tex(1, cyl * 512 + t * 520 + up * 300 / (S + 0.4), up * 420 / (S + 0.4) - cyl * 120); tw2 = s.tex(0, cyl * 768 + t * 800, up * 700 / (S + 0.4))
        lit = np.clip(0.55 - 0.7 * un * 0.0 + 0.5 * np.sqrt(np.clip(1 - un * un, 0, 1)), 0, 1)
        colr = (DUSK * 0.55 + INK)[None, None, :] + (CLOUD * 0.95)[None, None, :] * (sstep(0.3, 0.85, tw * 0.65 + tw2 * 0.35) * lit)[:, :, None] + (SUN * 0.35)[None, None, :] * (np.clip(-un, 0, 1) ** 2 * 0.0 + np.exp(-up / 0.12) * 0.6)[:, :, None]
        f[:] = f * (1 - ins * 0.94)[:, :, None] + colr * (ins * 0.94)[:, :, None]
        bw = S * W * 0.42; base = np.exp(-(((s.XX - X0) / bw) ** 2 + ((s.YY - yb) / (bw * 0.22)) ** 2)) * min(1.0, g * 6)                       # 足もとの 雲の しぶき
        f[:] = f * (1 - base * 0.5)[:, :, None] + (base * (0.45 + 0.6 * s.tex(2, s.XX * 0.6 / (S + 0.3) + t * 90, s.YY * 0.6 / (S + 0.3))))[:, :, None] * CLOUD[None, None, :]
        n = 9                                                                        # まわりを まわる 羽
        for i in range(n):
            r = s.rnd[200 + i]; hgt = 0.08 + 0.5 * r[0]; a = t * (3.0 + 2 * r[1]) + r[2] * 6.28; ww = S * (W * 0.10 + W * 1.1 * hgt ** 1.6) * 1.25
            if math.sin(a) > -0.2: s.feather(f, (X0 + math.cos(a) * ww, yb - hgt * H * min(1.0, S * 2.2) - math.sin(a) * ww * 0.12), S * W * 0.22 * (0.7 + 0.6 * r[3]), a * 1.7 + r[4] * 6, min(1.0, g * 5))
        return X0, yb

    # 3) うちわを ふる → 竜巻が 近づく → のまれる → うずの 中
    def sc_gale(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_ACT; sh = 0.0
        if t < T_IN + 0.25:
            g = clamp((t - T_TOR) / (T_IN - T_TOR)); f = s.dusk(t, 1.0 - 0.35 * g)
            if t > T_TOR: s.funnel(f, t, g); sh += 1 + 6 * g ** 2
            sw = t - T_SWING; rise = ease2(clamp((t - 6.2) / 1.5))
            if rise < 0.98:
                fp = s.boy(f, t, 1.0 - clamp((rise - 0.6) / 0.38), -H * 0.75 * rise, sway=0.3, breath=0.4, squash=(0.04 * ease(u / 0.3) if sw < 0 else -0.04 * math.exp(-sw * 4)))
                if 0 < sw < 0.5:                                                    # ひとふりの 弧
                    q = sw / 0.5; pts = [(fp[0] + math.cos(-2.4 + j * 0.11 * 2.6 * ease(q * 2)) * 95 * k + 30 * k, fp[1] + math.sin(-2.4 + j * 0.11 * 2.6 * ease(q * 2)) * 95 * k + 40 * k) for j in range(12)]
                    s.glowline(f, pts, PALE, 1.0 - q, 2.2, 0.9)
            wv = math.exp(-max(0.0, sw) * 1.6) * (sw > 0) + 0.25 * g + 0.5 * g ** 3
            s.gust(f, t, wv); s.leaves(f, t, 0.9, wv)
            if 0 < sw < 0.12: f = f + (1 - sw / 0.12) * 0.35; sh += 7 * (1 - sw / 0.12)
            if t > T_IN - 0.25: f = f * (1 - 0.0) + clamp((t - (T_IN - 0.25)) / 0.25) * (1 - clamp((t - T_IN) / 0.25)) * 0.0
            fo = f
        if t >= T_IN:                                                                # うずの 中から 目を 見上げる
            e = clamp((t - T_IN) / (T_HIT - T_IN)); ee = e ** 2.2; C = (W * 0.5 + W * 0.05 * math.sin(t * 2.3) * (1 - e), H * (0.36 + 0.06 * e))
            dxp = s.XX - C[0]; dyp = s.YY - C[1]; dist = np.hypot(dxp, dyp) + 1e-3; th = np.arctan2(dyp, dxp); lg = np.log(dist / k + 6.0)
            w1 = s.tex(1, th / 6.2832 * 768 + lg * 150 + t * 420, lg * 60 - t * 90); w2 = s.tex(0, th / 6.2832 * 1280 + lg * 260 + t * 700, lg * 34 - t * 50); w3 = s.tex(2, th / 6.2832 * 512 + lg * 90 + t * 250, lg * 40)
            dep = np.clip(dist / (W * 0.9), 0, 1)
            fi = (INK * 1.2 + DUSK * 0.45)[None, None, :] * (0.5 + 0.8 * w3)[:, :, None] + (CLOUD * 0.9)[None, None, :] * (sstep(0.45, 0.9, w1) * (0.25 + 0.6 * (1 - dep)))[:, :, None] + (PALE * 0.5)[None, None, :] * (sstep(0.6, 0.95, w2) * 0.6)[:, :, None]
            Re = k * 28 + W * 0.9 * ee
            fi = fi * (1 - np.exp(-(dist / (Re * 1.7)) ** 2) * 0.5)[:, :, None] + (np.exp(-(dist / Re) ** 2) * (1.0 + 0.8 * e))[:, :, None] * np.array([1.0, 0.93, 0.78], np.float32)[None, None, :]         # 目(空の 光)
            fi = fi.astype(np.float32)
            for i in range(10):                                                      # 黒い 羽が 奥から うずを えがいて 飛んで くる
                r = s.rnd[220 + i]; q = ((t - T_IN) * (0.75 + 0.5 * r[0]) + r[1] * 1.3) % 1.3 / 1.3
                if e > 0.9 and q < 0.3: continue
                a = r[2] * 6.28 + q * 3.2 + t * 0.6; rad = (k * 30 + W * 0.85 * q ** 1.8)
                s.feather(fi, (C[0] + math.cos(a) * rad, C[1] + math.sin(a) * rad * 1.15), k * (20 + 330 * q ** 2.2), a + 1.9 + r[3], min(1.0, q * 6) * clamp((1.0 - q) / 0.1), 0.01 * q)
            fi = s.godrays(fi, C, 0.10 + 0.3 * e, 1.05, 8, 0.88, 0.4); sh += 3 + 8 * ee
            if e > 0.82: fi = fi + ((e - 0.82) / 0.18) ** 2 * 1.6
            x = clamp((t - T_IN) / 0.25)
            f = fi if t >= T_IN + 0.25 else (fo * (1 - x) + fi * x + (1 - abs(2 * x - 1)) * 0.25)
        if u < 0.12: f = f + (1 - u / 0.12) * 0.25
        return f, sh

    # ---- この子だけの曲: 山伏の 法螺貝と 太鼓 → ひとふりの 風 → うなりが 太る → うずの 中の 高い 笛 → 一打 ----
    def audio(s, path):
        a = Snd(DUR, 411); G = 146.8; pn = [0, 2, 5, 7, 9]
        def fq(d, o=0): return G * 2 ** ((pn[d % 5] + 12 * (d // 5 + o)) / 12.0)
        a.note(fq(0, 0), 0.2, 1.1, 0.07, "fm"); a.note(fq(3, 0), 1.2, 1.3, 0.08, "fm")                                 # 法螺貝
        for d in (0, 3): a.note(fq(d, -1), 0.1, 2.6, 0.04, "pad", 0.7)
        for tk in (0.9, 1.9, 2.3): a.boom(tk, 70, 0.35, 0.3)
        a.boom(T_CUT, 58, 0.6, 0.4); a.whoosh(T_CUT - 0.3, T_CUT, 0.2, 0.1)
        for i in range(3): a.note(fq([5, 7, 8][i], 1), T_CUT + 0.5 + i * 0.14, 0.7, 0.08, "bell"); a.boom(T_CUT + 0.5 + i * 0.14, 90, 0.2, 0.2)
        for d in (0, 2, 3): a.note(fq(d, 0), T_CUT, 1.9, 0.04, "pad", 0.05)
        a.whoosh(T_SWING - 0.1, T_SWING + 0.5, 0.9, 0.3); a.boom(T_SWING, 55, 0.8, 0.4)
        wn = a.lp(a.nz, 0.22) - a.lp(a.nz, 0.03); hi = a.nz - a.lp(a.nz, 0.5); lo = a.lp(a.nz, 0.02)
        g = np.clip((a.t - T_TOR) / (T_IN - T_TOR), 0, 1) ** 1.6 * (a.t < T_HIT + 0.2) * (0.75 + 0.25 * np.sin(a.t * 7.0 + 2 * np.sin(a.t * 1.9)))
        a.a += wn * g * 1.7 + hi * g * 0.15 + lo * g * 2.5
        a.rise(T_TOR, T_IN, 60, 180, 0.10)
        for i in range(6): a.boom(T_TOR + 0.3 + i * 0.42 * (1 - 0.04 * i), 62, 0.4, 0.25 + 0.05 * i)
        a.boom(T_IN, 44, 1.3, 0.7)
        mel = [5, 7, 8, 10, 8, 7, 10, 12, 13, 12, 15, 17]; st = (T_HIT - T_IN - 0.3) / 12
        for i in range(12): a.note(fq(mel[i], 1), T_IN + 0.15 + i * st, st * 1.2, 0.07, "fm")
        for j, d in enumerate((0, 2, 3, 5)): a.note(fq(d, 0), T_IN + j * 0.08, T_HIT - T_IN + 0.2, 0.045, "pad", 0.5)
        a.rise(T_IN, T_HIT, 140, 520, 0.10)
        a.boom(T_HIT, 40, 1.5, 0.8)
        for j, d in enumerate((0, 3, 5, 8)): a.note(fq(d, 1), T_HIT + 0.02 + j * 0.03, 0.8, 0.05, "bell")
        s.fin_snd(a, DUR, 411); a.save(path)


if __name__ == "__main__":
    r = Tengu()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
