"""サーベ(★4 二刀の副船長) 専用の必殺技「波切リ二刀」— 高品質版(hs_fx)。二本の 曲刀から 水の 三日月が 2つ、ななめ十字に こちらへ 飛んで きて、画面が ×の 字に 四つに 切れて ずれる
月夜の 海の 岩に 立つ → 刀身の 形の ななめの 窓の カットイン+技名 → 一の太刀(三日月 1)→ 二の太刀(三日月 2)→
2つの 三日月が 大きく なって ななめ十字に 重なる → 画面が 四つに 切れて 外へ ずれ、切れ目から 光 → 白 → 暗転
使い方: python3 hs_422.py 出力.mp4 / python3 hs_422.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.2
T_CUT, T_ACT, T_S1, T_S2, T_X, T_HIT = 2.7, 4.7, 5.15, 5.6, 7.5, 9.3
NAVY = np.array([0.02, 0.04, 0.12], np.float32); TEAL = np.array([0.08, 0.26, 0.40], np.float32); SEA0 = np.array([0.02, 0.09, 0.18], np.float32); SEA1 = np.array([0.10, 0.36, 0.52], np.float32); FOAM = np.array([0.92, 0.98, 1.0], np.float32); PALE = np.array([0.80, 0.90, 1.0], np.float32); AQUA = np.array([0.40, 0.85, 1.0], np.float32)
EYE = (500, 225); HANDS = [(250, 455), (745, 455)]; TIPS = [(10, 220), (965, 270)]
A1, A2 = math.radians(-40), math.radians(-140)


class Sabe(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 422, (0.5, 0.85, 1.0), "波切リ二刀", "Wave Cutter", W); s.fx_init(422)
        s._vt = [s.text_img(c, hz.SERIF, int(70 * s.k), 3) for c in "波切リ二刀"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; s.hy = 0.56

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 月夜の 海と 岩 ----
    def nsea(s, t, lum=1.0):
        W, H, k = s.W, s.H, s.k; hy = s.hy; v = (s.yn / hy).clip(0, 1)
        f = NAVY[None, None, :] * (1 - v ** 1.5)[:, :, None] + TEAL[None, None, :] * (v ** 1.5)[:, :, None]
        dm = np.hypot(s.xn - 0.74, (s.yn - 0.15) * H / W)
        n = s.fbm(0.0009, t * 6, 0, 3, 80 * k, 1); n = cv2.GaussianBlur(n, (0, 0), 4 * k); cu = sstep(0.48, 0.68, n)
        f += (np.exp(-dm / 0.20) * 0.55 + sstep(0.085, 0.078, dm) * 1.2)[:, :, None] * np.array([0.90, 0.95, 1.0], np.float32)[None, None, :] * (1 - cu * 0.6)[:, :, None]
        lit = np.clip(np.exp(-dm / 0.35) * 1.4 * (0.5 + s.fbm(0.003, t * 9, 3, 3, 0, 2)), 0, 1.2)
        f = f * (1 - cu * 0.7)[:, :, None] + (cu * 0.7)[:, :, None] * ((NAVY * 1.8)[None, None, :] + (PALE * 0.75)[None, None, :] * lit[:, :, None])
        d = np.clip((s.yn - hy) / (1 - hy), 0, 1); pu = (s.XX - W / 2) / (d + 0.08) * 0.16; pv = 46.0 / (d + 0.08)
        w1 = s.tex(2, pu + t * 18, pv + t * 26); w2 = s.tex(0, pu * 2.1 - t * 24, pv * 2.1 + t * 36); wv = w1 * 0.6 + w2 * 0.4
        sw = SEA0[None, None, :] * (0.7 + 0.6 * wv)[:, :, None] + (SEA1 * 0.8)[None, None, :] * (d * (0.3 + 0.8 * wv))[:, :, None] + (sstep(0.72, 0.92, wv) * (0.15 + 0.5 * d))[:, :, None] * FOAM[None, None, :]
        sw += (np.exp(-np.abs(s.xn - 0.74 + (s.yn - hy) * 0.3) / (0.05 + 0.1 * d)) * sstep(0.45, 0.8, w2) * (0.9 - 0.5 * d))[:, :, None] * PALE[None, None, :]              # 月の 道
        m = sstep(hy - 0.002, hy + 0.003, s.yn)[:, :, None]; f = f * (1 - m) + sw * m
        f += (np.exp(-((s.yn - hy) / 0.012) ** 2) * 0.18)[:, :, None] * PALE[None, None, :]
        rt = 0.80 - 0.035 * np.exp(-((s.xn - 0.5) / 0.20) ** 2) + 0.9 * np.clip(np.abs(s.xn - 0.5) - 0.19, 0, 1) ** 1.25 - 0.012 * s.tex(1, s.XX * 0.8, 3)                  # 岩
        rm = sstep(-0.003, 0.003, s.yn - rt) * sstep(0.0, -0.006, s.yn - 0.895 - 0.012 * s.tex(2, s.XX * 0.5, 9))
        rc = np.array([0.06, 0.08, 0.12], np.float32)[None, None, :] * (0.5 + 1.4 * s.tex(1, s.XX * 0.9, s.YY * 0.9))[:, :, None] + (PALE * 0.45)[None, None, :] * (np.exp(-np.clip(s.yn - rt, 0, 1) / 0.012) * (0.4 + 0.6 * np.clip((s.xn - 0.4) * 3, 0, 1)))[:, :, None]
        f = f * (1 - rm)[:, :, None] + rc * rm[:, :, None]
        fo = np.exp(-((s.yn - 0.90) / 0.012) ** 2) * sstep(0.40, 0.30, np.abs(s.xn - 0.5)) * (0.5 + 0.7 * s.tex(2, s.XX * 1.2 + t * 30, s.YY * 1.2))
        f += (fo * 0.7)[:, :, None] * FOAM[None, None, :]
        return (f * lum).astype(np.float32)

    def mist(s, f, t, v=1.0):
        W, H = s.W, s.H; r = s.rnd[:100]
        x = ((r[:, 0] - t * (0.04 + 0.06 * r[:, 4])) % 1.0) * W; y = (0.45 + 0.5 * ((r[:, 1] - t * (0.03 + 0.04 * r[:, 2])) % 1.0)) * H
        s.splat(f, x, y, (0.3 + 0.6 * r[:, 2]) * v, FOAM, 1.1, 0.3)

    # ---- 水の 三日月(式): 弧の 帯。中ほどが 太く 両はしが 細い。大きく なると 画面を よこぎる 線に なる ----
    def moonblade(s, f, m, R, a, v=1.0, wd=1.0):
        if v <= 0.01: return
        k = s.k; c = (m[0] - R * math.cos(a), m[1] - R * math.sin(a)); dx = s.XX - c[0]; dy = s.YY - c[1]; d = np.hypot(dx, dy); th = np.arctan2(dy, dx)
        da = np.abs((th - a + math.pi) % (2 * math.pi) - math.pi); arcl = da * R; half = min(R * 1.15, s.W * 2.2); tp = np.clip(1 - arcl / half, 0, 1)
        w = (5 * k + R * 0.10 * min(1.0, 260 * k / R) * 1.0) * wd * tp ** 0.7 + 0.5
        inn = (d - R) / w; body = np.exp(-(inn / 0.9) ** 2) * (tp > 0); edge = np.exp(-((inn - 0.55) / 0.22) ** 2) * (tp > 0)
        wt = 0.6 + 0.6 * s.tex(2, arcl * 0.6, inn * 30 + R * 0.3); tail = np.exp(-np.clip(-inn, 0, 99) / 3.0) * (inn < 0) * tp * 0.35 * wt                                   # うしろに ひく 水
        f[:] = f * (1 - np.clip(body * 0.8 + tail * 0.5, 0, 1) * v)[:, :, None] + ((body * wt * 0.9)[:, :, None] * AQUA[None, None, :] + (edge * 1.3)[:, :, None] * FOAM[None, None, :] + tail[:, :, None] * PALE[None, None, :]) * v
        s.glowmask(f, body * tp, AQUA, 0.5 * v, 1.0, 0.6, 8.0)

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        s.draw_lit(f, img, cx, cy, hfrac, rim=(0.85, 0.93, 1.0), rim_dir=(0.6, -0.8), rim_v=0.6 * alpha, amb=(0.70, 0.82, 1.0), amb_v=0.20, aura=AQUA, aura_v=0.06, alpha=alpha, shade=0.88)

    def frame(s, t):
        o = {}
        if t < T_CUT: f, sh = s.sc_rock(t)
        elif t < T_ACT: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.5, thr=0.95)
        else: f, sh = s.sc_slash(t)
        f, sh = s.finale(f, sh, t, DUR, None, (0.85, 0.95, 1), (0.4, 0.8, 1.0), -1)
        return s.film(f, t, sh * s.k, 0.45, o.get("thr", 0.9), (0.90, 0.96, 1.0), 1.0 + 0.15 * min(sh, 8), 0.03, o.get("vig", 0.5), 1.2, 0.96)

    def mate(s, f, t, alpha=1.0, **kw):
        W, H, k = s.W, s.H, s.k; cy = H * 0.635
        img = s.pose(t, **kw); s.draw(f, img, W / 2, cy, 0.36, alpha)
        return [s.at(*s.A(*p)) for p in HANDS], [s.at(*s.A(*p)) for p in TIPS]

    # 1) 月夜の 岩に 立つ
    def sc_rock(s, t):
        f = s.nsea(t); s.mate(f, t, clamp(t / 0.6), sway=0.25, breath=0.5); s.mist(f, t, 0.6)
        return f * clamp(t / 0.5), 0

    # 2) 目の カットイン(刀身の 形の ななめの 窓)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.nsea(t, 0.34)
        a = ease(u / 0.25); cy = H * 0.32
        bandc = (NAVY * 2.5)[None, None, :] + (AQUA * 0.6)[None, None, :] * (0.25 + 0.9 * sstep(0.5, 0.92, s.tex(0, s.XX * 0.15 - t * 300, s.YY * 0.8 + s.XX * 0.2)))[:, :, None]
        ex, ey = s.A(*EYE); zf = 1.5; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(bandc, img, W / 2 - (ex - s.aw / 2) * sc, cy - (ey - s.ah / 2) * sc, zf)
        ca, sa = math.cos(-0.17), math.sin(-0.17); lx = (s.XX - W / 2) * ca + (s.YY - cy) * sa; ly = -(s.XX - W / 2) * sa + (s.YY - cy) * ca
        Lh = W * 0.62 * a + 1; hw = H * 0.13 * np.clip(1 - (lx / Lh) ** 2, 0, 1) ** 0.8 * (0.75 + 0.25 * np.clip(lx / Lh, -1, 1))                                        # 曲刀の 身: 片がわが ふくらむ
        d = np.where(ly < 0, -ly - hw * 1.25, ly - hw * 0.75)
        m = (sstep(0, -2, d) * (a > 0.05) * (np.abs(lx) < Lh))[:, :, None]; f = f * (1 - m) + bandc * m
        s.glowmask(f, np.exp(-(d / (2.2 * k)) ** 2) * (a > 0.05) * (np.abs(lx) < Lh), FOAM, 0.9 * a, 1.0, 0.8, 7.0)
        n = len(s._vt); step = W * 0.172
        for i, T in enumerate(s._vt):                                                # 技名: 左右から 交互に すべりこむ
            p = (u - 0.5 - i * 0.12) / 0.25
            if p > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step + (1 - ease(p)) * 70 * k * (1 if i % 2 else -1), H * 0.69, clamp(p) ** 1.2, 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.35 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.69 - 74 * k, 1.0, 1.0)
        if u > 1.4:
            g = ease((u - 1.4) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.69 + 56 * k), (W / 2 + W * 0.42 * g, H * 0.69 + 56 * k)], FOAM, 1.0, 0.7, 0.7)
            s.text(f, s.txt_en, W / 2, H * 0.69 + 82 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.5
        return f, 3 * math.exp(-u * 6)

    # 3) 一の太刀・二の太刀 → 三日月が 2つ 飛んで くる → ×に 切れて ずれる
    def sc_slash(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_ACT; sh = 0.0; C = (W * 0.5, H * 0.47)
        f = s.nsea(t)
        sq = 0.05 * ease(u / (T_S1 - T_ACT)) if t < T_S1 else -0.04 * (math.exp(-(t - T_S1) * 5) + (math.exp(-(t - T_S2) * 5) if t > T_S2 else 0))
        hands, tips = s.mate(f, t, 1.0, sway=0.2, breath=0.4, squash=sq)
        s.mist(f, t, 0.6)
        st = (W * 0.5, H * 0.58)
        for j, (t0, a) in enumerate(((T_S1, A1), (T_S2, A2))):
            q = t - t0
            if q < 0: continue
            if q < 0.22:                                                             # 刀の ひらめき
                hx, hy_ = hands[j]; tx, ty = tips[j]; g = 1 - q / 0.22
                s.glowline(f, [(hx, hy_), (tx + (tx - hx) * 0.25, ty + (ty - hy_) * 0.25)], FOAM, g, 2.5, 1.0); sh += 8 * g
                if q < 0.08: f[:] = f + (1 - q / 0.08) * 0.35
            if t < T_X + 0.12:
                p = clamp(q / (T_X - t0)); g = p ** 2.0; m = (st[0] + (C[0] - st[0]) * g, st[1] + (C[1] - st[1]) * g)
                s.moonblade(f, m, k * 26 + W * 3.0 * p ** 3.0, a, min(1.0, q * 8), 1.0 + 2.2 * p ** 2); sh += 5 * p ** 3
        if t >= T_X:                                                                 # 四つに 切れて ずれる
            e = clamp((t - T_X) / (T_HIT - T_X)); sft = k * (3 + 26 * clamp((t - T_X) / 0.25)) + W * 0.42 * e ** 2.4
            n1 = (math.cos(A1), math.sin(A1)); n2 = (math.cos(A2), math.sin(A2)); out = np.zeros_like(f); cov = s.L0()
            for s1 in (-1, 1):
                for s2 in (-1, 1):
                    ox = s1 * n1[0] + s2 * n2[0]; oy = s1 * n1[1] + s2 * n2[1]; ol = math.hypot(ox, oy); ox, oy = ox / ol * sft, oy / ol * sft * (1.0 + 0.3 * (s1 * s2 > 0))
                    sx = s.XX - ox; sy = s.YY - oy; inr = sstep(-1.0, 1.0, s1 * ((sx - C[0]) * n1[0] + (sy - C[1]) * n1[1])) * sstep(-1.0, 1.0, s2 * ((sx - C[0]) * n2[0] + (sy - C[1]) * n2[1]))
                    out += cv2.remap(f, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT) * inr[:, :, None]; cov += inr
            gap = np.clip(1 - cov, 0, 1); gl = cv2.GaussianBlur(gap, (0, 0), 10 * k)
            wtx = 0.55 + 0.6 * s.tex(2, s.XX * 0.5 + t * 60, s.YY * 0.5 - t * 90) + 0.25 * sstep(0.6, 0.9, s.tex(0, s.XX * 1.5, s.YY * 1.5 - t * 200))
            f = out * (1 - 0.25 * e) + (gap * 1.1 * wtx)[:, :, None] * FOAM[None, None, :] + (gl * (0.7 + 0.8 * e))[:, :, None] * AQUA[None, None, :]
            nn = 120; r = s.rnd[230:230 + nn]; side = np.where(r[:, 0] > 0.5, A1, A2) + math.pi / 2; ln = (r[:, 1] - 0.5) * 2 * H * 0.9; off = (r[:, 2] - 0.5) * 2 * (sft + 10 * k) * (0.3 + 1.2 * ((t - T_X) * (0.5 + r[:, 3])) % 1.0)   # 切れ目の しぶき
            s.splat(f, C[0] + np.cos(side) * ln - np.sin(side) * off, C[1] + np.sin(side) * ln + np.cos(side) * off, 0.5 + 0.8 * r[:, 3], FOAM, 1.6, 0.4)
            q = t - T_X
            if q < 0.14: f = f + (1 - q / 0.14) * 0.9
            sh += 12 * math.exp(-q * 4) + 2 + 6 * e ** 2
            if e > 0.8: f = f + ((e - 0.8) / 0.2) ** 2 * 1.6
        if u < 0.12: f = f + (1 - u / 0.12) * 0.25
        return f, sh

    # ---- この子だけの曲: 低い 弦の ような ひき波 → しゃっ(1)・しゃっ(2)→ 2つの 音が 近づいて 重なる → きん → 割れる ----
    def audio(s, path):
        a = Snd(DUR, 422); G = 130.8; pn = [0, 2, 3, 7, 8]
        def fq(d, o=0): return G * 2 ** ((pn[d % 5] + 12 * (d // 5 + o)) / 12.0)
        for d in (0, 3): a.note(fq(d, -1), 0.1, 2.6, 0.045, "pad", 0.7)
        wn = a.lp(a.nz, 0.18) - a.lp(a.nz, 0.03); hi = a.nz - a.lp(a.nz, 0.55)
        a.a += wn * (0.5 + 0.5 * np.sin(a.t * 1.6)) ** 2 * (a.t < T_ACT) * 0.5                                              # 波
        for i, d in enumerate((5, 7, 8, 7, 5, 3)): a.note(fq(d, 0), 0.5 + i * 0.34, 0.5, 0.06, "fm")
        a.boom(T_CUT, 58, 0.6, 0.4); a.whoosh(T_CUT - 0.3, T_CUT, 0.2, 0.1)
        for i in range(5): a.note(fq([5, 7, 8, 10, 12][i], 1), T_CUT + 0.5 + i * 0.12, 0.7, 0.07, "bell")
        for d in (0, 2, 3): a.note(fq(d, 0), T_CUT, 1.9, 0.04, "pad", 0.05)
        a.rise(T_ACT, T_S1, 120, 300, 0.07)
        for j, t0 in enumerate((T_S1, T_S2)):
            a.a += hi * a.env(t0, 0.003, 0.16) * 0.9; a.whoosh(t0, t0 + 0.3, 1.2, 0.2); a.note(fq(12 + j * 2, 1), t0 + 0.02, 0.5, 0.07, "bell"); a.boom(t0, 70, 0.3, 0.3)
            a.rise(t0 + 0.1, T_X, 260 + 90 * j, 620 + 20 * j, 0.06)
        g = np.clip((a.t - T_S1) / (T_X - T_S1), 0, 1) ** 2 * (a.t < T_X); a.a += wn * g * 1.3 + hi * g * 0.2
        for j, d in enumerate((0, 3, 7)): a.note(fq(d, 0), T_S2 + j * 0.06, T_X - T_S2, 0.045, "pad", 0.6)
        a.boom(T_X, 40, 1.4, 0.9); a.a += hi * a.env(T_X, 0.001, 0.3) * 1.0
        for j, d in enumerate((7, 12, 14)): a.note(fq(d, 2), T_X + j * 0.015, 1.4, 0.07, "bell")                          # きん
        g2 = np.clip((a.t - T_X) / (T_HIT - T_X), 0, 1) ** 1.5 * (a.t < T_HIT + 0.1) * (a.t > T_X); a.a += wn * g2 * 1.6 + a.lp(a.nz, 0.02) * g2 * 2.0
        a.rise(T_X + 0.2, T_HIT, 90, 380, 0.10)
        for j, d in enumerate((0, 2, 3, 7)): a.note(fq(d, 0), T_X + 0.3 + j * 0.08, T_HIT - T_X - 0.2, 0.045, "pad", 0.6)
        a.boom(T_HIT, 40, 1.5, 0.8)
        for j, d in enumerate((0, 3, 7, 10)): a.note(fq(d, 1), T_HIT + 0.02 + j * 0.03, 0.8, 0.05, "bell")
        s.fin_snd(a, DUR, 422); a.save(path)


if __name__ == "__main__":
    r = Sabe()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
