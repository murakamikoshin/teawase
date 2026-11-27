"""ユキメ(★4 雪女) 専用の必殺技「吹雪ノ吐息」— 高品質版(hs_fx)。口もとからの 白い 吐息が 吹雪の 筒に なって こちらへ 吹きつけ、雪の 結晶が つぎつぎ 飛んで きて、さいごに 大きな 結晶が 画面を うめる
夜の 雪原(月あかり・しずかに ふる 雪)に 立つ。手のひらの 上で 結晶が 育つ → 六角の 窓の カットイン+技名 →
ふる 雪が 口もとへ すいこまれる(息を すう)→ ふっと 吐く: 白い 息が ひろがり 放射の 吹雪に → 結晶が 飛んで くる → 大きな 結晶 → 白 → 暗転
使い方: python3 hs_410.py 出力.mp4 / python3 hs_410.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.3
T_CUT, T_ACT, T_BLOW, T_BIG, T_HIT = 2.7, 4.7, 5.55, 8.2, 9.6
ICE = np.array([0.55, 0.82, 1.0], np.float32); NAVY = np.array([0.03, 0.05, 0.14], np.float32); FOAM = np.array([0.93, 0.98, 1.0], np.float32); LILA = np.array([0.62, 0.60, 0.95], np.float32)
EYE = (325, 262); MOUTH = (323, 318); PALM = (100, 335)
FLY = [(5.9, 1.3, -0.62, -0.50), (6.3, 1.2, 0.70, -0.30), (6.75, 1.2, -0.50, 0.55), (7.1, 1.1, 0.55, 0.60), (7.45, 1.0, 0.10, -0.75), (7.7, 0.9, -0.80, 0.05)]   # (はじまり, 長さ, 向き x, y)


class Yukime(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 410, (0.6, 0.85, 1.0), "吹雪ノ吐息", "Blizzard Breath", W); s.fx_init(410)
        s._vt = [s.text_img(c, hz.SERIF, int(70 * s.k), 3) for c in "吹雪ノ吐息"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; s.hy = 0.62

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 夜の 雪原 ----
    def night(s, t, lum=1.0):
        W, H, k = s.W, s.H, s.k; hy = s.hy; v = (s.yn / hy).clip(0, 1)
        f = NAVY[None, None, :] * (1 - v ** 1.5)[:, :, None] + np.array([0.15, 0.25, 0.46], np.float32)[None, None, :] * (v ** 1.5)[:, :, None]
        n = s.fbm(0.0009, t * 4, 0, 3, 80 * k, 1); n = cv2.GaussianBlur(n, (0, 0), 5 * k); cu = sstep(0.48, 0.72, n)
        f = f + (cu * 0.16)[:, :, None] * LILA[None, None, :]
        dm = np.hypot(s.xn - 0.78, (s.yn - 0.13) * H / W)                                                           # 月と かさ
        f += (np.exp(-dm / 0.16) * 0.55 + sstep(0.062, 0.055, dm) * 1.1 + np.exp(-((dm - 0.19) / 0.012) ** 2) * 0.10)[:, :, None] * np.array([0.85, 0.92, 1.0], np.float32)[None, None, :]
        rd = hy - 0.035 - 0.075 * s.tex(2, s.XX * 0.22, 7) - 0.05 * np.exp(-((s.xn - 0.2) / 0.25) ** 2)             # とおくの 雪山
        hm = sstep(-0.003, 0.003, s.yn - rd); hc = np.array([0.20, 0.30, 0.50], np.float32)[None, None, :] * (0.55 + 0.9 * np.clip((s.yn - rd) / 0.10, 0, 1) + 0.25 * s.tex(1, s.XX * 0.6, s.YY * 0.6))[:, :, None]
        f = f * (1 - hm)[:, :, None] + hc * hm[:, :, None]
        d = np.clip((s.yn - hy) / (1 - hy), 0, 1); g = s.fbm(0.003, 0, 3, 3, 0, 2)
        sn = np.array([0.42, 0.56, 0.86], np.float32)[None, None, :] * (0.55 + 0.55 * d + 0.35 * g * (0.4 + d))[:, :, None]
        sn += (np.exp(-np.hypot(s.xn - 0.72, (s.yn - hy) * 3.5) / 0.20) * 0.30)[:, :, None]                         # 月の てり返し
        sn += (sstep(0.93, 1.0, s.tex(0, s.XX * 1.7 + t * 3, s.YY * 1.7)) * 0.5 * d)[:, :, None]                      # きらめき
        m = sstep(hy - 0.004, hy + 0.004, s.yn)[:, :, None]; f = f * (1 - m) + sn * m
        f += (np.exp(-((s.yn - hy) / 0.02) ** 2) * 0.12)[:, :, None] * ICE[None, None, :]
        return (f * lum).astype(np.float32)

    def snowfall(s, f, t, v=1.0, pull=0.0, to=None):
        W, H, k = s.W, s.H, s.k; r = s.rnd[:260]
        x = ((r[:, 0] + 0.03 * np.sin(t * (0.8 + r[:, 2]) + r[:, 3] * 6.28) + t * 0.01) % 1.0) * W; y = ((r[:, 1] + t * (0.05 + 0.09 * r[:, 4])) % 1.0) * H
        if pull > 0 and to is not None:
            g = np.clip(pull * (0.6 + 0.8 * r[:, 5]), 0, 1) ** 1.5; x = x + (to[0] - x) * g; y = y + (to[1] - y) * g; v = v * (1 - 0.0 * g)
        br = (0.35 + 0.65 * r[:, 2]) * v; big = r[:, 4] > 0.6
        s.splat(f, x[~big], y[~big], br[~big], FOAM, 0.9, 0.3); s.splat(f, x[big], y[big], br[big] * 1.6, FOAM, 2.0, 0.3)

    # ---- 雪の 結晶(式): 六本の 枝 + 小枝 + 芯の 六角 ----
    def crystal(s, f, c, R, rot, v=1.0, soft=0.0):
        if v <= 0.01 or R < 3: return
        k = s.k; L = s.L0(); th = max(1, int(R * 0.055)); cx, cy = c
        for i in range(6):
            a = rot + i * math.pi / 3; ca, sa = math.cos(a), math.sin(a)
            cv2.line(L, (int(cx), int(cy)), (int(cx + ca * R), int(cy + sa * R)), 1.0, th, cv2.LINE_AA)
            for u, bl in ((0.34, 0.30), (0.56, 0.24), (0.78, 0.15)):
                px, py = cx + ca * R * u, cy + sa * R * u
                for sg in (-1, 1):
                    b = a + sg * math.pi / 3; cv2.line(L, (int(px), int(py)), (int(px + math.cos(b) * R * bl), int(py + math.sin(b) * R * bl)), 1.0, max(1, int(th * 0.75)), cv2.LINE_AA)
            cv2.circle(L, (int(cx + ca * R), int(cy + sa * R)), max(1, int(th * 0.9)), 1.0, -1, cv2.LINE_AA)
        hx = np.array([(cx + math.cos(rot + math.pi / 6 + i * math.pi / 3) * R * 0.20, cy + math.sin(rot + math.pi / 6 + i * math.pi / 3) * R * 0.20) for i in range(6)], np.int32)
        cv2.polylines(L, [hx], True, 1.0, max(1, int(th * 0.8)), cv2.LINE_AA)
        L = cv2.GaussianBlur(L, (0, 0), max(0.6, R * (0.006 + soft))) * v
        fac = 0.55 + 0.6 * s.tex(1, s.XX * 0.9 + rot * 20, s.YY * 0.9)                                                # 氷の 面の むら
        body = (ICE * 0.55)[None, None, :] + (FOAM * 0.75)[None, None, :] * fac[:, :, None]
        f[:] = f * (1 - L * 0.80)[:, :, None] + body * (L * 0.95)[:, :, None]
        s.glowmask(f, L, ICE, 0.55 * v, 1.0, 0.7, max(3.0, R * 0.06 / k))

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        s.draw_lit(f, img, cx, cy, hfrac, rim=(0.85, 0.93, 1.0), rim_dir=(0.6, -0.8), rim_v=0.55 * alpha, amb=(0.72, 0.82, 1.0), amb_v=0.22, aura=ICE, aura_v=0.10, alpha=alpha, shade=0.9)

    def frame(s, t):
        o = {}
        if t < T_CUT: f, sh = s.sc_snow(t)
        elif t < T_ACT: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.5, thr=0.95)
        else: f, sh = s.sc_blow(t)
        f, sh = s.finale(f, sh, t, DUR, None, (0.8, 0.92, 1), (0.5, 0.8, 1.0), -1)
        return s.film(f, t, sh * s.k, 0.45, o.get("thr", 0.9), (0.90, 0.96, 1.0), 1.0 + 0.15 * min(sh, 8), 0.03, o.get("vig", 0.5), 1.2, 0.96)

    def lady(s, f, t, alpha=1.0, grow=1.0, **kw):
        W, H, k = s.W, s.H, s.k; cy = H * 0.575; fy = cy + 0.44 * H * 0.47
        sd = np.exp(-(((s.XX - W / 2) / (W * 0.22)) ** 2 + ((s.YY - fy) / (H * 0.016)) ** 2)); f *= (1 - 0.45 * alpha * sd)[:, :, None]          # 雪の 上の かげ
        img = s.pose(t, **kw); s.draw(f, img, W / 2, cy, 0.44, alpha)
        p = s.at(*s.A(*PALM)); m = s.at(*s.A(*MOUTH))
        if grow > 0.02:
            c = (p[0], p[1] - 30 * k - 3 * k * math.sin(t * 2.2)); s.crystal(f, c, k * (4 + 22 * grow), t * 0.8, alpha * min(1.0, grow * 3))
            s.glowmask(f, np.exp(-(np.hypot(s.XX - c[0], s.YY - c[1]) / (26 * k)) ** 2), ICE, 0.35 * alpha * grow, 1.0, 0.5, 6.0)
        return p, m

    # 1) 夜の 雪原に 立つ。手のひらで 結晶が 育つ
    def sc_snow(s, t):
        f = s.night(t); s.snowfall(f, t, 0.8); s.lady(f, t, clamp(t / 0.6), ease(clamp((t - 0.7) / 1.6)), sway=0.25, breath=0.5); s.snowfall(f, t + 31.7, 0.5)
        return f * clamp(t / 0.5), 0

    # 2) 目の カットイン(六角の 窓)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.night(t, 0.34); s.snowfall(f, t, 0.4)
        a = ease(u / 0.28); cy = H * 0.33; R = H * 0.15 * a
        band = (NAVY * 2.2)[None, None, :] + (ICE * 0.55)[None, None, :] * (0.35 + 0.8 * sstep(0.5, 0.92, s.tex(0, s.XX * 0.4 - t * 40, s.YY * 0.4 + s.XX * 0.25)))[:, :, None]
        ex, ey = s.A(*EYE); zf = 1.55; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(band, img, W / 2 - (ex - s.aw / 2) * sc, cy - (ey - s.ah / 2) * sc, zf)
        dx = np.abs(s.XX - W / 2) / 1.75; dy = np.abs(s.YY - cy); d = np.maximum(dx * 0.866 + dy * 0.5, dy)                # よこ長の 六角
        m = (sstep(R, R - 2, d) * (R > 4))[:, :, None]; f = f * (1 - m) + band * m
        s.glowmask(f, np.exp(-((d - R) / (2.2 * k)) ** 2) * (R > 4), FOAM, 0.9 * a, 1.0, 0.8, 7.0)
        if a > 0.5:
            for sg in (-1, 1): s.crystal(f, (W / 2 + sg * W * 0.40, cy - sg * H * 0.118), 20 * k * a, t * 0.9 * sg, a)
        n = len(s._vt); step = W * 0.172
        for i, T in enumerate(s._vt):                                                # 技名: 霜が おりる ように
            p = (u - 0.5 - i * 0.12) / 0.35
            if p > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.68 - (1 - ease(p)) * 20 * k, clamp(p) ** 1.2, 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.35 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.68 - 72 * k, 1.0, 1.0)
        if u > 1.4:
            g = ease((u - 1.4) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.68 + 54 * k), (W / 2 + W * 0.42 * g, H * 0.68 + 54 * k)], FOAM, 1.0, 0.7, 0.7)
            s.text(f, s.txt_en, W / 2, H * 0.68 + 80 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.5
        return f, 3 * math.exp(-u * 6)

    # 3) 息を すう(雪が 口もとへ)→ 吐く: 吹雪の 筒 → 結晶が 飛んで くる → 大きな 結晶
    def sc_blow(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_ACT; ub = t - T_BLOW; sh = 0.0
        eb = clamp(ub / 2.3) if ub > 0 else 0.0; e = clamp((t - T_BIG) / (T_HIT - T_BIG)) if t > T_BIG else 0.0
        f = s.night(t, 1.0 - 0.25 * eb)
        inh = clamp(u / (T_BLOW - T_ACT)) if ub < 0 else 0.0
        p, m = s.lady(f, t, 1.0, max(0.0, 1 - u / 0.5), sway=0.2, breath=0.4, squash=(0.03 * ease(inh) if ub < 0 else -0.03 * math.exp(-ub * 3)))
        s.snowfall(f, t, 0.9 * (1 - eb), inh ** 1.6 if ub < 0 else 0.0, m)
        if ub < 0: s.glowmask(f, np.exp(-(np.hypot(s.XX - m[0], s.YY - m[1]) / ((6 + 16 * inh) * k)) ** 2), FOAM, 0.9 * inh ** 2, 1.0, 0.6, 5.0)
        else:
            dxp = s.XX - m[0]; dyp = s.YY - m[1]; dist = np.hypot(dxp, dyp) + 1e-3; th = np.arctan2(dyp, dxp)
            Rb = k * 14 + W * 1.75 * eb ** 1.5
            inside = sstep(Rb, Rb * 0.45, dist); edge = np.exp(-((dist - Rb * 0.85) / (Rb * 0.16 + 3 * k)) ** 2)
            lg = np.log(dist / k + 4.0)
            st1 = sstep(0.50, 0.92, s.tex(0, th / 6.2832 * 1536, lg * 16 - t * 150)); st2 = sstep(0.55, 0.95, s.tex(1, th / 6.2832 * 2560 + 40, lg * 26 - t * 260))
            mist = s.tex(2, th / 6.2832 * 512 + t * 12, lg * 22 - t * 70)
            den = 0.30 + 0.55 * eb
            f[:] = f * (1 - inside * den * 0.6 * (0.5 + 0.5 * mist))[:, :, None] + ((inside * (0.30 + 0.45 * mist) * den + inside * st1 * (0.45 + 0.45 * eb) + inside * st2 * 0.55 + edge * (0.25 + 0.4 * mist) * (1 - eb * 0.5))[:, :, None] * (FOAM * 0.85 + ICE * 0.2)[None, None, :])
            f[:] = f + (np.exp(-dist / (W * (0.05 + 0.25 * eb))) * (0.5 + 0.5 * eb))[:, :, None] * FOAM[None, None, :] * 0.35 * (1 - e)      # 吹き出し口の 白
            n = 150; r = s.rnd[100:100 + n]; q = (r[:, 0] + t * (0.5 + 0.7 * r[:, 1])) % 1.0; ang = r[:, 2] * 6.2832; rr = Rb * 1.05 * q ** 1.8           # 飛ぶ 雪つぶ
            px = m[0] + np.cos(ang) * rr; py = m[1] + np.sin(ang) * rr; br = (0.3 + 0.7 * r[:, 3]) * np.clip((q - 0.2) * 5, 0, 1)
            for lo, hi_, bl in ((0, 0.45, 0.9), (0.45, 0.75, 2.2), (0.75, 1.01, 4.5)):
                w = (q >= lo) & (q < hi_); s.splat(f, px[w], py[w], br[w] * bl, FOAM, bl, 0.3)
            sh += 2 + 4 * eb
            if ub < 0.12: f[:] = f + (1 - ub / 0.12) * 0.45; sh += 6 * (1 - ub / 0.12)
            for j, (t0, dur, vx, vy) in enumerate(FLY):                              # 結晶が 飛んで くる(口もとから ふちへ、大きく なり ながら)
                q = (t - t0) / dur
                if 0 < q < 1.15:
                    g = q ** 2.2; c = (m[0] + vx * W * 0.95 * g, m[1] + vy * H * 0.75 * g + 0 * k); s.crystal(f, c, k * (6 + 250 * q ** 2.6), t * (1.6 + 0.4 * j) * (1 if j % 2 else -1), min(1.0, q * 5) * clamp((1.15 - q) / 0.12), 0.012 * q)
                    sh += 2 * q
            if e > 0:                                                                # 大きな 結晶が まっすぐ こちらへ
                ee = e ** 2.4; c = (m[0] + (W * 0.5 - m[0]) * ease2(e), m[1] + (H * 0.44 - m[1]) * ease2(e))
                s.glowmask(f, np.exp(-(np.hypot(s.XX - c[0], s.YY - c[1]) / (k * 40 + W * 0.8 * ee)) ** 2), ICE, 0.12 + 0.2 * e, 1.0, 0.3, 9.0)
                s.crystal(f, c, k * 10 + W * 1.25 * ee, 0.3 + 1.1 * e, min(1.0, e * 6), 0.002)
                f[:] = s.godrays(f, c, 0.06 + 0.2 * e, 1.05, 8, 0.88, 0.4); sh += 2 + 9 * ee
                if e > 0.82: f[:] = f + ((e - 0.82) / 0.18) ** 2 * 1.6
        if u < 0.12: f = f + (1 - u / 0.12) * 0.25
        return f, sh

    # ---- この子だけの曲: 凍った 鈴(高い 短調)→ 息を すう 無音 → 風の うなり+笛 → 結晶が 鳴る → 大きな 一打 ----
    def audio(s, path):
        a = Snd(DUR, 410); G = 220.0; pn = [0, 2, 3, 7, 8]
        def fq(d, o=0): return G * 2 ** ((pn[d % 5] + 12 * (d // 5 + o)) / 12.0)
        for d in (0, 3): a.note(fq(d, -1), 0.1, 2.6, 0.04, "pad", 0.7)
        rg = np.random.RandomState(410)
        for i in range(9): a.note(fq([9, 7, 8, 5, 7, 4, 5, 3, 5][i], 1), 0.45 + i * 0.25, 0.9, 0.055, "bell")
        a.boom(T_CUT, 58, 0.6, 0.35); a.whoosh(T_CUT - 0.3, T_CUT, 0.2, 0.1)
        for i in range(5): a.note(fq([5, 7, 8, 9, 12][i], 1), T_CUT + 0.5 + i * 0.12, 0.8, 0.08, "bell")
        for d in (0, 2, 3): a.note(fq(d, 0), T_CUT, 1.9, 0.04, "pad", 0.05)
        a.rise(T_ACT + 0.1, T_BLOW, 900, 300, 0.05); a.whoosh(T_ACT + 0.1, T_BLOW, 0.25, 0.5)                              # 息を すう
        wn = a.lp(a.nz, 0.30) - a.lp(a.nz, 0.05); hi = a.nz - a.lp(a.nz, 0.5)                                            # 吹雪
        g = np.clip((a.t - T_BLOW) / 0.25, 0, 1) * (a.t < T_HIT + 0.2) * (0.55 + 0.45 * np.clip((a.t - T_BLOW) / (T_HIT - T_BLOW), 0, 1)) * (0.8 + 0.2 * np.sin(a.t * 5.3) * np.sin(a.t * 1.7))
        a.a += wn * g * 1.5 + hi * g * 0.22
        a.boom(T_BLOW, 50, 1.0, 0.45)
        for i, d in enumerate((5, 8, 7)): a.note(fq(d, 0), T_BLOW + 0.1 + i * 0.75, 1.1, 0.05, "fm")                    # 風の 笛
        for j, (t0, dur, vx, vy) in enumerate(FLY):
            a.note(fq(9 + j, 1), t0 + 0.1, 0.5, 0.07, "bell"); a.note(fq(9 + j, 2), t0 + dur * 0.8, 0.25, 0.05, "bell"); a.whoosh(t0 + dur * 0.5, t0 + dur, 0.3, 0.35)
        for j, d in enumerate((0, 2, 3, 7)): a.note(fq(d, 0), T_BIG + j * 0.08, T_HIT - T_BIG + 0.2, 0.05, "pad", 0.5)
        a.rise(T_BIG, T_HIT, 110, 420, 0.11)
        for i in range(10): tk = T_BIG + 0.1 + (T_HIT - T_BIG - 0.2) * (i / 10) ** 0.8; a.note(fq(5 + i, 1), tk, 0.3, 0.07, "bell")
        a.boom(T_HIT, 40, 1.5, 0.8)
        for j, d in enumerate((0, 3, 5, 7)): a.note(fq(d, 2), T_HIT + 0.02 + j * 0.03, 0.9, 0.05, "bell")
        s.fin_snd(a, DUR, 410); a.save(path)


if __name__ == "__main__":
    r = Yukime()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
