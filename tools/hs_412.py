"""オニマル(★4 赤鬼の若武者) 専用の必殺技「金棒落トシ」— 高品質版(hs_fx)。地を けって 空へ 消え、見上げた 空から 落ちて きて、とげの 金棒の 頭が 真上から ふって きて 画面を うめる
溶岩の ひびが 光る 岩場に 立つ(火の粉)→ ぎざぎざの 帯の カットイン+技名 → しゃがんで とぶ(土けむりの 輪)→
見上げる(地平が 下へ)→ 空の 点が 鬼に なって 落ちて くる → 金棒の 頭(とげの 鉄球)が 大きく なって 画面を うめる → 白 → 暗転
使い方: python3 hs_412.py 出力.mp4 / python3 hs_412.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.2
T_CUT, T_ACT, T_JUMP, T_UP, T_FALL, T_CLUB, T_HIT = 2.7, 4.7, 5.15, 5.5, 6.45, 7.7, 9.3
LAVA = np.array([1.0, 0.45, 0.10], np.float32); HOT = np.array([1.0, 0.85, 0.45], np.float32); SKY0 = np.array([0.07, 0.02, 0.04], np.float32); SKY1 = np.array([0.72, 0.20, 0.07], np.float32); ROCK = np.array([0.13, 0.06, 0.06], np.float32); IRON = np.array([0.20, 0.19, 0.22], np.float32)
EYE = (325, 262); CLUB = (590, 170)
SPK = [(0.0, 0.0), (0.55, 0.05), (-0.50, 0.22), (0.18, 0.56), (-0.22, -0.55), (0.50, -0.42), (-0.58, -0.30), (-0.30, 0.62), (0.62, 0.50)]


class Onimaru(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 412, (1.0, 0.55, 0.3), "金棒落トシ", "Club Drop", W); s.fx_init(412)
        s._vt = [s.text_img(c, hz.SERIF, int(70 * s.k), 3) for c in "金棒落トシ"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 溶岩の ひびが 光る 岩場。hy を 下げると 見上げた 空に なる ----
    def hell(s, t, lum=1.0, hy=0.62):
        W, H, k = s.W, s.H, s.k; v = (s.yn / hy).clip(0, 1); sy = s.yn - (hy - 0.62)                                  # sy: 空の もようを いっしょに 流す
        f = SKY0[None, None, :] * (1 - v ** 1.8)[:, :, None] + SKY1[None, None, :] * (v ** 1.8)[:, :, None]
        n = s.fbm(0.0009, t * 7, (hy - 0.62) * -900, 3, 80 * k, 1); n = cv2.GaussianBlur(n, (0, 0), 4 * k); sm = sstep(0.45, 0.70, n)
        glowu = np.clip(0.2 + 1.5 * (s.fbm(0.003, t * 10, (hy - 0.62) * -900 + 3, 3, 0, 2) - 0.35), 0, 1.2)
        f = f * (1 - sm * 0.8)[:, :, None] + (sm * 0.8)[:, :, None] * (SKY0[None, None, :] * 1.3 + (SKY1 * 0.55)[None, None, :] * (glowu * (0.25 + 0.75 * v))[:, :, None])
        if hy < 1.25:
            rd = hy - 0.015 - 0.16 * s.tex(2, s.XX * 0.33, 5) ** 1.7 - 0.04 * np.abs(((s.xn * 9) % 1.0) - 0.5) * s.tex(1, s.XX * 0.2, 9)                      # ぎざぎざの 岩山
            hm = sstep(-0.003, 0.003, s.yn - rd); hc = ROCK[None, None, :] * (0.5 + 1.3 * s.tex(1, s.XX * 0.8, s.YY * 0.8))[:, :, None] + (SKY1 * 0.25)[None, None, :] * np.clip((s.yn - rd) / 0.12, 0, 1)[:, :, None]
            f = f * (1 - hm)[:, :, None] + hc * hm[:, :, None]
            d = np.clip((s.yn - hy) / (1 - 0.62), 0, 1.5); pu = (s.XX - W / 2) / (d + 0.10) * 0.20; pv = 55.0 / (d + 0.10)
            g1 = s.tex(0, pu, pv); g2 = s.tex(2, pu * 0.5 + 40, pv * 0.5)
            crk = np.exp(-((g2 - 0.5) / 0.035) ** 2) * (0.75 + 0.25 * np.sin(t * 2.5 + g1 * 9))
            gr = ROCK[None, None, :] * (0.45 + 1.5 * g1 * (0.5 + 0.6 * d))[:, :, None] + (crk * 1.5)[:, :, None] * LAVA[None, None, :] + (crk ** 3 * 0.9)[:, :, None] * HOT[None, None, :]
            gr += (cv2.GaussianBlur(crk, (0, 0), 9 * k) * 0.45)[:, :, None] * LAVA[None, None, :]
            m = sstep(hy - 0.003, hy + 0.006, s.yn)[:, :, None]; f = f * (1 - m) + gr * m
            f += (np.exp(-((s.yn - hy) / 0.02) ** 2) * 0.22)[:, :, None] * LAVA[None, None, :]
        return (f * lum).astype(np.float32)

    def embers(s, f, t, v=1.0, fall=0.0):
        W, H = s.W, s.H; r = s.rnd[:140]
        x = ((r[:, 0] + 0.03 * np.sin(t * (1 + r[:, 2]) + r[:, 3] * 6.28)) % 1.0) * W; y = ((r[:, 1] - t * (0.05 + 0.10 * r[:, 4]) * (1 - 6 * fall)) % 1.0) * H
        br = (0.4 + 0.6 * r[:, 2]) * v * (0.6 + 0.4 * np.sin(t * 9 + r[:, 5] * 20)); big = r[:, 4] > 0.6
        s.splat(f, x[~big], y[~big], br[~big] * 1.4, LAVA, 1.0, 0.3); s.splat(f, x[big], y[big], br[big] * 2.0, HOT * np.array([1, 0.8, 0.5], np.float32), 1.8, 0.3)

    # ---- 金棒の 頭(式): 鉄の 球 + ふちの とげ + 手前の とげ ----
    def ball(s, f, c, R, rot, v=1.0):
        if v <= 0.01 or R < 4: return
        k = s.k; dx = (s.XX - c[0]) / R; dy = (s.YY - c[1]) / R; d = np.hypot(dx, dy); z = np.sqrt(np.clip(1 - d * d, 0, 1))
        A = s.L0(); B = np.zeros((s.H, s.W, 3), np.float32)
        lo = tuple(float(x) for x in IRON * 0.6); mid = tuple(float(x) for x in IRON * 1.5); hi = tuple(float(x) for x in (IRON * 1.6 + LAVA * 0.8))
        P = lambda x, y: (int(c[0] + x * R), int(c[1] + y * R))
        for i in range(12):                                                          # ふちの とげ
            a = rot + i * math.pi / 6; ca, sa = math.cos(a), math.sin(a); tip = P(ca * 1.42, sa * 1.42); b0 = P(ca * 0.9 - sa * 0.16, sa * 0.9 + ca * 0.16); b1 = P(ca * 0.9 + sa * 0.16, sa * 0.9 - ca * 0.16); bm = P(ca * 0.9, sa * 0.9)
            cv2.fillPoly(A, [np.array([b0, b1, tip], np.int32)], 1.0, cv2.LINE_AA)
            cv2.fillPoly(B, [np.array([b0, bm, tip], np.int32)], hi if sa > -0.2 else mid, cv2.LINE_AA); cv2.fillPoly(B, [np.array([bm, b1, tip], np.int32)], lo, cv2.LINE_AA)
        lam = np.clip(0.35 + 1.0 * (-dx * 0.35 + dy * 0.75) + 0.75 * z, 0, 1.9); tx = s.tex(1, dx * 110 + rot * 40, dy * 110); t2 = s.tex(0, dx * 300, dy * 300)
        body = IRON[None, None, :] * ((0.35 + 1.0 * tx) * lam)[:, :, None] + (LAVA * 1.5)[None, None, :] * (np.clip(dy * 0.8 + 0.3, 0, 1) ** 2 * (1 - z) ** 1.2)[:, :, None] + (np.exp(-(np.hypot(dx + 0.35, dy + 0.45) / 0.22) ** 2) * 0.40)[:, :, None]
        inb = sstep(1.0, 0.985, d); A = np.maximum(A, inb); B = B * (1 - inb)[:, :, None] + body * inb[:, :, None]
        cr, sr = math.cos(rot * 0.6), math.sin(rot * 0.6)
        for (px, py) in SPK:                                                         # 手前の とげ(こちらを 向く)
            qx, qy = px * cr - py * sr, px * sr + py * cr; rb = 0.15 * (1 - 0.35 * (qx * qx + qy * qy)); tx_, ty_ = qx * 1.34 + 0.02, qy * 1.34 - 0.03
            nx, ny = -(ty_ - qy), (tx_ - qx); nl = math.hypot(nx, ny)
            if nl < 0.02: nx, ny, nl = 1.0, 0.0, 1.0
            nx, ny = nx / nl * rb, ny / nl * rb
            cv2.circle(B, P(qx, qy), int(rb * R), lo, -1, cv2.LINE_AA)
            cv2.fillPoly(B, [np.array([P(qx - nx, qy - ny), P(qx, qy + rb * 0.9), P(tx_, ty_)], np.int32)], hi, cv2.LINE_AA); cv2.fillPoly(B, [np.array([P(qx, qy + rb * 0.9), P(qx + nx, qy + ny), P(tx_, ty_)], np.int32)], mid, cv2.LINE_AA)
            cv2.fillPoly(B, [np.array([P(qx - nx, qy - ny), P(qx, qy - rb * 0.9), P(tx_, ty_)], np.int32)], mid, cv2.LINE_AA); cv2.fillPoly(B, [np.array([P(qx, qy - rb * 0.9), P(qx + nx, qy + ny), P(tx_, ty_)], np.int32)], lo, cv2.LINE_AA)
            cv2.circle(B, P(tx_, ty_), max(1, int(R * 0.012)), tuple(float(x) for x in HOT * 0.9), -1, cv2.LINE_AA)
        B = cv2.GaussianBlur(B, (0, 0), max(0.6, R * 0.004)) * (0.72 + 0.5 * t2)[:, :, None]
        A = cv2.GaussianBlur(A, (0, 0), max(0.7, R * 0.004)) * v
        f[:] = f * (1 - A)[:, :, None] + B * A[:, :, None]

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(LAVA), rim_dir=(0.0, 1.0), rim_v=0.7 * alpha, amb=(1.0, 0.75, 0.65), amb_v=0.14, aura=LAVA, aura_v=0.08, alpha=alpha, shade=0.9)

    def frame(s, t):
        o = {}
        if t < T_CUT: f, sh = s.sc_stand(t)
        elif t < T_ACT: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.5, thr=0.95)
        else: f, sh = s.sc_drop(t)
        f, sh = s.finale(f, sh, t, DUR, None, (1.0, 0.9, 0.75), (1.0, 0.6, 0.3), -1)
        return s.film(f, t, sh * s.k, 0.45, o.get("thr", 0.9), (1.0, 0.93, 0.88), 1.0 + 0.15 * min(sh, 8), 0.03, o.get("vig", 0.5), 1.2, 0.96)

    def oni(s, f, t, alpha=1.0, dy=0.0, **kw):
        W, H, k = s.W, s.H, s.k; cy = H * 0.575 + dy; fy = H * 0.575 + 0.44 * H * 0.47
        sd = np.exp(-(((s.XX - W / 2) / (W * 0.24)) ** 2 + ((s.YY - fy) / (H * 0.016)) ** 2)) * math.exp(dy / (H * 0.25)); f *= (1 - 0.6 * alpha * sd)[:, :, None]
        img = s.pose(t, **kw); s.draw(f, img, W / 2, cy, 0.44, alpha); return fy

    # 1) 岩場に 立つ
    def sc_stand(s, t):
        f = s.hell(t); s.oni(f, t, clamp(t / 0.6), sway=0.3, breath=0.6); s.embers(f, t, 0.9)
        return f * clamp(t / 0.5), 0

    # 2) 目の カットイン(とげの ように ぎざぎざの 帯)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.hell(t, 0.32); s.embers(f, t, 0.5)
        a = ease(u / 0.26); cy = H * 0.33; R = H * 0.115 * a
        band = (ROCK * 1.2)[None, None, :] + (LAVA * 0.75)[None, None, :] * (0.25 + 0.9 * sstep(0.45, 0.9, s.tex(0, s.XX * 0.3 + t * 30, s.YY * 0.5 - t * 80)))[:, :, None]
        ex, ey = s.A(*EYE); zf = 1.55; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(band, img, W / 2 - (ex - s.aw / 2) * sc, cy - (ey - s.ah / 2) * sc, zf)
        yy = (s.YY - cy) + (s.XX - W / 2) * 0.10; saw = np.abs(((s.XX / (46 * k) + t * 0.6) % 1.0) - 0.5) * 2 * 30 * k                                          # ぎざぎざ
        d = np.abs(yy) - R - saw * (R > 4)
        m = (sstep(0, -2, d) * (R > 4))[:, :, None]; f = f * (1 - m) + band * m
        s.glowmask(f, np.exp(-(d / (2.2 * k)) ** 2) * (R > 4), HOT, 0.9 * a, 1.0, 0.8, 7.0)
        n = len(s._vt); step = W * 0.172
        for i, T in enumerate(s._vt):                                                # 技名: 1文字ずつ 上から 落ちて 止まる
            p = (u - 0.45 - i * 0.13) / 0.22
            if p > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.69 - (1 - min(1.0, p) ** 2) * 120 * k + (6 * k * math.exp(-(p - 1) * 5) * math.sin((p - 1) * 14) if p > 1 else 0), clamp(p * 2), 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.3 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.69 - 74 * k, 1.0, 1.0)
        if u > 1.4:
            g = ease((u - 1.4) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.69 + 56 * k), (W / 2 + W * 0.42 * g, H * 0.69 + 56 * k)], HOT, 1.0, 0.7, 0.7)
            s.text(f, s.txt_en, W / 2, H * 0.69 + 82 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.5
        sh = 3 * math.exp(-u * 6)
        for i in range(n):
            q = u - 0.45 - i * 0.13 - 0.22
            if 0 < q < 0.2: sh += 4 * (1 - q / 0.2)
        return f, sh

    # 3) とぶ → 見上げる → 落ちて くる → 金棒の 頭が うめる
    def sc_drop(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_ACT; sh = 0.0
        hy = 0.62 + 0.95 * ease2(clamp((t - T_UP) / (T_FALL - T_UP)))
        p = clamp((t - T_FALL) / (T_HIT - T_FALL)); e = clamp((t - T_CLUB) / (T_HIT - T_CLUB))
        f = s.hell(t, 1.0 - 0.3 * e, hy)
        if t < T_UP + 0.5:
            j = t - T_JUMP
            if j < 0: s.oni(f, t, 1.0, 0.0, sway=0.2, breath=0.4, squash=0.10 * ease(u / (T_JUMP - T_ACT)))
            else:
                dyj = -H * (3.2 * j + 6.0 * j * j) + (hy - 0.62) * H
                if dyj > -H * 1.3: s.oni(f, t, 1.0, dyj, sway=0.0, breath=0.0, squash=-0.10)
                fy = H * (hy - 0.62) + H * 0.575 + 0.44 * H * 0.47; L = s.L0(); q = clamp(j / 0.5)                           # 土けむりの 輪
                cv2.ellipse(L, (W // 2, int(fy)), (int(W * (0.08 + 0.5 * q)), int(H * (0.008 + 0.05 * q))), 0, 0, 360, float((1 - q) ** 1.3), max(1, int(7 * k)), cv2.LINE_AA)
                s.glowmask(f, cv2.GaussianBlur(L, (0, 0), 3 * k) * (0.6 + 0.6 * s.tex(2, s.XX * 0.8, s.YY * 0.8)), (1.0, 0.75, 0.55), 1.2, 1.0, 0.6, 7.0)
                if j < 0.12: f = f + (1 - j / 0.12) * 0.4
                sh += 9 * math.exp(-j * 5)
        s.embers(f, t, 0.9 * (1 - 0.6 * p), 0.5 * clamp((t - T_UP) / 0.6))
        if t > T_FALL - 0.5:                                                         # 空の 点 → 鬼が 落ちて くる
            C0 = (W * 0.5, H * (0.20 + 0.20 * p))
            dxp = s.XX - C0[0]; dyp = s.YY - C0[1]; dist = np.hypot(dxp, dyp) + 1e-3; th = np.arctan2(dyp, dxp); lg = np.log(dist / k + 5.0)
            stv = sstep(0.60, 0.95, s.tex(0, th / 6.2832 * 1536, lg * 14 - t * 120)) * np.clip(dist / (W * 0.5), 0, 1)
            f += (stv * (0.10 + 0.45 * p))[:, :, None] * LAVA[None, None, :]                                               # 落ちる 勢いの 筋
            if t < T_FALL:
                q = (t - (T_FALL - 0.5)) / 0.5; s.glowmask(f, np.exp(-(dist / (5 * k)) ** 2), HOT, q * (0.6 + 0.4 * math.sin(t * 40)), 1.0, 0.8, 4.0)                   # きらり
            else:
                hf = 0.04 + 0.85 * p ** 2.1; img = s.pose(t, breath=0.0, sway=0.0)
                s.draw(f, img, C0[0], C0[1] + hf * H * 0.12, hf, min(1.0, p * 8))
                sh += 1 + 5 * p ** 2
                if e > 0:
                    cp0 = s.at(*s.A(*CLUB)); ee = e ** 2.3; g = ease2(clamp(e * 1.6)); c = (cp0[0] + (W * 0.5 - cp0[0]) * g, cp0[1] + (H * 0.46 - cp0[1]) * g)
                    R = hf * H * 0.13 * (1 - g * 0.0) + W * 1.25 * ee
                    sdw = np.exp(-(dist * 0 + np.hypot(s.XX - c[0], s.YY - c[1]) / (R * 1.5 + 1)) ** 2); f *= (1 - 0.5 * sdw * e)[:, :, None]
                    s.ball(f, c, R, 0.4 + 1.3 * e, min(1.0, e * 8)); sh += 10 * ee
                    if e > 0.86: f = f + ((e - 0.86) / 0.14) ** 2 * 1.8
        if u < 0.12: f = f + (1 - u / 0.12) * 0.25
        return f, sh

    # ---- この子だけの曲: 低い 太鼓の どん・どん → ふみきり → 無音の 間(風だけ)→ 落ちて くる うなり → 大太鼓 ----
    def audio(s, path):
        a = Snd(DUR, 412); G = 110.0; pn = [0, 1, 5, 7, 8]
        def fq(d, o=0): return G * 2 ** ((pn[d % 5] + 12 * (d // 5 + o)) / 12.0)
        for d in (0, 3): a.note(fq(d, -1), 0.1, 2.6, 0.045, "pad", 0.7)
        for i, tk in enumerate((0.5, 1.1, 1.7, 2.0, 2.3)): a.boom(tk, 62 - 3 * (i % 2), 0.5, 0.35 + 0.05 * i)
        a.note(fq(5, 0), 0.6, 1.6, 0.06, "fm"); a.note(fq(7, 0), 1.7, 0.9, 0.06, "fm")
        a.boom(T_CUT, 55, 0.7, 0.45); a.whoosh(T_CUT - 0.3, T_CUT, 0.2, 0.1)
        for i in range(5): a.boom(T_CUT + 0.67 + i * 0.13, 80 - i * 4, 0.25, 0.3); a.note(fq(5 + i, 0), T_CUT + 0.67 + i * 0.13, 0.4, 0.06, "bell")
        for d in (0, 2, 3): a.note(fq(d, 0), T_CUT, 1.9, 0.04, "pad", 0.05)
        a.rise(T_ACT, T_JUMP, 80, 200, 0.07); a.boom(T_JUMP, 45, 1.2, 0.8); a.whoosh(T_JUMP, T_JUMP + 0.5, 0.8, 0.2)
        wn = a.lp(a.nz, 0.20) - a.lp(a.nz, 0.03); lo = a.lp(a.nz, 0.02)
        a.a += wn * np.clip((a.t - T_UP) / 0.5, 0, 1) * (a.t < T_FALL) * 0.35                                               # 見上げる 間: 風だけ
        a.note(fq(14, 2), T_FALL - 0.45, 0.5, 0.05, "bell")                                                              # きらり
        g = np.clip((a.t - T_FALL) / (T_HIT - T_FALL), 0, 1) ** 2 * (a.t < T_HIT)
        a.a += wn * g * 1.6 + lo * g * 3.0
        a.rise(T_FALL, T_HIT, 520, 70, 0.10); a.rise(T_CLUB, T_HIT, 60, 30, 0.12)
        nb = 9
        for i in range(nb): tk = T_FALL + (T_HIT - T_FALL) * (1 - (1 - i / nb) ** 1.8); a.boom(tk, 70, 0.3, 0.2 + 0.05 * i)   # だんだん つまる 太鼓
        for j, d in enumerate((0, 1, 5)): a.note(fq(d, 0), T_CLUB + j * 0.06, T_HIT - T_CLUB + 0.1, 0.05, "pad", 0.4)
        a.boom(T_HIT, 36, 1.8, 1.0); a.boom(T_HIT + 0.02, 70, 0.5, 0.5); a.a += (a.nz - a.lp(a.nz, 0.4)) * a.env(T_HIT, 0.002, 0.25) * 0.5
        for j, d in enumerate((0, 3, 5)): a.note(fq(d, 1), T_HIT + 0.03 + j * 0.03, 0.8, 0.05, "bell")
        s.fin_snd(a, DUR, 412); a.save(path)


if __name__ == "__main__":
    r = Onimaru()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
