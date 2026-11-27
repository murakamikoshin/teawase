"""ゼフィ(★4 疾風の影) 専用の必殺技「疾風ノ影斬リ」— 高品質版(hs_fx)。草原を ジグザグに 駆けて 近づき、目の前を 影が よぎって、おくれて 一すじ 斬られる
月夜の 草原、風に なびく 草。低く かまえる → 目の カットイン+技名 →
消える: 奥を 右から 左へ(草が 一すじ 分かれる)→ 中ほどを 左から 右へ → 手前を 右から 左へ(だんだん 大きい)→ しずけさ →
黒い 影が 画面いっぱいに よぎる → 一拍 おいて 画面に 光の 一すじ → 上下が ずれて、すき間の 光と 草の 切れはしの 突風に のまれる → 白 → 暗転
使い方: python3 hs_250.py 出力.mp4 / python3 hs_250.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.1
T_CUT, T_DASH, T_SHADOW, T_LINE, T_HIT = 2.7, 4.7, 7.35, 8.05, 9.4
PASS = [(5.05, 0.50, 0.625, 0.15, -1), (5.75, 0.45, 0.735, 0.30, 1), (6.40, 0.38, 0.880, 0.58, -1)]     # (はじまり, 長さ, 画面の 高さ, キャラの 大きさ, 向き)
GREEN = np.array([0.20, 0.62, 0.40], np.float32); MOON = np.array([0.72, 0.86, 0.95], np.float32); NIGHT = np.array([0.02, 0.06, 0.10], np.float32); PALE = np.array([0.85, 1.0, 0.92], np.float32)
EYE = (370, 276); TIP = (19, 493)


class Zephy(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 250, (0.5, 1.0, 0.75), "疾風ノ影斬リ", "Gale Shadow Cut", W); s.fx_init(250)
        s._vt = [s.text_img(c, hz.SERIF, int(58 * s.k), 3) for c in "疾風ノ影斬リ"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; s.hy = 0.56
        rg = np.random.RandomState(2500); s._cl = rg.rand(3, 220, 5).astype(np.float32); s._lf = rg.rand(320, 6).astype(np.float32); s._wd = rg.rand(260, 4).astype(np.float32)

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 月夜の 草原 ----
    def field(s, t, wind=1.0, lum=1.0):
        W, H, k = s.W, s.H, s.k; hy = s.hy
        f = NIGHT[None, None, :] * (0.6 + 2.4 * (s.yn / hy).clip(0, 1) ** 1.6)[:, :, None] + np.array([0.10, 0.26, 0.28], np.float32)[None, None, :] * np.exp(-((s.yn - hy) / 0.13) ** 2)[:, :, None]
        s.starfield(f, t, 500, (0.85, 0.95, 1.0), 2, 0.7, (0.0, 0.0))
        cl = s.tex(0, s.XX * 0.18 + t * 30 * wind, s.YY * 0.9) * 0.6 + s.fbm(0.002, t * 60 * wind, 0, 3, 0, 1) * 0.5                                # 風に 流れる 雲
        f += (sstep(0.52, 0.85, cl) * np.exp(-((s.yn - 0.26) / 0.20) ** 2) * 0.30)[:, :, None] * MOON[None, None, :]
        mt = hy - 0.05 - 0.035 * np.sin(s.xn * 5.0 + 1.0) - 0.02 * np.sin(s.xn * 11.0); mm = sstep(mt - 0.003, mt + 0.003, s.yn) * (s.yn < hy + 0.01)
        f = f * (1 - mm)[:, :, None] + np.array([0.02, 0.05, 0.07], np.float32)[None, None, :] * mm[:, :, None]
        d = np.clip((s.yn - hy) / (1 - hy), 0, 1); z = 1.0 / (d + 0.09); sw = np.sin(t * 2.6 * wind + s.xn * 5 + d * 4) * 5 * wind + np.sin(t * 6.1 * wind + s.xn * 17) * 1.5 * wind
        u = (s.xn - 0.5) * z * 260 + sw * z * 1.2; v = z * 5.5
        b = s.tex(1, u, v) * 0.6 + s.tex(3, u * 2.3 + 50, v * 2.0) * 0.4; gust = 0.75 + 0.5 * s.tex(0, s.XX * 0.2 - t * 70 * wind, s.YY * 0.5)             # 風の 波
        g = np.array([0.012, 0.045, 0.040], np.float32)[None, None, :] + (GREEN * 0.55)[None, None, :] * (b ** 1.6 * (0.35 + 0.9 * d) * gust)[:, :, None] + (MOON * 0.30)[None, None, :] * (sstep(0.72, 0.95, b) * gust * (0.4 + 0.6 * d))[:, :, None]
        gm = sstep(hy - 0.004, hy + 0.004, s.yn + (b - 0.5) * 0.012)[:, :, None]
        s._b = b
        return ((f * (1 - gm) + g * gm) * lum).astype(np.float32)

    def wake(s, f, t):
        """駆けぬけた あと: 草が 一すじ 分かれ、切れはしが 舞う"""
        W, H, k = s.W, s.H, s.k
        for j, (t0, dur, yj, size, dr) in enumerate(PASS):
            p = (t - t0) / dur
            if p <= 0: continue
            xh = W * (0.5 - dr * (0.75 - 1.5 * min(p, 1.0))) if True else 0; behind = sstep(-8 * k, 8 * k, (xh - s.XX) * dr) if p < 1 else 1.0
            age = max(0.0, t - t0 - dur); wj = 0.010 + 0.030 * size / 0.58
            lane = np.exp(-((s.yn - yj - size * 0.20) / wj) ** 2) * behind * math.exp(-age * 0.35)
            f *= (1 - 0.65 * lane)[:, :, None]; f += (np.exp(-((s.yn - yj - size * 0.20 + wj * 1.3) / (wj * 0.35)) ** 2) * behind * math.exp(-age * 1.2) * 0.35 * (0.5 + s._b))[:, :, None] * PALE[None, None, :]
            c = s._cl[j]; xb = W * (0.5 - dr * (0.75 - 1.5 * c[:, 0])); tb = t0 + c[:, 0] * dur; a = t - tb; ok = (a > 0) & (a < 1.6)                     # 切れはし
            x = xb + dr * a * W * (0.10 + 0.25 * c[:, 1]) * -1 + np.sin(a * 6 + c[:, 2] * 9) * 10 * k; y = H * (yj + size * 0.20) - (a * (0.5 + c[:, 3]) - a * a * 0.45) * H * 0.22 * (0.4 + size)
            s.splat(f, x[ok], y[ok], (0.4 + c[ok, 4]) * (1 - a[ok] / 1.6) * (0.5 + size), GREEN * 0.6 + PALE * 0.4, 1.0 + 1.5 * size, 0.3)

    def streak(s, f, img, cx, cy, size, sx, gain=1.5):
        fc = f.copy(); s.draw(fc, img, cx, cy, size); d = cv2.GaussianBlur(fc - f, (0, 0), sigmaX=max(1.0, sx), sigmaY=max(0.6, sx * 0.03)); f += d * gain

    def windlines(s, f, t, v=1.0, speed=1.0):
        W, H, k = s.W, s.H, s.k; w = s._wd; B = np.zeros((H, W), np.float32)
        x = ((w[:, 0] + t * speed * (0.8 + 1.2 * w[:, 2])) % 1.3 - 0.15) * W; y = (w[:, 1] * 0.95 + 0.02 * np.sin(t * 3 + w[:, 3] * 20)) * H
        xi = x.astype(int); yi = y.astype(int); ok = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H); np.add.at(B, (yi[ok], xi[ok]), (0.4 + w[ok, 3]) * v)
        f += (cv2.GaussianBlur(B, (0, 0), sigmaX=38 * k, sigmaY=0.7 * k) * 26)[:, :, None] * PALE[None, None, :]

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(MOON), rim_dir=(0.3, -0.95), rim_v=0.55 * alpha, amb=(0.6, 0.9, 0.85), amb_v=0.18, aura=None, aura_v=0.0, alpha=alpha, shade=0.78)

    def frame(s, t):
        o = {}
        if t < T_CUT: f, sh = s.sc_field(t)
        elif t < T_DASH: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.92)
        else: f, sh = s.sc_dash(t)
        f, sh = s.finale(f, sh, t, DUR, None, (0.7, 1, 0.85), (0.4, 0.9, 0.7), -1)
        return s.film(f, t, sh * s.k, 0.5, o.get("thr", 0.84), (0.85, 1.0, 0.93), 1.0 + 0.15 * min(sh, 8), 0.04, o.get("vig", 0.58), 1.15, 1.12)

    def crouch(s, f, t, alpha=1.0, **kw):
        W, H, k = s.W, s.H, s.k; cy = H * 0.66; fy = cy + 0.40 * H * 0.40
        sh_ = s.L0(); cv2.ellipse(sh_, (int(W / 2), int(fy)), (int(W * 0.20), int(H * 0.012)), 0, 0, 360, 0.5 * alpha, -1, cv2.LINE_AA); f *= (1 - cv2.GaussianBlur(sh_, (0, 0), 4 * k))[:, :, None]
        img = s.pose(t, **kw); s.draw(f, img, W / 2, cy, 0.40, alpha)

    # 1) 月夜の 草原。低く かまえる
    def sc_field(s, t):
        W, H, k = s.W, s.H, s.k
        f = s.field(t, 0.8); s.crouch(f, t, 1.0, sway=0.45, breath=0.4)
        if t > 1.5: s.flare(f, s.at(*s.A(*TIP)), (6 + 10 * math.exp(-(t - 1.5) * 4)) * k, PALE, clamp((t - 1.5) * 5), 0.9, 0.3, False)     # 切っ先の 光
        s.windlines(f, t, 0.15, 0.5)
        return f * clamp(t / 0.5), 0

    # 2) 目の カットイン(ななめの 切り口の 帯)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.field(t, 0.8, 0.32)
        a = ease(u / 0.22); cy = H * 0.33; hh = H * 0.085; sl = -0.16
        band = np.array([0.02, 0.07, 0.07], np.float32)[None, None, :] + (GREEN * 0.5)[None, None, :] * (0.3 + 0.9 * s.tex(1, s.XX * 0.08 - t * 260, s.YY * 2.0))[:, :, None]
        ex, ey = s.A(*EYE); zf = 1.7; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(band, img, W / 2 - (ex - s.aw / 2) * sc + (1 - a) * W * 1.1 - u * 8 * k, cy - (ey - s.ah / 2) * sc, zf)
        dd = (s.YY - cy) + (s.XX - W / 2) * sl; lim = W * 0.56 * a
        m = (sstep(hh, hh - 2, np.abs(dd)) * sstep(lim, lim - 30 * k, np.abs(s.XX - W / 2 - (s.YY - cy) * 0.9)))[:, :, None]
        f = f * (1 - m) + band * m
        for sg in (-1, 1): s.glowline(f, [(W / 2 - lim, cy + sg * hh + lim * sl), (W / 2 + lim, cy + sg * hh - lim * sl)], PALE, 1.2, a, 0.9)
        n = len(s._vt); step = W * 0.148
        for i, T in enumerate(s._vt):                                                # 技名: 横から すべりこむ(はやい)
            p = (u - 0.45 - i * 0.07) / 0.12
            if p > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step + (1 - ease(p)) * 120 * k, H * 0.66, clamp(p * 2), 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.3 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.66 - 66 * k, 1.0, 1.0)
        if u > 1.1:
            g = ease((u - 1.1) / 0.3); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.66 + 50 * k), (W / 2 + W * 0.42 * g, H * 0.66 + 50 * k)], PALE, 1.0, 0.8, 0.8)
            s.text(f, s.txt_en, W / 2, H * 0.66 + 76 * k, g, 1.0)
        s.windlines(f, t, 0.2, 1.2)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.7
        return f, 5 * math.exp(-u * 6)

    # 3) 消える → 奥・中・手前と 駆けて 近づく → 影が よぎる → おくれて 一すじ
    def sc_dash(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_DASH; us = t - T_SHADOW; ul = t - T_LINE; sh = 0.0
        calm = 1.0 if us < 0.3 or ul > 0 else 0.25
        f = s.field(t, (1.2 + 0.6 * clamp(u / 0.4)) * calm + (2.5 * clamp(ul / 0.3) if ul > 0 else 0), 1.0)
        s.wake(f, t)
        if u < 0.36:                                                                 # ためて 消える
            g = u / 0.36; img = s.pose(t, sway=0.3, lean=-10 * k * g, squash=0.08 * g)
            if g < 0.75: s.draw(f, img, W / 2, H * 0.66, 0.40)
            else: q = (g - 0.75) / 0.25; s.streak(f, img, W / 2 - W * 0.9 * q, H * 0.66, 0.40, 60 * k * q + 2, 1.5 * (1 - q))
        for j, (t0, dur, yj, size, dr) in enumerate(PASS):                          # 駆けぬける 姿(ぶれ)
            p = (t - t0) / dur
            if 0 < p < 1:
                img = s.pose(t, lean=-8 * k); img = img if dr < 0 else np.ascontiguousarray(img[:, ::-1])
                xh = W * (0.5 - dr * (0.75 - 1.5 * p)); s.streak(f, img, xh - dr * size * H * 0.10, H * yj - size * H * 0.12, size, (10 + 55 * size) * k, 1.1); s.streak(f, img, xh, H * yj - size * H * 0.12, size, (2 + 7 * size) * k, 1.0)
                tip = (xh + dr * size * H * 0.42, H * yj + size * H * 0.02); s.flare(f, tip, (5 + 14 * size) * k, PALE, 0.9, 1.4, 0.0, False); sh += 3 + 8 * size
        s.windlines(f, t, 0.18 + 0.25 * (u < 2.2), 1.0 + 1.0 * (u < 2.2))
        if 0 < us < 0.42:                                                            # 黒い 影が 画面いっぱいに よぎる
            p = us / 0.42; xc = W * (-0.6 + 2.2 * p); edge = (s.tex(2, s.YY * 0.9, 3.0) - 0.5) * W * 0.5
            m = sstep(W * 0.75, W * 0.25, np.abs(s.XX - xc - edge)) * (0.75 + 0.25 * sstep(0.1, 0.3, s.yn))
            f *= (1 - 0.97 * m)[:, :, None]; s.glowline(f, [(xc - W * 0.9, H * 0.48), (xc + W * 0.5, H * 0.46)], PALE, 1.0, 0.8 * math.sin(p * 3.14), 0.6); sh += 10
        if ul > 0:                                                                   # おくれて 一すじ → 上下が ずれる → すき間の 光と 突風
            y0 = H * 0.47; sl = -0.07; ln = s.YY - y0 - (s.XX - W / 2) * sl; g = ease(ul / 0.10)
            e = clamp((ul - 0.30) / (T_HIT - T_LINE - 0.30)); gap = H * (0.004 + 0.75 * e ** 2.2) * (ul > 0.30); sl_ = 46 * k * ease2(e * 3) + W * 0.5 * e ** 2
            up = cv2.warpAffine(f, np.float32([[1, 0, -sl_], [0, 1, -gap]]), (W, H), borderMode=cv2.BORDER_REPLICATE); dn = cv2.warpAffine(f, np.float32([[1, 0, sl_], [0, 1, gap]]), (W, H), borderMode=cv2.BORDER_REPLICATE)
            mu = sstep(-gap + 1, -gap - 1, ln)[:, :, None]; md = sstep(gap - 1, gap + 1, ln)[:, :, None]
            light = (PALE * (1.3 + 1.0 * e))[None, None, :] * (0.7 + 0.5 * s.tex(1, s.XX * 0.05 - t * 400, s.YY * 1.5))[:, :, None]
            f = up * mu + dn * md + light * (1 - mu - md).clip(0, 1)
            vis = sstep(W * (-0.1 + 1.3 * g) + 20 * k, W * (-0.1 + 1.3 * g) - 20 * k, s.XX)
            f += (np.exp(-((np.abs(ln) - gap) / (2.2 * k)) ** 2) * vis * (1.6 * math.exp(-ul * 2) + 0.6))[:, :, None] * PALE[None, None, :] + (np.exp(-((np.abs(ln) - gap) / (14 * k)) ** 2) * vis * 0.5 * math.exp(-ul * 1.5))[:, :, None] * GREEN[None, None, :]
            lf = s._lf; a = ul * (1.2 + 1.5 * lf[:, 2]) + lf[:, 0]; x = (a % 1.3 - 0.15) * W; y = (lf[:, 1] + 0.05 * np.sin(a * 9 + lf[:, 3] * 9)) * H; L = s.L0()        # 草の 切れはしの 突風
            for i in range(len(lf)):
                sz = (3 + 9 * lf[i, 4] * (0.5 + 1.5 * e)) * k; cv2.ellipse(L, (int(x[i]), int(y[i])), (max(1, int(sz * 2.2)), max(1, int(sz * 0.35))), float(lf[i, 5] * 50 - 25 + math.sin(a[i] * 12) * 20), 0, 360, float(0.55 + 0.4 * lf[i, 3]), -1, cv2.LINE_AA)
            L = cv2.GaussianBlur(L, (0, 0), sigmaX=3 * k, sigmaY=0.6 * k) * min(1.0, ul * 4); f = f * (1 - 0.5 * np.clip(L, 0, 1))[:, :, None] + (np.clip(L, 0, 1) * 0.8)[:, :, None] * (GREEN * 0.7 + PALE * 0.3)[None, None, :]
            s.windlines(f, t, 1.2, 3.0)
            if ul < 0.08: f = f + (1 - ul / 0.08) * 0.9
            sh += 12 * math.exp(-ul * 4) + 3 + 8 * e
            if e > 0.80: f = f + ((e - 0.80) / 0.20) ** 2 * 1.4
        return f, sh

    # ---- この子だけの曲: 風の 音と 尺八ふうの 一音 → 太鼓の きざみ → 3回の 風切り(だんだん 近い)→ 無音 → 低い よぎり → 無音 → きん、と 一閃 → 突風 ----
    def audio(s, path):
        a = Snd(DUR, 250); E = 164.81; pn = [0, 2, 3, 7, 8]                                                           # 都節ふうの 5音
        def fq(d, o=0): return E * 2 ** ((pn[d % 5] + 12 * (d // 5 + o)) / 12.0)
        wind = (a.lp(a.nz, 0.10) - a.lp(a.nz, 0.02)) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.4 * a.t + 1) * np.sin(2 * np.pi * 0.13 * a.t))
        a.a += wind * np.clip(a.t / 0.8, 0, 1) * a.gate(0, 6.9, 0.2) * 1.1
        for (d, o, t0, ln) in ((2, 1, 0.5, 1.0), (3, 1, 1.4, 0.5), (2, 1, 1.9, 0.7)):                                 # 笛
            a.a += (np.sin(2 * np.pi * fq(d, o) * a.t + 2.5 * np.sin(2 * np.pi * 5 * a.t) * np.clip((a.t - t0) / ln, 0, 1)) + 0.25 * a.lp(a.nz, 0.3)) * a.env(t0, 0.12, ln * 0.6) * 0.09
        a.boom(T_CUT, 50, 1.0, 0.4); a.whoosh(T_CUT - 0.25, T_CUT, 0.4, 0.4)
        for i in range(6): a.note(fq(i, 1), T_CUT + 0.45 + i * 0.07, 0.2, 0.09, "fm"); a.hat(T_CUT + 0.45 + i * 0.07, 0.06)
        for i in range(10): a.boom(T_CUT + 1.0 + i * 0.1 * (1 - i * 0.03), 80, 0.25 + 0.03 * i, 0.06)                   # 太鼓の きざみ
        a.whoosh(T_DASH, T_DASH + 0.36, 0.6, 0.5); a.boom(T_DASH + 0.34, 60, 0.7, 0.2)
        for j, (t0, dur, yj, size, dr) in enumerate(PASS):
            g = 0.5 + 1.2 * size; a.a += (a.nz - a.lp(a.nz, 0.35)) * np.exp(-((a.t - t0 - dur / 2) / (dur * 0.28)) ** 2) * 0.5 * g + a.lp(a.nz, 0.15) * np.exp(-((a.t - t0 - dur / 2) / (dur * 0.3)) ** 2) * 0.5 * g
            a.note(fq(4 - j, 2), t0 + dur * 0.5, 0.25, 0.06 + 0.03 * j, "bell"); a.boom(t0 + dur * 0.5, 70 - 8 * j, 0.4 + 0.5 * size, 0.15)
        a.a += a.lp(a.nz, 0.03) * np.exp(-((a.t - T_SHADOW - 0.21) / 0.13) ** 2) * 2.4; a.boom(T_SHADOW + 0.2, 34, 1.5, 0.5)                  # 影が よぎる
        a.note(fq(9, 2), T_LINE, 1.6, 0.16, "bell"); a.note(fq(9, 2) * 1.5, T_LINE, 0.9, 0.07, "sine"); a.a += (a.nz - a.lp(a.nz, 0.8)) * a.env(T_LINE, 0.001, 0.05) * 0.7   # きん
        tg = T_LINE + 0.30; a.boom(tg, 40, 1.4, 0.7)
        a.a += ((a.lp(a.nz, 0.2) - a.lp(a.nz, 0.03)) * 1.2 + (a.nz - a.lp(a.nz, 0.5)) * 0.25) * np.clip((a.t - tg) / 0.15, 0, 1) * (a.t < T_HIT) * (0.7 + 0.3 * np.sin(2 * np.pi * 3 * a.t))
        for j, d in enumerate((0, 2, 3, 5, 7)): a.note(fq(d, 0), tg + j * 0.05, T_HIT - tg + 0.2, 0.045, "pad", 0.3)
        a.rise(tg, T_HIT, 100, 380, 0.11)
        a.boom(T_HIT, 38, 1.5, 0.8)
        for j, d in enumerate((0, 2, 5, 7)): a.note(fq(d, 1), T_HIT + 0.02, 0.6, 0.06, "bell")
        s.fin_snd(a, DUR, 250); a.save(path)


if __name__ == "__main__":
    r = Zephy()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
