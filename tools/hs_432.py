"""パティシエル(★5 菓子の国の女王) 専用の必殺技「女王ノ晩餐」— 高品質版(hs_fx)。白い 食卓が こちらへ のびて 皿と ろうそくが ならび、上から 銀の ふた(クロッシュ)が 降りて きて かぶさり、まっ暗の 中で ベルが 鳴って ひらく
晩餐の 広間(赤い 幕・シャンデリア・市松の 床)に 立つ → ふちが 波うつ レースの 丸窓の カットイン+技名 →
泡立て器を かかげる(クリームの リボン)→ 食卓が 手前へ のびる(皿・ろうそくが ぽんぽん)→ 銀の ふたが 上から 降りて 大きく なり 画面を おおう →
暗やみ・下の すき間の 光が とじる → ちーん(金の 星)→ ふたが ひらく 光 → 白 → 暗転
使い方: python3 hs_432.py 出力.mp4 / python3 hs_432.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.9
T_CUT, T_ACT, T_TABLE, T_LID, T_DARK, T_BELL, T_HIT = 2.7, 4.7, 5.0, 6.7, 8.5, 9.25, 9.95
VELV = np.array([0.50, 0.08, 0.16], np.float32); GOLD = np.array([1.0, 0.82, 0.42], np.float32); CREAM = np.array([1.0, 0.95, 0.86], np.float32); PINK = np.array([1.0, 0.70, 0.74], np.float32); WHITE_ = np.array([1.0, 0.99, 0.96], np.float32)
SILV = np.array([0.78, 0.82, 0.90], np.float32); WARM = np.array([1.0, 0.86, 0.60], np.float32)
EYE = (360, 320); WHISK = (70, 190)


class Patissiel(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 432, (1.0, 0.75, 0.8), "女王ノ晩餐", "Queen's Banquet", W); s.fx_init(432)
        s._vt = [s.text_img(c, hz.SERIF, int(70 * s.k), 3) for c in "女王ノ晩餐"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; s.hy = 0.56

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 晩餐の 広間 ----
    def hall(s, t, lum=1.0):
        W, H, k = s.W, s.H, s.k; hy = s.hy
        fold = 0.5 + 0.5 * np.sin(s.XX / (17 * k) + 1.4 * np.sin(s.XX / (53 * k)) + 0.25 * np.sin(s.YY / (90 * k) + t * 0.5))                                   # 幕の ひだ
        f = VELV[None, None, :] * (0.38 + 0.95 * fold ** 1.6 + 0.12 * s.tex(1, s.XX * 0.6, s.YY * 0.15))[:, :, None] * (0.55 + 0.6 * (s.yn / hy).clip(0, 1))[:, :, None]
        f += (PINK * 0.25)[None, None, :] * (fold ** 6)[:, :, None]
        tr = np.exp(-((s.yn - 0.085 - 0.02 * np.abs(np.sin(s.xn * 9.42))) / 0.010) ** 2); f = f * (1 - tr * 0.9)[:, :, None] + (tr * (0.7 + 0.5 * s.tex(0, s.XX * 2, 3)))[:, :, None] * GOLD[None, None, :]      # 金の ふさ
        for cxh, cyh, sz in ((0.22, 0.20, 1.0), (0.78, 0.20, 1.0), (0.5, 0.13, 1.3)):                                                                          # シャンデリア
            d = np.hypot(s.xn - cxh, (s.yn - cyh) * H / W); th = np.arctan2((s.yn - cyh) * H / W, s.xn - cxh)
            f += (np.exp(-d / (0.10 * sz)) * 0.55 + np.exp(-d / (0.02 * sz)) * 1.0 + np.exp(-d / (0.06 * sz)) * np.abs(np.cos(th * 4 + t * 0.3)) ** 24 * 0.8 * (0.8 + 0.2 * math.sin(t * 5 + cxh * 9)))[:, :, None] * WARM[None, None, :]
        d = np.clip((s.yn - hy) / (1 - hy), 0, 1); pu = (s.XX - W / 2) / (d * 0.9 + 0.22); pz = 1.0 / (d + 0.12)                                              # 市松の 床
        cu_ = pu / (120 * k); cv_ = pz * 1.8; chk = ((np.floor(cu_) + np.floor(cv_)) % 2)
        fl = (CREAM * 0.9)[None, None, :] * (chk * (0.85 + 0.2 * s.tex(1, pu * 0.3, pz * 50)))[:, :, None] + (VELV * 1.1)[None, None, :] * ((1 - chk) * (0.7 + 0.5 * s.tex(2, pu * 0.3, pz * 50)))[:, :, None]
        fl = fl * (0.55 + 0.5 * d)[:, :, None] + (np.exp(-np.abs(s.xn - 0.5) / 0.3) * np.exp(-d / 0.4) * 0.30)[:, :, None] * WARM[None, None, :]
        m = sstep(hy - 0.002, hy + 0.003, s.yn)[:, :, None]; f = f * (1 - m) + fl * m
        sk = np.exp(-((s.yn - hy) / 0.008) ** 2); f = f * (1 - sk * 0.8)[:, :, None] + (sk * 0.8)[:, :, None] * GOLD[None, None, :]                               # 金の はばき
        return (f * lum).astype(np.float32)

    def motes(s, f, t, v=1.0):
        W, H = s.W, s.H; r = s.rnd[:80]
        x = ((r[:, 0] + 0.02 * np.sin(t * (1 + r[:, 2]) + r[:, 3] * 6.28)) % 1.0) * W; y = ((r[:, 1] - t * (0.015 + 0.03 * r[:, 4])) % 1.0) * H
        s.splat(f, x, y, (0.3 + 0.5 * r[:, 2]) * v * (0.6 + 0.4 * np.sin(t * 6 + r[:, 5] * 20)), WARM, 1.1, 0.3)

    # ---- 食卓(手前へ のびる 白い クロス・赤い 帯・皿・ろうそく)----
    def table(s, f, t, g):
        if g <= 0.01: return
        W, H, k = s.W, s.H, s.k; y0 = 0.71; y1 = y0 + (1.08 - y0) * g
        v = np.clip((s.yn - y0) / (1.08 - y0), 0, 1); hw = 0.16 + 0.62 * v ** 1.25; ax = np.abs(s.xn - 0.5)
        m = sstep(-0.003, 0.003, s.yn - y0) * sstep(0.004, -0.004, s.yn - y1) * sstep(0.004, -0.004, ax - hw)
        cl = WHITE_[None, None, :] * (0.90 + 0.10 * s.tex(1, (s.xn - 0.5) / (hw + 0.01) * 120, v * 300) - 0.10 * np.clip((ax / hw - 0.86) / 0.14, 0, 1))[:, :, None]
        rn = sstep(0.16, 0.14, ax / hw); cl = cl * (1 - rn * 0.9)[:, :, None] + (VELV * 1.5)[None, None, :] * (rn * 0.9 * (0.8 + 0.3 * s.tex(2, s.XX * 0.6, s.YY * 0.6)))[:, :, None]
        ln = np.exp(-(((ax / hw) - 0.16) / 0.012) ** 2) + np.exp(-(((ax / hw) - 0.93) / 0.012) ** 2) * (0.5 + 0.5 * (np.sin(v ** 0.8 * 160) > 0))
        cl = cl * (1 - np.clip(ln, 0, 1) * 0.8)[:, :, None] + (np.clip(ln, 0, 1) * 0.8)[:, :, None] * GOLD[None, None, :]
        f[:] = f * (1 - m)[:, :, None] + cl * m[:, :, None]
        edge = np.exp(-((s.yn - y1) / 0.006) ** 2) * sstep(0.004, -0.004, ax - hw) * (g < 0.99); f[:] = f + (edge * 0.6)[:, :, None]
        for i in range(8):                                                           # 皿と ろうそく(のびた 所から 順に ぽん)
            vv = 0.08 + i * 0.125; yy = y0 + (1.08 - y0) * vv
            if yy > y1 - 0.01: break
            hwv = 0.16 + 0.62 * vv ** 1.25; sc = 0.35 + 1.6 * vv; age = (g - (vv + 0.02)) * 6
            pop = back(clamp(age)) if age > 0 else 0
            for sg in (-1, 1):
                px = W * (0.5 + sg * hwv * 0.58); py = H * yy; rx = 30 * k * sc * pop; ry = rx * 0.34
                if rx < 2: continue
                dd = np.hypot((s.XX - px) / rx, (s.YY - py) / ry); pm = sstep(1.0, 0.9, dd)
                pc = WHITE_[None, None, :] * (0.80 + 0.2 * sstep(0.75, 0.6, dd) - 0.12 * np.exp(-((dd - 0.72) / 0.05) ** 2))[:, :, None] + (GOLD * 0.5)[None, None, :] * (np.exp(-((dd - 0.94) / 0.04) ** 2))[:, :, None]
                f[:] = f * (1 - pm)[:, :, None] + pc * pm[:, :, None]
                ck = sstep(1.0, 0.85, np.hypot((s.XX - px) / (rx * 0.45), (s.YY - (py - ry * 0.5)) / (ry * 1.1)))                                        # 小さな ケーキ
                f[:] = f * (1 - ck)[:, :, None] + (ck * (0.85 + 0.25 * s.tex(1, s.XX * 0.9, s.YY * 0.9)))[:, :, None] * (PINK if (i + (sg > 0)) % 2 else CREAM)[None, None, :]
                cxp = W * (0.5 + sg * hwv * 0.92); cyp = py - 26 * k * sc * pop                                                                           # ろうそくの 火
                s.glowmask(f, np.exp(-(np.hypot(s.XX - cxp, (s.YY - cyp) * 0.7) / (5 * k * sc)) ** 2) * pop, WARM, 1.4 * (0.8 + 0.2 * math.sin(t * 11 + i * 2 + sg)), 1.0, 0.7, 5.0 * sc)

    # ---- 銀の ふた(クロッシュ): 上から 降りて 大きく なる ----
    def cloche(s, f, t, d):
        if d <= 0.001: return
        W, H, k = s.W, s.H, s.k; cx = W * 0.5; Wd = W * (0.42 + 1.75 * d ** 1.4); rim = H * (-0.02 + 1.22 * d ** 1.15); Hd = Wd * 0.62
        u = (s.XX - cx) / (Wd / 2); vv = (rim - s.YY) / Hd; m = sstep(1.0, 0.985, np.hypot(u, vv)) * sstep(-0.004, 0.004, vv)
        bulge = np.sqrt(np.clip(1 - u * u - vv * vv, 0, 1))
        sh = 0.28 + 0.30 * bulge + 0.95 * np.exp(-((u + 0.42 - 0.2 * vv) / 0.07) ** 2) * (0.5 + 0.5 * vv) + 0.30 * np.exp(-((u - 0.30) / 0.22) ** 2) + 0.18 * np.exp(-((vv - 0.75) / 0.2) ** 2)
        refl = s.tex(2, u * 60 + t * 6, vv * 30)
        colr = SILV[None, None, :] * (sh * (0.86 + 0.2 * refl))[:, :, None] + (VELV * 0.9)[None, None, :] * (np.exp(-vv / 0.22) * 0.35 * (0.6 + 0.6 * refl))[:, :, None] + (WARM * 0.4)[None, None, :] * (np.exp(-((u - 0.05) / 0.5) ** 2) * np.exp(-((vv - 0.35) / 0.25) ** 2) * 0.4)[:, :, None]
        rb = np.exp(-((vv - 0.03) / 0.022) ** 2) * (np.abs(u) < 1.02); colr = colr * (1 - np.clip(rb, 0, 1) * 0.85)[:, :, None] + (np.clip(rb, 0, 1) * (0.7 + 0.6 * np.exp(-((u + 0.4) / 0.2) ** 2)))[:, :, None] * GOLD[None, None, :]
        f[:] = f * (1 - m)[:, :, None] + colr * m[:, :, None]
        ky = rim - Hd * 1.0; kr = Wd * 0.045                                                                                                             # つまみ
        dk = np.hypot(s.XX - cx, (s.YY - (ky - kr * 0.6))) / kr; km = sstep(1.0, 0.9, dk)
        f[:] = f * (1 - km)[:, :, None] + (GOLD[None, None, :] * (0.55 + 0.7 * np.exp(-(np.hypot(s.XX - cx + kr * 0.3, s.YY - ky + kr * 0.9) / (kr * 0.5)) ** 2))[:, :, None]) * km[:, :, None]
        sd = sstep(0.0, 0.10, -vv) * np.exp(-(u / 1.1) ** 2) * sstep(0.35, 0.0, -vv) * 0.55 * min(1.0, d * 3); f[:] = f * (1 - sd * (1 - m))[:, :, None]                # ふたの 下の かげ

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        s.draw_lit(f, img, cx, cy, hfrac, rim=(1.0, 0.9, 0.7), rim_dir=(0.0, -1.0), rim_v=0.45 * alpha, amb=(1.0, 0.92, 0.88), amb_v=0.08, aura=WARM, aura_v=0.06, alpha=alpha, shade=0.98)

    def frame(s, t):
        o = {}
        if t < T_CUT: f, sh = s.sc_hall(t)
        elif t < T_ACT: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.5, thr=0.96)
        else: f, sh = s.sc_feast(t)
        f, sh = s.finale(f, sh, t, DUR, None, (1.0, 0.95, 0.88), (1.0, 0.8, 0.6), -1)
        return s.film(f, t, sh * s.k, 0.4, o.get("thr", 0.95), (1.0, 0.96, 0.92), 1.0 + 0.15 * min(sh, 8), 0.03, o.get("vig", 0.45), 1.18, 0.97)

    def queen(s, f, t, alpha=1.0, **kw):
        W, H, k = s.W, s.H, s.k; cy = H * 0.505; fy = cy + 0.38 * H * 0.47
        sd = np.exp(-(((s.XX - W * 0.5) / (W * 0.22)) ** 2 + ((s.YY - fy) / (H * 0.012)) ** 2)); f *= (1 - 0.45 * alpha * sd)[:, :, None]
        img = s.pose(t, **kw); s.draw(f, img, W / 2, cy, 0.38, alpha); return s.at(*s.A(*WHISK))

    # 1) 広間に 立つ
    def sc_hall(s, t):
        f = s.hall(t); s.queen(f, t, clamp(t / 0.6), sway=0.25, breath=0.5); s.motes(f, t, 0.6)
        return f * clamp(t / 0.5), 0

    # 2) 目の カットイン(レースの ふちの 丸窓)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.hall(t, 0.34)
        a = back(clamp(u / 0.4)); c = (W / 2, H * 0.32); R = H * 0.165 * a
        bandc = (VELV * 1.3)[None, None, :] + (PINK * 0.8)[None, None, :] * (0.25 + 0.9 * sstep(0.45, 0.9, s.tex(2, s.XX * 0.5 + t * 30, s.YY * 0.5 - t * 14)))[:, :, None]
        ex, ey = s.A(*EYE); zf = 1.4; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(bandc, img, c[0] - (ex - s.aw / 2) * sc, c[1] - (ey - s.ah / 2) * sc, zf)
        if R > 6:
            d = np.hypot(s.XX - c[0], s.YY - c[1]); th = np.arctan2(s.YY - c[1], s.XX - c[0]) + t * 0.15
            m = sstep(R, R - 2, d)[:, :, None]; f = f * (1 - m) + bandc * m
            Ro = R * (1.16 + 0.045 * np.abs(np.cos(th * 9))); lace = sstep(-1, 1, d - R) * sstep(1, -1, d - Ro)                                              # レース
            hole = np.exp(-(np.hypot((d - R * 1.085) / (R * 0.03), np.sin(th * 9) * 0.9)) ** 2)
            f = f * (1 - lace)[:, :, None] + WHITE_[None, None, :] * (lace * (0.95 - 0.5 * hole) * (0.92 + 0.1 * s.tex(1, th * 60, d)))[:, :, None]
            s.glowmask(f, np.exp(-((d - R) / (1.8 * k)) ** 2), GOLD, 0.8 * a, 1.0, 0.6, 5.0)
        n = len(s._vt); step = W * 0.172
        for i, T in enumerate(s._vt):                                                # 技名: おじぎを する ように 下から ふわり
            p = (u - 0.55 - i * 0.13) / 0.4
            if p > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.70 + (1 - ease(p)) * 28 * k, clamp(p) ** 1.2, 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.4 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.70 - 74 * k, 1.0, 1.0)
        if u > 1.4:
            g = ease((u - 1.4) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.70 + 56 * k), (W / 2 + W * 0.42 * g, H * 0.70 + 56 * k)], GOLD, 1.0, 0.7, 0.7)
            s.text(f, s.txt_en, W / 2, H * 0.70 + 82 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.5
        return f, 3 * math.exp(-u * 6)

    # 3) 食卓が のびる → 銀の ふたが かぶさる → 暗やみ → ちーん → ひらく
    def sc_feast(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_ACT; sh = 0.0
        g = ease2(clamp((t - T_TABLE) / 1.5)); d = clamp((t - T_LID) / (T_DARK - T_LID)); dk = clamp((t - T_DARK) / 0.45); e = clamp((t - T_BELL) / (T_HIT - T_BELL))
        f = s.hall(t, 1.0 - 0.25 * d)
        wp = s.queen(f, t, 1.0, sway=0.2, breath=0.4, squash=(0.04 * ease(u / 0.3) if t < T_TABLE else -0.03 * math.exp(-(t - T_TABLE) * 4)))
        s.table(f, t, g); s.motes(f, t, 0.6)
        q = t - T_TABLE
        if 0 < q < 1.9:                                                              # クリームの リボン(泡立て器から うずを まいて 上へ)
            pts = []; n = 26
            for j in range(n):
                a_ = q * 5.0 - j * 0.30; r_ = (16 + j * 5.0) * k * min(1.0, q * 2.5); pts.append((wp[0] + math.cos(a_) * r_ + j * 5 * k, wp[1] - 26 * k + math.sin(a_) * r_ * 0.55 - j * 7 * k * min(1.0, q)))
            s.glowline(f, pts, WHITE_, clamp(1.9 - q) * 1.0, 7.0, 0.35); s.glowline(f, pts, PINK, clamp(1.9 - q) * 0.5, 2.0, 0.2)
            if q < 0.1: f = f + (1 - q / 0.1) * 0.3; sh += 5 * (1 - q / 0.1)
        if d > 0:
            s.cloche(f, t, d); sh += 1.5 + 5 * d ** 2
        if dk > 0:                                                                   # 暗やみ: 下の すき間の 光が とじる
            slit = np.exp(-((s.yn - (0.93 + 0.10 * dk)) / (0.05 * (1 - dk) + 0.004)) ** 2) * (1 - dk) ** 0.7
            f = f * (1 - dk) * 0.9 + (slit * 0.9)[:, :, None] * WARM[None, None, :] + np.array([0.03, 0.02, 0.04], np.float32)[None, None, :] * ((0.5 + s.tex(2, s.XX * 0.4, s.YY * 0.4)) * dk)[:, :, None]
            if 0.98 < dk and t < T_DARK + 0.6: sh += 8 * (1 - (t - T_DARK - 0.45) / 0.15) if t < T_DARK + 0.6 else 0
        if e > 0:                                                                    # ちーん → ひらく
            C = (W * 0.5, H * 0.45); dist = np.hypot(s.XX - C[0], s.YY - C[1]); th = np.arctan2(s.YY - C[1], s.XX - C[0])
            star = np.exp(-dist / (k * (14 + 120 * e))) * (0.4 + 2.0 * np.abs(np.cos(th * 2)) ** 30) + np.exp(-((dist - k * 420 * e) / (k * (5 + 26 * e))) ** 2) * (1 - e) * 0.9
            f = f + (star * (1.2 + 1.5 * e))[:, :, None] * GOLD[None, None, :]
            op = sstep(0.25, 1.0, e)                                                 # 下から 光が さしこむ(ふたが 上がる)
            f = f + (np.clip((s.yn - (1.0 - 1.15 * op)) / 0.25, 0, 1) * op * 1.6)[:, :, None] * WARM[None, None, :]
            sh += 4 * math.exp(-e * 6) + 8 * op
            if e > 0.6: f = f + ((e - 0.6) / 0.4) ** 2 * 1.6
        if u < 0.12: f = f + (1 - u / 0.12) * 0.25
        return f.astype(np.float32), sh

    # ---- この子だけの曲: ワルツ(3拍子)の オルゴール → 皿が ならぶ 音(ちりん)→ 低い 金属の うなり(ふた)→ 無音 → ちーん → 和音 ----
    def audio(s, path):
        a = Snd(DUR, 432); G = 196.0; pn = [0, 2, 4, 5, 7, 9, 11]
        def fq(d, o=0): return G * 2 ** ((pn[d % 7] + 12 * (d // 7 + o)) / 12.0)
        for bar in range(3):                                                                                              # ずん・ちゃっ・ちゃっ
            tb = 0.3 + bar * 0.78; a.note(fq([0, 3, 4][bar], -1), tb, 0.5, 0.07, "fm")
            for j in (1, 2): a.note(fq([2, 5, 6][bar], 0), tb + j * 0.26, 0.2, 0.045, "pad", 0.01); a.note(fq([4, 7, 8][bar], 0), tb + j * 0.26, 0.2, 0.04, "pad", 0.01)
        for i, d in enumerate((7, 9, 11, 9, 10, 8, 9, 7, 11)): a.note(fq(d, 0), 0.3 + i * 0.26, 0.4, 0.06, "bell")
        a.boom(T_CUT, 60, 0.6, 0.35); a.whoosh(T_CUT - 0.3, T_CUT, 0.2, 0.1)
        for i in range(5): a.note(fq([7, 9, 11, 12, 14][i], 0), T_CUT + 0.55 + i * 0.13, 0.8, 0.07, "bell")
        for d in (0, 2, 4): a.note(fq(d, 0), T_CUT, 1.9, 0.04, "pad", 0.05)
        a.whoosh(T_TABLE, T_TABLE + 1.4, 0.5, 0.6); a.rise(T_TABLE, T_TABLE + 1.5, 200, 420, 0.05)
        rg = np.random.RandomState(432)
        for i in range(8): tk = T_TABLE + 0.25 + i * 0.16; a.note(fq(9 + i, 1), tk, 0.25, 0.05, "bell", 0.002); a.note(2400 + rg.rand() * 900, tk + 0.02, 0.05, 0.03, "bell", 0.001)
        lo = a.lp(a.nz, 0.02); wn = a.lp(a.nz, 0.18) - a.lp(a.nz, 0.03); hi = a.nz - a.lp(a.nz, 0.6)
        g = np.clip((a.t - T_LID) / (T_DARK - T_LID), 0, 1) ** 1.6 * (a.t < T_DARK); a.a += lo * g * 3.2 + wn * g * 0.9
        a.rise(T_LID, T_DARK, 220, 55, 0.11)
        for j, d in enumerate((0, 1, 4)): a.note(fq(d, -1), T_LID + j * 0.08, T_DARK - T_LID, 0.05, "pad", 0.6)
        a.boom(T_DARK, 38, 1.6, 1.0); a.boom(T_DARK + 0.02, 180, 0.25, 0.5); a.a += hi * a.env(T_DARK, 0.001, 0.08) * 0.6                 # がこん
        for j, fqq in enumerate((2093.0, 4186.0, 6272.0, 3136.0)): a.note(fqq, T_BELL, 1.2 - j * 0.2, 0.09 - j * 0.015, "bell")       # ちーん
        for j, d in enumerate((0, 2, 4, 7, 9)): a.note(fq(d, 1), T_BELL + 0.2 + j * 0.06, T_HIT - T_BELL + 0.3, 0.05, "bell")
        a.rise(T_BELL + 0.15, T_HIT, 200, 700, 0.08)
        a.boom(T_HIT, 42, 1.4, 0.7)
        for j, d in enumerate((0, 2, 4, 7)): a.note(fq(d, 2), T_HIT + 0.02 + j * 0.03, 0.9, 0.05, "bell")
        s.fin_snd(a, DUR, 432); a.save(path)


if __name__ == "__main__":
    r = Patissiel()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
