"""サジタ(★4 射手座の狩人) 専用の必殺技「流星射チ」— 高品質版(hs_fx)。放った 矢が 流れ星に なって 夜空を ひとめぐりし、向きを 変えて まっすぐ こちらへ 落ちて くる
星空を うつす 湖の ほとり。流れ星が ひとつ 横ぎる(前ふり) → 目の カットイン+技名 →
放つ: 矢は 右へ 飛び去り、流れ星に なって 空を 大きく うずを えがいて まわる(光の 尾が 残り、湖にも うつる)→
うずの まん中で 止まり、こちらを 向く: 点が ふくらみ、青白い しんと 緑・金の 炎の 尾を ひいて 正面から 落ちて くる → 白 → 暗転
使い方: python3 hs_407.py 出力.mp4 / python3 hs_407.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.3
T_CUT, T_DRAW, T_SHOT, T_TURN, T_HIT = 2.7, 4.7, 5.2, 7.4, 9.6
NIGHT = np.array([0.02, 0.04, 0.12], np.float32); STAR = np.array([0.80, 0.90, 1.0], np.float32); GOLD = np.array([1.0, 0.82, 0.36], np.float32); TEAL = np.array([0.35, 1.0, 0.80], np.float32); CORE = np.array([0.90, 0.97, 1.0], np.float32)
EYE = (400, 262); TIPA = (815, 328); BOWS = [(625, 60), (630, 625)]


class Sagitta(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 407, (0.6, 0.85, 1.0), "流星射チ", "Meteor Shot", W); s.fx_init(407)
        s._vt = [s.text_img(c, hz.SERIF, int(76 * s.k), 3) for c in "流星射チ"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; s.hy = 0.69; H = s.H; k = s.k; s.VP = (W * 0.52, H * 0.30)
        band = np.exp(-(((s.YY - H * 0.30) - (s.XX - W * 0.5) * 0.7) / (H * 0.13)) ** 2); s._mw = (band * s.fbm(0.0016, 3, 9, 4, 50 * k, 2)).astype(np.float32)
        s._tree = (s.hy - 0.012 - 0.022 * s.tex(1, s.XX * 0.5, 4.0) - 0.012 * s.tex(3, s.XX * 2.2, 8.0)).astype(np.float32)
        s._shore = (0.835 + 0.03 * ((s.xn - 0.42) / 0.6) ** 2 - 0.01 * s.tex(2, s.XX * 0.4, 2.0)).astype(np.float32)

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    def heaven(s, t, lum=1.0):
        f = NIGHT[None, None, :] * (0.6 + 1.6 * (s.yn / s.hy).clip(0, 1) ** 1.5)[:, :, None] + np.array([0.05, 0.14, 0.24], np.float32)[None, None, :] * np.exp(-((s.yn - s.hy) / 0.12) ** 2)[:, :, None]
        f = f + s.nebula(t, [(0.04, 0.07, 0.24), (0.14, 0.09, 0.30), (0.40, 0.50, 0.80)], 0.0008, (3.0, 0.6), 0.6, 0) + (s._mw * 0.34)[:, :, None] * STAR[None, None, :]
        s.starfield(f, t, 1300, (0.9, 0.94, 1.0), 2, 1.0, (0.0, 0.0)); s.starfield(f, t, 120, tuple(GOLD * 0.8 + 0.2), 7, 1.5, (0.0, 0.0))
        return (f * lum).astype(np.float32)

    def lake(s, f, t):
        """空を 湖に うつす + 向こう岸の 森 + 手前の 岸"""
        W, H, k = s.W, s.H, s.k; hy = s.hy; d = np.clip((s.yn - hy) / (1 - hy), 0, 1)
        mx = s.XX + (np.sin(s.YY * 0.33 + t * 2.6) * 3.0 + np.sin(s.YY * 0.9 - t * 4.0) * 1.2) * k * (0.3 + 2.5 * d); my = 2 * hy * H - s.YY - 2
        rf = cv2.remap(f, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        rf = rf * np.array([0.50, 0.58, 0.72], np.float32)[None, None, :] * (0.95 - 0.35 * d)[:, :, None] + NIGHT[None, None, :] * 0.5
        rf *= (0.85 + 0.3 * s.tex(0, s.XX * 0.25 + t * 6, s.YY * 3.0))[:, :, None]                                       # さざ波
        m = sstep(hy - 0.002, hy + 0.002, s.yn)[:, :, None]; f = f * (1 - m) + rf * m
        tm = (sstep(s._tree - 0.002, s._tree + 0.002, s.yn) * sstep(hy + 0.006, hy + 0.002, s.yn))[:, :, None]; f = f * (1 - tm) + np.array([0.012, 0.022, 0.04], np.float32)[None, None, :] * tm
        f += (np.exp(-((s.yn - hy - 0.004) / 0.003) ** 2) * 0.22)[:, :, None] * STAR[None, None, :]
        sm = sstep(s._shore - 0.003, s._shore + 0.003, s.yn)[:, :, None]; gn = s.tex(1, s.XX * 1.4 + np.sin(t * 2 + s.YY * 0.04) * 3 * k, s.YY * 0.6) * 0.6 + s.tex(3, s.XX * 0.8, s.YY * 0.9) * 0.4
        f = f * (1 - sm) + (np.array([0.012, 0.03, 0.045], np.float32)[None, None, :] * (0.5 + 1.6 * gn)[:, :, None] + (STAR * 0.07)[None, None, :] * np.exp(-np.clip(s.yn - s._shore, 0, 1) / 0.02)[:, :, None]) * sm
        return f

    def path(s, q):
        a = 0.25 - 4.9 * q; R = s.W * (0.78 - 0.76 * q ** 0.85); return (s.VP[0] + math.cos(a) * R, s.VP[1] + math.sin(a) * R * 0.72)

    def trail(s, f, q, fade=1.0):
        """流れ星の 尾(新しい ほど 明るい)"""
        if q <= 0: return
        k = s.k; n = 90; L = s.L0(); L2 = s.L0(); qs = [q * i / n for i in range(n + 1)]; P = [s.path(x) for x in qs]
        for i in range(n):
            age = q - qs[i + 1]; v = (math.exp(-age * 4.0) + 0.32) * fade
            cv2.line(L, (int(P[i][0]), int(P[i][1])), (int(P[i + 1][0]), int(P[i + 1][1])), float(min(1.0, v)), max(1, int((2.0 + 4.5 * math.exp(-age * 7)) * k)), cv2.LINE_AA)
            if age < 0.12: cv2.line(L2, (int(P[i][0]), int(P[i][1])), (int(P[i + 1][0]), int(P[i + 1][1])), float(1 - age / 0.12), max(1, int(2 * k)), cv2.LINE_AA)
        Lb = cv2.GaussianBlur(L, (0, 0), 0.8 * k); f += (Lb * 1.0 + cv2.GaussianBlur(L, (0, 0), 5 * k) * 0.9)[:, :, None] * (STAR * 0.6 + TEAL * 0.4)[None, None, :] + (cv2.GaussianBlur(L2, (0, 0), 1.0 * k) * 1.2)[:, :, None]
        hp = P[-1]; s.flare(f, hp, 9 * k, CORE, 1.0 * fade, 0.8, 0.5, False)

    def meteor(s, f, C, R, e, t):
        """正面から 落ちて くる 流星: 青白い しん・緑の ふち・金の 炎の 尾(放射)"""
        dx = s.XX - C[0]; dy = s.YY - C[1]; r = np.hypot(dx, dy) / R + 1e-4; th = np.arctan2(dy, dx); lr = np.log(r + 0.05)
        n = s.tex(1, th / 6.2832 * 768, lr * 46 - t * 210); n2 = s.tex(3, th / 6.2832 * 1536 + 20, lr * 110 - t * 480)
        head = np.exp(-(r / 0.55) ** 2); rim = np.exp(-((r - 1.0) / 0.22) ** 2)
        tail = np.exp(-np.clip(r - 1.0, 0, None) / (1.6 + 2.5 * e)) * (r > 0.8) * (0.25 + 1.0 * n ** 1.6 + 0.7 * n2 ** 3)
        f *= (1 - sstep(1.15, 0.8, r) * 0.9)[:, :, None]
        f += (head * 2.4)[:, :, None] * CORE[None, None, :] + (rim * (0.9 + 0.8 * n))[:, :, None] * TEAL[None, None, :] + (sstep(1.25, 0.85, r) * 0.55 * (0.5 + n))[:, :, None] * (STAR * 0.7)[None, None, :]
        wt = sstep(1.0, 3.5, r)[:, :, None]; f += tail[:, :, None] * (TEAL[None, None, :] * (1 - wt) + (GOLD * 1.1)[None, None, :] * wt) * (0.8 + 0.5 * e)

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        m = getattr(s, '_ml', 0.0)
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(STAR * (1 - m) + (TEAL * 0.5 + CORE * 0.5) * m), rim_dir=(0.2 * m, -1.0), rim_v=(0.5 + 0.5 * m) * alpha, amb=(0.6, 0.75, 1.0), amb_v=0.22, aura=None, aura_v=0.0, alpha=alpha, shade=0.72 + 0.25 * m)

    def frame(s, t):
        o = {}; s._ml = 0.0
        if t < T_CUT: f, sh = s.sc_lake(t)
        elif t < T_DRAW: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.92)
        else: f, sh = s.sc_meteor(t)
        f, sh = s.finale(f, sh, t, DUR, None, (0.7, 0.9, 1), (0.4, 1, 0.8), -1)
        return s.film(f, t, sh * s.k, 0.55, o.get("thr", 0.82), (0.9, 0.97, 1.0), 1.0 + 0.15 * min(sh, 8), 0.04, o.get("vig", 0.58), 1.15, 1.12)

    def hunter(s, f, t, alpha=1.0, **kw):
        W, H, k = s.W, s.H, s.k; img = s.pose(t, **kw); s.draw(f, img, W * 0.42, H * 0.655, 0.40, alpha)
        for i, b in enumerate(BOWS): s.flare(f, s.at(*s.A(*b)), (5 + 2.5 * math.sin(t * 5 + i * 2)) * k, GOLD, 0.7 * alpha, 0.4, 0.0, False)

    # 1) 星空を うつす 湖。流れ星が ひとつ 横ぎる
    def sc_lake(s, t):
        W, H, k = s.W, s.H, s.k
        f = s.heaven(t); q = (t - 1.0) / 0.55
        if 0 < q < 1.6:                                                              # 前ふりの 流れ星
            a = (W * (0.15 + 0.5 * min(q, 1.0)), H * (0.10 + 0.16 * min(q, 1.0))); b = (a[0] - W * 0.16 * min(1.0, q * 3), a[1] - H * 0.05 * min(1.0, q * 3)); L = s.L0(); cv2.line(L, (int(b[0]), int(b[1])), (int(a[0]), int(a[1])), 1.0, max(1, int(2 * k)), cv2.LINE_AA)
            s.glowmask(f, cv2.GaussianBlur(L, (0, 0), 0.8 * k), STAR, 1.0 * max(0.0, 1 - max(0.0, q - 1) / 0.6), 1.0, 0.9, 6.0)
        f = s.lake(f, t); s.hunter(f, t, 1.0, sway=0.15, breath=0.5)
        return f * clamp(t / 0.5), 0

    # 2) 目の カットイン(流れ星の 尾の 形の 帯)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.lake(s.heaven(t, 0.32), t)
        a = ease(u / 0.24); cy = H * 0.33; hh = H * 0.105
        band = (NIGHT * 2.0)[None, None, :] + (STAR * 0.42)[None, None, :] * (0.3 + 0.9 * s.tex(1, s.XX * 0.07 - t * 220, s.YY * 2.0))[:, :, None] + (TEAL * 0.12)[None, None, :]
        ex, ey = s.A(*EYE); zf = 1.5; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(band, img, W / 2 - (ex - s.aw / 2) * sc - (1 - a) * W * 0.9 + u * 6 * k, cy - (ey - s.ah / 2) * sc, zf)
        hd = W * (1.05 * a); edge = hh * np.clip((s.XX - (hd - W * 1.15)) / (W * 1.1), 0, 1) ** 0.6 * (s.XX < hd); dd = np.abs(s.YY - cy + (s.XX - W / 2) * 0.10)      # 左が 細い 尾、右が 太い 頭
        m = (sstep(edge, edge - 2, dd) * (edge > 1))[:, :, None]; f = f * (1 - m) + band * m
        s.glowmask(f, np.exp(-((dd - edge) / (1.6 * k)) ** 2) * (edge > 1), STAR, 0.9 * a, 1.0, 0.8, 8.0)
        n = len(s._vt); step = W * 0.20
        for i, T in enumerate(s._vt):                                                # 技名: ななめ上から 流れて 入る
            p = (u - 0.45 - i * 0.11) / 0.16
            if p > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step - (1 - ease(p)) * 110 * k, H * 0.68 - (1 - ease(p)) * 50 * k, clamp(p * 2), 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.3 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.68 - 76 * k, 1.0, 1.0)
        if u > 1.25:
            g = ease((u - 1.25) / 0.35); s.glowline(f, [(W * 0.08, H * 0.68 + 58 * k), (W * (0.08 + 0.84 * g), H * 0.68 + 58 * k)], STAR, 1.0, 0.8, 0.8)
            s.text(f, s.txt_en, W / 2, H * 0.68 + 84 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.6
        return f, 4 * math.exp(-u * 6)

    # 3) 放つ → 流れ星が 空を めぐる → こちらを 向いて 落ちて くる
    def sc_meteor(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_DRAW; us = t - T_SHOT; ut = t - T_TURN; sh = 0.0; VP = s.VP
        e = clamp(ut / (T_HIT - T_TURN)) if ut > 0 else 0.0; s._ml = e
        f = s.heaven(t, 1.0 - 0.3 * e)
        q = clamp((us - 0.12) / (T_TURN - T_SHOT - 0.12)) if us > 0 else 0.0
        if us > 0: s.trail(f, ease2(q) if q < 1 else 1.0, 1.0 - 0.75 * clamp(ut / 0.8) if ut > 0 else 1.0)
        if ut > 0:
            R = k * (5 + 26 * e) + W * 0.95 * e ** 3.0; C = (VP[0] + (W * 0.5 - VP[0]) * e, VP[1] + (H * 0.40 - VP[1]) * e ** 2)
            s.meteor(f, C, R, e, t); s.rushdust(f, t, C, 900, 0.8 + 2.2 * e, GOLD * 0.6 + CORE * 0.4, 0.2 + 0.6 * e, 4, 1.0)
            if ut < 0.15: s.flare(f, VP, 30 * k * (1 - ut / 0.15) + 4, CORE, 1.0, 1.2, 0.4, True)                           # きらり、と 向きを 変える
            sh += 1 + 11 * e ** 2
        f = s.lake(f, t)
        a_ = 1 - clamp((e - 0.62) / 0.2)
        if a_ > 0.02:
            g = ease(u / 0.45) if us < 0 else 0.0
            s.hunter(f, t, a_, sway=0.1, lean=(-5 * k * g if us < 0 else 6 * k * math.exp(-us * 6)), squash=(-0.03 * g if us < 0 else 0.04 * math.exp(-us * 8)))
            tp = s.at(*s.A(*TIPA))
            if us < 0: s.flare(f, tp, (8 + 14 * g + 3 * math.sin(t * 20) ** 2) * k, CORE, 1.0, 0.8, t * 2, False)
            elif us < 0.16:
                p = us / 0.16; L = s.L0(); cv2.line(L, (int(tp[0] + W * 0.9 * max(0.0, p - 0.4)), int(tp[1] - H * 0.02 * p)), (int(tp[0] + W * 0.9 * p), int(tp[1] - H * 0.03 * p)), 1.0, max(2, int(4 * k)), cv2.LINE_AA)
                s.glowmask(f, cv2.GaussianBlur(L, (0, 0), 0.8 * k), CORE, 1.3, 1.0, 1.0, 8.0); f = f + (1 - p) * 0.5; sh += 7 * (1 - p)
        if ut > 0:
            f = s.godrays(f, (W * 0.5, H * 0.38), 0.2 + 0.5 * e, 0.95, 8, 0.88, 0.45)
            if e > 0.84: f = f + ((e - 0.84) / 0.16) ** 2 * 1.5
        if u < 0.12: f = f + (1 - u / 0.12) * 0.3
        return f, sh

    # ---- この子だけの曲: しずかな 夜の 分散和音 → 流れ星の 一音 → 弦を はじく → 高い 音が 空を めぐる(だんだん 上がる)→ 一拍 → 低い うなりが ふくらんで 落ちる ----
    def audio(s, path):
        a = Snd(DUR, 407); A = 220.0; ly = [0, 2, 4, 6, 7, 9, 11]
        def fq(d, o=0): return A * 2 ** ((ly[d % 7] + 12 * (d // 7 + o)) / 12.0)
        for d in (0, 4): a.note(fq(d, -1), 0.1, 2.6, 0.045, "pad", 0.7)
        for i in range(9): a.note(fq([0, 4, 7, 9, 11, 9, 7, 4, 7][i], 0), 0.4 + i * 0.24, 0.6, 0.06, "bell")
        a.rise(1.0, 1.55, 2400, 1200, 0.04); a.note(fq(14, 1), 1.0, 0.7, 0.07, "sine")                                   # 前ふりの 流れ星
        a.boom(T_CUT, 55, 0.8, 0.4); a.whoosh(T_CUT - 0.25, T_CUT, 0.4, 0.5)
        for i in range(4): a.note(fq([9, 7, 6, 4][i], 1), T_CUT + 0.45 + i * 0.11, 0.5, 0.10, "bell")
        for d in (0, 2, 4, 6): a.note(fq(d, 0), T_CUT, 1.9, 0.04, "pad", 0.05)
        a.rise(T_DRAW, T_SHOT, 180, 520, 0.08)
        a.note(fq(0, -1), T_SHOT, 0.5, 0.20, "fm", 0.001); a.a += (a.nz - a.lp(a.nz, 0.6)) * a.env(T_SHOT, 0.001, 0.05) * 0.6; a.whoosh(T_SHOT, T_SHOT + 0.3, 0.6, 0.6)
        ph = 2 * np.pi * np.cumsum(900 + 1500 * np.clip((a.t - T_SHOT) / (T_TURN - T_SHOT), 0, 1) ** 1.5) / 44100.0 if hasattr(a, 't') else 0     # 空を めぐる 高い 音
        try: a.a += np.sin(ph + 2 * np.sin(2 * np.pi * 6 * a.t)) * a.gate(T_SHOT + 0.15, T_TURN, 0.1) * 0.045
        except Exception: a.rise(T_SHOT + 0.15, T_TURN, 900, 2400, 0.05)
        for i in range(14): a.note(fq(i, 1), T_SHOT + 0.2 + i * (T_TURN - T_SHOT - 0.3) / 14, 0.3, 0.05, "bell")
        a.note(fq(14, 2), T_TURN, 0.9, 0.14, "bell"); a.boom(T_TURN, 80, 0.5, 0.2)
        a.rise(T_TURN + 0.1, T_HIT, 50, 260, 0.17); a.whoosh(T_TURN + 0.4, T_HIT, 1.0, 0.3)
        a.a += (a.lp(a.nz, 0.05) * 1.3 + (a.nz - a.lp(a.nz, 0.5)) * 0.15) * np.clip((a.t - T_TURN) / (T_HIT - T_TURN), 0, 1) ** 2.2 * (a.t < T_HIT) * (a.t > T_TURN)
        for j, d in enumerate((0, 4, 7, 9, 11, 14)): a.note(fq(d, -1 if j < 2 else 0), T_TURN + 0.3 + j * 0.08, T_HIT - T_TURN, 0.045, "pad", 0.6)
        a.boom(T_HIT, 34, 1.9, 0.9)
        for j, d in enumerate((0, 4, 7, 11)): a.note(fq(d, 1), T_HIT + 0.02, 0.6, 0.06, "bell")
        s.fin_snd(a, DUR, 407); a.save(path)


if __name__ == "__main__":
    r = Sagitta()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
