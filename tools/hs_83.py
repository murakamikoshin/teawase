"""エイオン(★8 永劫の天秤) 専用の必殺技「永劫ノ天秤」— 高品質版(hs_fx)・通し直し。光と 夜を はかり、釣り合った 輪を こちらへ 下す
まっ白な 無の 地平に 星が 流れこむ → ふり向くと 世界が 光(金の 星雲)と 夜(すみれの 星空)に 割れる → 目の カットイン →
金の 天秤(金属)。左の 皿に 太陽、右の 皿に 月。ゆれて → ぴたり(色も 音も 消える) → 開眼 →
太陽と 月が のぼって 重なり 金環に なる → こちらを 指さす → 金環が こちらへ 飛んでくる。輪の 中は 光と 夜の うず。輪を くぐって のみこまれる → 白 → 暗転
使い方: python3 hs_83.py 出力.mp4 / python3 hs_83.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old83", "/home/claude/tools/pre_fin/hs_83_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = 14.7
GOLD, VIO, ICE = old.GOLD, old.VIO, old.ICE
T_TURN, T_CUT, T_SCALE, T_STILL, T_OPEN, T_POINT = old.T_TURN, old.T_CUT, old.T_SCALE, old.T_STILL, old.T_OPEN, old.T_POINT
T_GO, T_HIT = 11.95, 14.0
EYES, HANDS2, FINGER4 = old.EYES, old.HANDS2, old.FINGER4
SUNC = np.array([1.0, 0.93, 0.72], np.float32); MOONC = np.array([0.80, 0.88, 1.0], np.float32)


class Aion(FX, old.Aion):
    def __init__(s, W=540):
        old.Aion.__init__(s, W); s.fx_init(83); s._mix = 1.0

    def polar(s, c, r):
        d = np.hypot(s.XX - c[0], s.YY - c[1]) / max(float(r), 1.0); th = np.arctan2(s.YY - c[1], s.XX - c[0]); return d.astype(np.float32), th.astype(np.float32)

    def tex(s, i, u, v):
        return cv2.remap(s._tiles[i], (u % 256).astype(np.float32), (v % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 光の 空(左)と 夜の 空(右)。mix=0 は まっ白な 無 ----
    def twin(s, t, mix=1.0, lum=1.0, mono=0.0):
        W, H, k = s.W, s.H, s.k; s._mix = mix
        wob = (s.fbm(0.0012, t * 4, t * 9, 3, 0, 1) - 0.5) * 0.10; xb = s.xn + wob
        b = sstep(0.44, 0.56, xb)[:, :, None]
        day = np.array([0.50, 0.31, 0.10], np.float32)[None, None, :] * (0.55 + 0.75 * (1 - s.yn) + 0.5 * np.exp(-np.hypot(s.xn - 0.1, (s.yn - 0.15) * 1.6) / 0.35))[:, :, None] + s.nebula(t, [(0.80, 0.50, 0.16), (1.0, 0.80, 0.40), (1.0, 0.95, 0.78)], 0.00045, (7.0, -3.0), 0.55, 0)
        ngt = np.array([0.045, 0.03, 0.12], np.float32)[None, None, :] * (0.6 + 0.6 * (1 - s.yn))[:, :, None] + s.nebula(t, [(0.14, 0.07, 0.38), (0.34, 0.20, 0.78), (0.58, 0.68, 1.0)], 0.0009, (-6.0, 2.0), 0.85, 1)
        st = np.zeros((H, W, 3), np.float32); s.starfield(st, t, 900, (0.85, 0.9, 1.0), 2, 1.0, (0.0, 2.5)); ngt += st
        f = day * (1 - b) + ngt * b
        f += (np.exp(-((xb - 0.5) / 0.010) ** 2) * 0.45)[:, :, None]                                                 # 割れ目の 光
        f *= lum
        if mono: g = f.mean(2, keepdims=True); f = f * (1 - mono) + g * mono * 1.2
        if mix < 1:
            hz = s.fbm(0.0009, t * 5, t * 2, 3, 30 * k, 2)
            void = ((0.93 - 0.22 * np.abs(s.yn - 0.62)) * (0.92 + 0.16 * hz) * (1 - 0.10 * sstep(0.66, 0.9, s.yn)))[:, :, None] * (ICE * 0.18 + 0.82)[None, None, :]
            f = void * (1 - mix) + f * mix
        return f.astype(np.float32)

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        m = s._mix
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(GOLD * 0.6 + 0.4), rim_dir=getattr(s, '_rdir', (-0.8, -0.5)), rim_v=0.75 * alpha * m, amb=(0.8, 0.86, 1.0), amb_v=0.10, aura=ICE, aura_v=0.10 * m, alpha=alpha, shade=0.95)

    def add(s, f, L, col, gain=1.0, b1=3, b2=12):
        s.glowmask(f, np.clip(L, 0, 1), col, gain * 0.8, 1.0, 1.0, float(b2))

    # ---- 太陽: つぶつぶの 面 + ゆらぐ コロナ ----
    def sun(s, f, c, r, t, v=1.0):
        if v <= 0.01: return
        d, th = s.polar(c, r); n = s.tex(1, th / 6.2832 * 512 + t * 6, d * 22 - t * 22); gr = s.tex(3, (s.XX - c[0]) / r * 60 + t * 3, (s.YY - c[1]) / r * 60)
        disc = sstep(1.03, 0.93, d); cor = np.exp(-np.clip(d - 1, 0, None) / 0.60) * (d > 0.95) * (0.30 + 1.2 * n ** 2)
        f += (disc * (1.25 + 0.5 * gr) * (1.15 - 0.35 * d ** 2) * v)[:, :, None] * SUNC[None, None, :] + (cor * 0.85 * v)[:, :, None] * GOLD[None, None, :]
        s.flare(f, c, r * 0.8, GOLD, 0.35 * v, 0.5, t * 0.3, False)

    # ---- 月: 黒い 玉 + 太陽の側だけ 光る 細い ふち ----
    def moon(s, f, c, r, t, v=1.0):
        if v <= 0.01: return
        d, th = s.polar(c, r); core = sstep(1.0, 0.95, d); n = s.tex(2, (s.XX - c[0]) / r * 46 + 90, (s.YY - c[1]) / r * 46 + 30)
        f *= (1 - core * 0.97 * v)[:, :, None]
        f += (core * (0.05 + 0.10 * n) * v)[:, :, None] * (VIO * 0.8 + 0.2)[None, None, :]
        cres = core * sstep(0.62, 1.0, d) * np.clip(-np.cos(th - 0.5), 0, 1) ** 1.3
        f += (cres * 1.7 * v * (0.8 + 0.3 * n))[:, :, None] * MOONC[None, None, :]
        f += (np.exp(-np.clip(d - 1, 0, None) / 0.35) * (d >= 0.98) * 0.30 * v)[:, :, None] * VIO[None, None, :]

    # ---- 金の 天秤(金属): さお・支柱・くさり・皿 ----
    def scales(s, f, tilt, t, lit=1.0, orbs=True):
        W, H, k = s.W, s.H, s.k; B = (W / 2, H * 0.17); hl = W * 0.36
        ca, sa = math.cos(tilt), math.sin(tilt); ends = [(B[0] - ca * hl, B[1] - sa * hl), (B[0] + ca * hl, B[1] + sa * hl)]
        M = s.L0(); S = s.L0(); ip = lambda p: (int(round(p[0])), int(round(p[1])))
        def bar(p, q, w, base=0.42):
            dx, dy = q[0] - p[0], q[1] - p[1]; ln = math.hypot(dx, dy) + 1e-6; nx, ny = dy / ln, -dx / ln
            if ny > 0 or (ny == 0 and nx > 0): nx, ny = -nx, -ny                       # 上・左が 明るい
            o = w * 0.26; cv2.line(M, ip(p), ip(q), 1.0, int(w), cv2.LINE_AA); cv2.line(S, ip(p), ip(q), base, int(w), cv2.LINE_AA)
            cv2.line(S, ip((p[0] - nx * o, p[1] - ny * o)), ip((q[0] - nx * o, q[1] - ny * o)), 0.13, max(1, int(w * 0.3)), cv2.LINE_AA)
            cv2.line(S, ip((p[0] + nx * o, p[1] + ny * o)), ip((q[0] + nx * o, q[1] + ny * o)), 1.0, max(1, int(w * 0.28)), cv2.LINE_AA)
        def knob(p, r, base=0.5):
            cv2.circle(M, ip(p), int(r), 1.0, -1, cv2.LINE_AA); cv2.circle(S, ip(p), int(r), base, -1, cv2.LINE_AA); cv2.circle(S, ip((p[0] + r * 0.25, p[1] + r * 0.3)), int(r * 0.55), 0.2, -1, cv2.LINE_AA); cv2.circle(S, ip((p[0] - r * 0.3, p[1] - r * 0.35)), max(1, int(r * 0.32)), 1.0, -1, cv2.LINE_AA)
        bar(B, (B[0], H * 0.52), 10 * k)
        pans = []
        for e in ends:
            py = e[1] + H * 0.15; pw = W * 0.15
            for dx in (-pw * 0.96, 0, pw * 0.96):                                      # くさり
                q = (e[0] + dx, py); cv2.line(M, ip(e), ip(q), 0.9, 1, cv2.LINE_AA); cv2.line(S, ip(e), ip(q), 0.45, 1, cv2.LINE_AA); n = 13
                for j in range(1, n):
                    p = (e[0] + dx * j / n, e[1] + (py - e[1]) * j / n); cv2.circle(M, ip(p), max(1, int(1.6 * k)), 1.0, -1, cv2.LINE_AA); cv2.circle(S, ip(p), max(1, int(1.6 * k)), 1.0 if j % 2 else 0.5, -1, cv2.LINE_AA)
            c = ip((e[0], py)); ax = (int(pw), int(pw * 0.34))
            cv2.ellipse(M, c, ax, 0, 0, 180, 1.0, -1, cv2.LINE_AA); cv2.ellipse(S, c, ax, 0, 0, 180, 0.36, -1, cv2.LINE_AA)                 # 皿の おわん
            cv2.ellipse(S, c, (int(pw * 0.80), int(pw * 0.24)), 0, 100, 165, 1.0, max(2, int(4 * k)), cv2.LINE_AA)
            cv2.ellipse(S, c, (int(pw * 0.86), int(pw * 0.27)), 0, 15, 75, 0.12, max(2, int(6 * k)), cv2.LINE_AA)
            cv2.ellipse(S, c, (int(pw * 0.45), int(pw * 0.31)), 0, 60, 120, 0.62, max(1, int(2 * k)), cv2.LINE_AA)
            rim = (int(pw), max(2, int(pw * 0.075)))
            cv2.ellipse(M, c, rim, 0, 0, 360, 1.0, -1, cv2.LINE_AA); cv2.ellipse(S, c, rim, 0, 0, 360, 0.80, -1, cv2.LINE_AA)
            cv2.ellipse(S, c, (int(pw * 0.93), max(1, int(pw * 0.048))), 0, 0, 360, 0.16, -1, cv2.LINE_AA)
            cv2.ellipse(S, c, rim, 0, 110, 250, 1.0, max(1, int(1.5 * k)), cv2.LINE_AA)
            knob(e, 6 * k); pans.append((e[0], py - W * 0.085))
        bar(ends[0], ends[1], 15 * k)
        for i in range(-3, 4):                                                       # さおの 目もり(きざみ)
            if i == 0: continue
            p = (B[0] + ca * hl * i / 4.0, B[1] + sa * hl * i / 4.0); cv2.line(S, ip((p[0], p[1] - 4 * k)), ip((p[0], p[1] + 4 * k)), 0.08, 1, cv2.LINE_AA)
        knob(B, 14 * k, 0.55); knob((B[0], B[1] - 24 * k), 6 * k); bar((B[0], B[1] - 13 * k), (B[0], B[1] - 20 * k), 4 * k)
        S = cv2.GaussianBlur(S, (0, 0), 0.9 * k); M = cv2.GaussianBlur(M, (0, 0), 0.6 * k)
        S = S * (0.78 + 0.44 * s.tex(3, s.XX * 1.7, s.YY * 0.22))
        dk = np.array([0.17, 0.09, 0.03], np.float32); md = np.array([0.88, 0.60, 0.20], np.float32); hi = np.array([1.25, 1.15, 0.92], np.float32)
        a1 = sstep(0.0, 0.55, S)[:, :, None]; a2 = sstep(0.55, 1.0, S)[:, :, None]; metal = dk + (md - dk) * a1 + (hi - md) * a2
        b = sstep(0.42, 0.58, s.xn)[:, :, None]; g = metal.mean(2, keepdims=True); metal = metal * (1 - 0.6 * b) + g * np.array([0.66, 0.70, 1.15], np.float32)[None, None, :] * 0.6 * b      # 夜の側は 月の色
        a = (M * lit)[:, :, None]; f *= (1 - a); f += metal * a
        s.flare(f, B, 9 * k, WHITE * 0.6 + GOLD * 0.4, 0.5 * lit, 0.5, t * 0.4, False)
        if orbs: s.sun(f, pans[0], W * 0.060, t, lit); s.moon(f, pans[1], W * 0.060, t, lit)
        return pans

    # ---- 金環の 門: 外は コロナ、中は 光と 夜の うず ----
    def portal(s, f, C, R, rot, opn, e, t, v=1.0):
        W, H, k = s.W, s.H, s.k; r, th = s.polar(C, R); r = r + 1e-4; lr = np.log(r + 0.04)
        ins = sstep(1.0, 0.94, r); ph = th - rot - 2.4 * lr
        side = 0.5 + 0.5 * np.tanh(2.6 * np.sin(ph))
        fl = t * (50 + 260 * e)
        n = s.tex(0, ph / 6.2832 * 512, lr * 70 - fl); n2 = s.tex(2, ph / 6.2832 * 1024, lr * 150 - fl * 2.1)
        n3 = s.tex(3, ph / 6.2832 * 2048, lr * 420 - fl * 4.0)
        I = (0.16 + 1.0 * n ** 1.6 + 0.6 * n2 ** 3 + 0.5 * n3 ** 4 * sstep(0.15, 0.5, r)) * (0.28 + 0.95 * np.clip(r, 0, 1.3))
        edge = np.exp(-(np.sin(ph) / 0.10) ** 2) * sstep(0.05, 0.3, r)                                              # 光と 夜の さかい目
        col = I[:, :, None] * ((GOLD * np.array([1.45, 1.2, 0.8], np.float32))[None, None, :] * (1 - side)[:, :, None] + (VIO * 0.80)[None, None, :] * side[:, :, None]) + (edge * 0.55)[:, :, None]
        col += (np.exp(-(r / 0.15) ** 2) * (0.7 + 3.2 * e) + np.exp(-r / 0.40) * (0.15 + 0.5 * e))[:, :, None] * np.array([1.0, 0.95, 0.85], np.float32)[None, None, :]
        f *= (1 - ins)[:, :, None]; f += col * (ins * opn)[:, :, None]
        if opn < 1: f += (ins * (1 - opn) * (0.03 + 0.05 * n))[:, :, None] * VIO[None, None, :]
        na = s.tex(1, th / 6.2832 * 512 + t * 9, (r - 1) * 46 - t * 26); wr = (0.5 + 0.5 * np.cos(th))[:, :, None]
        cor = np.exp(-np.clip(r - 1, 0, None) / 0.45) * (r > 0.97) * (0.22 + 1.25 * na ** 2)
        f += (cor * v)[:, :, None] * (GOLD[None, None, :] * (1 - wr) + (VIO * 0.85 + 0.15)[None, None, :] * wr)
        f += (np.exp(-((r - 1) / 0.035) ** 2) * 2.3 * v)[:, :, None] * SUNC[None, None, :]

    def frame(s, t):
        o = {}; s._rdir = (-0.8, -0.5); sat = 1.12; expo = 1.0; thr = 0.80
        if t < T_CUT:
            f, sh, g = s.sc_void(t); m = s._mix; expo = 1.0 + 1.2 * (1 - m); thr = 0.80 + 2.4 * (1 - m); o = dict(vig=0.55 - 0.3 * (1 - m))
        elif t < T_SCALE: f, sh, g = s.sc_cut(t - T_CUT); o = dict(vig=0.45); thr = 0.9
        elif t < T_OPEN:
            s._rdir = (0.0, -1.0); f, sh, g = s.sc_scale(t); sat = 1.12 * (1 - 0.85 * ease((t - T_STILL) / 0.5))
            if t < T_STILL: f = s.godrays(f, (s.W * 0.14, s.H * 0.24), 0.25, 1.0, 8, 0.86, 0.3)
        else: s._rdir = (0.0, -1.0); f, sh = s.sc_judge(t)
        f, sh = s.finale(f, sh, t, DUR, None, (1, 0.85, 0.4), (0.56, 0.4, 1.0), -1)
        return s.film(f, t, sh * s.k, 0.55, thr, (1.0, 0.96, 0.9), 1.0 + 0.15 * min(sh, 8), 0.04, o.get("vig", 0.55), sat, expo)

    # 4) 裁き: 開眼 → 太陽と 月が 重なり 金環 → 指さす → 金環が こちらへ。くぐって のみこまれる
    def sc_judge(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_OPEN; pt = t - T_POINT; go = t - T_GO; e = clamp(go / (T_HIT - T_GO)); ee = e ** 2.2
        f = s.twin(t, 1.0, (0.85 + 0.6 * math.exp(-u * 3)) * (1 - 0.6 * e))
        O = (W / 2, H * 0.25); R0 = W * 0.125; m = ease2(u / 1.1)
        if u < 0.9: s.scales(f, 0.0, t, 1 - ease(u / 0.8), False)                                                     # 役目を 終えた 天秤は 消える
        p0 = (W / 2 - W * 0.36, H * 0.32 - W * 0.085)
        if m < 1:
            x = p0[0] + (O[0] - p0[0]) * m; y = p0[1] + (O[1] - p0[1]) * m - H * 0.06 * math.sin(m * math.pi); rr = W * 0.060 + (R0 - W * 0.060) * m
            s.sun(f, (x, y), rr, t); s.moon(f, (W - x, y), rr * 0.97, t)
            C = O
        else:
            v = u - 1.1; C = (W / 2, H * (0.25 + 0.21 * ease2(e))); R = R0 * (1 + 0.03 * math.sin(t * 9) * (1 - e)) * (1 + 16 * ee)
            s.portal(f, C, R, t * 0.5 + 3.2 * ee, ease(e / 0.28), e, t)
            if v < 0.5: s.flare(f, (C[0] - R * 0.71, C[1] - R * 0.71), 16 * k * (1 - v / 0.5) + 1, SUNC, 1.4 * (1 - v / 0.5), 1.0, 0.4, True)   # ダイヤモンドリング
            L = s.L0()
            for i in range(2):
                q = (v - i * 0.12) / 0.9
                if 0 < q < 1: s.circ(L, O, R0 + W * 1.0 * q, (1 - q) ** 2 * 0.7, 2)
            if v < 1.1: s.glowmask(f, L, SUNC, 0.7, 1.0, 1.0, 10.0)
        a_ = 1 - clamp((e - 0.48) / 0.2); cy = H * (0.72 + 0.10 * ee * 4)
        if e > 0: s.rushdust(f, t, C, 900, 0.5 + 2.6 * e, GOLD * 0.8 + 0.2, 0.15 + 0.6 * e, 4, 1.0); s.rushdust(f, t * 1.07, C, 900, 0.5 + 2.6 * e, VIO * 0.7 + 0.3, 0.15 + 0.6 * e, 9, 1.0)
        if a_ > 0.02:
            if pt < 0:
                s.hero(f, 3, t, W / 2, cy, 0.47, a_, sway=0.8 * math.exp(-u * 3), squash=-0.05 * math.exp(-u * 7)); s.gaze(f, 1.0, ICE, 1.2)
            else:
                s.hero(f, 4, t, W / 2, cy, 0.47 + 0.03 * ease(pt / 0.3), a_, squash=-0.04 * math.exp(-pt * 9)); s.gaze(f, 0.9 * a_, ICE, 0.8)
                fp = s.at(*s.P(FINGER4)); bv = (1 - clamp(pt / 0.75)) ** 1.5
                if bv > 0.01:                                                        # 指先から こちらへ のびる 光(手前ほど 太い)
                    g = ease(pt / 0.14); d = (0.50, 0.87); ln = H * 0.9 * g; L = s.L0()
                    for sp_, val in ((0.10, 0.35), (0.035, 1.0)):
                        pts = np.array([fp, (fp[0] + (d[0] * math.cos(sp_) - d[1] * math.sin(sp_)) * ln, fp[1] + (d[1] * math.cos(sp_) + d[0] * math.sin(sp_)) * ln), (fp[0] + (d[0] * math.cos(sp_) + d[1] * math.sin(sp_)) * ln, fp[1] + (d[1] * math.cos(sp_) - d[0] * math.sin(sp_)) * ln)], np.int32)
                        cv2.fillPoly(L, [pts], val, cv2.LINE_AA)
                    s.glowmask(f, cv2.GaussianBlur(L, (0, 0), 1.5 * k), WHITE * 0.7 + GOLD * 0.3, 1.1 * bv, 1.0, 1.0, 12.0)
                s.flare(f, fp, (9 + 6 * math.sin(t * 14) ** 2) * k, WHITE * 0.6 + GOLD * 0.4, 1.0 * a_, 0.8, t * 2, False)
                if e < 0.35 and go > 0: s.glowline(f, [fp, C], SUNC, 1.0, (0.35 + 0.2 * math.sin(t * 31)) * (1 - e / 0.35), 0.8)      # 指と 金環を つなぐ 糸
        f = s.godrays(f, C, 0.22 + 0.55 * e, 0.95, 10, 0.88, 0.5)
        if u < 0.12: f = f + (1 - u / 0.12) * 0.9
        if 0 <= u - 1.1 < 0.12: f = f + (1 - (u - 1.1) / 0.12) * 0.8
        if 0 <= pt < 0.08: f = f + (1 - pt / 0.08) * 0.5
        if e > 0.86: f = f + ((e - 0.86) / 0.14) ** 2 * 1.3
        return f, 8 * math.exp(-u * 5) + (5 * math.exp(-go * 4) + 1.5 + 10 * ee if go > 0 else 0)

    # ---- この子だけの曲: 光の調べ(リディア)と 夜の調べ(フリギア)が 交互に鳴り、釣り合った所で 無音。開眼で 同時に鳴り、金環が 来るにつれ 2つが まざって ふくらむ ----
    def audio(s, path):
        a = Snd(DUR, 83); D = 146.83; lyd = [0, 2, 4, 6, 7, 9, 11]; phr = [0, 1, 3, 5, 7, 8, 10]
        def fl(d, o=0): return D * 2 ** ((lyd[d % 7] + 12 * (d // 7 + o)) / 12.0)
        def fp(d, o=0): return D * 2 ** ((phr[d % 7] + 12 * (d // 7 + o)) / 12.0)
        a.a += a.tone(D / 2) * 0.08 * np.clip(a.t / 2, 0, 1) * a.gate(0, T_STILL, 0.3)
        a.rise(0.3, T_TURN, 1200, 200, 0.05); a.whoosh(0.5, T_TURN, 0.35, 0.05)                                    # 星が 吸いこまれる
        a.boom(T_TURN, 40, 1.3, 0.7); a.note(fl(7, 1), T_TURN, 1.4, 0.12, "bell"); a.note(fp(7, 0), T_TURN, 1.4, 0.10, "bell")
        for j in range(3): a.boom(T_CUT + j * 0.14, 60 - j * 6, 0.8, 0.2); a.note([fl(4, 1), fl(7, 1), fp(8, 1)][j], T_CUT + j * 0.14, 0.5, 0.10, "bell")
        for i in range(5): a.note(fl([0, 4, 7, 4, 0][i], 1), T_CUT + 0.75 + abs(i - 2) * 0.12 + i * 0.001, 0.4, 0.09, "bell")
        for d in (0, 4): a.note(fl(d, -1), T_CUT + 0.75, 1.6, 0.06, "pad", 0.2)
        k = 0; per = math.pi / 4.2                                                                                   # 天秤が かたむくたび 光と夜が 交互に
        while T_SCALE + k * per < T_STILL - 0.05:
            tk = T_SCALE + k * per; g = math.exp(-k * per * 0.55)
            a.a += (a.nz - a.lp(a.nz, 0.4)) * a.env(tk, 0.001, 0.03) * 0.25 * g                                      # かちり
            if k % 2 == 0:
                for j, d in enumerate((0, 2, 4, 6)): a.note(fl(d + (k // 2) % 3, 0), tk + j * 0.06, 0.5, 0.075 * (0.5 + g), "bell")
            else:
                for j, d in enumerate((7, 5, 3, 1)): a.note(fp(d, -1), tk + j * 0.06, 0.5, 0.085 * (0.5 + g), "fm")
            a.boom(tk, 46, 0.35 * g, 0.2); k += 1
        a.note(fl(14, 1), T_STILL + 0.1, 0.9, 0.07, "sine")                                                          # 釣り合い: 細い一音だけ
        a.rise(T_STILL + 0.5, T_OPEN, 100, 180, 0.07)
        a.boom(T_OPEN, 38, 1.5, 0.9)                                                                                 # 開眼: 両方の和音が 同時に
        for d in (0, 2, 4, 6, 7): a.note(fl(d, 0), T_OPEN, 1.8, 0.045, "pad", 0.05)
        for d in (0, 3, 4): a.note(fp(d, -1), T_OPEN, 1.8, 0.055, "saw", 0.05)
        a.rise(T_OPEN, T_OPEN + 1.1, 200, 800, 0.10); a.boom(T_OPEN + 1.1, 55, 1.0, 0.4); a.note(fl(11, 1), T_OPEN + 1.1, 1.2, 0.12, "bell")
        a.note(fl(14, 2), T_POINT, 0.5, 0.11, "sine"); a.whoosh(T_POINT, T_POINT + 0.4, 0.4, 0.4); a.boom(T_POINT, 70, 0.5, 0.2)
        a.boom(T_GO, 36, 1.4, 0.8); a.rise(T_GO, T_HIT, 70, 260, 0.15); a.whoosh(T_GO + 0.5, T_HIT, 0.9, 0.3)
        a.a += a.lp(a.nz, 0.05) * np.clip((a.t - T_GO) / (T_HIT - T_GO), 0, 1) ** 2 * (a.t < T_HIT) * 0.7
        for j, (d, o) in enumerate(((0, -2), (0, -1), (4, -1), (0, 0), (2, 0), (4, 0), (6, 0), (7, 0))): a.note(fl(d, o), T_GO + j * 0.04, T_HIT - T_GO + 0.3, 0.05, "pad", 0.3)
        for d in (0, 3, 4): a.note(fp(d, -1), T_GO + 0.6, T_HIT - T_GO - 0.3, 0.04, "saw", 0.4)
        n = 16
        for i in range(n):                                                                                           # 光と 夜が かわるがわる、だんだん はやく
            tk = T_GO + 0.15 + (T_HIT - T_GO - 0.2) * (1 - (1 - i / n) ** 1.6)
            if i % 2 == 0: a.note(fl(i // 2 + 4, 0), tk, 0.35, 0.09, "bell")
            else: a.note(fp(12 - i // 2, 0), tk, 0.35, 0.08, "fm")
        a.boom(T_HIT, 34, 1.7, 0.9)
        for j, d in enumerate((0, 4, 7, 11, 14)): a.note(fl(d, 0), T_HIT + 0.02, 0.6, 0.06, "bell")
        s.fin_snd(a, DUR, 83); a.save(path)


if __name__ == "__main__":
    r = Aion()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
