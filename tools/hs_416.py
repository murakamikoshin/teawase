"""シュテン(★6 酒呑の鬼王) 専用の必殺技「百鬼ノ王」— 高品質版(hs_fx)。酒を 吹いて 大太刀に 火を つけ、山に 百鬼の 目が ひらき、赤黒い 三日月の 斬撃と ともに 押しよせる
夜の 山(赤い もや・山なみ)。ひょうたんの 酒を あおり、刃に 吹きかけると 赤黒い 火が つく(前ふり) → 目の カットイン+技名 →
うしろの 山に 鬼の 目と 角が 手前から 奥へ 波の ように 灯る(百鬼) → 大太刀を ふりぬく:
赤黒い 炎の 三日月が こちらへ 飛び、百鬼の 目と 角が いっせいに 押しよせる → 三日月が 画面を おおい 炎に のまれる → 白 → 暗転
使い方: python3 hs_416.py 出力.mp4 / python3 hs_416.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 11.2
T_CUT, T_EYES, T_SLASH, T_HIT = 3.0, 5.0, 7.5, 10.5
RED = np.array([1.0, 0.16, 0.10], np.float32); EMB = np.array([1.0, 0.50, 0.16], np.float32); PALE = np.array([1.0, 0.90, 0.70], np.float32); INK = np.array([0.03, 0.012, 0.03], np.float32)
DR = [(0.0, (0.0, 0.0, 0.0)), (0.28, (0.22, 0.0, 0.03)), (0.58, (0.80, 0.07, 0.05)), (0.84, (1.0, 0.45, 0.14)), (1.0, (1.0, 0.92, 0.72))]
EYE = (515, 245); GOURD = (650, 460); GUARD = (310, 490); TIP = (30, 725); MOUTH = (515, 300)


class Shuten(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 416, (1.0, 0.25, 0.15), "百鬼ノ王", "King of a Hundred Demons", W); s.fx_init(416)
        s._vt = [s.text_img(c, hz.SERIF, int(78 * s.k), 3) for c in "百鬼ノ王"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; s.hy = 0.70; H = s.H; k = s.k
        s._rg = [(0.30 + 0.11 * i + 0.07 * s.tex(i % 4, s.XX * (0.22 + 0.12 * i), 13.0 * (i + 1)) + 0.02 * s.tex((i + 2) % 4, s.XX * 1.1, 7.0 * (i + 1))).astype(np.float32) for i in range(3)]
        rg = np.random.RandomState(4160); n = 100; d = np.sort(rg.rand(n))[::-1]                                         # 百鬼: 手前(下・大)から 奥(上・小)へ
        x = 0.04 + 0.92 * rg.rand(n); y = 0.60 - 0.50 * (1 - d) ** 0.8 + 0.03 * rg.rand(n); keep = ~((np.abs(x - 0.5) < 0.22) & (y > 0.40))
        s._dm = np.stack([x, y, 0.35 + 0.65 * d, rg.rand(n), rg.rand(n), d], 1)[keep].astype(np.float32)

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 夜の 山 ----
    def mount(s, t, lum=1.0, red=0.0):
        W, H, k = s.W, s.H, s.k; hy = s.hy
        f = INK[None, None, :] * (0.8 + 2.0 * (s.yn / hy).clip(0, 1) ** 1.5)[:, :, None] + (np.array([0.30, 0.04, 0.06], np.float32) * (0.5 + 0.9 * red))[None, None, :] * np.exp(-((s.yn - 0.42) / 0.20) ** 2)[:, :, None]
        f = f + s.nebula(t, [(0.10, 0.02, 0.08), (0.22, 0.03, 0.06), (0.55, 0.14, 0.10)], 0.0008, (4.0, 1.0), 0.6 + 0.3 * red, 0)
        s.starfield(f, t, 500, (1.0, 0.85, 0.8), 2, 0.6 * (1 - 0.6 * red), (0.0, 0.0))
        f += (np.exp(-np.hypot(s.xn - 0.5, (s.yn - 0.36) * 0.9) / 0.20) * (0.30 + 0.5 * red))[:, :, None] * np.array([0.9, 0.12, 0.08], np.float32)[None, None, :]      # 山の むこうの 赤い 光
        for i, r in enumerate(s._rg):
            m = sstep(r - 0.003, r + 0.003, s.yn)[:, :, None] * (s.yn < hy + 0.02)[:, :, None]; dr = np.clip((s.yn - r) / 0.10, 0, 1)
            col = (np.array([0.045, 0.012, 0.035], np.float32) * (1.0 - 0.25 * i))[None, None, :] * (0.7 + 0.6 * s.tex(2, s.XX * 0.9, s.YY * 0.9))[:, :, None] + (np.array([0.5, 0.06, 0.05], np.float32) * (0.5 + red))[None, None, :] * (np.exp(-dr / 0.10) * (0.5 - 0.12 * i))[:, :, None]
            f = f * (1 - m) + col * m
            ms = s.fbm(0.0018, t * (14 + 6 * i), i * 40.0, 3, 30 * k, i) * np.exp(-((s.yn - r - 0.05) / 0.05) ** 2); f += (sstep(0.35, 0.8, ms) * (0.10 + 0.22 * red))[:, :, None] * np.array([0.8, 0.10, 0.10], np.float32)[None, None, :]      # 赤い もや
        d = np.clip((s.yn - hy) / (1 - hy), 0, 1); gm = sstep(hy - 0.003, hy + 0.003, s.yn)[:, :, None]; gn = s.tex(1, s.XX * 0.8, s.YY * 2.4 / (0.4 + d)) * 0.6 + s.tex(3, s.XX * 1.7, s.YY * 1.7) * 0.4
        g = np.array([0.035, 0.014, 0.03], np.float32)[None, None, :] * (0.5 + 1.3 * gn)[:, :, None] + (np.array([0.5, 0.07, 0.05], np.float32) * (0.25 + 0.8 * red))[None, None, :] * (np.exp(-d / 0.08) * (0.4 + gn))[:, :, None]
        f = f * (1 - gm) + g * gm
        em = np.zeros_like(f); s.starfield(em, t, 160, tuple(EMB), 9, 1.3 * (0.4 + red), (10.0, -50.0)); f += em
        return (f * lum).astype(np.float32)

    # ---- 百鬼: つり上がった 目 2つ と 角 2本(顔は 描かない)----
    def demons(s, f, t, lit, rush=0.0, C=None, v=1.0):
        """lit: 0..1(手前から 奥へ 灯る)、rush: 0..1(こちらへ 押しよせる)"""
        if lit <= 0 or v <= 0.01: return
        W, H, k = s.W, s.H, s.k; E = s.L0(); Hn = s.L0(); C = C or (W / 2, H * 0.42)
        for i in range(len(s._dm)):
            x, y, sz, r1, r2, d = s._dm[i]; a = clamp((lit * 1.25 - (1 - d)) * 4)
            if a <= 0: continue
            px = x * W; py = y * H; z = (11 + 34 * sz ** 1.5) * k * 0.72
            if rush > 0: g = rush ** 2 * (0.6 + 3.2 * d + 1.5 * r1); px = C[0] + (px - C[0]) * (1 + g); py = C[1] + (py - C[1]) * (1 + g) + rush * rush * H * 0.05; z = z * (1 + g * 0.9)
            if px < -80 * k or px > W + 80 * k or py < -80 * k or py > H + 80 * k: continue
            bl = 1.0 if math.sin(t * (1.5 + 2 * r1) + r2 * 30) > -0.96 else 0.1; a = a * bl * (0.75 + 0.25 * math.sin(t * 5 + r2 * 20))
            for sg in (-1, 1):
                cv2.ellipse(E, (int(px + sg * z * 0.55), int(py)), (max(1, int(z * 0.36)), max(1, int(z * 0.11))), float(sg * 22), 0, 360, float(a), -1, cv2.LINE_AA)
                pts = [(px + sg * z * (0.55 + 0.55 * q + 0.25 * q * q), py - z * (0.55 + 1.6 * q - 0.35 * q * q)) for q in (0.0, 0.3, 0.6, 0.85, 1.0)]
                cv2.polylines(Hn, [np.array(pts, np.int32)], False, float(a * 0.8), max(1, int(z * 0.24)), cv2.LINE_AA)
        Eb = cv2.GaussianBlur(E, (0, 0), 0.9 * k)
        f += (Eb * 1.6 * v)[:, :, None] * (EMB * 0.6 + PALE * 0.4)[None, None, :] + (cv2.GaussianBlur(E, (0, 0), 6 * k) * 1.3 * v)[:, :, None] * RED[None, None, :]
        Hb = cv2.GaussianBlur(Hn, (0, 0), 0.8 * k); f += (Hb * 0.55 * v)[:, :, None] * np.array([0.9, 0.2, 0.14], np.float32)[None, None, :] + (cv2.GaussianBlur(Hn, (0, 0), 5 * k) * 0.35 * v)[:, :, None] * RED[None, None, :]

    # ---- 赤黒い 炎の 三日月(飛ぶ 斬撃)----
    def crescent(s, f, c, S, t, v=1.0, rot=-0.30):
        if v <= 0.01 or S < 4: return
        cr, sr = math.cos(rot), math.sin(rot); dx = s.XX - c[0]; dy = s.YY - c[1]; x = (dx * cr + dy * sr) / S; y = (-dx * sr + dy * cr) / S
        r1 = np.hypot(x, y); th = np.arctan2(y, x); r2 = np.hypot(x, y - 0.24) / 0.955
        teeth = 0.035 * np.cos(th * 46 + t * 6) * sstep(0.6, 1.2, np.abs(th + 1.5708) * 0 + 1)                          # 内がわの ふち: 刃文の ような 波
        ins = sstep(1.0, 0.985, r1) * sstep(1.0 + teeth - 0.01, 1.0 + teeth + 0.02, r2) * (y < 0.25)
        dep = np.clip((1.0 - r1) / 0.22, 0, 1)                                                                          # 外の ふちからの 深さ
        n = s.tex(1, th / 6.2832 * 768 + t * 30, r1 * 26 - t * 90); n2 = s.tex(3, th / 6.2832 * 1536, r1 * 70 - t * 220)
        val = np.clip(1.05 * np.exp(-(dep / 0.10) ** 2) + (0.55 * n ** 1.3 + 0.35 * n2 ** 3) * (1 - 0.55 * dep) + 0.18, 0, 1)
        f *= (1 - ins * 0.94 * v)[:, :, None]; f += s.ramp(val, DR) * (ins * 1.5 * v)[:, :, None]
        wk = sstep(1.0 + teeth + 0.02, 1.0 + teeth - 0.03, r2) * np.exp(-np.clip(1.0 - r2, 0, None) / 0.55) * (r1 < 1.0) * sstep(0.55, -0.05, y)                               # うしろに 引く 炎
        wv = np.clip((0.35 + 0.75 * n ** 1.4 + 0.4 * n2 ** 3) * wk, 0, 1); f *= (1 - 0.8 * wk * v)[:, :, None]; f += s.ramp(wv, DR) * (1.25 * v)
        f += (np.exp(-((r1 - 1.0) / 0.012) ** 2) * (y < 0.2) * sstep(-0.05, 0.2, r2 - 1.0) * 1.8 * v)[:, :, None] * PALE[None, None, :]      # 先の ふち(白く 熱い)
        f += (np.exp(-np.clip(r1 - 1.0, 0, None) / 0.10) * (r1 > 1.0) * (y < 0.1) * 0.5 * v)[:, :, None] * RED[None, None, :]

    def bladefire(s, f, t, v):
        """刃に そって 燃える 赤黒い 火"""
        if v <= 0.01: return
        k = s.k; a = s.at(*s.A(*GUARD)); b = s.at(*s.A(*TIP)); L = s.L0(); cv2.line(L, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 1.0, max(2, int(16 * k)), cv2.LINE_AA)
        m = cv2.GaussianBlur(L, (0, 0), 7 * k); m = np.maximum(m, cv2.warpAffine(m, np.float32([[1, 0, 4 * k], [0, 1, -16 * k]]), (s.W, s.H)) * 0.8)
        fl = s.flame(t, np.clip(m * 2.2, 0, 1), 300.0, 0.003, 1.1); f += s.ramp(np.clip(fl * v, 0, 1), DR) * 1.4
        s.flare(f, b, 7 * k, EMB, 0.6 * v, 0.6, t, False)

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        w = getattr(s, '_fr', 0.0)
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(RED * 0.6 + EMB * 0.4), rim_dir=(-0.5 * w, -1.0 + 0.6 * w), rim_v=(0.45 + 0.45 * w) * alpha, amb=(1.0, 0.7, 0.65), amb_v=0.14, aura=RED, aura_v=0.10 + 0.14 * w, alpha=alpha, shade=0.80 + 0.12 * w)

    def frame(s, t):
        o = {}; s._fr = 0.0
        if t < T_CUT: f, sh = s.sc_drink(t)
        elif t < T_EYES: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.92)
        else: f, sh = s.sc_horde(t)
        f, sh = s.finale(f, sh, t, DUR, None, (1, 0.3, 0.2), (1, 0.6, 0.2), -1)
        return s.film(f, t, sh * s.k, 0.6, o.get("thr", 0.80), (1.0, 0.75, 0.65), 1.0 + 0.15 * min(sh, 8), 0.04, o.get("vig", 0.60), 1.15, 1.12)

    def king(s, f, t, alpha=1.0, dy=0.0, **kw):
        W, H, k = s.W, s.H, s.k; cy = H * 0.675 + dy; fy = H * 0.675 + 0.42 * H * 0.46
        sh_ = s.L0(); cv2.ellipse(sh_, (int(W / 2), int(fy)), (int(W * 0.24), int(H * 0.013)), 0, 0, 360, 0.6 * alpha, -1, cv2.LINE_AA); f *= (1 - cv2.GaussianBlur(sh_, (0, 0), 4 * k))[:, :, None]
        img = s.pose(t, **kw); s.draw(f, img, W / 2, cy, 0.42, alpha)

    # 1) 夜の 山。酒を あおり、刃に 吹きかけて 火を つける
    def sc_drink(s, t):
        W, H, k = s.W, s.H, s.k; dr = math.sin(clamp((t - 1.1) / 0.8) * 3.1416); sp = t - 2.05; s._fr = ease(sp / 0.5) if sp > 0 else 0.0
        f = s.mount(t, 1.0, 0.25 * s._fr)
        s.king(f, t, 1.0, 0.0, sway=0.2, breath=0.5, lean=7 * k * dr - (5 * k * math.exp(-sp * 5) if sp > 0 else 0), squash=-0.02 * dr)
        gp = s.at(*s.A(*GOURD)); s.flare(f, gp, (5 + 9 * dr) * k, PALE, 0.4 + 0.5 * dr, 0.5, t, False)
        if 0 < sp < 0.6:                                                             # 吹きかける しぶき
            mp = s.at(*s.A(*MOUTH)); tp = s.at(*s.A(*GUARD)); rg = s.rnd[:160]; q = np.clip(sp / 0.35 - rg[:, 0] * 0.5, 0, 1)
            x = mp[0] + (tp[0] - 40 * k - mp[0]) * q + (rg[:, 1] - 0.5) * 70 * k * q; y = mp[1] + (tp[1] + 60 * k - mp[1]) * q + (rg[:, 2] - 0.5) * 60 * k * q
            s.splat(f, x, y, rg[:, 3] * 0.9 * (q > 0) * (1 - sp / 0.6), PALE * 0.6 + EMB * 0.4, 1.2, 0.5)
        s.bladefire(f, t, s._fr)
        if 0 <= sp < 0.1: f = f + (1 - sp / 0.1) * 0.6
        return f * clamp(t / 0.5), 5 * math.exp(-sp * 6) if sp > 0 else 0

    # 2) 目の カットイン(炎の ふちの 帯)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT; s._fr = 1.0
        f = s.mount(t, 0.30, 0.3)
        a = ease(u / 0.26); cy = H * 0.33; hh = H * 0.105
        band = np.array([0.05, 0.0, 0.02], np.float32)[None, None, :] + s.ramp(np.clip(s.flame(t, np.ones((H, W), np.float32), 320.0, 0.002, 0.95), 0, 1), DR) * 0.6
        ex, ey = s.A(*EYE); zf = 1.5; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(band, img, W / 2 - (ex - s.aw / 2) * sc + (1 - a) * W * 0.9 - u * 6 * k, cy - (ey - s.ah / 2) * sc, zf)
        jag = (s.tex(1, s.XX * 1.4 + t * 60, 4.0) - 0.5) * 22 * k; dd = np.abs(s.YY - cy); edge = hh * a + jag * a
        m = (sstep(edge, edge - 2, dd) * (a > 0.02))[:, :, None]; f = f * (1 - m) + band * m
        s.glowmask(f, np.exp(-((dd - edge) / (2.0 * k)) ** 2) * (a > 0.02), EMB, 1.0 * a, 1.0, 0.9, 9.0)
        n = len(s._vt); step = W * 0.205; sh = 5 * math.exp(-u * 6)
        for i, T in enumerate(s._vt):                                                # 技名: 1文字ずつ 重く 落ちる
            p = (u - 0.55 - i * 0.18) / 0.12
            if p > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.68 - (1 - ease(p)) * 70 * k, clamp(p * 2), 1.0); sh += 7 * math.exp(-max(0.0, p - 1) * 1.6) * (p > 1) * (p < 3.5)
        for i, T in enumerate(s._vo):
            if u > 0.35 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.68 - 78 * k, 1.0, 1.0)
        if u > 1.45:
            g = ease((u - 1.45) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.68 + 60 * k), (W / 2 + W * 0.42 * g, H * 0.68 + 60 * k)], RED * 0.7 + EMB * 0.3, 1.0, 0.8, 0.8)
            s.text(f, s.txt_en, W / 2, H * 0.68 + 86 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.7
        return f, sh

    # 3) 百鬼の 目が 灯る → ふりぬく → 三日月と 百鬼が 押しよせる
    def sc_horde(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_EYES; us = t - T_SLASH; sh = 0.0
        lit = clamp(u / 2.0); e = clamp((us - 0.12) / (T_HIT - T_SLASH - 0.12)) if us > 0 else 0.0; ee = e ** 2.3
        s._fr = 1.0
        f = s.mount(t, 1.0 - 0.3 * e, 0.3 + 0.7 * lit)
        C = (W / 2, H * 0.44)
        s.demons(f, t, lit, e, C, 1.0 - 0.4 * clamp((e - 0.7) / 0.3))
        a_ = 1 - clamp((e - 0.30) / 0.22)
        if a_ > 0.02:
            if us < 0: g = ease(clamp((u - 1.5) / 0.8)); s.king(f, t, a_, -H * 0.012 * g, sway=0.2, breath=0.4, lean=9 * k * g, squash=-0.04 * g)        # ためる
            else: s.king(f, t, a_, H * 0.006, sway=0.5, lean=-12 * k * math.exp(-us * 2.5), squash=0.06 * math.exp(-us * 7), jit=2 * k * math.exp(-us * 5))
            s.bladefire(f, t, a_ * (1.0 + 0.5 * clamp((u - 1.5) / 0.8)))
        if us > 0:
            if us < 0.30:                                                            # ふりぬく 光の 弧
                q = us / 0.30; pts = [(W * 0.5 + math.cos(a) * W * 0.50, H * 0.66 + math.sin(a) * H * 0.20) for a in np.linspace(2.9 - 3.4 * q, 2.9 - 3.4 * max(0.0, q - 0.55), 16)[::-1]]
                L = s.L0(); cv2.polylines(L, [np.array(pts, np.int32)], False, 1.0, max(2, int(9 * k)), cv2.LINE_AA); s.glowmask(f, cv2.GaussianBlur(L, (0, 0), 1.2 * k), PALE * 0.6 + RED * 0.4, 1.3 * (1 - q * 0.5), 1.0, 1.2, 12.0)
            S = W * (0.18 + 0.34 * ease(us / 0.25)) * (1 + 5.5 * ee); cc = (C[0], H * (0.66 - 0.10 * ease2(e)) + S * 0.55 * (1 - 0.3 * e))
            s.crescent(f, cc, S, t, min(1.0, us * 8))
            s.rushdust(f, t, C, 1200, 0.7 + 2.4 * e, EMB * 0.7 + RED * 0.3, 0.2 + 0.6 * e, 6, 1.0)
            f = s.godrays(f, C, 0.2 + 0.4 * e, 0.95, 8, 0.88, 0.45)
            if us < 0.10: f = f + (1 - us / 0.10) * 1.0
            sh += 14 * math.exp(-us * 3) + 2 + 10 * ee
            if e > 0.85: f = f + ((e - 0.85) / 0.15) ** 2 * 1.4
        else: sh += 1.5 * lit * abs(math.sin(u * 5.2))
        if u < 0.12: f = f + (1 - u / 0.12) * 0.3
        return f, sh

    # ---- この子だけの曲: 低い 太鼓と 笛(都節)→ 酒を あおる 一拍 → 火が つく → 百鬼の 太鼓が ふえていく → 一閃 → うなりと 鬨の声の ような 和音 ----
    def audio(s, path):
        a = Snd(DUR, 416); D = 146.83; ms = [0, 1, 5, 7, 8]
        def fq(d, o=0): return D * 2 ** ((ms[d % 5] + 12 * (d // 5 + o)) / 12.0)
        a.a += (a.tone(D / 2) + 0.5 * a.tone(D / 2 * 1.498)) * 0.09 * np.clip(a.t / 1.2, 0, 1) * a.gate(0, T_HIT, 0.3)
        for t0 in (0.3, 0.9): a.boom(t0, 46, 1.0, 0.5)
        for (d, o, t0, ln) in ((3, 1, 0.5, 0.7), (4, 1, 1.25, 0.35), (3, 1, 1.6, 0.4)): a.a += (np.sin(2 * np.pi * fq(d, o) * a.t + 2.5 * np.sin(2 * np.pi * 5.5 * a.t)) + 0.2 * a.lp(a.nz, 0.3)) * a.env(t0, 0.1, ln * 0.6) * 0.08
        a.a += (a.nz - a.lp(a.nz, 0.4)) * a.env(2.05, 0.03, 0.2) * 0.4; a.boom(2.3, 60, 0.8, 0.3); a.a += a.lp(a.nz, 0.12) * a.env(2.3, 0.05, 0.6) * 0.5      # 吹く → 火が つく
        a.boom(T_CUT, 42, 1.3, 0.55); a.whoosh(T_CUT - 0.3, T_CUT, 0.3)
        for i in range(4): a.boom(T_CUT + 0.67 + i * 0.18, 54 - i * 3, 1.1, 0.25); a.note(fq([0, 2, 3, 5][i], 0), T_CUT + 0.67 + i * 0.18, 0.5, 0.10, "saw", 0.01)
        for d in (0, 2): a.note(fq(d, -1), T_CUT + 0.4, 1.6, 0.07, "saw", 0.2)
        n = 26; rg = np.random.RandomState(416)
        for i in range(n):                                                                                              # 百鬼の 太鼓: だんだん ふえて はやく
            tk = T_EYES + (T_SLASH - T_EYES - 0.25) * (i / n) ** 0.8; g = 0.35 + 0.65 * i / n
            a.boom(tk, 58 + (i % 3) * 14, 0.5 * g + 0.2, 0.12); a.hat(tk + 0.04, 0.03 * g)
            if i % 3 == 0: a.note(fq(i // 3 % 5, 0), tk, 0.25, 0.06 * g, "fm")
        for i in range(24): a.note(fq(int(rg.rand() * 5) + 5, 1), T_EYES + rg.rand() * (T_SLASH - T_EYES), 0.08, 0.03, "sine", 0.01)      # 目が ひらく 音
        a.rise(T_EYES + 1.4, T_SLASH, 70, 260, 0.13)
        a.whoosh(T_SLASH - 0.1, T_SLASH + 0.25, 0.8, 0.6); a.boom(T_SLASH + 0.1, 34, 2.0, 0.9); a.note(fq(9, 2), T_SLASH + 0.05, 1.0, 0.10, "bell")
        ts = T_SLASH + 0.12; roar = a.lp(a.nz, 0.05) * (0.55 + 0.45 * np.sin(2 * np.pi * 27 * a.t + 2 * np.sin(2 * np.pi * 4 * a.t)))
        a.a += roar * np.clip((a.t - ts) / 0.2, 0, 1) * (a.t < T_HIT) * (0.9 + 1.2 * np.clip((a.t - ts) / (T_HIT - ts), 0, 1))
        for j, d in enumerate((0, 2, 3, 5, 7)): a.note(fq(d, -1 if j < 2 else 0), ts + j * 0.06, T_HIT - ts + 0.2, 0.055, "saw", 0.3)
        st = 0.24; tk = ts + 0.2
        while tk < T_HIT - 0.1: a.boom(tk, 60, 0.7, 0.1); tk += st; st = max(0.07, st * 0.9)                               # 押しよせる 足音
        a.rise(ts, T_HIT, 60, 240, 0.13)
        a.boom(T_HIT, 34, 1.8, 0.9)
        for j, d in enumerate((0, 3, 5, 8)): a.note(fq(d, 0), T_HIT + 0.02, 0.6, 0.06, "bell")
        s.fin_snd(a, DUR, 416); a.save(path)


if __name__ == "__main__":
    r = Shuten()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
