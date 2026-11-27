"""ヌエ(★5 鵺の仮面の男) 専用の必殺技「鵺ノ夜鳴キ」— 高品質版(hs_fx)。黒雲が 月を かくし、稲光の たびに 近づいて くる
月夜の 御殿の 屋根に 立つ。黒い 雲が 流れこみ、月が かげる(前ふり) → 仮面の 目の カットイン+技名 →
手を かかげると 黒雲が うずを 巻いて 空も 姿も のみこむ(まっくら・細い 鳴き声)→
稲光 1: 遠くの 雲の 中に 小さな 姿 → 闇 → 稲光 2: 近い → 闇 → 稲光 3: 目の前に 仮面 → 闇の 中、仮面の 目だけが 赤く 近づく → 稲妻が こちらへ 落ちる → 白 → 暗転
使い方: python3 hs_415.py 出力.mp4 / python3 hs_415.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.2
T_CUT, T_CLOUD, T_HIT = 2.8, 4.8, 9.45
FL = [(6.75, 0.20, 0.30), (7.60, 0.46, 0.38), (8.40, 1.30, 0.40)]                       # 稲光: (時刻, 姿の 大きさ, 仮面の 高さ)
VIO = np.array([0.62, 0.50, 1.0], np.float32); BOLT = np.array([0.90, 0.86, 1.0], np.float32); MOONC = np.array([0.80, 0.86, 1.0], np.float32); RED = np.array([1.0, 0.18, 0.12], np.float32); NIGHT = np.array([0.03, 0.04, 0.10], np.float32)
EYE = (290, 228); EYES = [(245, 228), (335, 228)]; HAND = (85, 335)


class Nue(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 415, (0.6, 0.5, 1.0), "鵺ノ夜鳴キ", "Nue's Night Cry", W); s.fx_init(415)
        s._vt = [s.text_img(c, hz.SERIF, int(66 * s.k), 3) for c in "鵺ノ夜鳴キ"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; H = s.H; k = s.k; rg = np.random.RandomState(4150); s._bolts = []
        for j, tg in enumerate(((W * 0.30, H * 0.52), (W * 0.74, H * 0.60), (W * 0.42, H * 0.70), (W * 0.50, H * 0.46))):   # 稲妻の 道すじ(主と 枝)
            x0 = W * (0.2 + 0.6 * rg.rand()); pts = [(x0, -10.0)]; n = 13
            for i in range(1, n + 1): q = i / n; pts.append((x0 + (tg[0] - x0) * q + (rg.rand() - 0.5) * 70 * k * math.sin(q * 3.14), tg[1] * q + (rg.rand() - 0.5) * 16 * k))
            br = []
            for b in range(4):
                i0 = rg.randint(3, n - 2); p = pts[i0]; bp = [p]; a = (rg.rand() - 0.5) * 2.2 + 1.57
                for _ in range(5): a += (rg.rand() - 0.5) * 0.9; p = (p[0] + math.cos(a) * 30 * k, p[1] + math.sin(a) * 30 * k); bp.append(p)
                br.append(bp)
            s._bolts.append((pts, br))
        s._roofy = (0.800 - 0.045 * np.cos((s.xn - 0.5) * 3.1416 * 1.5) + 0.05 * np.abs(s.xn - 0.5) ** 2 * 0).astype(np.float32)

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 月夜の 屋根 ----
    def roofnight(s, t, lum=1.0):
        W, H, k = s.W, s.H, s.k
        f = NIGHT[None, None, :] * (0.7 + 1.6 * (s.yn / 0.8).clip(0, 1) ** 1.4)[:, :, None]
        s.starfield(f, t, 500, (0.85, 0.9, 1.0), 2, 0.6, (0.0, 0.0))
        d = np.hypot(s.xn - 0.5, (s.yn - 0.25) * H / W) / 0.17; mn = s.tex(2, s.XX * 0.9, s.YY * 0.9)
        f += (sstep(1.0, 0.96, d) * (0.95 + 0.25 * mn))[:, :, None] * MOONC[None, None, :] * 1.1 + (np.exp(-np.clip(d - 1, 0, None) / 0.9) * 0.45)[:, :, None] * MOONC[None, None, :]
        ry = s._roofy; m = sstep(ry - 0.002, ry + 0.002, s.yn)[:, :, None]; dd = np.clip((s.yn - ry) / 0.2, 0, 1)
        tile = 0.55 + 0.45 * np.abs(np.cos(s.XX * 0.16 + (s.YY - ry * H) * 0.02)) ; row = 1 - 0.5 * sstep(0.85, 0.98, np.abs(np.cos((s.YY - ry * H) * 0.18)))
        col = np.array([0.055, 0.065, 0.10], np.float32)[None, None, :] * (tile * row * (1.4 - 0.8 * dd))[:, :, None] + (MOONC * 0.28)[None, None, :] * (np.exp(-dd / 0.03) * (0.6 + 0.4 * tile))[:, :, None]
        f = f * (1 - m) + col * m
        return (f * lum).astype(np.float32)

    # ---- 黒雲(cover: 0..1 おおう 量、lit: 稲光の 明るさ)----
    def clouds(s, f, t, cover, lit=0.0, swirl=0.0, C=None):
        if cover <= 0.005: return f
        W, H, k = s.W, s.H, s.k; C = C or (W / 2, H * 0.40)
        dx = s.XX - C[0]; dy = s.YY - C[1]; r = np.hypot(dx, dy) / W
        ang = swirl * 2.5 * np.exp(-r * 1.6); X = C[0] + dx * np.cos(ang) - dy * np.sin(ang); Y = C[1] + dx * np.sin(ang) + dy * np.cos(ang)
        n = s.tex(0, X * 0.22 + t * 14, Y * 0.30 + t * 3) * 0.55 + s.tex(2, X * 0.6 - t * 9, Y * 0.7) * 0.30 + s.tex(1, X * 1.5, Y * 1.5 + t * 5) * 0.15
        edge = np.clip(1.25 * np.maximum(np.abs(s.xn - 0.5) * 2, np.abs(s.yn - 0.35) * 1.6), 0, 1.4)                      # ふちから おおって くる
        den = sstep(1.05 - cover * 1.35, 1.30 - cover * 1.35, n * 0.55 + edge * 0.6)
        if cover < 1.0: den = den * sstep(s._roofy + 0.02, s._roofy - 0.10, s.yn) ** (1 - cover)
        rim = sstep(0.0, 0.5, den) * (1 - sstep(0.5, 1.0, den))
        col = np.array([0.018, 0.014, 0.03], np.float32)[None, None, :] * (0.5 + 1.2 * n)[:, :, None] + (VIO * 0.10)[None, None, :] * rim[:, :, None]
        if lit > 0: col = col + ((BOLT * 0.65 + VIO * 0.35) * lit)[None, None, :] * ((0.10 + 0.9 * n ** 2.2) * (0.45 + 0.55 * rim + 0.4 * np.exp(-r * 2.0)))[:, :, None]
        return f * (1 - den)[:, :, None] + col * den[:, :, None]

    def bolt(s, f, j, v, grow=1.0):
        if v <= 0.01: return
        k = s.k; pts, br = s._bolts[j]; n = max(2, int(len(pts) * clamp(grow))); L = s.L0(); L2 = s.L0()
        cv2.polylines(L, [np.array(pts[:n], np.int32)], False, 1.0, max(2, int(3.5 * k)), cv2.LINE_AA)
        for bp in br: cv2.polylines(L2, [np.array(bp, np.int32)], False, 0.7, max(1, int(1.5 * k)), cv2.LINE_AA)
        if grow < 1: L2 *= 0
        L = L + L2; f += ((L * 1.6 + cv2.GaussianBlur(L, (0, 0), 3 * k) * 1.6 + cv2.GaussianBlur(L, (0, 0), 16 * k) * 2.2) * v)[:, :, None] * BOLT[None, None, :] + (L * v)[:, :, None]

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        m = getattr(s, '_lt', 0)
        if m == 1: return s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(BOLT), rim_dir=(0.3, -0.95), rim_v=0.9 * alpha, amb=(0.75, 0.7, 1.0), amb_v=0.30, aura=None, aura_v=0.0, alpha=alpha, shade=0.80)
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(MOONC), rim_dir=(0.0, -1.0), rim_v=0.65 * alpha * getattr(s, '_mo', 1.0), amb=(0.6, 0.7, 1.0), amb_v=0.22, aura=None, aura_v=0.0, alpha=alpha, shade=0.50 + 0.28 * getattr(s, '_mo', 1.0))

    def frame(s, t):
        o = {}; s._lt = 0; s._mo = 1.0
        if t < T_CUT: f, sh = s.sc_roof(t)
        elif t < T_CLOUD: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.92)
        else: f, sh = s.sc_dark(t)
        f, sh = s.finale(f, sh, t, DUR, None, (0.8, 0.75, 1), (0.6, 0.4, 1.0), -1)
        return s.film(f, t, sh * s.k, 0.6, o.get("thr", 0.80), (0.9, 0.88, 1.0), 1.0 + 0.2 * min(sh, 8), 0.045, o.get("vig", 0.60), 1.10, 1.12)

    def stand(s, f, t, alpha=1.0, **kw):
        W, H, k = s.W, s.H, s.k; img = s.pose(t, **kw); s.draw(f, img, W / 2, H * 0.585, 0.40, alpha)

    # 1) 月夜の 屋根。黒雲が 流れこみ、月が かげる
    def sc_roof(s, t):
        W, H, k = s.W, s.H, s.k; cv = 0.34 * ease2((t - 0.8) / 1.9); s._mo = 1 - 0.5 * cv / 0.34
        f = s.roofnight(t, 1.0 - 0.25 * cv / 0.34); f = s.clouds(f, t, cv)
        s.stand(f, t, 1.0, sway=0.25, breath=0.5)
        return f * clamp(t / 0.5), 0

    # 2) 仮面の 目の カットイン(雲の 切れ間の 形)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT; s._mo = 0.5
        f = s.clouds(s.roofnight(t, 0.30), t, 0.34)
        a = ease(u / 0.28); cy = H * 0.33; hh = H * 0.11
        band = np.array([0.03, 0.02, 0.07], np.float32)[None, None, :] + (VIO * 0.35)[None, None, :] * (0.3 + 0.9 * s.tex(0, s.XX * 0.25 + t * 40, s.YY * 0.6))[:, :, None]
        ex, ey = s.A(*EYE); zf = 1.6; sc = zf * H / s.ah; s._lt = 1
        img = s.pose(t, breath=0.2); s.draw(band, img, W / 2 - (ex - s.aw / 2) * sc + (1 - a) * W * 0.9 - u * 5 * k, cy - (ey - s.ah / 2) * sc, zf); s._lt = 0
        puf = (s.tex(2, s.XX * 0.5 + t * 25, 6.0) - 0.5) * 34 * k + (s.tex(1, s.XX * 1.6 - t * 18, 2.0) - 0.5) * 12 * k; dd = np.abs(s.YY - cy); edge = (hh + puf) * a   # もくもくした ふち
        m = (sstep(edge, edge - 3, dd) * (a > 0.02))[:, :, None]; f = f * (1 - m) + band * m
        s.glowmask(f, np.exp(-((dd - edge) / (2.2 * k)) ** 2) * (a > 0.02), VIO, 0.8 * a, 1.0, 0.8, 8.0)
        n = len(s._vt); step = W * 0.165
        for i, T in enumerate(s._vt):                                                # 技名: 闇から にじみ出る
            p = (u - 0.5 - i * 0.14) / 0.45
            if p > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.69, clamp(p) ** 1.5, 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.35 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.69 - 70 * k, 1.0, 1.0)
        if u > 1.45:
            g = ease((u - 1.45) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.69 + 52 * k), (W / 2 + W * 0.42 * g, H * 0.69 + 52 * k)], VIO, 1.0, 0.8, 0.8)
            s.text(f, s.txt_en, W / 2, H * 0.69 + 78 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.6
        return f, 3 * math.exp(-u * 6)

    def flash(s, x):
        """稲光の 明るさ(2度 またたく)"""
        if x < 0: return 0.0
        return max(math.exp(-x * 9), 0.75 * math.exp(-max(0.0, x - 0.11) * 12) * (x > 0.11), 0.0) * (1.0 if x < 0.5 else 0.0)

    # 3) 黒雲に のまれる → 稲光の たびに 近づく → 稲妻が 落ちる
    def sc_dark(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_CLOUD; sh = 0.0
        if u < 1.75:                                                                 # 黒雲が うずを 巻いて のみこむ
            cv = 0.34 + 0.86 * ease2(u / 1.6); s._mo = max(0.0, 0.5 - u * 0.5)
            f = s.roofnight(t, max(0.05, 0.75 - u * 0.5)); a_ = 1 - clamp((u - 0.9) / 0.6)
            if a_ > 0.02: g = ease(u / 0.5); s.stand(f, t, a_, sway=0.4, breath=0.3, lean=4 * k * g, squash=-0.04 * g)
            f = s.clouds(f, t, cv, 0.05 * ease2(u / 1.6), ease2(u / 1.6))
            return f + (1 - u / 0.12) * 0.3 * (u < 0.12), 1.5 * clamp(u / 1.0)
        I = 0.0; idx = -1
        for j, (tf, sz, my) in enumerate(FL):
            x = t - tf; v = s.flash(x)
            if v > I: I = v; idx = j
        last = -1
        for j, (tf, sz, my) in enumerate(FL):
            if t >= tf: last = j
        f = np.zeros((H, W, 3), np.float32) + NIGHT[None, None, :] * 0.25
        f = s.clouds(f, t, 1.2, 0.05 + I * 1.25, 1.0 + 0.2 * (t - T_CLOUD))
        if idx >= 0 and I > 0.02:
            tf, sz, my = FL[idx]; s._lt = 1; sc = sz * H / s.ah; ex, ey = s.A(*EYE)
            g = np.zeros_like(f); img = s.pose(t, sway=0.3 + 0.4 * idx, lean=(6 + 10 * idx) * k)
            s.draw(g, img, W / 2 - (ex - s.aw / 2) * sc + (idx - 1) * W * 0.05 * (idx < 2), H * my - (ey - s.ah / 2) * sc, sz); s._lt = 0
            al = (g.max(2, keepdims=True) > 0.004).astype(np.float32); al = cv2.GaussianBlur(al, (0, 0), 1.0)[:, :, None]
            f = f * (1 - al * min(1.0, I * 3)) + g * (I * 1.15)
            s.bolt(f, idx, I * 1.2); sh += 10 * I
            f = f + I * 0.10
        tr = t - (FL[2][0] + 0.42)                                                   # 闇の 中、仮面の 目だけが 赤く 近づく
        if tr > 0:
            q = clamp(tr / (T_HIT - 0.22 - FL[2][0] - 0.42)); sz = FL[2][1] * (1 + 0.9 * q ** 1.5); sc = sz * H / s.ah; ex, ey = s.A(*EYE); E = s.L0()
            for e_ in EYES:
                p = s.A(*e_); x = W / 2 + (p[0] - ex) * sc; y = H * (FL[2][2] + 0.04 * q) + (p[1] - ey) * sc
                cv2.ellipse(E, (int(x), int(y)), (max(2, int(27 * sc)), max(1, int(11 * sc))), (14 if p[0] < ex else -14), 0, 360, 1.0, -1, cv2.LINE_AA)
            E = cv2.GaussianBlur(E, (0, 0), 3 * k) * (0.25 + 0.75 * q) * (0.85 + 0.15 * math.sin(t * 23))
            f += (E * 1.5)[:, :, None] * RED[None, None, :] + (cv2.GaussianBlur(E, (0, 0), 22 * k) * 1.2)[:, :, None] * RED[None, None, :] + (E ** 3 * 0.8)[:, :, None]
            sh += 2 * q
        ub = t - (T_HIT - 0.22)                                                      # 稲妻が こちらへ 落ちる
        if ub > 0:
            g = clamp(ub / 0.10); f = s.clouds(np.zeros((H, W, 3), np.float32) + f, t, 0.0) ; lit = s.clouds(np.zeros((H, W, 3), np.float32), t, 1.2, 1.4 * g, 1.6)
            f = f * (1 - 0.6 * g) + lit * g; s.bolt(f, 3, 1.5, g)
            d = np.hypot(s.XX - W * 0.5, s.YY - H * 0.46); f += (np.exp(-(d / (W * (0.05 + 1.4 * clamp((ub - 0.08) / 0.16) ** 1.5))) ** 2) * 2.5 * clamp((ub - 0.06) / 0.1))[:, :, None]
            sh += 18 * g
        return f, sh

    # ---- この子だけの曲: 低い 持続と 細く ふるえる 鳴き声(ひょう、ひょう)→ 雷の 3つの 一撃(だんだん 近い)→ 無音の 闇に 心音 → 落雷 ----
    def audio(s, path):
        a = Snd(DUR, 415); D = 138.59; sc_ = [0, 1, 4, 6, 7, 10]
        def fq(d, o=0): return D * 2 ** ((sc_[d % 6] + 12 * (d // 6 + o)) / 12.0)
        a.a += (a.tone(D / 2) + 0.5 * a.tone(D / 2 * 1.06)) * 0.09 * np.clip(a.t / 1.5, 0, 1) * a.gate(0, FL[0][0], 0.3)
        a.a += (a.lp(a.nz, 0.07) - a.lp(a.nz, 0.02)) * a.gate(0.5, FL[0][0], 0.4) * (0.4 + 0.6 * np.clip(a.t / 5, 0, 1)) * 0.8                 # 風
        def cry(t0, f0, ln=0.7, g=0.07): a.a += np.sin(2 * np.pi * (f0 * a.t + f0 * 0.05 * np.clip(a.t - t0, 0, ln) ** 2) + 5 * np.sin(2 * np.pi * 7 * a.t)) * a.env(t0, 0.15, ln * 0.5) * g
        cry(1.0, 1245, 0.7); cry(1.9, 1175, 0.9, 0.06)
        a.boom(T_CUT, 46, 1.1, 0.5); a.whoosh(T_CUT - 0.3, T_CUT, 0.3)
        for d in (0, 1, 3): a.note(fq(d, 0), T_CUT, 1.9, 0.04, "pad", 0.1)
        for i in range(5): a.note(fq([5, 4, 3, 1, 0][i], 1), T_CUT + 0.5 + i * 0.14, 0.8, 0.07, "bell")
        a.rise(T_CLOUD, T_CLOUD + 1.7, 50, 110, 0.14); a.a += a.lp(a.nz, 0.03) * a.gate(T_CLOUD, T_CLOUD + 1.75, 0.3) * 1.6
        cry(T_CLOUD + 0.5, 1245, 0.8, 0.09); cry(T_CLOUD + 1.3, 1319, 0.5, 0.08)
        for j, (tf, sz, my) in enumerate(FL):                                                                            # 雷: 近いほど 大きく、すぐ 鳴る
            g = 0.5 + 0.25 * j; a.a += (a.nz - a.lp(a.nz, 0.6)) * a.env(tf, 0.001, 0.05) * 0.8 * g; a.boom(tf + 0.03, 40 - 3 * j, 1.5 * g, 0.5 + 0.2 * j)
            a.a += a.lp(a.nz, 0.05) * a.env(tf + 0.05, 0.02, 0.5 + 0.2 * j) * 1.6 * g; cry(tf + 0.25, 1245 * 2 ** (j / 12.0), 0.3, 0.05)
        hb = FL[2][0] + 0.45
        while hb < T_HIT - 0.3: a.boom(hb, 50, 0.9, 0.12); a.boom(hb + 0.16, 44, 0.7, 0.12); hb += 0.36                    # 心音
        a.rise(FL[2][0] + 0.42, T_HIT - 0.22, 800, 2400, 0.05)
        tb = T_HIT - 0.22; a.a += (a.nz - a.lp(a.nz, 0.6)) * a.env(tb, 0.001, 0.08) * 1.2; a.boom(tb + 0.03, 32, 2.4, 1.0); a.a += a.lp(a.nz, 0.06) * a.env(tb + 0.04, 0.02, 0.9) * 2.2
        for j, d in enumerate((0, 3, 4, 6)): a.note(fq(d, 1), T_HIT + 0.02, 0.6, 0.05, "bell")
        s.fin_snd(a, DUR, 415); a.save(path)


if __name__ == "__main__":
    r = Nue()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
