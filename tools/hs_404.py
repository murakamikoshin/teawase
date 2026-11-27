"""レオニス(★5 獅子座の王子) 専用の必殺技「獅子王ノ咆哮」— 高品質版(hs_fx)。夜空の しし座が 炎の 獅子に なり、ほえる 声の 輪が こちらへ 来る
夜の 草原に 王子が 立つ → 頭上に しし座の 星が 1つずつ 灯り、線で むすばれる(前ふり) → 目の カットイン+技名 →
星座が 燃えあがり、炎の たてがみと 2つの 目(顔は 描かない: 闇と 目だけ) → 口の 光が ひらいて ほえる:
声の 輪(画面が ゆがむ 波)が 次々に こちらへ ひろがり、獅子が せまって 口の 光に のまれる → 白 → 暗転
使い方: python3 hs_404.py 出力.mp4 / python3 hs_404.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.8
T_CUT, T_LION, T_ROAR, T_HIT = 2.7, 4.7, 7.3, 10.1
GOLD = np.array([1.0, 0.78, 0.30], np.float32); EMB = np.array([1.0, 0.45, 0.10], np.float32); PALE = np.array([1.0, 0.92, 0.72], np.float32); NIGHT = np.array([0.05, 0.06, 0.16], np.float32)
EYE = (401, 233); HILT = (486, 355)
# しし座: 頭の「？」の形 + 体の 三角。(x, y) は 中心からの 比
LEO = [(-0.55, 0.42), (-0.62, 0.10), (-0.40, -0.15), (-0.28, -0.46), (-0.52, -0.64), (-0.76, -0.50), (0.42, -0.12), (0.47, 0.24), (0.96, 0.18)]
LEOL = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (2, 6), (6, 8), (8, 7), (7, 0)]


class Leonis(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 404, (1.0, 0.6, 0.2), "獅子王ノ咆哮", "Lion King's Roar", W); s.fx_init(404)
        s._vt = [s.text_img(c, hz.SERIF, int(60 * s.k), 3) for c in "獅子王ノ咆哮"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H
        s._gt = s.fbm(0.004, 11, 5, 4, 0, 1)

    def tex(s, i, u, v):
        return cv2.remap(s._tiles[i], (u % 256).astype(np.float32), (v % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 夜空と 草原 ----
    def night(s, t, warm=0.0):
        W, H, k = s.W, s.H, s.k
        base = NIGHT[None, None, :] * (0.5 + 1.3 * s.yn ** 1.5)[:, :, None] + np.array([0.10, 0.05, 0.12], np.float32)[None, None, :] * np.exp(-((s.yn - 0.80) / 0.14) ** 2)[:, :, None]
        f = base + s.nebula(t, [(0.06, 0.07, 0.22), (0.16, 0.10, 0.30), (0.40, 0.42, 0.70)], 0.0008, (4.0, 1.0), 0.55, 0)
        s.starfield(f, t, 1000, (0.85, 0.9, 1.0), 3, 0.9 * (1 - 0.5 * warm), (0.0, 0.0))
        if warm > 0:
            f = f * (1 - 0.35 * warm) + (np.array([0.42, 0.12, 0.03], np.float32)[None, None, :] * (np.exp(-np.hypot(s.xn - 0.5, (s.yn - 0.30) * 0.8) / 0.45) * warm)[:, :, None])
        return f.astype(np.float32)

    def ground(s, f, t, warm=0.0):
        W, H, k = s.W, s.H, s.k; hy = 0.815 + 0.012 * np.sin(s.xn * 5.0 + 1.0) + 0.008 * np.sin(s.xn * 13.0)
        m = sstep(hy - 0.004, hy + 0.004, s.yn); d = np.clip((s.yn - hy) / 0.2, 0, 1)
        gr = s.tex(2, s.XX * 0.9 + np.sin(t * 1.3 + s.YY * 0.02) * 3 * k, s.YY * 3.5 / (0.3 + d))
        col = (np.array([0.020, 0.030, 0.045], np.float32) * (1 - warm) + np.array([0.16, 0.06, 0.02], np.float32) * warm)[None, None, :] * (0.5 + 1.2 * gr)[:, :, None] * (1.2 - 0.6 * d)[:, :, None]
        col += (np.exp(-d / 0.06) * (0.10 + 0.5 * warm) * (0.5 + gr))[:, :, None] * (PALE * (1 - warm) * 0.4 + EMB * warm)[None, None, :]      # 地平の ふち光
        f *= (1 - m)[:, :, None]; f += col * m[:, :, None]

    # ---- しし座 ----
    def leo(s, f, t, ns, nl, v=1.0, C=None, S=None):
        if v <= 0.01: return
        W, H, k = s.W, s.H, s.k; C = C or (W / 2 - W * 0.04, H * 0.27); S = S or W * 0.36
        P = [(C[0] + x * S, C[1] + y * S) for x, y in LEO]
        for j, (a, b) in enumerate(LEOL):
            q = clamp(nl - j)
            if q > 0: s.glowline(f, [P[a], (P[a][0] + (P[b][0] - P[a][0]) * q, P[a][1] + (P[b][1] - P[a][1]) * q)], PALE * 0.8 + 0.2, 0.9, 0.42 * v, 0.7)
        for i, p in enumerate(P):
            q = ns - i
            if q <= 0: continue
            pop = 1 + 1.5 * math.exp(-q * 6); big = 1.5 if i in (0, 2, 8) else 1.0
            s.flare(f, p, (4.5 + 1.2 * math.sin(t * 5 + i * 2)) * k * big * pop, PALE, 0.85 * v * clamp(q * 5), 0.35, 0.0, False)

    # ---- 炎の 獅子: たてがみ(火)・闇の 顔・目・口の 光 ----
    def lion(s, f, C, R, t, v, eye, mouth):
        if v <= 0.01: return
        W, H, k = s.W, s.H, s.k; dx = (s.XX - C[0]) / R; dy = (s.YY - C[1]) / R; d = np.hypot(dx, dy) + 1e-4; th = np.arctan2(dy, dx)
        n = s.tex(1, th / 6.2832 * 512, d * 34 - t * 55); n2 = s.tex(3, th / 6.2832 * 1024 + 40, d * 80 - t * 120); tuft = s.tex(0, th / 6.2832 * 768, t * 14 + 0 * d)
        Ro = 0.80 + 0.75 * tuft ** 1.4 + 0.12 * np.abs(np.cos(th)); chin = 1 - 0.45 * np.exp(-((np.abs(th - 1.5708)) / 0.55) ** 2)
        m = sstep(0.40, 0.62, np.minimum(d, np.hypot(dx, dy - 0.20) * 1.45)) * sstep(Ro, Ro * 0.45, d) * chin
        fl = np.clip((n * 0.85 + n2 * 0.55 - 0.42) * 2.3, 0, 1) * m + m * 0.22
        f += s.ramp(np.clip(fl * v, 0, 1), s.FIRE) * 1.25
        f += (np.exp(-d / 0.9) * 0.22 * v)[:, :, None] * EMB[None, None, :]
        dq = np.minimum(d, np.hypot(dx, dy - 0.20) * 1.45); face = sstep(0.56, 0.38, dq) * 0.92 * v; f *= (1 - face)[:, :, None]
        f += (face * (0.03 + 0.06 * s.tex(2, dx * 60 + 128, dy * 60 + 128)))[:, :, None] * EMB[None, None, :]
        if eye > 0:                                                                  # 目: つり上がった 2つの 火
            for sg in (-1, 1):
                ex = dx - sg * 0.185; ey = dy + 0.08; a = -sg * 0.42; xr = ex * math.cos(a) + ey * math.sin(a); yr = -ex * math.sin(a) + ey * math.cos(a)
                g = np.exp(-((xr / 0.105) ** 2 + (yr / ((0.010 + 0.034 * eye) * (0.55 + 0.45 * sstep(-0.12, 0.10, -sg * xr)))) ** 2))
                f += (g * 2.6 * eye * v)[:, :, None] * PALE[None, None, :] + (np.exp(-((xr / 0.20) ** 2 + (yr / 0.09) ** 2)) * 0.55 * eye * v)[:, :, None] * EMB[None, None, :]
        if mouth > 0.02:
            my = dy - 0.24; hgt = 0.02 + 0.17 * mouth; wid = 0.11 + 0.07 * mouth
            g = np.exp(-(np.abs(dx / wid) ** 2.6 + (np.abs(my) / hgt) ** 3))
            fang = np.zeros_like(g)
            for x0, w0, h0, up in ((-0.090, 0.042, 0.34, 1), (0.090, 0.042, 0.34, 1), (-0.050, 0.030, 0.22, -1), (0.050, 0.030, 0.22, -1)):    # きば(光の 中の 影)
                yy = (my + hgt * 0.95 * up) * up; q = np.clip(1 - yy / (hgt * 2 * h0), 0, 1) * (yy > -hgt * 0.3)
                fang = np.maximum(fang, sstep(0.0, 0.25, q - np.abs(dx - x0) / w0))
            g = g * (1 - 0.92 * fang * sstep(0.25, 0.6, mouth))
            f += (g * (0.8 + 1.5 * mouth) * v)[:, :, None] * PALE[None, None, :] + (np.exp(-(np.hypot(dx, my) / (0.22 + 0.3 * mouth)) ** 2) * mouth * 0.45 * v)[:, :, None] * EMB[None, None, :]

    def ripple(s, f, c, fronts):
        """声の 輪: 画面を 輪の 形に ゆがませる + 輪の ふちの 光。fronts: [(半径, 強さpx, 幅px)]"""
        dx = s.XX - c[0]; dy = s.YY - c[1]; d = np.hypot(dx, dy) + 1e-3; disp = np.zeros_like(d); edge = np.zeros_like(d)
        for rf, amp, wd in fronts:
            disp += amp * np.exp(-((d - rf) / wd) ** 2); edge += np.exp(-((d - rf - wd * 0.4) / (wd * 0.20)) ** 2) * min(1.0, amp / (22 * s.k))
        f = cv2.remap(f, (s.XX - dx / d * disp).astype(np.float32), (s.YY - dy / d * disp).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        return f + (edge * 0.55)[:, :, None] * (PALE * 0.6 + EMB * 0.4)[None, None, :]

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        w = getattr(s, '_warm', 0.0)
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(PALE * (1 - w) + (EMB * 0.7 + 0.3) * w), rim_dir=(0.0, -1.0), rim_v=(0.45 + 0.5 * w) * alpha, amb=tuple(np.array([0.7, 0.8, 1.0]) * (1 - w) + np.array([1.0, 0.75, 0.5]) * w), amb_v=0.14, aura=EMB, aura_v=0.12 * w, alpha=alpha, shade=0.80 + 0.14 * w)

    def frame(s, t):
        o = {}; s._warm = 0.0
        if t < T_CUT: f, sh = s.sc_night(t)
        elif t < T_LION: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.9)
        else: f, sh = s.sc_lion(t)
        f, sh = s.finale(f, sh, t, DUR, None, (1, 0.8, 0.3), (1, 0.45, 0.1), -1)
        return s.film(f, t, sh * s.k, 0.6, o.get("thr", 0.80), (1.0, 0.82, 0.6), 1.0 + 0.15 * min(sh, 8), 0.04, o.get("vig", 0.55), 1.15, 1.08)

    # 1) 夜の 草原。しし座が 1つずつ 灯る
    def sc_night(s, t):
        W, H, k = s.W, s.H, s.k
        f = s.night(t); s.ground(f, t)
        s.leo(f, t, (t - 0.55) / 0.19, (t - 1.25) / 0.14)
        img = s.pose(t, sway=0.25, breath=0.5); s.draw(f, img, W / 2, H * 0.70, 0.43)
        if t > 2.2: s.flare(f, s.at(*s.A(*HILT)), 9 * k * ease((t - 2.2) / 0.3), GOLD, 0.8, 0.8, t, False)         # つかが 光る
        return f * clamp(t / 0.5), 0

    # 2) 目の カットイン + 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.night(t) * 0.35; s.leo(f, t, 9, 9, 0.5)
        a = ease(u / 0.28); cy = H * 0.34; hh = H * 0.115; sl = 0.20
        band = np.zeros((H, W, 3), np.float32) + (EMB * 0.20)[None, None, :] + s.ramp(np.clip(s.flame(t, np.ones((H, W), np.float32), 300.0, 0.002, 0.9), 0, 1), s.FIRE) * 0.45
        ex, ey = s.A(*EYE); zf = 1.55; sc = zf * H / s.ah; s._warm = 0.6
        img = s.pose(t, breath=0.2); s.draw(band, img, W / 2 - (ex - s.aw / 2) * sc + (1 - a) * W * 0.9 - u * 6 * k, cy - (ey - s.ah / 2) * sc, zf)
        dd = (s.YY - cy) + (s.XX - W / 2) * sl; m = sstep(hh, hh - 2, np.abs(dd))[:, :, None] * a
        f = f * (1 - m) + band * m
        for sg in (-1, 1): s.glowline(f, [(0, cy + sg * hh + W / 2 * sl), (W, cy + sg * hh - W / 2 * sl)], GOLD, 1.4, a, 0.9)
        n = len(s._vt); step = W * 0.148
        for i, T in enumerate(s._vt):                                                # 技名: 1文字ずつ たたきつける
            q = (u - 0.55 - i * 0.11) / 0.16
            if q > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.66 - (1 - ease(q)) * 26 * k, clamp(q * 2), 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.4 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.66 - 66 * k, 1.0, 1.0)
        if u > 1.3:
            g = ease((u - 1.3) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.66 + 50 * k), (W / 2 + W * 0.42 * g, H * 0.66 + 50 * k)], GOLD, 1.0, 0.8, 0.8)
            s.text(f, s.txt_en, W / 2, H * 0.66 + 76 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.8
        return f, 6 * math.exp(-u * 6)

    # 3) 星座が 燃えて 炎の 獅子に → ほえる。声の 輪が こちらへ
    def sc_lion(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_LION; g = ease(u / 1.7); ro = t - T_ROAR; e = clamp((ro - 0.45) / (T_HIT - T_ROAR - 0.45)); ee = e ** 2.4
        s._warm = g; f = s.night(t, 0.85 * g); s.ground(f, t, g)
        C = (W / 2, H * (0.27 + 0.17 * ease2(e))); R = W * 0.40 * (0.30 + 0.70 * g) * (1 + 0.02 * math.sin(t * 7)) * (1 + 7.5 * ee)
        s.leo(f, t, 9, 9, 1 - ease(u / 1.2))
        op = ease(ro / 0.22) if ro > 0 else 0.0
        s.lion(f, C, R, t, clamp(u / 0.5) * (0.4 + 0.6 * g), ease((u - 1.0) / 0.35), op)
        Cm = (C[0], C[1] + 0.22 * R)
        em = np.zeros_like(f); s.starfield(em, t, 260, tuple(EMB * 0.8 + 0.2), 8, 1.6 * g, (14.0, -70.0)); f += em * (s.yn < 0.84)[:, :, None]      # 火の粉
        a_ = 1 - clamp((e - 0.42) / 0.25)
        if a_ > 0.02:
            blow = (1 - math.exp(-ro * 5)) if ro > 0 else 0.0
            img = s.pose(t, sway=0.25 + 1.3 * blow, breath=0.4, lean=-5 * k * blow, squash=0.03 * blow * math.exp(-ro * 2))
            s.draw(f, img, W / 2, H * (0.70 + 0.5 * ee), 0.43 + 0.03 * g, a_)
            s.flare(f, s.at(*s.A(*HILT)), (8 + 3 * math.sin(t * 9) ** 2) * k, GOLD, 0.8 * a_, 0.7, t, False)
        sh = 1.2 * g * abs(math.sin(u * 5))
        if ro > 0:
            s.rushdust(f, t, Cm, 1300, 0.7 + 2.4 * e, EMB * 0.7 + PALE * 0.3, 0.25 + 0.6 * e, 6, 1.0)
            fr = []
            for i in range(5):                                                       # 声の 輪: 5つ、あとの ほど 強い
                q = ro - i * 0.38
                if q > 0: fr.append((q ** 1.35 * W * 1.25 + 10 * k, (20 + 5 * i) * k * math.exp(-q * 0.9), (26 + 30 * q) * k))
            f = s.ripple(f, Cm, fr)
            sh = 13 * math.exp(-ro * 2.2) + 3 + 9 * ee + 4 * sum(math.exp(-((ro - i * 0.38) * 9) ** 2) for i in range(5))
        f = s.godrays(f, Cm, 0.15 * g + 0.22 * op, 1.1, 10, 0.88, 0.5)
        if 0 <= ro < 0.12: f = f + (1 - ro / 0.12) * 0.9
        if u < 0.15: f = f + (1 - u / 0.15) * 0.4
        if e > 0.84: f = f + ((e - 0.84) / 0.16) ** 2 * 1.3
        return f, sh

    # ---- この子だけの曲: 低い D の 上に 星の 鈴が 9つ → 金管ふうの 和音 → うなり → ほえ声(ふるえる 低い 雑音)と 太鼓 5つ ----
    def audio(s, path):
        a = Snd(DUR, 404); D = 146.83; dor = [0, 2, 3, 5, 7, 9, 10]
        def fq(d, o=0): return D * 2 ** ((dor[d % 7] + 12 * (d // 7 + o)) / 12.0)
        a.a += (a.tone(D / 2) + 0.4 * a.tone(D * 0.75)) * 0.09 * np.clip(a.t / 1.5, 0, 1) * a.gate(0, T_ROAR, 0.2)
        for i in range(9): a.note(fq([7, 9, 11, 12, 14, 16, 11, 9, 14][i], 1), 0.55 + i * 0.19, 0.7, 0.10, "bell")
        for i in range(9): a.note(fq(i % 5, 0), 1.25 + i * 0.14, 0.2, 0.03, "sine")
        a.note(fq(7, 0), 2.2, 0.5, 0.10, "bell")
        a.boom(T_CUT, 46, 1.1, 0.5); a.whoosh(T_CUT - 0.3, T_CUT, 0.3)
        for d in (0, 4, 7, 9): a.note(fq(d, 0), T_CUT, 1.8, 0.05, "saw", 0.03)
        for i in range(6): a.boom(T_CUT + 0.55 + i * 0.11, 80 - i * 4, 0.45, 0.12); a.note(fq([0, 2, 4, 7, 4, 7][i], 1), T_CUT + 0.55 + i * 0.11, 0.3, 0.10, "bell")
        gr = a.lp(a.nz, 0.03) * (0.6 + 0.4 * np.sin(2 * np.pi * 23 * a.t))                                             # うなり
        a.a += gr * np.clip((a.t - T_LION) / (T_ROAR - T_LION), 0, 1) ** 1.5 * (a.t < T_ROAR) * 1.6
        a.rise(T_LION, T_ROAR, 50, 110, 0.12)
        for j, d in enumerate((0, 4, 7)): a.note(fq(d, -1), T_LION + j * 0.5, T_ROAR - T_LION - j * 0.5, 0.06, "saw", 0.4)
        for i in range(6): a.note(fq(i + 2, 0), T_LION + 1.0 + i * 0.22, 0.4, 0.07, "bell")
        a.note(fq(14, 1), T_ROAR - 0.25, 0.25, 0.08, "sine")
        roar = (a.lp(a.nz, 0.06) * 1.6 + a.lp(a.nz, 0.3) * 0.5) * (0.55 + 0.45 * np.sin(2 * np.pi * 31 * a.t + 2 * np.sin(2 * np.pi * 5 * a.t)))
        a.a += roar * np.clip((a.t - T_ROAR) / 0.06, 0, 1) * np.exp(-np.clip(a.t - T_ROAR - 1.6, 0, None) / 0.8) * (a.t >= T_ROAR) * 1.5
        a.a += a.tone(62) * np.sin(2 * np.pi * 31 * a.t) * a.env(T_ROAR, 0.05, 1.4) * 0.3
        for i in range(5): a.boom(T_ROAR + i * 0.38, 44 - i * 2, 1.1 + 0.15 * i, 0.5); a.whoosh(T_ROAR + i * 0.38, T_ROAR + i * 0.38 + 0.5, 0.3, 0.3)
        for j, d in enumerate((0, 4, 7, 9, 11, 14)): a.note(fq(d, -1 if j < 2 else 0), T_ROAR + 0.1 + j * 0.05, T_HIT - T_ROAR, 0.045, "saw", 0.3)
        a.rise(T_ROAR + 0.6, T_HIT, 80, 300, 0.13)
        a.boom(T_HIT, 36, 1.7, 0.9)
        for j, d in enumerate((0, 4, 7, 11)): a.note(fq(d, 0), T_HIT + 0.02, 0.6, 0.06, "bell")
        s.fin_snd(a, DUR, 404); a.save(path)


if __name__ == "__main__":
    r = Leonis()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
