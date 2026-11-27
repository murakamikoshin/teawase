"""キツネビ(★5 九尾の巫女) 専用の必殺技「九尾ノ狐火」— 高品質版(hs_fx)。九つの 狐火が 輪に なり、九つ折りの 万華鏡に 化けて こちらを のみこむ
夜の 杉の 森(きり・こけ)。巫女が 浮かび、青い 狐火が 1つずつ 灯って 九つに なる(前ふり) → 目の カットイン+技名 →
九つの 狐火が まわりを まわる(だんだん はやく、青い 尾)→ ひとつの 輪に そろうと、輪の 中が 九つ折りの 万華鏡(狐火の 模様)に 化ける →
万華鏡が ひろがって 画面を おおい、模様が こちらへ 流れて くる(化かされて のみこまれる)→ まん中の 光 → 白 → 暗転
使い方: python3 hs_413.py 出力.mp4 / python3 hs_413.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.9
T_CUT, T_SPIN, T_KAL, T_HIT = 2.8, 4.8, 7.0, 10.2
FOX = np.array([0.20, 0.55, 1.0], np.float32); CYAN = np.array([0.55, 0.92, 1.0], np.float32); RED = np.array([1.0, 0.28, 0.20], np.float32); NIGHT = np.array([0.015, 0.035, 0.06], np.float32)
BR = [(0.0, (0.0, 0.0, 0.0)), (0.28, (0.02, 0.07, 0.36)), (0.58, (0.10, 0.42, 1.0)), (0.84, (0.50, 0.90, 1.0)), (1.0, (1.0, 1.0, 1.0))]
EYE = (365, 310); FIRES = [(115, 320), (50, 530), (705, 490)]; PALM = (130, 400)


class Kitsunebi(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 413, (0.35, 0.7, 1.0), "九尾ノ狐火", "Ninefold Foxfire", W); s.fx_init(413)
        s._vt = [s.text_img(c, hz.SERIF, int(66 * s.k), 3) for c in "九尾ノ狐火"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; H = s.H; k = s.k; rg = np.random.RandomState(4130)
        s._tr = []                                                                   # 杉の 幹(奥 → 手前)
        for lay, (n, w0, dk) in enumerate(((9, 16, 0.55), (6, 30, 0.35), (4, 58, 0.16))):
            xs = (np.arange(n) + rg.rand(n) * 0.7) / n * 1.1 - 0.05
            for x in xs:
                if lay == 2 and 0.28 < x < 0.72: continue
                s._tr.append((lay, x * W, (w0 + rg.rand() * w0 * 0.5) * k, dk, rg.rand() * 200))
        s.HC = (W / 2, H * 0.60)

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 夜の 杉の 森 ----
    def forest(s, t, lum=1.0, blue=0.0):
        W, H, k = s.W, s.H, s.k; gy = 0.80
        f = NIGHT[None, None, :] * (1.0 + 1.4 * np.exp(-((s.yn - 0.45) / 0.3) ** 2))[:, :, None] + (FOX * 0.10 * (0.4 + blue))[None, None, :] * np.exp(-np.hypot(s.xn - 0.5, (s.yn - 0.55) * 0.8) / 0.35)[:, :, None]
        last = -1
        for lay, x, w, dk, ph in s._tr:
            if lay != last:                                                          # 層の 間に きり
                ms = s.fbm(0.0014, t * (8 + 5 * lay), lay * 60.0, 3, 40 * k, lay); f += (sstep(0.35, 0.85, ms) * (0.05 + 0.03 * lay) * (0.6 + blue))[:, :, None] * np.array([0.35, 0.6, 0.8], np.float32)[None, None, :]; last = lay
            x0 = max(0, int(x - w)); x1 = min(W, int(x + w) + 1)
            if x1 <= x0: continue
            xx = (s.XX[:, x0:x1] - x) / w; m = sstep(1.0, 0.92, np.abs(xx)) * (s.YY[:, x0:x1] < H * (gy + 0.03 * lay))
            bark = 0.6 + 0.6 * s.tex(2, s.XX * 3.0 + ph, s.YY * 0.25 + ph)[:, x0:x1]; sh_ = 0.45 + 0.55 * np.sqrt(np.clip(1 - xx ** 2, 0, 1))
            col = np.array([0.075, 0.065, 0.08], np.float32)[None, None, :] * (bark * sh_ * (1 - dk) * 1.8)[:, :, None] + (FOX * 0.10 * (0.3 + blue))[None, None, :] * (np.exp(-((xx - 0.2 * np.sign(W / 2 - x)) / 0.5) ** 2) * (0.3 + 0.25 * lay))[:, :, None]
            f[:, x0:x1] = f[:, x0:x1] * (1 - m)[:, :, None] + col * m[:, :, None]
        gm = sstep(gy - 0.004, gy + 0.004, s.yn + (s.tex(1, s.XX * 0.6, 3.0) - 0.5) * 0.02)[:, :, None]; d = np.clip((s.yn - gy) / (1 - gy), 0, 1)
        gn = s.tex(1, s.XX * 0.9, s.YY * 2.2) * 0.6 + s.tex(3, s.XX * 2.0, s.YY * 2.0) * 0.4
        g = np.array([0.015, 0.04, 0.03], np.float32)[None, None, :] * (0.4 + 1.5 * gn)[:, :, None] + (FOX * 0.14 * (0.3 + blue))[None, None, :] * (np.exp(-np.hypot((s.xn - 0.5) * 1.5, (s.yn - gy) * 3) / 0.3) * (0.4 + gn))[:, :, None]
        return ((f * (1 - gm) + g * gm) * lum).astype(np.float32)

    def box(s, c, rad):
        x0 = max(0, int(c[0] - rad)); x1 = min(s.W, int(c[0] + rad) + 1); y0 = max(0, int(c[1] - rad)); y1 = min(s.H, int(c[1] + rad) + 1)
        return (slice(y0, y1), slice(x0, x1)) if x1 > x0 + 1 and y1 > y0 + 1 else None

    # ---- 狐火(式): 下が 丸く 上へ とがる 青い 火 ----
    def foxfire(s, f, c, R, t, ph=0.0, v=1.0):
        if v <= 0.01 or R < 1.5: return
        sl = s.box(c, R * 3.2)
        if sl is None: return
        x = (s.XX[sl] - c[0]) / R; y = (c[1] - s.YY[sl]) / R                                                           # y: 上が 正
        sw = 0.22 * np.sin(y * 1.6 + t * 7 + ph) * np.clip(y, 0, 3) * 0.5; xx = x - sw
        tip = np.clip(1 - y / 2.4, 0, 1) ** 1.25; wdt = np.where(y < 0, np.sqrt(np.clip(1 - y ** 2, 0, 1)), tip + 0.0) ; q = np.abs(xx) / np.maximum(wdt, 1e-3)
        ins = sstep(1.0, 0.72, q) * (y > -1.0) * (y < 2.4); core = sstep(1.0, 0.5, np.hypot(xx / 0.48, (y + 0.15) / 0.62))
        val = np.clip(0.50 * ins * (1 - 0.35 * q) + 0.55 * core + 0.12 * ins * np.sin(t * 13 + ph * 3 + y * 3), 0, 1)
        f[sl] += s.ramp_box(val, BR) * (1.5 * v) + (np.exp(-(np.hypot(x, y - 0.3) / 2.0) ** 2) * 0.22 * v)[:, :, None] * FOX[None, None, :]

    def ramp_box(s, v, stops):
        out = np.zeros(v.shape + (3,), np.float32)
        for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:]):
            w = np.clip((v - p0) / (p1 - p0 + 1e-9), 0, 1)[..., None]; seg = (v >= p0) & ((v < p1) if p1 < 1.0 else True)
            out[seg] = (np.asarray(c0, np.float32) * (1 - w) + np.asarray(c1, np.float32) * w)[seg]
        return out

    # ---- 九つ折りの 万華鏡(狐火の 模様)----
    def kaleido(s, f, C, S, t, flow, v=1.0, core=0.0):
        W, H, k = s.W, s.H, s.k; dx = s.XX - C[0]; dy = s.YY - C[1]; r = np.hypot(dx, dy) / S + 1e-4; th = np.arctan2(dy, dx)
        a = (th / 6.2832 * 9 + t * 0.05) % 1.0; m = np.abs(a - 0.5) * 2                                                 # 0: くさびの 軸 / 1: くさびの さかい
        lr = np.log(r + 0.03) * 1.25 - flow; q = lr % 1.0; q2 = (lr + 0.5) % 1.0
        n1 = s.tex(1, m * 96, lr * 52); n2 = s.tex(3, m * 200 + 31, lr * 120)
        b1 = sstep(0.34, 0.20, np.hypot((1 - m) * 0.75, (q - 0.5) * 0.9)) ; b2 = sstep(0.30, 0.18, np.hypot(m * 0.75, (q2 - 0.5) * 0.9))    # 軸の 上と さかいの 上に 狐火の 玉が ならぶ
        lines = np.exp(-(m / 0.03) ** 2) * 0.35 + np.exp(-((1 - m) / 0.03) ** 2) * 0.25 + np.exp(-((q - 0.5) / 0.04) ** 2) * 0.22 * (0.4 + m)
        val = np.clip(0.16 + 0.42 * n1 ** 1.4 * (1 - 0.4 * m) + 0.25 * n2 ** 3 + 0.75 * b1 + lines, 0, 1)
        col = s.ramp(val, BR) * 1.25 + (b2 * 0.95)[:, :, None] * RED[None, None, :] + (b2 ** 3 * 0.6)[:, :, None]
        col += (np.exp(-(r / (0.10 + 0.5 * core)) ** 2) * (0.8 + 2.6 * core))[:, :, None] * CYAN[None, None, :] * 0.6 + (np.exp(-(r / (0.07 + 0.4 * core)) ** 2) * (0.5 + 2.5 * core))[:, :, None]
        ins = sstep(1.0, 0.97, r) * v
        f *= (1 - ins)[:, :, None]; f += col * ins[:, :, None]
        f += (np.exp(-((r - 1.0) / 0.025) ** 2) * 1.6 * v)[:, :, None] * CYAN[None, None, :] + (np.exp(-np.clip(r - 1, 0, None) / 0.3) * (r > 1) * 0.35 * v)[:, :, None] * FOX[None, None, :]

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        b = getattr(s, '_bl', 0.3)
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(CYAN), rim_dir=(0.0, 1.0), rim_v=(0.3 + 0.45 * b) * alpha, amb=(0.6, 0.8, 1.0), amb_v=0.20, aura=FOX, aura_v=0.10 + 0.16 * b, alpha=alpha, shade=0.72 + 0.2 * b)

    def frame(s, t):
        o = {}; s._bl = 0.3
        if t < T_CUT: f, sh = s.sc_nine(t)
        elif t < T_SPIN: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.92)
        else: f, sh = s.sc_trick(t)
        f, sh = s.finale(f, sh, t, DUR, None, (0.5, 0.85, 1), (0.2, 0.5, 1.0), -1)
        return s.film(f, t, sh * s.k, 0.6, o.get("thr", 0.80), (0.85, 0.95, 1.0), 1.0 + 0.15 * min(sh, 8), 0.04, o.get("vig", 0.58), 1.15, 1.12)

    def miko(s, f, t, alpha=1.0, **kw):
        W, H, k = s.W, s.H, s.k; cy = s.HC[1] + 7 * k * math.sin(t * 2.0)
        sh_ = s.L0(); cv2.ellipse(sh_, (int(W / 2), int(H * 0.845)), (int(W * 0.16), int(H * 0.010)), 0, 0, 360, 0.45 * alpha, -1, cv2.LINE_AA); f *= (1 - cv2.GaussianBlur(sh_, (0, 0), 5 * k))[:, :, None]
        img = s.pose(t, **kw); s.draw(f, img, W / 2, cy, 0.42, alpha)

    def ringpos(s, i, ang, rx, ry, cy=None):
        a = ang + i * 6.2832 / 9; return (s.HC[0] + math.cos(a) * rx, (cy if cy is not None else s.HC[1]) + math.sin(a) * ry, math.sin(a))

    # 1) 夜の 森。狐火が 1つずつ 灯って 九つに
    def sc_nine(s, t):
        W, H, k = s.W, s.H, s.k; n = (t - 0.7) / 0.21; s._bl = 0.3 + 0.5 * clamp(n / 9)
        f = s.forest(t, 1.0, 0.5 * clamp(n / 9))
        pts = [s.ringpos(i, -1.9 + t * 0.25, W * 0.40, H * 0.20) for i in range(9)]
        for i, (x, y, dp) in enumerate(pts):
            if dp < 0 and n > i: s.foxfire(f, (x, y), (21 + 6 * dp) * k * (1 + 1.2 * math.exp(-(n - i) * 3)), t, i * 1.3, clamp((n - i) * 3))
        s.miko(f, t, clamp(t / 0.6), sway=0.3, breath=0.5)
        for i, (x, y, dp) in enumerate(pts):
            if dp >= 0 and n > i: s.foxfire(f, (x, y), (21 + 6 * dp) * k * (1 + 1.2 * math.exp(-(n - i) * 3)), t, i * 1.3, clamp((n - i) * 3))
        return f * clamp(t / 0.5), 0

    # 2) 目の カットイン(きつねの 目の 形の 帯)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT; s._bl = 0.8
        f = s.forest(t, 0.32, 0.5)
        a = ease(u / 0.26); cy = H * 0.33; hh = H * 0.12
        band = np.array([0.01, 0.03, 0.10], np.float32)[None, None, :] + s.ramp(np.clip(s.flame(t, np.ones((H, W), np.float32), 300.0, 0.002, 0.9), 0, 1), BR) * 0.55
        ex, ey = s.A(*EYE); zf = 1.5; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(band, img, W / 2 - (ex - s.aw / 2) * sc - (1 - a) * W * 0.9 + u * 5 * k, cy - (ey - s.ah / 2) * sc, zf)
        xr = (s.XX - W / 2) / (W * 0.60); edge = hh * a * np.clip(1 - np.abs(xr) ** 1.6, 0, 1) ** 0.8; dd = np.abs(s.YY - cy - xr * H * 0.035)                # つり上がった 目の 形
        m = (sstep(edge, edge - 2, dd) * (edge > 1))[:, :, None]; f = f * (1 - m) + band * m
        s.glowmask(f, np.exp(-((dd - edge) / (1.8 * k)) ** 2) * (edge > 1), CYAN, 0.9 * a, 1.0, 0.8, 8.0)
        n = len(s._vt); step = W * 0.165
        for i, T in enumerate(s._vt):                                                # 技名: ゆらめいて 1文字ずつ 灯る
            p = (u - 0.5 - i * 0.13) / 0.3
            if p > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.69 + 4 * k * math.sin(t * 5 + i * 1.3), clamp(p) * (0.85 + 0.15 * math.sin(t * 11 + i)), 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.35 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.69 - 70 * k, 1.0, 1.0)
        if u > 1.4:
            g = ease((u - 1.4) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.69 + 52 * k), (W / 2 + W * 0.42 * g, H * 0.69 + 52 * k)], CYAN, 1.0, 0.8, 0.8)
            s.text(f, s.txt_en, W / 2, H * 0.69 + 78 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.7
        return f, 4 * math.exp(-u * 6)

    # 3) 九つが まわる → 輪に そろう → 万華鏡に 化けて のみこむ
    def sc_trick(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_SPIN; uk = t - T_KAL; sh = 0.0
        e = clamp(uk / (T_HIT - T_KAL)) if uk > 0 else 0.0; ee = e ** 2.2; s._bl = 1.0
        f = s.forest(t, 1.0 - 0.4 * e, 0.5 + 0.5 * clamp(u / 1.0))
        al = ease2(clamp((u - 1.2) / 1.0))                                           # 輪に そろう(だ円 → 正円、上へ)
        ang = -1.9 + T_SPIN * 0.25 + u * 0.8 + u * u * 1.1; rx = W * (0.40 - 0.08 * al); ry = H * 0.20 * (1 - al) + rx * al; cyr = s.HC[1] - H * 0.16 * al
        C = (W / 2, cyr)
        S = rx * (1 + 9.0 * ee) if uk > 0 else rx
        pts = [s.ringpos(i, ang, rx, ry, cyr) for i in range(9)]
        def fires(front):
            for i, (x, y, dp) in enumerate(pts):
                if (dp >= 0) != front and al < 0.98: continue
                if al >= 0.98 and not front: continue
                R = (22 + 6 * dp * (1 - al)) * k
                if uk > 0: x = C[0] + (x - C[0]) * S / rx; y = C[1] + (y - C[1]) * S / rx; R = R * (1 + 4 * ee)
                tl = []
                for j in range(7, 0, -1):                                            # 青い 尾
                    px, py, _ = s.ringpos(i, ang - j * 0.05 * (0.6 + u * 0.5), rx, ry, cyr)
                    if uk > 0: px = C[0] + (px - C[0]) * S / rx; py = C[1] + (py - C[1]) * S / rx
                    tl.append((px, py))
                tl.append((x, y)); L = s.L0()
                for j in range(len(tl) - 1): cv2.line(L, (int(tl[j][0]), int(tl[j][1])), (int(tl[j + 1][0]), int(tl[j + 1][1])), float((j + 1) / 8.0), max(1, int(R * 0.5 * (j + 1) / 8.0)), cv2.LINE_AA)
                s.glowmask(f, cv2.GaussianBlur(L, (0, 0), 1.0 * k), FOX * 0.8 + CYAN * 0.2, 0.7 * (1 - 0.7 * e), 1.0, 0.8, 7.0)
                s.foxfire(f, (x, y), R, t, i * 1.3, 1.0 - 0.6 * clamp((e - 0.6) / 0.3))
        fires(False)
        if uk > 0: s.kaleido(f, C, S * 0.93, t, uk * 0.25 + 2.4 * ee, min(1.0, uk * 3), e ** 3.2 * 0.8)
        a_ = 1 - clamp((e - 0.18) / 0.22)
        if a_ > 0.02: s.miko(f, t, a_, sway=0.4 + 0.3 * clamp(u / 1.5), breath=0.4, squash=-0.03 * al)
        fires(True)
        if uk > 0:
            s.rushdust(f, t, C, 900, 0.6 + 2.2 * e, CYAN, 0.15 + 0.5 * e, 5, 1.0); f = s.godrays(f, C, 0.2 + 0.45 * e, 0.95, 8, 0.88, 0.45)
            if uk < 0.12: f = f + (1 - uk / 0.12) * 0.9
            sh += 8 * math.exp(-uk * 4) + 1 + 9 * ee
            if e > 0.90: f = f + ((e - 0.90) / 0.10) ** 2 * 1.4
        if u < 0.12: f = f + (1 - u / 0.12) * 0.3
        return f, sh

    # ---- この子だけの曲: 神楽鈴と 笛(陽音階)。狐火 9つ = 9つの 鈴 → まわるほど 鈴が こまかく → 化ける 一音 → ガラスの ような 和音が くり返し 上がる ----
    def audio(s, path):
        a = Snd(DUR, 413); D = 293.66; yo = [0, 2, 5, 7, 9]
        def fq(d, o=0): return D * 2 ** ((yo[d % 5] + 12 * (d // 5 + o)) / 12.0)
        a.a += (a.tone(D / 4) + 0.4 * a.tone(D / 4 * 1.5)) * 0.07 * np.clip(a.t / 1.5, 0, 1) * a.gate(0, T_HIT, 0.3)
        a.a += (a.lp(a.nz, 0.06) - a.lp(a.nz, 0.02)) * a.gate(0, T_KAL, 0.3) * 0.5
        for i in range(9): tk = 0.7 + i * 0.21; a.note(fq([0, 1, 2, 3, 4, 5, 6, 7, 9][i], 1), tk, 0.6, 0.10, "bell"); a.a += (a.nz - a.lp(a.nz, 0.85)) * a.env(tk, 0.001, 0.06) * 0.10      # 鈴
        for (d, o, t0, ln) in ((2, 0, 0.3, 0.9), (4, 0, 1.3, 0.5), (3, 0, 1.9, 0.8)): a.a += np.sin(2 * np.pi * fq(d, o) * a.t + 2.2 * np.sin(2 * np.pi * 5 * a.t)) * a.env(t0, 0.12, ln * 0.6) * 0.07
        a.boom(T_CUT, 55, 0.8, 0.4); a.whoosh(T_CUT - 0.3, T_CUT, 0.25)
        for d in (0, 2, 3): a.note(fq(d, 0), T_CUT, 1.9, 0.045, "pad", 0.05)
        for i in range(5): a.note(fq([2, 3, 4, 5, 7][i], 1), T_CUT + 0.5 + i * 0.13, 0.5, 0.10, "bell")
        tk = T_SPIN + 0.1; st = 0.30; i = 0
        while tk < T_KAL - 0.05:                                                                                       # まわる: 鈴が こまかく なる
            a.note(fq(i % 9, 1), tk, 0.25, 0.07, "bell"); a.a += (a.nz - a.lp(a.nz, 0.85)) * a.env(tk, 0.001, 0.04) * 0.08
            if i % 3 == 0: a.boom(tk, 75, 0.25, 0.08)
            tk += st; st = max(0.07, st * 0.93); i += 1
        a.rise(T_SPIN + 1.0, T_KAL, 200, 700, 0.08); a.whoosh(T_SPIN, T_KAL, 0.3, 0.2)
        a.boom(T_KAL, 40, 1.4, 0.7); a.note(fq(9, 2), T_KAL, 1.4, 0.14, "bell"); a.note(fq(9, 2) * 1.498, T_KAL, 1.0, 0.07, "sine")
        st = 0.20; tk = T_KAL + 0.15; i = 0
        while tk < T_HIT - 0.05:                                                                                       # 万華鏡: 上がりつづける 分散和音
            for j, d in enumerate((0, 2, 4)): a.note(fq(i + d, 1), tk + j * 0.02, 0.3, 0.055, "bell")
            a.note(fq(i % 5, -1), tk, 0.3, 0.05, "fm"); tk += st; st = max(0.06, st * 0.94); i += 1
        for j, d in enumerate((0, 2, 3, 5, 7)): a.note(fq(d, -1 if j < 2 else 0), T_KAL + j * 0.06, T_HIT - T_KAL + 0.2, 0.045, "pad", 0.4)
        a.rise(T_KAL, T_HIT, 70, 260, 0.12); a.a += a.lp(a.nz, 0.08) * np.clip((a.t - T_KAL) / (T_HIT - T_KAL), 0, 1) ** 2 * (a.t < T_HIT) * (a.t > T_KAL) * 0.6
        a.boom(T_HIT, 36, 1.6, 0.9)
        for j, d in enumerate((0, 2, 3, 5)): a.note(fq(d, 1), T_HIT + 0.02, 0.6, 0.06, "bell")
        s.fin_snd(a, DUR, 413); a.save(path)


if __name__ == "__main__":
    r = Kitsunebi()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
