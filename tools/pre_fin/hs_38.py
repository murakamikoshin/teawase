"""フェンリス(★5 鎖の狼) 専用の必殺技「鎖断チノ咆哮」
夜の雪原(氷のとげ)。両うでの鎖が 画面の外へ ぴんと 張っている → ガシャン、ガシャン。四方から 鎖が ふえて しばられ、霜が 鎖を つたって せまる →
鎖の輪の形の 窓のカットイン → 遠ぼえ。うしろに 氷の けずり出しの 狼の頭。声の輪が 広がり、とどいた所から 鎖が くだけ散る →
画面が 霜で まっ白に こおる → 三本の つめあとが 霜を さいて、向こうに 本人 → 霜が はがれ、白い息
使い方: python3 hs_38.py 出力.mp4 / python3 hs_38.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 11.5
ICE = np.array([0.66, 0.88, 1.0], np.float32); IRON = np.array([0.40, 0.42, 0.50], np.float32); FRO = np.array([0.86, 0.94, 1.0], np.float32)
CLK = (0.9, 1.25, 1.55, 1.8); T_FROST, T_CUT, T_HOWL, T_FRZ, T_CLAW, T_PEEL = 2.1, 3.6, 5.5, 7.5, 8.45, 8.95
ANCH = ((-0.12, 0.93), (1.15, 0.62), (-0.12, 0.12), (1.12, 0.06), (0.22, -0.06), (1.12, 0.95))
ATT = ((75, 520), (455, 350), (290, 345), (300, 345), (250, 120), (330, 470))
WOLF = [(0.00, 1.0), (0.06, 0.70), (0.16, 0.48), (0.20, 0.40), (0.08, 0.10), (0.36, 0.30), (0.50, 0.26), (0.70, 0.12), (0.93, 0.0), (1.0, 0.03), (0.96, 0.10), (0.88, 0.14), (0.62, 0.36), (0.50, 0.50), (0.80, 0.56), (0.74, 0.64), (0.50, 0.70), (0.40, 0.72), (0.45, 0.80), (0.35, 0.84), (0.42, 0.92), (0.36, 1.0)]


class Fenris(Kit):
    def __init__(s, W=540):
        super().__init__(38, (0.66, 0.88, 1.0), "鎖断チノ咆哮", "Chainbreaker Howl", W)
        H, k = s.H, s.k
        s.eL = s.A(150, 275); s.eR = s.A(255, 255); s.fc = s.A(215, 265); s.mth = s.A(190, 330)
        s.att = [s.A(*p) for p in ATT]
        s.ht = [s.text_img(c, hz.SERIF, int(70 * k), 3) for c in "鎖断チノ咆哮"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * k), 1) for c in "奥義"]
        s.HF = 0.44; s.CX = W / 2; s.flr = H * 0.89; s.CY = s.flr - s.HF * H / 2
        rg = np.random.RandomState(38); s.yn = s.YY / H
        m = s.L0()                                                                   # 地平の 氷のとげ
        x = -20 * k
        while x < W + 20 * k:
            w = (18 + 40 * rg.rand()) * k; h = (40 + 150 * rg.rand()) * k * (0.5 + abs(x / W - 0.5) * 1.6)
            cv2.fillPoly(m, [np.array([(x, s.flr), (x + w * (0.3 + 0.4 * rg.rand()), s.flr - h), (x + w, s.flr)], np.int32)], 1.0, cv2.LINE_AA); x += w * 0.7
        s.spk = m
        # 霜: ふちから のびる 結晶
        n = cv2.resize(rg.rand(40, 24).astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC); ed = np.minimum(np.minimum(s.XX, W - s.XX) / (W * 0.5), np.minimum(s.YY, H - s.YY) / (H * 0.5))
        s.fth = np.clip(ed * 0.85 + (n - 0.5) * 0.35, 0, 1.2)                           # この値より 進みが 大きい所が こおる
        L = s.L0()
        for i in range(70):
            side = i % 4; u0 = rg.rand(); p = [(u0 * W, 0), (W, u0 * H), (u0 * W, H), (0, u0 * H)][side]; a = [1.57, 3.14, -1.57, 0.0][side] + (rg.rand() - 0.5) * 1.0; ln = (120 + 240 * rg.rand()) * k
            q = (p[0] + math.cos(a) * ln, p[1] + math.sin(a) * ln); cv2.line(L, (int(p[0]), int(p[1])), (int(q[0]), int(q[1])), 1.0, max(1, int(2 * k)), cv2.LINE_AA)
            for j in range(1, 7):
                b = (p[0] + (q[0] - p[0]) * j / 7.0, p[1] + (q[1] - p[1]) * j / 7.0); bl = ln * 0.22 * (1 - j / 8.0)
                for sg in (-1, 1): cv2.line(L, (int(b[0]), int(b[1])), (int(b[0] + math.cos(a + sg * 1.05) * bl), int(b[1] + math.sin(a + sg * 1.05) * bl)), 0.8, 1, cv2.LINE_AA)
        s.den = cv2.GaussianBlur(L, (0, 0), 0.7 * k)
        s.ftex = (0.80 + 0.12 * n)[:, :, None] * FRO[None, None, :] + s.den[:, :, None] * 0.25
        s.rs = rg.rand(200, 4).astype(np.float32)
        # つめあと 3本(右上→左下)
        s.claw = []
        for j in range(3):
            c = np.array([W * (0.50 + (j - 1) * 0.21), H * (0.46 + (j - 1) * 0.03)], np.float32); d = np.array([-0.26, 0.966], np.float32); nn = np.array([0.966, 0.26], np.float32); ln = H * (0.40 if j != 1 else 0.47)
            s.claw.append((c, d, nn, ln))

    # ---- 大道具 ----
    def field(s, t, dark=0.0):
        f = (np.array([0.02, 0.04, 0.10], np.float32)[None, None, :] * (1 - s.yn)[:, :, None] + np.array([0.10, 0.17, 0.30], np.float32)[None, None, :] * s.yn[:, :, None]).astype(np.float32)
        f = f * (1 - s.spk[:, :, None] * 0.75) + (s.spk * 0.75)[:, :, None] * np.array([0.16, 0.26, 0.40], np.float32)[None, None, :] * (0.4 + 0.6 * s.yn)[:, :, None]
        g = int(s.flr); f[g:] = np.array([0.52, 0.62, 0.78], np.float32) * (1.25 - 0.5 * s.yn[g:])[:, :, None]
        return f * (1 - 0.45 * dark)

    def chain(s, f, i, t, grow=1.0, frost=0.0, vib=0.0, howl=None):
        """鎖 1本: 画面の外(ANCH) → 体(ATT)。frost: 外から こおる割合。howl: (中心, 時刻) 声の輪が とどくと くだける"""
        W, H, k = s.W, s.H, s.k; a = np.array([ANCH[i][0] * W, ANCH[i][1] * H], np.float32); b = np.array(s.at(*s.att[i]), np.float32)
        d = b - a; L = float(np.linalg.norm(d)); d /= L; nn = np.array([-d[1], d[0]], np.float32); ll = 44 * k; n = int(L / (ll * 0.72)); ang = math.degrees(math.atan2(d[1], d[0]))
        for j in range(n):
            u = (j + 0.5) / n
            if u > grow: break
            p = a + d * L * u + nn * math.sin(u * 3.1416) * math.sin(u * 14 + t * 60) * vib
            fr = clamp((frost - u) / 0.06); col = IRON * (1 - fr) + FRO * fr
            if howl is not None:
                hd = math.hypot(p[0] - howl[0][0], p[1] - howl[0][1]); v = t - (howl[1] + 0.2 + hd / (W * 1.5))
                if v > 0:                                                            # くだけて とぶ
                    if v < 0.7:
                        r = s.rs[(i * 31 + j) % 200]; o = np.array([p[0] - howl[0][0], p[1] - howl[0][1]], np.float32) / (hd + 1e-3)
                        q = p + o * W * 0.5 * v * (0.5 + r[0]) + np.array([0, 300 * k * v * v], np.float32); ro = r[1] * 6.28 + v * (r[2] - 0.5) * 30; sz = 16 * k
                        cv2.ellipse(f, (int(q[0]), int(q[1])), (int(sz), int(sz * 0.6)), math.degrees(ro), 20, 200, tuple(float(c) for c in FRO * (1 - v / 0.7)), max(1, int(5 * k)), cv2.LINE_AA)
                    continue
            c3 = (float(col[0]), float(col[1]), float(col[2]))
            if j % 2 == 0:
                cv2.ellipse(f, (int(p[0]), int(p[1])), (int(ll * 0.5), int(ll * 0.30)), ang, 0, 360, (0.02, 0.03, 0.05), max(1, int(13 * k)), cv2.LINE_AA)
                cv2.ellipse(f, (int(p[0]), int(p[1])), (int(ll * 0.5), int(ll * 0.30)), ang, 0, 360, c3, max(1, int(8 * k)), cv2.LINE_AA)
            else:
                e0 = p - d * ll * 0.5; e1 = p + d * ll * 0.5
                cv2.line(f, (int(e0[0]), int(e0[1])), (int(e1[0]), int(e1[1])), (0.02, 0.03, 0.05), max(1, int(14 * k)), cv2.LINE_AA); cv2.line(f, (int(e0[0]), int(e0[1])), (int(e1[0]), int(e1[1])), c3, max(1, int(9 * k)), cv2.LINE_AA)

    def hero(s, f, t, **kw):
        img = s.pose(t, **kw); s.draw(f, img, s.CX, s.CY, s.HF)

    def eyes(s, f, v):
        if v <= 0.01: return
        sc = s._pl[2]; L = s.L0()
        for e in (s.eL, s.eR): p = s.at(*e); cv2.circle(L, (int(p[0]), int(p[1])), max(1, int(11 * sc)), float(min(1.0, v)), -1, cv2.LINE_AA)
        s.add(f, L, ICE, 0.8, 2, 7)

    def breath(s, f, t, v=1.0):
        p = s.at(*s.mth); u = (t * 0.6) % 1.0
        s.soft(f, (p[0] - 30 * s.k * u - 10 * s.k, p[1] - 26 * s.k * u), (10 + 26 * u) * s.k, 0.5 * v * math.sin(u * 3.1416), FRO)

    def wolf(s, f, g, t):
        """氷の けずり出しの 狼の頭(横顔・上を向いて ほえる)"""
        W, H, k = s.W, s.H, s.k; bw, bh = W * 0.92 * g, H * 0.60 * g; ox = W * 0.50 - bw * 0.46; oy = s.CY - H * 0.02 - bh
        P = [(ox + x * bw, oy + y * bh) for x, y in WOLF]; m = s.L0(); cv2.fillPoly(m, [np.array(P, np.int32)], 1.0, cv2.LINE_AA)
        sh = (0.20 + 0.10 * np.sin(s.XX / (37 * k) + s.YY / (23 * k)) * np.sign(np.sin(s.XX / (61 * k) - s.YY / (47 * k))))
        f += (m * sh)[:, :, None] * ICE[None, None, :]
        L = s.L0(); cv2.polylines(L, [np.array(P, np.int32)], True, 1.0, max(1, int(3 * k)), cv2.LINE_AA)
        e = (ox + 0.47 * bw, oy + 0.36 * bh)
        for j in (1, 2, 4, 7, 12, 13, 16, 18): cv2.line(L, (int(e[0]), int(e[1])), (int(P[j][0]), int(P[j][1])), 0.45, 1, cv2.LINE_AA)       # けずり面の 線
        for a0, a1, a2 in (((0.86, 0.155), (0.845, 0.28), (0.80, 0.20)), ((0.76, 0.24), (0.745, 0.35), (0.70, 0.29)), ((0.77, 0.552), (0.755, 0.44), (0.715, 0.54)), ((0.67, 0.53), (0.66, 0.44), (0.62, 0.52))):
            cv2.fillPoly(L, [np.array([(ox + a0[0] * bw, oy + a0[1] * bh), (ox + a1[0] * bw, oy + a1[1] * bh), (ox + a2[0] * bw, oy + a2[1] * bh)], np.int32)], 1.0, cv2.LINE_AA)   # きば
        cv2.fillPoly(L, [np.array([(e[0] - 26 * k * g, e[1] + 4 * k), (e[0], e[1] - 12 * k * g), (e[0] + 30 * k * g, e[1] - 8 * k), (e[0], e[1] + 8 * k * g)], np.int32)], 1.0, cv2.LINE_AA)   # 目
        s.add(f, L, ICE, 0.75, 2, 8)
        return (ox + 0.74 * bw, oy + 0.38 * bh)

    def frame(s, t):
        if t < T_CUT: f, sh = s.sc_bind(t)
        elif t < T_HOWL: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_FRZ + 0.95: f, sh = s.sc_howl(t)
        else: f, sh = s.sc_claw(t)
        return s.post(f, ICE, 0.30, sh * s.k, t, 0.97)

    # 1) しばられる
    def sc_bind(s, t):
        W, H, k = s.W, s.H, s.k
        nc = sum(1 for c in CLK if t >= c); last = max([c for c in CLK if t >= c] + [-9]); v = t - last
        f = s.field(t, clamp((t - 0.8) / 1.5))
        s._pose = dict(t=t, sway=0, breath=0, lean=0, squash=0, stretch=None); s._pl = (s.CX, s.CY, s.HF * H / s.ah)
        fr = 0.9 * ease2((t - T_FROST) / (T_CUT - T_FROST - 0.1)) if t > T_FROST else 0.0
        for i in (2, 3, 4):
            if i - 2 < nc - 0 and t >= CLK[min(i - 2, 3)]: s.chain(f, i, t, ease((t - CLK[i - 2]) / 0.10), fr, 5 * k * math.exp(-(t - CLK[i - 2]) * 7))
        s.hero(f, t, jit=1.6 * k * math.exp(-v * 12) if v < 0.5 else 0)
        for i in (0, 1): s.chain(f, i, t, 1.0, fr, 3 * k * math.exp(-v * 7) if v < 1 else 0)
        if nc >= 4: s.chain(f, 5, t, ease((t - CLK[3]) / 0.10), fr, 5 * k * math.exp(-(t - CLK[3]) * 7))
        s.breath(f, t, 0.8)
        if fr > 0.5: s.eyes(f, (fr - 0.5) * 2)
        if 0 <= v < 0.06: f = f + (1 - v / 0.06) * 0.25
        return f * clamp(t / 0.4), 5 * math.exp(-v * 12) if v < 0.5 else 0

    # 2) 鎖の輪の形の 窓
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = np.zeros((H, W, 3), np.float32) + np.array([0.02, 0.04, 0.09], np.float32) + (s.den * 0.10)[:, :, None] * ICE[None, None, :]
        g = back(u / 0.3, 1.3); C = (W / 2, H * 0.38); hw, hh = W * 0.43 * g, H * 0.17 * g
        def stad(img, c, a, b, v, th):
            if th < 0: cv2.rectangle(img, (int(c[0] - a + b), int(c[1] - b)), (int(c[0] + a - b), int(c[1] + b)), v, -1); cv2.circle(img, (int(c[0] - a + b), int(c[1])), int(b), v, -1, cv2.LINE_AA); cv2.circle(img, (int(c[0] + a - b), int(c[1])), int(b), v, -1, cv2.LINE_AA)
            else:
                cv2.line(img, (int(c[0] - a + b), int(c[1] - b)), (int(c[0] + a - b), int(c[1] - b)), v, th, cv2.LINE_AA); cv2.line(img, (int(c[0] - a + b), int(c[1] + b)), (int(c[0] + a - b), int(c[1] + b)), v, th, cv2.LINE_AA)
                cv2.ellipse(img, (int(c[0] - a + b), int(c[1])), (int(b), int(b)), 0, 90, 270, v, th, cv2.LINE_AA); cv2.ellipse(img, (int(c[0] + a - b), int(c[1])), (int(b), int(b)), 0, -90, 90, v, th, cv2.LINE_AA)
        for sg in (-1, 1):                                                           # となりの輪(横向き)
            cv2.line(f, (int(C[0] + sg * W * 0.36), int(C[1])), (int(C[0] + sg * W * 0.70), int(C[1])), (0.02, 0.03, 0.05), int(40 * k), cv2.LINE_AA); cv2.line(f, (int(C[0] + sg * W * 0.36), int(C[1])), (int(C[0] + sg * W * 0.70), int(C[1])), (0.36, 0.38, 0.46), int(28 * k), cv2.LINE_AA)
        m = s.L0(); stad(m, C, hw, hh, 1.0, -1)
        win = np.zeros((H, W, 3), np.float32) + np.array([0.10, 0.20, 0.36], np.float32)
        zf = 1.10 + 0.05 * ease(u / 1.8); sc = zf * H / s.ah; img = s.pose(t)
        s.draw(win, img, C[0] - (s.fc[0] - s.aw / 2) * sc, C[1] + H * 0.01 - (s.fc[1] - s.ah / 2) * sc, zf); s.eyes(win, 0.9 + 1.2 * math.exp(-max(0.0, u - 0.45) * 5) * (u > 0.45))
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        stad(f, C, hw, hh, (0.02, 0.03, 0.05), int(34 * k)); stad(f, C, hw, hh, (0.42, 0.44, 0.52), int(24 * k)); stad(f, (C[0], C[1] - 3 * k), hw, hh, (0.62, 0.66, 0.76), int(5 * k))
        cr = (u - 0.9) / 0.5                                                         # 輪に ひびが 入る
        if cr > 0:
            L = s.L0(); p = (C[0] + hw * 0.55, C[1] - hh)
            for j, (dx, dy) in enumerate(((0, -18), (10, -6), (-4, 6), (12, 18))):
                if cr > j * 0.2: q = (p[0] + dx * k, p[1] + dy * k); s.line(L, p, q, 1.0, 2); p = q
            s.add(f, L, ICE, 0.9, 1, 5)
        n = len(s.ht); step = W * 0.148
        for i, T in enumerate(s.ht):                                                 # 技名: ガシャンと 1字ずつ 落ちて 止まる
            a = (u - 0.5 - i * 0.09) / 0.12
            if a > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.82 - (1 - min(1.0, a)) ** 2 * 80 * k, clamp(a * 2), 1.0)
        for i, T in enumerate(s.vo):
            if u > 0.4 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.745, 1.0, 1.0)
        if u > 1.25: s.text(f, s.txt_en, W / 2, H * 0.82 + 56 * k, ease((u - 1.25) / 0.3), 1.0)
        if u < 0.08: f = f + (1 - u / 0.08) * 0.7
        return f, 0

    def scene(s, t, chains=True):
        """遠ぼえの 場面(霜の下に しく絵 にも使う)"""
        W, H, k = s.W, s.H, s.k; u = t - T_HOWL
        f = s.field(t, 0.6 * (1 - clamp(u / 1.5)))
        s._pose = dict(t=t, sway=0, breath=0, lean=0, squash=0, stretch=None); s._pl = (s.CX, s.CY, s.HF * H / s.ah)
        hw = clamp(1 - (t - T_FRZ - 0.2) / 0.5)
        mc = s.at(*s.mth)
        if hw > 0 and u > 0.08:
            fw = np.zeros_like(f); mc = s.wolf(fw, back((u - 0.08) / 0.35, 1.2), t); f += fw * hw
        if chains:
            for i in (2, 3, 4): s.chain(f, i, t, 1.0, 0.9, 0, (mc, T_HOWL))
        s.hero(f, t, squash=-0.06 * math.exp(-u * 5) * (u > 0), jit=1.2 * k * clamp(1.4 - u) * (u > 0))
        if chains:
            for i in (0, 1, 5): s.chain(f, i, t, 1.0, 0.9, 0, (mc, T_HOWL))
        if hw > 0 and u > 0.15:                                                      # 声の輪
            L = s.L0()
            for j in range(7):
                v = (u - 0.15 - j * 0.2) / 0.9
                if 0.08 < v < 1: s.circ(L, mc, W * 1.5 * v + 10 * k, (1 - v) * hw, max(2, int(6 * k * (1 - v) + 2)))
            s.add(f, L, ICE, 0.7, 2, 9)
        s.eyes(f, 1.0)
        return f, mc

    # 3) 遠ぼえ → 鎖が くだける → 霜で こおる
    def sc_howl(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_HOWL; z = t - T_FRZ
        f, mc = s.scene(t)
        if 0 <= u < 0.1: f = f + (1 - u / 0.1) * 0.6
        if z > 0:
            pr = ease2(z / 0.7) * 1.25; m = np.clip((pr - s.fth) / 0.08, 0, 1)
            f = f * (1 - m[:, :, None]) + s.ftex * m[:, :, None]; f += (np.clip(1 - np.abs(pr - s.fth) / 0.05, 0, 1) * 0.5)[:, :, None]
        return f, (7 * clamp(1.5 - u) * (u > 0) if z < 0 else 0)

    # 4) つめあとが 霜を さく → はがれる → 白い息
    def sc_claw(s, t):
        W, H, k = s.W, s.H, s.k; c = t - T_CLAW; p = t - T_PEEL
        f = s.field(t, 0.0); s._pose = dict(t=t, sway=0, breath=0, lean=0, squash=0, stretch=None); s._pl = (s.CX, s.CY, s.HF * H / s.ah)
        for i, (x, r) in enumerate(((0.16, 20), (0.30, -35), (0.71, 60), (0.84, 10), (0.60, -20))):    # 雪の上の 鎖のかけら
            cv2.ellipse(f, (int(W * x), int(s.flr + (18 + 9 * (i % 3)) * k)), (int(15 * k), int(8 * k)), r, 0, 360, (0.25, 0.28, 0.36), max(1, int(5 * k)), cv2.LINE_AA)
        s.hero(f, t, squash=(0.05 * math.exp(-c * 9) if c > 0 else 0)); s.eyes(f, 1.0 if p < 0.6 else max(0.0, 1.6 - p)); s.breath(f, t, clamp(p))
        m = s.L0() + 1.0; edge = s.L0(); sh = 0
        if c > 0:
            wd = (26 + (W * 1.2 / k) * ease2(p / 0.55) if p > 0 else 26) * k
            for j, (cc, d, nn, ln) in enumerate(s.claw):
                v = (c - j * 0.07) / 0.09
                if v <= 0: continue
                e = ease(v); a = cc - d * ln; b = cc - d * ln + d * 2 * ln * e; mid = (a + b) / 2
                P = np.array([a, mid + nn * wd, b, mid - nn * wd], np.int32); cv2.fillPoly(m, [P], 0.0, cv2.LINE_AA)
                if p < 0.3: cv2.polylines(edge, [P], True, float(1 - max(0.0, p) / 0.3), max(1, int(3 * k)), cv2.LINE_AA)
            sh = 13 * math.exp(-c * 7)
        if p > 0: m *= (1 - ease(p / 0.55))
        f = f * (1 - m[:, :, None]) + s.ftex * m[:, :, None]
        if c > 0: s.add(f, edge, WHITE, 0.8, 2, 6)
        if 0 <= c < 0.08: f = f + (1 - c / 0.08) * 0.5
        end = DUR - t
        if end < 0.45: v = clamp((0.45 - end) / 0.4); f = f * (1 - v) + v
        return f, sh

    # ---- この子だけの曲: 低い うなりと 風。鎖は 金物の「ガシャン」。カットインは 戦太鼓と 低い 男声ふうの 和音(ホのフリギア)。遠ぼえは のびる 声(すべり上がって ゆれる)と 鎖の 高い くだけ音。霜は ぱりぱり。つめは 3回の するどい 風切り。さいごに 遠くの 遠ぼえ ----
    def audio(s, path):
        a = Snd(DUR, 38); E = 82.41
        def clank(t0, g=0.3):
            for fr, gg, dc in ((523, 1.0, 0.10), (1318, 0.7, 0.07), (2217, 0.5, 0.05), (3729, 0.3, 0.03)): a.note(fr * (1 + 0.02 * math.sin(t0 * 9)), t0, dc, g * gg * 0.3, "sq", 0.0005)
            a.hat(t0, g, 0.03); a.boom(t0, 75, g * 1.6, 0.14)
        def howl(t0, ln, g=0.12, f0=300.0, f1=520.0):
            u = np.clip((a.t - t0) / ln, 0, 1); fr = f0 + (f1 - f0) * np.minimum(1, u * 4) ** 0.6 - (f1 - f0) * 0.35 * np.clip((u - 0.7) / 0.3, 0, 1) + 6 * np.sin(2 * np.pi * 5.5 * a.t) * u
            ph = 2 * np.pi * np.cumsum(fr) / a.sr; e = np.clip(u * 10, 0, 1) * np.clip((1 - u) * 3, 0, 1) * ((a.t >= t0) & (a.t < t0 + ln))
            a.a += (np.sin(ph) + 0.45 * np.sin(2 * ph) + 0.22 * np.sin(3 * ph) + 0.1 * np.sin(4 * ph)) * e * g
        def choir(semi, t0, ln, g=0.04):
            for dt in (0.997, 1.0, 1.004): a.note(E * 2 * 2 ** (semi / 12.0) * dt, t0, ln, g, "pad", 0.15)
        a.a += a.lp(a.nz, 0.04) * (0.5 + 0.3 * np.sin(a.t * 1.7)) * 0.7 * a.gate(0, DUR, 0.4) * (1 - 0.6 * a.gate(T_CUT, T_HOWL, 0.2)) + a.tone(41.2) * 0.07 * a.gate(0, T_CUT, 0.4)
        for i, c in enumerate(CLK): clank(c, 0.35 + 0.06 * i)
        tk = T_FROST                                                                                                            # 霜が つたう: ぱり、ぱり
        rg = np.random.RandomState(5)
        while tk < T_CUT - 0.1: a.hat(tk, 0.10, 0.006); a.note(3000 + 2500 * rg.rand(), tk, 0.015, 0.03, "sine", 0.0005); tk += 0.03 + 0.09 * rg.rand()
        a.rise(T_FROST, T_CUT, 200, 500, 0.05)
        a.boom(T_CUT, 50, 1.0, 0.35); clank(T_CUT + 0.02, 0.3)
        for i in range(6): clank(T_CUT + 0.5 + i * 0.09, 0.16)
        for dt in (0.3, 0.75, 1.05, 1.2, 1.5): a.boom(T_CUT + dt, 62, 0.7, 0.2)
        choir(0, T_CUT + 0.2, 0.7); choir(7, T_CUT + 0.2, 0.7, 0.03); choir(1, T_CUT + 1.0, 0.7); choir(8, T_CUT + 1.0, 0.7, 0.03)
        a.note(2400, T_CUT + 0.95, 0.1, 0.05, "sine")                                                                           # ひび
        howl(T_HOWL, 1.9, 0.14); howl(T_HOWL + 0.03, 1.9, 0.07, 150, 260); a.boom(T_HOWL, 42, 1.3, 0.6)
        choir(0, T_HOWL, 1.6, 0.04); choir(7, T_HOWL, 1.6, 0.03); choir(12, T_HOWL + 0.4, 1.2, 0.03)
        for i in range(34): t0 = T_HOWL + 0.3 + 1.3 * rg.rand(); a.note(1800 + 3200 * rg.rand(), t0, 0.06, 0.035, "bell", 0.0005); a.hat(t0, 0.06, 0.01)   # 鎖が くだける
        tk = T_FRZ
        while tk < T_FRZ + 0.75: a.hat(tk, 0.12, 0.006); a.note(2500 + 3500 * rg.rand(), tk, 0.012, 0.035, "sine", 0.0005); tk += 0.012 + 0.03 * rg.rand()
        a.a += (a.nz - a.lp(a.nz, 0.8)) * a.gate(T_FRZ, T_FRZ + 0.8, 0.2) * 0.08; a.rise(T_FRZ, T_FRZ + 0.75, 800, 3200, 0.05)
        for j in range(3): t0 = T_CLAW + j * 0.07; a.rise(t0, t0 + 0.09, 3000 - j * 300, 900, 0.16); a.hat(t0 + 0.03, 0.5, 0.06); a.boom(t0 + 0.08, 60 - j * 6, 0.8, 0.2)
        a.a += a.lp(a.nz, 0.4) * a.env(T_PEEL, 0.02, 0.4) * 0.35
        for i in range(12): a.note(2200 + 2600 * rg.rand(), T_PEEL + 0.03 * i, 0.05, 0.03, "bell", 0.0005)
        howl(T_PEEL + 0.8, 1.5, 0.045, 260, 440); choir(0, T_PEEL + 0.7, 1.4, 0.03); choir(7, T_PEEL + 0.7, 1.4, 0.02)
        a.save(path)


if __name__ == "__main__":
    r = Fenris()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
