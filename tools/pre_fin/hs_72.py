"""アマツ(★6 雷の帝) 専用の必殺技「天裂ク雷帝」
流れる黒雲の中で 遠雷が光る → 大きな落雷の中から アマツが現れる(一瞬 白黒が反転) → 稲妻の形に裂けた窓の カットイン →
剣を天へ。うずを巻く雲から 雷が 剣に何度も落ち(だんだん速く)、剣が 白く焼ける → 斜めに駆け抜けて 一閃(止め絵) →
雲そのものが 斜めに裂けて ずれ、すき間から 金の空と 雷の雨 → とどめの大落雷
使い方: python3 hs_72.py 出力.mp4 / python3 hs_72.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 12.8
YEL = np.array([1.0, 0.86, 0.22], np.float32); HOT = np.array([1.0, 0.97, 0.80], np.float32); SLATE = np.array([0.20, 0.23, 0.34], np.float32); AMBER = np.array([1.0, 0.62, 0.12], np.float32)
T_BOLT, T_CUT, T_CHARGE, T_DASH, T_STOP, T_TEAR, T_FINAL = 1.7, 2.5, 4.6, 8.4, 8.62, 9.05, 10.9
STRIKES = [T_CHARGE + x for x in (0.7, 1.4, 1.95, 2.35, 2.65, 2.88, 3.06)]


class Amatsu(Kit):
    def __init__(s, W=540):
        super().__init__(72, (1.0, 0.86, 0.22), "天裂ク雷帝", "Heavensplitter", W)
        s.eyeL = s.A(396, 263); s.eyeR = s.A(496, 263); s.tip = s.A(22, 72); s.guard = s.A(322, 300)
        s.vt = [s.text_img(c, hz.SERIF, int(70 * s.k), 3) for c in "天裂ク雷帝"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        rg = np.random.RandomState(72)
        n = cv2.GaussianBlur(rg.rand(s.H // 12 + 2, s.W * 3 // 12 + 2).astype(np.float32), (0, 0), 2.2)
        n = (n - n.min()) / (n.max() - n.min()); s.cl = cv2.resize(n, (s.W * 3, s.H), interpolation=cv2.INTER_CUBIC)
        s.rr = rg.rand(120, 5).astype(np.float32)
        s.diag = (s.YY - (s.H * 0.64 - s.XX / s.W * s.H * 0.34))                    # 裂け目の線からの距離(下が正)

    # ---- この子だけの絵 ----
    def clouds(s, t, lum=1.0, flashes=()):
        W = s.W; o1 = int((t * 60) % (2 * W)); o2 = int((t * 34 + 500) % (2 * W))
        d = np.clip(s.cl[:, o1:o1 + W] * 0.75 + s.cl[::-1, o2:o2 + W] * 0.75 - 0.42, 0, 1)
        f = SLATE[None, None, :] * (0.10 + 1.5 * d)[:, :, None] * lum
        for (x, y, v) in flashes:                                                    # 雲の中の 稲光
            if v > 0: f += (np.exp(-(((s.XX - x) ** 2 + (s.YY - y) ** 2) / (W * 0.30) ** 2)) * d * v * 2.2)[:, :, None] * (YEL * 0.7 + 0.3)[None, None, :]
        return f.astype(np.float32), d

    def bolt(s, L, p0, p1, seed, rough=0.26, v=1.0, th=2, depth=2):
        rg = np.random.RandomState(int(seed) % 100000); pts = [np.array(p0, np.float32), np.array(p1, np.float32)]
        for _ in range(5):
            new = [pts[0]]
            for a, b in zip(pts, pts[1:]):
                dv = b - a; nrm = np.array([-dv[1], dv[0]], np.float32); new += [(a + b) / 2 + nrm * (rg.rand() - 0.5) * rough, b]
            pts = new
        cv2.polylines(L, [np.array(pts, np.int32)], False, float(v), th, cv2.LINE_AA)
        if depth > 0:
            for _ in range(2):
                i = rg.randint(4, len(pts) - 4); a = pts[i]; dv = (np.array(p1, np.float32) - np.array(p0, np.float32)) * (0.25 + 0.2 * rg.rand())
                ang = (rg.rand() - 0.5) * 1.6; c, sn = math.cos(ang), math.sin(ang)
                s.bolt(L, a, a + np.array([dv[0] * c - dv[1] * sn, dv[0] * sn + dv[1] * c]), rg.randint(99999), rough, v * 0.55, 1, depth - 1)

    def hero(s, f, t, cx, cy, hfrac, deg=0.0, alpha=1.0, white=0.0, **kw):
        img = s.pose(t, **kw)
        if deg:
            M = cv2.getRotationMatrix2D((s.pw / 2, s.ph / 2), deg, 1.0); img = cv2.warpAffine(img, M, (s.pw, s.ph), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        s.draw(f, img, cx, cy, hfrac, alpha, white); s._deg = deg

    def pt(s, a):
        p = s.at(*a); cx, cy, _ = s._pl; r = math.radians(s._deg); dx, dy = p[0] - cx, p[1] - cy
        return (cx + math.cos(r) * dx + math.sin(r) * dy, cy - math.sin(r) * dx + math.cos(r) * dy)

    def eyes(s, f, v):
        if v <= 0: return
        L = s.L0(); sc = s._pl[2]
        for e in (s.eyeL, s.eyeR):
            p = s.pt(e); cv2.ellipse(L, (int(p[0]), int(p[1])), (max(1, int(15 * sc)), max(1, int(19 * sc))), 0, 0, 360, float(v), -1, cv2.LINE_AA)
            s.bolt(L, p, (p[0] + (60 if e is s.eyeR else -60) * sc * 2, p[1] - 50 * sc * 2), int(s._t * 15) + (7 if e is s.eyeR else 0), 0.35, 0.8 * v, 1, 0)     # 目から 小さな稲妻
        s.add(f, L, YEL, 1.3, 2, 8)

    def blade(s, f, heat, t):
        """剣が 焼けて 白く光る"""
        if heat <= 0: return
        a = s.pt(s.guard); b = s.pt(s.tip); L = s.L0()
        cv2.line(L, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), float(min(1.0, heat)), max(1, int((5 + 7 * heat) * s.k)), cv2.LINE_AA)
        for i in range(int(2 + heat * 4)):                                           # 剣を はう 小さな放電
            q = s.rr[(int(t * 20) + i * 7) % 120]; p = (a[0] + (b[0] - a[0]) * q[0], a[1] + (b[1] - a[1]) * q[0])
            s.bolt(L, p, (p[0] + (q[1] - 0.5) * 120 * s.k, p[1] + (q[2] - 0.5) * 120 * s.k), int(t * 20) * 31 + i, 0.4, 0.7, 1, 0)
        s.add(f, L, YEL * (1 - heat * 0.4) + HOT * heat * 0.4, 0.5 + 0.4 * heat, 2, 8)

    def frame(s, t):
        k = s.k; s._t = t
        if t < T_CUT: f, sh = s.sc_storm(t)
        elif t < T_CHARGE: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_DASH: f, sh = s.sc_charge(t)
        else: f, sh = s.sc_split(t)
        return s.post(f, YEL, 0.45, sh * k, t, 0.95)

    # 1) 黒雲と遠雷 → 落雷の中から 現れる
    def sc_storm(s, t):
        W, H, k = s.W, s.H, s.k
        fl = [(W * 0.25, H * 0.2, math.exp(-abs(t - 0.5) * 14)), (W * 0.8, H * 0.35, math.exp(-abs(t - 1.0) * 14)), (W * 0.4, H * 0.12, math.exp(-abs(t - 1.35) * 14))]
        f, d = s.clouds(t, clamp(t / 0.4), fl)
        u = t - T_BOLT; C = (W / 2, H * 0.60)
        if u > 0:
            if u < 0.34:                                                             # 落雷(2コマごとに 形が変わる)
                L = s.L0(); sd = int(u * 15)
                s.bolt(L, (W * 0.5 + 40 * k, -10), (C[0], C[1] + H * 0.14), 100 + sd, 0.22, 1.0, 5); s.bolt(L, (W * 0.5 - 60 * k, -10), (C[0], C[1] + H * 0.14), 200 + sd, 0.3, 0.7, 2)
                s.add(f, L, HOT, 1.6, 4, 20)
            inv = u < 0.14
            s.hero(f, t, C[0], C[1], 0.46, 0, 1.0, (1.0 if u < 0.07 else 0.0), squash=0.10 * math.exp(-u * 9) * math.cos(u * 24))
            if inv: f = 1 - np.clip(f, 0, 1)                                         # 一瞬の 反転
            L = s.L0()
            for i in range(3):
                v = (u - i * 0.1) / 0.7
                if 0 < v < 1: s.ell(L, (C[0], C[1] + H * 0.19), (W * 0.9 * v, W * 0.15 * v), 0, 1 - v, 3)
            for i in range(4):                                                       # からだに 残る 放電
                if (int(t * 12) + i) % 3 == 0:
                    q = s.rr[(int(t * 12) * 3 + i) % 120]; p = (C[0] + (q[0] - 0.5) * W * 0.5, C[1] + (q[1] - 0.5) * H * 0.3)
                    s.bolt(L, p, (p[0] + (q[2] - 0.5) * 140 * k, p[1] + (q[3] - 0.5) * 140 * k), int(t * 12) * 17 + i, 0.4, 0.8, 1, 0)
            s.add(f, L, YEL, 1.0)
            s.eyes(f, clamp((u - 0.3) / 0.3))
        return f, (16 * math.exp(-u * 5) if u > 0 else 0)

    # 2) 稲妻の形に裂けた窓 と 雷に打たれて出る 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = np.zeros((H, W, 3), np.float32) + np.array([0.02, 0.02, 0.03], np.float32)
        ex = (s.eyeL[0] + s.eyeR[0]) / 2; ey = s.eyeL[1]; E = (W * 0.60, H * 0.36)
        L = s.L0()
        for i in range(48):                                                          # 目に集まる 集中線
            a = i * 0.1309; r0 = W * (0.42 + 0.08 * s.rr[i][0]); cv2.fillPoly(L, [np.array([(E[0] + math.cos(a) * r0, E[1] + math.sin(a) * r0), (E[0] + math.cos(a - 0.018) * W * 2, E[1] + math.sin(a - 0.018) * W * 2), (E[0] + math.cos(a + 0.018) * W * 2, E[1] + math.sin(a + 0.018) * W * 2)], np.int32)], 0.5, cv2.LINE_AA)
        f += L[:, :, None] * YEL[None, None, :] * 0.5 * ease(u / 0.3)
        op = ease(u / 0.22); hw = W * 0.21 * op
        zig = [(W * 0.86, -10), (W * 0.52, H * 0.26), (W * 0.74, H * 0.30), (W * 0.36, H * 0.60), (W * 0.58, H * 0.63), (W * 0.20, H + 10)]
        left = [(x - hw, y) for x, y in zig]; right = [(x + hw, y) for x, y in zig]
        m = s.L0(); cv2.fillPoly(m, [np.array(left + right[::-1], np.int32)], 1.0, cv2.LINE_AA)
        win = np.zeros((H, W, 3), np.float32) + (AMBER * 0.35 + YEL * 0.25)[None, None, :]
        zf = 1.2; sc = zf * H / s.ah
        s.hero(win, t, E[0] - (ex - s.aw / 2) * sc, E[1] - (ey - s.ah / 2) * sc, zf)
        s.eyes(win, 0.55)
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        L = s.L0(); cv2.polylines(L, [np.array(left, np.int32)], False, 1.0, 3, cv2.LINE_AA); cv2.polylines(L, [np.array(right, np.int32)], False, 1.0, 3, cv2.LINE_AA)
        s.add(f, L, YEL, 1.1)
        for i, T in enumerate(s.vt):                                                 # 1字ずつ 雷に打たれて 出る
            a = (u - 0.45 - i * 0.16) / 0.1
            if a > 0:
                p = (W * 0.13, H * 0.44 + i * 80 * k); s.text(f, T, p[0], p[1], 1.0, 1.0)
                if a < 1.6:
                    L2 = s.L0(); s.bolt(L2, (p[0] + 30 * k, -10), p, 300 + i * 9 + int(a * 3), 0.2, 1.0, 3, 1); s.add(f, L2, HOT, 1.3 * clamp(1.6 - a))
        for i, T in enumerate(s.vo):
            if u > 0.25 + i * 0.1: s.text(f, T, W * 0.13, H * 0.30 + i * 36 * k, 1.0, 1.0)
        if u > 1.4: s.text(f, s.txt_en, W * 0.62, H * 0.93, ease((u - 1.4) / 0.3), 1.0)
        if u < 0.1: f += (1 - u / 0.1) * 0.9
        return f, 0

    # 3) 剣を天へ。うず雲から 雷が 何度も落ちる
    def sc_charge(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_CHARGE
        O = (W / 2, H * 0.17)
        last = max([x for x in STRIKES if x <= t] or [-9]); since = t - last; hits = len([x for x in STRIKES if x <= t]); heat = hits / len(STRIKES)
        f, d = s.clouds(t * 0.5, 0.75, [(O[0], O[1], 0.15 + 0.9 * math.exp(-since * 10))])
        L = s.L0()
        for j in range(5):                                                           # うずを巻く雲
            for i in range(26):
                a = j * 1.2566 + i * 0.16 - t * (1.2 + heat * 2.5); rr = W * (0.06 + i * 0.03)
                cv2.circle(L, (int(O[0] + math.cos(a) * rr), int(O[1] + math.sin(a) * rr * 0.45)), int((6 + i * 1.3) * k), 0.10, -1)
        f += cv2.GaussianBlur(L, (0, 0), 9 * k)[:, :, None] * (SLATE * 2.2 + YEL * 0.5 * heat)[None, None, :]
        hole = np.exp(-(((s.XX - O[0]) / (W * 0.10)) ** 2 + ((s.YY - O[1]) / (W * 0.05)) ** 2)); f = f * (1 - hole * 0.9)[:, :, None]
        crouch = ease((u - 3.15) / 0.5)
        cy = H * 0.70 + crouch * H * 0.02
        s.soft(f, (W / 2, cy + H * 0.1), W * 0.30, 0.05 + 0.10 * heat, YEL)
        s.hero(f, t, W / 2, cy, 0.42, -24 * ease(u / 0.4) + 16 * crouch, squash=0.05 * math.exp(-since * 12) + 0.07 * crouch, jit=(2.0 * k * crouch))
        tip = s.pt(s.tip)
        if 0 <= since < 0.13:                                                        # 雷が 剣に 落ちる
            L = s.L0(); s.bolt(L, O, tip, int(last * 100) + int(since * 20), 0.3, 1.0, 4 + hits // 2); s.add(f, L, HOT, 1.7, 4, 18)
            f += math.exp(-since * 30) * 0.45
        s.blade(f, heat, t); s.eyes(f, 0.25 + 0.35 * heat)
        if u < 0.15: f += (1 - u / 0.15) * 0.7
        return f, 9 * math.exp(-since * 9) * (0.6 + heat) + 3 * crouch

    # 4) 一閃 → 雲が 裂けて ずれる → 雷の雨 → 大落雷
    def sc_split(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_DASH
        P0 = (-W * 0.15, H * 0.64 + W * 0.15 / W * H * 0.34 + H * 0.06); P1 = (W * 1.15, H * 0.30 - 0.15 * H * 0.34)
        zz = [(P0[0] + (P1[0] - P0[0]) * i / 14.0, P0[1] + (P1[1] - P0[1]) * i / 14.0 + (22 * k if i % 2 else -22 * k) * (1 if 0 < i < 14 else 0)) for i in range(15)]
        if t < T_STOP:                                                               # 駆け抜ける
            a = ease2(u / (T_STOP - T_DASH)); f, d = s.clouds(t * 0.5, 0.7)
            L = s.L0(); n = max(2, int(a * 15)); cv2.polylines(L, [np.array(zz[:n], np.int32)], False, 1.0, int(7 * k), cv2.LINE_AA); s.add(f, L, HOT, 1.5, 4, 16)
            x = P0[0] + (P1[0] - P0[0]) * a; y = P0[1] + (P1[1] - P0[1]) * a
            for j in (3, 2, 1): s.hero(f, t, x - j * W * 0.10, y + j * H * 0.027, 0.36, 18, 0.25, 0.8)
            s.hero(f, t, x, y, 0.36, 18, 1.0, 0.3); return f, 5
        if t < T_TEAR:                                                               # 止め絵: 白地に 黒い一閃
            f = np.ones((H, W, 3), np.float32) * (YEL * 0.25 + 0.75)[None, None, :]
            L = s.L0(); cv2.polylines(L, [np.array(zz, np.int32)], False, 1.0, int(11 * k), cv2.LINE_AA)
            for i in range(30):
                q = s.rr[i]; x = q[0] * W; y = P0[1] + (P1[1] - P0[1]) * ((x - P0[0]) / (P1[0] - P0[0])); s.line(L, (x, y), (x + (q[1] - 0.5) * 220 * k, y + (q[2] - 0.5) * 300 * k), 1.0, 1)
            f *= (1 - np.clip(L, 0, 1))[:, :, None]
            return f, 0
        v = t - T_TEAR; fin = t - T_FINAL
        gap = H * 0.24 * ease(v / 1.1) + H * 0.02 * math.sin(v * 2)
        f = (AMBER * 0.5 + YEL * 0.5)[None, None, :] * (0.55 + 0.45 * np.exp(-(s.diag / (H * 0.25)) ** 2))[:, :, None] * np.ones((H, W, 1), np.float32)
        L = s.L0()
        for i in range(10):                                                          # すき間から さす 光のすじ
            x = W * (0.05 + i * 0.1); y = H * 0.64 - x / W * H * 0.34
            cv2.fillPoly(L, [np.array([(x - 8 * k, y), (x + 8 * k, y), (x + 60 * k + (i - 5) * 30 * k, H), (x - 60 * k + (i - 5) * 30 * k, H)], np.int32)], 0.25, cv2.LINE_AA)
        f += cv2.GaussianBlur(L, (0, 0), 6 * k)[:, :, None] * HOT[None, None, :]
        Lb = s.L0()
        for i in range(7):                                                           # すき間から ふる 雷の雨
            ph = (v * 3.1 + i * 0.37);
            if ph % 1.0 < 0.34:
                q = s.rr[(int(ph) * 7 + i * 13) % 120]; x = W * (0.08 + 0.84 * q[0]); y = H * 0.64 - x / W * H * 0.34
                s.bolt(Lb, (x, y), (x + (q[1] - 0.5) * 200 * k, H + 10), int(ph) * 101 + i * 17 + int((ph % 1.0) * 12), 0.24, 1.0, 3)
        C, d = s.clouds(t * 0.5, 0.75)
        dx = gap * 0.34; dy = gap
        up = (s.diag < 0).astype(np.float32); f2 = f.copy()
        for sgn, msk in ((-1, up), (1, 1 - up)):                                     # 雲の上半分と 下半分が ずれる
            M = np.float32([[1, 0, sgn * dx * 0.5], [0, 1, sgn * dy * 0.5]])
            cm = cv2.warpAffine(C * msk[:, :, None], M, (W, H)); am = cv2.warpAffine(msk, M, (W, H))
            f2 = f2 * (1 - am[:, :, None]) + cm
        f = f2; s.add(f, Lb, HOT, 1.4, 3, 14)
        L = s.L0()                                                                   # 裂け目の ふち
        for sgn in (-1, 1): cv2.polylines(L, [np.array([(x + sgn * dx * 0.5, y + sgn * dy * 0.5) for x, y in zz], np.int32)], False, 1.0, 3, cv2.LINE_AA)
        s.add(f, L, YEL, 1.1)
        land = v - 0.5; cy = H * 0.78
        if land > 0:                                                                 # 着地
            s.soft(f, (W / 2, cy), W * 0.3, 0.3, YEL)
            s.hero(f, t, W / 2, cy - H * 0.5 * (1 - ease(land / 0.25)), 0.40, 0, squash=0.12 * math.exp(-max(0, land - 0.25) * 9) * (land > 0.25))
            s.blade(f, 0.8, t); s.eyes(f, 0.6)
        if fin > 0:                                                                  # とどめの 大落雷
            w = W * (0.10 + 0.22 * math.exp(-fin * 2.5)) * (1 + 0.1 * math.sin(fin * 70))
            L = s.L0(); cv2.rectangle(L, (int(W / 2 - w / 2), 0), (int(W / 2 + w / 2), int(cy + H * 0.16)), 1.0, -1)
            for i in range(4): s.bolt(L, (W / 2 + (i - 1.5) * w * 0.5, -10), (W / 2 + (i - 1.5) * w * 0.9, cy + H * 0.16), int(fin * 15) * 13 + i, 0.3, 1.0, 3)
            f += cv2.GaussianBlur(L, (0, 0), 14 * k)[:, :, None] * HOT[None, None, :] * 1.3 * clamp(1.6 - fin * 0.9)
            if land > 0: s.hero(f, t, W / 2, cy, 0.40, 0, 1.0, 0.0)                  # 光の前に 黒く 立つ
            for i in range(3):
                q = (fin - i * 0.12) / 0.9
                if 0 < q < 1:
                    L2 = s.L0(); s.ell(L2, (W / 2, cy + H * 0.15), (W * 1.2 * q, W * 0.2 * q), 0, 1 - q, 4); s.add(f, L2, HOT, 1.0)
            if fin < 0.12: f += (1 - fin / 0.12) * 1.3
        if v < 0.12: f = f + (1 - v / 0.12) * 1.0
        end = DUR - t
        if end < 0.8: q = clamp((0.8 - end) / 0.35); f = f * (1 - q) + q
        if end < 0.45: f = np.ones_like(f) * clamp(end / 0.4) ** 2
        return f, 12 * math.exp(-v * 2.5) + (20 * math.exp(-fin * 3) if fin > 0 else 0)

    # ---- この子だけの曲: 太鼓の乱れ打ちと ひずんだ旋律(ホ短調の五音)。雷が落ちる間が だんだん詰まる ----
    def audio(s, path):
        a = Snd(DUR, 72); E = 164.81; pen = [0, 3, 5, 7, 10]
        def fq(d, o=0): return E * 2 ** ((pen[d % 5] + 12 * (d // 5 + o)) / 12.0)
        def thunder(t0, g=1.0): a.a += a.lp(a.nz, 0.06) * a.env(t0, 0.01, 0.9) * 1.6 * g + (a.nz - a.lp(a.nz, 0.4)) * a.env(t0, 0.001, 0.05) * 0.5 * g; a.boom(t0, 45, 0.9 * g, 0.5)
        def lead(d, t0, dur, g=0.07, o=1):
            a.note(fq(d, o), t0, dur, g, "saw", 0.005); a.note(fq(d, o) * 1.5, t0, dur * 0.7, g * 0.35, "sq", 0.005)
        a.a += a.lp(a.nz, 0.015) * 0.6 * a.gate(0, T_BOLT, 0.3)
        for x in (0.5, 1.0, 1.35): a.a += a.lp(a.nz, 0.04) * a.env(x, 0.02, 0.4) * 0.5
        thunder(T_BOLT, 1.3)
        a.boom(T_CUT, 50, 1.0)
        for i in range(5): a.a += (a.nz - a.lp(a.nz, 0.5)) * a.env(T_CUT + 0.45 + i * 0.16, 0.001, 0.04) * 0.35; lead([4, 3, 2, 3, 5][i], T_CUT + 0.45 + i * 0.16, 0.14, 0.08)
        bt = 60.0 / 172 / 2; tk = T_CUT + 1.3; i = 0
        riff = [0, 0, 3, 0, 2, 0, 4, 3]
        while tk < T_DASH:                                                           # 16分の 太鼓 と 低い刻み
            if i % 4 in (0, 3): a.boom(tk, 62, 0.4, 0.09)
            if i % 4 == 2: a.a += a.lp(a.nz, 0.5) * a.env(tk, 0.001, 0.05) * 0.25
            a.hat(tk, 0.04); a.note(fq(riff[i % 8], -1), tk, 0.09, 0.10, "saw", 0.003)
            if tk > T_CHARGE and i % 2 == 0: lead([5, 7, 5, 4, 5, 7, 8, 9][(i // 2) % 8], tk, 0.15, 0.055)
            tk += bt; i += 1
        for j, x in enumerate(STRIKES): thunder(x, 0.6 + j * 0.07); a.note(fq(5 + j, 1), x, 0.3, 0.07, "bell")
        a.rise(T_CHARGE + 3.15, T_DASH, 200, 1400, 0.16)
        a.whoosh(T_DASH - 0.1, T_STOP, 0.7, 0.4); a.note(fq(9, 2), T_STOP, 0.5, 0.12, "sine")                     # 止め絵は 高い一音だけ
        thunder(T_TEAR, 1.4)
        for d, o in ((0, -1), (2, 0), (4, 0), (5, 0)): a.note(fq(d, o), T_TEAR, 1.6, 0.05, "saw", 0.02)
        tk = T_TEAR; i = 0
        while tk < T_FINAL:
            if i % 2 == 0: a.boom(tk, 58, 0.4, 0.1)
            a.hat(tk, 0.05); lead([9, 8, 7, 5, 7, 8, 9, 10][i % 8], tk, 0.12, 0.06);
            if i % 3 == 0: a.a += a.lp(a.nz, 0.1) * a.env(tk, 0.005, 0.2) * 0.35
            tk += bt; i += 1
        thunder(T_FINAL, 1.7); thunder(T_FINAL + 0.25, 0.9)
        for d, o in ((0, -1), (0, 0), (2, 0), (4, 0), (5, 1)): a.note(fq(d, o), T_FINAL + 0.05, 1.5, 0.05, "saw", 0.02)
        a.save(path)


if __name__ == "__main__":
    r = Amatsu()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
