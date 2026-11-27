"""セイリュウ(★5 青竜の将) 専用の必殺技「蒼天雷鳴」— 決めは 画面の奥から こちらへ(Koshin 指定)
まっ青な 空と 石の柱の山。柱の上に 立つ → 石づきを 打ちつけると 刃の 竜の彫りが 光り、青い すじが 空の 奥の 1点へ のぼって 小さく消える →
偃月刀の 刃の形の 窓のカットイン → 奥の 1点から 青竜が らせんを えがいて こちらへ。どんどん 大きくなり、口を あけて 画面いっぱいに →
口が 画面を のみこみ、そのまま こちら(画面の先)へ つきぬけていく: まっ白 → 青い輪と いなずまが 通りすぎる → 目の前に 本人
使い方: python3 hs_67.py 出力.mp4 / python3 hs_67.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 8.9
AZ = np.array([0.30, 0.62, 1.0], np.float32); BOLT = np.array([0.78, 0.92, 1.0], np.float32); GOLD = np.array([1.0, 0.82, 0.32], np.float32)
T_TAP, T_CUT, T_COME, T_HIT, T_END = 1.8, 3.6, 5.5, 7.75, 8.6
BLADE = [(-0.25, -1.0), (0.45, -0.55), (0.62, 0.1), (0.50, 0.75), (0.15, 1.0), (-0.35, 1.0), (-0.20, 0.45), (-0.48, 0.30), (-0.30, 0.0), (-0.42, -0.5)]


class Seiryu(Kit):
    def __init__(s, W=540):
        super().__init__(67, (0.30, 0.62, 1.0), "蒼天雷鳴", "Azure Thunder", W)
        H, k = s.H, s.k
        s.eL = s.A(225, 365); s.eR = s.A(335, 355); s.fc = s.A(280, 350); s.btip = s.A(95, 32); s.bgd = s.A(95, 415); s.butt = s.A(95, 735)
        s.ht = [s.text_img(c, hz.SERIF, int(84 * k), 3) for c in "蒼天雷鳴"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * k), 1) for c in "奥義"]
        s.HF = 0.40; s.CX = W / 2; s.flr = H * 0.82; s.CY = s.flr - s.HF * H / 2; s.VP = (W / 2, H * 0.30)
        rg = np.random.RandomState(67); s.yn = s.YY / H
        s.vr = np.hypot(s.XX - s.VP[0], s.YY - s.VP[1]) / (H * 0.75)
        far = s.L0(); near = s.L0()                                                   # 石の柱の山(遠・近)
        for m, n0, hmin, hmax, wmin, wmax in ((far, 11, 0.10, 0.30, 16, 40), (near, 7, 0.06, 0.20, 34, 80)):
            for i in range(n0):
                x = W * (i + rg.rand() * 0.8) / n0; w = (wmin + (wmax - wmin) * rg.rand()) * k; h = H * (hmin + (hmax - hmin) * rg.rand())
                if m is near and abs(x - W / 2) < 90 * k: continue
                P = [(x - w / 2, H), (x - w * 0.42, H - h * 0.5), (x - w * 0.34, H - h), (x - w * 0.05, H - h - w * 0.25), (x + w * 0.3, H - h + w * 0.1), (x + w * 0.44, H - h * 0.6), (x + w / 2, H)]
                cv2.fillPoly(m, [np.array(P, np.int32)], 1.0, cv2.LINE_AA)
        w = 150 * k; cv2.fillPoly(near, [np.array([(W / 2 - w * 0.62, H), (W / 2 - w * 0.5, s.flr + 30 * k), (W / 2 - w * 0.42, s.flr), (W / 2 + w * 0.40, s.flr - 6 * k), (W / 2 + w * 0.52, s.flr + 40 * k), (W / 2 + w * 0.6, H)], np.int32)], 1.0, cv2.LINE_AA)
        s.far = far; s.near = near
        s.sl = rg.rand(80, 3).astype(np.float32); s.sp = rg.rand(70, 4).astype(np.float32)
        s.cr = []                                                                    # いなずまの ひび(まん中→ふち)
        for j in range(11):
            a = j * 6.283 / 11 + (rg.rand() - 0.5) * 0.4; p = np.array([W / 2, H * 0.42], np.float32); pts = [p.copy()]; br = []
            while 0 < p[0] < W and 0 < p[1] < H:
                a2 = a + (rg.rand() - 0.5) * 1.1; st = (30 + 50 * rg.rand()) * k; p = p + np.array([math.cos(a2), math.sin(a2)], np.float32) * st; pts.append(p.copy())
                if rg.rand() < 0.4:
                    q = p.copy(); bb = [q.copy()]; a3 = a + (1 if rg.rand() < 0.5 else -1) * (0.6 + 0.5 * rg.rand())
                    for _ in range(3): q = q + np.array([math.cos(a3 + (rg.rand() - 0.5) * 0.9), math.sin(a3 + (rg.rand() - 0.5) * 0.9)], np.float32) * (22 + 30 * rg.rand()) * k; bb.append(q.copy())
                    br.append((len(pts) - 1, bb))
            s.cr.append((pts, br))

    # ---- 大道具 ----
    def sky(s, t, deep=0.0, haze=1.0):
        top = np.array([0.06, 0.30, 0.86], np.float32) * (1 - deep) + np.array([0.01, 0.06, 0.30], np.float32) * deep
        low = np.array([0.62, 0.86, 1.0], np.float32) * (1 - deep) + np.array([0.10, 0.30, 0.70], np.float32) * deep
        f = (top[None, None, :] * (1 - s.yn)[:, :, None] + low[None, None, :] * s.yn[:, :, None]).astype(np.float32)
        if deep > 0: f *= (1 - 0.45 * deep * np.clip(s.vr, 0, 1))[:, :, None]
        return f

    def peaks(s, f, deep=0.0):
        c1 = np.array([0.46, 0.66, 0.90], np.float32) * (1 - 0.6 * deep); c2 = np.array([0.14, 0.26, 0.46], np.float32) * (1 - 0.5 * deep)
        f = f * (1 - s.far[:, :, None]) + c1[None, None, :] * s.far[:, :, None]
        return f * (1 - s.near[:, :, None]) + (c2[None, None, :] * (0.7 + 0.5 * (1 - s.yn))[:, :, None]) * s.near[:, :, None]

    def rush(s, f, t, v=1.0, speed=1.0):
        """奥の 1点から こちらへ 流れる すじ"""
        L = s.L0(); VP = s.VP
        for r in s.sl:
            a = r[0] * 6.283; q = ((r[1] + t * speed * (0.6 + r[2])) % 1.0) ** 2; r0 = q * s.H * 0.9 + 12 * s.k; r1 = r0 * (1.12 + 0.25 * q)
            s.line(L, (VP[0] + math.cos(a) * r0, VP[1] + math.sin(a) * r0), (VP[0] + math.cos(a) * r1, VP[1] + math.sin(a) * r1), float(v * (0.2 + 0.8 * q)), 1 if q < 0.5 else 2)
        f += cv2.GaussianBlur(L, (0, 0), 1.0 * s.k)[:, :, None] * BOLT[None, None, :] * 0.6

    def hero(s, f, t, hf=None, cy=None, **kw):
        hf = s.HF if hf is None else hf; img = s.pose(t, **kw); s.draw(f, img, s.CX, s.CY if cy is None else cy, hf)

    def carve(s, f, v):
        """刃の 竜の彫りが 光る"""
        if v <= 0.01: return
        a = s.at(*s.bgd); b = s.at(*s.btip); L = s.L0(); n = 9; pts = []
        for j in range(n + 1): u = j / float(n); pts.append((a[0] + (b[0] - a[0]) * u + math.sin(u * 9.4) * 7 * s._pl[2] * 2, a[1] + (b[1] - a[1]) * u))
        cv2.polylines(L, [np.array(pts, np.int32)], False, float(min(1.0, v)), max(1, int(3 * s.k)), cv2.LINE_AA); s.add(f, L, BOLT, 0.9 * min(1.4, v), 2, 8)

    def dragon(s, f, t, zh, op=0.0, sky_c=(0.03, 0.12, 0.45)):
        """奥から こちらへ 来る 青竜。zh: 頭の おくゆき(小さいほど 近い)。op: 口の ひらき"""
        W, H, k = s.W, s.H, s.k; VP = s.VP; N = 30; segs = []
        for i in range(N):
            z = zh + i * 0.42; th = t * 2.2 + z * 1.15; Rw = W * 0.62 * min(1.0, 0.12 + i / 7.0)
            c = (VP[0] + math.cos(th) * Rw / z, VP[1] + math.sin(th) * Rw / z * 0.8); r = 92 * k / z * (1.0 if i < 20 else (N - i) / 10.0)
            segs.append((c, r, z, th))
        for i in range(N - 1, 0, -1):
            c, r, z, th = segs[i]
            if r < 1.2: continue
            fog = clamp((z - 2.0) / 9.0); col = np.array([0.08, 0.36, 0.98], np.float32) * (1 - fog) + np.array(sky_c, np.float32) * fog; bel = GOLD * (1 - fog) + np.array(sky_c, np.float32) * fog
            x, y = int(c[0]), int(c[1])
            if i % 2 == 0 and r > 4: cv2.fillPoly(f, [np.array([(x - r * 0.4, y - r * 0.8), (x, y - r * 1.55), (x + r * 0.4, y - r * 0.8)], np.int32)], (float(col[0] * 0.5), float(col[1] * 0.5), float(col[2] * 0.6)), cv2.LINE_AA)
            cv2.circle(f, (x, y), int(r + max(1, r * 0.08)), (0.0, 0.03, 0.12), -1, cv2.LINE_AA); cv2.circle(f, (x, y), int(r), (float(col[0]), float(col[1]), float(col[2])), -1, cv2.LINE_AA)
            cv2.ellipse(f, (x, int(y + r * 0.45)), (int(r * 0.62), int(r * 0.42)), 0, 0, 180, (float(bel[0]), float(bel[1]), float(bel[2])), -1, cv2.LINE_AA)
            if r > 6: cv2.ellipse(f, (x, y), (int(r * 0.7), int(r * 0.7)), 0, 200, 290, (0.45, 0.72, 1.0), max(1, int(r * 0.12)), cv2.LINE_AA)
        L = s.L0()                                                                   # 体を はしる 電気
        for j in range(5):
            i = 1 + (int(t * 30) * 7 + j * 5) % (N - 4); a, b = segs[i], segs[i + 2]
            if a[1] < 3: continue
            mid = ((a[0][0] + b[0][0]) / 2 + math.sin(t * 90 + j) * a[1], (a[0][1] + b[0][1]) / 2 + math.cos(t * 70 + j) * a[1]); s.line(L, a[0], mid, 1.0, 2); s.line(L, mid, b[0], 1.0, 2)
        s.add(f, L, BOLT, 0.9, 1, 6)
        c, r, z, th = segs[0]; r *= 1.55; x, y = c                                    # 頭(正面)
        for sg in (-1, 1):                                                           # ひげ・角
            w = [(x + sg * r * 0.5, y + r * 0.35), (x + sg * r * 1.5, y + r * 0.1 + math.sin(t * 9 + sg) * r * 0.2), (x + sg * r * 2.1, y + r * 0.6 + math.cos(t * 7) * r * 0.2)]
            cv2.polylines(f, [np.array(w, np.int32)], False, (1.0, 0.86, 0.40), max(1, int(r * 0.07)), cv2.LINE_AA)
            hb = (x + sg * r * 0.45, y - r * 0.75); ht = (x + sg * r * 0.95, y - r * 1.85)
            cv2.line(f, (int(hb[0]), int(hb[1])), (int(ht[0]), int(ht[1])), (0.0, 0.03, 0.12), max(2, int(r * 0.26)), cv2.LINE_AA); cv2.line(f, (int(hb[0]), int(hb[1])), (int(ht[0]), int(ht[1])), (1.0, 0.82, 0.32), max(1, int(r * 0.16)), cv2.LINE_AA)
            cv2.line(f, (int(hb[0] + (ht[0] - hb[0]) * 0.55), int(hb[1] + (ht[1] - hb[1]) * 0.55)), (int(x + sg * r * 1.45), int(y - r * 1.25)), (1.0, 0.82, 0.32), max(1, int(r * 0.12)), cv2.LINE_AA)
        cv2.circle(f, (int(x), int(y)), int(r * 1.06), (0.0, 0.03, 0.12), -1, cv2.LINE_AA); cv2.circle(f, (int(x), int(y)), int(r), (0.08, 0.36, 0.98), -1, cv2.LINE_AA)
        for j in range(7): a = 3.6 + j * 0.37; cv2.fillPoly(f, [np.array([(x + math.cos(a - 0.14) * r * 0.95, y + math.sin(a - 0.14) * r * 0.95), (x + math.cos(a) * r * 1.4, y + math.sin(a) * r * 1.4), (x + math.cos(a + 0.14) * r * 0.95, y + math.sin(a + 0.14) * r * 0.95)], np.int32)], (0.04, 0.20, 0.70), cv2.LINE_AA)   # たてがみ
        my = y + r * 0.34; mw = r * (0.52 + 0.2 * op); mh = r * (0.10 + 0.62 * op)
        cv2.ellipse(f, (int(x), int(y + r * 0.22)), (int(r * 0.74), int(r * 0.5 + mh * 0.5)), 0, 0, 360, (0.22, 0.52, 1.0), -1, cv2.LINE_AA)      # 鼻づら
        cv2.ellipse(f, (int(x), int(my + mh * 0.3)), (int(mw), int(mh)), 0, 0, 360, (0.45, 0.02, 0.08), -1, cv2.LINE_AA)
        if op > 0.3:
            s.soft(f, (x, my + mh * 0.3), mw * 0.9, 0.9 * (op - 0.3) / 0.7, BOLT); cv2.circle(f, (int(x), int(my + mh * 0.3)), int(mw * 0.45 * (op - 0.3) / 0.7), (1.0, 1.0, 1.0), -1, cv2.LINE_AA)
        for sg in (-1, 1):
            cv2.fillPoly(f, [np.array([(x + sg * mw * 0.75, my + mh * 0.3 - mh * 0.75), (x + sg * mw * 0.45, my + mh * 0.3 - mh * 0.85), (x + sg * mw * 0.58, my + mh * 0.3 - mh * 0.85 + r * 0.26)], np.int32)], (1.0, 1.0, 0.94), cv2.LINE_AA)
            e = (x + sg * r * 0.42, y - r * 0.28); cv2.ellipse(f, (int(e[0]), int(e[1])), (int(r * 0.24), int(r * 0.13)), sg * 22, 0, 360, (1.0, 0.92, 0.30), -1, cv2.LINE_AA); cv2.ellipse(f, (int(e[0]), int(e[1])), (max(1, int(r * 0.05)), int(r * 0.12)), 0, 0, 360, (0.0, 0.0, 0.0), -1, cv2.LINE_AA)
            cv2.line(f, (int(x + sg * r * 0.15), int(y - r * 0.5)), (int(x + sg * r * 0.72), int(y - r * 0.55)), (0.02, 0.12, 0.5), max(1, int(r * 0.09)), cv2.LINE_AA)
            cv2.circle(f, (int(x + sg * r * 0.16), int(y + r * 0.02)), max(1, int(r * 0.05)), (0.02, 0.10, 0.40), -1, cv2.LINE_AA)
        return (x, my + mh * 0.3), mw

    def forks(s, f, t, v, n=3):
        """いなずまの 枝(コマごとに ちがう 数本だけ ちらつく)"""
        if v <= 0.01: return
        L = s.L0()
        for j in range(n):
            pts, br = s.cr[(int(t * 30) * 5 + j * 4) % len(s.cr)]; cv2.polylines(L, [np.array(pts, np.int32)], False, 1.0, max(1, int(4 * s.k)), cv2.LINE_AA)
            for at_i, bb in br: cv2.polylines(L, [np.array(bb, np.int32)], False, 0.8, max(1, int(2 * s.k)), cv2.LINE_AA)
        s.add(f, L, BOLT, v, 2, 10)

    def tunnel(s, f, t, h, v=1.0):
        """竜が こちらを つきぬけていく: 青い輪が 次々 手前へ 広がって 通りすぎる"""
        C = (s.W / 2, s.H * 0.42)
        for j in range(9):
            q = (h * 1.7 + j / 9.0) % 1.0; r = q * q * s.H * 1.15 + 6 * s.k; L = s.L0(); s.circ(L, C, r, 1.0, max(2, int((3 + 34 * q * q) * s.k)))
            f += (cv2.GaussianBlur(L, (0, 0), (1 + 5 * q) * s.k) * (1 - q) * 1.2 * v)[:, :, None] * (AZ * (0.6 + 0.4 * (j % 2)) + (0.25 if j % 3 == 0 else 0.0) * GOLD)[None, None, :]

    def frame(s, t):
        if t < T_CUT: f, sh = s.sc_call(t)
        elif t < T_COME: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_END: f, sh = s.sc_come(t)
        else: f, sh = s.sc_end(t - T_END)
        f, sh = s.finale(f, sh, t, DUR, s.btip, (0.3, 0.62, 1), (1, 0.82, 0.32), -1)
        return s.post(f, BOLT, 0.30, sh * s.k, t, 0.95)

    # 1) 柱の上 → 石づきを 打つ → すじが 奥へ
    def sc_call(s, t):
        W, H, k = s.W, s.H, s.k; tp = t - T_TAP; deep = 0.8 * ease2((tp - 0.2) / 1.2) if tp > 0.2 else 0.0
        f = s.sky(t, deep); f = s.peaks(f, deep)
        if deep > 0.05: s.rush(f, t, deep * 0.5, 0.35)
        s.hero(f, t, squash=0.07 * math.exp(-tp * 9) * math.cos(tp * 18) if tp > 0 else 0)
        gl = (t - 0.7) / 0.6                                                         # 刃を 光が のぼる
        if 0 < gl < 1: a = s.at(*s.bgd); b = s.at(*s.btip); L = s.L0(); s.star4(L, (a[0] + (b[0] - a[0]) * gl, a[1] + (b[1] - a[1]) * gl), 40 * k * math.sin(gl * 3.1416) + 1, 1.0, 0.3, 0.1); s.add(f, L, WHITE, 0.8, 2, 6)
        sh = 0
        if tp > 0:
            s.carve(f, 0.5 + 0.9 * math.exp(-tp * 3))
            p = s.at(*s.butt); L = s.L0()
            for j in range(2):
                v = (tp - j * 0.08) / 0.45
                if 0 < v < 1: s.ell(L, p, (160 * k * ease(v), 34 * k * ease(v)), 0, 1 - v, 3)
            b = s.at(*s.btip); g = ease((tp - 0.25) / 0.35)                           # 刃先から 奥の 1点へ
            if g > 0:
                h0 = max(0.0, g - 0.5 - (tp - 0.6)); s.line(L, (b[0] + (s.VP[0] - b[0]) * h0, b[1] + (s.VP[1] - b[1]) * h0), (b[0] + (s.VP[0] - b[0]) * g, b[1] + (s.VP[1] - b[1]) * g), 1.0, max(1, int(4 * k * (1 - 0.7 * g))))
            s.add(f, L, BOLT, 0.9, 2, 8)
            tw = tp - 0.6
            if tw > 0: Ls = s.L0(); s.star4(Ls, s.VP, (26 * math.exp(-tw * 3) + 8 + 3 * math.sin(t * 20)) * k, 1.0, t, 0.12); s.add(f, Ls, BOLT, 0.9, 1, 5)
            sh = 8 * math.exp(-tp * 9)
        return f * clamp(t / 0.3), sh

    # 2) 偃月刀の 刃の形の 窓
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.sky(t, 1.0) * 0.7; s.rush(f, t, 0.5, 0.5)
        C = (W / 2 + 10 * k, H * 0.375); g = back(u / 0.3, 1.2); sx, sy = W * 0.66, H * 0.31 * g
        P = [(C[0] + x * sx, C[1] + y * sy) for x, y in BLADE]
        m = s.L0(); cv2.fillPoly(m, [np.array(P, np.int32)], 1.0, cv2.LINE_AA)
        win = np.zeros((H, W, 3), np.float32) + np.array([0.10, 0.30, 0.78], np.float32) * (1.2 - 0.6 * s.yn)[:, :, None]
        zf = 0.92 + 0.05 * ease(u / 1.8); sc = zf * H / s.ah; img = s.pose(t)
        s.draw(win, img, C[0] + W * 0.03 - (s.fc[0] - s.aw / 2) * sc, C[1] - (s.fc[1] - s.ah / 2) * sc, zf)
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        cv2.polylines(f, [np.array(P, np.int32)], True, (0.0, 0.03, 0.12), max(1, int(14 * k)), cv2.LINE_AA); cv2.polylines(f, [np.array(P, np.int32)], True, (1.0, 0.84, 0.36), max(1, int(6 * k)), cv2.LINE_AA)
        cv2.polylines(f, [np.array(P[:5], np.int32)], False, (0.94, 0.97, 1.0), max(1, int(6 * k)), cv2.LINE_AA)                  # 刃の側は 白
        z = (u - 0.5) / 0.25                                                         # 刃に 電気が はしる
        if 0 < z < 1:
            L = s.L0(); pts = [(P[0][0] + (P[4][0] - P[0][0]) * j / 8.0 + (30 * k * math.sin(j * 2.4 + 1) if 0 < j < 8 else 0) + W * 0.30 * math.sin(j / 8.0 * 3.1416), P[0][1] + (P[4][1] - P[0][1]) * j / 8.0) for j in range(int(9 * z) + 1)]
            if len(pts) > 1: cv2.polylines(L, [np.array(pts, np.int32)], False, 1.0, max(1, int(3 * k)), cv2.LINE_AA); s.add(f, L, BOLT, 1.0, 2, 8)
        n = len(s.ht); step = W * 0.20
        for i, T in enumerate(s.ht):                                                 # 技名: 奥から こちらへ(小さく→大きく は 使えないので 光って 現れる)
            a = (u - 0.6 - i * 0.12) / 0.15
            if a > 0:
                s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.835, clamp(a * 2), 1.0)
                if a < 1.5: s.soft(f, (W / 2 + (i - (n - 1) / 2) * step, H * 0.835), 60 * k, 0.7 * (1 - a / 1.5), BOLT)
        for i, T in enumerate(s.vo):
            if u > 0.45 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.758, 1.0, 1.0)
        if u > 1.35: s.text(f, s.txt_en, W / 2, H * 0.835 + 62 * k, ease((u - 1.35) / 0.3), 1.0)
        if u < 0.08: f = f + (1 - u / 0.08) * 0.7
        return f, 0

    # 3) 奥から 青竜が こちらへ → 画面に ぶつかる → ひび
    def sc_come(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_COME; h = t - T_HIT
        if h < 0:
            g = clamp(u / (T_HIT - T_COME)); zh = 13 * (1 - g) ** 2.2 + 0.62
            f = s.sky(t, 1.0); s.rush(f, t, 0.5 + 0.5 * g, 0.5 + 1.2 * g)
            if g < 0.75:
                f = s.peaks(f, 1.0); s.hero(f, t); s.carve(f, 0.8)
            op = ease2((g - 0.55) / 0.4); mc, mw = s.dragon(f, t, zh, op)
            pre = (g - 0.93) / 0.07                                                  # 口の 光が 画面を うめる
            if pre > 0: s.soft(f, mc, mw * (1 + 5 * pre), 1.2 * pre, BOLT); f = f + ease2(pre) * 0.9
            if u < 0.1: f = f * (u / 0.1) + (1 - u / 0.1) * 0.5
            return f, 1.5 + 9 * g ** 3
        f = s.sky(t, 1.0) * (0.35 + 0.3 * clamp(h / 0.8))                              # ぶつかった あと
        L = s.L0()
        for r in s.sp:                                                               # 火花が こちらへ とびちる(大きくなりながら 外へ)
            a = r[0] * 6.283; d = (0.1 + r[1]) * H * 0.9 * (1 - math.exp(-h * 3.0)); p = (W / 2 + math.cos(a) * d, H * 0.42 + math.sin(a) * d)
            cv2.circle(L, (int(p[0]), int(p[1])), max(1, int((2 + 14 * r[2] * (1 - math.exp(-h * 3))) * k)), float(clamp(1.3 - h * 1.5)), -1, cv2.LINE_AA)
        f += (L + cv2.GaussianBlur(L, (0, 0), 5 * k))[:, :, None] * BOLT[None, None, :]
        s.tunnel(f, t, h, clamp(1.4 - h * 1.2)); s.forks(f, t, 1.1 * clamp(1.3 - h * 1.3))
        if h < 0.22: f = f + (1 - h / 0.22) ** 2 * 1.3
        return f, 24 * math.exp(-h * 4.5)

    # 4) ひびが 引くと、目の前に 本人
    def sc_end(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_END
        f = s.sky(t, 0.25 * (1 - clamp(u / 1.0))); f = s.peaks(f, 0.0)
        hf = 0.50 + 0.07 * back(u / 0.35, 1.6)                                         # ぐっと こちらへ 寄る
        s.hero(f, t, hf, H * 0.985 - hf * H / 2); s.carve(f, 0.4 + 0.8 * math.exp(-u * 2))
        if u < 0.5:
            L = s.L0()
            for r in s.sp[:24]: a = r[0] * 6.283; d = H * (0.5 + 0.6 * r[1]) * (0.7 + u); cv2.circle(L, (int(W / 2 + math.cos(a) * d), int(H * 0.42 + math.sin(a) * d)), max(1, int(8 * r[2] * k)), float(1 - u / 0.5), -1, cv2.LINE_AA)
            f += cv2.GaussianBlur(L, (0, 0), 3 * k)[:, :, None] * BOLT[None, None, :]
        if u < 0.12: f = f + (1 - u / 0.12) * 0.5
        end = DUR - t
        if end < 0.45: v = clamp((0.45 - end) / 0.4); f = f * (1 - v) + v
        return f, 0

    # ---- この子だけの曲: 五音(ニ・ホ・嬰ヘ・イ・ロ)の 大太鼓と どら、二胡ふうの すべる ふし。奥から 来る間は うなりが 上がり 太鼓が つまる。ぶつかりは かみなり(ばりっ → ごろごろ)と どら。さいごは 二胡の ひと節 ----
    def audio(s, path):
        a = Snd(DUR, 67); D = 146.83; pen = [0, 2, 4, 7, 9]
        def fq(d, o=0): return D * 2 ** ((pen[d % 5] + 12 * (d // 5 + o)) / 12.0)
        def drum(t0, g=0.9, f0=52): a.boom(t0, f0, g, 0.3); a.hat(t0, g * 0.08, 0.01)
        def gong(t0, g=0.10):
            for fr, gg in ((73.4, 1.0), (110.5, 0.7), (187, 0.5), (251, 0.4), (392, 0.25), (577, 0.15)): a.note(fr, t0, 2.0, g * gg, "sine", 0.003)
            a.a += a.lp(a.nz, 0.2) * a.env(t0, 0.01, 0.3) * g * 2
        def erhu(d, t0, ln, g=0.07, o=1, sl=0.0):
            u = np.clip((a.t - t0) / ln, 0, 1); fr = fq(d, o) * (1 + sl * (1 - np.minimum(1, u * 5))) * (1 + 0.012 * np.sin(2 * np.pi * 6 * a.t) * np.minimum(1, u * 3))
            ph = 2 * np.pi * np.cumsum(fr) / a.sr; w = sum(np.sin(m * ph) / m for m in range(1, 8)); e = np.clip(u * 12, 0, 1) * np.clip((1 - u) * 6, 0, 1) * ((a.t >= t0) & (a.t < t0 + ln))
            a.a += a.lp((w * e).astype(np.float32), 0.25) * g
        def zheng(d, t0, g=0.08, o=1): a.note(fq(d, o), t0, 0.35, g, "bell", 0.001); a.note(fq(d, o), t0, 0.03, g * 0.5, "saw", 0.001)
        def thunder(t0, g=1.0):
            a.a += (a.nz - a.lp(a.nz, 0.5)) * a.env(t0, 0.001, 0.07) * 0.8 * g + a.lp(a.nz, 0.05) * a.env(t0 + 0.03, 0.03, 0.9) * (1 + 0.5 * np.sin(a.t * 23)) * 1.6 * g; a.boom(t0, 36, 1.8 * g, 0.9)
        a.a += a.lp(a.nz, 0.03) * 0.35 * a.gate(0, T_TAP, 0.3)                                                                   # 高い所の 風
        erhu(2, 0.3, 0.6, 0.05, 1, -0.06); erhu(4, 0.95, 0.7, 0.05, 1, -0.04); a.note(3200, 0.7, 0.5, 0.03, "sine", 0.2)
        drum(T_TAP, 1.2); gong(T_TAP, 0.07); a.rise(T_TAP + 0.25, T_TAP + 0.6, 500, 3000, 0.10); a.note(4200, T_TAP + 0.6, 0.6, 0.04, "sine")
        a.a += a.tone(55) * 0.07 * a.gate(T_TAP + 0.3, T_CUT, 0.3)
        drum(T_CUT, 1.1); gong(T_CUT, 0.05)
        for i, (d, dt) in enumerate(((5, 0.2), (7, 0.4), (9, 0.6), (7, 0.9), (5, 1.1), (4, 1.3), (5, 1.5))): erhu(d, T_CUT + dt, 0.22 if i != 2 else 0.3, 0.08, 0, -0.04)
        for i in range(10): zheng(i + 2, T_CUT + 0.5 + i * 0.03, 0.05)
        for dt in (0.6, 0.9, 1.2, 1.35, 1.5): drum(T_CUT + dt, 0.6, 62)
        a.a += a.lp(a.nz, 0.12) * np.clip((a.t - T_COME) / (T_HIT - T_COME), 0, 1) ** 2 * a.gate(T_COME, T_HIT, 0.02) * 1.0        # こちらへ 来る: うなりが 上がる
        a.rise(T_COME, T_HIT, 60, 420, 0.16); a.rise(T_COME + 1.0, T_HIT, 200, 1500, 0.08)
        tk = T_COME + 0.1; i = 0
        while tk < T_HIT - 0.04: drum(tk, 0.5 + 0.5 * (tk - T_COME) / 2.2, 58 if i % 2 else 48); tk += max(0.05, 0.34 - i * 0.022); i += 1
        for i in range(6): erhu(5 + i, T_COME + 0.3 + i * 0.3, 0.26, 0.05, 0, 0.05)
        thunder(T_HIT, 1.2); gong(T_HIT + 0.02, 0.12)
        for i in range(16): a.note(2000 + (i * 733) % 4000, T_HIT + 0.02 + i * 0.03, 0.03, 0.04, "sq", 0.0005)                    # ひびの ぱりぱり
        thunder(T_END + 0.05, 0.35)
        for d, dt, ln in ((9, 0.5, 0.35), (7, 0.9, 0.25), (5, 1.2, 0.9)): erhu(d, T_END + dt, ln, 0.07, 0, -0.03)
        drum(T_END + 1.2, 0.7); gong(T_END + 1.2, 0.06); zheng(5, T_END + 1.2, 0.07, 0); zheng(7, T_END + 1.25, 0.06, 0); zheng(10, T_END + 1.3, 0.06, 0)
        s.fin_snd(a, DUR, 67); a.save(path)


if __name__ == "__main__":
    r = Seiryu()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
