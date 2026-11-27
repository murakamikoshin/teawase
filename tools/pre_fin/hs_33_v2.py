"""ジークフリート(★5 竜鱗の騎士) 専用の必殺技「竜鱗ノ剣」
溶岩の洞くつ。うしろの闇で 大きな竜の目が ふたつ ひらく → 竜が 火をはく。ふりむかずに 立ったまま、うろこ模様の 見えない盾で 火が 左右に われる →
竜の ひとみ(たての細い瞳孔)の中に うつる 騎士のカットイン → まっ暗に ななめの一閃 → しずかな間 → 竜が 線にそって ずれる →
竜が うろこに ほどけて 剣に はりつき、うろこの波が 広がる
使い方: python3 hs_33.py 出力.mp4 / python3 hs_33.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 11.5
GREEN = np.array([0.30, 1.0, 0.50], np.float32); AMBER = np.array([1.0, 0.70, 0.12], np.float32); LAVA = np.array([1.0, 0.36, 0.06], np.float32); YEL = np.array([1.0, 0.92, 0.30], np.float32)
T_FIRE, T_CUT, T_SL, T_BLK, T_AFT, T_SPLIT, T_SC, T_WAVE = 2.3, 4.5, 6.4, 6.75, 7.05, 7.65, 8.5, 9.6


class Sieg(Kit):
    def __init__(s, W=540):
        super().__init__(33, (0.30, 1.0, 0.50), "竜鱗ノ剣", "Dragonscale Blade", W)
        H, k = s.H, s.k
        s.eL = s.A(345, 205); s.eR = s.A(462, 205); s.tip = s.A(100, 195); s.grd = s.A(372, 380); s.fc = s.A(403, 215)
        s.ht = [s.text_img(c, hz.SERIF, int(84 * k), 3) for c in "竜鱗ノ剣"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * k), 1) for c in "奥義"]
        rg = np.random.RandomState(33); s.n1 = rg.rand(300, 34).astype(np.float32); s.n2 = rg.rand(200, 12).astype(np.float32); s.em = rg.rand(46, 4).astype(np.float32)
        s.yn = s.YY / H; s.flr = H * 0.86; s.CX = W / 2; s.CY = s.flr - H * 0.205
        # うろこ模様(画面いっぱい)
        L = s.L0(); R = 24 * k; j = 0; y = 0.0
        while y < H + R:
            x = -R + (R * 0.92 if j % 2 else 0)
            while x < W + R: cv2.ellipse(L, (int(x), int(y)), (int(R), int(R)), 0, 0, 180, 1.0, max(1, int(2 * k)), cv2.LINE_AA); x += R * 1.84
            y += R * 0.62; j += 1
        s.scl = L
        # 竜の頭(正面)
        hp = [(0, 0.045), (-0.28, 0.03), (-0.50, 0.10), (-0.97, 0.012), (-0.74, 0.17), (-0.90, 0.25), (-0.66, 0.29), (-0.50, 0.38), (-0.34, 0.45), (-0.20, 0.49), (0, 0.50)]
        pts = [(W / 2 + x * W / 2, y * H) for x, y in hp] + [(W / 2 - x * W / 2, y * H) for x, y in hp[-2:0:-1]]
        m = s.L0(); cv2.fillPoly(m, [np.array(pts, np.int32)], 1.0, cv2.LINE_AA)
        for sx in (-1, 1):                                                           # 大きく そった角 と ほおの とげ
            hn = [(0.30, 0.06), (0.50, -0.01), (0.78, -0.03), (0.99, 0.03), (0.80, 0.02), (0.62, 0.06), (0.52, 0.13)]
            cv2.fillPoly(m, [np.array([(W / 2 + sx * x * W / 2, (y + 0.035) * H) for x, y in hn], np.int32)], 1.0, cv2.LINE_AA)
            for x0, y0, x1, y1 in ((0.80, 0.26, 1.0, 0.30), (0.66, 0.31, 0.86, 0.37), (0.54, 0.37, 0.70, 0.43)):
                cv2.fillPoly(m, [np.array([(W / 2 + sx * (x0 - 0.08) * W / 2, (y0 - 0.03) * H), (W / 2 + sx * x1 * W / 2, y1 * H), (W / 2 + sx * (x0 - 0.10) * W / 2, (y0 + 0.03) * H)], np.int32)], 1.0, cv2.LINE_AA)
        for i in range(-3, 4):                                                       # 頭の てっぺんの とげ
            x = W / 2 + i * W * 0.045; cv2.fillPoly(m, [np.array([(x - W * 0.02, H * 0.06), (x, H * (0.012 + 0.006 * abs(i))), (x + W * 0.02, H * 0.06)], np.int32)], 1.0, cv2.LINE_AA)
        s.dm = cv2.GaussianBlur(m, (0, 0), 1.2 * k)
        s.EY = [(W * 0.29, H * 0.205, 16), (W * 0.71, H * 0.205, -16)]
        # 一閃の線: 剣の向き(右下→左上)
        d = np.array([s.tip[0] - s.grd[0], s.tip[1] - s.grd[1]], np.float32); d /= np.linalg.norm(d); s.dir = d; s.nrm = np.array([-d[1], d[0]], np.float32)
        s.P0 = np.array([W * 0.5, H * 0.25], np.float32)
        s.side = np.where((s.XX - s.P0[0]) * s.nrm[0] + (s.YY - s.P0[1]) * s.nrm[1] > 0, 1.0, 0.0).astype(np.float32)
        # ひとみの まわりの 虹彩
        ang = np.arctan2(s.YY - H * 0.38, s.XX - W / 2); rad = np.sqrt((s.XX - W / 2) ** 2 + (s.YY - H * 0.38) ** 2) / (W * 0.62)
        st = rg.rand(360).astype(np.float32); st = np.convolve(np.concatenate([st, st[:4]]), np.ones(3) / 3, "same")[:360]
        v = st[((ang + math.pi) / (2 * math.pi) * 359).astype(np.int32)]
        ir = (0.35 + 0.65 * v) * np.clip(1.25 - rad, 0, 1) * np.clip((1 - rad) / 0.06, 0, 1) * (0.55 + 0.45 * np.clip(rad * 1.3, 0, 1))
        s.iris = cv2.GaussianBlur(ir.astype(np.float32), (0, 0), 1.0 * k)[:, :, None] * AMBER[None, None, :]
        # うろこの つぶ: 竜の中から 生まれる
        P = []
        while len(P) < 150:
            x, y = rg.rand() * W, rg.rand() * H * 0.52
            if s.dm[int(y), int(x)] > 0.5: P.append((x, y, rg.rand(), rg.rand(), rg.rand()))
        s.sp = np.array(P, np.float32)
        # 岩: まん中の 足場と 両わきの 柱
        m = s.L0(); f0 = s.flr
        cv2.fillPoly(m, [np.array([(W * 0.08, f0), (W * 0.92, f0), (W * 0.80, H), (W * 0.20, H)], np.int32)], 1.0, cv2.LINE_AA)
        cv2.fillPoly(m, [np.array([(0, H * 0.55), (W * 0.07, H * 0.60), (W * 0.04, H * 0.80), (W * 0.10, H), (0, H)], np.int32)], 1.0, cv2.LINE_AA)
        cv2.fillPoly(m, [np.array([(W, H * 0.50), (W * 0.92, H * 0.58), (W * 0.96, H * 0.82), (W * 0.89, H), (W, H)], np.int32)], 1.0, cv2.LINE_AA)
        s.rock = m

    # ---- 大道具 ----
    def noise(s, n, off, rows):
        cell = s.H / float(rows); b = math.floor(off); fr = off - b
        idx = (int(b) + np.arange(rows + 2)) % n.shape[0]
        big = cv2.resize(n[idx], (s.W, int((rows + 2) * cell)), interpolation=cv2.INTER_CUBIC); y0 = int((1 - fr) * cell)
        return big[y0:y0 + s.H]

    def cave(s, t, tint=0.0):
        """tint: 0=溶岩の赤 1=うろこの緑"""
        col = LAVA * (1 - tint) + GREEN * 0.7 * tint
        f = np.zeros((s.H, s.W, 3), np.float32) + np.array([0.022, 0.02, 0.03], np.float32)
        g = np.exp(-((s.yn - 1.02) / 0.26) ** 2) * (0.85 + 0.08 * math.sin(t * 3.1))
        f += g[:, :, None] * col[None, None, :]
        return f

    def rocks(s, f): return f * (1 - s.rock[:, :, None]) + np.array([0.035, 0.03, 0.035], np.float32)[None, None, :] * s.rock[:, :, None]

    def embers(s, f, t, tint=0.0):
        L = s.L0()
        for r in s.em:
            y = s.H - ((r[1] + t * 0.07 * (0.5 + r[2])) % 1.0) * s.H * 0.95; x = r[0] * s.W + math.sin(t * 1.3 + r[3] * 9) * 14 * s.k
            cv2.circle(L, (int(x), int(y)), max(1, int((1.2 + 1.6 * r[3]) * s.k)), float(0.9 * (y / s.H)), -1, cv2.LINE_AA)
        f += (L + cv2.GaussianBlur(L, (0, 0), 3 * s.k))[:, :, None] * (LAVA * (1 - tint) + GREEN * tint + 0.15)[None, None, :]

    def dragon(s, t, a, op, glow=1.0, tint=0.0):
        """竜の層 (色, 不透明度)。op: 目の ひらき"""
        H, W, k = s.H, s.W, s.k
        c = np.zeros((H, W, 3), np.float32) + np.array([0.05, 0.10, 0.07], np.float32)
        c += np.clip((s.yn - 0.24) / 0.30, 0, 1)[:, :, None] * (np.array([0.22, 0.075, 0.02], np.float32) * (1 - tint) + GREEN * 0.12 * tint)[None, None, :]
        c += (s.scl * 0.055)[:, :, None] * GREEN[None, None, :]
        for sx in (-1, 1):                                                           # まゆの 岩角 と 鼻の穴
            cv2.line(c, (int(W / 2 + sx * W * 0.10), int(H * 0.165)), (int(W / 2 + sx * W * 0.34), int(H * 0.125)), (0.02, 0.04, 0.03), int(9 * k), cv2.LINE_AA)
            cv2.ellipse(c, (int(W / 2 + sx * W * 0.055), int(H * 0.452)), (int(9 * k), int(5 * k)), sx * 25, 0, 360, (0.30 * glow, 0.08 * glow, 0.01), -1, cv2.LINE_AA)
        for i in range(-4, 5):                                                       # 鼻すじの うろこ板 と 上あごの きば
            y = H * (0.25 + 0.02 * (4 - abs(i))); cv2.ellipse(c, (int(W / 2), int(H * (0.10 + 0.036 * (i + 4)))), (int(W * (0.035 + 0.004 * (i + 4))), int(H * 0.014)), 0, 0, 180, (0.02, 0.045, 0.03), max(1, int(3 * k)), cv2.LINE_AA)
        for i in range(-3, 4):
            if i == 0: continue
            x = W / 2 + i * W * 0.045; yb = H * (0.492 - 0.012 * abs(i)); ln = H * (0.030 if abs(i) == 3 else 0.018)
            cv2.fillPoly(c, [np.array([(x - W * 0.014, yb - H * 0.012), (x + W * 0.014, yb - H * 0.012), (x, yb + ln)], np.int32)], (0.80, 0.76, 0.62), cv2.LINE_AA)
        if op > 0.02:
            for ex, ey, rot in s.EY:
                cv2.ellipse(c, (int(ex), int(ey)), (int(W * 0.115), max(1, int(W * 0.046 * op))), rot, 0, 360, tuple(float(v) for v in AMBER * glow), -1, cv2.LINE_AA)
                cv2.ellipse(c, (int(ex), int(ey)), (int(W * 0.115), max(1, int(W * 0.046 * op))), rot, 0, 360, (0.25 * glow, 0.10 * glow, 0.0), max(1, int(3 * k)), cv2.LINE_AA)
                cv2.ellipse(c, (int(ex), int(ey)), (max(1, int(W * 0.011)), max(1, int(W * 0.042 * op))), 0, 0, 360, (0.0, 0.0, 0.0), -1, cv2.LINE_AA)
        return c, s.dm * a

    def put_dragon(s, f, t, a, op, glow=1.0, tint=0.0):
        c, m = s.dragon(t, a, op, glow, tint); f = f * (1 - m[:, :, None]) + c * m[:, :, None]
        if op > 0.02 and glow > 0.05:
            for ex, ey, rot in s.EY: s.soft(f, (ex, ey), s.W * 0.07, 0.30 * op * glow * a, AMBER)
        return f

    def hero(s, f, t, dx=0.0, dy=0.0, **kw):
        img = s.pose(t, **kw); s.draw(f, img, s.CX + dx, s.CY + dy, 0.42)

    def heyes(s, f, v):
        if v <= 0.01: return
        sc = s._pl[2]; L = s.L0()
        for e in (s.eL, s.eR):
            p = s.at(*e); cv2.circle(L, (int(p[0]), int(p[1])), max(1, int(9 * sc)), float(min(1.0, v)), -1, cv2.LINE_AA)
            if v > 1: s.star4(L, p, 30 * sc * v, 0.8, 0.0, 0.10)
        s.add(f, L, YEL, 0.8, 2, 7)

    def blade(s, f, v, col=GREEN, th=5):
        if v <= 0.01: return
        a = s.at(*s.grd); b = s.at(*s.tip); L = s.L0(); s.line(L, a, b, min(1.0, v), max(1, int(th * s.k))); s.add(f, L, col, 0.9 * min(1.5, v), 2, 9)

    def frame(s, t):
        if t < T_FIRE: f, sh = s.sc_cave(t)
        elif t < T_CUT: f, sh = s.sc_fire(t - T_FIRE)
        elif t < T_SL: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_SC: f, sh = s.sc_slash(t)
        else: f, sh = s.sc_scales(t - T_SC)
        f, sh = s.finale(f, sh, t, DUR, s.tip, (0.3, 1, 0.5), (1, 0.7, 0.15), -1)
        return s.post(f, GREEN * 0.5 + 0.5, 0.45, sh * s.k, t, 0.82)

    # 1) 闇の中で 目が ひらく
    def sc_cave(s, t):
        W, H, k = s.W, s.H, s.k
        f = s.cave(t); f = s.put_dragon(f, t, 0.9 * ease((t - 0.5) / 1.0), ease2((t - 1.0) / 0.45), 1.0)
        mg = clamp((t - 1.7) / 0.6)
        if mg > 0: s.soft(f, (W / 2, H * 0.475), W * 0.10 * mg, 0.9 * mg, LAVA + 0.15)
        f = s.rocks(f); s.embers(f, t); s.hero(f, t)
        return f * clamp(t / 0.4), 0

    # 2) 火を はかれても ふりむかない。うろこの盾で 火が われる
    def sc_fire(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_FIRE
        f = s.cave(t); f = s.put_dragon(f, t, 0.9, 1.0, 1.3); f = s.rocks(f)
        pw = clamp(u / 0.25) * (1 - ease((u - 1.55) / 0.4))
        n = 0.55 * s.noise(s.n1, -t * 90, 30) + 0.45 * s.noise(s.n2, -t * 45, 12)
        front = np.clip((H * 1.5 * ease(u / 0.45) + H * 0.44 - s.YY) / (H * 0.10), 0, 1) * np.clip((s.YY - H * 0.43) / (H * 0.07), 0, 1)
        cone = np.clip(1.25 - np.abs(s.XX - W / 2) / (W * 0.10 + (s.YY - H * 0.44) * 1.3 + 1), 0, 1)
        fi = np.clip(n * 1.25 + 0.22, 0, 1) * front * np.clip(cone * 2, 0, 1) * pw
        ay = s.CY - H * 0.245; wy = np.minimum(np.sqrt(np.clip(s.YY - ay, 0, None) / H) * W * 0.66, W * 0.36)
        dd = wy - np.abs(s.XX - s.CX); ins = np.clip(dd / (10 * k), 0, 1) * (s.YY > ay)
        fi = fi * (1 - ins)
        fc = np.stack([np.clip(fi * 1.9, 0, 1), np.clip(fi * 1.45 - 0.34, 0, 1), np.clip(fi * 1.6 - 1.28, 0, 1)], 2)
        f = f * (1 - np.clip(fi * 1.6, 0, 1))[:, :, None] + fc
        s.hero(f, t, jit=0.5 * k * pw)
        rim = np.clip(1 - np.abs(dd) / (7 * k), 0, 1) * (s.YY > ay - 4 * k) * np.clip(1.4 - (s.YY - ay) / (H * 0.34), 0, 1)
        band = np.clip(1 - dd / (64 * k), 0, 1) * ins * np.clip(1.3 - (s.YY - ay) / (H * 0.32), 0, 1)
        hit = pw * (0.65 + 0.35 * n)
        f += ((cv2.GaussianBlur(rim, (0, 0), 3 * k) * 1.2 + band * s.scl * 0.9) * hit)[:, :, None] * GREEN[None, None, :]
        s.heyes(f, clamp((u - 1.6) / 0.3) * 1.3)
        if u > 1.75:                                                                 # 火が やんで うろこが ひとなで 光る
            v = (u - 1.75) / 0.4; f += (np.exp(-((s.YY - (ay + H * 0.5 * v)) / (30 * k)) ** 2) * ins * s.scl * (1 - v) * 1.2)[:, :, None] * GREEN[None, None, :]
        if u < 0.1: f += (1 - u / 0.1) * 0.5 * LAVA[None, None, :]
        return f, 2.2 * pw

    # 3) 竜の ひとみに うつる 騎士
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        g = back(u / 0.3, 1.3); C = (W / 2, H * 0.38); hw, hh = W * 0.27 * g, H * 0.30
        f = s.iris * (0.75 + 0.25 * ease(u / 0.5))
        pts = [(C[0] + hw * math.sin(math.pi * i / 24.0), C[1] - hh + 2 * hh * i / 24.0) for i in range(25)] + [(C[0] - hw * math.sin(math.pi * i / 24.0), C[1] + hh - 2 * hh * i / 24.0) for i in range(25)]
        m = s.L0(); cv2.fillPoly(m, [np.array(pts, np.int32)], 1.0, cv2.LINE_AA)
        n = s.noise(s.n2, -t * 40, 20)
        win = np.zeros((H, W, 3), np.float32) + np.array([0.02, 0.03, 0.03], np.float32) + (n * 0.22 * np.clip(1.2 - s.yn * 1.6, 0, 1))[:, :, None] * LAVA[None, None, :]
        zf = 1.02 + 0.05 * ease(u / 1.9); sc = zf * H / s.ah; img = s.pose(t)
        s.draw(win, img, C[0] - (s.fc[0] - s.aw / 2) * sc, C[1] - H * 0.02 - (s.fc[1] - s.ah / 2) * sc, zf)
        s.heyes(win, 0.9 + 1.2 * math.exp(-max(0.0, u - 0.45) * 5) * (u > 0.45))
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        L = s.L0(); cv2.polylines(L, [np.array(pts, np.int32)], True, 1.0, max(1, int(3 * k)), cv2.LINE_AA); s.add(f, L, AMBER, 0.5, 2, 6)
        nn = len(s.ht); step = W * 0.20
        for i, T in enumerate(s.ht):                                                 # 技名: 右から 斬りこんで 止まる
            a = (u - 0.55 - i * 0.12) / 0.2
            if a > 0: s.text(f, T, W / 2 + (i - (nn - 1) / 2) * step + (1 - ease(a)) * 60 * k, H * 0.835, clamp(a * 2), 1.0)
        for i, T in enumerate(s.vo):
            if u > 0.4 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.762, 1.0, 1.0)
        if u > 1.2: s.text(f, s.txt_en, W / 2, H * 0.835 + 62 * k, ease((u - 1.2) / 0.3), 1.0)
        if u < 0.08: f = f + (1 - u / 0.08) * 0.7
        return f, 0

    def cutline(s, L, pr, v, th):
        a = s.P0 - s.dir * s.H * 0.9; b = a + s.dir * s.H * 1.8 * pr; s.line(L, a, b, v, th)

    # 4) 一閃 → 間 → ずれる
    def sc_slash(s, t):
        W, H, k = s.W, s.H, s.k
        if T_BLK <= t < T_AFT:                                                       # まっ暗に 線 1本
            v = t - T_BLK; f = np.zeros((H, W, 3), np.float32); L = s.L0(); s.cutline(L, ease(v / 0.09), 1.0, max(1, int(4 * k)))
            s.add(f, L, GREEN, 1.3, 2, 14)
            if v < 0.05: f += 0.6
            return f, 0
        f = s.cave(t)
        if t < T_BLK:                                                                # にらみあい: 刃に 光が 走る
            v = t - T_SL; f = s.put_dragon(f, t, 0.9, 1.0, 1.2); f = s.rocks(f); s.embers(f, t); s.hero(f, t); s.heyes(f, 1.0)
            q = clamp(v / 0.3); a = s.at(*s.grd); b = s.at(*s.tip); p = (a[0] + (b[0] - a[0]) * q, a[1] + (b[1] - a[1]) * q)
            L = s.L0(); s.star4(L, p, 34 * k * math.sin(q * 3.1416) + 1, 1.0, 0.4, 0.12); s.add(f, L, WHITE, 0.9, 2, 6)
            if v < 0.1: f = f * (v / 0.1)
            return f, 0
        v = t - T_AFT; w = t - T_SPLIT
        lunge = math.exp(-v * 4.5)
        if w < 0:                                                                    # しずかな間
            f = s.put_dragon(f, t, 0.9, 1.0 + 0.25 * clamp(v / 0.2), 1.2); L = s.L0(); s.cutline(L, 1.0, 0.55 * (1 - v / 0.9), max(1, int(1.5 * k))); s.add(f, L, GREEN, 0.5, 1, 4)
            sh = 0
        else:
            c, m = s.dragon(t, 0.9, 1.25, 1.2 * (1 - ease(w / 0.5)), ease(w / 0.6)); d = W * 0.11 * ease(w / 0.7); gp = 9 * k * ease(w / 0.25)
            L = s.L0(); s.cutline(L, 1.0, 1.0, max(1, int((4 + 26 * ease(w / 0.3)) * k)))
            f += (cv2.GaussianBlur(L, (0, 0), 16 * k) * 1.1 * (1 - 0.5 * clamp(w / 0.8)))[:, :, None] * GREEN[None, None, :]
            for sg, hm in ((1, s.side), (-1, 1 - s.side)):
                M = np.float32([[1, 0, (s.dir[0] * d + s.nrm[0] * gp) * sg], [0, 1, (s.dir[1] * d + s.nrm[1] * gp) * sg]])
                mm = cv2.warpAffine(m * hm, M, (W, H)); cc = cv2.warpAffine(c, M, (W, H)); f = f * (1 - mm[:, :, None]) + cc * mm[:, :, None]
            if w < 0.1: f = f + (1 - w / 0.1) * 0.8
            sh = 15 * math.exp(-w * 6)
        f = s.rocks(f); s.embers(f, t, ease(w / 0.6) if w > 0 else 0.0)
        s.hero(f, t, dx=-W * 0.07 * lunge, dy=H * 0.012 * lunge, lean=-10 * k * lunge, squash=0.04 * lunge)
        s.heyes(f, 1.0); s.blade(f, 0.9 * lunge + 0.25, GREEN * 0.5 + 0.5, 4)
        if v < 0.06: f = f + (1 - v / 0.06) * 0.5
        return f, sh

    # 5) 竜が うろこに ほどけて 剣に はりつく
    def sc_scales(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_SC
        f = s.cave(t, 1.0); fd = 1 - ease(u / 0.9)
        if fd > 0.01:
            c, m = s.dragon(t, 0.9 * fd, 0.0, 0.0, 1.0); d = W * 0.11; gp = 9 * k
            for sg, hm in ((1, s.side), (-1, 1 - s.side)):
                M = np.float32([[1, 0, (s.dir[0] * d + s.nrm[0] * gp) * sg], [0, 1, (s.dir[1] * d + s.nrm[1] * gp) * sg]])
                mm = cv2.warpAffine(m * hm, M, (W, H)); cc = cv2.warpAffine(c, M, (W, H)); f = f * (1 - mm[:, :, None]) + cc * mm[:, :, None]
        f = s.rocks(f); s.embers(f, t, 1.0)
        wv = t - T_WAVE
        s.hero(f, t, squash=(-0.04 * math.exp(-wv * 7) if wv > 0 else 0))
        ga = s.at(*s.grd); tp = s.at(*s.tip); got = 0; L = s.L0()
        for x, y, r1, r2, r3 in s.sp:
            v = (u - r1 * 0.5) / 0.55
            if v <= 0: continue
            if v >= 1: got += 1; continue
            e = ease2(v); sgn = 1 if s.side[int(y), int(x)] > 0.5 else -1
            x0 = x + (s.dir[0] * W * 0.11) * sgn; y0 = y + (s.dir[1] * W * 0.11) * sgn
            tx = ga[0] + (tp[0] - ga[0]) * r2; ty = ga[1] + (tp[1] - ga[1]) * r2
            px = x0 + (tx - x0) * e + math.sin(e * 3.1416) * (r3 - 0.5) * 220 * k; py = y0 + (ty - y0) * e - math.sin(e * 3.1416) * 40 * k
            a = v * 9 + r3 * 6; sz = (9 - 4 * e) * k
            q = [(px + math.cos(a + j * 1.5708) * sz * (1.0 if j % 2 == 0 else 0.62), py + math.sin(a + j * 1.5708) * sz * (1.0 if j % 2 == 0 else 0.62)) for j in range(4)]
            cv2.fillPoly(f, [np.array(q, np.int32)], (0.10, float(0.52 + 0.3 * r3), 0.26), cv2.LINE_AA); cv2.circle(L, (int(px), int(py)), max(1, int(2 * k)), 0.9, -1, cv2.LINE_AA)
        f += (L + cv2.GaussianBlur(L, (0, 0), 4 * k))[:, :, None] * GREEN[None, None, :]
        fr = got / float(len(s.sp))
        s.blade(f, 0.25 + 1.1 * fr + (0.5 * math.exp(-wv * 4) if wv > 0 else 0), GREEN, 4 + 5 * fr)
        if fr > 0.02:                                                                # 刃に うろこ模様が のる
            Lb = s.L0(); s.line(Lb, ga, (ga[0] + (tp[0] - ga[0]) * 1.0, ga[1] + (tp[1] - ga[1]) * 1.0), 1.0, max(1, int(26 * s._pl[2])))
            f += (cv2.GaussianBlur(Lb, (0, 0), 2 * k) * s.scl * 0.8 * fr)[:, :, None] * (GREEN * 0.8 + 0.2)[None, None, :]
        s.heyes(f, 1.0 + (1.3 * math.exp(-wv * 4) if wv > 0 else 0))
        sh = 0
        if wv > 0:                                                                   # うろこが 竜の頭に 組み上がり、こちらへ かみついてくる
            g = clamp(wv / 1.15); z = 0.10 + 3.6 * g ** 2.3; c, m = s.dragon(t, 1.0, 1.0 + 0.25 * g, 1.5, 1.0)
            c = c * 2.2 + (s.scl * 0.30)[:, :, None] * GREEN[None, None, :]
            px, py = W / 2, H * 0.24; M = np.float32([[z, 0, px * (1 - z)], [0, z, py * (1 - z) + H * 0.16 * g]])
            mm = cv2.warpAffine(m, M, (W, H)) * clamp(wv / 0.15); cc = cv2.warpAffine(c, M, (W, H)); f = f * (1 - mm[:, :, None]) + cc * mm[:, :, None]
            for ex, ey, rot in s.EY:
                e = (px + (ex - px) * z, py + (ey - py) * z + H * 0.16 * g)
                if -W * 0.3 < e[0] < W * 1.3: s.soft(f, e, min(160 * k, W * 0.07 * z), 0.35, AMBER)
            if wv < 0.1: f = f + (1 - wv / 0.1) * 0.5
            sh = 3 + 11 * g ** 2
        if u < 0.001: pass
        end = DUR - t
        if end < 0.45: v = clamp((0.45 - end) / 0.4); f = f * (1 - v) + v
        return f, sh

    # ---- この子だけの曲: ニ短調の 重い角笛と かなとこ(金づちの カン・カカン)。火の間は かなとこが 速くなる。一閃のあとは 無音。竜が ずれて 低い金管、うろこが つく たびに 小さな鈴、さいごは 長調で しめる ----
    def audio(s, path):
        a = Snd(DUR, 33); D = 146.83; mn = [0, 2, 3, 5, 7, 8, 10]
        def fq(d, o=0): return D * 2 ** ((mn[d % 7] + 12 * (d // 7 + o)) / 12.0)
        def horn(d, t0, ln, g=0.07, o=0):
            n0 = len(a.a); b = Snd(DUR, 1); b.note(fq(d, o), t0, ln, 1.0, "saw", 0.05); a.a += a.lp(b.a.astype(np.float32), 0.10) * g * 3
        def anvil(t0, g=0.10, hi=False):
            for fr, gg in ((1870 if hi else 1480, 1.0), (2790 if hi else 2310, 0.6), (4100, 0.3)): a.note(fr, t0, 0.07, g * gg, "sine", 0.0005)
        a.a += a.lp(a.nz, 0.02) * 0.9 * a.gate(0, T_FIRE, 0.4) + a.tone(36.7) * 0.10 * a.gate(0, T_CUT, 0.3)                    # 洞くつの うなり
        horn(0, 1.0, 0.9, 0.08, -1); horn(4, 1.0, 0.9, 0.05, -1); a.boom(1.0, 38, 0.6, 0.5)                                      # 目が ひらく
        a.rise(1.7, T_FIRE, 60, 240, 0.10)
        a.a += (a.lp(a.nz, 0.12) * 0.55 + a.lp(a.nz, 0.5) * 0.12) * a.gate(T_FIRE, T_FIRE + 1.85, 0.15) * (0.8 + 0.2 * np.sin(a.t * 31))  # 火
        a.boom(T_FIRE, 44, 1.0, 0.5)
        tk = T_FIRE + 0.15; i = 0
        while tk < T_FIRE + 1.8: anvil(tk, 0.07, i % 3 == 2); tk += [0.20, 0.10, 0.20, 0.16, 0.08, 0.16, 0.12, 0.06, 0.12, 0.10, 0.05, 0.10][min(i, 11)]; i += 1
        horn(0, T_FIRE + 0.2, 0.7, 0.06); horn(2, T_FIRE + 0.9, 0.7, 0.06); horn(4, T_FIRE + 1.5, 0.8, 0.07)
        a.note(fq(7, 2), T_FIRE + 1.95, 0.5, 0.06, "bell")
        a.boom(T_CUT, 55, 0.9, 0.3)                                                                                              # カットイン: 角笛の ふし
        for i, (d, ln) in enumerate(((0, 0.18), (4, 0.18), (7, 0.36), (9, 0.18), (8, 0.18), (7, 0.18), (4, 0.5))):
            t0 = T_CUT + 0.25 + [0, 0.2, 0.4, 0.8, 1.0, 1.2, 1.4][i]; horn(d, t0, ln, 0.09); horn(d, t0, ln, 0.04, -1)
        for i in range(4): anvil(T_CUT + 0.55 + i * 0.12, 0.09, True); a.boom(T_CUT + 0.55 + i * 0.12, 70, 0.35, 0.12)
        a.note(2600, T_SL + 0.05, 0.25, 0.05, "sine"); a.rise(T_SL, T_BLK, 900, 3000, 0.04)                                       # 刃の光
        a.rise(T_BLK, T_BLK + 0.09, 1800, 7000, 0.22); a.hat(T_BLK + 0.02, 0.5, 0.10); a.note(5200, T_BLK + 0.09, 0.5, 0.07, "sine")  # 一閃
        a.note(fq(0, -2), T_AFT + 0.05, 0.5, 0.05, "sine")
        a.boom(T_SPLIT, 34, 1.6, 0.8); a.a += a.lp(a.nz, 0.2) * a.env(T_SPLIT, 0.01, 0.5) * 0.5                                    # ずれる
        horn(0, T_SPLIT + 0.05, 0.8, 0.10, -1); horn(2, T_SPLIT + 0.05, 0.8, 0.06, -1); horn(4, T_SPLIT + 0.05, 0.8, 0.06, -1)
        a.whoosh(T_SPLIT + 0.1, T_SC, 0.25, 0.08)
        pen = [0, 2, 4, 7, 9]; order = np.argsort(s.sp[:, 2])                                                                     # うろこが つく たびに 鈴
        for j, idx in enumerate(order[::3]):
            t0 = T_SC + s.sp[idx, 2] * 0.5 + 0.55; a.note(D * 4 * 2 ** ((pen[j % 5] + 12 * (j // 15 % 2)) / 12.0), t0, 0.10, 0.035, "bell", 0.001)
        for i, d in enumerate((0, 2, 4, 5)): horn(d, T_SC + 0.2 + i * 0.45, 0.42, 0.06)
        a.rise(T_SC + 1.2, T_WAVE, 150, 600, 0.07)
        a.boom(T_WAVE, 40, 1.4, 0.7); anvil(T_WAVE, 0.14, True)                                                                  # うろこの波: 長調
        for semi, g in ((0, 0.10), (4, 0.07), (7, 0.07), (12, 0.06)):
            b = Snd(DUR, 1); b.note(D * 2 ** (semi / 12.0), T_WAVE, 0.9, 1.0, "saw", 0.03); a.a += a.lp(b.a.astype(np.float32), 0.12) * g * 3
        for i, semi in enumerate((12, 16, 19, 24)): a.note(D * 2 * 2 ** (semi / 12.0), T_WAVE + 0.08 + i * 0.07, 0.6, 0.05, "bell")
        s.fin_snd(a, DUR, 33); a.save(path)


if __name__ == "__main__":
    r = Sieg()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
