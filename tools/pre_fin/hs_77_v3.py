"""エクリプス(★6 黒い太陽の王) 専用の必殺技「日食ノ王冠」
まひるの野。ひざまずく王と、長くのびる影 → 金の目が ひらく(左は白地に黒い目、右は黒地に金の目の 二つ割りカットイン) →
杖の「黒い太陽」が 空へ のぼり、日を かくす → 世界が 影絵に 反転する(黒地に 金の線) → 足もとの影が 立ち上がり、巨大な「影の王」になる →
影の王の頭に とげの王冠が おりる → とげが 金の刃になって ふりそそぐ → ダイヤモンドリングの 一点の光 → まひるに もどる
使い方: python3 hs_77.py 出力.mp4 / python3 hs_77.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 13.6
GOLD = np.array([1.0, 0.80, 0.30], np.float32); SKY = np.array([0.62, 0.82, 1.0], np.float32); SAND = np.array([0.93, 0.86, 0.66], np.float32); INK = np.array([0.03, 0.025, 0.05], np.float32)
T_OPEN, T_CUT, T_RISE, T_FLIP, T_KING, T_CROWN, T_RAIN, T_RING, T_DAY = 1.9, 2.7, 4.8, 6.5, 6.9, 8.5, 9.9, 11.5, 12.2


class Eclipse(Kit):
    def __init__(s, W=540):
        super().__init__(77, (1.0, 0.80, 0.30), "日食ノ王冠", "Eclipse Crown", W)
        s.eyeB = s.A(272, 300); s.eyeG = s.A(396, 282); s.orb = s.A(512, 142); s.crownA = s.A(322, 100)
        s.vt = [s.text_img(c, hz.SERIF, int(70 * s.k), 3) for c in "日食ノ王冠"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        rg = np.random.RandomState(77); s.rd = rg.rand(200, 5).astype(np.float32)
        al = s.cp[:, :, 3]; s.sil = np.zeros_like(s.cp); s.sil[:, :, 3] = al
        rim = cv2.dilate(al, np.ones((5, 5), np.uint8)) - al; s.rim = np.zeros_like(s.cp); s.rim[:, :, :3] = 1.0; s.rim[:, :, 3] = np.clip(rim, 0, 1)
        s.yn = s.YY / s.H; s.gnd = s.H * 0.82

    # ---- この子だけの絵 ----
    def noon(s, t, lum=1.0):
        f = (SKY[None, None, :] * (1.15 - 0.5 * s.yn)[:, :, None] + (WHITE * 0.25)[None, None, :] * (s.yn ** 2)[:, :, None]) * lum
        g = int(s.gnd); f[g:, :, :] = SAND * (1.0 - 0.25 * (s.yn[g:, :, None] - 0.82) / 0.18) * lum
        return f.astype(np.float32)

    def sun(s, f, c, r, v=1.0):
        L = s.L0(); cv2.circle(L, (int(c[0]), int(c[1])), int(r), 1.0, -1, cv2.LINE_AA)
        f += (L + cv2.GaussianBlur(L, (0, 0), r * 0.8) * 1.2)[:, :, None] * (WHITE * 0.7 + GOLD * 0.3)[None, None, :] * v

    def black_sun(s, f, c, r, t, v=1.0, spikes=12, glow=1.0):
        """とげのある 黒い太陽(杖の先と 同じ形)"""
        L = s.L0()
        for i in range(spikes):
            a = i * 6.2832 / spikes + t * 0.3; w = 0.16
            cv2.fillPoly(L, [np.array([(c[0] + math.cos(a - w) * r * 1.02, c[1] + math.sin(a - w) * r * 1.02), (c[0] + math.cos(a) * r * (1.55 + 0.2 * (i % 2)), c[1] + math.sin(a) * r * (1.55 + 0.2 * (i % 2))), (c[0] + math.cos(a + w) * r * 1.02, c[1] + math.sin(a + w) * r * 1.02)], np.int32)], 1.0, cv2.LINE_AA)
        cv2.circle(L, (int(c[0]), int(c[1])), int(r * 1.06), 1.0, max(2, int(r * 0.10)), cv2.LINE_AA)
        f[:] = f * (1 - np.clip(L, 0, 1) * v)[:, :, None] + np.clip(L, 0, 1)[:, :, None] * GOLD[None, None, :] * v
        f += cv2.GaussianBlur(L, (0, 0), max(2, r * 0.25))[:, :, None] * GOLD[None, None, :] * 0.6 * glow * v
        core = s.L0(); cv2.circle(core, (int(c[0]), int(c[1])), int(r), 1.0, -1, cv2.LINE_AA); f *= (1 - core * v)[:, :, None]

    def gold_eye(s, f, v, streak=1.0):
        if v <= 0: return
        p = s.at(*s.eyeG); sc = s._pl[2]; L = s.L0()
        cv2.ellipse(L, (int(p[0]), int(p[1])), (max(1, int(13 * sc)), max(1, int(20 * sc))), 0, 0, 360, float(v), -1, cv2.LINE_AA)
        s.line(L, (p[0] - 20 * sc, p[1]), (p[0] + 200 * sc * streak * v, p[1] - 30 * sc * streak * v), 0.8 * v, 1)
        s.add(f, L, GOLD, 1.3, 2, 8)

    def place(s, f, img, cx, cy, hfrac, alpha=1.0, tint=None):
        sc = hfrac * s.H / s.ah; im = img
        if tint is not None: im = img.copy(); im[:, :, :3] = np.asarray(tint, np.float32)[None, None, :]
        s.put(f, im, cx, cy, sc, alpha)

    def frame(s, t):
        k = s.k
        if t < T_CUT: f, sh = s.sc_noon(t)
        elif t < T_RISE: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_CROWN: f, sh = s.sc_flip(t)
        else: f, sh = s.sc_crown(t)
        f, sh = s.finale(f, sh, t, DUR, s.orb, (1, 0.82, 0.3), (0.5, 0.3, 0.9), 4)
        return s.post(f, GOLD, 0.45, sh * k, t, 0.97 if (t < T_FLIP or t > T_DAY) else 0.88)

    def ground_shadow(s, f, cx, feet, hfrac, ln, alpha):
        """足もとから のびる影(絵の形を つぶして たおす)"""
        sc = hfrac * s.H / s.ah; w = int(s.pw * sc); h = max(2, int(s.ph * sc * ln))
        im = cv2.resize(s.sil[:, :, 3], (w, h), interpolation=cv2.INTER_AREA)[::-1]
        M = np.float32([[1, 0.9, 0], [0, 1, 0]]); im = cv2.warpAffine(im, M, (w + h, h))
        x0 = int(cx - w / 2); y0 = int(feet - 4 * s.k)
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(s.W, x0 + w + h), min(s.H, y0 + h)
        if xa < xb and ya < yb: f[ya:yb, xa:xb] *= (1 - im[ya - y0:yb - y0, xa - x0:xb - x0] * alpha)[:, :, None]

    # 1) まひる
    def sc_noon(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_OPEN
        f = s.noon(t, clamp(t / 0.4)); s.sun(f, (W * 0.5, H * 0.15), W * 0.085)
        L = s.L0()
        for i in range(16):                                                          # ゆらめく 日ざし
            a = i * 0.3927 + t * 0.12; s.line(L, (W * 0.5 + math.cos(a) * W * 0.13, H * 0.15 + math.sin(a) * W * 0.13), (W * 0.5 + math.cos(a) * W * (0.2 + 0.05 * math.sin(t * 2 + i)), H * 0.15 + math.sin(a) * W * (0.2 + 0.05 * math.sin(t * 2 + i))), 0.5, 2)
        f += cv2.GaussianBlur(L, (0, 0), 2 * k)[:, :, None]
        cy = s.gnd - H * 0.215
        s.ground_shadow(f, W / 2, s.gnd, 0.44, 0.30 + 0.25 * ease(t / 2.0), 0.55)
        img = s.pose(t); s.draw(f, img, W / 2, cy, 0.44)
        if u > 0:                                                                    # 金の目が ひらく
            s.gold_eye(f, clamp(1.6 - u), 1.0)
            if u < 0.1: f = f * 0.4 + (1 - u / 0.1) * 0.1
            L = s.L0(); p = s.at(*s.eyeG)
            for i in range(2):
                v = (u - i * 0.1) / 0.6
                if 0 < v < 1: s.circ(L, p, W * 1.1 * v, 1 - v, 3)
            f *= (1 - np.clip(L, 0, 1) * 0.5)[:, :, None]
        return f, (5 * math.exp(-u * 8) if u > 0 else 0)

    # 2) 二つ割り: 白地に黒い目 / 黒地に金の目
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        img = s.pose(t); zf = 1.55; sc = zf * H / s.ah; cy = H * 0.36
        ex = (s.eyeB[0] + s.eyeG[0]) / 2; ey = (s.eyeB[1] + s.eyeG[1]) / 2
        A = np.ones((H, W, 3), np.float32) * 0.97; B = np.zeros((H, W, 3), np.float32) + INK[None, None, :]
        s.draw(A, img, W * 0.5 - (ex - s.aw / 2) * sc, cy - (ey - s.ah / 2) * sc, zf)
        s.place(B, s.rim, W * 0.5 - (ex - s.aw / 2) * sc, cy - (ey - s.ah / 2) * sc, zf, 1.0, GOLD)       # 右半分は 金のふちどりだけ
        s._pl = (W * 0.5 - (ex - s.aw / 2) * sc, cy - (ey - s.ah / 2) * sc, sc); s.gold_eye(B, 0.8 + 0.2 * math.sin(u * 10), 0.6)
        split = W * 0.5 + (1 - ease(u / 0.25)) * W * 0.6
        m = (s.XX < split).astype(np.float32)[:, :, None]; f = A * m + B * (1 - m)
        band = (np.abs(s.YY - cy) < H * 0.16 * ease(u / 0.3)).astype(np.float32)[:, :, None]
        f = f * band + (INK * 0.6)[None, None, :] * (1 - band)
        L = s.L0(); s.line(L, (split, cy - H * 0.16), (split, cy + H * 0.16), 1.0, 2); s.line(L, (0, cy - H * 0.16), (W, cy - H * 0.16), 0.8, 2); s.line(L, (0, cy + H * 0.16), (W, cy + H * 0.16), 0.8, 2); s.add(f, L, GOLD, 0.9)
        n = len(s.vt); step = W * 0.17
        for i, T in enumerate(s.vt):                                                 # 技名: 影が 先に のびて、字が あとから 追いつく
            a = (u - 0.55 - i * 0.13) / 0.3
            if a <= 0: continue
            x = W / 2 + (i - (n - 1) / 2) * step; y = H * 0.70
            L2 = s.L0(); s.line(L2, (x, y + 38 * k), (x + 70 * k * ease(a), y + 38 * k + 26 * k * ease(a)), 0.5, int(8 * k)); f += cv2.GaussianBlur(L2, (0, 0), 3 * k)[:, :, None] * (GOLD * 0.35)[None, None, :]
            if a > 0.4: s.text(f, T, x, y, clamp((a - 0.4) * 3), 1.0)
        for i, T in enumerate(s.vo):
            if u > 0.45 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.585, 1.0, 1.0)
        if u > 1.4: s.text(f, s.txt_en, W / 2, H * 0.70 + 84 * k, ease((u - 1.4) / 0.3), 1.0)
        if u < 0.08: f = f * (u / 0.08) + (1 - u / 0.08)
        return f, 0

    # 3) 黒い太陽が 日をかくす → 反転 → 影の王
    def sc_flip(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_RISE; fl = t - T_FLIP; kg = t - T_KING
        sun = (W * 0.5, H * 0.15); cy = s.gnd - H * 0.215
        if fl < 0:                                                                   # まひるのまま、黒い太陽が のぼる
            cov = ease2(u / (T_FLIP - T_RISE - 0.1)); f = s.noon(t, 1.0 - 0.45 * cov); s.sun(f, sun, W * 0.085, 1 - cov * 0.5)
            s.ground_shadow(f, W / 2, s.gnd, 0.44, 0.55 + 0.5 * cov, 0.55 + 0.3 * cov)
            img = s.pose(t, lean=-6 * k * ease(u / 0.4)); s.draw(f, img, W / 2, cy, 0.44)
            o = s.at(*s.orb); c = (o[0] + (sun[0] - o[0]) * cov, o[1] + (sun[1] - o[1]) * cov)
            s.black_sun(f, c, (26 + 24 * cov) * k, t, 1.0, 12, 0.6); s.gold_eye(f, 0.6)
            return f, 0
        f = np.zeros((H, W, 3), np.float32) + INK[None, None, :]                      # 影絵の世界
        L = s.L0(); s.line(L, (0, s.gnd), (W, s.gnd), 0.9, 2)
        for i in range(14): r = s.rd[i]; s.line(L, (r[0] * W, s.gnd + (4 + r[1] * 60) * k), (r[0] * W + (20 + 60 * r[2]) * k, s.gnd + (4 + r[1] * 60) * k), 0.3, 1)
        s.add(f, L, GOLD, 0.7)
        s.black_sun(f, sun, 50 * k, t, 1.0, 12, 1.0)
        if kg > 0:                                                                   # 影が 立ち上がる
            g = ease(kg / 1.3); hk = 0.44 + 0.34 * g; ky = s.gnd - hk * H * 0.47 + (1 - g) * H * 0.02
            s.place(f, s.sil, W / 2, ky, hk, 0.92, (0.17, 0.12, 0.27)); s.place(f, s.rim, W / 2, ky, hk, 0.95 * g, GOLD)
            s._pl = (W / 2, ky, hk * H / s.ah); s._pose = dict(t=t, sway=0, breath=0, lean=0, squash=0, stretch=None); s.gold_eye(f, g, 1.8)
        img = s.pose(t); s.place(f, s.sil, W / 2, cy, 0.44, 1.0, (0.0, 0.0, 0.0)); s.place(f, s.rim, W / 2, cy, 0.44, 1.0, GOLD)
        s.draw(f, img, W / 2, cy, 0.44, 1.0, light=np.full((s.ph, s.pw), 0.8, np.float32)); s.gold_eye(f, 1.0, 0.8)
        if fl < 0.14: f = (1 - f) * (1 - fl / 0.14) + f * (fl / 0.14)                # 反転の一瞬
        return f, 12 * math.exp(-fl * 7) + (3 * clamp(kg / 1.3) * (kg > 0) * (kg < 1.3))

    # 4) 王冠 → 金の刃 → ダイヤモンドリング → まひる
    def sc_crown(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_CROWN; rn = t - T_RAIN; dr = t - T_RING; dy = t - T_DAY
        cy = s.gnd - H * 0.215
        if dy > 0:                                                                   # まひるに もどる
            f = s.noon(t, 1.0); s.sun(f, (W * 0.5, H * 0.15), W * 0.085)
            s.ground_shadow(f, W / 2, s.gnd, 0.44, 0.30, 0.55)
            img = s.pose(t, squash=0.03 * math.exp(-dy * 8)); s.draw(f, img, W / 2, cy, 0.44); s.gold_eye(f, clamp(1 - dy / 0.8) * 0.8, 0.6)
            if dy < 0.5: f = f * (dy / 0.5) + (1 - dy / 0.5)
            end = DUR - t
            if end < 0.4: f = f * clamp(end / 0.35)
            return f, 0
        f = np.zeros((H, W, 3), np.float32) + INK[None, None, :]
        L = s.L0(); s.line(L, (0, s.gnd), (W, s.gnd), 0.9, 2); s.add(f, L, GOLD, 0.7)
        hk = 0.78; ky = s.gnd - hk * H * 0.47
        s.place(f, s.sil, W / 2, ky, hk, 0.92, (0.17, 0.12, 0.27)); s.place(f, s.rim, W / 2, ky, hk, 0.95, GOLD)
        s._pl = (W / 2, ky, hk * H / s.ah); s._pose = dict(t=t, sway=0, breath=0, lean=0, squash=0, stretch=None); s.gold_eye(f, 1.0, 1.8)
        ca = s.at(*s.crownA)                                                         # 影の王の 頭の上へ 王冠が おりる
        g = ease(u / 1.1); C = (ca[0] + 10 * k, -H * 0.2 + (ca[1] + H * 0.02 + H * 0.2) * g); R = W * 0.19
        s.black_sun(f, C, R, t * (1 + 3 * clamp(rn / 1.4) if rn > 0 else 1), 1.0, 16, 1.0 + (1.5 * math.exp(-(u - 1.1) * 5) if u > 1.1 else 0))
        if rn > 0 and dr < 0.3:                                                      # とげが 金の刃になって ふりそそぐ
            L = s.L0()
            for i in range(46):
                r = s.rd[i]; v = (rn * (1.6 + r[2]) - r[0] * 1.2)
                if v <= 0: continue
                v = v % 1.0; x0 = C[0] + (r[1] - 0.5) * R * 2.2; x1 = x0 + (r[1] - 0.5) * W * 0.9; y1 = C[1] + R * 0.6 + (s.gnd - C[1] - R * 0.6) * v; x = x0 + (x1 - x0) * v
                dx = (x1 - x0) * 0.08; dyv = (s.gnd - C[1]) * 0.08
                cv2.fillPoly(L, [np.array([(x, y1), (x - dx - 3 * k, y1 - dyv), (x - dx * 2.2, y1 - dyv * 2.2), (x - dx + 3 * k, y1 - dyv)], np.int32)], 1.0, cv2.LINE_AA)
                if v > 0.9: s.ell(L, (x1, s.gnd), (30 * k * (v - 0.9) * 10, 6 * k * (v - 0.9) * 10), 0, 1 - (v - 0.9) * 10, 2)
            s.add(f, L, GOLD, 1.1, 2, 8)
        img = s.pose(t); s.place(f, s.sil, W / 2, cy, 0.44, 1.0, (0.0, 0.0, 0.0)); s.place(f, s.rim, W / 2, cy, 0.44, 1.0, GOLD)
        s.draw(f, img, W / 2, cy, 0.44, 1.0, light=np.full((s.ph, s.pw), 0.8, np.float32)); s.gold_eye(f, 1.0, 0.8)
        if dr > 0:                                                                   # ダイヤモンドリング: ふちの 一点が 光る
            a = -0.7; p = (C[0] + math.cos(a) * R * 1.03, C[1] + math.sin(a) * R * 1.03); v = ease(dr / 0.5)
            L = s.L0(); s.circ(L, C, R * 1.03, 0.9, 3); s.star4(L, p, (30 + 260 * v) * k, 1.0, 0.0, 0.07); s.star4(L, p, (16 + 110 * v) * k, 1.0, 0.7854, 0.10)
            cv2.circle(L, (int(p[0]), int(p[1])), int((8 + 30 * v) * k), 1.0, -1, cv2.LINE_AA)
            s.add(f, L, WHITE * 0.75 + GOLD * 0.25, 1.4, 4, 20)
            wo = clamp((dr - 0.3) / 0.4); f = f * (1 - wo) + wo
        if u < 0.1: f += (1 - u / 0.1) * 0.4
        return f, (6 * math.exp(-(u - 1.1) * 6) if u > 1.1 else 0) + (5 * (rn > 0) * (dr < 0)) + (10 * clamp(dr / 0.5) if dr > 0 else 0)

    # ---- この子だけの曲: まひるは 明るい長調の たてごと。目がひらくと 同じ旋律が 短調に うら返る。影の王は 低い金管、刃は 16分の 鉄琴、さいごに 長調へ もどる ----
    def audio(s, path):
        a = Snd(DUR, 77); C4 = 261.63; maj = [0, 2, 4, 5, 7, 9, 11]; mn = [0, 1, 3, 5, 7, 8, 10]
        def fq(d, o=0, sc=maj): return C4 * 2 ** ((sc[d % 7] + 12 * (d // 7 + o)) / 12.0)
        tune = [4, 2, 0, 2, 4, 7, 4, 2]
        for i, d in enumerate(tune): a.note(fq(d), 0.2 + i * 0.21, 0.5, 0.09, "bell")                               # まひる
        for d in (0, 4): a.note(fq(d, -1), 0.2, 1.6, 0.05, "pad", 0.2)
        a.a += a.lp(a.nz, 0.02) * 0.25 * a.gate(0, T_OPEN, 0.3)
        a.boom(T_OPEN, 50, 0.9, 0.5); a.note(fq(7, 1, mn), T_OPEN, 0.9, 0.12, "bell")
        a.boom(T_CUT, 46, 1.0)
        for i in range(5): a.note(fq(tune[i], 0, mn), T_CUT + 0.55 + i * 0.13, 0.4, 0.11, "bell"); a.note(fq(tune[i], -1, mn), T_CUT + 0.55 + i * 0.13, 0.4, 0.06, "saw", 0.01)
        a.rise(T_RISE, T_FLIP, 400, 90, 0.16); a.a += a.tone(55) * 0.12 * a.gate(T_RISE, T_FLIP, 0.4)                # 日が かげる
        for i, d in enumerate(tune): a.note(fq(d, 0, mn) * (1 - 0.01 * i), T_RISE + i * 0.2, 0.5, 0.07 * (1 - i / 10.0), "bell")
        a.boom(T_FLIP, 36, 1.5, 0.8); a.a += (a.nz - a.lp(a.nz, 0.5)) * a.env(T_FLIP, 0.001, 0.05) * 0.5              # 反転
        for j, (d, o) in enumerate(((0, -2), (4, -2), (0, -1), (2, -1))): a.note(fq(d, o, mn), T_KING + j * 0.3, 1.8, 0.10, "saw", 0.15)   # 影の王
        a.rise(T_KING, T_CROWN, 60, 140, 0.14)
        a.whoosh(T_CROWN, T_CROWN + 1.1, 0.5, 0.2); a.boom(T_CROWN + 1.1, 40, 1.3, 0.6)
        for d in (0, 2, 4, 7): a.note(fq(d, 0, mn), T_CROWN + 1.1, 1.2, 0.06, "saw", 0.02)
        tk = T_RAIN; i = 0
        while tk < T_RING:                                                           # 金の刃
            a.note(fq([7, 9, 11, 14, 11, 9][i % 6], 1, mn), tk, 0.12, 0.06, "bell"); a.hat(tk, 0.05)
            if i % 4 == 0: a.boom(tk, 58, 0.5, 0.12)
            if i % 2 == 0: a.note(fq([0, 0, 3, 4][(i // 2) % 4], -1, mn), tk, 0.18, 0.10, "saw", 0.005)
            tk += 0.1; i += 1
        a.note(fq(14, 2), T_RING, 1.2, 0.16, "sine", 0.02); a.rise(T_RING, T_DAY, 600, 2400, 0.10)                  # 一点の光
        a.boom(T_DAY, 48, 0.7, 0.5)
        for j, (d, o) in enumerate(((0, -1), (4, -1), (0, 0), (2, 0), (4, 0))): a.note(fq(d, o), T_DAY + j * 0.05, 1.6, 0.05, "pad", 0.3)
        for i, d in enumerate([4, 7, 9, 11]): a.note(fq(d), T_DAY + 0.2 + i * 0.2, 0.6, 0.09, "bell")
        s.fin_snd(a, DUR, 77); a.save(path)


if __name__ == "__main__":
    r = Eclipse()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
