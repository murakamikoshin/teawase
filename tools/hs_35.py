"""ヴァルキリア(★5 戦乙女) 専用の必殺技「戦乙女ノ槍」— 高品質版(hs_fx)。話は v3 と 同じ: 敵は さいしょから 画面の こちら側。槍は こちらへ 投げる
あらしの雲が 流れる 戦場あとの丘(ささった槍の 影・地を はう もや)。下から とび上がって 宙で止まる → 盾の 鳥の紋が 光る → 金の 丸盾の窓のカットイン →
紋の鳥が 光に なって 盾から ぬけ出し うしろで はばたく → 鳥が 槍に 重なり、ふりかぶって こちらへ 投げる →
こちらは 丸盾を かまえる(木目・鉄のふち・革の 持ち手の うら側)。槍は 板の すきまから 光を もらしながら 1枚、2枚、3枚と ぶちぬいて 近づき、光の鳥を まとった はがねの 穂先が 目の前に → 白 → 暗転
使い方: python3 hs_35.py 出力.mp4 / python3 hs_35.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old35", "/home/claude/tools/pre_fin/hs_35_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = old.DUR; ICE, GOLD, BLUE = old.ICE, old.GOLD, old.BLUE
T_CUT, T_SKY, T_ANT, T_TH, T_FLY, T_NEAR, HITS, UPS, SHC = old.T_CUT, old.T_SKY, old.T_ANT, old.T_TH, old.T_FLY, old.T_NEAR, old.HITS, old.UPS, old.SHC


class Val(FX, old.Val):
    def __init__(s, W=540):
        old.Val.__init__(s, W); s.fx_init(35); H, k = s.H, s.k; s._amb = (0.75, 0.85, 1.0); s._shade = 0.92; s._gt = {}
        s.hrim = np.clip(s.hill - cv2.warpAffine(s.hill, np.float32([[1, 0, 0], [0, 1, 3 * k]]), (W, H)), 0, 1)
        s.Cs = (W / 2, H * 0.50); s.gang = np.arctan2(s.YY - s.Cs[1], s.XX - s.Cs[0]); s.grad = np.hypot(s.XX - s.Cs[0], s.YY - s.Cs[1])

    # ---- 空と 丘 ----
    def storm(s, t, br=0.0):
        top = np.array([0.020, 0.035, 0.10], np.float32) * (1 - br) + np.array([0.30, 0.55, 0.95], np.float32) * br
        low = np.array([0.13, 0.17, 0.30], np.float32) * (1 - br) + np.array([0.85, 0.92, 1.0], np.float32) * br
        f = (top[None, None, :] * (1 - s.yn)[:, :, None] + low[None, None, :] * s.yn[:, :, None]).astype(np.float32)
        n = s.fbm(0.0007, t * 55, t * 8, 4, 90 * s.k, 0); n2 = s.fbm(0.0018, t * 90 + 40, t * 12, 3, 0, 2)
        cl = sstep(0.40, 0.72, n * 0.75 + n2 * 0.35); lit = cv2.GaussianBlur(sstep(0.44, 0.80, n * 0.8 + n2 * 0.3), (0, 0), 6 * s.k)
        cc = np.array([0.26, 0.31, 0.48], np.float32) * (1 - br) + np.array([0.95, 0.97, 1.0], np.float32) * br
        f = f * (1 - 0.7 * cl * (1 - br))[:, :, None] + (cl * (0.10 + 0.75 * lit ** 1.5) * (0.9 + 0.5 * br))[:, :, None] * cc[None, None, :]
        d = np.zeros_like(f); s.starfield(d, t, 260, ICE, 2, 1.0, (620.0, -70.0)); s.starfield(d, t + 3, 90, WHITE, 4, 1.6, (980.0, -110.0))        # 風に とぶ しぶき(横に 尾を ひく)
        f += cv2.blur(d, (int(54 * s.k), 1)) * 5.0
        return f

    def ground(s, f, br=0.0):
        c = np.array([0.010, 0.014, 0.030], np.float32) * (1 - br) + np.array([0.16, 0.26, 0.42], np.float32) * br
        f = f * (1 - s.hill[:, :, None]) + c[None, None, :] * s.hill[:, :, None]
        f += (cv2.GaussianBlur(s.hrim, (0, 0), 1.0 * s.k) * (0.35 + 0.4 * br))[:, :, None] * ICE[None, None, :]
        mist = s.fbm(0.0018, s._t * 30, 0, 3, 0, 3) * sstep(0.80, 0.97, s.yn) * 0.30
        return f + mist[:, :, None] * (ICE * 0.8)[None, None, :]

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        s.draw_lit(f, img, cx, cy, hfrac, rim=ICE * 0.6 + 0.4, rim_dir=(0.35, -0.94), rim_v=0.85, amb=s._amb, amb_v=0.14, alpha=alpha, shade=s._shade)

    def bird(s, L, c, size, rot=0.0, v=1.0, sy=1.0):
        T = s.L0(); old.Val.bird(s, T, c, size, rot, 1.0, sy); q = size / s.eh
        if q > 1.5: T = sstep(0.42, 0.58, cv2.GaussianBlur(np.clip(T, 0, 1), (0, 0), min(12 * s.k, q * 0.75)))                         # 大きくしても なめらかな ふち
        L += T * v

    def crest(s, f, v):
        if v <= 0.01: return
        c = s.at(*s.ec); L = s.L0(); s.bird(L, c, s.eh * s._pl[2], 0.0, min(1.0, v)); s.glowmask(f, np.clip(L, 0, 1), ICE, 0.8 * min(1.6, v), 0.7, 1.0, 10.0)
        s.flare(f, c, 6 * s.k * v, ICE * 0.5 + 0.5, 0.5 * min(1.5, v), 1.2, 0.0, False)

    def add(s, f, L, col, gain=1.0, b1=3, b2=12):                                        # 光の鳥・槍の光: にじみを 深く
        s.glowmask(f, np.clip(L, 0, 1), col, gain * 0.9, 0.75, 1.0, float(b2))
        ys, xs = np.nonzero(L > 0.5)
        if len(xs) > 40:                                                             # 羽の きらめき
            i = (s.rnd[:90, 0] * len(xs)).astype(np.int32); s.splat(f, xs[i] + (s.rnd[:90, 1] - 0.5) * 30 * s.k, ys[i] + ((s.rnd[:90, 2] + s._t * 0.6) % 1.0) * 50 * s.k, s.rnd[:90, 3] * gain, (1, 1, 1), 0.9, 0.5)

    # ---- こちらを 向いた 槍 ----
    def lance(s, f, c, S, t):
        k = s.k; W, H = s.W, s.H; fl = math.sin(t * 13) ** 2
        B = s.L0(); s.bird(B, (c[0], c[1] + S * 0.05), S * (1 + 0.04 * fl), 0.0, 1.0, 1 - 0.20 * fl); B = np.clip(B, 0, 1)
        sh = s.fbm(0.004, t * 60, t * 200, 3, 0, 1)
        f[:] = f * (1 - 0.55 * B[:, :, None]) + (B * (0.50 + 0.5 * sh))[:, :, None] * (ICE * 0.85 + 0.15)[None, None, :]             # 光の鳥(うすく すける)
        e_ = np.clip(B - cv2.GaussianBlur(B, (0, 0), 3 * k), 0, 1); f += (e_ * 2.0 + cv2.GaussianBlur(B, (0, 0), 14 * k) * 0.5)[:, :, None] * (ICE * 0.7 + 0.3)[None, None, :]
        r = S * 0.15; gm = s.L0()
        for sg in (-1, 1): cv2.fillPoly(gm, [np.array([(c[0] + sg * r * 0.3, c[1] - r * 0.16), (c[0] + sg * r * 1.5, c[1] - r * 0.10), (c[0] + sg * r * 1.75, c[1]), (c[0] + sg * r * 1.5, c[1] + r * 0.10), (c[0] + sg * r * 0.3, c[1] + r * 0.16)], np.int32)], 1.0, cv2.LINE_AA)
        gy_ = np.clip((s.YY - c[1]) / max(1.0, r * 0.16), -1, 1); gcol = GOLD[None, None, :] * (0.85 - 0.45 * gy_ + 0.5 * np.exp(-((gy_ + 0.45) / 0.18) ** 2))[:, :, None]
        f[:] = f * (1 - gm[:, :, None]) + gcol * gm[:, :, None]
        a = r * 0.55; b = r * 1.25; dx = (s.XX - c[0]) / a; dy = (s.YY - c[1]) / b; d = np.abs(dx) + np.abs(dy); aa = min(0.2, 2.0 / max(4.0, a)); m = sstep(1.0, 1.0 - aa, d)
        base = np.where(dx < 0, np.where(dy < 0, 1.25, 0.50), np.where(dy < 0, 0.80, 0.22)).astype(np.float32)
        ridge = np.exp(-(dx * a / (1.2 * k + r * 0.012)) ** 2) + np.exp(-(dy * b / (1.2 * k + r * 0.012)) ** 2)
        val = base * (0.62 + 0.38 * (1 - d)) + 0.25 * sstep(0.2, 0.0, np.abs(dx + dy * 0.6 + 0.25)) * (dy < 0) + ridge * 0.45
        steel = (val[:, :, None] * np.array([0.84, 0.92, 1.0], np.float32)[None, None, :] + (np.clip(dy, 0, 1) * 0.22)[:, :, None] * np.array([0.2, 0.45, 1.0], np.float32)[None, None, :]) * (1 - 0.75 * sstep(0.90, 1.0, d))[:, :, None]
        f[:] = f * (1 - m[:, :, None]) + steel * m[:, :, None]
        s.flare(f, c, max(3.0, min(34 * k, r * 0.22)), WHITE, 0.9, 1.6, 0.0, False)

    # ---- こちらが かまえた 丸盾の うら側 ----
    def gtex(s, j):
        if j in s._gt: return s._gt[j]
        W, H, k = s.W, s.H, s.k; R = W * (0.80 + 0.06 * j); x, y = s.Cs; col = np.asarray(SHC[j], np.float32); rr = s.grad / R; m = sstep(1.0, 0.994, rr)
        u = (s.XX - x) / (R * 0.2) + 0.5; pj = np.floor(u); fr = u - pj; gap = sstep(0.40, 0.49, np.abs(fr - 0.5))
        tone = np.mod(np.sin(pj * 12.9898 + j * 4.1) * 43758.5453, 1.0).astype(np.float32)
        gr = cv2.remap(s._tiles[0], ((s.XX * 1.9 + pj * 37) % 256).astype(np.float32), ((s.YY * 0.10 + pj * 53) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        gr2 = cv2.remap(s._tiles[1], ((s.XX * 0.6 + pj * 11) % 256).astype(np.float32), ((s.YY * 0.04) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        wood = (0.55 + 0.55 * gr) * (0.75 + 0.5 * gr2) * (0.80 + 0.35 * tone) * (1 - 0.85 * gap) * (1 + 0.25 * sstep(0.30, 0.38, np.abs(fr - 0.5)) * (1 - gap))
        wood *= (1.15 - 0.55 * rr) * (0.85 + 0.3 * np.clip(1 - np.hypot(s.XX - (x - R * 0.3), s.YY - (y - R * 0.4)) / (R * 0.9), 0, 1))
        c = wood[:, :, None] * (col * 1.5)[None, None, :]
        band = sstep(0.875, 0.885, rr); p = np.clip((rr - 0.88) / 0.12, 0, 1)                                                      # 鉄の ふち
        met = (0.16 + 0.42 * (np.cos(s.gang + 2.2) * 0.5 + 0.5) ** 2) * (0.45 + 0.75 * np.sin(p * 3.1416)) * (0.8 + 0.4 * s.fbm(0.02, j * 9, 3, 2, 0, 3))
        c = c * (1 - band[:, :, None]) + (met[:, :, None] * np.array([0.82, 0.86, 1.0], np.float32)[None, None, :]) * band[:, :, None]
        c *= (1 - 0.6 * np.exp(-((rr - 0.86) / 0.02) ** 2))[:, :, None]
        for q in range(12):
            a = q * 0.5236; px, py = x + math.cos(a) * R * 0.94, y + math.sin(a) * R * 0.94; rv = max(2, int(R * 0.024))
            cv2.circle(c, (int(px), int(py)), rv, (0.20, 0.21, 0.25), -1, cv2.LINE_AA); cv2.circle(c, (int(px - rv * 0.3), int(py - rv * 0.3)), max(1, int(rv * 0.45)), (0.75, 0.78, 0.88), -1, cv2.LINE_AA)
        bo = sstep(0.21, 0.20, rr); c = c * (1 - bo[:, :, None]) + (np.array([0.05, 0.05, 0.06], np.float32)[None, None, :] * (0.6 + 2.5 * np.clip(rr / 0.2, 0, 1) ** 3)[:, :, None]) * bo[:, :, None]   # こぶの うら
        st = sstep(R * 0.060, R * 0.052, np.abs(s.YY - y)) * sstep(R * 0.64, R * 0.62, np.abs(s.XX - x)); v_ = (s.YY - y) / (R * 0.055)                 # 革の 持ち手
        lea = (0.45 + 0.35 * np.cos(v_ * 1.4) + 0.25 * np.exp(-((v_ + 0.5) / 0.2) ** 2)) * (0.8 + 0.4 * gr2); stitch = (np.mod(s.XX, 9 * k) < 3 * k) * (np.abs(np.abs(v_) - 0.78) < 0.08)
        c = c * (1 - st[:, :, None]) + (lea[:, :, None] * (col * 0.8)[None, None, :] + stitch[:, :, None] * 0.25) * st[:, :, None]
        s._gt[j] = (c.astype(np.float32), m.astype(np.float32), gap.astype(np.float32)); return s._gt[j]

    def guard_at(s, f, j, oy, bulge=0.0):
        c, m, gap = s.gtex(j); W, H, k = s.W, s.H, s.k; R = W * (0.80 + 0.06 * j)
        if abs(oy) > 0.5:
            M = np.float32([[1, 0, 0], [0, 1, oy]]); c = cv2.warpAffine(c, M, (W, H)); m = cv2.warpAffine(m, M, (W, H)); gap = cv2.warpAffine(gap, M, (W, H))
        f[:] = f * (1 - m[:, :, None]) + c * (0.55 + 0.25 * bulge) * m[:, :, None]
        if bulge > 0:                                                                # 板の すきまから 光が もれ、まん中に ひびが 走る
            C = (s.Cs[0], s.Cs[1] + oy); near = np.exp(-(np.hypot(s.XX - C[0], s.YY - C[1]) / (R * (0.25 + 0.5 * bulge))) ** 2)
            f += (gap * m * near * bulge * 1.6)[:, :, None] * (ICE * 0.8 + 0.2)[None, None, :]
            for q in range(7):
                a = q * 0.9 + 0.3 + j; P = [C]
                for w in range(1, 5): P.append((C[0] + math.cos(a + 0.25 * math.sin(w * 2.1 + q)) * R * 0.14 * w * bulge, C[1] + math.sin(a + 0.25 * math.sin(w * 2.1 + q)) * R * 0.14 * w * bulge))
                s.glowline(f, P, ICE * 0.7 + 0.3, 2.4, 1.0 * bulge, 1.3)
            s.flare(f, C, 10 * k * bulge, WHITE, 0.9 * bulge, 1.2, 0.3, False)

    def shards(s, f, j, v):
        c, m, gap = s.gtex(j); W, H, k = s.W, s.H, s.k; Cx, Cy = s.Cs
        for i in range(6):
            a0 = i * 1.0472 + 0.3 * j; am = a0 + 0.5236; fly = W * (0.25 + 2.6 * v) * (1 - math.exp(-v * 7)) * (0.7 + 0.3 * ((i * 7 + j) % 3)); g = 1 + 1.2 * v
            da = np.mod(s.gang - a0, 6.2832); sect = ((da < 1.0472)).astype(np.float32) * m; ro = v * (2 + i) * (1 if i % 2 else -1) * 0.25
            M = cv2.getRotationMatrix2D((Cx, Cy), math.degrees(-ro), g); M[0, 2] += math.cos(am) * fly; M[1, 2] += math.sin(am) * fly
            mm = cv2.warpAffine(sect, M, (W, H)); cc = cv2.warpAffine(c, M, (W, H))
            if mm.max() <= 0.01: continue
            ed = np.clip(mm - cv2.GaussianBlur(mm, (0, 0), 3 * k), 0, 1)
            f[:] = f * (1 - mm[:, :, None]) + cc * (0.55 + 0.5 * math.exp(-v * 5)) * mm[:, :, None]; f += (ed * (1.8 * math.exp(-v * 4) + 0.3))[:, :, None] * (ICE * 0.7 + 0.3)[None, None, :]
        r = s.rnd[:260]; a_ = r[:, 0] * 6.2832; d_ = (20 + (300 + 900 * r[:, 1]) * v) * k                                              # 木の くず
        s.splat(f, Cx + np.cos(a_) * d_, Cy + np.sin(a_) * d_ + 300 * k * v * v, (1 - v / 0.7) * r[:, 2] * 1.6, (1.0, 0.85, 0.6), 1.2, 0.3)

    def frame(s, t):
        s._t = t; o = {}
        if t < T_CUT: f, sh = s.sc_rise(t)
        elif t < T_SKY: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.4)
        elif t < T_FLY: f, sh = s.sc_throw(t)
        else: f, sh = s.sc_fly(t)
        f, sh = s.finale(f, sh, t, DUR, s.stip, (0.72, 0.87, 1), (1, 0.82, 0.36), -1)
        return s.film(f, t, sh * s.k, 0.5, 0.85, (0.85, 0.93, 1.0), 1.0 + 0.15 * sh, 0.035, o.get("vig", 0.55), 1.12, 1.05)

    # 2) 金の 丸盾の 窓
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.storm(t) * 0.45
        g = back(u / 0.32, 1.4); C = (W / 2, H * 0.38); R = W * 0.40 * g
        rr = np.hypot(s.XX - C[0], s.YY - C[1]) / max(1.0, R); ang = np.arctan2(s.YY - C[1], s.XX - C[0]); m = sstep(1.0, 0.99, rr)
        f *= (1 - 0.6 * np.exp(-((rr - 1.0) / 0.25) ** 2) * (rr > 1))[:, :, None]
        win = (BLUE * 0.75)[None, None, :] * (1.15 - 0.55 * rr)[:, :, None] + s.nebula(t, [(0.05, 0.12, 0.40), (0.20, 0.40, 0.85), (0.8, 0.9, 1.0)], 0.0015, (60.0, -8.0), 0.8)
        s.starfield(win, t, 500, WHITE, 6, 0.8, (300.0, -40.0))
        zf = 1.05 + 0.05 * ease(u / 1.9); sc = zf * H / s.ah; img = s.pose(t)
        s.draw(win, img, C[0] - (s.fc[0] - s.aw / 2) * sc, C[1] + H * 0.01 - (s.fc[1] - s.ah / 2) * sc, zf)
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        band = sstep(0.955, 0.965, rr) * sstep(1.045, 1.035, rr); p = np.clip((rr - 0.955) / 0.09, 0, 1)                             # 金の ふち(まるみと てり)
        met = (0.35 + 0.55 * (np.cos(ang + 2.2) * 0.5 + 0.5) ** 1.5 + 0.5 * np.exp(-((np.mod(ang + 2.2 + 3.1416, 6.2832) - 3.1416) / 0.25) ** 2)) * (0.45 + 0.8 * np.sin(p * 3.1416))
        f = f * (1 - band[:, :, None]) + (met[:, :, None] * GOLD[None, None, :]) * band[:, :, None]
        for i in range(8):                                                           # びょう
            a = i * 0.7854 + 0.39; px, py = C[0] + math.cos(a) * R, C[1] + math.sin(a) * R
            cv2.circle(f, (int(px), int(py)), max(1, int(5 * k)), (0.55, 0.40, 0.14), -1, cv2.LINE_AA); cv2.circle(f, (int(px - 1.5 * k), int(py - 1.5 * k)), max(1, int(2.2 * k)), (1.0, 0.95, 0.70), -1, cv2.LINE_AA)
        sw = (u - 0.45) / 0.35                                                       # ふちを 光が ひとまわり
        if 0 < sw < 1:
            a = math.radians(-70 + 360 * sw); s.flare(f, (C[0] + math.cos(a) * R, C[1] + math.sin(a) * R), 11 * k, WHITE, 0.9, 1.2, a, False)
        f *= (1 - 0.6 * sstep(0.70, 0.79, s.yn) * sstep(0.96, 0.90, s.yn))[:, :, None]
        n = len(s.ht); step = W * 0.175
        for i, T in enumerate(s.ht):                                                 # 技名: 槍の向きに そって ささる
            a = (u - 0.55 - i * 0.11) / 0.2
            if a > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step - (1 - ease(a)) * 50 * k, H * 0.835 + (1 - ease(a)) * 30 * k, clamp(a * 2), 1.0)
        for i, T in enumerate(s.vo):
            if u > 0.4 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.755, 1.0, 1.0)
        if u > 1.25: s.text(f, s.txt_en, W / 2, H * 0.835 + 60 * k, ease((u - 1.25) / 0.3), 1.0)
        if u < 0.08: f = f + (1 - u / 0.08) * 0.7
        return f, 0

    # 4) こちらの 丸盾を 3枚 ぶちぬいて、槍が 目の前に
    def sc_fly(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_FLY; e = clamp((t - T_FLY) / (T_NEAR - T_FLY)); C = (W / 2, H * 0.40 + H * 0.08 * e)
        br = clamp((t - HITS[0]) / (HITS[2] - HITS[0])) * 0.7
        f = s.storm(t, br); f = s.ground(f, br)
        s.rushdust(f, t, C, 1300, 0.7 + 1.8 * e, ICE * 0.6 + 0.4, 0.5 + 0.6 * e, 9, 1.0)
        if e < 0.55: s.hero(f, t, lean=-8 * k); s.crest(f, 0.6)
        S = (60 + 2340 * e ** 2.6) * k; s.lance(f, C, S, t)
        f = s.godrays(f, C, 0.25 + 0.35 * e, 0.8, 8, 0.86, 0.4)
        sh = 2 + 6 * e; fl = 0.0
        for j in range(3):                                                           # 盾: 下から かまえる → すきまから 光 → 6つに 割れて 外へ
            up = t - UPS[j]; v = t - HITS[j]
            if up > 0 and v < 0:
                rise = back(up / 0.14, 1.4); s.guard_at(f, j, H * 1.2 * (1 - rise), clamp((v + 0.22) / 0.22))
                if up < 0.2: sh = max(sh, 6 * math.exp(-up * 14))
            elif 0 <= v < 0.7:
                s.shards(f, j, v)
                if v < 0.25: s.lance(f, C, S, t)                                     # 割れ目から 槍が のぞく
                fl = max(fl, (1 - v / 0.09) * (0.6 + 0.2 * j) if v < 0.09 else 0); sh = max(sh, (10 + 6 * j) * math.exp(-v * 7))
        f = f + fl
        if u < 0.08: f = f + (1 - u / 0.08) * 0.3
        return f, sh


if __name__ == "__main__":
    r = Val()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
