"""ノア(★6 方舟の旅人) 専用の必殺技「方舟ノ大洪水」— 高品質版(hs_fx)・通し直し。山のような 大波が 頭の上から くずれて こちらを のむ
雨の夜の 海。小舟の上、ランタンの 明かりだけが この子を てらす。水かさが じわじわ 上がる(前ふり) → 肩の ハトが 飛び立つ → 木の板が 上下から 閉じる カットイン(すき間に 目) →
水かさが 上がり、小舟が 大きな 方舟に なって 窓に 明かりが 2つずつ ともる → 沖の 水平線が もり上がり、山のような 水の 壁に なって せまる →
波がしらが 頭の上を こえ、白い 泡の 天井が 上から くずれ落ちて 目の前を おおう → 白 → 暗転
使い方: python3 hs_73.py 出力.mp4 / python3 hs_73.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old73", "/home/claude/tools/pre_fin/hs_73_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = 10.0
TEAL, DEEP, LAMP, WOOD, FOAM = old.TEAL, old.DEEP, old.LAMP, old.WOOD, old.FOAM
T_DOVE, T_CUT, T_FLOOD = old.T_DOVE, old.T_CUT, old.T_FLOOD
T_WAVE, T_CRASH = 7.4, 9.3


class Noah(FX, old.Noah):
    def __init__(s, W=540):
        old.Noah.__init__(s, W); s.fx_init(73); s.yv = (s.YY / s.H).astype(np.float32); s._lampx = None; s._lv = 0.6

    # ---- 雨の夜の 空 ----
    def night(s, t, lum=1.0, dawn=0.0):
        W, H, k = s.W, s.H, s.k; n = s.fbm(0.0008, t * 30, t * 4, 4, 90 * k, 0); rel = np.clip((n - np.roll(n, int(26 * k), axis=0)) * 2.0, -0.3, 0.4)
        f = DEEP[None, None, :] * (0.35 + 1.3 * s.yv)[:, :, None] * (0.55 + 0.9 * n)[:, :, None] + (np.clip(rel, 0, 1) * 0.10)[:, :, None] * np.array([0.45, 0.62, 0.72], np.float32)[None, None, :]
        f += (TEAL * 0.20)[None, None, :] * (s.yv ** 2)[:, :, None]
        return (f * lum).astype(np.float32)

    def rain(s, f, t, amt=1.0):
        g = np.zeros_like(f); old.Noah.rain(s, g, t, amt); old.Noah.rain(s, g, t * 0.8 + 3.1, amt * 0.7)
        f += g * 0.55 + cv2.GaussianBlur(g, (0, 0), 2.5 * s.k) * 0.5

    def sea(s, f, level, t, front=None, lampv=None):
        """奥へ つづく 海(波の てり・ランタンの うつりこみ)。front: 手前の 波の 線(x ごとの y)"""
        W, H, k = s.W, s.H, s.k; y0 = int(max(0, min(H - 2, (level if front is None else float(front.min())) - 8 * k))); X = s.XX[y0:]; Y = s.YY[y0:]
        dz = (H * 0.5) / (np.clip(Y - level, 0, None) + 7 * k); u1 = (X - W / 2) * dz * 0.10 / k + t * 6; v1 = dz * 18 + t * 14
        w1 = cv2.remap(s._tiles[1], (u1 % 256).astype(np.float32), (v1 % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        w2 = cv2.remap(s._tiles[3], ((u1 * 2.3 + 50) % 256).astype(np.float32), ((v1 * 2.1 - t * 20) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        near = np.clip((Y - level) / (H - level + 1), 0, 1)
        w3 = cv2.remap(s._tiles[0], ((X * 1.1 / k + 30 * np.sin(Y * 0.02 + t)) % 256).astype(np.float32), ((Y * 3.2 / k - t * 46) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        wv = (w1 * 0.6 + w2 * 0.4) * (1 - 0.55 * near) + w3 * 0.55 * near; spec = sstep(0.62, 0.86, wv)
        col = DEEP[None, None, :] * (0.45 + 1.0 * wv)[:, :, None] + TEAL[None, None, :] * ((0.08 + 0.40 * sstep(0.5, 0.85, wv)) * (0.35 + 0.65 * near))[:, :, None] + (spec * 0.07)[:, :, None] * FOAM[None, None, :]
        lv = s._lv if lampv is None else lampv
        if s._lampx is not None and lv > 0:
            col += ((spec * 1.3 + 0.10) * np.exp(-((X - s._lampx) / (W * (0.04 + 0.22 * near))) ** 2) * lv * (0.35 + 0.65 * near))[:, :, None] * LAMP[None, None, :]
        col += (np.exp(-np.clip(Y - level, 0, None) / (9 * k)) * 0.10)[:, :, None] * np.array([0.5, 0.75, 0.8], np.float32)[None, None, :]
        ys = np.full(W, level, np.float32) if front is None else front.astype(np.float32); m = sstep(-1.0, 1.5, Y - ys[None, :])
        F = f[y0:]; F[:] = F * (1 - m[:, :, None]) + col * m[:, :, None]
        if front is not None: F += (np.exp(-((Y - ys[None, :]) / (2.6 * k)) ** 2) * (0.25 + 0.6 * w2))[:, :, None] * FOAM[None, None, :] * 0.6

    # ---- 木の 舟 ----
    def wood(s, f, poly, rowh, vertical=False, lum=1.0, warm=0.3):
        W, H, k = s.W, s.H, s.k; P = np.array(poly, np.float32); x0 = int(max(0, P[:, 0].min() - 3)); x1 = int(min(W, P[:, 0].max() + 3)); y0 = int(max(0, P[:, 1].min() - 3)); y1 = int(min(H, P[:, 1].max() + 3))
        if x1 - x0 < 3 or y1 - y0 < 3: return
        m = np.zeros((y1 - y0, x1 - x0), np.float32); cv2.fillPoly(m, [np.array(P - np.array([x0, y0], np.float32), np.int32)], 1.0, cv2.LINE_AA)
        X = s.XX[y0:y1, x0:x1]; Y = s.YY[y0:y1, x0:x1]; a = (X - P[:, 0].min() if vertical else Y - P[:, 1].min()) / max(1.5, rowh); fr = a - np.floor(a); seam = 0.45 + 0.55 * sstep(0.0, 0.14, fr) * sstep(1.0, 0.9, fr)
        g = cv2.remap(s._tiles[0], ((X * (3.0 if vertical else 0.4) / k) % 256).astype(np.float32), ((Y * (0.4 if vertical else 3.0) / k) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        var = s._tiles[2][(np.floor(a).astype(np.int32) * 53) % 256, 7]; d = max(1, int(2.5 * k)); edge = np.clip(m - np.roll(m, d, axis=0), 0, 1); edge[:d] = m[:d]
        col = WOOD[None, None, :] * ((0.26 + 0.60 * g) * (0.75 + 0.5 * var) * seam * lum)[:, :, None] + (edge * (0.20 + 0.5 * warm))[:, :, None] * LAMP[None, None, :]
        F = f[y0:y1, x0:x1]; F[:] = F * (1 - m[:, :, None]) + col * m[:, :, None]

    def ark(s, f, cx, wy, size, t, lit=0, glow=1.0, house=1.0):
        k = s.k * size
        if house > 0.02:
            top = wy - 46 * k - 72 * k * house
            s.wood(f, [(cx - 130 * k, wy - 46 * k), (cx - 130 * k, top), (cx + 130 * k, top), (cx + 130 * k, wy - 46 * k)], 13 * k, True, 0.95, 0.4)
            s.wood(f, [(cx - 150 * k, top), (cx, top - 52 * k * house), (cx + 150 * k, top)], 7 * k, False, 0.65, 0.6)
            Wn = s.L0()
            for j in range(8):                                                       # 窓(2つで 1組)
                if j < lit and house > 0.9:
                    for q in (-1, 1):
                        wx = cx + (j - 3.5) * 31 * k + q * 6 * k; wyy = wy - (96 if j % 2 else 70) * k
                        cv2.rectangle(Wn, (int(wx - 4 * k), int(wyy - 6 * k)), (int(wx + 4 * k), int(wyy + 6 * k)), 1.0, -1)
            s.glowmask(f, Wn, LAMP, 0.9 * glow, 1.0, 1.0, 7.0)
        s.wood(f, [(cx - 190 * k, wy - 46 * k), (cx + 190 * k, wy - 46 * k), (cx + 174 * k, wy - 8 * k), (cx + 142 * k, wy + 22 * k), (cx + 90 * k, wy + 34 * k), (cx - 90 * k, wy + 34 * k), (cx - 142 * k, wy + 22 * k), (cx - 174 * k, wy - 8 * k)], 15 * k, False, 1.0, 0.5)
        s.glowline(f, [(cx - 190 * k, wy - 46 * k), (cx + 190 * k, wy - 46 * k)], LAMP * 0.6 + WOOD * 0.6, 1.2, 0.30 * glow, 0.5)

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(LAMP), rim_dir=(-0.8, -0.5), rim_v=0.75 * alpha, amb=(0.55, 0.80, 0.90), amb_v=0.20, alpha=alpha, shade=0.80)

    def add(s, f, L, col, gain=1.0, b1=3, b2=12):
        s.glowmask(f, np.clip(L, 0, 1), col, gain * 0.8, 1.0, 1.0, float(b2))

    def lantern(s, f, t, v=1.0):
        p = s.at(*s.lamp); fl = 1 + 0.10 * math.sin(t * 13) + 0.06 * math.sin(t * 29); sc = s._pl[2] / 0.55
        s.flare(f, p, 13 * s.k * sc * fl, LAMP, 0.75 * v, 0.35, 0.0, False)
        f += (np.exp(-np.hypot(s.XX - p[0], s.YY - p[1]) / (110 * s.k * sc)) * 0.16 * v * fl)[:, :, None] * LAMP[None, None, :]
        s._lampx = p[0]; s._lv = 0.6 * v; return p

    def frame(s, t):
        o = {}
        if t < T_CUT: f, sh = s.sc_rain(t)
        elif t < T_FLOOD: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.4, thr=0.9)
        elif t < T_WAVE: f, sh = s.sc_flood(t)
        else: f, sh = s.sc_wave(t)
        f, sh = s.finale(f, sh, t, DUR, None, (0.6, 0.9, 1), (1, 0.9, 0.6), -1)
        return s.film(f, t, sh * s.k, 0.5, o.get("thr", 0.80), (0.85, 0.97, 1.0), 1.0 + 0.15 * sh, 0.045, o.get("vig", 0.6), 1.12, 1.15)

    # 1) 雨の夜。小舟の上、ランタンの 明かり。ハトが 飛び立つ
    def sc_rain(s, t):
        W, H, k = s.W, s.H, s.k; lv = H * (0.875 - 0.035 * ease2(t / T_CUT)); bob = math.sin(t * 1.7) * 4 * k
        f = s.night(t, 1.0); s._lampx = W * 0.30; s._lv = 0.6 * ease(t / 0.6); s.sea(f, lv - H * 0.05, t)
        s.ark(f, W / 2, H * 0.842 + bob, 1.30, t, 0, 1.0, 0.0)
        img = s.pose(t, breath=0.25); s.draw(f, img, W / 2, H * 0.565 + bob, 0.45); s.lantern(f, t, ease(t / 0.6))
        s.sea(f, lv, t, s.surf(lv, t, 6 * k))
        L = s.L0()
        for i in range(10):                                                          # 水面の 雨の輪
            r = s.rn[i + 50]; v = (t * 0.9 + r[0]) % 1.0; s.ell(L, (r[1] * W, lv + H * (0.03 + 0.09 * r[2])), (40 * k * v, 8 * k * v), 0, 0.5 * (1 - v), 1)
        f += (L + cv2.GaussianBlur(L, (0, 0), 2 * k))[:, :, None] * FOAM[None, None, :] * 0.4
        s.rain(f, t, 1.0)
        fly = t - T_DOVE
        if fly > 0:                                                                  # ハト: 肩から 右上へ(白い 光の 影)
            p0 = s.at(*s.dove0); a = ease2(fly / 0.8); p = (p0[0] + W * 0.55 * a, p0[1] - H * 0.52 * a - math.sin(a * 3.14) * H * 0.05)
            L = s.L0(); s.dove(L, p, t, 1.5 * k + 0.6 * k * a)
            for i in range(5):                                                       # 抜けた羽
                v = fly * (0.5 + 0.2 * i); q = (p0[0] + (i - 2) * 14 * k + math.sin(fly * 4 + i) * 12 * k, p0[1] - 20 * k + v * 70 * k)
                cv2.ellipse(L, (int(q[0]), int(q[1])), (int(7 * k), int(2 * k)), 30 * i + fly * 90, 0, 360, 0.6, -1, cv2.LINE_AA)
            s.glowmask(f, cv2.GaussianBlur(L, (0, 0), 0.8 * k), (1.0, 0.98, 0.94), 0.9, 1.0, 0.8, 6.0)
        return f * clamp(t / 0.5), 0

    # 3) 水かさが 上がる。舟が 方舟になる
    def sc_flood(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_FLOOD; rise = ease2(u / (T_WAVE - T_FLOOD)); level = H * (0.88 - 0.30 * rise); amp = (5 + 12 * rise) * k
        f = s.night(t, 0.95); s._lampx = W * 0.40; s._lv = 0.6
        hz_ = level - H * 0.06; s.sea(f, hz_, t)
        ys = s.surf(level, t, amp); wy = float(ys[W // 2]) - H * 0.02; grow = ease(u / 1.3); size = 0.40 + 0.70 * grow
        s.ark(f, W / 2, wy + 8 * k, size, t, int(clamp((u - 1.0) / 1.1) * 8.99), 1.0, ease(clamp((u - 0.25) / 0.9)))
        img = s.pose(t); cyc = wy - (118 + 170) * 0.5 * k * size - H * 0.085
        s.draw(f, img, W / 2, cyc, 0.22 + 0.04 * grow); s.lantern(f, t, 1.0)
        s.sea(f, level, t, ys + 26 * k * size * 0.6)
        s.rain(f, t, 1.4); s.rushdust(f, t, (W / 2, hz_), 300, 0.2, FOAM, 0.15, 3, 0.0)
        if u < 0.2: f = f * (u / 0.2)
        return f, 1.5 * rise

    def bigwave(s, f, t, g, level):
        """沖から もり上がる 水の 壁(画面の 座標の 式)"""
        W, H, k = s.W, s.H, s.k; x = s.xr / W
        hh = H * (0.02 + 1.25 * g ** 1.7) * (0.80 + 0.20 * np.sin(x * 3.0 + 0.9) + 0.05 * np.sin(x * 11 + t * 2)); yc = (level - hh).astype(np.float32)
        y1 = int(min(H, level + 2)); y0 = int(max(0, min(y1 - 2, float(yc.min()) - 40 * k))); X = s.XX[y0:y1]; Y = s.YY[y0:y1]; span = np.maximum(level - yc, 2.0)[None, :]; v = np.clip((Y - yc[None, :]) / span, 0, 1)
        m = sstep(-1.0, 1.5, Y - yc[None, :])
        tx = cv2.remap(s._tiles[0], ((X * 0.8 / k + 20 * np.sin(Y * 0.01)) % 256).astype(np.float32), ((Y * 0.22 / k + t * 70 * (1 + g)) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        fo = cv2.remap(s._tiles[2], ((X * 1.7 / k) % 256).astype(np.float32), ((Y * 0.5 / k + t * 130 * (1 + g)) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        thin = (1 - v) ** 1.6
        col = np.array([0.02, 0.09, 0.13], np.float32)[None, None, :] * (0.5 + 0.9 * tx)[:, :, None] + (thin * (0.5 + 0.7 * tx) * (0.35 + 0.65 * g))[:, :, None] * np.array([0.16, 0.62, 0.60], np.float32)[None, None, :]
        col += ((0.25 + 0.75 * tx) * g * 0.55)[:, :, None] * np.array([0.05, 0.30, 0.32], np.float32)[None, None, :]
        col += (sstep(0.66, 0.90, fo) * (0.15 + 0.7 * thin))[:, :, None] * FOAM[None, None, :] * 0.55
        cw = (5 + 26 * g) * k; n = s.fbm(0.004, t * 60, -t * 90, 3, 0, 1)[y0:y1]
        crest = np.exp(-((Y - yc[None, :]) / cw) ** 2) * (0.5 + 1.0 * n); col += crest[:, :, None] * FOAM[None, None, :] * 1.1
        F = f[y0:y1]; F[:] = F * (1 - m[:, :, None]) + col * m[:, :, None]
        F += (np.exp(-np.clip(yc[None, :] - Y, 0, None) / (cw * 1.6)) * (Y < yc[None, :]) * sstep(0.45, 0.8, n) * 0.6)[:, :, None] * FOAM[None, None, :]          # 波がしらの 上に まう しぶき
        rg = np.random.RandomState(int(t * 30)); nn = int(150 + 450 * g); xs = rg.rand(nn) * W; s.splat(f, xs, yc[np.clip(xs.astype(np.int32), 0, W - 1)] - rg.rand(nn) ** 2 * (20 + 110 * g) * k, rg.rand(nn) * 0.3, FOAM, 1.6, 0.6)
        return yc

    def lip(s, f, t, q):
        """頭の上を こえた 波がしらが 上から くずれ落ちる"""
        W, H, k = s.W, s.H, s.k; x = s.xr / W; yl = (H * (-0.30 + 1.65 * q ** 1.6) + H * 0.035 * np.sin(x * 5 + t * 3)).astype(np.float32)
        n = s.fbm(0.0028, t * 40, -t * 420, 4, 30 * k, 2); n2 = s.fbm(0.0030, 0, -t * 300, 2, 0, 0)
        st = cv2.remap(s._tiles[1], ((s.XX * 1.3 / k) % 256).astype(np.float32), ((s.YY * 0.22 / k - t * 330) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        d = yl[None, :] - s.YY + (n2 - 0.5) * H * 0.20 + (st - 0.5) * H * 0.10; m = sstep(-2.0, 5.0 * k, d)
        col = FOAM[None, None, :] * (0.55 + 0.55 * n + 0.45 * sstep(0.45, 0.8, st))[:, :, None]
        sh = (sstep(0.52, 0.25, n) * 0.65)[:, :, None]; col = col * (1 - sh) + np.array([0.10, 0.46, 0.50], np.float32)[None, None, :] * sh
        col += (np.exp(-(np.clip(d, 0, None) / (16 * k)) ** 2) * 0.7)[:, :, None]                                                                   # ふちの 厚い 泡
        e = (np.exp(np.clip(d, None, 0) / (80 * k)) * (d < 0) * 0.65)[:, :, None]; f[:] = f * (1 - e) + np.array([0.02, 0.12, 0.15], np.float32)[None, None, :] * e   # 波の 腹の 影
        f[:] = f * (1 - m[:, :, None]) + col * m[:, :, None]
        rg = np.random.RandomState(int(t * 30) + 7); nn = 700; xs = rg.rand(nn) * W; s.splat(f, xs, yl[np.clip(xs.astype(np.int32), 0, W - 1)] + rg.rand(nn) ** 1.5 * H * 0.25, rg.rand(nn) * 0.35, FOAM, 1.8, 0.6)

    # 4) 沖の 水平線が もり上がる → 水の 壁 → 頭の上から くずれる
    def sc_wave(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_WAVE; g = clamp(u / (T_CRASH - T_WAVE - 0.35)) ; q = clamp((u - 0.95) / (T_CRASH - T_WAVE - 0.95))
        level = H * 0.58; amp = 17 * k * (1 + 1.2 * g); f = s.night(t, 0.95 - 0.35 * g); s._lampx = W * 0.40; s._lv = 0.6
        hz_ = level - H * 0.06; s.sea(f, hz_, t); s.bigwave(f, t, ease2(g) * 0.5 + g * 0.5, hz_ + 2)
        ys = s.surf(level, t, amp); wy = float(ys[W // 2]) - H * 0.02 - H * 0.03 * g
        s.ark(f, W / 2, wy + 8 * k, 1.1, t, 8, 1.0, 1.0)
        img = s.pose(t, jit=1.0 * k * g); s.draw(f, img, W / 2, wy - 144 * k * 1.1 - H * 0.085, 0.26); s.lantern(f, t, 1.0)
        s.sea(f, level, t, ys + 26 * k * 0.66)
        s.rain(f, t, 1.6 + g)
        if q > 0: s.lip(f, t, q)
        if u < 0.15: f = f * (0.6 + 0.4 * u / 0.15)
        if q > 0.85: f = f + ((q - 0.85) / 0.15) ** 2 * 0.9
        return f, 2 + 5 * g + 12 * q ** 2

    # ---- この子だけの曲: 雨の音 と、8分の6拍子の 舟歌(ニ調ドリア)。窓が ともるたび 1音。沖の うなりが 大きく なって、頭の上から くずれる ----
    def audio(s, path):
        a = Snd(DUR, 73); D = 146.83; dor = [0, 2, 3, 5, 7, 9, 10]; maj = [0, 2, 4, 5, 7, 9, 11]
        def fq(d, o=0, sc=dor): return D * 2 ** ((sc[d % 7] + 12 * (d // 7 + o)) / 12.0)
        a.a += (a.nz - a.lp(a.nz, 0.5)) * 0.10 * a.gate(0, T_CRASH, 0.4) * (1 + 0.8 * np.clip((a.t - T_FLOOD) / 3, 0, 1))        # 雨
        a.a += a.lp(a.nz, 0.02) * 0.5 * np.clip((a.t - T_FLOOD) / 2.2, 0, 1) * a.gate(T_FLOOD, T_CRASH, 0.3)                  # 水の うなり
        for i, d in enumerate([0, 4, 7, 4]): a.note(fq(d, 0), 0.5 + i * 0.45, 0.8, 0.08, "fm")
        for i in range(6): a.hat(T_DOVE + i * 0.06, 0.07, 0.03)                                                                # はばたき
        a.note(fq(9, 1), T_DOVE, 0.6, 0.07, "bell")
        a.boom(T_CUT + 0.33, 70, 1.0, 0.18); a.a += a.lp(a.nz, 0.6) * a.env(T_CUT + 0.33, 0.001, 0.06) * 0.5                   # 板が ばたん
        for i in range(6): a.note(fq([7, 5, 4, 2, 4, 0][i], 0), T_CUT + 0.5 + i * 0.14, 0.4, 0.10, "fm"); a.note(fq(14, 1), T_CUT + 0.5 + i * 0.14, 0.08, 0.04, "sine")
        bt = 0.22; tk = T_FLOOD; i = 0; mel = [0, 2, 4, 7, 4, 2, 3, 4, 5, 7, 9, 7, 5, 4, 2, 0]                                  # 舟歌(強・弱・弱)
        while tk < T_CRASH - 0.1:
            g = 1.0 + 0.5 * clamp((tk - T_WAVE) / (T_CRASH - T_WAVE))
            if i % 3 == 0: a.boom(tk, 52, 0.45 * g, 0.16); a.note(fq([0, 0, 3, 4][(i // 6) % 4], -1), tk, 0.5, 0.14, "saw", 0.01)
            else: a.note(fq([4, 7][i % 2], 0), tk, 0.12, 0.035, "sq")
            if i % 3 != 1: a.note(fq(mel[(i * 2 // 3) % 16] + (2 if tk > T_WAVE else 0), 1), tk, 0.3, 0.085, "fm")
            tk += bt; i += 1
        for j in range(8): a.note(fq(j, 1, maj), T_FLOOD + 1.0 + j * 1.1 / 8, 0.5, 0.07, "bell")                               # 窓が 2つずつ
        a.rise(T_WAVE, T_CRASH, 45, 150, 0.28); a.whoosh(T_WAVE + 0.4, T_CRASH, 0.9, 0.3)                                       # 水の 壁が せまる
        a.a += a.lp(a.nz, 0.05) * np.clip((a.t - T_WAVE) / (T_CRASH - T_WAVE), 0, 1) ** 1.6 * (a.t < T_CRASH) * 1.3
        a.a += (a.nz - a.lp(a.nz, 0.35)) * np.clip((a.t - T_WAVE - 0.95) / (T_CRASH - T_WAVE - 0.95), 0, 1) ** 2 * (a.t < T_CRASH) * 0.45      # くずれる 泡
        a.boom(T_CRASH, 36, 1.8, 0.9); a.a += a.lp(a.nz, 0.25) * a.env(T_CRASH, 0.02, 0.5) * 0.9
        s.fin_snd(a, DUR, 73); a.save(path)


if __name__ == "__main__":
    r = Noah()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
