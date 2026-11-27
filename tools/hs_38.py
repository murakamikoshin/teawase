"""フェンリス(★5 鎖の狼) 専用の必殺技「鎖断チノ咆哮」— 高品質版(hs_fx)・通し直し。敵は さいしょから 画面の こちら側
オーロラの 夜の雪原(氷のとげ・ふぶき)。両うでの鎖が 画面の外へ ぴんと 張っている → ガシャン、ガシャン。四方から 鎖が ふえて しばられ、霜が 鎖を つたって せまる(前ふり: この鎖と霜が 武器に なる) →
鎖の輪の形の 窓のカットイン → 遠ぼえ。声の輪が とどいた所から 鎖が くだけ、こおった 輪の かけらと 氷の つぶが 宙に 止まる(ため) →
うでを ふる = 合図。かけらが いっせいに こちらへ 流れ、その まん中を 本人が つめを 立てて とんでくる(逆光の影・青い目)→ 三本の つめの 光が 画面を さく → 白 → 暗転
使い方: python3 hs_38.py 出力.mp4 / python3 hs_38.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old38", "/home/claude/tools/pre_fin/hs_38_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = 9.2
ICE, IRON, FRO, CLK, ANCH = old.ICE, old.IRON, old.FRO, old.CLK, old.ANCH
T_FROST, T_CUT, T_HOWL = old.T_FROST, old.T_CUT, old.T_HOWL
T_GO, T_LEAP, T_CLAW = 7.2, 7.5, 8.28
AUR = [(0.02, 0.20, 0.16), (0.05, 0.30, 0.45), (0.45, 0.95, 0.80)]


class Fenris(FX, old.Fenris):
    def __init__(s, W=540):
        old.Fenris.__init__(s, W); s.fx_init(38); H, k = s.H, s.k
        s.srim = np.clip(s.spk - cv2.warpAffine(s.spk, np.float32([[1, 0, 3 * k], [0, 1, 2 * k]]), (W, H)), 0, 1)
        s.sfac = 0.55 + 0.9 * s.fbm(0.012, 4, 9, 3, 0, 1) * (0.6 + 0.4 * np.sign(np.sin(s.XX / (23 * k) + s.YY / (51 * k))))
        s.snow = 0.75 + 0.5 * s.fbm(0.006, 2, 5, 4, 0, 2)
        s.VC = (W / 2, H * 0.42)

    # ---- 雪原 ----
    def field(s, t, dark=0.0):
        H, W, k = s.H, s.W, s.k
        f = (np.array([0.010, 0.022, 0.07], np.float32)[None, None, :] * (1 - s.yn)[:, :, None] + np.array([0.05, 0.10, 0.22], np.float32)[None, None, :] * s.yn[:, :, None]).astype(np.float32)
        s.starfield(f, t, 800, (0.85, 0.92, 1.0), 2, 0.9, (1.0, 0.0))
        rib = cv2.remap(s._tiles[2], ((s.XX * 0.35 + t * 5 + 40 * np.sin(s.YY / (120 * k) + t * 0.5)) % 256).astype(np.float32), ((s.YY * 0.05 + t * 2) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        au = sstep(0.50, 0.80, rib) * sstep(0.02, 0.18, s.yn) * sstep(0.62, 0.30, s.yn) * (0.6 + 0.4 * s.fbm(0.002, t * 10, 0, 2, 0, 3))
        f += (au * 0.55 * (1 - 0.5 * dark))[:, :, None] * (np.array([0.20, 0.95, 0.70], np.float32)[None, None, :] * (1 - s.yn * 1.4).clip(0, 1)[:, :, None] + np.array([0.35, 0.45, 1.0], np.float32)[None, None, :] * (s.yn * 1.4).clip(0, 1)[:, :, None])
        sp = s.spk[:, :, None]; f = f * (1 - sp * 0.9) + (s.spk * s.sfac * (0.25 + 0.75 * s.yn))[:, :, None] * np.array([0.10, 0.20, 0.34], np.float32)[None, None, :]
        f += (cv2.GaussianBlur(s.srim, (0, 0), 0.8 * k) * 0.5)[:, :, None] * ICE[None, None, :]
        g = sstep(s.flr - 2 * k, s.flr + 2 * k, s.YY); dep = np.clip((s.YY - s.flr) / (H - s.flr), 0, 1)
        gr = (np.array([0.30, 0.40, 0.58], np.float32)[None, None, :] * (s.snow * (0.75 + 0.45 * dep))[:, :, None]); f = f * (1 - g[:, :, None]) + gr * g[:, :, None]
        s.starfield(f, t, 520, FRO, 5, 1.1, (-260.0, 190.0)); s.starfield(f, t + 0.012, 520, FRO, 5, 0.6, (-260.0, 190.0)); s.starfield(f, t + 0.024, 520, FRO, 5, 0.3, (-260.0, 190.0))      # ふぶき(尾を ひく)
        mist = s.fbm(0.0018, -t * 40, 0, 3, 0, 0) * np.exp(-((s.YY - s.flr) / (H * 0.06)) ** 2); f += (mist * 0.22)[:, :, None] * FRO[None, None, :]
        return f * (1 - 0.45 * dark)

    def link(s, f, p, ang, ll, col, flat, th=1.0, a0=0, a1=360):
        """鎖の輪 1つ(ふち・地・てり の 3度ぬり)"""
        k = s.k; c3 = tuple(float(v) for v in col); hi = tuple(float(v) for v in np.clip(np.asarray(col) * 1.55 + 0.18, 0, 1.4)); dk = (0.012, 0.018, 0.035); x, y = int(p[0]), int(p[1])
        if not flat:
            ax = (max(1, int(ll * 0.5)), max(1, int(ll * 0.30)))
            cv2.ellipse(f, (x, y), ax, ang, a0, a1, dk, max(1, int(13 * k * th)), cv2.LINE_AA); cv2.ellipse(f, (x, y), ax, ang, a0, a1, c3, max(1, int(8 * k * th)), cv2.LINE_AA)
            cv2.ellipse(f, (int(x - 1.2 * k * th), int(y - 1.8 * k * th)), ax, ang, max(a0, 185), min(a1, 300), hi, max(1, int(2.4 * k * th)), cv2.LINE_AA)
        else:
            d = (math.cos(math.radians(ang)) * ll * 0.5, math.sin(math.radians(ang)) * ll * 0.5); e0 = (int(p[0] - d[0]), int(p[1] - d[1])); e1 = (int(p[0] + d[0]), int(p[1] + d[1]))
            cv2.line(f, e0, e1, dk, max(1, int(14 * k * th)), cv2.LINE_AA); cv2.line(f, e0, e1, c3, max(1, int(9 * k * th)), cv2.LINE_AA)
            cv2.line(f, (e0[0], int(e0[1] - 2.2 * k * th)), (e1[0], int(e1[1] - 2.2 * k * th)), hi, max(1, int(2.4 * k * th)), cv2.LINE_AA)

    def geom(s, i):
        W, H = s.W, s.H; a = np.array([ANCH[i][0] * W, ANCH[i][1] * H], np.float32); b = np.array(s.at(*s.att[i]), np.float32)
        d = b - a; L = float(np.linalg.norm(d)); d /= L; ll = 44 * s.k; return a, d, L, ll, int(L / (ll * 0.72)), math.degrees(math.atan2(d[1], d[0]))

    def chain(s, f, i, t, grow=1.0, frost=0.0, vib=0.0, howl=None):
        W, H, k = s.W, s.H, s.k; a, d, L, ll, n, ang = s.geom(i); nn = np.array([-d[1], d[0]], np.float32); gx_, gy_ = [], []
        for j in range(n):
            u = (j + 0.5) / n
            if u > grow: break
            p = a + d * L * u + nn * math.sin(u * 3.1416) * math.sin(u * 14 + t * 60) * vib
            if howl is not None and t - s.tshat(p) > 0: continue
            fr = clamp((frost - u) / 0.06); s.link(f, p, ang, ll, IRON * (1 - fr) + FRO * fr, j % 2 == 1)
            if fr > 0.5: gx_.append(p[0]); gy_.append(p[1])
        if gx_:                                                                      # 霜の きらめき
            r = s.rs[:len(gx_) * 3]; s.splat(f, np.repeat(gx_, 3) + (r[:, 0] - 0.5) * 30 * k, np.repeat(gy_, 3) + (r[:, 1] - 0.5) * 30 * k, (0.5 + 0.5 * np.sin(t * 9 + r[:, 2] * 40)) * 1.2, FRO, 0.9, 0.5)

    def tshat(s, p):
        m = s.at(*s.mth); return T_HOWL + 0.2 + math.hypot(p[0] - m[0], p[1] - m[1]) / (s.W * 1.5)

    def debris(s, f, t):
        """くだけた 輪の かけら: とびちる → 宙に 止まる → 合図で こちらへ"""
        W, H, k = s.W, s.H, s.k; m = s.at(*s.mth); C = s.VC; gl = []
        for i in range(6):
            a, d, L, ll, n, ang = s.geom(i)
            for j in range(n):
                p = a + d * L * (j + 0.5) / n; v = t - s.tshat(p)
                if v <= 0: continue
                r = s.rs[(i * 31 + j) % 200]; hd = math.hypot(p[0] - m[0], p[1] - m[1]) + 1e-3; o = ((p[0] - m[0]) / hd, (p[1] - m[1]) / hd); dr = 1 - math.exp(-v * 4)
                q = (p[0] + o[0] * W * 0.22 * dr * (0.5 + r[0]), p[1] + o[1] * W * 0.22 * dr * (0.5 + r[0]) - 26 * k * dr); ro = ang + (r[1] - 0.5) * 300 * dr + math.sin(t * 1.3 + r[2] * 9) * 6
                e = clamp((t - T_GO - r[3] * 0.85) / 0.60) ** 2.2; z = 1 + 9 * e
                q = (C[0] + (q[0] - C[0]) * z, C[1] + (q[1] - C[1]) * z)
                if not (-90 * k < q[0] < W + 90 * k and -90 * k < q[1] < H + 90 * k): continue
                s.link(f, q, ro + e * 200 * (r[2] - 0.5), ll * (1 + 3.5 * e), (IRON * 0.55 + FRO * 0.45) * (0.8 + 0.3 * r[0]), False, 1 + 3.0 * e, 20, 200)
                if r[2] > 0.7: gl.append((q, 0.5 + 0.5 * math.sin(t * 7 + r[1] * 30)))
        for q, v in gl[:14]: s.flare(f, q, 4 * k, FRO, 0.6 * v, 0.6, 0.0, False)

    def hero(s, f, t, shade=0.92, **kw):
        img = s.pose(t, **kw); s.draw_lit(f, img, s.CX, s.CY, s.HF, rim=ICE * 0.7 + 0.3, rim_dir=(-0.4, -0.9), rim_v=0.85, amb=(0.75, 0.9, 1.0), amb_v=0.14, shade=shade)
        L = s.L0(); cv2.ellipse(L, (int(s.CX), int(s.flr + 3 * s.k)), (int(s.W * 0.20), int(8 * s.k)), 0, 0, 360, 1.0, -1, cv2.LINE_AA); 

    def eyes(s, f, v):
        if v <= 0.01: return
        sc = s._pl[2]
        for e in (s.eL, s.eR): p = s.at(*e); s.flare(f, p, 5 * sc * 2, ICE, 0.5 * min(1.6, v), 0.8 + 0.8 * max(0.0, v - 1), 0.0, False)

    def breath(s, f, t, v=1.0):
        p = s.at(*s.mth); u = (t * 0.6) % 1.0; c = (p[0] - 30 * s.k * u - 10 * s.k, p[1] - 26 * s.k * u); r = (10 + 26 * u) * s.k
        m = np.exp(-(((s.XX - c[0]) / r) ** 2 + ((s.YY - c[1]) / r) ** 2)) * s.fbm(0.01, t * 30, t * 50, 2, 0, 1); f += (m * 0.9 * v * math.sin(u * 3.1416))[:, :, None] * FRO[None, None, :]

    def lsil(s, f, img, cx, cy, sc, col, a=1.0, dark=0.0, rimv=1.0):
        ph, pw = img.shape[:2]; M = np.float32([[sc, 0, cx - pw / 2 * sc], [0, sc, cy - ph / 2 * sc]]); col = np.asarray(col, np.float32)
        al = cv2.warpAffine(img[:, :, 3], M, (s.W, s.H), flags=cv2.INTER_LINEAR)
        if sc > 1.0: d = 0.45 / sc; al = sstep(0.5 - d, 0.5 + d, al)
        rgb = cv2.warpAffine(img[:, :, :3], M, (s.W, s.H), flags=cv2.INTER_LINEAR)
        body = rgb * (1 - dark) + col[None, None, :] * (0.02 * dark) + rgb * col[None, None, :] * 0.16 * dark
        m = (al * a)[:, :, None]; f[:] = f * (1 - m) + body * m
        r = max(2.0, min(14.0 * s.k, 3.0 * s.k * sc)); edge = np.clip(al - cv2.GaussianBlur(al, (0, 0), r), 0, 1) * 2.2
        f += ((edge + cv2.GaussianBlur(edge, (0, 0), 6 * s.k) * 0.8) * a * rimv)[:, :, None] * (col * 0.8 + 0.2)[None, None, :]

    def frame(s, t):
        s._t = t; o = {}
        if t < T_CUT: f, sh = s.sc_bind(t)
        elif t < T_HOWL: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.4, thr=0.9)
        else: f, sh = s.sc_howl(t)
        f, sh = s.finale(f, sh, t, DUR, s.mth, (0.66, 0.88, 1), (1, 1, 1), -1)
        return s.film(f, t, sh * s.k, 0.55, o.get("thr", 0.84), (0.8, 0.92, 1.0), 1.0 + 0.2 * sh, 0.035, o.get("vig", 0.6), 1.12, 1.06)

    # 1) しばられる
    def sc_bind(s, t):
        W, H, k = s.W, s.H, s.k
        nc = sum(1 for c in CLK if t >= c); last = max([c for c in CLK if t >= c] + [-9]); v = t - last
        f = s.field(t, clamp((t - 0.8) / 1.5))
        s._pose = dict(t=t, sway=0, breath=0, lean=0, squash=0, stretch=None); s._pl = (s.CX, s.CY, s.HF * H / s.ah)
        fr = 0.9 * ease2((t - T_FROST) / (T_CUT - T_FROST - 0.1)) if t > T_FROST else 0.0
        for i in (2, 3, 4):
            if i - 2 < nc and t >= CLK[min(i - 2, 3)]: s.chain(f, i, t, ease((t - CLK[i - 2]) / 0.10), fr, 5 * k * math.exp(-(t - CLK[i - 2]) * 7))
        s.hero(f, t, jit=1.6 * k * math.exp(-v * 12) if v < 0.5 else 0)
        for i in (0, 1): s.chain(f, i, t, 1.0, fr, 3 * k * math.exp(-v * 7) if v < 1 else 0)
        if nc >= 4: s.chain(f, 5, t, ease((t - CLK[3]) / 0.10), fr, 5 * k * math.exp(-(t - CLK[3]) * 7))
        s.breath(f, t, 0.8)
        if fr > 0.5: s.eyes(f, (fr - 0.5) * 2)
        if 0 <= v < 0.06: f = f + (1 - v / 0.06) * 0.25
        return f * clamp(t / 0.4), 5 * math.exp(-v * 12) if v < 0.5 else 0

    # 2) 鎖の輪の形の 窓(鉄の まるみ・霜)
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.field(t, 0.2) * 0.35
        g = back(u / 0.3, 1.3); C = (W / 2, H * 0.38); hw, hh = W * 0.43 * g, H * 0.17 * g
        qx = np.maximum(np.abs(s.XX - C[0]) - (hw - hh), 0); qy = s.YY - C[1]; d = np.sqrt(qx * qx + qy * qy) - hh; th = 13 * k
        m = sstep(-th + 1, -th - 1, d); band = sstep(th + 1, th - 1, np.abs(d))
        f *= (1 - 0.6 * np.exp(-np.clip(d, 0, None) / (40 * k)))[:, :, None]
        win = np.zeros((H, W, 3), np.float32) + np.array([0.04, 0.09, 0.20], np.float32) + s.nebula(t, [(0.05, 0.16, 0.36), (0.10, 0.40, 0.45), (0.7, 0.9, 1.0)], 0.0014, (20.0, -6.0), 0.9)
        s.starfield(win, t, 400, FRO, 6, 1.0, (-200.0, 140.0)); s.starfield(win, t + 0.015, 400, FRO, 6, 0.5, (-200.0, 140.0))
        zf = 1.10 + 0.05 * ease(u / 1.8); sc = zf * H / s.ah; img = s.pose(t)
        s.draw_lit(win, img, C[0] - (s.fc[0] - s.aw / 2) * sc, C[1] + H * 0.01 - (s.fc[1] - s.ah / 2) * sc, zf, rim=ICE * 0.6 + 0.4, rim_dir=(-0.5, -0.85), rim_v=0.8, amb=(0.8, 0.9, 1.0), amb_v=0.12, shade=0.97)
        s.eyes(win, 0.9 + 1.2 * math.exp(-max(0.0, u - 0.45) * 5) * (u > 0.45))
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        p = np.clip((d + th) / (2 * th), 0, 1); ang = np.arctan2(qy, np.sign(s.XX - C[0]) * qx + 1e-3)                                 # 鉄の輪
        met = (0.20 + 0.50 * (np.cos(ang + 2.3) * 0.5 + 0.5) ** 2) * (0.40 + 0.85 * np.sin(p * 3.1416)) * (0.8 + 0.4 * s.fbm(0.006, 3, 3, 2, 0, 3)) + 0.5 * np.exp(-((p - 0.32) / 0.07) ** 2) * (np.cos(ang + 2.3) > 0.2)
        frost = sstep(0.45, 0.75, s.fbm(0.003, 1, 7, 3, 0, 0)) * clamp(u / 1.2)
        ring = met[:, :, None] * np.array([0.78, 0.84, 1.0], np.float32)[None, None, :] * (1 - 0.6 * frost)[:, :, None] + (frost * 0.75)[:, :, None] * FRO[None, None, :]
        f = f * (1 - band[:, :, None]) + ring * band[:, :, None]
        cr = (u - 0.9) / 0.5                                                         # 輪に ひびが 入る
        if cr > 0:
            pp = (C[0] + hw * 0.55, C[1] - hh - th); P = [pp]
            for j, (dx, dy) in enumerate(((0, 8), (10, 14), (-4, 20), (9, 27))):
                if cr > j * 0.2: P.append((pp[0] + dx * k, pp[1] + dy * k))
            if len(P) > 1: s.glowline(f, P, ICE * 0.6 + 0.4, 1.4, 1.0, 0.9); s.flare(f, P[-1], 5 * k, WHITE, 0.7, 0.8, 0.0, False)
        f *= (1 - 0.7 * sstep(0.68, 0.77, s.yn) * sstep(0.95, 0.89, s.yn))[:, :, None]
        n = len(s.ht); step = W * 0.148
        for i, T in enumerate(s.ht):                                                 # 技名: ガシャンと 1字ずつ 落ちて 止まる
            a = (u - 0.5 - i * 0.09) / 0.12
            if a > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.82 - (1 - min(1.0, a)) ** 2 * 80 * k, clamp(a * 2), 1.0)
        for i, T in enumerate(s.vo):
            if u > 0.4 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.745, 1.0, 1.0)
        if u > 1.25: s.text(f, s.txt_en, W / 2, H * 0.82 + 56 * k, ease((u - 1.25) / 0.3), 1.0)
        if u < 0.08: f = f + (1 - u / 0.08) * 0.7
        return f, 0

    # 3) 遠ぼえ → 鎖が くだけて 宙に 止まる → 合図 → かけらと いっしょに こちらへ → つめ
    def sc_howl(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_HOWL; b = t - T_GO; lp = clamp((t - T_LEAP) / 0.85); e = lp ** 2.2
        f = s.field(t, 0.5 * (1 - clamp(u / 1.5))) * (1 - 0.35 * e)
        s._pose = dict(t=t, sway=0, breath=0, lean=0, squash=0, stretch=None); s._pl = (s.CX, s.CY, s.HF * H / s.ah)
        mc = s.at(*s.mth)
        for j in range(4):                                                           # 声の輪
            v = (u - 0.08 - j * 0.28) / 1.1
            if 0 < v < 1:
                R = (20 + 900 * v ** 0.8) * k; P = [(mc[0] + math.cos(q * 0.0524) * R, mc[1] + math.sin(q * 0.0524) * R) for q in range(121)]
                s.glowline(f, P, ICE * 0.7 + 0.3, 1.5 + 3.0 * v, 1.0 * (1 - v) ** 1.2, 1.4, True)
        for i in (2, 3, 4): s.chain(f, i, t, 1.0, 0.9, 2 * k * clamp(1 - u), True)
        hl = clamp(u / 0.15) * clamp((1.9 - u) / 0.3)
        if lp <= 0:
            cr = clamp(b / 0.3) if b > 0 else 0.0; pre = clamp((t - (T_GO - 0.35)) / 0.35)
            s.hero(f, t, 0.92, lean=(-10 * k * hl if b < 0 else 0) + (10 * k * pre if b < 0 else -14 * k * math.exp(-b * 6)), squash=(-0.03 * hl if b < 0 else 0.07 * cr), jit=0.8 * k * hl)
            s.eyes(f, 1.0 + 0.6 * hl); s.breath(f, t, 1.0 + 2.0 * hl)
        for i in (0, 1, 5): s.chain(f, i, t, 1.0, 0.9, 2 * k * clamp(1 - u), True)
        s.debris(f, t)
        r = s.rs[:200]; v0 = np.clip((u - 0.3 - r[:, 3] * 0.9) * 3, 0, 1); ee = np.clip((t - T_GO - r[:, 2] * 0.85) / 0.6, 0, 1) ** 2.2                   # 氷の つぶ(同じく 止まって → こちらへ)
        px = W * r[:, 0]; py = H * (0.08 + 0.8 * r[:, 1]) - 20 * k * v0; px = s.VC[0] + (px - s.VC[0]) * (1 + 9 * ee); py = s.VC[1] + (py - s.VC[1]) * (1 + 9 * ee)
        s.splat(f, px, py, v0 * (0.6 + 0.6 * np.sin(t * 6 + r[:, 2] * 50) ** 2) * (1 + 2 * ee), FRO, 1.0 + 1.5 * ee.mean(), 0.6)
        sh = 4 * hl
        if b > 0:
            s.rushdust(f, t, s.VC, 1400, 1.0 + 1.6 * clamp(b), FRO, 0.4 + 0.7 * clamp(b), 9, 1.0)
            if b < 0.08: f = f + (1 - b / 0.08) * 0.45
            sh = max(sh, 3 + 3 * clamp(b))
        if lp > 0:                                                                   # とんでくる: 逆光の影・青い目
            hf = s.HF + 3.3 * e; sc = hf * H / s.ah; img = s.pose(t, lean=-10 * k)
            tx, ty = (s.fc[0] - s.aw / 2) * sc, (s.fc[1] - s.ah / 2) * sc; q = clamp(lp * 1.4)
            cx = s.CX + (W / 2 - tx - s.CX) * q; cy = s.CY + (H * 0.40 - ty - s.CY) * q - H * 0.10 * math.sin(lp * 3.1416) * (1 - e)
            s.lsil(f, img, cx, cy, sc, ICE, 1.0, clamp(lp * 2.2), 0.8 + 1.2 * e); s._pl = (cx, cy, sc); s._pose = dict(t=t, sway=0, breath=0, lean=-10 * k, squash=0, stretch=None)
            for ey_ in (s.eL, s.eR): p = s.at(*ey_); s.flare(f, p, min(22 * k, 7 * sc * (0.6 + e)), ICE * 0.7 + 0.3, 1.0, 2.2, 0.0, False)
            sh = max(sh, 5 + 9 * e)
        cl = t - T_CLAW
        if cl > 0:                                                                   # 三本の つめ
            for j in range(3):
                v = clamp((cl - j * 0.05) / 0.10)
                if v <= 0: continue
                a0 = (W * (1.10 - 0.22 * j), H * (-0.04 + 0.06 * j)); a1 = (W * (0.34 - 0.24 * j), H * (1.04 + 0.02 * j)); bend = W * 0.20; M_ = s.L0(); P1, P2 = [], []
                for q in range(25):
                    w_ = q / 24.0 * v; x = a0[0] + (a1[0] - a0[0]) * w_ - bend * math.sin(w_ * 3.1416); y = a0[1] + (a1[1] - a0[1]) * w_ + bend * 0.15 * math.sin(w_ * 3.1416)
                    hw_ = (5 + 30 * max(0.0, math.sin(q / 24.0 * 3.1416)) ** 0.7) * k; P1.append((x - hw_, y - hw_ * 0.3)); P2.append((x + hw_, y + hw_ * 0.3))
                cv2.fillPoly(M_, [np.array(P1 + P2[::-1], np.int32)], 1.0, cv2.LINE_AA)
                f[:] = f * (1 - M_[:, :, None]) + M_[:, :, None] * 1.6; s.glowmask(f, M_, ICE, 1.2, 0.0, 1.3, 22.0)
            f += 0.35 * math.exp(-cl * 10) * (cl < 0.15); sh = max(sh, 16 * math.exp(-cl * 5))
        return f, sh

    # ---- この子だけの曲 ----
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
        for i in range(10): a.note(3200 + 2600 * rg.rand(), T_HOWL + 1.9 + i * 0.05 + 0.03 * rg.rand(), 0.25, 0.022, "bell", 0.0005)                # 宙に 止まる: かすかな きらめき
        clank(T_GO, 0.6); a.whoosh(T_GO - 0.15, T_GO + 0.1, 0.35, 0.3); a.boom(T_GO, 48, 1.0, 0.3)                                 # 合図(はじめの ガシャン と 同じ音)
        tk = T_GO + 0.06; i = 0                                                                                                 # かけらが 流れる: ちりちりが つまり、太鼓が 速く
        while tk < T_CLAW:
            a.note(1800 + 3600 * rg.rand(), tk, 0.05, 0.03 + 0.003 * i, "bell", 0.0005); a.hat(tk, 0.05, 0.008)
            if i % 3 == 0: a.boom(tk, 70 - min(i, 20), 0.35 + 0.02 * i, 0.12)
            tk += max(0.022, 0.07 - i * 0.0018); i += 1
        a.rise(T_GO, T_CLAW, 200, 1300, 0.10); a.a += (a.nz - a.lp(a.nz, 0.8)) * np.clip((a.t - T_GO) / (T_CLAW - T_GO), 0, 1) ** 2 * (a.t < T_CLAW) * 0.10
        howl(T_LEAP, T_CLAW - T_LEAP + 0.1, 0.13, 320, 640); howl(T_LEAP + 0.02, T_CLAW - T_LEAP + 0.1, 0.06, 160, 320)              # とびながら ほえる
        for j in range(3): t0 = T_CLAW + j * 0.05; a.rise(t0, t0 + 0.09, 3000 - j * 300, 900, 0.18); a.hat(t0 + 0.03, 0.55, 0.06); a.boom(t0 + 0.08, 60 - j * 6, 1.0, 0.25)
        choir(0, T_CLAW, 0.6, 0.05); choir(7, T_CLAW, 0.6, 0.04)
        s.fin_snd(a, DUR, 38); a.save(path)


if __name__ == "__main__":
    r = Fenris()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
