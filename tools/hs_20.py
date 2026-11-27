"""アカツキ(★5 夜明けの王) 専用の必殺技「夜明ケノ王蹴」— 高品質版(hs_fx)。話は v3 と 同じ: 敵は こちら側。とびげりが まっすぐ こちらへ
夜明け前。星と 雲の海(下から 朝やけ)と 山のかげ → 雲を つきやぶって 真上へ。のぼる 朝日を 背に、黒い影で 止まる →
日の出の形(半円)の窓の カットイン → 朝日ごと こちらへ 落ちてくる とびげり。近づくほど 空が 朝に なり、くつ底(ゴムの みぞ・金の ふち・日の紋)が 目の前を ふさぐ → 白 → 暗転
使い方: python3 hs_20.py 出力.mp4 / python3 hs_20.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old20", "/home/claude/tools/pre_fin/hs_20_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = old.DUR; SUNC, RED, GOLD, N_TOP, N_LOW, D_TOP, D_LOW = old.SUNC, old.RED, old.GOLD, old.N_TOP, old.N_LOW, old.D_TOP, old.D_LOW
T_LAUNCH, T_APEX, T_CUT, T_DIVE, T_HIT = old.T_LAUNCH, old.T_APEX, old.T_CUT, old.T_DIVE, old.T_HIT


class Akatsuki(FX, old.Akatsuki):
    def __init__(s, W=540):
        old.Akatsuki.__init__(s, W); s.fx_init(20); H, k = s.H, s.k
        M = s.L0(); cv2.fillPoly(M, [np.array(s.peak, np.int32)], 1.0, cv2.LINE_AA); s.mtn = M
        s.mrim = np.clip(M - cv2.warpAffine(M, np.float32([[1, 0, 0], [0, 1, 3 * k]]), (W, H)), 0, 1); s.mdet = 0.6 + 0.8 * s.fbm(0.008, 3, 9, 3, 0, 1)

    def sky(s, day):
        d = day if isinstance(day, np.ndarray) else np.full((s.H, s.W), float(day), np.float32); t = getattr(s, "_t", 0.0)
        y = s.yn[:, :, None]; n = N_TOP[None, None, :] * 0.7 * (1 - y) + N_LOW[None, None, :] * y; dd = D_TOP[None, None, :] * (1 - y) ** 1.2 + D_LOW[None, None, :] * y
        n = n + s.nebula(t, [(0.05, 0.04, 0.16), (0.14, 0.06, 0.22), (0.5, 0.3, 0.5)], 0.0008, (3.0, 0.0), 0.45)
        st = np.zeros_like(n); s.starfield(st, t, 900, (0.9, 0.93, 1.0), 2, 1.0, (0.6, 0.0)); n += st * np.clip(1.1 - s.yn * 1.5, 0, 1)[:, :, None]
        dd = dd + (sstep(0.5, 0.85, s.fbm(0.0012, t * 6, 0, 3, 0, 2)) * 0.18 * sstep(0.7, 0.2, s.yn))[:, :, None]                    # 朝の うす雲
        return (n * (1 - d[:, :, None]) + dd * d[:, :, None]).astype(np.float32), d

    def clouds(s, f, t, d, y0=0.72, n=4, drift=1.0):
        """雲の海(fbm の もこもこ・上の ふちが 朝日に そまる)"""
        k = s.k; H = s.H
        for j in range(n):
            base = H * (y0 + j * 0.07); x = s.xr + t * (10 + 8 * j) * k * drift + j * 137
            top = base - np.abs(np.sin(x / ((38 + 9 * j) * k))) * (16 + 5 * j) * k - np.abs(np.sin(x / ((97 + 13 * j) * k) + j)) * 14 * k
            tex = s.fbm(0.0035 - 0.0004 * j, t * (10 + 8 * j) * drift + j * 60, j * 40, 4, 30 * k, j % 4)
            top2 = top[None, :] - (tex - 0.5) * 46 * k; dep = np.clip((s.YY - top2) / (70 * k), 0, 1)
            m = sstep(-3 * k, 3 * k, s.YY - top2)[:, :, None]; sh = 0.55 + 0.15 * j
            ncol = np.array([0.13, 0.12, 0.28], np.float32) * sh; dcol = np.array([1.0, 0.86, 0.62], np.float32) * (0.72 + 0.09 * j)
            col = ncol[None, None, :] * (1 - d[:, :, None]) + dcol[None, None, :] * d[:, :, None]
            col = col * ((0.70 + 0.55 * tex) * (1.0 - 0.35 * dep))[:, :, None]
            col += (np.exp(-np.clip(s.YY - top2, 0, None) / (16 * k)) * (0.30 + 0.5 * d))[:, :, None] * (SUNC * 0.8 + RED * 0.2)[None, None, :]    # 上の ふちの 朝やけ
            f[:] = f * (1 - m) + col * m

    def sun(s, f, c, r, v=1.0):
        """朝日: やわらかい 円盤 + 空気の にじみ + 横に のびる 光"""
        d = np.sqrt((s.XX - c[0]) ** 2 + (s.YY - c[1]) ** 2) / r; disc = sstep(1.0, 1.0 - 3.0 / r, d)[:, :, None] * v
        col = (GOLD * 1.25 + 0.15)[None, None, :] * (1 - np.clip(d, 0, 1) ** 2.5 * 0.45)[:, :, None] + (RED * 0.35)[None, None, :] * (np.clip(d, 0, 1) ** 3)[:, :, None]
        f[:] = f * (1 - disc) + col * disc
        o = np.clip(d - 1, 0, None); f += ((np.exp(-o / 0.22) * 0.60 + np.exp(-o / 1.1) * 0.22) * (d > 1) * v)[:, :, None] * (SUNC * 0.8 + 0.1)[None, None, :]
        f += (np.exp(-((s.YY - c[1]) / (r * 0.10)) ** 2) * np.exp(-np.abs(s.XX - c[0]) / (r * 2.6)) * 0.30 * v)[:, :, None] * (GOLD * 0.8 + 0.2)[None, None, :]

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if alpha <= 0.01: s._pl = (cx, cy, hfrac * s.H / s.ah); return
        s.draw_lit(f, img, cx, cy, hfrac, rim=GOLD * 0.8 + 0.2, rim_dir=(0.0, -1.0), rim_v=0.9 * alpha, amb=(1.0, 0.85, 0.65), amb_v=0.12, alpha=alpha, shade=0.95)

    def add(s, f, L, col, gain=1.0, b1=3, b2=12):
        s.glowmask(f, np.clip(L, 0, 1), col, gain, 0.8, 1.0, float(b2))

    def emblem(s, f, v, t):
        if v <= 0: return
        p = s.at(*s.emb); sc = s._pl[2]; s.flare(f, p, 7 * sc * 2 * v, GOLD * 0.7 + 0.3, 0.8 * v, 1.4, t, False)

    def sole(s, f, c, S, rot):
        """こちらへ せまる くつ底(式で 描く: ゴムの みぞ・金の ふち・日の紋)"""
        W, H, k = s.W, s.H, s.k; x0 = int(max(0, c[0] - S * 1.3)); x1 = int(min(W, c[0] + S * 1.3)); y0 = int(max(0, c[1] - S * 1.3)); y1 = int(min(H, c[1] + S * 1.3))
        if x1 - x0 < 3 or y1 - y0 < 3: return
        ca, sa = math.cos(rot), math.sin(rot); dx = s.XX[y0:y1, x0:x1] - c[0]; dy = s.YY[y0:y1, x0:x1] - c[1]; x = (dx * ca + dy * sa) / S; y = (-dx * sa + dy * ca) / S
        toe = np.hypot(x / 0.52, (y + 0.38) / 0.66); heel = np.hypot(x / 0.40, (y - 0.62) / 0.42); wst = np.maximum(np.abs(x) / (0.42 - 0.067 * np.clip((y + 0.2) / 0.9, 0, 1)), np.maximum(-(y + 0.2) / 0.001 * 0 + (-(y + 0.2) > 0) * 9, (y - 0.7 > 0) * 9))
        dd = np.minimum(np.minimum(toe, heel), wst); aa = min(0.2, 2.5 / S)
        mo = sstep(1.12, 1.12 - aa, dd); mg = sstep(1.0, 1.0 - aa, dd); mr = sstep(0.88, 0.88 - aa, dd)
        tread = sstep(0.35, 0.75, np.sin((y + 0.22 * np.abs(x)) * 6.2832 / 0.15)) * sstep(0.80, 0.70, dd) * (1 - sstep(0.34, 0.30, np.hypot(x, y - 0.12)))
        gr = cv2.remap(s._tiles[1], ((x * 200 + 300) % 256).astype(np.float32), ((y * 200 + 300) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        lit = 0.55 + 0.6 * np.clip(0.6 - (x * 0.6 + y * 0.5), 0, 1.2)
        rub = np.array([0.20, 0.12, 0.08], np.float32)[None, None, :] * (lit * (0.8 + 0.4 * gr) * (1 - 0.72 * tread))[:, :, None]
        rub += (sstep(0.2, 0.7, np.sin((y + 0.22 * np.abs(x)) * 6.2832 / 0.15 + 0.9)) * (1 - tread) * sstep(0.80, 0.70, dd) * 0.05)[:, :, None]                # みぞの ふちの てり
        p = np.clip((dd - 0.88) / 0.12, 0, 1); gold = GOLD[None, None, :] * (0.45 + 0.85 * np.sin(p * 3.1416) * (0.6 + 0.6 * np.clip(0.5 - (x + y) * 0.5, 0, 1)))[:, :, None]
        col = gold * (1 - mr[:, :, None]) + rub * mr[:, :, None]
        ex = x; ey = y - 0.12; er = np.hypot(ex, ey); ea = np.arctan2(ey, ex)                                                          # 日の紋
        ray = sstep(0.55, 0.80, np.cos(ea * 12)) * sstep(0.125, 0.14, er) * sstep(0.27, 0.25, er) + sstep(0.092, 0.085, er)
        col = col * (1 - ray[:, :, None]) + (GOLD * (1.0 + 0.3 * np.clip(0.3 - (ex + ey), 0, 1))[:, :, None]) * ray[:, :, None]
        col = col * (1 - 0.6 * sstep(0.105, 0.098, er) * sstep(0.085, 0.092, er))[:, :, None]
        F = f[y0:y1, x0:x1]; F[:] = F * (1 - mo[:, :, None]) + np.array([0.02, 0.014, 0.012], np.float32)[None, None, :] * mo[:, :, None]
        F[:] = F * (1 - mg[:, :, None]) + col * mg[:, :, None]

    def frame(s, t):
        s._t = t; o = {}
        if t < T_CUT: f, sh = s.sc_rise(t)
        elif t < T_DIVE: f, sh = s.sc_cut(t - T_CUT); s.starfield(f, t, 300, GOLD, 5, 0.5, (0.0, -20.0)); o = dict(vig=0.45, thr=0.92)
        else:
            tt = min(t, T_HIT); a = ease2((tt - T_DIVE) / (T_HIT - T_DIVE)) ** 1.6
            f, sh = s.sc_dive(tt)
            if a < 0.75: s.rushdust(f, t, (s.W * 0.5, s.H * 0.34), 1300, 0.8 + 1.6 * a, GOLD * 0.6 + 0.4, 0.5 * (1 - a), 3, 1.0)
        f, sh = s.finale(f, sh, t, DUR, s.boot, (1, 0.6, 0.2), (1, 0.9, 0.5), -1)
        return s.film(f, t, sh * s.k, 0.55, o.get("thr", 0.84), (1.0, 0.86, 0.62), 1.0 + 0.2 * sh, 0.035, o.get("vig", 0.55), 1.12, 1.05)

    # 1) 雲を つきやぶって、朝日を 背に 止まる
    def sc_rise(s, t):
        W, H, k = s.W, s.H, s.k; la = t - T_LAUNCH; ap = t - T_APEX
        f, d = s.sky(0.0)
        hz_ = H * 0.70; glow = ease(t / 1.2)
        f += np.exp(-((s.YY - hz_) / (H * 0.10)) ** 2)[:, :, None] * (SUNC * 0.5 + RED * 0.25)[None, None, :] * glow                 # 地平の 朝やけ
        C = (W / 2, H * 0.34)
        if ap > 0:
            sc_ = (W / 2, hz_ + H * 0.10 - (hz_ + H * 0.10 - C[1]) * ease(ap / 1.1)); s.sun(f, sc_, W * 0.36 * (0.6 + 0.4 * ease(ap / 1.1)))
        M = s.mtn[:, :, None]; f = f * (1 - M) + M * (np.array([0.035, 0.03, 0.08], np.float32)[None, None, :] * s.mdet[:, :, None])
        f += (cv2.GaussianBlur(s.mrim, (0, 0), 0.8 * k) * (0.25 + 0.5 * glow))[:, :, None] * (SUNC * 0.7 + RED * 0.3)[None, None, :]
        s.clouds(f, t, d, 0.76, 4)
        if la > 0:
            a = ease(la / (T_APEX - T_LAUNCH)); cy = H * 1.05 - (H * 1.05 - C[1]) * a
            if ap < 0.5:                                                             # 上がる すじ
                v = clamp(1 - ap * 2 if ap > 0 else 1); s.glowline(f, [(W / 2, H), (W / 2, cy + H * 0.1)], GOLD * 0.6 + 0.4, 4.0, 0.8 * v, 1.4)
                r = s.rnd[:160]; s.splat(f, W / 2 + (r[:, 0] - 0.5) * 70 * k, cy + H * 0.1 + r[:, 1] * (H - cy), r[:, 2] * v * 1.4, GOLD, 1.0, 0.5)
            for i in range(3):                                                       # 雲の 輪
                v = (la - 0.15 - i * 0.1) / 0.8
                if 0 < v < 1: P = [(W / 2 + math.cos(q * 0.1309) * W * 0.75 * v, H * 0.78 + math.sin(q * 0.1309) * W * 0.10 * v) for q in range(48)]; s.glowline(f, P, GOLD * 0.6 + 0.4, 1.6, (1 - v) * 0.8, 0.8, True)
            img = s.pose(t, squash=(-0.10 * (1 - a) if ap < 0 else 0.05 * math.exp(-ap * 9) * math.cos(ap * 20)))
            s.draw(f, img, W / 2, cy, 0.42, 0.0); s.place(f, cv2.resize(s.sil, (s.pw, s.ph)), W / 2, cy, 0.42, 1.0)                   # 影のまま
            if ap > 0.75: s.draw(f, img, W / 2, cy, 0.42, ease((ap - 0.75) / 0.4)); s.emblem(f, ease((ap - 0.8) / 0.3), t)
            if ap > 0: f = s.godrays(f, (W / 2, cy), 0.35 * clamp(ap * 2), 0.8, 8, 0.86, 0.35)
        return f * clamp(t / 0.3), (7 * math.exp(-la * 6) if la > 0 else 0)


if __name__ == "__main__":
    r = Akatsuki()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
