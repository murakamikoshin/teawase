"""セイリュウ(★5 青竜の将) 専用の必殺技「蒼天雷鳴」— 高品質版(hs_fx)・通し直し。敵は さいしょから 画面の こちら側。空から 青竜の 胴と 雷が 降りてくる
まっ青な 空と 石の柱の山(岩の しま・もや)。柱の上に 立つ → 石づきを 打ちつけると 刃の 竜の彫りが 光り、青い すじが 空の 奥の 1点へ のぼって 消える(前ふり: そこが 竜の 出口) →
偃月刀の 刃の形の 窓のカットイン → その 1点から 雷雲が うずを まいて 広がり、青竜の 胴(うろこの 筒・金の 背びれ・目の 光)が らせんを えがいて こちらへ 降りてくる。
胴が 画面の ふちを まわりこんで こちらを かこんだ まん中を、いなずまが つらぬいて 落ちる → 白 → 暗転
使い方: python3 hs_67.py 出力.mp4 / python3 hs_67.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old67", "/home/claude/tools/pre_fin/hs_67_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = old.DUR; AZ, BOLT, GOLD = old.AZ, old.BOLT, old.GOLD
T_TAP, T_CUT, T_COME, T_HIT = old.T_TAP, old.T_CUT, old.T_COME, old.T_HIT


class Seiryu(FX, old.Seiryu):
    def __init__(s, W=540):
        old.Seiryu.__init__(s, W); s.fx_init(67); H, k = s.H, s.k
        st = cv2.remap(s._tiles[0], ((s.XX * 0.9) % 256).astype(np.float32), ((s.YY * 0.10) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        st2 = s.fbm(0.006, 3, 8, 3, 0, 1); s.rock = (0.55 + 0.7 * st * (0.6 + 0.8 * st2)).astype(np.float32)                             # 岩の たての しま
        sh = lambda m, dx: np.clip(m - cv2.warpAffine(m, np.float32([[1, 0, dx], [0, 1, 2 * k]]), (W, H)), 0, 1)
        s.frim = sh(s.far, 3 * k); s.nrim = sh(s.near, 4 * k)
        dx = s.XX - s.VP[0]; dy = s.YY - s.VP[1]; s.pr = np.hypot(dx, dy) / (H * 0.75); s.pa = np.arctan2(dy, dx)

    # ---- 空・山 ----
    def sky(s, t, deep=0.0, haze=1.0):
        top = np.array([0.05, 0.28, 0.86], np.float32) * (1 - deep) + np.array([0.006, 0.03, 0.16], np.float32) * deep
        low = np.array([0.62, 0.86, 1.0], np.float32) * (1 - deep) + np.array([0.06, 0.20, 0.52], np.float32) * deep
        f = (top[None, None, :] * (1 - s.yn)[:, :, None] + low[None, None, :] * s.yn[:, :, None]).astype(np.float32)
        cl = sstep(0.50, 0.80, s.fbm(0.0011, t * 12, 0, 4, 60 * s.k, 2)) * sstep(0.95, 0.35, s.yn)                                     # 雲
        f += (cl * (0.55 * (1 - deep) + 0.10 * deep))[:, :, None] * np.array([1.0, 1.0, 1.0], np.float32)[None, None, :]
        if deep > 0: f *= (1 - 0.45 * deep * np.clip(s.vr, 0, 1))[:, :, None]
        return f

    def peaks(s, f, deep=0.0):
        k = s.k; t = getattr(s, "_t", 0.0)
        c1 = np.array([0.42, 0.62, 0.88], np.float32) * (1 - 0.6 * deep); c2 = np.array([0.10, 0.20, 0.38], np.float32) * (1 - 0.5 * deep)
        f = f * (1 - s.far[:, :, None]) + (c1[None, None, :] * (0.75 + 0.35 * s.rock)[:, :, None]) * s.far[:, :, None]
        f += (cv2.GaussianBlur(s.frim, (0, 0), 0.8 * k) * 0.30 * (1 - 0.6 * deep))[:, :, None]
        mist = s.fbm(0.0018, t * 14, 0, 3, 0, 3) * np.exp(-((s.yn - 0.86) / 0.10) ** 2); f += (mist * 0.55 * (1 - 0.6 * deep))[:, :, None] * np.array([0.8, 0.92, 1.0], np.float32)[None, None, :]   # 山あいの もや
        f = f * (1 - s.near[:, :, None]) + (c2[None, None, :] * (s.rock * (0.6 + 0.6 * (1 - s.yn)))[:, :, None]) * s.near[:, :, None]
        f += (cv2.GaussianBlur(s.nrim, (0, 0), 0.8 * k) * (0.45 - 0.2 * deep))[:, :, None] * (BOLT * 0.8 + 0.2)[None, None, :]
        return f

    def rush(s, f, t, v=1.0, speed=1.0):
        s.rushdust(f, t, s.VP, 1100, speed * 1.4, BOLT, 0.8 * v, 3, 1.0)

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        s.draw_lit(f, img, cx, cy, hfrac, rim=getattr(s, "_rimc", WHITE), rim_dir=(0.0, -1.0), rim_v=0.8 * alpha, amb=(0.8, 0.9, 1.0), amb_v=0.12, alpha=alpha, shade=getattr(s, "_hsh", 0.96))

    def add(s, f, L, col, gain=1.0, b1=3, b2=12):
        s.glowmask(f, np.clip(L, 0, 1), col, gain, 0.8, 1.0, float(b2))

    # ---- 雷雲・竜の胴・いなずま ----
    def vortex(s, f, t, R, v=1.0):
        """1点から うずを まいて 広がる 雷雲"""
        if R <= 0.01: return
        sw = s.pa + 2.2 * np.log(s.pr + 0.05) - t * 0.9
        n = cv2.remap(s._tiles[1], ((sw / 6.2832 * 256 * 2) % 256).astype(np.float32), ((s.pr * 170 - t * 30) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        n2 = cv2.remap(s._tiles[3], ((sw / 6.2832 * 256 * 5) % 256).astype(np.float32), ((s.pr * 420 - t * 55) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        dens = sstep(0.30, 0.75, n * 0.7 + n2 * 0.4) * sstep(R, R * 0.55, s.pr) * v
        col = np.array([0.04, 0.09, 0.24], np.float32)[None, None, :] * (0.4 + 1.6 * n2)[:, :, None] + (np.exp(-s.pr / (0.10 + 0.25 * R)) * 0.55 * (0.6 + 0.4 * math.sin(t * 31) * math.sin(t * 17)))[:, :, None] * BOLT[None, None, :]
        f[:] = f * (1 - 0.92 * dens[:, :, None]) + col * dens[:, :, None]

    def body(s, f, t, zh, v=1.0):
        """青竜の 胴: らせんの 筒(うろこ・金の 背びれ)。zh: 頭の 近さ 0(奥)..1(目の前を 通りすぎる)"""
        W, H, k = s.W, s.H, s.k; N = 120; segs = []
        for i in range(N):
            u = i / float(N - 1); z = zh - u * 0.95                                    # うしろほど 奥
            if z <= 0.02: continue
            q = z ** 2.4; a = 1.2 + u * 13.0 - t * 1.6 + zh * 2.0; rho = (0.02 + 0.80 * q) * H; R = (4 + 150 * q) * k * (1.0 - 0.45 * u) * (1.25 if i < 3 else 1.0)
            segs.append((z, s.VP[0] + math.cos(a) * rho * 0.62, s.VP[1] + math.sin(a) * rho, R, a, i, u))
        segs.sort(key=lambda g: g[0])
        for z, x, y, R, a, i, u in segs:
            x0 = int(max(0, x - R - 2)); x1 = int(min(W, x + R + 2)); y0 = int(max(0, y - R - 2)); y1 = int(min(H, y + R + 2))
            if x1 - x0 < 2 or y1 - y0 < 2 or R < 1.2: continue
            X = (s.XX[y0:y1, x0:x1] - x) / R; Y = (s.YY[y0:y1, x0:x1] - y) / R; d2 = X * X + Y * Y; m = sstep(1.0, 1.0 - min(0.5, 2.0 / R), np.sqrt(d2)) * v
            zz = np.sqrt(np.clip(1 - d2, 0, 1)); ox, oy = math.cos(a), math.sin(a)                                                    # 外向き(背)
            back_ = np.clip(X * ox + Y * oy, -1, 1); lit = 0.22 + 0.85 * zz * (0.65 + 0.35 * np.clip(-X * 0.5 - Y * 0.8, -1, 1)) + 0.55 * np.exp(-(((X + 0.35) ** 2 + (Y + 0.45) ** 2) / 0.05))
            band = 0.75 + 0.25 * (i % 2); belly = sstep(-0.50, -0.85, back_); lit = lit * (0.80 + 0.35 * np.cos(X * 7.0 + i) * np.cos(Y * 7.0 - i * 0.7))
            col = (np.array([0.05, 0.26, 0.72], np.float32) * band)[None, None, :] * (1 - belly)[:, :, None] + np.array([0.75, 0.86, 0.80], np.float32)[None, None, :] * belly[:, :, None]
            col = col * lit[:, :, None] + (sstep(0.75, 1.0, np.sqrt(d2)) * 0.35)[:, :, None] * BOLT[None, None, :]
            F = f[y0:y1, x0:x1]; F[:] = F * (1 - m[:, :, None]) + col * m[:, :, None]
            if i % 3 == 1 and R > 5 * k:                                               # 金の 背びれ
                P = [(x + ox * R * 0.8 - oy * R * 0.35, y + oy * R * 0.8 + ox * R * 0.35), (x + ox * R * 2.0, y + oy * R * 2.0), (x + ox * R * 0.8 + oy * R * 0.35, y + oy * R * 0.8 - ox * R * 0.35)]
                M_ = s.L0(); cv2.fillPoly(M_, [np.array(P, np.int32)], 1.0, cv2.LINE_AA); f[:] = f * (1 - M_[:, :, None] * v) + (GOLD * 0.9)[None, None, :] * M_[:, :, None] * v
            if i == 0 and R > 3 * k:                                                   # 頭: 目の 光と ひげ
                for sg in (-1, 1):
                    e = (x - oy * sg * R * 0.45 - ox * R * 0.1, y + ox * sg * R * 0.45 - oy * R * 0.1); s.flare(f, e, min(22 * k, R * 0.28), GOLD, 0.9 * v, 1.2, 0.0, False)
                    P = [(x - oy * sg * R * 0.7, y + ox * sg * R * 0.7)] + [(x - oy * sg * R * (0.9 + 0.5 * j) - ox * R * 0.5 * j + math.sin(t * 6 + j) * R * 0.2, y + ox * sg * R * (0.9 + 0.5 * j) - oy * R * 0.5 * j) for j in range(1, 5)]
                    s.glowline(f, P, GOLD, max(1.0, R * 0.03 / k), 0.7 * v, 0.6)

    def bolts(s, f, t, v, n=3, main=0.0):
        """いなずま(まん中 → ふち。枝分かれ)"""
        if v <= 0.01 and main <= 0.01: return
        k = s.k
        for j, (pts, br) in enumerate(s.cr):
            on = main if main > 0 else (v if (int(t * 22) * 7 + j * 3) % 11 < n else 0.0)
            if on <= 0.01: continue
            P = [(float(p[0]), float(p[1])) for p in pts]; s.glowline(f, P, BOLT, 1.6 + 2.5 * main, 0.9 * on, 1.3 + main)
            for idx, bb in br: s.glowline(f, [(float(p[0]), float(p[1])) for p in bb], BOLT, 1.0 + main, 0.6 * on, 0.8)

    def frame(s, t):
        s._t = t; o = {}; s._rimc = WHITE; s._hsh = 0.96
        if t < T_CUT: f, sh = s.sc_call(t)
        elif t < T_COME: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.92)
        else: f, sh = s.sc_storm(t)
        f, sh = s.finale(f, sh, t, DUR, s.btip, (0.3, 0.62, 1), (1, 0.82, 0.32), -1)
        return s.film(f, t, sh * s.k, 0.55, o.get("thr", 0.84), (0.8, 0.9, 1.0), 1.0 + 0.2 * sh, 0.035, o.get("vig", 0.55), 1.12, 1.04)

    # 3) 雷雲 → 竜の胴が らせんで 降りてくる → いなずまが つらぬく
    def sc_storm(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_COME; g = clamp(u / (T_HIT - T_COME)); hit = t - T_HIT
        f = s.sky(t, 1.0); s.vortex(f, t, 0.15 + 1.6 * ease(u / 1.2), 1.0)
        f = s.peaks(f, 1.0); s._rimc = BOLT; s._hsh = 0.75 - 0.25 * g
        s.rush(f, t, 0.4 + 0.6 * g, 0.5 + 1.0 * g)
        s.hero(f, t, jit=0.6 * k * g); s.carve(f, 0.8 + 0.5 * math.sin(t * 20))
        zh = 0.06 + 1.55 * g ** 1.6
        s.bolts(f, t, 0.5 * clamp(u / 0.5), 2)
        if u > 0.25: s.body(f, t, zh, clamp((u - 0.25) / 0.3))
        s.flare(f, s.VP, (10 + 6 * math.sin(t * 25)) * k, BOLT, 0.7, 1.4, t, False)
        sh = 2 + 6 * g
        if hit > 0:                                                                  # 決め: いなずま
            e = math.exp(-hit * 6); s.bolts(f, t, 1.0, 11, 1.0); s.flare(f, (W / 2, H * 0.42), (40 + 200 * clamp(hit * 3)) * k, BOLT, 1.0, 2.5, 0.3, False)
            f = f + (0.9 * e + 0.6 * clamp(hit / 0.4) ** 2); sh = 16
        if u < 0.1: f = f * (u / 0.1) + (1 - u / 0.1) * 0.5
        return f, sh


if __name__ == "__main__":
    r = Seiryu()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
