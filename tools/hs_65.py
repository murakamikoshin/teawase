"""ヒミカ(★5 日の巫女王) 専用の必殺技「日輪ノ裁キ」— 高品質版(hs_fx)・通し直し。敵は さいしょから 画面の こちら側。鏡の日が こちらを 焼く
まっ暗な 古い社(朱の丸柱・しめなわ・ほこりの 舞う 日ざしが 1本)。鏡を 向けると 日ざしが 鏡に 乗り、宙に 現れる 銅鏡へ キィン、キィンと 6回 はね返って 光の あみを 張る(前ふり: この鏡と 光が 武器) →
銅鏡の形の 窓のカットイン → 宙の 6枚が 1枚ずつ くるりと こちらを 向く(まぶしい 円盤に なる)。あみの 光は ぜんぶ 巫女の 手の鏡へ もどって たまる →
手の鏡を こちらへ 向ける = 決め。鏡の中の 日輪(もえる 面・光の かんむり)が ふくらんで 画面を おおう → 白 → 暗転
使い方: python3 hs_65.py 出力.mp4 / python3 hs_65.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old65", "/home/claude/tools/pre_fin/hs_65_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = 9.3
GOLD, SUN, VERM = old.GOLD, old.SUN, old.VERM
T_CATCH, T_R0, RSTEP, T_CUT, T_AIM = old.T_CATCH, old.T_R0, old.RSTEP, old.T_CUT, old.T_AIM
T_FACE, FSTEP, T_TURN, T_HIT = 5.7, 0.14, 7.2, 8.6
DUST = [(0.10, 0.05, 0.03), (0.22, 0.12, 0.05), (0.55, 0.38, 0.16)]
BRZ = np.array([0.62, 0.44, 0.18], np.float32)


class Himika(FX, old.Himika):
    def __init__(s, W=540):
        old.Himika.__init__(s, W); s.fx_init(65); H, k = s.H, s.k
        sh = np.ones((H, W), np.float32)
        for x, w in ((0.03, 34), (0.21, 22), (0.80, 22), (0.97, 34)):                    # 丸柱の まるみ・うるしの てり・木目
            q = (s.XX - x * W) / (w * k / 2 + 1.0); ins = np.abs(q) < 1.0
            v = 0.30 + 1.0 * np.sqrt(np.clip(1 - q * q, 0, 1)) * (0.8 - 0.3 * q) + 0.8 * np.exp(-((q + 0.42) / 0.13) ** 2)
            sh = np.where(ins, v, sh)
        gr = cv2.remap(s._tiles[0], ((s.XX * 2.2) % 256).astype(np.float32), ((s.YY * 0.07) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        s.psh = (sh * (0.8 + 0.4 * gr)).astype(np.float32)
        dep = np.clip((s.YY - s.flr) / (H - s.flr), 0, 1); pl = np.floor((s.XX - W / 2) / ((60 + 240 * dep) * k) + 0.5)
        fx_ = (s.XX - W / 2) / ((60 + 240 * dep) * k) + 0.5 - pl - 0.5
        g2 = cv2.remap(s._tiles[1], ((s.XX * 0.3 + pl * 31) % 256).astype(np.float32), ((dep * 600 / (0.4 + dep) + pl * 17) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        s.flo = ((0.6 + 0.6 * g2) * (0.85 + 0.25 * np.mod(pl * 0.37, 1.0)) * (1 - 0.75 * sstep(0.44, 0.5, np.abs(fx_)))).astype(np.float32)
        tw = (np.sin((s.XX + s.YY * 0.8) / (5 * k)) * 0.5 + 0.5); s.rtw = (0.6 + 0.6 * tw).astype(np.float32)                         # なわの より
        L = s.L0(); cv2.fillPoly(L, [np.array([(s.SRC[0] - 7 * k, -10), (s.SRC[0] + 7 * k, -10), (s.FLO[0] + 26 * k, s.FLO[1]), (s.FLO[0] - 26 * k, s.FLO[1])], np.int32)], 1.0, cv2.LINE_AA)
        s.shaft = cv2.GaussianBlur(L, (0, 0), 4 * k)

    # ---- 社 ----
    def shrine(s, t, lit=0.0):
        W, H, k = s.W, s.H, s.k; a = 0.10 + 0.90 * lit
        f = (np.array([0.030, 0.020, 0.016], np.float32)[None, None, :] * (0.4 + 0.6 * s.yn)[:, :, None] * (0.6 + 2.0 * lit)).astype(np.float32)
        f += s.nebula(t, DUST, 0.0009, (3.0, -2.0), 0.25 + 0.6 * lit)
        f = f * (1 - s.pil[:, :, None]) + (VERM * a)[None, None, :] * (s.pil * s.psh)[:, :, None]
        f = f * (1 - s.rope[:, :, None]) + (np.array([0.78, 0.64, 0.34], np.float32) * a)[None, None, :] * (s.rope * s.rtw)[:, :, None]
        f = f * (1 - s.shide[:, :, None]) + (np.array([0.96, 0.94, 0.88], np.float32) * (0.2 + 0.8 * lit))[None, None, :] * s.shide[:, :, None]
        g = sstep(s.flr - 1, s.flr + 1, s.YY); dep = np.clip((s.YY - s.flr) / (H - s.flr), 0, 1)
        fl = np.array([0.30, 0.19, 0.10], np.float32)[None, None, :] * (s.flo * (0.25 + 0.9 * lit) * (0.7 + 0.5 * dep))[:, :, None]
        return f * (1 - g[:, :, None]) + fl * g[:, :, None]

    def sunshaft(s, f, t, v=1.0):
        if v <= 0.01: return
        hz_ = 0.5 + 0.9 * s.fbm(0.002, t * 8, t * 14, 3, 0, 2); d = np.zeros_like(f); s.starfield(d, t, 1500, (1, 1, 1), 3, 1.5, (4.0, 16.0))
        f += (s.shaft * (hz_ * 0.30 + d[:, :, 0] * 1.6) * v)[:, :, None] * SUN[None, None, :]

    def beam(s, f, a, b, v=1.0, th=1.6):
        s.glowline(f, [a, b], GOLD, th, 0.75 * v, 1.0)

    def bronze(s, f, c, r, lit=0.0, sq=1.0, rot=0.0):
        """銅鏡(式で 描く)。sq: 横の つぶれ(1 で こちら向き)。lit: 光を うけた 明るさ"""
        W, H, k = s.W, s.H, s.k
        if r < 2: return
        x0 = int(max(0, c[0] - r - 2)); x1 = int(min(W, c[0] + r + 2)); y0 = int(max(0, c[1] - r - 2)); y1 = int(min(H, c[1] + r + 2))
        if x1 - x0 < 3 or y1 - y0 < 3: return
        X = (s.XX[y0:y1, x0:x1] - c[0]) / (r * max(0.08, sq)); Y = (s.YY[y0:y1, x0:x1] - c[1]) / r; d = np.hypot(X, Y); ang = np.arctan2(Y, X); m = sstep(1.0, 1.0 - 2.0 / r, d)
        met = 0.35 + 0.65 * (np.cos(ang + 2.3) * 0.5 + 0.5) ** 1.5 + 0.6 * np.exp(-((np.mod(ang + 2.3 + 3.1416, 6.2832) - 3.1416) / 0.3) ** 2)
        rimp = np.clip((d - 0.86) / 0.14, 0, 1); col = BRZ[None, None, :] * (met * (0.45 + 0.8 * np.sin(rimp * 3.1416)))[:, :, None]                 # ふち
        teeth = sstep(0.0, 0.25, np.cos(ang * 14 + rot)) * sstep(0.60, 0.64, d) * sstep(0.86, 0.82, d); mid = sstep(0.60, 0.62, d) * sstep(0.86, 0.84, d)
        col = col * (1 - mid[:, :, None]) + (BRZ * 0.55)[None, None, :] * ((0.6 + 0.5 * met) * (1 - 0.55 * teeth))[:, :, None] * mid[:, :, None]
        face = sstep(0.60, 0.58, d); fc = GOLD[None, None, :] * (0.55 + 0.45 * (1 - d / 0.6) + 0.5 * np.exp(-(((X + Y) * 0.7 + 0.18) / 0.10) ** 2))[:, :, None] * (0.75 + 0.25 * met)[:, :, None]
        fc = fc * (1 - lit) + SUN[None, None, :] * (lit * (1.2 + 1.2 * (1 - d / 0.6)))[:, :, None]
        col = col * (1 - face[:, :, None]) + fc * face[:, :, None]
        col *= (1 - 0.7 * sstep(0.955, 1.0, d))[:, :, None]
        f[y0:y1, x0:x1] = f[y0:y1, x0:x1] * (1 - m[:, :, None]) + col * m[:, :, None]
        if lit > 0.05: s.flare(f, c, r * 0.30 * (0.5 + lit), SUN, 0.55 * lit * min(1.0, sq + 0.3), 1.0 + lit, 0.0, False)

    def motes(s, f, t, v=1.0, rise=0.0):
        s.starfield(f, t, 500, GOLD, 4, 0.9 * v, (3.0, -8.0 - 40 * rise))

    def hero(s, f, t, light=1.0, **kw):
        img = s.pose(t, **kw); s.draw_lit(f, img, s.CX, s.CY, s.HF, rim=SUN, rim_dir=(-0.7, -0.7), rim_v=0.3 + 0.6 * light, amb=(1.0, 0.85, 0.6), amb_v=0.14, shade=0.30 + 0.66 * light)
        L = s.L0(); cv2.ellipse(L, (int(s.CX), int(s.flr + 5 * s.k)), (int(s.W * 0.20), int(8 * s.k)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)

    def web(s, f, t, n, v=1.0, back_=0.0):
        """日ざし → 手の鏡 → 銅鏡 n枚 の 光の あみ。back_: 0..1 で あみが 手の鏡へ もどる"""
        pts = [s.M0] + s.mp[:n]; s.beam(f, s.SRC, s.M0, v, 2.2)
        for i in range(len(pts) - 1):
            g = ease((t - (T_R0 + i * RSTEP)) / 0.07); a, b = pts[i], pts[i + 1]
            q0 = clamp(back_ * len(pts) - (len(pts) - 2 - i))                         # うしろの 線から 順に ちぢむ
            if q0 >= 1: continue
            s.beam(f, a, (a[0] + (b[0] - a[0]) * g * (1 - q0), a[1] + (b[1] - a[1]) * g * (1 - q0)), v, 1.6)

    def nichirin(s, f, c, R, t, v=1.0, frame_=1.0):
        """鏡の中の 日輪(式で 描く: もえる 面・ふちの かげ・光の かんむり・放射の すじ)"""
        k = s.k; X = (s.XX - c[0]) / R; Y = (s.YY - c[1]) / R; d = np.hypot(X, Y); ang = np.arctan2(Y, X)
        rm = lambda T, sc_, o: cv2.remap(s._tiles[T], ((X * sc_ + o + t * 6) % 256).astype(np.float32), ((Y * sc_ + o * 0.7 - t * 4) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        gran = rm(0, 60.0, 30.0) * 0.6 + rm(2, 170.0, 80.0) * 0.4
        if R > 220 * k: gran = gran * 0.7 + rm(3, 520.0, 11.0) * 0.3
        z = np.sqrt(np.clip(1 - d * d, 0, 1)); heat = np.clip(0.42 + 0.50 * z + 1.5 * (gran - 0.5), 0, 1.3)
        disc = s.ramp(np.clip(heat / 1.3, 0, 1), [(0, (0.75, 0.16, 0.01)), (0.40, (1.0, 0.45, 0.05)), (0.72, (1.0, 0.80, 0.30)), (1.0, (1.0, 1.0, 0.90))]) * 1.2
        m = sstep(1.0, 1.0 - 2.0 / R, d) * v; f[:] = f * (1 - m[:, :, None]) + disc * m[:, :, None]
        st = cv2.remap(s._tiles[1], ((ang / 6.2832 + 0.5) * 256 * 4 % 256).astype(np.float32), ((d * 14 - t * 30) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        st2 = cv2.remap(s._tiles[2], ((ang / 6.2832 + 0.5) * 256 * 11 % 256).astype(np.float32), ((d * 5 - t * 12) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        out = np.clip(d - 1, 0, None); cor = (np.exp(-out / 0.10) * 0.9 + np.exp(-out / 0.45) * (0.25 + 1.3 * st * st2) + np.exp(-out / 1.6) * 0.5 * st2 ** 2) * (1 - sstep(1.0, 1.0 - 2.0 / R, d)) * v
        f += cor[:, :, None] * np.array([1.0, 0.80, 0.42], np.float32)[None, None, :]
        if frame_ > 0.02:                                                            # まわりの 銅の ふち(鏡の 中から 出てくる)
            band = sstep(1.10, 1.12, d) * sstep(1.30, 1.28, d) * frame_; met = 0.35 + 0.65 * (np.cos(ang + 2.3) * 0.5 + 0.5) ** 1.5; p = np.clip((d - 1.10) / 0.20, 0, 1)
            teeth = sstep(0.0, 0.25, np.cos(ang * 14)) * sstep(1.14, 1.16, d) * sstep(1.24, 1.22, d)
            f[:] = f * (1 - band[:, :, None]) + (BRZ * 2.2)[None, None, :] * (met * (0.5 + 0.7 * np.sin(p * 3.1416)) * (1 - 0.5 * teeth))[:, :, None] * band[:, :, None]

    def frame(s, t):
        s._t = t; o = {}
        if t < T_CUT: f, sh = s.sc_relay(t)
        elif t < T_AIM: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.4, thr=0.95)
        else: f, sh = s.sc_sun(t)
        f, sh = s.finale(f, sh, t, DUR, s.mir, (1, 0.86, 0.46), (1, 0.4, 0.2), -1)
        return s.film(f, t, sh * s.k, 0.6, o.get("thr", 0.82), (1.0, 0.86, 0.6), 1.0 + 0.2 * sh, 0.035, o.get("vig", 0.6), 1.12, 1.05)

    # 1) 日ざしを 鏡で うけて、はね返していく
    def sc_relay(s, t):
        W, H, k = s.W, s.H, s.k; c = t - T_CATCH; n = s.nref(t); lit = 0.05 + 0.11 * n + (0.08 * clamp(c / 0.3) if c > 0 else 0)
        f = s.shrine(t, lit)
        for i in range(n):
            v = t - (T_R0 + i * RSTEP); s.bronze(f, s.mp[i], 48 * k * back((v + 0.05) / 0.15, 2.0), 0.35 + 0.65 * math.exp(-v * 5), 0.62 + 0.2 * math.sin(i * 2.1), t)
        s.hero(f, t, clamp(0.15 + lit * 1.6 + (0.4 if c > 0 else 0)), squash=(-0.05 * math.exp(-c * 10) if c > 0 else 0))
        if c <= 0.25:                                                                # 床に 落ちる 日ざし → 鏡へ
            g = ease2(c / 0.25) if c > 0 else 0.0
            if c <= 0:
                s.sunshaft(f, t, 1.0); L = s.L0(); cv2.ellipse(L, (int(s.FLO[0]), int(s.FLO[1])), (int(26 * k), int(7 * k)), 0, 0, 360, 1.0, -1, cv2.LINE_AA); s.glowmask(f, L, SUN, 0.9, 0.9, 0.8, 10.0)
            else: s.beam(f, s.SRC, (s.FLO[0] + (s.M0[0] - s.FLO[0]) * g, s.FLO[1] + (s.M0[1] - s.FLO[1]) * g), 0.9, 2.4)
        else: s.web(f, t, n)
        if c > 0: s.flare(f, s.M0, 12 * k * (0.6 + 0.6 * math.exp(-c * 6)), SUN, 0.7, 1.0, 0.0, False)
        for i in range(n):
            v = t - (T_R0 + i * RSTEP + 0.07)
            if 0 < v < 0.2: s.flare(f, s.mp[i], 22 * k * (1 - v / 0.2), SUN, 1.0, 1.4, 0.4, False)
        s.motes(f, t, 0.25 + 0.5 * lit)
        if lit > 0.2: f = s.godrays(f, s.M0, 0.25 * lit, 0.75, 8, 0.86, 0.3)
        return f * clamp(t / 0.4), 0

    # 2) 銅鏡の形の 窓
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        C = (W / 2, H * 0.38); g = back(u / 0.3, 1.3); R = W * 0.44 * g; ri = R * 0.74
        dx = s.XX - C[0]; dy = s.YY - C[1]; d = np.hypot(dx, dy); ang = np.arctan2(dy, dx)
        ray = cv2.remap(s._tiles[1], ((ang / 6.2832 + 0.5) * 256 * 5 + t * 6).astype(np.float32) % 256, ((d * 0.02 - t * 8) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        f = (np.array([0.07, 0.02, 0.012], np.float32)[None, None, :] + (sstep(0.45, 0.85, ray) * 0.22 * np.exp(-d / (H * 0.6)))[:, :, None] * (VERM * 1.5 + GOLD * 0.3)[None, None, :]).astype(np.float32)
        s.starfield(f, t, 500, GOLD, 5, 0.6, (4.0, -20.0))
        f *= (1 - 0.55 * np.exp(-np.clip(d - R, 0, None) / (36 * k)))[:, :, None]
        win = GOLD[None, None, :] * (0.95 - 0.55 * d / max(1.0, ri))[:, :, None] + s.nebula(t, [(0.5, 0.2, 0.05), (0.9, 0.6, 0.2), (1.0, 0.95, 0.7)], 0.0014, (10.0, -6.0), 0.6)
        zf = 1.02 + 0.05 * ease(u / 1.8); sc = zf * H / s.ah; img = s.pose(t)
        s.draw_lit(win, img, C[0] - (s.fc[0] - s.aw / 2) * sc, C[1] + H * 0.015 - (s.fc[1] - s.ah / 2) * sc, zf, rim=SUN, rim_dir=(-0.6, -0.8), rim_v=0.7, amb=(1.0, 0.9, 0.7), amb_v=0.10, shade=0.98)
        s.bronze(f, C, R, 0.0, 1.0, t * 0.2)
        m = sstep(ri + 1, ri - 1, d); f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        f *= (1 - 0.5 * sstep(ri - 9 * k, ri, d) * m)[:, :, None]
        sw = (u - 0.45) / 0.4                                                        # 鏡の面を 光が よぎる
        if 0 < sw < 1: x = C[0] - ri * 1.3 + ri * 2.6 * sw; f += (np.exp(-(((dx - (x - C[0])) + dy * 0.35) / (22 * k)) ** 2) * m * 0.55)[:, :, None]
        f *= (1 - 0.7 * sstep(0.70, 0.79, s.yn) * sstep(0.96, 0.90, s.yn))[:, :, None]
        n = len(s.ht); step = W * 0.175
        for i, T in enumerate(s.ht):                                                 # 技名: まん中から 日が さすように
            a = (u - 0.55 - abs(i - 2) * 0.12) / 0.2
            if a > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.835, clamp(a * 2), 1.0)
        for i, T in enumerate(s.vo):
            if u > 0.4 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.755, 1.0, 1.0)
        if u > 1.3: s.text(f, s.txt_en, W / 2, H * 0.835 + 60 * k, ease((u - 1.3) / 0.3), 1.0)
        if u < 0.08: f = f + (1 - u / 0.08) * 0.7
        return f, 0

    # 3) 宙の 鏡が こちらを 向く → 光が 手の鏡へ もどる → 手の鏡を こちらへ → 日輪
    def sc_sun(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_AIM; b = t - T_TURN; g = clamp(b / (T_HIT - T_TURN)); gg = g ** 2.0
        nf = sum(1 for i in range(6) if t >= T_FACE + i * FSTEP); bk = clamp((t - (T_FACE + 6 * FSTEP)) / 0.45)
        f = s.shrine(t, 0.72 + 0.28 * g)
        for i in range(6):
            v = t - (T_FACE + i * FSTEP); tr = ease(v / 0.16) if v > 0 else 0.0; sq = (0.62 + 0.2 * math.sin(i * 2.1)) * (1 - tr) + tr
            sq = sq * abs(math.cos(tr * 3.1416)) if 0 < tr < 1 else sq               # くるりと 1回 うら返って こちらへ
            s.bronze(f, s.mp[i], 48 * k * (1 + 0.12 * math.exp(-max(0.0, v) * 8) * (v > 0)), 0.25 + 0.75 * tr, max(0.08, sq), t)
        sw = math.exp(-b * 5) if b > 0 else 0.0; pre = clamp((t - (T_TURN - 0.3)) / 0.3)
        s.hero(f, t, 1.0, lean=(6 * k * pre if b < 0 else -12 * k * sw), squash=0.04 * sw)
        if bk < 1: s.web(f, t, 6, 1.0, bk)
        else: s.beam(f, s.SRC, s.M0, 1.0, 2.2)
        hold = clamp((t - (T_FACE + 6 * FSTEP)) / 0.6)                                   # 手の鏡に 光が たまる
        c = (s.M0[0] + (W / 2 - s.M0[0]) * ease2(g), s.M0[1] + (H * 0.42 - s.M0[1]) * ease2(g)); R = (15 + 8 * hold) * k * (1 + 0.06 * math.sin(t * 40) * hold) + gg * H * 1.25
        if b <= 0: s.flare(f, s.M0, (10 + 14 * hold) * k, SUN, 0.6 + 0.5 * hold, 1.0 + hold, t * 0.6, hold > 0.5)
        else:
            s.nichirin(f, c, R, t, 1.0, 1.0); s.flare(f, c, min(60 * k, R * 0.5), SUN, 0.9 * (1 - 0.6 * g), 2.0, t * 0.3, g < 0.5)
            s.rushdust(f, t, c, 1500, 0.9 + 1.8 * g, SUN, 0.4 + 0.9 * g, 5, 1.0); f = s.godrays(f, c, 0.4 * (1 - 0.5 * g), 0.8, 10, 0.88, 0.5)
            if b < 0.08: f = f + (1 - b / 0.08) * 0.6
        s.motes(f, t, 0.8, 1.0)
        fl = 0.0
        for i in range(nf):
            v = t - (T_FACE + i * FSTEP + 0.16)
            if 0 < v < 0.15: fl = max(fl, 1 - v / 0.15)
        if u < 0.1: f = f * (u / 0.1) + (1 - u / 0.1) * 0.5
        return f + fl * 0.10, (2 * fl if b < 0 else 3 + 11 * gg)

    def audio(s, path):
        a = Snd(DUR, 65); G = 196.0
        def fq(semi): return G * 2 ** (semi / 12.0)
        def suzu(t0, g=0.06, n=5):
            for q in range(n):
                for fr in (2637, 3520, 4699, 6272): a.note(fr * (1 + 0.01 * q), t0 + q * 0.035, 0.12, g * (1 - q / (n + 2.0)) * 0.6, "sine", 0.0005)
                a.hat(t0 + q * 0.035, g * 0.5, 0.02)
        def kane(semi, t0, g=0.12):
            f0 = fq(semi) * 2
            for m, gg, dc in ((1.0, 1.0, 0.7), (2.76, 0.5, 0.4), (5.4, 0.3, 0.2), (8.9, 0.15, 0.1)): a.note(f0 * m, t0, dc, g * gg, "sine", 0.0005)
        def odaiko(t0, g=1.0): a.boom(t0, 44, g, 0.5); a.boom(t0, 88, g * 0.3, 0.15)
        def organ(semi, t0, ln, g=0.035):
            for m, gg in ((1, 1.0), (2, 0.5), (3, 0.3), (4, 0.2)): a.note(fq(semi) * m, t0, ln, g * gg, "sine", 0.12)
        a.a += a.tone(49.0) * 0.06 * a.gate(0, T_CUT, 0.4) + a.lp(a.nz, 0.015) * 0.5 * a.gate(0, T_CATCH, 0.3)
        suzu(0.5, 0.04, 3); odaiko(1.1, 0.5)
        a.rise(T_CATCH, T_CATCH + 0.25, 600, 1800, 0.08); kane(0, T_CATCH + 0.25, 0.10); suzu(T_CATCH + 0.25, 0.05, 3)
        pent = (2, 4, 7, 9, 12, 14)
        for i in range(6): t0 = T_R0 + i * RSTEP + 0.07; kane(pent[i], t0, 0.11); a.hat(t0, 0.15, 0.01)
        odaiko(T_R0 + 6 * RSTEP + 0.05, 0.9)
        odaiko(T_CUT, 1.0); suzu(T_CUT + 0.05, 0.07, 6)
        for i, (sm, dt) in enumerate(((7, 0.25), (9, 0.5), (12, 0.75), (14, 1.0), (12, 1.25), (16, 1.5))): kane(sm - 12, T_CUT + dt, 0.10); organ(sm - 12, T_CUT + dt, 0.3, 0.03)
        for dt in (0.75, 1.5): odaiko(T_CUT + dt, 0.6)
        for i in range(6): t0 = T_FACE + i * FSTEP + 0.16; kane(pent[5 - i], t0, 0.11); a.hat(t0, 0.15, 0.01); a.whoosh(t0 - 0.14, t0, 0.15, 0.3)     # 鏡が くるりと こちらを 向く(はね返りの キィン を 逆の 順で)
        tb = T_FACE + 6 * FSTEP; a.rise(tb, T_TURN, 300, 1800, 0.08); tk = tb; i = 0                                                # 光が 手の鏡へ もどって たまる
        while tk < T_TURN - 0.05: a.note(fq(pent[i % 6] + 12 * (i // 6 % 2)) * 2, tk, 0.15, 0.04, "bell", 0.001); tk += max(0.04, 0.12 - i * 0.008); i += 1
        odaiko(T_TURN, 1.4); kane(0, T_TURN, 0.14); a.rise(T_TURN - 0.02, T_TURN + 0.12, 4000, 600, 0.10)                           # 手の鏡を こちらへ
        for sm, g in ((0, 0.05), (7, 0.04), (12, 0.04), (16, 0.035), (19, 0.03)): organ(sm, T_TURN, T_HIT - T_TURN + 0.3, g)
        tk = T_TURN + 0.15; i = 0
        while tk < T_HIT - 0.05: odaiko(tk, 0.4 + 0.7 * (tk - T_TURN) / (T_HIT - T_TURN)); kane((0, 4, 7, 12, 16, 19, 24, 28)[i % 8] - 12, tk + 0.04, 0.06); tk += max(0.08, 0.30 - i * 0.035); i += 1
        suzu(T_TURN + 0.2, 0.05, 14); suzu(T_TURN + 0.8, 0.06, 14)
        a.rise(T_TURN, T_HIT, 80, 420, 0.13); a.a += a.lp(a.nz, 0.10) * np.clip((a.t - T_TURN) / (T_HIT - T_TURN), 0, 1) ** 2 * (a.t < T_HIT) * 0.9
        odaiko(T_HIT, 1.8)
        s.fin_snd(a, DUR, 65); a.save(path)


if __name__ == "__main__":
    r = Himika()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
