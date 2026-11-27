"""スチム(★4 蒸気の拳闘士) 専用の必殺技「蒸気鉄拳」— 高品質版(hs_fx)。背中の 機関の 圧力計が 赤に ふりきれ、笛と ともに 蒸気が 画面を 白く うめ、蒸気の 輪が 三つ ぬけて きた あと、その まん中から 鉄の こぶしが まっすぐ 飛んで くる
夕ぐれの 機巧の 街(レンガ・煙突・回る 歯車)に 立つ → 圧力計の 丸い 窓の カットイン+技名 →
かまえて 圧を ためる(大きな 圧力計の 針が 赤へ)→ ぴーっ(蒸気が ふき出して 画面が 白く)→ 輪が 三つ →
こぶしが 蒸気を 割って 大きく なる → 白 → 暗転
使い方: python3 hs_435.py 出力.mp4 / python3 hs_435.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.4
T_CUT, T_ACT, T_BLOW, T_RING, T_FIST, T_HIT = 2.7, 4.7, 6.0, 6.5, 7.3, 9.4
DUSK = np.array([1.0, 0.56, 0.26], np.float32); DEEP = np.array([0.16, 0.20, 0.42], np.float32); BRICK = np.array([0.50, 0.24, 0.16], np.float32)
BRASS = np.array([0.95, 0.70, 0.28], np.float32); IRON = np.array([0.22, 0.22, 0.26], np.float32); STEAM = np.array([1.0, 0.98, 0.96], np.float32)
HOT = np.array([1.0, 0.45, 0.15], np.float32); FACE = np.array([0.96, 0.92, 0.80], np.float32); RED = np.array([0.90, 0.16, 0.12], np.float32)
EYE = (285, 265); FIST = (650, 455); FR = 150.0; PIPE = (560, 100)


class Steam(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 435, (0.95, 0.6, 0.3), "蒸気鉄拳", "Steam Knuckle", W); s.fx_init(435)
        s._vt = [s.text_img(c, hz.SERIF, int(70 * s.k), 3) for c in "蒸気鉄拳"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; s.hy = 0.60
        s._bg = s.townimg()

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 機巧の 街(動かない 部分。1回だけ 作る): 夕空・工場の かげ・明かりの ついた 窓・レンガの 地面 ----
    def townimg(s):
        W, H, k = s.W, s.H, s.k; hy = s.hy; v = (s.yn / hy).clip(0, 1)
        f = DEEP[None, None, :] * (1 - v ** 1.5)[:, :, None] + DUSK[None, None, :] * (v ** 1.5)[:, :, None]
        f = f + (np.exp(-np.hypot((s.xn - 0.5) * 1.2, (s.yn - hy) * 2.2) / 0.22) * 0.55)[:, :, None] * np.array([1.0, 0.85, 0.55], np.float32)[None, None, :]
        for lay, (base, amp, dark) in enumerate(((0.30, 0.16, 0.42), (0.40, 0.14, 0.24))):                                # 工場の かげ 2層
            cell = 0.085 - 0.02 * lay; ix = np.floor(s.xn / cell + lay * 0.5); rr = np.sin(ix * (12.9898 + lay * 3.1)) * 43758.5453; rr = rr - np.floor(rr)
            top = hy - (base * 0.5 + amp * rr) * (1 - 0.25 * lay); fu = (s.xn / cell + lay * 0.5) % 1.0
            chim = (rr > 0.55) & (np.abs(fu - 0.5) < 0.16); top = np.where(chim, top - 0.09 - 0.05 * rr, top)
            m = sstep(-0.002, 0.002, s.yn - top) * (s.yn < hy + 0.01)
            wy = ((s.yn - top) / 0.035) % 1.0; wx = (fu * 3.0) % 1.0; win = (np.abs(wx - 0.5) < 0.22) * (np.abs(wy - 0.5) < 0.25) * (rr * 7 % 1.0 > 0.45) * (~chim) * (s.yn - top > 0.02)
            col = (IRON * (dark / 0.24))[None, None, :] * (0.8 + 0.4 * s.tex(1, s.XX * 0.5, s.YY * 0.5 + lay * 77))[:, :, None] + (DUSK * 0.25)[None, None, :] * (1 - dark)
            col = col + (win * (0.9 - 0.3 * lay))[:, :, None] * np.array([1.0, 0.80, 0.42], np.float32)[None, None, :]
            f = f * (1 - m)[:, :, None] + col * m[:, :, None]
        d = np.clip((s.yn - hy) / (1 - hy), 0, 1); pu = (s.XX - W / 2) / (d * 0.9 + 0.25); pz = 1.0 / (d + 0.12)                    # レンガの 地面
        row = np.floor(pz * 2.6); bu = pu / (80 * k) + 0.5 * (row % 2); fu = bu % 1.0; fv = (pz * 2.6) % 1.0
        ln = np.maximum(np.exp(-((np.abs(fu - 0.5) - 0.5) / 0.035) ** 2), np.exp(-((np.abs(fv - 0.5) - 0.5) / 0.07) ** 2))
        rb = np.sin((np.floor(bu) + row * 7.3) * 12.9898) * 43758.5453; rb = rb - np.floor(rb)
        gc = BRICK[None, None, :] * ((0.70 + 0.35 * rb + 0.25 * s.tex(1, pu * 0.4, pz * 60)) * (1 - 0.55 * ln) * (0.55 + 0.6 * d))[:, :, None]
        gc = gc + (np.exp(-np.abs(s.xn - 0.5) / 0.18) * np.exp(-d / 0.35) * 0.30)[:, :, None] * DUSK[None, None, :]
        m = sstep(hy - 0.002, hy + 0.003, s.yn)[:, :, None]; f = f * (1 - m) + gc * m
        return f.astype(np.float32)

    def gear(s, f, c, R, n, ang, v=1.0):                                                    # 歯車の かげ(空に 回る)
        dx = s.XX - c[0]; dy = s.YY - c[1]; d = np.hypot(dx, dy); a = np.arctan2(dy, dx) + ang
        edge = R * (1 + 0.10 * np.clip(np.cos(a * n) * 3.0, -1, 1)); m = sstep(1.5, -1.5, d - edge) * sstep(R * 0.30, R * 0.30 + 2, d)
        sp = (np.abs(np.sin(a * 3)) < 0.22) | (d > R * 0.72) | (d < R * 0.42); m = m * sp
        col = IRON * 0.9 + BRASS * 0.10; f *= (1 - m * 0.85 * v)[:, :, None]; f += (m * v * 0.85)[:, :, None] * col[None, None, :] * (0.8 + 0.5 * np.exp(-np.abs(d - edge) / (3 * s.k)))[:, :, None]

    def town(s, t, lum=1.0):
        W, H, k = s.W, s.H, s.k; f = s._bg.copy()
        s.gear(f, (W * 0.13, H * 0.20), W * 0.20, 10, t * 0.5); s.gear(f, (W * 0.34, H * 0.11), W * 0.10, 8, -t * 1.0 + 0.3); s.gear(f, (W * 0.90, H * 0.26), W * 0.15, 9, t * 0.66 + 1.0)
        n = s.fbm(0.0011, t * 6, -t * 9, 3, 60 * k, 1); n = cv2.GaussianBlur(n, (0, 0), 4 * k)                                       # 煙突の けむり
        for cxh, ph in ((0.23, 0.0), (0.72, 0.4)):
            sm = sstep(0.50, 0.64, n * 0.5 + 0.55 * np.exp(-((s.xn - cxh - 0.16 * (s.hy - 0.25 - s.yn).clip(0, 1)) / (0.035 + 0.10 * (s.hy - 0.25 - s.yn).clip(0, 1))) ** 2) - 0.30) * sstep(s.hy - 0.20, s.hy - 0.26, s.yn) * sstep(0.0, 0.10, s.yn)
            f = f * (1 - sm * 0.7)[:, :, None] + (sm * 0.7)[:, :, None] * (STEAM * 0.90)[None, None, :] * np.clip(0.9 + 0.3 * (n - 0.5), 0.86, 1.05)[:, :, None]
        return (f * lum).astype(np.float32)

    def cloud(s, f, t, amt, c, spread=1.0):                                                 # 白い 蒸気(amt: 0〜1.4 で 画面を うめる)
        W, H, k = s.W, s.H, s.k
        if amt <= 0.005: return 0
        n = s.fbm(0.0013, t * 30, -t * 55, 4, 90 * k, 2); d = np.hypot((s.XX - c[0]) / W, (s.YY - c[1]) / H * 0.8) / spread
        m = sstep(0.0, 0.20, n * 1.2 + amt * 1.0 - d * 2.05 - 0.45) * min(1.0, amt * 4)
        nb = cv2.GaussianBlur(n, (0, 0), 30 * k); shd = np.clip(1.0 + (cv2.GaussianBlur(n, (0, 0), 9 * k) - nb) * 4.0 + 0.10 * (0.5 - s.yn), 0.86, 1.12)                # もこもこの かげ(上から 光)
        low = np.exp(-(1 - s.yn) / 0.30) * 0.16                                                                                 # 下から 夕やけの 色
        f *= (1 - m * 0.97)[:, :, None]; f += (m * 0.97 * shd * 1.3)[:, :, None] * STEAM[None, None, :] + (m * low)[:, :, None] * DUSK[None, None, :]; f -= (m * np.clip(1.02 - shd, 0, 1) * 1.3)[:, :, None] * np.array([0.55, 0.40, 0.22], np.float32)[None, None, :]; np.maximum(f, 0, out=f)
        return m

    def dial(s, f, c, R, val, alpha):                                                       # 大きな 圧力計(式): しんちゅうの 枠・目もり・赤い はんい・針
        k = s.k; dx = s.XX - c[0]; dy = s.YY - c[1]; d = np.hypot(dx, dy); a = np.arctan2(dy, dx)
        if alpha <= 0.01: return
        face = sstep(R * 0.86 + 1, R * 0.86 - 1, d); rim = sstep(R + 1.5, R - 1.5, d) * (1 - face)
        A0, A1 = math.radians(140), math.radians(400); aa = np.where(a < math.radians(100), a + 2 * math.pi, a); u = (aa - A0) / (A1 - A0)
        tick = (np.abs(((u * 20) % 1.0) - 0.5) > 0.42) * (d > R * 0.66) * (d < R * 0.80) * (u > -0.01) * (u < 1.01)
        redz = (u > 0.78) * (u < 1.01) * (d > R * 0.55) * (d < R * 0.64)
        fc = FACE[None, None, :] * (0.92 + 0.10 * s.tex(1, s.XX * 0.5, s.YY * 0.5) - 0.25 * (d / R) ** 3)[:, :, None]
        fc = fc * (1 - tick * 0.85)[:, :, None]; fc = fc * (1 - redz)[:, :, None] + redz[:, :, None] * RED[None, None, :]
        na = A0 + (A1 - A0) * val; nx, ny = math.cos(na), math.sin(na); pr = dx * nx + dy * ny; pd = np.abs(-dx * ny + dy * nx)
        ndl = sstep(3.0 * k, 1.6 * k, pd * (1 + pr.clip(0, None) / (R * 0.9))) * (pr > -R * 0.15) * (pr < R * 0.74)
        fc = fc * (1 - ndl)[:, :, None] + (ndl * 0.12)[:, :, None]; hub = sstep(R * 0.09 + 1, R * 0.09 - 1, d); fc = fc * (1 - hub)[:, :, None] + (hub * 0.7)[:, :, None] * BRASS[None, None, :]
        lightside = (dx * -0.5 + dy * -0.8) / (d + 1e-3)
        rc = BRASS[None, None, :] * (0.55 + 0.75 * lightside.clip(0, 1) + 0.25 * np.exp(-((d - R * 0.93) / (2 * k)) ** 2))[:, :, None]
        rv = (np.abs(((a / (2 * math.pi) * 12) % 1.0) - 0.5) < 0.10) * (np.abs(d - R * 0.93) < R * 0.035); rc = rc * (1 - 0.45 * rv)[:, :, None]
        hot = clamp((val - 0.78) / 0.22)
        f *= (1 - alpha * 0.35 * sstep(R * 1.25, R, d) * (1 - face) * (1 - rim))[:, :, None]
        f *= (1 - alpha * (face + rim))[:, :, None]; f += (alpha * face)[:, :, None] * fc + (alpha * rim)[:, :, None] * rc
        if hot > 0: f += (np.exp(-np.abs(d - R) / (10 * k)) * hot * alpha * (0.6 + 0.4 * math.sin(val * 90)))[:, :, None] * HOT[None, None, :]

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        s.draw_lit(f, img, cx, cy, hfrac, rim=(1.0, 0.75, 0.45), rim_dir=(0.0, -0.9), rim_v=0.45 * alpha, amb=(1.0, 0.85, 0.7), amb_v=0.10, aura=HOT, aura_v=0.04, alpha=alpha, shade=0.95)

    def frame(s, t):
        o = {}
        if t < T_CUT: f, sh = s.sc_town(t)
        elif t < T_ACT: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.97)
        else: f, sh = s.sc_punch(t)
        f, sh = s.finale(f, sh, t, DUR, None, (1.0, 0.96, 0.9), (1.0, 0.7, 0.4), -1)
        return s.film(f, t, sh * s.k, 0.35, o.get("thr", 0.96), (1.0, 0.96, 0.9), 1.0 + 0.15 * min(sh, 8), 0.03, o.get("vig", 0.38), 1.12, 0.97)

    def hero(s, f, t, alpha=1.0, **kw):
        W, H, k = s.W, s.H, s.k; cy = H * 0.63; fy = cy + 0.40 * H * 0.47
        sd = np.exp(-(((s.XX - W * 0.5) / (W * 0.26)) ** 2 + ((s.YY - fy) / (H * 0.014)) ** 2)); f *= (1 - 0.45 * alpha * sd)[:, :, None]
        img = s.pose(t, **kw); s.draw(f, img, W / 2, cy, 0.40, alpha)
        return s.at(*s.A(*FIST)), s.at(*s.A(*PIPE))

    def puff(s, f, t, c, v):                                                                # 背中の 煙突から ぽっ ぽっ
        W, H, k = s.W, s.H, s.k; n = 26; r = s.rnd[40:40 + n]; ph = (t * (0.5 + 0.5 * r[:, 0]) + r[:, 1]) % 1.0
        s.splat(f, c[0] + (r[:, 2] - 0.5) * 30 * k * ph + 30 * k * ph ** 2, c[1] - ph * 150 * k * (0.5 + r[:, 3]), (1 - ph) * v * 1.2, STEAM, 4.0, 0.9)

    # 1) 夕ぐれの 街に 立つ
    def sc_town(s, t):
        f = s.town(t); fp, pp = s.hero(f, t, clamp(t / 0.6), sway=0.2, breath=0.6); s.puff(f, t, pp, 0.7 * clamp(t / 0.6))
        return f * clamp(t / 0.5), 0

    # 2) 目の カットイン(圧力計の 丸い 窓)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.town(t, 0.34)
        a = back(clamp(u / 0.35)); c = (W / 2, H * 0.31); R = W * 0.36 * a
        bandc = (IRON * 0.8)[None, None, :] + (HOT * 0.8)[None, None, :] * (0.2 + 0.9 * sstep(0.45, 0.9, s.tex(0, s.XX * 0.4 - t * 30, s.YY * 0.4 - t * 60)))[:, :, None]
        ex, ey = s.A(*EYE); zf = 0.98; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(bandc, img, c[0] - (ex - s.aw / 2) * sc, c[1] - (ey - s.ah / 2) * sc, zf)
        if R > 8:
            dx = s.XX - c[0]; dy = s.YY - c[1]; d = np.hypot(dx, dy); an = np.arctan2(dy, dx)
            m = sstep(R + 1, R - 1, d)[:, :, None]; f = f * (1 - m) + bandc * m
            bw = 22 * k; fr = sstep(R - 1, R + 1, d) * sstep(R + bw + 1, R + bw - 1, d); ls = ((dx * -0.5 + dy * -0.8) / (d + 1e-3)).clip(0, 1)
            rv = (np.abs(((an / (2 * math.pi) * 16 + t * 0.05) % 1.0) - 0.5) < 0.09) * (np.abs(d - R - bw * 0.5) < bw * 0.22)
            fv = (0.55 + 0.8 * ls + 0.3 * np.exp(-((d - R - bw * 0.5) / (2 * k)) ** 2)) * (0.85 + 0.3 * s.tex(1, s.XX * 0.4, s.YY * 0.4)) * (1 - 0.5 * rv)
            f = f * (1 - fr)[:, :, None] + BRASS[None, None, :] * (fv * fr)[:, :, None]
            f *= (1 - 0.35 * sstep(R + bw, R + bw + 10 * k, d) * sstep(R + bw + 22 * k, R + bw + 8 * k, d))[:, :, None]
            tk = (np.abs(((an / (2 * math.pi) * 40) % 1.0) - 0.5) > 0.40) * (d > R * 0.90) * (d < R - 2); f = f * (1 - 0.5 * tk)[:, :, None] + (tk * 0.5)[:, :, None] * FACE[None, None, :]
        n = len(s._vt); step = W * 0.20
        for i, T in enumerate(s._vt):                                                # 技名: どん と 1文字ずつ 上から 落ちて はまる
            p = (u - 0.5 - i * 0.13) / 0.14
            if p > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.70 - (1 - min(1.0, p)) ** 2 * 60 * k, clamp(p * 2), 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.35 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.70 - 76 * k, 1.0, 1.0)
        if u > 1.4:
            g = ease((u - 1.4) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.70 + 56 * k), (W / 2 + W * 0.42 * g, H * 0.70 + 56 * k)], HOT, 1.0, 0.7, 0.7)
            s.text(f, s.txt_en, W / 2, H * 0.70 + 82 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.5
        sh = 3 * math.exp(-u * 6) + sum(2.5 * math.exp(-max(0.0, u - 0.5 - i * 0.13) * 14) * (u >= 0.5 + i * 0.13) for i in range(n))
        return f, sh

    # 3) 圧を ためる → ぴーっ → 蒸気で 白く → 輪 三つ → こぶしが 来る
    def sc_punch(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_ACT; sh = 0.0; C = (W * 0.5, H * 0.46)
        ch = clamp(u / (T_BLOW - T_ACT)); bl = t - T_BLOW
        f = s.town(t, 1.0 - 0.25 * ch if bl < 0 else 0.45)
        sq = 0.07 * ease(ch) if bl < 0 else -0.04 * math.exp(-bl * 5)
        fp, pp = s.hero(f, t, 1.0, sway=0.15, breath=0.3, squash=sq, jit=1.6 * k * ch ** 2 if bl < 0 else 0.0)
        s.puff(f, t * (1 + 2 * ch), pp, 0.7 + 1.2 * ch)
        if bl < 0:                                                                   # 大きな 圧力計: 針が ふるえながら 赤へ
            val = 0.12 + 0.9 * ch ** 1.6 + 0.02 * math.sin(t * 40) * ch; al = clamp(u / 0.25)
            s.dial(f, (W * 0.5, H * 0.20), W * 0.24 * (0.9 + 0.1 * back(clamp(u / 0.3))), min(1.03, val), al)
            s.soft(f, fp, 60 * k * (0.6 + ch), 0.5 * ch, HOT); sh += 1 + 5 * ch ** 2
            if u < 0.12: f = f + (1 - u / 0.12) * 0.25
            return f, sh
        # ---- ふき出す ----
        amt = 1.15 * (1 - math.exp(-bl * 3.2)); org = (pp[0] + (C[0] - pp[0]) * clamp(bl / 0.5), pp[1] + (C[1] - pp[1]) * clamp(bl / 0.5))
        s.cloud(f, t, amt, org, 1.0)
        if bl < 0.10: f = f + (1 - bl / 0.10) * 0.6
        sh += 9 * math.exp(-bl * 3) + 1.5
        for i in range(3):                                                           # 蒸気の 輪が 三つ ぬけて くる
            q = (t - T_RING - i * 0.26) / 0.75
            if 0 < q < 1.25:
                r = W * (0.03 + 1.25 * q ** 2.0); d = np.hypot(s.XX - C[0], (s.YY - C[1]) * 0.92); wd = (6 + 34 * q) * k
                ring = np.exp(-((d - r) / wd) ** 2); f *= (1 - 0.50 * np.exp(-((d - r + wd * 1.6) / (wd * 1.2)) ** 2) * (1 - q * 0.6))[:, :, None]
                f += (ring * (0.7 * (1 - q * 0.5)))[:, :, None] * STEAM[None, None, :]
                if q < 0.08: sh += 5
        p = clamp((t - T_FIST) / (T_HIT - T_FIST))
        if t >= T_FIST - 0.25:                                                       # まん中の 蒸気が 赤く 光って から こぶしが 出る
            gl = clamp((t - (T_FIST - 0.25)) / 0.25) * (1 - 0.6 * p)
            f += (np.exp(-np.hypot(s.XX - C[0], s.YY - C[1]) / (W * (0.10 + 0.5 * p))) * 0.55 * gl)[:, :, None] * HOT[None, None, :]
        if p > 0:
            g = p ** 2.3; zf = 0.40 * (1 + 0.6 * p) + 2.9 * g
            wob = (math.sin(t * 31) * 5 * k * (1 - p), math.cos(t * 27) * 4 * k * (1 - p))
            s.rushdust(f, t, C, 700, 1.4, (1.0, 0.97, 0.94), 0.5 * clamp(p * 3), 5, 0.8)
            s.fist(f, t, C[0] + wob[0], C[1] + wob[1] + H * 0.03 * g, zf, clamp(p / 0.10))
            R = FR / 760.0 * zf * H
            tr = cv2.GaussianBlur(s.fbm(0.0016, t * 40, t * 70, 3, 70 * k, 2), (0, 0), 3 * k); d = np.hypot(s.XX - C[0], s.YY - C[1])                                # こぶしの ふちを まく 蒸気
            col = sstep(0.42, 0.66, tr + 0.40 * np.exp(-((d - R * 1.02) / (R * 0.20 + 6 * k)) ** 2) - 0.12) * clamp(p / 0.15) * (1 - 0.5 * g)
            f *= (1 - col * 0.9)[:, :, None]; f += (col * 0.9 * 1.8 * np.clip(0.95 + 0.5 * (tr - 0.5), 0.86, 1.08))[:, :, None] * STEAM[None, None, :]
            sh += 2 + 9 * g
            if p > 0.86: f = f + ((p - 0.86) / 0.14) ** 2 * 1.7
        return f, sh

    def fist(s, f, t, cx, cy, zf, alpha):                                              # 本人の 絵の こぶしを 大きく して 置く(まわりは 丸く 切る)
        W, H = s.W, s.H; img = s.pose(t); fx, fy = s.A(*FIST); sc = zf * H / s.ah
        ox = cx - (fx - s.aw / 2) * sc; oy = cy - (fy - s.ah / 2) * sc
        b0 = np.zeros((H, W, 3), np.float32); b1 = np.ones((H, W, 3), np.float32)
        Kit.draw(s, b0, img, ox, oy, zf); Kit.draw(s, b1, img, ox, oy, zf)
        al = np.clip(1 - (b1 - b0).mean(2), 0, 1); R = FR / 760.0 * zf * H; d = np.hypot(s.XX - cx, s.YY - cy)
        m = al * sstep(R * 1.04, R * 0.90, d) * alpha
        lit = b0 * (0.92 + 0.25 * np.clip(1 - d / (R * 1.1), 0, 1))[:, :, None] + (np.exp(-np.abs(d - R * 0.85) / (R * 0.12)) * 0.10)[:, :, None] * HOT[None, None, :]
        f *= (1 - m)[:, :, None]; f += lit * m[:, :, None]

    # ---- この子だけの曲: しゅっ しゅっ(機関)+ 低い 金管 → 針が 上がる 音 → ぴーっ(汽笛)→ しゅごー → ぽん ぽん ぽん → ごおお → どん ----
    def audio(s, path):
        a = Snd(DUR, 435); G = 146.8; pn = [0, 2, 3, 7, 10]
        def fq(d, o=0): return G * 2 ** ((pn[d % 5] + 12 * (d // 5 + o)) / 12.0)
        hi = a.nz - a.lp(a.nz, 0.6); wn = a.lp(a.nz, 0.2) - a.lp(a.nz, 0.03); lo = a.lp(a.nz, 0.02)
        for i in range(10):                                                          # 機関の しゅっ しゅっ
            tk = 0.3 + i * 0.24; a.a += wn * a.env(tk, 0.01, 0.07) * (0.5 if i % 2 == 0 else 0.3); a.boom(tk, 70, 0.12, 0.08)
        for d in (0, 3): a.note(fq(d, -1), 0.1, 2.6, 0.04, "pad", 0.6)
        for i, d in enumerate((0, 2, 3, 2, 5, 3)): a.note(fq(d, 0), 0.5 + i * 0.36, 0.5, 0.06, "bell")
        a.boom(T_CUT, 60, 0.6, 0.35); a.whoosh(T_CUT - 0.3, T_CUT, 0.2, 0.1)
        for i in range(4): tk = T_CUT + 0.5 + i * 0.13; a.boom(tk, 80, 0.5, 0.18); a.a += hi * a.env(tk, 0.001, 0.03) * 0.5; a.note(fq(5 + i, 0), tk, 0.4, 0.06, "bell")
        for d in (0, 2, 4): a.note(fq(d, 0), T_CUT, 1.9, 0.04, "pad", 0.05)
        n = 16
        for i in range(n):                                                           # ためる: しゅっ しゅっ が 速く なる
            x = i / (n - 1.0); tk = T_ACT + (T_BLOW - T_ACT - 0.1) * (1 - (1 - x) ** 1.6); a.a += wn * a.env(tk, 0.006, 0.05) * (0.3 + 0.5 * x); a.boom(tk, 70 + 40 * x, 0.10 + 0.15 * x, 0.07)
        a.rise(T_ACT, T_BLOW, 160, 520, 0.08); g = np.clip((a.t - T_ACT) / (T_BLOW - T_ACT), 0, 1) * (a.t < T_BLOW); a.a += lo * g ** 2 * 2.5
        wg = a.env(T_BLOW, 0.03, 0.5) * 0.14                                         # 汽笛(2つの 音が うなる)
        a.a += (np.sin(2 * np.pi * (880 * a.t + 6 * np.sin(2 * np.pi * 5 * a.t) / (2 * np.pi))) + 0.8 * np.sin(2 * np.pi * 1108 * a.t)) * wg
        a.boom(T_BLOW, 46, 1.0, 0.7); gs = np.clip((a.t - T_BLOW) / 0.08, 0, 1) * np.exp(-np.clip(a.t - T_BLOW - 0.3, 0, None) / 1.4); a.a += (hi * 0.55 + wn * 0.8) * gs
        for i in range(3): tk = T_RING + i * 0.26; a.boom(tk, 90 - i * 8, 0.6, 0.22); a.a += wn * a.env(tk, 0.004, 0.12) * 0.7
        a.rise(T_FIST - 0.2, T_HIT, 70, 300, 0.14); gr = np.clip((a.t - T_FIST) / (T_HIT - T_FIST), 0, 1) * (a.t < T_HIT); a.a += (lo * 3.0 + wn * 0.9) * gr ** 1.5
        st = (T_HIT - T_FIST - 0.2) / 10
        for i in range(10): a.note(fq(i, 0), T_FIST + 0.1 + i * st, 0.35, 0.06, "bell")
        for j, d in enumerate((0, 3, 5)): a.note(fq(d, 0), T_FIST + j * 0.08, T_HIT - T_FIST + 0.2, 0.05, "pad", 0.5)
        a.boom(T_HIT, 38, 1.6, 0.9); a.boom(T_HIT, 110, 0.4, 0.3); a.a += hi * a.env(T_HIT, 0.001, 0.15) * 0.9 + wn * a.env(T_HIT, 0.003, 0.6) * 1.0
        for j, d in enumerate((0, 2, 4, 7)): a.note(fq(d, 1), T_HIT + 0.02 + j * 0.03, 0.9, 0.05, "bell")
        s.fin_snd(a, DUR, 435); a.save(path)


if __name__ == "__main__":
    r = Steam()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
