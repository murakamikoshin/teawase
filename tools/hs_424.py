"""ドレイク(★6 七つの海の大船長) 専用の必殺技「七海制覇」— 高品質版(hs_fx)。合図の 一発で 朝ぎりの 水平線に 七せきの 艦隊が 灯を ともし、砲火を ならべ、旗艦が 波を けたてて まっすぐ 来て へさきが 画面に つっこむ
朝やけの 波止場に 立つ(きり)→ 羅針図の 星形の 窓の カットイン+技名 → 短銃を 空へ(信号弾が のぼって はじける)→
きりが はれ 水平線に 七せきの 影、灯が 1つずつ ともる → 寄る: 左右の 六せきが 順に 砲火 → 旗艦が ぐんぐん 大きく →
へさき(板・金の 錨の かざり・へさきの 波しぶき)が 画面を うめる → 白 → 暗転
使い方: python3 hs_424.py 出力.mp4 / python3 hs_424.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 12.6
T_CUT, T_ACT, T_SHOT, T_FLEET, T_RAM, T_HIT = 2.8, 4.8, 5.1, 6.9, 9.3, 11.7
INDIGO = np.array([0.09, 0.11, 0.30], np.float32); PEACH = np.array([1.0, 0.70, 0.52], np.float32); GOLD = np.array([1.0, 0.80, 0.36], np.float32); SEAD = np.array([0.05, 0.16, 0.26], np.float32); SEAL = np.array([0.20, 0.42, 0.50], np.float32)
FOGC = np.array([0.96, 0.86, 0.80], np.float32); FOAM = np.array([0.96, 0.98, 1.0], np.float32); HULL = np.array([0.20, 0.11, 0.07], np.float32); SAIL = np.array([0.98, 0.92, 0.80], np.float32); STONE = np.array([0.32, 0.30, 0.33], np.float32)
EYE = (400, 335); GUN = (700, 240); BLADE = (60, 130)
FLEETX = [0.0, -0.13, 0.13, -0.27, 0.27, -0.40, 0.40]; FLEETS = [1.0, 0.8, 0.8, 0.64, 0.64, 0.5, 0.5]
LAMP = [5.9 + i * 0.13 for i in range(7)]; BROAD = [7.2 + i * 0.26 for i in range(6)]


class Drake(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 424, (1.0, 0.82, 0.4), "七海制覇", "Seven Seas Conquest", W); s.fx_init(424)
        s._vt = [s.text_img(c, hz.SERIF, int(84 * s.k), 3) for c in "七海制覇"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; s.hy = 0.50

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 朝やけの 海(きり fog 0..1)。quay=波止場の 石だたみの 上の ふちの 高さ(1.2 で 見えない) ----
    def dawn(s, t, lum=1.0, fog=1.0, quay=0.80, rough=0.0):
        W, H, k = s.W, s.H, s.k; hy = s.hy; v = (s.yn / hy).clip(0, 1)
        f = INDIGO[None, None, :] * (1 - v ** 1.2)[:, :, None] + np.array([0.55, 0.36, 0.55], np.float32)[None, None, :] * (v * (1 - v) * 2.0)[:, :, None] + (PEACH * 1.05)[None, None, :] * (v ** 2.6)[:, :, None]
        n = s.fbm(0.0009, t * 5, 0, 3, 80 * k, 1); n = cv2.GaussianBlur(n, (0, 0), 4 * k); cu = sstep(0.50, 0.68, n + 0.10 * np.exp(-((s.yn - 0.22) / 0.14) ** 2))
        lit = np.clip(0.45 + 1.5 * (s.fbm(0.003, t * 8, 3, 3, 0, 2) - 0.4) + 0.8 * v, 0.3, 1.4)
        f = f * (1 - cu * 0.8)[:, :, None] + (cu * 0.8)[:, :, None] * (np.array([0.30, 0.22, 0.40], np.float32)[None, None, :] + (np.array([1.0, 0.66, 0.50], np.float32) * 0.85)[None, None, :] * lit[:, :, None])
        f += (np.exp(-np.hypot(s.xn - 0.5, (s.yn - hy) * 1.6) / 0.17) * 0.9)[:, :, None] * np.array([1.0, 0.88, 0.60], np.float32)[None, None, :]                     # のぼる 日
        d = np.clip((s.yn - hy) / (1 - hy), 0, 1); pu = (s.XX - W / 2) / (d + 0.08) * 0.16; pv = 46.0 / (d + 0.08)
        w1 = s.tex(2, pu + t * 12, pv + t * (22 + 30 * rough)); w2 = s.tex(0, pu * 2.1 - t * 18, pv * 2.1 + t * (30 + 40 * rough)); wv = w1 * 0.6 + w2 * 0.4
        sw = SEAD[None, None, :] * (0.7 + 0.6 * wv)[:, :, None] + SEAL[None, None, :] * (d * (0.3 + 0.9 * wv))[:, :, None] + (sstep(0.74 - 0.12 * rough, 0.92, wv) * (0.15 + 0.6 * d))[:, :, None] * FOAM[None, None, :]
        sw += (np.exp(-np.abs(s.xn - 0.5) / (0.07 + 0.22 * d)) * sstep(0.40, 0.8, w2) * (1.0 - 0.5 * d))[:, :, None] * GOLD[None, None, :]                           # 日の 道
        m = sstep(hy - 0.002, hy + 0.003, s.yn)[:, :, None]; f = f * (1 - m) + sw * m
        if fog > 0.01:                                                               # 水平線の 朝ぎり
            fb = np.exp(-((s.yn - hy - 0.01) / (0.05 + 0.03 * s.tex(2, s.XX * 0.25 + t * 6, 5))) ** 2) * fog * (0.75 + 0.3 * s.tex(1, s.XX * 0.3 - t * 9, s.YY * 0.8))
            f = f * (1 - np.clip(fb, 0, 1) * 0.92)[:, :, None] + (np.clip(fb, 0, 1) * 0.92)[:, :, None] * FOGC[None, None, :]
        if quay < 1.1:                                                               # 波止場の 石だたみ
            dq = np.clip((s.yn - quay) / (1.0 - 0.80), 0, 1.5); qu = (s.XX - W / 2) / (dq * 0.8 + 0.5); row = np.floor(dq * 5.0 ** 1.0 + 0.0); bx = (qu / (96 * k) + row * 0.5); by = dq * 5.0
            jl = np.maximum(np.exp(-((np.abs((bx % 1.0) - 0.5) - 0.5) / 0.03) ** 2), np.exp(-((np.abs((by % 1.0) - 0.5) - 0.5) / 0.05) ** 2))
            sc_ = STONE[None, None, :] * ((0.55 + 0.9 * s.tex(1, qu * 0.6 + row * 31, by * 40)) * (1 - 0.55 * jl) * (0.7 + 0.4 * dq))[:, :, None] + (PEACH * 0.30)[None, None, :] * (np.exp(-dq / 0.12))[:, :, None]
            m = sstep(quay - 0.002, quay + 0.003, s.yn)[:, :, None]; f = f * (1 - m) + sc_ * m
        return (f * lum).astype(np.float32)

    def haze(s, f, t, v=1.0):
        W, H = s.W, s.H; r = s.rnd[:80]
        x = ((r[:, 0] + t * (0.02 + 0.03 * r[:, 4])) % 1.0) * W; y = ((r[:, 1] - t * (0.02 + 0.03 * r[:, 2])) % 1.0) * H
        s.splat(f, x, y, (0.3 + 0.6 * r[:, 2]) * v, GOLD, 1.2, 0.3)

    # ---- とおくの 船(かげ絵): 正面を 向いた 帆船。fogm で きりの 色に とける ----
    def farship(s, f, cx, wy, S, v=1.0, fogm=0.5, lamp=0.0):
        if v <= 0.01 or S < 3: return
        P = lambda x, y: (int(cx + x * S), int(wy + y * S)); M = s.L0(); Q = s.L0()
        cv2.fillPoly(M, [np.array([P(-0.30, -0.50), P(0, -0.60), P(0.30, -0.50), P(0.22, -0.22), P(0, 0.03), P(-0.22, -0.22)], np.int32)], 1.0, cv2.LINE_AA)
        for mx, sc in ((-0.36, 0.62), (0.36, 0.62), (0.0, 1.0)):
            cv2.line(M, P(mx, -0.5), P(mx, -0.5 - 1.28 * sc), 1.0, max(1, int(S * 0.02)), cv2.LINE_AA)
            for yb, yt, hw in ((-0.66, -1.00, 0.50), (-1.04, -1.36, 0.42), (-1.40, -1.66, 0.30)):
                pts = [P(mx - hw * sc, -0.5 + (yt + 0.5) * sc), P(mx + hw * sc, -0.5 + (yt + 0.5) * sc), P(mx + hw * sc * 1.04, -0.5 + (yb + 0.5) * sc), P(mx, -0.5 + (yb + 0.5 + 0.05) * sc), P(mx - hw * sc * 1.04, -0.5 + (yb + 0.5) * sc)]
                cv2.fillPoly(Q, [np.array(pts, np.int32)], 1.0, cv2.LINE_AA)
        bl = max(0.6, S * 0.012); M = cv2.GaussianBlur(M, (0, 0), bl); Q = cv2.GaussianBlur(Q, (0, 0), bl)
        hc = HULL * (1 - fogm) + FOGC * 0.8 * fogm; sc2 = (SAIL * 1.0) * (1 - fogm) + FOGC * 0.92 * fogm
        tx = 0.85 + 0.3 * s.tex(1, s.XX * 0.7, s.YY * 0.7)
        f[:] = f * (1 - Q * v)[:, :, None] + (Q * v * tx)[:, :, None] * sc2[None, None, :]; f[:] = f * (1 - M * v)[:, :, None] + (M * v * tx)[:, :, None] * hc[None, None, :]
        fo = np.exp(-(((s.XX - cx) / (S * 0.42)) ** 2 + ((s.YY - wy) / (S * 0.07)) ** 2)) * v * 0.7; f[:] = f + fo[:, :, None] * FOAM[None, None, :] * (1 - 0.5 * fogm)
        if lamp > 0.01: s.glowmask(f, np.exp(-(np.hypot(s.XX - cx, s.YY - (wy - 0.62 * S)) / max(2.0, S * 0.06)) ** 2), GOLD, 1.6 * lamp * v, 1.0, 0.8, max(3.0, S * 0.12 / s.k))

    # ---- 旗艦(大写し): 板の すじ・金の ふち・へさきの 金の 錨・帆の かげ・へさきの 波 ----
    def flagship(s, f, cx, wy, S, t, v=1.0):
        W, H, k = s.W, s.H, s.k; x = (s.XX - cx) / S; y = (s.YY - wy) / S; ax = np.abs(x)
        for mx, sc in ((-0.36, 0.62), (0.36, 0.62), (0.0, 1.0)):                    # 帆(うしろから 前へ)
            lx = (x - mx) / sc; ly = (y + 0.5) / sc; sm = np.zeros_like(x); shd = np.zeros_like(x)
            for yb, yt, hw in ((-0.16, -0.50, 0.50), (-0.54, -0.86, 0.42), (-0.90, -1.16, 0.30)):
                vv = np.clip((ly - yt) / (yb - yt), 0, 1); wdt = hw * (1 + 0.04 * vv); bot = yb + 0.05 * (1 - (lx / wdt) ** 2)
                mk = sstep(0.0, 0.01, ly - yt) * sstep(0.0, -0.01, ly - bot) * sstep(wdt, wdt - 0.012, np.abs(lx)); sm = np.maximum(sm, mk)
                shd = np.where(mk > 0.01, 0.88 + 0.40 * np.sin(np.clip(vv, 0, 1) * 2.6) * (1 - 0.35 * (lx / wdt) ** 2) + 0.10 * np.sin(lx * 60 / 1.0) * 0.0, shd)
            mast = sstep(0.012, 0.006, np.abs(lx)) * sstep(0.0, 0.01, ly + 1.30) * sstep(0.02, 0.0, ly)
            sc3 = SAIL[None, None, :] * (shd * (0.90 + 0.18 * s.tex(1, lx * 300, ly * 300)) * (0.82 if mx else 1.0))[:, :, None] + (PEACH * 0.25)[None, None, :] * (shd * np.clip(-lx, 0, 1))[:, :, None]
            f[:] = f * (1 - mast * v * 0.9)[:, :, None] + (mast * v * 0.9)[:, :, None] * (HULL * 0.8)[None, None, :]
            f[:] = f * (1 - sm * v)[:, :, None] + sc3 * (sm * v)[:, :, None]
        top = -0.50 - 0.10 * np.clip(1 - ax / 0.30, 0, 1) ** 1.3; side = 0.30 - 0.30 * np.clip((y + 0.50) / 0.53, 0, 1) ** 1.6                                      # 船体(正面)
        hm = sstep(0.0, 0.006, y - top) * sstep(0.0, 0.008, side - ax) * sstep(0.04, 0.03, y)
        plank = 1 - 0.55 * np.exp(-((np.abs(((y + ax * 0.55) / 0.045) % 1.0 - 0.5) - 0.5) / 0.07) ** 2)
        rnd_ = 0.45 + 0.75 * np.clip(1 - ax / (side + 0.02), 0, 1) ** 0.6
        hc = HULL[None, None, :] * (plank * rnd_ * (0.6 + 0.9 * s.tex(1, x * 420, np.floor((y + ax * 0.55) / 0.045) * 17 + y * 60)) * 1.5)[:, :, None] + (PEACH * 0.22)[None, None, :] * (np.clip(-x / 0.3, 0, 1) * rnd_)[:, :, None]
        trim = np.exp(-((y - top - 0.03) / 0.012) ** 2) + np.exp(-((y - top - 0.11) / 0.006) ** 2) * 0.7                                                            # 金の ふち
        hc = hc * (1 - np.clip(trim, 0, 1) * 0.9)[:, :, None] + (np.clip(trim, 0, 1) * (0.6 + 0.6 * s.tex(0, x * 300, y * 300)))[:, :, None] * GOLD[None, None, :]
        stem = np.exp(-(ax / 0.014) ** 2); hc = hc + (stem * 0.35)[:, :, None] * GOLD[None, None, :]                                                                 # へさきの 柱
        ex, ey = x, y + 0.33; er = np.hypot(ex, ey)                                                                                                                # 金の 錨の かざり
        emb = np.maximum.reduce([np.exp(-((er - 0.085) / 0.008) ** 2), np.exp(-(ex / 0.008) ** 2) * (np.abs(ey) < 0.062), np.exp(-((ey + 0.035) / 0.007) ** 2) * (np.abs(ex) < 0.035), np.exp(-((np.hypot(ex, ey - 0.005) - 0.045) / 0.008) ** 2) * (ey > 0.005)])
        disc = sstep(0.085, 0.080, er); hc = hc * (1 - disc * 0.6)[:, :, None]; hc = hc + (emb * 1.2 * (0.7 + 0.5 * s.tex(0, x * 500, y * 500)))[:, :, None] * GOLD[None, None, :]
        f[:] = f * (1 - hm * v)[:, :, None] + hc * (hm * v)[:, :, None]
        s.glowmask(f, np.exp(-(np.hypot(x, y + 0.62) / 0.03) ** 2), GOLD, 1.2 * v, 1.0, 0.8, max(3.0, S * 0.08 / k))                                               # へさきの 灯
        bw = np.exp(-((y - 0.0) / (0.05 + 0.10 * np.clip(ax / 0.5, 0, 1))) ** 2) * np.exp(-(ax / 0.55) ** 2) + np.exp(-(((ax - 0.20 - np.clip(-y, 0, 1) * 0.5) / 0.10) ** 2)) * sstep(-0.42, -0.05, y) * sstep(0.06, 0.0, y) * 0.9   # へさきの 波
        fo = np.clip(bw * (0.55 + 0.8 * s.tex(2, x * 160 + t * 60, y * 160 - t * 150)), 0, 1) * v
        f[:] = f * (1 - fo * 0.85)[:, :, None] + (fo * 1.05)[:, :, None] * FOAM[None, None, :]

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        s.draw_lit(f, img, cx, cy, hfrac, rim=(1.0, 0.85, 0.6), rim_dir=(0.0, -1.0), rim_v=0.6 * alpha, amb=(1.0, 0.86, 0.8), amb_v=0.14, aura=GOLD, aura_v=0.08, alpha=alpha, shade=0.93)

    def frame(s, t):
        o = {}
        if t < T_CUT: f, sh = s.sc_quay(t)
        elif t < T_ACT: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.5, thr=0.95)
        else: f, sh = s.sc_fleet(t)
        f, sh = s.finale(f, sh, t, DUR, None, (1.0, 0.95, 0.85), (1.0, 0.8, 0.4), -1)
        return s.film(f, t, sh * s.k, 0.45, o.get("thr", 0.93), (1.0, 0.96, 0.9), 1.0 + 0.15 * min(sh, 8), 0.03, o.get("vig", 0.45), 1.2, 0.96)

    def captain(s, f, t, alpha=1.0, dy=0.0, **kw):
        W, H, k = s.W, s.H, s.k; cy = H * 0.635 + dy; fy = cy + 0.44 * H * 0.47
        sd = np.exp(-(((s.XX - W * 0.5) / (W * 0.26)) ** 2 + ((s.YY - fy) / (H * 0.014)) ** 2)); f *= (1 - 0.55 * alpha * sd)[:, :, None]
        img = s.pose(t, **kw); s.draw(f, img, W / 2, cy, 0.44, alpha); return s.at(*s.A(*GUN)), s.at(*s.A(*BLADE))

    # 1) 朝やけの 波止場に 立つ
    def sc_quay(s, t):
        f = s.dawn(t); s.captain(f, t, clamp(t / 0.6), sway=0.25, breath=0.5); s.haze(f, t, 0.5)
        return f * clamp(t / 0.5), 0

    # 2) 目の カットイン(羅針図の 八方星の 窓)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.dawn(t, 0.32)
        a = back(clamp(u / 0.4)); c = (W / 2, H * 0.32); R = H * 0.215 * a
        bandc = (INDIGO * 1.3)[None, None, :] + (GOLD * 0.75)[None, None, :] * (0.25 + 0.9 * sstep(0.45, 0.9, s.tex(0, s.XX * 0.4 + t * 40, s.YY * 0.4 - t * 25)))[:, :, None]
        ex, ey = s.A(*EYE); zf = 1.25; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(bandc, img, c[0] - (ex - s.aw / 2) * sc, c[1] - (ey - s.ah / 2) * sc, zf)
        if R > 6:
            d = np.hypot(s.XX - c[0], s.YY - c[1]); th = np.arctan2(s.YY - c[1], s.XX - c[0]) + 0.10 * math.sin(t * 0.8)
            star = R * (0.70 + 0.30 * np.abs(np.cos(th * 2)) ** 5 + 0.14 * np.abs(np.sin(th * 2)) ** 9)                                                     # 4つの 長い はり+4つの 短い はり
            dd = d - star; m = sstep(0, -2, dd)[:, :, None]; f = f * (1 - m) + bandc * m
            s.glowmask(f, np.exp(-(dd / (2.4 * k)) ** 2), GOLD, 1.0 * a, 1.0, 0.8, 7.0)
            ring = np.exp(-((d - R * 1.12) / (1.6 * k)) ** 2) * (0.5 + 0.5 * (np.sin(th * 32) > 0)); s.glowmask(f, ring, GOLD, 0.7 * a, 1.0, 0.5, 4.0)                 # 目もりの 輪
        n = len(s._vt); step = W * 0.215
        for i, T in enumerate(s._vt):                                                # 技名: 日の 出の ように 下から
            p = (u - 0.55 - i * 0.14) / 0.4
            if p > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.73 + (1 - ease(p)) * 36 * k, clamp(p) ** 1.2, 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.4 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.73 - 82 * k, 1.0, 1.0)
        if u > 1.4:
            g = ease((u - 1.4) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.73 + 62 * k), (W / 2 + W * 0.42 * g, H * 0.73 + 62 * k)], GOLD, 1.0, 0.7, 0.7)
            s.text(f, s.txt_en, W / 2, H * 0.73 + 88 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.5
        return f, 3 * math.exp(-u * 6)

    # 3) 合図 → 艦隊が あらわれる → 寄る・砲火 → 旗艦の へさきが つっこむ
    def sc_fleet(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_ACT; sh = 0.0; hy = s.hy
        g = ease2(clamp((t - T_FLEET) / (T_RAM - T_FLEET))); e = clamp((t - T_RAM) / (T_HIT - T_RAM))
        fogv = 1.0 - 0.85 * ease2(clamp((t - 5.5) / 1.2)); drop = ease2(clamp((t - T_FLEET) / 1.1))
        f = s.dawn(t, 1.0 - 0.12 * e, fogv, 0.80 + 0.45 * drop, g + e)
        app = clamp((t - 5.55) / 0.9)                                                # 艦隊
        if app > 0:
            for i in (5, 6, 3, 4, 1, 2):
                S = k * 26 * FLEETS[i] * math.exp(g * 1.5 + e * 1.2); cx = W * (0.5 + FLEETX[i] * (1 + 1.3 * g + 4.0 * e ** 1.5)); wy = H * (hy + 0.012) + S * 0.10 + H * 0.10 * g * FLEETS[i] + H * 0.5 * e ** 1.5
                if -S < cx < W + S and wy - 2 * S < H:
                    s.farship(f, cx, wy, S, app, 0.75 * fogv + 0.25 * (1 - g), clamp((t - LAMP[i]) / 0.15))
                    j = i - 1; q = t - BROAD[j]
                    if 0 < q < 1.2:                                                  # 砲火
                        bx = cx + (S * 0.35 if FLEETX[i] < 0 else -S * 0.35); by = wy - S * 0.28
                        if q < 0.12: s.glowmask(f, np.exp(-(np.hypot(s.XX - bx, s.YY - by) / (S * 0.30)) ** 2), (1.0, 0.85, 0.5), 2.5 * (1 - q / 0.12), 1.0, 0.8, 8.0); sh += 6 * (1 - q / 0.12)
                        pm = np.exp(-(np.hypot(s.XX - bx, (s.YY - by + q * S * 0.3) * 1.2) / (S * (0.18 + 0.35 * q))) ** 2) * clamp(1.2 - q) * (0.6 + 0.6 * s.tex(2, s.XX * 1.5, s.YY * 1.5 - q * 40))
                        f[:] = f * (1 - np.clip(pm, 0, 1) * 0.8)[:, :, None] + (np.clip(pm, 0, 1) * 0.8)[:, :, None] * FOGC[None, None, :]
            S0 = k * 26 * math.exp(g * 2.3 + e * 3.0); wy0 = H * (hy + 0.012) + S0 * (0.10 + 0.16 * e) + H * 0.10 * g
            if S0 < 70 * k: s.farship(f, W * 0.5, wy0, S0, app, 0.75 * fogv + 0.25 * (1 - g), clamp((t - LAMP[0]) / 0.15))
            else: s.flagship(f, W * 0.5 + W * 0.012 * math.sin(t * 2.2) * (1 - e), wy0 + S0 * 0.012 * math.sin(t * 3.1), S0, t, 1.0)
            sh += 3 * g ** 2 + 9 * e ** 2
        dy = H * 0.62 * drop                                                         # 船長(寄ると 下へ 流れる)
        if drop < 0.98:
            st = t - T_SHOT
            gp, bp = s.captain(f, t, 1.0, dy, sway=0.2, breath=0.4, squash=(0.04 * ease(u / (T_SHOT - T_ACT)) if st < 0 else -0.04 * math.exp(-st * 5)))
            if 0 <= st < 1.6:                                                        # 信号弾
                if st < 0.10: s.glowmask(f, np.exp(-(np.hypot(s.XX - gp[0], s.YY - gp[1]) / (40 * k)) ** 2), (1.0, 0.9, 0.6), 2.4 * (1 - st / 0.10), 1.0, 0.8, 9.0); sh += 9 * (1 - st / 0.10)
                top = (gp[0] + (W * 0.62 - gp[0]) * 0.6, H * 0.10); p = clamp(st / 0.55); pp = 1 - (1 - p) ** 2; hd = (gp[0] + (top[0] - gp[0]) * pp, gp[1] + (top[1] - gp[1]) * pp)
                if p < 1: s.glowline(f, [(gp[0] + (hd[0] - gp[0]) * max(0, pp - 0.3) / max(pp, 1e-3), gp[1] + (hd[1] - gp[1]) * max(0, pp - 0.3) / max(pp, 1e-3)), hd], GOLD, 1.0, 2.0, 0.9)
                else:
                    q = st - 0.55; n = 90; r = s.rnd[250:250 + n]; a = r[:, 0] * 6.283; rr = k * 150 * (1 - np.exp(-q * (2.5 + 2 * r[:, 1]))); 
                    s.splat(f, top[0] + np.cos(a) * rr, top[1] + np.sin(a) * rr + q * q * 60 * k, (0.5 + 0.8 * r[:, 2]) * clamp(1.4 - q * 1.4) * 1.6, GOLD, 1.5, 0.5)
                    if q < 0.12: s.glowmask(f, np.exp(-(np.hypot(s.XX - top[0], s.YY - top[1]) / (60 * k)) ** 2), (1.0, 0.95, 0.75), 2.0 * (1 - q / 0.12), 1.0, 0.8, 10.0)
        s.haze(f, t, 0.5 * (1 - e))
        if e > 0:                                                                    # しぶきが 画面に
            n = 120; r = s.rnd[100:100 + n]; q = (r[:, 0] + t * (0.6 + 0.8 * r[:, 1])) % 1.0; sx = W * (0.5 + (r[:, 2] - 0.5) * (0.4 + 1.4 * q)); sy = H * (1.05 - q * (0.5 + 0.9 * r[:, 3]) * (0.4 + e))
            s.splat(f, sx, sy, (0.4 + 0.6 * r[:, 3]) * e * 1.5, FOAM, 1.8, 0.4)
            if e > 0.84: f = f + ((e - 0.84) / 0.16) ** 2 * 1.7
        if u < 0.12: f = f + (1 - u / 0.12) * 0.25
        return f, sh

    # ---- この子だけの曲: 低い 金管の ような 主題 → 合図の 一発 → 七つの 鐘(灯)→ 六つの 砲声 → 主題が 高く・波の ごう音 → 一打 ----
    def audio(s, path):
        a = Snd(DUR, 424); G = 116.5; pn = [0, 2, 4, 7, 9]
        def fq(d, o=0): return G * 2 ** ((pn[d % 5] + 12 * (d // 5 + o)) / 12.0)
        for d in (0, 3): a.note(fq(d, -1), 0.1, 2.7, 0.05, "pad", 0.8)
        th1 = [(5, 0.5, 0.5), (8, 1.0, 0.25), (9, 1.25, 0.25), (10, 1.5, 0.7), (8, 2.2, 0.5)]
        for d, tk, ln in th1: a.note(fq(d, 0), tk, ln, 0.08, "fm"); a.note(fq(d, -1), tk, ln, 0.05, "fm")
        wn = a.lp(a.nz, 0.18) - a.lp(a.nz, 0.03); hi = a.nz - a.lp(a.nz, 0.5); lo = a.lp(a.nz, 0.02)
        a.a += wn * (0.5 + 0.5 * np.sin(a.t * 1.4)) ** 2 * (a.t < T_FLEET) * 0.35
        a.boom(T_CUT, 55, 0.7, 0.45); a.whoosh(T_CUT - 0.3, T_CUT, 0.2, 0.1)
        for i in range(4): a.note(fq([5, 8, 10, 12][i], 1), T_CUT + 0.55 + i * 0.14, 0.9, 0.08, "bell"); a.note(fq([5, 8, 10, 12][i], 0), T_CUT + 0.55 + i * 0.14, 0.5, 0.06, "fm")
        for d in (0, 2, 4): a.note(fq(d, 0), T_CUT, 2.0, 0.045, "pad", 0.05)
        a.boom(T_SHOT, 60, 0.5, 0.8); a.a += hi * a.env(T_SHOT, 0.001, 0.10) * 1.0; a.rise(T_SHOT, T_SHOT + 0.55, 500, 1500, 0.05)
        a.boom(T_SHOT + 0.55, 90, 0.4, 0.4); a.a += hi * a.env(T_SHOT + 0.55, 0.002, 0.5) * 0.4
        for i, tk in enumerate(LAMP): a.note(fq(5 + i, 1), tk, 0.9, 0.07, "bell")                                         # 灯が ともる
        for j, d in enumerate((0, 2, 4)): a.note(fq(d, 0), 5.7 + j * 0.1, T_FLEET - 5.6, 0.045, "pad", 0.6)
        th2 = [(10, 0.0, 0.45), (12, 0.5, 0.2), (13, 0.75, 0.2), (15, 1.0, 0.6), (13, 1.65, 0.3), (15, 2.0, 0.3), (17, 2.4, 1.0)]
        for d, tk, ln in th2: a.note(fq(d, 0), T_FLEET + 0.1 + tk, ln, 0.085, "fm"); a.note(fq(d, -1), T_FLEET + 0.1 + tk, ln, 0.06, "fm")
        for tk in BROAD: a.boom(tk, 44, 0.9, 0.6); a.a += (a.lp(a.nz, 0.3) - a.lp(a.nz, 0.05)) * a.env(tk, 0.003, 0.25) * 1.0
        g = np.clip((a.t - T_FLEET) / (T_HIT - T_FLEET), 0, 1) ** 1.7 * (a.t < T_HIT + 0.15); a.a += wn * g * 1.9 + lo * g * 2.6 + hi * g * 0.2
        for d, tk, ln in ((15, 0.0, 0.5), (17, 0.5, 0.5), (19, 1.0, 0.5), (20, 1.5, 0.9)): a.note(fq(d, 0), T_RAM + 0.1 + tk, ln, 0.09, "fm"); a.note(fq(d, 1), T_RAM + 0.1 + tk, ln, 0.05, "bell")
        for j, d in enumerate((0, 2, 4, 7)): a.note(fq(d, 0), T_RAM + j * 0.08, T_HIT - T_RAM + 0.2, 0.05, "pad", 0.6)
        a.rise(T_RAM, T_HIT, 70, 300, 0.12)
        nb = 10
        for i in range(nb): tk = T_RAM + (T_HIT - T_RAM) * (1 - (1 - i / nb) ** 1.6); a.boom(tk, 66, 0.3, 0.2 + 0.05 * i)
        a.boom(T_HIT, 34, 2.0, 1.1); a.a += wn * a.env(T_HIT, 0.004, 0.8) * 2.0
        for j, d in enumerate((0, 2, 4, 7, 9)): a.note(fq(d, 1), T_HIT + 0.02 + j * 0.03, 1.0, 0.05, "bell")
        s.fin_snd(a, DUR, 424); a.save(path)


if __name__ == "__main__":
    r = Drake()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
