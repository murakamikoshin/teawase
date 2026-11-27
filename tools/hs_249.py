"""シェラ(★4 千夜の語り部) 専用の必殺技「千夜ノ灯」— 高品質版(hs_fx)。夜の 都に 千の ランタンが のぼり、さいごの 1つが 目の前に 来て すかし模様の 光で つつむ
星空の 下、丸屋根の 都の 影。屋上に 語り部が 立ち、手の ランタンが 灯る → 都の 窓が 1つずつ 灯る(前ふり) → 目の カットイン+技名 →
ランタンを かかげる: 都から 灯が つぎつぎに のぼり、夜空を うめる → 灯が こちらへ 流れて くる(大きな ぼけ玉が 通りすぎる)→
さいごの 1つが 目の前に: 金具の すかし模様(星と 格子)ごしの 光が 画面を おおい、中の 火に のまれる → 白 → 暗転
使い方: python3 hs_249.py 出力.mp4 / python3 hs_249.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.7
T_CUT, T_RISE, T_COME, T_HIT = 2.7, 4.7, 7.1, 10.0
NIGHT = np.array([0.03, 0.05, 0.15], np.float32); LAMP = np.array([1.0, 0.70, 0.30], np.float32); FLAME = np.array([1.0, 0.92, 0.66], np.float32); BRONZE = np.array([0.20, 0.12, 0.05], np.float32); GOLD = np.array([1.0, 0.80, 0.36], np.float32)
EYE = (357, 259); LANT = (599, 477)


class Shera(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 249, (1.0, 0.75, 0.35), "千夜ノ灯", "Thousand Nights Lamp", W); s.fx_init(249)
        s._vt = [s.text_img(c, hz.SERIF, int(74 * s.k), 3) for c in "千夜ノ灯"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; H = s.H; k = s.k; rg = np.random.RandomState(2490)
        M = np.zeros((H, W), np.float32); base = int(H * 0.70); wins = []                                             # 都の 影(丸屋根・塔)
        x = -20
        while x < W + 20:
            w = int((26 + rg.rand() * 46) * k); h = int((40 + rg.rand() * 120) * k); kind = rg.randint(0, 4); top = base - h
            cv2.rectangle(M, (x, top), (x + w, H), 1.0, -1)
            if kind == 0: cv2.circle(M, (x + w // 2, top), w // 2, 1.0, -1, cv2.LINE_AA); cv2.line(M, (x + w // 2, top - w // 2), (x + w // 2, top - w // 2 - int(14 * k)), 1.0, max(1, int(2 * k)), cv2.LINE_AA)
            elif kind == 1:
                tw = max(4, int(9 * k)); th = int(h * 0.7); cx = x + w // 2; cv2.rectangle(M, (cx - tw // 2, top - th), (cx + tw // 2, top), 1.0, -1); cv2.fillPoly(M, [np.array([(cx - tw, top - th), (cx + tw, top - th), (cx, top - th - int(22 * k))], np.int32)], 1.0, cv2.LINE_AA)
            elif kind == 2: cv2.ellipse(M, (x + w // 2, top), (w // 2, int(w * 0.34)), 0, 180, 360, 1.0, -1, cv2.LINE_AA)
            for _ in range(int(w * h / (700 * k * k)) + 1):
                wx = x + 5 * k + rg.rand() * max(1, w - 10 * k); wy = top + 8 * k + rg.rand() * max(1, h - 10 * k)
                if wy < base + 20 * k: wins.append((wx, wy, rg.rand(), rg.rand()))
            x += w + int(rg.rand() * 8 * k)
        s._city = cv2.GaussianBlur(M, (0, 0), 0.7); s._win = np.array(wins, np.float32); s._win = s._win[np.argsort(s._win[:, 2])]
        s._ln = rg.rand(150, 6).astype(np.float32); s._far = rg.rand(900, 5).astype(np.float32)

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 夜の 都 ----
    def city(s, t, nwin=0.0, lum=1.0, warm=0.0):
        W, H, k = s.W, s.H, s.k
        f = NIGHT[None, None, :] * (0.5 + 1.7 * (s.yn / 0.7).clip(0, 1) ** 1.5)[:, :, None] + np.array([0.07, 0.20, 0.26], np.float32)[None, None, :] * np.exp(-((s.yn - 0.66) / 0.14) ** 2)[:, :, None]
        f = f + s.nebula(t, [(0.04, 0.07, 0.22), (0.12, 0.08, 0.28), (0.35, 0.40, 0.65)], 0.0008, (3.0, 0.6), 0.55, 0)
        s.starfield(f, t, 1000, (0.9, 0.93, 1.0), 2, 0.9 * (1 - 0.4 * warm), (0.0, 0.0))
        f += (np.exp(-((s.yn - 0.68) / 0.10) ** 2) * (0.10 * min(1.0, nwin) + 0.22 * warm))[:, :, None] * LAMP[None, None, :]                     # 灯の てり返し
        m = s._city[:, :, None]; cc = np.array([0.018, 0.022, 0.05], np.float32)[None, None, :] * (0.7 + 0.6 * s.tex(2, s.XX * 1.2, s.YY * 1.2))[:, :, None] + (LAMP * 0.05 * (min(1.0, nwin) + warm))[None, None, :] * (s.yn > 0.55)[:, :, None]
        f = f * (1 - m) + cc * m
        n = int(len(s._win) * clamp(nwin))
        if n > 0:
            w = s._win[:n]; tw = 0.75 + 0.25 * np.sin(t * (2 + 3 * w[:, 3]) + w[:, 2] * 40); s.splat(f, w[:, 0], w[:, 1], tw * 0.8, LAMP * 0.9 + 0.1, 1.2, 0.45)
        py = 0.865                                                                   # 屋上の 手すり
        pm = sstep(py - 0.002, py + 0.002, s.yn)[:, :, None]; pc = np.array([0.05, 0.04, 0.07], np.float32)[None, None, :] * (0.6 + 0.8 * s.tex(1, s.XX * 0.8, s.YY * 0.8))[:, :, None] * (1 + 2.5 * warm) + (LAMP * 0.10 * (0.3 + warm))[None, None, :]
        pc = pc * (1 - 0.5 * sstep(0.9, 0.98, np.abs(np.cos(s.XX * 0.045))) * (s.yn < py + 0.04))[:, :, None]
        f = f * (1 - pm) + pc * pm; f += (np.exp(-((s.yn - py) / 0.0035) ** 2) * 0.35)[:, :, None] * GOLD[None, None, :]
        return (f * lum).astype(np.float32)

    # ---- のぼる ランタン(小さい もの)----
    def lamps(s, f, t, u, come=0.0, v=1.0):
        W, H, k = s.W, s.H, s.k
        fr = s._far; q = (u * (0.10 + 0.10 * fr[:, 2]) - fr[:, 3] * 1.2); ok = q > 0; q = np.clip(q, 0, 2.0)                                # 遠くの 点
        x = fr[:, 0] * W + np.sin(t * (0.6 + fr[:, 4]) + fr[:, 3] * 30) * 10 * k; y = H * 0.70 - q * H * 0.62 * (0.5 + fr[:, 1])
        s.splat(f, x[ok], y[ok] % (H * 0.78), (0.35 + 0.5 * fr[ok, 4]) * np.clip(q[ok] * 6, 0, 1) * v, LAMP * 0.9 + 0.1, 1.1, 0.5)
        L = s.L0(); C_ = s.L0(); ln = s._ln
        for i in range(len(ln)):
            p = ln[i]; q = u * (0.13 + 0.12 * p[2]) - p[3] * 1.6
            if q <= 0: continue
            dp = 0.35 + 0.65 * p[4]; x = p[0] * W + math.sin(t * (0.7 + p[5]) + i) * 14 * k * dp; y = H * 0.72 - q * H * 0.55 * (0.5 + dp)
            if come > 0: g = come ** 2 * (0.6 + 2.2 * p[4]); x = W / 2 + (x - W / 2) * (1 + g * 1.6); y = H * 0.42 + (y - H * 0.42) * (1 + g * 1.6); dp = dp * (1 + g * 1.4)
            if y < -40 * k or x < -40 * k or x > W + 40 * k: continue
            w = max(2, int(5.5 * k * dp)); h = max(3, int(8 * k * dp)); a = min(1.0, q * 5) * v
            cv2.ellipse(L, (int(x), int(y)), (w, h), 0, 0, 360, float(a), -1, cv2.LINE_AA); cv2.ellipse(C_, (int(x), int(y + h * 0.25)), (max(1, w // 2), max(1, h // 2)), 0, 0, 360, float(a), -1, cv2.LINE_AA)
            cv2.line(L, (int(x - w), int(y - h)), (int(x + w), int(y - h)), float(a * 0.3), max(1, int(1.2 * k * dp)), cv2.LINE_AA)
        f += (L * 0.75 + cv2.GaussianBlur(L, (0, 0), 7 * k) * 0.9)[:, :, None] * LAMP[None, None, :] + (cv2.GaussianBlur(C_, (0, 0), 1.2 * k) * 0.9)[:, :, None] * FLAME[None, None, :]

    # ---- 大きな ランタン: 金具の すかし模様(星と 格子)ごしの 光 ----
    def lantern(s, f, c, S, t, v=1.0):
        W, H, k = s.W, s.H, s.k; x = (s.XX - c[0]) / S; y = (s.YY - c[1]) / S; ax = np.abs(x)
        body = sstep(1.0, 0.97, (ax / 0.62) ** 3 + (np.abs(y) / 0.95) ** 3); cap = (ax < 0.36 - 0.5 * (-(y + 0.95)).clip(0, 1)) * (y < -0.90) * (y > -1.30); foot = (ax < 0.40) * (y > 0.90) * (y < 1.04)
        ring = sstep(0.05, 0.03, np.abs(np.hypot(x, y + 1.42) - 0.12))
        n = 3.4; gx = (x * n + 0.5) % 1.0 - 0.5; gy = (y * n + 0.5) % 1.0 - 0.5; hx = (x * n) % 1.0 - 0.5; hy_ = (y * n) % 1.0 - 0.5                     # すかし: 八角の 星 + すみの ひし形
        sd = np.minimum(np.maximum(np.abs(gx), np.abs(gy)), (np.abs(gx) + np.abs(gy)) / 1.4142); dd_ = np.abs(hx) + np.abs(hy_)
        soft = min(0.05, 3.0 / (S * 0.2 + 1)) + 0.012
        hole = np.maximum(sstep(0.34, 0.34 - soft, sd), sstep(0.17, 0.17 - soft, dd_)) * body * (np.abs(y) < 0.84) * (ax < 0.57)
        lat = 1 - np.clip(np.minimum(np.abs(sd - 0.34), np.abs(dd_ - 0.17)) / 0.05, 0, 1)                                                         # ふちの 金の てり
        band = body * ((np.abs(np.abs(y) - 0.87) < 0.04) | (ax > 0.56))
        d = np.hypot(x, (y - 0.15) * 0.8); fl = np.exp(-(d / 0.45) ** 2) * (0.9 + 0.12 * math.sin(t * 13) + 0.08 * math.sin(t * 29)); glow = 0.35 + 1.4 * fl
        metal = (BRONZE[None, None, :] * (0.5 + 1.2 * s.tex(2, x * 70, y * 70))[:, :, None] + (GOLD * 0.55)[None, None, :] * (lat * body * 0.9 + band * 0.5)[:, :, None]) * (0.55 + 0.9 * fl)[:, :, None]
        solid = np.clip(body + cap + foot + ring, 0, 1) * v
        f *= (1 - solid)[:, :, None]; f += metal * solid[:, :, None]
        f += (hole * glow * v)[:, :, None] * (LAMP[None, None, :] * (1 - fl.clip(0, 1))[:, :, None] + FLAME[None, None, :] * fl.clip(0, 1)[:, :, None]) * 1.25
        f += (np.exp(-(np.hypot(x, y) / 1.6) ** 2) * 0.30 * v)[:, :, None] * LAMP[None, None, :]

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        w = getattr(s, '_wm', 0.3)
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(FLAME * w + np.array([0.7, 0.8, 1.0]) * (1 - w)), rim_dir=(0.8, -0.3), rim_v=(0.35 + 0.5 * w) * alpha, amb=tuple(np.array([0.6, 0.75, 1.0]) * (1 - w) + np.array([1.0, 0.8, 0.55]) * w), amb_v=0.16, aura=LAMP, aura_v=0.14 * w, alpha=alpha, shade=0.72 + 0.22 * w)

    def frame(s, t):
        o = {}; s._wm = 0.3
        if t < T_CUT: f, sh = s.sc_night(t)
        elif t < T_RISE: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.92)
        else: f, sh = s.sc_lamps(t)
        f, sh = s.finale(f, sh, t, DUR, None, (1, 0.8, 0.4), (1, 0.6, 0.2), -1)
        return s.film(f, t, sh * s.k, 0.6, o.get("thr", 0.80), (1.0, 0.85, 0.65), 1.0 + 0.15 * min(sh, 8), 0.04, o.get("vig", 0.58), 1.15, 1.10)

    def teller(s, f, t, alpha=1.0, lift=0.0, **kw):
        W, H, k = s.W, s.H, s.k; cy = H * (0.675 - 0.012 * lift)
        img = s.pose(t, **kw); s.draw(f, img, W / 2, cy, 0.42, alpha)
        lp = s.at(*s.A(*LANT)); s.flare(f, lp, (9 + 3 * math.sin(t * 7) ** 2 + 8 * lift) * k * s._wm ** 0.5, FLAME, 0.9 * alpha * min(1.0, s._wm * 2), 0.5, 0.0, False)
        return lp

    # 1) 夜の 都。手の ランタンが 灯り、都の 窓が 1つずつ 灯る
    def sc_night(s, t):
        W, H, k = s.W, s.H, s.k; lit = ease((t - 0.7) / 0.5); nw = clamp((t - 1.2) / 1.5) ** 1.5
        s._wm = 0.05 + 0.5 * lit
        f = s.city(t, nw * 0.7, 1.0, 0.1 * lit); s.teller(f, t, 1.0, 0.0, sway=0.15, breath=0.5)
        return f * clamp(t / 0.5), 0

    # 2) 目の カットイン(ランタンの 窓の 形)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT; s._wm = 0.8
        f = s.city(t, 0.7, 0.32, 0.1)
        a = ease(u / 0.28); cy = H * 0.33; hh = H * 0.115
        band = (BRONZE * 0.9)[None, None, :] + (LAMP * 0.55)[None, None, :] * (0.35 + 0.65 * np.exp(-((s.YY - cy) / (hh * 0.9)) ** 2) * (0.8 + 0.3 * s.tex(1, s.XX * 0.6 + t * 20, s.YY * 0.6)))[:, :, None]
        ex, ey = s.A(*EYE); zf = 1.5; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(band, img, W / 2 - (ex - s.aw / 2) * sc - (1 - a) * W * 0.9 + u * 5 * k, cy - (ey - s.ah / 2) * sc, zf)
        edge = hh * a * (0.84 + 0.16 * np.abs(np.cos(s.XX * 3.1416 / (W / 5.0)))); dd = np.abs(s.YY - cy)
        m = (sstep(edge, edge - 2, dd) * (a > 0.02))[:, :, None]; f = f * (1 - m) + band * m
        s.glowmask(f, np.exp(-((dd - edge) / (1.6 * k)) ** 2) * (a > 0.02), GOLD, 0.9 * a, 1.0, 0.8, 8.0)
        n = len(s._vt); step = W * 0.20
        for i, T in enumerate(s._vt):                                                # 技名: 灯が ともるように 1文字ずつ
            p = (u - 0.5 - i * 0.15) / 0.3
            if p > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.66, clamp(p) * (0.85 + 0.15 * math.sin(t * 9 + i)), 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.35 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.66 - 76 * k, 1.0, 1.0)
        if u > 1.3:
            g = ease((u - 1.3) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.66 + 58 * k), (W / 2 + W * 0.42 * g, H * 0.66 + 58 * k)], GOLD, 1.0, 0.8, 0.8)
            s.text(f, s.txt_en, W / 2, H * 0.66 + 84 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.6
        return f, 3 * math.exp(-u * 6)

    # 3) 千の 灯が のぼる → こちらへ 流れて くる → さいごの 1つの すかし模様に つつまれる
    def sc_lamps(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_RISE; lift = ease(u / 0.6); uc = t - T_COME; come = clamp(uc / (T_HIT - T_COME))
        TL = T_COME + 0.5; pl = clamp((t - TL) / (T_HIT - TL)); big = pl ** 2.6
        s._wm = 0.6 + 0.4 * lift
        f = s.city(t, 0.7 + 0.3 * clamp(u / 1.5), 1.0 - 0.35 * big, 0.1 + 0.9 * clamp(u / 2.2))
        s.lamps(f, t, (u + 0.3) * 1.8, come, 1 - 0.5 * big)
        a_ = 1 - clamp((pl - 0.45) / 0.25)
        if a_ > 0.02: s.teller(f, t, a_, lift, sway=0.2, breath=0.4, lean=3 * s.k * lift, squash=-0.02 * lift)
        if uc > 0:                                                                   # 通りすぎる 大きな ぼけ玉
            B = np.zeros((H, W), np.float32)
            for i in range(9):
                p = (uc - i * 0.22) / 0.9
                if 0 < p < 1:
                    b = s._ln[i]; a = b[0] * 6.2832; rr = p ** 2 * W * 0.9; cv2.circle(B, (int(W / 2 + math.cos(a) * rr), int(H * 0.42 + math.sin(a) * rr * 1.3)), int((10 + 150 * p ** 2) * k), float(0.5 * math.sin(p * 3.14)), -1, cv2.LINE_AA)
            f += cv2.GaussianBlur(B, (0, 0), 9 * k)[:, :, None] * LAMP[None, None, :]
        sh = 0.0
        if pl > 0:
            c = (W / 2 + W * 0.10 * (1 - pl) * math.sin(t * 1.6), H * (0.30 + 0.16 * ease2(pl)) + 6 * k * math.sin(t * 2.2)); S = k * (12 + 70 * pl) + H * 1.5 * big
            s.lantern(f, c, S, t, min(1.0, pl * 5))
            f = s.godrays(f, (c[0], c[1] + S * 0.15), 0.35 * pl + 0.3 * big, 0.95, 8, 0.88, 0.45); sh = 4 * big
            if pl > 0.86: f = f + ((pl - 0.86) / 0.14) ** 2 * 1.4
        if u < 0.12: f = f + (1 - u / 0.12) * 0.35
        return f, sh

    # ---- この子だけの曲: ウードふうの つまびき(中東の 音階)と 鈴。灯が のぼるほど 音が かさなり、さいごは あたたかい 和音に つつまれる ----
    def audio(s, path):
        a = Snd(DUR, 249); D = 146.83; hj = [0, 1, 4, 5, 7, 8, 11]
        def fq(d, o=0): return D * 2 ** ((hj[d % 7] + 12 * (d // 7 + o)) / 12.0)
        a.a += a.tone(D / 2) * 0.07 * np.clip(a.t / 1.5, 0, 1) * a.gate(0, T_HIT, 0.3)
        a.note(fq(7, 1), 0.7, 1.2, 0.14, "bell")                                                                      # ランタンが 灯る
        rg = np.random.RandomState(249)
        for i in range(14): a.note(fq(int(rg.rand() * 7) + 7, 1), 1.2 + (i / 14) ** 0.7 * 1.4, 0.3, 0.05, "bell")      # 窓が 灯る
        pl_ = [0, 4, 2, 4, 5, 4, 2, 1, 0, 2, 4, 7, 5, 4, 2, 4]
        for i, d in enumerate(pl_[:8]): a.note(fq(d, 0), 0.5 + i * 0.27, 0.35, 0.09, "fm"); a.note(fq(d, 0) * 1.003, 0.5 + i * 0.27 + 0.01, 0.3, 0.05, "saw")
        a.boom(T_CUT, 55, 0.8, 0.4); a.whoosh(T_CUT - 0.3, T_CUT, 0.25)
        for d in (0, 2, 4): a.note(fq(d, 0), T_CUT, 1.9, 0.045, "pad", 0.05)
        for i in range(4): a.note(fq([0, 4, 7, 11][i], 1), T_CUT + 0.5 + i * 0.15, 0.6, 0.11, "bell")
        st = 0.2; nb = int((T_HIT - T_RISE) / st)
        for i in range(nb):                                                                                           # つまびき: だんだん 厚く
            tk = T_RISE + i * st; g = 0.6 + 0.8 * i / nb; d = pl_[i % 16]
            a.note(fq(d, 0), tk, 0.3, 0.07 * g, "fm"); a.note(fq(d, 0) * 1.003, tk + 0.01, 0.25, 0.04 * g, "saw")
            if i % 4 == 0: a.note(fq([0, 3, 4, 0][(i // 4) % 4], -1), tk, 0.7, 0.10, "saw", 0.01); a.boom(tk, 75, 0.25, 0.1)
            if i % 2 == 1 and i > 8: a.hat(tk, 0.035)
            if i > nb // 3: a.note(fq(d + 7, 1), tk + st / 2, 0.25, 0.04 * g, "bell")
        for i in range(40): a.note(fq(int(rg.rand() * 7) + 7, 2), T_RISE + 0.5 + rg.rand() * (T_HIT - T_RISE - 0.6), 0.2, 0.02, "sine")   # 灯の きらめき
        for j, d in enumerate((0, 2, 4, 7, 9)): a.note(fq(d, 0), T_COME + j * 0.1, T_HIT - T_COME, 0.045, "pad", 0.6)
        a.rise(T_COME + 0.5, T_HIT, 90, 260, 0.10); a.whoosh(T_COME + 1.0, T_HIT, 0.6, 0.3)
        a.boom(T_HIT, 42, 1.3, 0.8)
        for j, d in enumerate((0, 4, 7, 11, 14)): a.note(fq(d, 1), T_HIT + 0.02 + j * 0.03, 0.7, 0.06, "bell")
        s.fin_snd(a, DUR, 249); a.save(path)


if __name__ == "__main__":
    r = Shera()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
