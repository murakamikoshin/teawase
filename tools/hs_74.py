"""クロガネ(★6 千の刃) 専用の必殺技「千刃地獄」— 高品質版(hs_fx)・通し直し。千の 刀の 切先が こちらを 向き、いっせいに 飛んでくる
うしろから 灯の さす 障子(和紙の せんい・木の さん)に うつる 大きな 影 → 一筋の 赤い 線が 走り、障子が 左右に ひらく(前ふり: 切り筋) → 筆の 一はけの 中に 鬼の面(カットイン) →
居合の 構えのまま 動かない。まわりの 闇に 鋼の 刀が 三→九→二十七→八十一→二百四十三→千 と ふえ、切先を こちらへ 向けて 輪に ならぶ(数が 漢字で 出る) →
外の 輪から 順に こちらへ 飛んでくる。通りすぎるたび、目の前に 赤い 切り筋が たまっていく → しずかに 納刀「カチン」 → ひと呼吸おいて 切り筋が いっせいに ひらく → 白 → 暗転
使い方: python3 hs_74.py 出力.mp4 / python3 hs_74.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old74", "/home/claude/tools/pre_fin/hs_74_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = 10.3
RED, STEEL, PAPER, WOODC, RINGS, KANJI = old.RED, old.STEEL, old.PAPER, old.WOODC, old.RINGS, old.KANJI
T_SLASH, T_CUT, T_GATHER = old.T_SLASH, old.T_CUT, old.T_GATHER
T_FLY, T_SHEATH, T_BURST = 7.7, 9.0, 9.6
STAGE = [T_GATHER + 0.35 + i * 0.38 for i in range(6)]
NCUT = 36


class Kurogane(FX, old.Kurogane):
    def __init__(s, W=540):
        old.Kurogane.__init__(s, W); s.fx_init(74); H = s.H; k = s.k; s.yv = (s.YY / H).astype(np.float32); s.O = (W * 0.5, H * 0.30)
        rg = np.random.RandomState(741); s.cut2 = [(rg.rand() * 3.1416, W * (0.1 + 0.8 * rg.rand()), H * (0.08 + 0.84 * rg.rand())) for _ in range(NCUT)]
        s._fib = (cv2.remap(s._tiles[0], ((s.XX * 0.25 / k) % 256).astype(np.float32), ((s.YY * 2.4 / k) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP) * 0.55 + cv2.remap(s._tiles[2], ((s.XX * 2.1 / k) % 256).astype(np.float32), ((s.YY * 2.1 / k) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP) * 0.45)
        s._wg = cv2.remap(s._tiles[1], ((s.XX * 3.0 / k) % 256).astype(np.float32), ((s.YY * 0.3 / k) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        gy = H * 0.86; dz = (H - gy) / (np.clip(s.YY - gy, 0, None) + 5 * k)
        s._tat = cv2.remap(s._tiles[3], (((s.XX - W / 2) * dz * 1.6 / k + 128) % 256).astype(np.float32), ((dz * 14) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        s._wall = s.fbm(0.0016, 0, 0, 4, 40 * k, 1); s._eyev = 0.3

    # ---- 灯の ほの暗い 座敷 ----
    def room(s, lum=1.0):
        W, H, k = s.W, s.H, s.k; gy = int(H * 0.86)
        f = np.array([0.085, 0.026, 0.032], np.float32)[None, None, :] * ((0.35 + 1.1 * s.yv) * (0.6 + 0.8 * s._wall))[:, :, None]
        f += (np.exp(-np.hypot((s.XX - W * 0.12) / (W * 0.5), (s.YY - H * 0.70) / (H * 0.35)) * 2.2) * 0.10)[:, :, None] * np.array([1.0, 0.55, 0.25], np.float32)[None, None, :]       # 行灯の ぬくい 光
        fl = np.array([0.075, 0.048, 0.034], np.float32)[None, None, :] * ((0.45 + 1.0 * s._tat) * (0.6 + 0.6 * np.clip((s.YY - gy) / (H - gy), 0, 1)))[:, :, None]
        fl += (np.exp(-((s.XX - W / 2) / (W * 0.35)) ** 2) * 0.05 * s._eyev)[:, :, None] * RED[None, None, :]
        m = sstep(gy - 1.5, gy + 1.5, s.YY)[:, :, None]; f = f * (1 - m) + fl * m
        L = s.L0()
        for i in range(-6, 7): cv2.line(L, (int(W / 2 + i * W * 0.09), gy), (int(W / 2 + i * W * 0.22), H), 0.6, 1, cv2.LINE_AA)         # たたみの へり
        f *= (1 - 0.5 * cv2.GaussianBlur(L, (0, 0), 0.8 * k))[:, :, None]
        f += (np.exp(-((s.YY - gy) / (5 * k)) ** 2) * 0.03)[:, :, None] * np.array([1.0, 0.6, 0.4], np.float32)[None, None, :]
        return (f * lum).astype(np.float32)

    def shoji2(s, op, glow):
        """障子: 和紙(せんい・うしろの 灯)と 木の さん。もどり: 絵, ぜんたいの 形, 紙の 形"""
        W, H, k = s.W, s.H, s.k; off = int(W * 0.52 * op); pane = np.zeros((H, W, 3), np.float32); pm = s.L0(); pap = s.L0()
        bk = 0.50 + 0.55 * np.exp(-(((s.XX - W / 2) / (W * 0.6)) ** 2 + ((s.YY - H * 0.52) / (H * 0.5)) ** 2)); pc = PAPER[None, None, :] * ((0.60 + 0.34 * s._fib) * bk * glow)[:, :, None]
        wc = WOODC[None, None, :] * (0.45 + 1.3 * s._wg)[:, :, None]
        for sg in (-1, 1):
            x0 = (0 if sg < 0 else W // 2) + sg * off; x1 = x0 + W // 2; a, b = max(0, x0), min(W, x1)
            if a >= b: continue
            Lm = s.L0()
            for i in range(4): xx = x0 + int(i * W / 6); cv2.rectangle(Lm, (xx - int(3 * k), 0), (xx + int(3 * k), H), 1.0, -1)
            for j in range(9): yy = int(j * H / 8); cv2.rectangle(Lm, (a, yy - int(3 * k)), (b, yy + int(3 * k)), 1.0, -1)
            ex = x1 if sg < 0 else x0; cv2.rectangle(Lm, (ex - int(7 * k), 0), (ex + int(7 * k), H), 1.0, -1)
            sl = slice(a, b); src = np.clip(np.arange(a, b) - sg * off, 0, W - 1); lm = Lm[:, sl]; sh = cv2.GaussianBlur(Lm, (0, 0), 3 * k)[:, sl]
            pane[:, sl] = pc[:, src] * (1 - 0.22 * sh * (1 - lm))[:, :, None] * (1 - lm[:, :, None]) + wc[:, src] * lm[:, :, None]; pm[:, sl] = 1.0; pap[:, sl] = 1 - lm
        return pane, pm, pap

    def steel(s, f, pf, pn, wn, t, ph=0.0, al=1.0):
        """刀 1本(式): 鋼の はだ・しのぎ・刃文・切先・赤い 柄と 金の つば。pf=つば(奥)、pn=切先(手前)、wn=はば"""
        W, H = s.W, s.H; ax, ay = pn[0] - pf[0], pn[1] - pf[1]; ln = math.hypot(ax, ay)
        if ln < 3 or al <= 0.02: return
        ux, uy = ax / ln, ay / ln; nx, ny = -uy, ux; hl = ln * 0.30; hx, hy = pf[0] - ux * hl, pf[1] - uy * hl; pad = wn * 0.6 + 2
        x0 = int(max(0, min(hx, pn[0]) - pad)); x1 = int(min(W, max(hx, pn[0]) + pad)); y0 = int(max(0, min(hy, pn[1]) - pad)); y1 = int(min(H, max(hy, pn[1]) + pad))
        if x1 - x0 < 2 or y1 - y0 < 2: return
        dx = s.XX[y0:y1, x0:x1] - pf[0]; dy = s.YY[y0:y1, x0:x1] - pf[1]; u = (dx * ux + dy * uy) / ln; w = (dx * nx + dy * ny) / wn; aa = min(0.25, 1.1 / wn)
        wid = 0.5 * (0.55 + 0.45 * np.clip(u, 0, 1)); tq = np.clip((u - 0.84) / 0.16, 0, 1) ** 1.5; wt = wid * (1 - tq); c = wid - wt
        m = sstep(wt + aa, wt - aa, np.abs(w - c)) * (u >= 0) * (u <= 1); rel = (w - c) / (wt + 1e-4)
        plane = 0.40 + 0.34 * sstep(0.32, 0.20, rel); yak = sstep(0.06, -0.06, rel + 0.50 - 0.13 * np.sin(u * 46 + ph * 7) - 0.05 * np.sin(u * 113 + ph))
        gl = np.exp(-((u - ((t * 0.6 + ph) % 1.4)) / 0.05) ** 2) * 0.9; gr = cv2.remap(s._tiles[0], ((u * ln * 0.6 + ph * 40) % 256).astype(np.float32), ((w * wn * 4 + 128) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        col = STEEL[None, None, :] * ((plane + 0.30 * yak + gl) * (0.85 + 0.3 * gr))[:, :, None] + (np.exp(-((rel + 1) / 0.14) ** 2) * 0.5 + np.exp(-((rel - 0.26) / 0.05) ** 2) * 0.18)[:, :, None]
        col += (0.10 * (1 - np.clip(u, 0, 1)))[:, :, None] * RED[None, None, :]
        uh = u * ln / hl; mh = sstep(0.34 + aa, 0.34 - aa, np.abs(w)) * (u < 0) * (uh > -1)
        dia = sstep(0.25, 0.45, np.abs(((uh * 6 + np.abs(w) * 1.4) % 1.0) - 0.5) * 2); ch = np.array([0.50, 0.05, 0.06], np.float32)[None, None, :] * (0.35 + 0.8 * dia)[:, :, None] * (0.6 + 0.6 * (0.34 - np.abs(w)) / 0.34)[:, :, None]
        mt = sstep(1.0, 0.7, np.abs(u * ln) / (wn * 0.16 + 0.8)) * sstep(0.80 + aa, 0.80 - aa, np.abs(w)); ct = np.array([0.78, 0.60, 0.24], np.float32)[None, None, :] * (0.5 + 0.7 * (1 - np.abs(w) / 0.8))[:, :, None]
        F = f[y0:y1, x0:x1]
        for mm, cc in ((mh, ch), (m, col), (mt, ct)): a_ = (mm * al)[:, :, None]; F[:] = F * (1 - a_) + cc * a_

    def rings(s, f, t, upto, fly=None):
        """切先を こちらへ 向けて ならぶ 刀の 輪。fly[i]=その輪が 飛んできた 割合(0..1)"""
        W, H, k = s.W, s.H, s.k; O = s.O; Ls = s.L0()
        for i in range(min(upto, 6) - 1, -1, -1):
            n = RINGS[i]; pop = back((t - STAGE[i]) / 0.3, 1.6) if fly is None else 1.0; q = fly[i] if fly else 0.0
            if q >= 1 or pop <= 0.02: continue
            z = 1 + 11 * q ** 2.6; rr = W * (0.10 + 0.075 * i) * (1.3 - 0.3 * pop) * (1 + 3.2 * q ** 2.2); rot = t * (0.30 if i % 2 else -0.30) * (1 - 0.1 * i) + i * 0.4
            ln = W * (0.085 + 0.012 * i) * pop * z; wn = ln * 0.11; al = clamp((1 - q) / 0.25)
            for j in range(n):
                a = rot + j * 6.2832 / n; ca, sa = math.cos(a), math.sin(a) * 0.92; pf = (O[0] + ca * rr, O[1] + sa * rr); pn = (O[0] + ca * (rr + ln), O[1] + sa * (rr + ln))
                if min(pf[0], pn[0]) > W + 20 or max(pf[0], pn[0]) < -20 or min(pf[1], pn[1]) > H + 20 or max(pf[1], pn[1]) < -20: continue
                if wn < 2.6: cv2.line(Ls, (int(pf[0]), int(pf[1])), (int(pn[0]), int(pn[1])), 0.9 * al, 1, cv2.LINE_AA)
                else: s.steel(f, pf, pn, wn, t, i * 1.7 + j * 0.37, al)
        s.glowmask(f, Ls, STEEL, 0.6, 1.0, 0.6, 4.0)
        s.glowmask(f, sstep(0.0, 1.0, 1 - np.hypot(s.XX - O[0], (s.YY - O[1]) / 0.92) / (W * 0.07)) * 0.5, RED, 0.25 + 0.08 * upto, 0.6, 1.0, 14.0)           # 輪の しんの 赤い ほむら
        return O

    def gashes(s, f, n, br, t):
        """目の前に たまる 赤い 切り筋"""
        W, H, k = s.W, s.H, s.k; L = s.L0(); Lw = s.L0(); th = 1 if br < 0 else int(1 + 90 * br ** 1.5)
        for i in range(min(n, NCUT)):
            a, cx, cy = s.cut2[i]; ln = W * 1.6; p0 = (int(cx - math.cos(a) * ln), int(cy - math.sin(a) * ln)); p1 = (int(cx + math.cos(a) * ln), int(cy + math.sin(a) * ln))
            cv2.line(L, p0, p1, 1.0, th, cv2.LINE_AA)
            if i >= n - 2 and br < 0: cv2.line(Lw, p0, p1, 1.0, 2, cv2.LINE_AA)
        s.glowmask(f, np.clip(L, 0, 1), RED if br < 0 else RED * 0.5 + 0.5, 0.9 + (2.0 * br if br > 0 else 0), 1.0, 1.0, 8.0 if br < 0 else 16.0)
        if br < 0: s.glowmask(f, Lw, (1.0, 0.85, 0.85), 0.8, 1.0, 1.0, 10.0)

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(RED * 0.8 + 0.2), rim_dir=(0.6, -0.8), rim_v=(0.35 + 0.5 * s._eyev) * alpha, amb=(1.0, 0.6, 0.45), amb_v=0.12, alpha=alpha, shade=0.82)

    def add(s, f, L, col, gain=1.0, b1=3, b2=12):
        s.glowmask(f, np.clip(L, 0, 1), col, gain * 0.8, 1.0, 1.0, float(b2))

    def mask_eye(s, f, v):
        if v <= 0.01: return
        old.Kurogane.mask_eye(s, f, min(1.0, v)); s.flare(f, s.at(*s.meye), 6 * s._pl[2] * 2, RED * 0.8 + 0.2, 0.5 * v, 0.9, 0.0, False)

    def frame(s, t):
        o = {}
        if t < T_CUT: f, sh = s.sc_shoji(t)
        elif t < T_GATHER: s._eyev = 0.8; f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.9)
        elif t < T_FLY: f, sh = s.sc_gather(t)
        else: f, sh = s.sc_release(t)
        f, sh = s.finale(f, sh, t, DUR, None, (1, 0.2, 0.25), (0.6, 0.1, 0.8), -1)
        return s.film(f, t, sh * s.k, 0.5, o.get("thr", 0.82), (1.0, 0.85, 0.82), 1.0 + 0.15 * sh, 0.045, o.get("vig", 0.65), 1.10, 1.12)

    # 1) 障子の 影 → 赤い 一筋 → ひらく
    def sc_shoji(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_SLASH; op = ease2((u - 0.35) / 0.7) if u > 0.35 else 0.0; s._eyev = 0.3 + 0.7 * op
        f = s.room(1.0); s.pose(t); s.draw(f, s.cp, W / 2, H * 0.66, 0.44); s.mask_eye(f, op)
        if op < 1:
            flick = 1 + 0.05 * math.sin(t * 17) + 0.03 * math.sin(t * 41); pane, pm, pap = s.shoji2(op, (0.65 + 0.35 * ease(t / 0.6)) * flick)
            sc = 0.80 * H / s.ah; im = cv2.resize(s.shadow[:, :, 3], None, fx=sc, fy=sc); hh, ww = im.shape[:2]; x0 = int(W / 2 - ww / 2); y0 = int(H * 0.52 - hh / 2)      # 紙に うつる 大きな 影
            A = s.L0(); xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + ww), min(H, y0 + hh); A[ya:yb, xa:xb] = im[ya - y0:yb - y0, xa - x0:xb - x0]
            al = (cv2.GaussianBlur(A, (0, 0), 5 * k) * 0.7 + cv2.GaussianBlur(A, (0, 0), 16 * k) * 0.25) * 0.88 * ease((t - 0.3) / 0.8) * (1 - op)
            pane = pane * (1 - (al * pap)[:, :, None]); f = f * (1 - pm[:, :, None]) + pane * pm[:, :, None]
        if u > 0:                                                                    # 赤い 一筋
            g = ease(u / 0.12); a = (W * 1.05, H * 0.12); b = (-W * 0.05, H * 0.80)
            s.glowline(f, [a, (a[0] + (b[0] - a[0]) * g, a[1] + (b[1] - a[1]) * g)], RED, 2.0 if u < 0.5 else 1.2, clamp(1.6 - u * 1.2), 1.0)
            if u < 0.07: f = f + (1 - u / 0.07) * 0.8
        return f * clamp(t / 0.3), (8 * math.exp(-u * 9) if u > 0 else 0)

    # 2) 筆の 一はけの 中に 鬼の面
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = np.array([0.03, 0.01, 0.012], np.float32)[None, None, :] * (0.5 + 1.2 * s._fib)[:, :, None] + s.nebula(t, [(0.10, 0.01, 0.015), (0.22, 0.02, 0.03), (0.5, 0.08, 0.06)], 0.0012, (6.0, 2.0), 0.5)
        g = ease(u / 0.3); cy = H * 0.34; top = []; bot = []
        for i in range(25):                                                          # かすれた 筆の ふち
            x = -20 + (W + 40) * i / 24.0 * g; r = s.rd[i]
            top.append((x, cy - H * 0.115 + (r[0] - 0.5) * 16 * k + i * 1.2 * k)); bot.append((x, cy + H * 0.115 + (r[1] - 0.5) * 22 * k + i * 1.2 * k))
        m = s.L0(); cv2.fillPoly(m, [np.array(top + bot[::-1], np.int32)], 1.0, cv2.LINE_AA)
        for i in range(14): r = s.rd[i + 30]; cv2.circle(m, (int(r[0] * W * g), int(cy + (r[1] - 0.5) * H * 0.34)), int((2 + 6 * r[2]) * k), 1.0, -1, cv2.LINE_AA)
        dry = cv2.remap(s._tiles[0], ((s.XX * 0.12 / k) % 256).astype(np.float32), ((s.YY * 5.0 / k) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)       # 筆の かすれ
        edge = np.clip(np.abs(s.YY - cy - s.XX * 0.05) / (H * 0.115), 0, 1.4); m = m * np.clip(1 - sstep(0.62, 1.0, edge) * sstep(0.35, 0.65, dry) - sstep(0.75, 1.0, s.XX / W) * sstep(0.4, 0.7, dry) * 0.8, 0, 1)
        win = s.nebula(t + 5, [(0.35, 0.03, 0.04), (0.75, 0.08, 0.07), (1.0, 0.45, 0.25)], 0.0018, (30.0, 8.0), 1.0, 1) + (RED * 0.30)[None, None, :] * (0.6 + 0.8 * dry)[:, :, None]
        zf = 1.55; sc = zf * H / s.ah
        s.pose(t); s.draw(win, s.cp, W * 0.56 - (s.mask[0] - 30 * k - s.aw / 2) * sc - u * 6 * k, cy - (s.mask[1] + 40 * k - s.ah / 2) * sc, zf); s.mask_eye(win, 0.7 + 0.3 * math.sin(u * 11))
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        for i, T in enumerate(s.vt):                                                 # 技名: 1字ずつ 斬りこまれる(切り筋つき)
            a = (u - 0.5 - i * 0.2) / 0.12
            if a > 0:
                p = (W * (0.20 + i * 0.20), H * 0.66 + (i % 2) * 22 * k); s.text(f, T, p[0], p[1], clamp(a), 1.0)
                if a < 2.5: q = clamp(a); s.glowline(f, [(p[0] + 70 * k, p[1] - 80 * k), (p[0] + 70 * k - 140 * k * q, p[1] - 80 * k + 160 * k * q)], RED * 0.5 + 0.5, 1.5, clamp(1.6 - a / 1.6), 0.9)
        for i, T in enumerate(s.vo):
            if u > 0.3 + i * 0.1: s.text(f, T, W * 0.5 + (i - 0.5) * 44 * k, H * 0.54, 1.0, 1.0)
        if u > 1.5: s.text(f, s.txt_en, W / 2, H * 0.66 + 92 * k, ease((u - 1.5) / 0.3), 1.0)
        if u < 0.08: f += (1 - u / 0.08) * 0.8
        return f, 0

    # 3) 動かない。刀だけが ふえて、切先が こちらを 向く
    def sc_gather(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_GATHER
        n = len([x for x in STAGE if x <= t]); last = max([x for x in STAGE if x <= t] or [-9]); since = t - last; s._eyev = 0.4 + 0.1 * n
        f = s.room(0.85); s.rings(f, t, n)
        s.pose(t, breath=0.2); s.draw(f, s.cp, W / 2, H * 0.72, 0.40); s.mask_eye(f, 0.4 + 0.1 * n)
        if n > 0:                                                                    # いまの 本数(漢字)
            T = s.kj[n - 1]; a = since / 0.14; z = 0.6 * (1 + (1 - min(1.0, a)) * 0.3)
            T = tuple(cv2.resize(q, None, fx=z, fy=z, interpolation=cv2.INTER_AREA) for q in T); s.text(f, T, W * 0.5, H * 0.942, clamp(a * 2), 1.0)
            if since < 0.1: f += (1 - since / 0.1) * 0.15
        s.rushdust(f, t, s.O, 400, 0.15, STEEL, 0.15 + 0.04 * n, 3, 0.0)
        if u < 0.15: f += (1 - u / 0.15) * 0.6
        return f, 2.5 * math.exp(-since * 12)

    # 4) こちらへ 飛んでくる → 切り筋が たまる → 納刀 → ひらく
    def sc_release(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_FLY; sh = t - T_SHEATH; br = t - T_BURST; s._eyev = 1.0
        fly = [clamp((u - (5 - i) * 0.16) / 0.36) for i in range(6)]; hits = sum(1 for q in fly if q >= 0.8); done = clamp((u - 1.2) / 0.3)
        f = s.room(0.85 * (1 - 0.55 * done)); s.rushdust(f, t, s.O, 900, 1.6, STEEL, 0.5 * (1 - done), 4, 1.0)
        s.pose(t, breath=0.2 if sh < 0 else 0.0, squash=(0.05 * math.exp(-sh * 14) if sh > 0 else 0.0)); s.draw(f, s.cp, W / 2, H * 0.72, 0.40); s.mask_eye(f, 1.0)
        s.rings(f, t, 6, fly)
        for i in range(6):                                                           # 通りすぎる 一瞬の 光
            q = u - (5 - i) * 0.16 - 0.30
            if 0 < q < 0.08: f += (1 - q / 0.08) * 0.35
        s.gashes(f, hits * 6, br, t)
        if sh > 0 and br < 0:                                                        # 納刀: つばもとの 小さな 火花
            p = s.at(*s.hilt); v = clamp(1 - sh / 0.35); s.flare(f, p, (14 * v + 3) * k, (1.0, 0.95, 0.9), v, 0.8, 0.4, False)
        if br > 0: f = f + min(1.0, br / 0.25) ** 2 * 0.6 * RED[None, None, :] + (br / 0.3) ** 2 * 0.5
        return f, 5 * sum(math.exp(-abs(u - (5 - i) * 0.16 - 0.30) * 18) for i in range(6)) + (3 * math.exp(-sh * 20) if sh > 0 else 0) + (14 * br if br > 0 else 0)

    # ---- この子だけの曲: 三味線ふうの はじく音(都節)と 拍子木。刀が ふえるたび 一音 上がる。飛んでくる 風切り → 納刀の「カチン」のあとは 無音、それから 大太鼓 ----
    def audio(s, path):
        a = Snd(DUR, 74); D = 146.83; miy = [0, 1, 5, 7, 8]
        def fq(d, o=0): return D * 2 ** ((miy[d % 5] + 12 * (d // 5 + o)) / 12.0)
        def pluck(d, t0, g=0.14, o=0): a.note(fq(d, o), t0, 0.22, g, "fm", 0.002); a.note(fq(d, o) * 2.01, t0, 0.07, g * 0.4, "sq", 0.001)
        def clack(t0, g=0.3): a.a += (a.nz - a.lp(a.nz, 0.7)) * a.env(t0, 0.0005, 0.02) * g; a.note(1900, t0, 0.03, g * 0.3, "sine", 0.0005)
        def taiko(t0, g=1.0): a.boom(t0, 48, g, 0.5)
        a.a += a.lp(a.nz, 0.01) * 0.4 * a.gate(0, T_SLASH, 0.3)
        for i, d in enumerate([0, 1, 0]): pluck(d, 0.4 + i * 0.45, 0.10)
        a.whoosh(T_SLASH - 0.12, T_SLASH + 0.1, 0.8, 0.5); clack(T_SLASH + 0.12, 0.5); taiko(T_SLASH + 0.35, 0.7)
        a.rise(T_SLASH + 0.35, T_SLASH + 1.0, 90, 50, 0.2)                                                           # 障子が すべる
        taiko(T_CUT, 1.0)
        for i in range(4): a.whoosh(T_CUT + 0.42 + i * 0.2, T_CUT + 0.5 + i * 0.2, 0.5, 0.6); pluck([7, 5, 3, 0][i], T_CUT + 0.5 + i * 0.2, 0.16)
        for i, x in enumerate(STAGE):                                                                                 # ふえるたび 一音 上がる
            pluck(i, x, 0.16, 1); clack(x, 0.25); a.note(fq(i, 2), x, 0.25, 0.04, "bell")
            for j in range(min(6, RINGS[i] // 3)): a.note(2400 + j * 200, x + j * 0.02, 0.05, 0.03, "sine")              # しゃりん
        tk = T_GATHER; i = 0
        while tk < T_FLY - 0.1:
            if i % 4 == 0: taiko(tk, 0.28)
            if i % 4 == 2: clack(tk, 0.12)
            tk += 0.15; i += 1
        a.rise(STAGE[-1], T_FLY, 150, 500, 0.12)
        for i in range(6):                                                                                            # 輪ごとに 風切り と 切る 音
            tk = T_FLY + i * 0.16; a.whoosh(tk, tk + 0.30, 0.8, 0.6); a.rise(tk + 0.1, tk + 0.32, 2600, 700, 0.10); pluck(9 - i, tk + 0.30, 0.15, 1); taiko(tk + 0.30, 0.45)
            a.a += (a.nz - a.lp(a.nz, 0.6)) * a.env(tk + 0.30, 0.001, 0.05) * 0.4
            for j in range(4): pluck(4 + (i + j) % 5, tk + j * 0.05, 0.06, 1)
        a.a *= 1 - 0.85 * np.clip((a.t - (T_SHEATH - 0.35)) / 0.2, 0, 1) * (a.t < T_SHEATH - 0.02)                                # しずまる
        clack(T_SHEATH, 0.7); a.note(2600, T_SHEATH, 0.5, 0.10, "bell")                                               # カチン
        taiko(T_BURST, 1.7); taiko(T_BURST + 0.16, 0.9); a.a += a.lp(a.nz, 0.3) * a.env(T_BURST, 0.005, 0.4) * 0.6; a.a += (a.nz - a.lp(a.nz, 0.5)) * a.env(T_BURST, 0.002, 0.25) * 0.4
        for d, o in ((0, -1), (1, 0), (3, 0), (0, 1)): a.note(fq(d, o), T_BURST, 0.6, 0.07, "saw", 0.01)
        s.fin_snd(a, DUR, 74); a.save(path)


if __name__ == "__main__":
    r = Kurogane()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
