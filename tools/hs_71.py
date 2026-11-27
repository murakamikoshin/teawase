"""ゼノン(★6 建国の王) 専用の必殺技「創国ノ王笏」— 高品質版(hs_fx)・通し直し。国の印(封ろうの はんこ)が こちらへ 押される
金の インクで 描かれた 古い地図の 夜空。空の 高みに うすい 王冠の 紋(前ふり) → ゼノンが 斜めに 飛びこみ、頂点で ぴたり → 王笏の「目」が ひらく(丸い窓のカットイン) →
ふりおろして 地を 打つ → 金の 道が こちらの 足もとへ 走り、線だった 城が 石の 城に なって 塔ごとに 立ち、窓に 灯。うしろから 夜明け →
空の 紋に 赤い ろうが ともり、国の印に なって こちらへ おりてくる(ろうの つや・もりあがった ふち・金の 王冠)→ どん、と 目の前を おおう → 白 → 暗転
使い方: python3 hs_71.py 出力.mp4 / python3 hs_71.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old71", "/home/claude/tools/pre_fin/hs_71_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = 9.7
GOLD, ROYAL, SKYB, RED, TOWERS = old.GOLD, old.ROYAL, old.SKYB, old.RED, old.TOWERS
T_STOP, T_CUT, T_RAISE, T_HIT = old.T_STOP, old.T_CUT, old.T_RAISE, old.T_HIT
T_SEAL, T_PRESS = 7.6, 9.0
WARM = np.array([1.0, 0.62, 0.30], np.float32)
CROWN = [(-0.40, 0.20), (-0.48, -0.20), (-0.24, 0.00), (0.0, -0.34), (0.24, 0.00), (0.48, -0.20), (0.40, 0.20)]


def psdf(X, Y, P):
    d = np.full(X.shape, 1e9, np.float32); ins = np.zeros(X.shape, bool); n = len(P)
    for i in range(n):
        ax, ay = P[i]; bx, by = P[(i + 1) % n]; ex, ey = bx - ax, by - ay; wx, wy = X - ax, Y - ay
        tt = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0, 1); d = np.minimum(d, np.hypot(wx - ex * tt, wy - ey * tt))
        if abs(by - ay) > 1e-6: ins ^= ((ay <= Y) != (by <= Y)) & (X < ax + (Y - ay) / (by - ay) * ex)
    return np.where(ins, -d, d).astype(np.float32)


class Zenon(FX, old.Zenon):
    def __init__(s, W=540):
        old.Zenon.__init__(s, W); s.fx_init(71); H = s.H; s.yv = (s.YY / H).astype(np.float32); s.gy0 = H * 0.80; s.CR = (W * 0.5, H * 0.17); s._day = 0.0; s._eye = 0.0

    # ---- 古い地図の 夜空 ----
    def atlas(s, t, lum=1.0, day=0.0):
        W, H, k = s.W, s.H, s.k; n = s.fbm(0.0012, t * 4, 0, 4, 40 * k, 0); n2 = s.fbm(0.003, 0, 0, 3, 0, 2)
        f = ROYAL[None, None, :] * ((0.16 + 0.70 * s.yv) * (0.55 + 0.9 * n))[:, :, None]
        f = f + (sstep(0.55, 0.9, n) * 0.05)[:, :, None] * SKYB[None, None, :]
        off = int((t * 12) % 70); m = s.map[:, off:off + W]; f = f + (m * (0.10 + 0.34 * n2) * (1 - 0.6 * day))[:, :, None] * GOLD[None, None, :]      # 金の インク(かすれ)
        s.starfield(f, t, 500, GOLD * 0.6 + 0.4, 5, 0.6 * (1 - 0.7 * day), (0.0, 0.0))
        if day > 0:
            g = np.exp(-np.clip(s.gy0 - s.YY, 0, None) / (H * 0.26)); f = f * (1 + 0.6 * day) + (g * day * 1.25)[:, :, None] * WARM[None, None, :] + (np.exp(-np.clip(s.gy0 - s.YY, 0, None) / (H * 0.07)) * day * 0.6)[:, :, None] * (WARM * 0.6 + 0.4)[None, None, :] + (day * 0.10 * (1 - g))[:, :, None] * SKYB[None, None, :]
        return (f * lum).astype(np.float32)

    def crest(s, f, v, t, R=None):
        """空の 王冠の 紋(金の 線)"""
        if v <= 0.01: return
        c = s.CR; R = R or s.W * 0.11; L = s.L0()
        for rr, b in ((1.0, 0.9), (0.80, 0.6)): cv2.circle(L, (int(c[0]), int(c[1])), int(R * rr), b, 1, cv2.LINE_AA)
        cv2.polylines(L, [np.array([(c[0] + x * R * 1.15, c[1] + y * R * 1.15) for x, y in CROWN], np.int32)], True, 1.0, 1, cv2.LINE_AA)
        for i in range(24): a = i * 0.2618 + t * 0.15; cv2.line(L, (int(c[0] + math.cos(a) * R * 1.0), int(c[1] + math.sin(a) * R * 1.0)), (int(c[0] + math.cos(a) * R * 1.10), int(c[1] + math.sin(a) * R * 1.10)), 0.7, 1, cv2.LINE_AA)
        s.glowmask(f, L, GOLD, v, 0.8, 1.0, 7.0)

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        d = s._day
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(GOLD * (1 - d) + (WARM * 0.7 + 0.3) * d), rim_dir=(0.0, -1.0) if d > 0.3 else (0.6, -0.8), rim_v=(0.7 + 0.3 * d) * alpha, amb=(0.6, 0.72, 1.0), amb_v=0.14 * (1 - d), aura=SKYB if s._eye > 0.3 and alpha > 0.9 else None, aura_v=0.10 * s._eye, alpha=alpha, shade=0.90)

    def add(s, f, L, col, gain=1.0, b1=3, b2=12):
        s.glowmask(f, np.clip(L, 0, 1), col, gain, 0.8, 1.0, float(b2))

    def scepter_eye(s, f, p, open_, t, big=1.0):
        if open_ <= 0: return
        old.Zenon.scepter_eye(s, f, p, open_, t, big); s.flare(f, p, 7 * s._pl[2] * 2 * (0.6 + 0.4 * big), SKYB * 0.8 + 0.2, 0.55 * open_, 0.8, 0.0, big > 1.3)

    # ---- 石の 城 ----
    def stone(s, f, x0, x1, top, bot, al, lineg, cren=True):
        W, H, k = s.W, s.H, s.k; d = s._day; a0 = int(max(0, x0 - 2)); a1 = int(min(W, x1 + 2)); b0 = int(max(0, top - 11 * k)); b1 = int(min(H, bot + 1))
        if a1 - a0 < 2 or b1 - b0 < 2: return
        X = s.XX[b0:b1, a0:a1]; Y = s.YY[b0:b1, a0:a1]
        m = sstep(x0 - 0.7, x0 + 0.7, X) * sstep(x1 + 0.7, x1 - 0.7, X) * sstep(top - 0.7, top + 0.7, Y)
        if cren: cw = 7 * k; m = np.maximum(m, sstep(x0 - 0.7, x0 + 0.7, X) * sstep(x1 + 0.7, x1 - 0.7, X) * sstep(top - 9 * k - 0.7, top - 9 * k + 0.7, Y) * (((X - x0) / cw) % 2 < 1) * (Y <= top + 1))
        by = Y / (8 * k); row = np.floor(by); fy = by - row; bx = X / (17 * k) + 0.5 * (row % 2); col_ = np.floor(bx); fx = bx - col_
        mort = (0.55 + 0.45 * sstep(0.0, 0.14, fy) * sstep(0.0, 0.08, fx)); var = s._tiles[0][(row.astype(np.int32) * 91) % 256, (col_.astype(np.int32) * 37) % 256]
        gr = cv2.remap(s._tiles[1], ((X * 2.2 / k) % 256).astype(np.float32), ((Y * 2.2 / k) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        up = np.clip((bot - Y) / max(1.0, bot - top), 0, 1); rim = np.exp(-(X - x0) / (3.5 * k)) + np.exp(-(x1 - X) / (3.5 * k)) + np.exp(-np.clip(Y - (top - 9 * k if cren else top), 0, None) / (3 * k)) * 0.8
        base = np.array([0.20, 0.22, 0.34], np.float32)[None, None, :] * (1 - 0.5 * d) + np.array([0.26, 0.17, 0.16], np.float32)[None, None, :] * 0.5 * d
        c = base * (mort * (0.75 + 0.5 * var) * (0.7 + 0.5 * gr) * (0.50 + 0.45 * up))[:, :, None] + (np.clip(rim, 0, 1.3) * (0.22 + 0.55 * d))[:, :, None] * (GOLD * (1 - d) + WARM * d)[None, None, :]
        F = f[b0:b1, a0:a1]; mm = (m * al)[:, :, None]; F[:] = F * (1 - mm) + c * mm
        if lineg > 0.02:
            L = s.L0(); cv2.rectangle(L, (int(x0), int(top)), (int(x1), int(bot)), 1.0, 1, cv2.LINE_AA); s.glowmask(f, L, GOLD, 0.9 * lineg, 0.9, 1.0, 6.0)

    def roof(s, f, cx, hw, base, rh, al):
        W, H, k = s.W, s.H, s.k; d = s._day; a0 = int(max(0, cx - hw - 2)); a1 = int(min(W, cx + hw + 2)); b0 = int(max(0, base - rh - 2)); b1 = int(min(H, base + 1))
        if a1 - a0 < 2 or b1 - b0 < 2 or rh < 2: return
        X = s.XX[b0:b1, a0:a1]; Y = s.YY[b0:b1, a0:a1]; q = np.clip((Y - (base - rh)) / rh, 0, 1); half = hw * q; m = sstep(half + 0.8, half - 0.8, np.abs(X - cx)) * (Y <= base) * (Y >= base - rh)
        tl = 0.6 + 0.4 * sstep(0.1, 0.3, ((Y / (5 * k)) % 1.0)) * sstep(0.05, 0.2, ((X / (7 * k) + 0.5 * (np.floor(Y / (5 * k)) % 2)) % 1.0))
        side = np.clip((X - cx) / (half + 1e-3), -1, 1); c = (ROYAL * 1.5 + 0.03)[None, None, :] * (tl * (0.55 + 0.35 * side))[:, :, None] + (np.exp(-(half - np.abs(X - cx)) / (2.5 * k)) * (0.25 + 0.5 * d))[:, :, None] * (GOLD * (1 - d) + WARM * d)[None, None, :]
        F = f[b0:b1, a0:a1]; mm = (m * al)[:, :, None]; F[:] = F * (1 - mm) + c * mm

    def castle(s, f, t, hit):
        W, H, k = s.W, s.H, s.k; gy = s.gy0; Wn = s.L0(); Fl = s.L0(); Pl = s.L0()
        wl = ease(clamp((hit - 0.15) / 0.6))
        if wl > 0: s.stone(f, W / 2 - W * 0.5 * wl, W / 2 + W * 0.5 * wl, gy - H * 0.07, gy, clamp((hit - 0.3) / 0.5), 1 - 0.8 * clamp((hit - 0.3) / 0.6))
        order = sorted(range(len(TOWERS)), key=lambda i: TOWERS[i][2])
        for i in order:
            x, w, h, kind = TOWERS[i]; t0 = 0.2 + i * 0.16; g = back((hit - t0) / 0.45, 1.3) if hit > t0 else 0.0
            if g <= 0: continue
            g = min(g, 1.06); x0 = (x - w / 2) * W; x1 = (x + w / 2) * W; top = gy - h * H * g; al = clamp((hit - t0 - 0.12) / 0.45)
            s.stone(f, x0, x1, top, gy, al, 1 - 0.85 * clamp((hit - t0 - 0.2) / 0.6))
            if kind:
                gr = ease(clamp((hit - t0 - 0.4) / 0.3)); rh = (60 if kind == 2 else 38) * k * gr; cx = x * W
                if gr > 0:
                    s.roof(f, cx, (x1 - x0) / 2 - 4 * k, top - 9 * k, rh, 1.0); fy = top - 9 * k - rh
                    if gr >= 1:
                        cv2.line(Pl, (int(cx), int(fy)), (int(cx), int(fy - 30 * k)), 0.8, 1, cv2.LINE_AA); wv = math.sin(t * 6 + i) * 5 * k
                        cv2.fillPoly(Fl, [np.array([(cx, fy - 30 * k), (cx + 28 * k, fy - 24 * k + wv), (cx, fy - 17 * k)], np.int32)], 1.0, cv2.LINE_AA)
        for j in range(60):                                                          # 窓に 灯が ともる
            r = s.win[j]; i = int(r[0] * len(TOWERS)); x, w, h, kind = TOWERS[i]
            if hit > 0.9 + i * 0.16 + r[2] * 1.0:
                wx = (x + (r[1] - 0.5) * w * 0.6) * W; wy = gy - (0.12 + r[2] * 0.8) * h * H
                cv2.ellipse(Wn, (int(wx), int(wy - 3 * k)), (int(3 * k), int(3 * k)), 0, 180, 360, 1.0, -1, cv2.LINE_AA); cv2.rectangle(Wn, (int(wx - 3 * k), int(wy - 3 * k)), (int(wx + 3 * k), int(wy + 5 * k)), 1.0, -1)
        s.glowmask(f, Wn, WARM * 0.7 + GOLD * 0.3, 0.9, 1.0, 1.0, 6.0); s.glowmask(f, Pl, GOLD, 0.5, 0.8, 0.5, 4.0)
        f += (Fl * 0.75 + cv2.GaussianBlur(Fl, (0, 0), 3 * k) * 0.35)[:, :, None] * RED[None, None, :] * (0.6 + 0.4 * np.clip(1 - (s.XX % (9 * k)) / (9 * k), 0, 1))[:, :, None]

    def ground(s, f, t, hit):
        W, H, k = s.W, s.H, s.k; gy = s.gy0; d = s._day; g0 = int(gy); X = s.XX[g0:]; Y = s.YY[g0:]; dep = (H - gy) / (Y - gy + 5 * k)
        tex = cv2.remap(s._tiles[2], (((X - W / 2) * dep * 0.35 / k + 128) % 256).astype(np.float32), ((dep * 60) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        near = np.clip((Y - gy) / (H - gy), 0, 1)
        c = np.array([0.025, 0.04, 0.12], np.float32)[None, None, :] * ((0.5 + 1.0 * tex) * (0.6 + 0.6 * near))[:, :, None]
        c += (np.exp(-((X - W / 2) / (W * 0.16 * (1 + 2.5 * near))) ** 2) * d * 0.45 * (0.4 + 0.9 * tex) * (1 - 0.6 * near))[:, :, None] * WARM[None, None, :]
        f[g0:] = c
        if hit > 0:                                                                  # 金の 道が こちらの 足もとへ 走る
            L = s.L0(); g = ease(clamp(hit / 0.5))
            for i in range(11):
                a = (i - 5) * 0.30; cv2.line(L, (int(W / 2), int(gy)), (int(W / 2 + math.tan(a) * (H - gy) * 1.4 * g * 2.2), int(gy + (H - gy) * 1.4 * g)), 0.8, 1, cv2.LINE_AA)
            for j in range(6):
                v = ((hit * 0.55 + j / 6.0) % 1.0); yy = gy + (H - gy) * 1.2 * v ** 2.2
                if v < g * 1.2: cv2.line(L, (0, int(yy)), (W, int(yy)), float(0.25 + 0.6 * v), 1 if v < 0.6 else 2, cv2.LINE_AA)
            for i in range(3):
                v = (hit - i * 0.1) / 0.8
                if 0 < v < 1: cv2.ellipse(L, (int(W / 2), int(gy)), (int(W * 1.0 * v), int(W * 0.16 * v)), 0, 0, 180, float(1 - v), max(1, int(2 * k)), cv2.LINE_AA)
            L[:g0] = 0; s.glowmask(f, L, GOLD, 0.55, 0.9, 1.0, 7.0)

    # ---- 国の印(封ろう) ----
    def seal(s, f, c, R, rot, al, wax=1.0):
        W, H = s.W, s.H; x0 = int(max(0, c[0] - R * 1.14)); x1 = int(min(W, c[0] + R * 1.14)); y0 = int(max(0, c[1] - R * 1.14)); y1 = int(min(H, c[1] + R * 1.14))
        if x1 - x0 < 3 or y1 - y0 < 3 or al <= 0.01: return
        Xs = (s.XX[y0:y1, x0:x1] - c[0]) / R; Ys = (s.YY[y0:y1, x0:x1] - c[1]) / R; ca, sa = math.cos(rot), math.sin(rot); X = Xs * ca + Ys * sa; Y = -Xs * sa + Ys * ca
        r = np.hypot(X, Y); ang = np.arctan2(Y, X); re = 1 + 0.05 * np.sin(5 * ang + 1) + 0.03 * np.sin(9 * ang + 2.3) + 0.018 * np.sin(14 * ang + 0.7); q = r / re
        aa = min(0.2, 2.0 / R); m = sstep(1.0, 1.0 - aa, q) * al
        plate = sstep(0.75, 0.70, r); h = (0.55 * np.exp(-((q - 0.84) / 0.10) ** 2) + 0.25 * sstep(1.0, 0.9, q)) * (1 - plate) + 0.10 * plate
        sd = np.minimum(psdf(X, Y, CROWN), np.maximum(np.abs(X) - 0.40, np.abs(Y - 0.30) - 0.045)); ew = max(0.012, 1.2 / R)
        emb = sstep(ew, -ew, sd) + np.exp(-((r - 0.64) / 0.014) ** 2) + np.exp(-((r - 0.685) / 0.018) ** 2) * sstep(0.5, 0.9, np.cos(ang * 36))
        for jx, jy in ((-0.48, -0.20), (0.0, -0.34), (0.48, -0.20)): emb = emb + np.exp(-(((X - jx) / 0.045) ** 2 + ((Y - jy) / 0.045) ** 2)) * 1.2
        emb = np.clip(emb, 0, 1.3) * plate; tex = cv2.remap(s._tiles[1], ((X * 46 + 128) % 256).astype(np.float32), ((Y * 46 + 128) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        h = h + 0.20 * emb + 0.03 * tex; h = cv2.GaussianBlur(h, (0, 0), max(0.8, R * 0.006))
        gyy, gxx = np.gradient(h); nx = -gxx * R * 1.5; ny = -gyy * R * 1.5; nn = np.sqrt(nx * nx + ny * ny + 1); nx /= nn; ny /= nn; nz = 1 / nn
        lam = np.clip(nx * -0.45 + ny * -0.60 + nz * 0.66, 0, 1); hh = np.clip(nx * -0.25 + ny * -0.33 + nz * 0.91, 0, 1); spec = hh ** 46
        wc = np.array([0.66, 0.045, 0.055], np.float32)[None, None, :] * ((0.20 + 1.0 * lam) * (0.75 + 0.5 * tex))[:, :, None] + (spec * 0.85)[:, :, None] * np.array([1.0, 0.82, 0.74], np.float32)[None, None, :]
        gm = (np.clip(emb, 0, 1) * 0.85)[:, :, None]; gc = GOLD[None, None, :] * (0.22 + 1.0 * lam + 1.4 * spec)[:, :, None] * (0.8 + 0.4 * tex)[:, :, None]
        col = wc * (1 - gm) + gc * gm
        if wax < 1: col = col * wax + (np.clip(emb, 0, 1) * 0.9)[:, :, None] * GOLD[None, None, :] * (1 - wax); m = m * (wax + (1 - wax) * np.clip(emb, 0, 1))
        F = f[y0:y1, x0:x1]; F[:] = F * (1 - m[:, :, None]) + col * m[:, :, None]

    def frame(s, t):
        o = {}; s._day = 0.0; s._eye = 0.2 if t > T_STOP else 0.0
        if t < T_CUT: f, sh = s.sc_leap(t); s.crest(f, 0.16 + 0.10 * math.sin(t * 2.0) + (0.5 * math.exp(-(t - T_STOP) * 4) if t > T_STOP else 0), t); f = f * clamp(t / 0.4)
        elif t < T_RAISE: s._eye = 1.0; f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.9)
        else: f, sh = s.sc_realm(t)
        f, sh = s.finale(f, sh, t, DUR, None, (1, 0.82, 0.3), (0.4, 0.6, 1), -1)
        return s.film(f, t, sh * s.k, 0.55, o.get("thr", 0.82), (1.0, 0.93, 0.78), 1.0 + 0.15 * sh, 0.04, o.get("vig", 0.6), 1.12, 1.10)

    # 2) 丸い窓の中で 王笏の目が ひらく
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.atlas(t, 0.45); s.rushdust(f, t, (W / 2, H * 0.34), 500, 0.25, GOLD, 0.35, 3, 0.0)
        C = (W / 2, H * 0.34); R = W * 0.40 * back(u / 0.35, 1.2)
        win = s.nebula(t, [(0.05, 0.10, 0.34), (0.10, 0.22, 0.60), (0.55, 0.70, 1.0)], 0.0014, (10.0, 4.0), 0.9, 1) + (ROYAL * 0.55)[None, None, :]
        win += (np.exp(-np.hypot(s.XX - C[0], s.YY - C[1]) / (W * 0.22)) * 0.35)[:, :, None] * SKYB[None, None, :]
        zf = 1.55; sc = zf * H / s.ah
        s.hero(win, t, C[0] - (s.orb[0] - s.aw / 2) * sc, C[1] - (s.orb[1] - s.ah / 2) * sc, zf)
        op = ease((u - 0.35) / 0.25); s.scepter_eye(win, s.pt(s.orb), op * (0.8 + 0.2 * math.sin(u * 10)), t, 1.6)
        win = s.godrays(win, C, 0.45 * op, 0.6)
        m = s.L0(); cv2.circle(m, (int(C[0]), int(C[1])), max(1, int(R)), 1.0, -1, cv2.LINE_AA)
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        L = s.L0(); s.circ(L, C, R, 1.0, 3); s.circ(L, C, R + 12 * k, 0.5, 1)
        for i in range(36):                                                          # 窓のふちの 目もり(回る)
            a = i * 0.1745 + u * 0.5; r0 = R + (20 if i % 3 else 30) * k
            s.line(L, (C[0] + math.cos(a) * (R + 12 * k), C[1] + math.sin(a) * (R + 12 * k)), (C[0] + math.cos(a) * r0, C[1] + math.sin(a) * r0), 0.7, 1)
        s.glowmask(f, L, GOLD, 0.8, 0.9, 1.0, 8.0)
        n = len(s.ht); step = W * 0.17
        for i, T in enumerate(s.ht):
            a = (u - 0.6 - i * 0.12) / 0.2
            if a > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.72 + (1 - ease(a)) * 50 * k, clamp(a), 1.0)
        if u > 0.6: g = ease((u - 0.6) / 0.5); s.glowline(f, [(W / 2 - W * 0.44 * g, H * 0.72 + 56 * k), (W / 2 + W * 0.44 * g, H * 0.72 + 56 * k)], GOLD, 1.5, 0.8, 0.8)
        for i, T in enumerate(s.vo):
            if u > 0.5 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.72 - 74 * k, 1.0, 1.0)
        if u > 1.3: s.text(f, s.txt_en, W / 2, H * 0.72 + 84 * k, ease((u - 1.3) / 0.3), 1.0)
        if u < 0.1: f += (1 - u / 0.1) * 0.8
        return f, 0

    # 3) 地を 打つ → 城が 立つ → 夜明け → 国の印が こちらへ
    def sc_realm(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_RAISE; hit = t - T_HIT; gy = s.gy0; se = t - T_SEAL; e = clamp(se / (T_PRESS - T_SEAL)) if se > 0 else 0.0
        day = ease2(clamp((hit - 0.6) / 1.3)) if hit > 0 else 0.0; s._day = day; s._eye = 1.0
        f = s.atlas(t, 0.9, day)
        if se <= 0.35: s.crest(f, 0.22 + 0.5 * clamp(hit / 0.3) * (hit > 0) + 0.5 * clamp(se / 0.35) * (se > 0), t)
        sun = (W / 2 + W * 0.09, gy - H * (0.04 + 0.27 * day))
        if day > 0.02: s.flare(f, sun, W * 0.10 * (0.4 + 0.6 * day), WARM * 0.6 + 0.4, 1.3 * day, 0.0, 0.0, False)
        if hit > 0: s.castle(f, t, hit)
        s.ground(f, t, hit)
        f += (np.exp(-((s.YY - gy) / (H * 0.022)) ** 2) * (0.10 + 0.25 * day))[:, :, None] * (SKYB * (1 - day) + WARM * day)[None, None, :]
        if day > 0.05: f = s.godrays(f, sun, 0.9 * day, 0.5, 12, 0.88, 0.6)
        if hit < 0:                                                                  # ため → ふりおろす
            r = ease(u / 0.5); dv = clamp((u - 0.5) / (T_HIT - T_RAISE - 0.5)) ** 2
            cy = H * (0.36 - 0.05 * r) + (gy - H * 0.17 - H * 0.31) * dv; deg = 22 * r - 60 * dv
            for j in (2, 1):
                if dv > 0.05: s.hero(f, t, W / 2, cy - j * H * 0.07 * dv, 0.40, deg + 14 * j, 0.2, 0.6)
            s.hero(f, t, W / 2, cy, 0.40, deg, squash=-0.06 * r * (1 - dv), jit=1.2 * k * r * (1 - dv)); s.scepter_eye(f, s.pt(s.orb), 1.0, t, 1.2 + r)
        else:
            s.hero(f, t, W / 2, gy - H * 0.17, 0.40, -38 * math.exp(-hit * 5) + (10 * ease(clamp(se / 0.3)) if se > 0 else 0), squash=0.10 * math.exp(-hit * 8) * math.cos(hit * 22) - (0.04 * ease(clamp(se / 0.3)) if se > 0 else 0), sway=0.8 * math.exp(-hit * 3))
            s.scepter_eye(f, s.pt(s.orb), 1.0, t, 1.2 + (0.8 * math.exp(-se * 3) if se > 0 else 0))
            if hit < 0.12: f = f + (1 - hit / 0.12) * 1.2
        s.rushdust(f, t, (W / 2, gy - H * 0.2), 500, 0.2 + 0.5 * e, GOLD * (1 - day) + WARM * day, 0.25 * clamp(hit / 0.5) * (hit > 0), 4, 0.0)
        sh = 16 * math.exp(-hit * 4.5) if hit > 0 else 0
        if se > 0:                                                                   # 国の印が おりてくる
            a = ease(clamp(se / 0.35)); R = W * (0.11 + 0.07 * a + 0.20 * e + 1.75 * e ** 4.5)
            c = (s.CR[0], s.CR[1] + (H * 0.50 - s.CR[1]) * ease2(e) ** 1.3)
            sd = np.exp(-(((s.XX - c[0]) / (R * 1.5 + 40 * k)) ** 2 + ((s.YY - c[1] - R * 0.5) / (R * 1.7 + 40 * k)) ** 2)); f *= (1 - 0.75 * sd * e ** 1.2)[:, :, None]           # 印の 影
            s.seal(f, c, R, 0.45 * (1 - e) ** 2, 1.0, a)
            if a < 1: s.flare(f, c, W * 0.06, RED * 0.7 + GOLD * 0.3, 0.9 * (1 - a), 0.6, 0.0, False)
            sh = max(sh, 2 + 14 * e ** 3)
            if e > 0.9: f = f + ((e - 0.9) / 0.1) ** 2 * 0.9
        if u < 0.15: f += (1 - u / 0.15) * 0.6
        return f, sh

    # ---- この子だけの曲: 王のファンファーレ(変ロ長調)。風 → 目がひらく音 → ため → ティンパニ → らっぱの和音が 塔ごとに 重なる → ファンファーレ → 印が おりる とどろき → どん ----
    def audio(s, path):
        a = Snd(DUR, 71); Bb = 233.08; maj = [0, 2, 4, 5, 7, 9, 11]
        def fq(d, o=0): return Bb * 2 ** ((maj[d % 7] + 12 * (d // 7 + o)) / 12.0)
        def brass(d, t0, dur, g=0.07, o=0):
            a.note(fq(d, o), t0, dur, g, "saw", 0.03); a.note(fq(d, o) * 1.004, t0, dur, g * 0.6, "saw", 0.03)
        a.whoosh(0.1, T_STOP, 0.5, 0.25); a.boom(T_STOP, 70, 0.6, 0.15)
        a.note(fq(4, 2), T_STOP + 0.05, 0.9, 0.10, "bell")
        a.boom(T_CUT, 50, 1.0); a.rise(T_CUT + 0.3, T_CUT + 0.6, 300, 900, 0.12)
        for d in (0, 2, 4): brass(d, T_CUT + 0.35, 0.9, 0.05)
        for i in range(5): brass([0, 2, 4, 2, 7][i], T_CUT + 0.6 + i * 0.12, 0.16, 0.07)
        a.rise(T_RAISE, T_HIT, 90, 400, 0.2)
        for i in range(14): a.hat(T_RAISE + i * (T_HIT - T_RAISE) / 14, 0.05 + i * 0.008, 0.03)                    # たいこの ロール
        a.boom(T_HIT, 40, 1.4, 0.7); a.boom(T_HIT + 0.02, 90, 0.6, 0.2)
        chord = [(0, -1), (4, -1), (0, 0), (2, 0), (4, 0), (7, 0), (9, 0)]                                         # 塔が立つたび 音が重なる
        for i, (d, o) in enumerate(chord):
            tk = T_HIT + 0.2 + i * 0.16; brass(d, tk, T_SEAL - tk + 0.2, 0.035, o); a.boom(tk, 60 + i * 6, 0.35, 0.12)
        fan = [(0, 0.0), (0, 0.14), (0, 0.21), (4, 0.28), (7, 0.56), (4, 0.70), (7, 0.84), (9, 0.98), (11, 1.12), (14, 1.26)]     # ファンファーレ(夜明け)
        for d, dt in fan: brass(d, T_SEAL - 1.45 + dt, 0.18, 0.09, 0); a.note(fq(d, 1), T_SEAL - 1.45 + dt, 0.16, 0.03, "sq")
        a.note(fq(7, 1), T_SEAL, 1.2, 0.09, "bell"); a.boom(T_SEAL, 60, 0.6, 0.2)                                     # 紋に ろうが ともる
        for d, o in ((0, -1), (0, 0), (2, 0), (4, 0), (7, 0)): brass(d, T_SEAL + 0.1, T_PRESS - T_SEAL, 0.04, o)
        a.rise(T_SEAL + 0.2, T_PRESS, 70, 30, 0.22); a.whoosh(T_SEAL + 0.3, T_PRESS, 0.9, 0.45)                         # おりてくる とどろき
        a.a += a.lp(a.nz, 0.06) * np.clip((a.t - T_SEAL) / (T_PRESS - T_SEAL), 0, 1) ** 2 * (a.t < T_PRESS) * 0.9
        tk = T_SEAL + 0.3; i = 0
        while tk < T_PRESS - 0.05: a.boom(tk, 52, 0.35 + 0.5 * (tk - T_SEAL) / (T_PRESS - T_SEAL), 0.12); tk += max(0.07, 0.24 - i * 0.02); i += 1
        a.boom(T_PRESS, 36, 2.0, 0.8); a.boom(T_PRESS + 0.01, 85, 0.9, 0.25); a.a += a.lp(a.nz, 0.25) * a.env(T_PRESS, 0.004, 0.35) * 0.7   # どん
        for d, o in ((0, -1), (0, 0), (4, 0), (7, 0), (14, 0)): brass(d, T_PRESS + 0.02, 0.6, 0.06, o)
        s.fin_snd(a, DUR, 71); a.save(path)


if __name__ == "__main__":
    r = Zenon()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
