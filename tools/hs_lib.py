"""必殺技動画: 場面の部品(登場6・カットイン5・技14・しめ4)を キャラごとの絵コンテで組む。
使い方: python3 hs_lib.py <id> [出力先フォルダ] / python3 hs_lib.py all [フォルダ] [n分割] [何番目]
絵コンテは hs_board.py の BOARD。"""
import sys, os, math, subprocess, wave
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from scipy.signal import lfilter

FPS = 30
WHITE = np.array([1, 1, 1], np.float32)


def ease(x): x = min(1, max(0, x)); return 1 - (1 - x) ** 3
def clamp(x): return min(1, max(0, x))


class S(hz.R):
    def __init__(s, cid, elem, name, en, W=540):
        super().__init__(cid, elem, name, en, W)
        s.cid = cid
        s.chw = max(s.ch.shape[1], s.ch.shape[0] * 0.7)
        s.C = (s.W / 2, s.H * 0.42); s.Cb = (s.W / 2, s.H * 0.78)
        s.vt = [s.text_img(c, hz.SERIF, int((64 if len(name) <= 5 else 52) * s.k), 3) for c in name]
        s.ht = [s.text_img(c, hz.SERIF, int((70 if len(name) <= 5 else 56) * s.k), 3) for c in name]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        rg = np.random.RandomState(cid * 31 + 7); s.rg = rg
        s.shards = [(rg.rand() * 6.283, 0.3 + rg.rand() * 1.2, rg.rand() * 6.283, 0.5 + rg.rand()) for _ in range(46)]
        s.crk = []
        for i in range(16):
            a = i * 0.3927 + rg.rand() * 0.2; pts = [(0.0, a)]; r = 0.0
            while r < 1.6:
                r += 0.08 + rg.rand() * 0.12; a += (rg.rand() - 0.5) * 0.5; pts.append((r, a))
            s.crk.append(pts)
        s.rnd = rg.rand(200, 4).astype(np.float32)

    # ---- 共通 ----
    rr = 3
    def _prep(s):
        yy, xx = np.mgrid[0:s.H, 0:s.W].astype(np.float32)
        d = np.sqrt(((xx - s.C[0]) / s.W) ** 2 + ((yy - s.C[1]) / s.W) ** 2)
        s.rad = np.exp(-(d / 0.55) ** 2).astype(np.float32)
        e = np.sqrt(((xx - s.W / 2) / (s.W * 0.75)) ** 2 + ((yy - s.H / 2) / (s.H * 0.62)) ** 2)
        s.vig = np.clip(1.08 - 0.5 * e ** 2.4, 0.25, 1.08).astype(np.float32)
        sc = np.clip(s.col.mean() + (s.col - s.col.mean()) * 2.2, 0.05, 1.3); s.scol = (sc / sc.max()).astype(np.float32)   # こい属性色
        s.tint = (s.scol * 0.85 + np.array([0.06, 0.03, 0.12], np.float32))
        s.em = np.random.RandomState(s.cid * 13 + 5).rand(160, 5).astype(np.float32)

    def bg(s, t, dim=1.0):
        if not hasattr(s, "rad"): s._prep()
        k = s.rr
        f = super().bg(t, dim) * (1.5 + 0.14 * k) * (0.45 + 0.95 * s.scol)[None, None, :]
        f += s.rad[:, :, None] * s.tint[None, None, :] * (0.16 + 0.03 * k) * dim * (0.85 + 0.15 * math.sin(t * 2.2))
        L = s.L0()
        s.circle_magic(L, s.C, s.W * 0.66, t * 0.45, 0.10 + 0.02 * k)                      # うしろで ゆっくり回る 大きな陣
        s.ell(L, s.C, (s.W * 0.80, s.W * 0.22), 18 + t * 9, 0.10 + 0.015 * k, 1)
        s.ell(L, s.C, (s.W * 0.80, s.W * 0.22), -18 - t * 7, 0.10 + 0.015 * k, 1)
        P = s.L0()
        for i in range(34 + 14 * k):                                                         # 立ちのぼる光のつぶ
            r = s.em[i]; y = ((r[1] - t * (0.05 + 0.16 * r[2])) % 1.0) * s.H; x = (r[0] + 0.035 * math.sin(t * (1 + r[3] * 2) + r[4] * 6.283)) * s.W
            b = (0.35 + 0.65 * math.sin(t * 3 + r[4] * 20) ** 2) * (0.5 + 0.5 * r[3])
            cv2.circle(P, (int(x), int(y)), 1 + int(r[2] * 2.2), float(b), -1)
        P = P + cv2.GaussianBlur(P, (0, 0), 4 * s.k) * 1.6
        f += ((L + cv2.GaussianBlur(L, (0, 0), 5 * s.k) * 1.2)[:, :, None] * s.col[None, None, :] + P[:, :, None] * (s.col * 0.7 + 0.3)[None, None, :]) * dim
        return f

    def char(s, f, cx, cy, size=0.62, alpha=1.0, white=0.0, add=0.0):
        size *= 1.2
        if alpha > 0.6 and white < 0.6:                                                      # うしろに 属性色の後光
            s.glow(f, (cx, cy), size * s.W * 0.46, 0.20 + 0.03 * s.rr)
            L = s.L0(); s.circ(L, (cx, cy), size * s.W * 0.58, 0.35, 1); s.bloom(f, L, gain=0.5)
        s.put(f, s.ch, cx, cy, size * s.W / s.chw, alpha, add, white)

    def post(s, f):
        if not hasattr(s, "rad"): s._prep()
        br = np.clip(f.max(2) - 0.72, 0, 1.5)
        b = cv2.GaussianBlur(br, (0, 0), 20 * s.k) + cv2.GaussianBlur(br, (0, 0), 6 * s.k) * 0.5
        f = f + b[:, :, None] * (s.col * 0.85 + 0.15)[None, None, :] * (0.9 + 0.08 * s.rr)   # 明るい所から 属性色が にじむ
        return f * s.vig[:, :, None]

    def glow(s, f, c, r, v=0.5, col=None):
        g = s.L0(); cv2.circle(g, (int(c[0]), int(c[1])), int(r), float(v), -1)
        f += cv2.GaussianBlur(g, (0, 0), max(2, r * 0.45))[:, :, None] * (s.col if col is None else col)[None, None, :]

    def speed(s, L, u, n=30, ang=0.0, v=0.35):
        ca, sa = math.cos(ang), math.sin(ang)
        for i in range(n):
            r = s.rnd[i]; p = ((u * (1.2 + r[2] * 2) + r[0]) % 1.3 - 0.15)
            x = p * s.W * 1.2 if abs(ca) > 0.5 else r[1] * s.W; y = r[1] * s.H if abs(ca) > 0.5 else p * s.H * 1.1
            ln = s.W * (0.08 + 0.2 * r[3])
            s.line(L, (x, y), (x + ca * ln, y + sa * ln), v * (0.4 + r[3] * 0.6), 1)

    def hname(s, f, cy, u, per=0.13, slam=False):
        n = len(s.ht); step = min(78 * s.k, s.W * 0.86 / n); x0 = s.W / 2 - step * (n - 1) / 2
        for i, T in enumerate(s.ht):
            a = (u - i * per) / 0.18
            if a <= 0: continue
            if slam and a < 1:
                core, m, g = T; sc = 1 + (1 - a) * 2.2
                T2 = tuple(cv2.resize(x, None, fx=sc, fy=sc) for x in T); s.text(f, T2, x0 + i * step, cy, a, 1.0)
            else: s.text(f, T, x0 + i * step, cy, 1.0, 1.0)

    def vname(s, f, cx, y0, u, per=0.15):
        step = (74 if len(s.vt) <= 5 else 60) * s.k
        for i, T in enumerate(s.vt):
            if u > i * per: s.text(f, T, cx, y0 + i * step, clamp((u - i * per) / 0.12), 1.0)

    # ================= 登場 =================
    def i_shadow(s, u, d, t):
        f = s.bg(t, clamp(u / 0.6)); L = s.L0()
        if u > d * 0.45: s.circle_magic(L, s.C, s.W * 0.47 * ease((u - d * 0.45) / 0.5), t, 0.75)
        s.bloom(f, L, gain=0.8); s.glow(f, s.C, s.W * 0.36, 0.5 * clamp(u))
        img = s.ch.copy(); img[:, :, :3] *= clamp((u - 0.3) / (d * 0.5))
        s.put(f, img, s.C[0], s.C[1], 0.62 * s.W / s.chw, clamp(u / 0.4)); return f

    def i_drop(s, u, d, t):
        f = s.bg(t); e = ease(u / (d * 0.55)); y = -s.H * 0.25 + (s.C[1] + s.H * 0.25) * e
        L = s.L0(); s.speed(L, u, 24, 1.5708, 0.3 * (1 - e))
        for i in range(3):
            v = (u - d * 0.55 - i * 0.15) / 0.8
            if 0 < v < 1: s.ell(L, (s.C[0], s.C[1] + s.W * 0.3), (s.W * 0.9 * v, s.W * 0.2 * v), 0, 1 - v, 3)
        s.bloom(f, L); s.char(f, s.C[0], y, 0.62, 1.0, max(0, 1 - u / (d * 0.7))); return f

    def i_dash(s, u, d, t):
        f = s.bg(t); e = ease(u / (d * 0.6)); x = -s.W * 0.5 + (s.C[0] + s.W * 0.5) * e
        L = s.L0(); s.speed(L, u, 34, 0.0, 0.5 * (1 - e * 0.7)); s.bloom(f, L, WHITE, 0.8)
        for j in (3, 2, 1): s.char(f, x - j * s.W * 0.13 * (1 - e), s.C[1], 0.62, 0.22 * (1 - e), 0.6)
        s.char(f, x, s.C[1], 0.62); return f

    def i_rise(s, u, d, t):
        f = s.bg(t, 0.8); e = ease(u / (d * 0.7)); col = s.L0()
        w = s.W * (0.04 + 0.2 * math.sin(clamp(u / d) * 3.14))
        cv2.rectangle(col, (int(s.W / 2 - w), 0), (int(s.W / 2 + w), s.H), 1.0, -1)
        f += cv2.GaussianBlur(col, (0, 0), 20 * s.k)[:, :, None] * (s.col * 0.8 + 0.3)[None, None, :]
        L = s.L0(); s.speed(L, u, 20, -1.5708, 0.4); s.bloom(f, L)
        s.char(f, s.C[0], s.H * 1.2 + (s.C[1] - s.H * 1.2) * e, 0.62, 1.0, 0.5 * (1 - e)); return f

    def i_spin(s, u, d, t):
        f = s.bg(t); e = ease(u / (d * 0.7)); L = s.L0()
        for i in range(3):
            pts = [(s.C[0] + math.cos(i * 2.094 + u * 5 + j * 0.16) * s.W * 0.7 * j / 30 * (1.2 - e * 0.4), s.C[1] + math.sin(i * 2.094 + u * 5 + j * 0.16) * s.W * 0.7 * j / 30 * (1.2 - e * 0.4)) for j in range(30)]
            for j in range(29): s.line(L, pts[j], pts[j + 1], 0.3 + 0.5 * j / 29, 2)
        s.bloom(f, L, gain=1 - e * 0.5); s.char(f, s.C[0], s.C[1], 0.05 + 0.57 * e, 1.0, 1 - e); return f

    def i_flash(s, u, d, t):
        f = s.bg(t, 0.15 + 0.85 * clamp((u - d * 0.6) / 0.4)); fl = [d * 0.15, d * 0.33, d * 0.5]
        on = any(0 <= u - x < 0.09 for x in fl)
        if on:
            L = s.L0(); rg = np.random.RandomState(int(u * 30))
            for i in range(3):
                x = rg.rand() * s.W; pts = [(x, 0)]
                for j in range(1, 8): pts.append((x + (rg.rand() - 0.5) * s.W * 0.3, s.H * j / 7))
                for j in range(7): s.line(L, pts[j], pts[j + 1], 1.0, 3)
            s.bloom(f, L, s.col * 0.5 + 0.5, 1.2); f += 0.25
        s.char(f, s.C[0], s.C[1], 0.62, 1.0 if (on or u > d * 0.6) else 0.0, 1.0 if u < d * 0.6 else max(0, 1 - (u - d * 0.6) / 0.3)); return f

    # ================= カットイン =================
    def _dim(s, t):
        f = s.bg(t, 0.22); s.char(f, s.W / 2, s.H * 0.62, 0.9, 0.12); return f

    def c_band(s, u, d, t):
        W, H = s.W, s.H; f = s._dim(t); band = np.zeros((H, W, 3), np.float32); bm = s.L0()
        zoom = 1.55 * W / s.ch.shape[1]; fy = s.ch.shape[0] * 0.33
        s.put(band, s.ch, W / 2 + (u - 1) * 14 * s.k, H * 0.385 + (s.ch.shape[0] / 2 - fy) * zoom, zoom, 1.0)
        bh = H * 0.105 * clamp(u / 0.18); sl = math.tan(math.radians(-6)) * W / 2
        pts = [(0, H * 0.385 - bh - sl), (W, H * 0.385 - bh + sl), (W, H * 0.385 + bh + sl), (0, H * 0.385 + bh - sl)]
        s.poly(bm, pts, 1.0, 1, True)
        tint = np.zeros((H, W, 3), np.float32) + s.col * 0.42
        f = f * (1 - bm[:, :, None]) + (tint * (1 - np.clip(band.sum(2, keepdims=True) * 9, 0, 1)) + band * 0.96) * bm[:, :, None]
        L = s.L0(); s.line(L, pts[0], pts[1], 1.0, 3); s.line(L, pts[3], pts[2], 1.0, 3); s.speed(L, u, 22, -0.35, 0.3); s.bloom(f, L, gain=0.9)
        s.text(f, s.txt_ougi, W / 2, H * 0.215, clamp(u / 0.25), 1.0)
        s.text(f, s.txt_main, W / 2, H * 0.60, 1.0, clamp((u - 0.25) / (0.13 * len(s.name))))
        if s.txt_en is not None: s.text(f, s.txt_en, W / 2, H * 0.665, clamp((u - 0.25 - 0.13 * len(s.name)) / 0.3), 1.0)
        return f

    def c_eye(s, u, d, t):
        W, H = s.W, s.H; f = s.bg(t, 0.2); band = np.zeros((H, W, 3), np.float32)
        zoom = 1.9 * W / s.ch.shape[1]; fy = s.ch.shape[0] * 0.36
        s.put(band, s.ch, W * 0.44 - u * 6 * s.k, H * 0.40 + (s.ch.shape[0] / 2 - fy) * zoom, zoom, 1.0)
        m = s.L0(); pts = s.lens((-W * 0.05, H * 0.40), 0, W * 1.1, H * 0.085 * ease(u / 0.35)); s.poly(m, pts, 1.0, 1, True)
        f = f * (1 - m[:, :, None]) + (s.col * 0.4 * (1 - np.clip(band.sum(2, keepdims=True) * 9, 0, 1)) + band) * m[:, :, None]
        L = s.L0(); s.poly(L, pts, 1.0, 3); s.speed(L, u, 18, 1.5708, 0.25); s.bloom(f, L, WHITE, 0.9)
        for i, T in enumerate(s.vo): s.text(f, T, W * 0.12, H * 0.16 + i * 36 * s.k, clamp(u / 0.3), 1.0)
        s.vname(f, W * 0.86, H * 0.53, u - 0.3)
        if s.txt_en is not None and u > 1.1: s.text(f, s.txt_en, W * 0.40, H * 0.60, clamp((u - 1.1) / 0.3), 1.0)
        return f

    def c_split(s, u, d, t):
        W, H = s.W, s.H; f = s.bg(t, 0.2); e = ease(u / 0.3)
        pan = np.zeros((H, W, 3), np.float32) + s.col * 0.35; zoom = 1.25 * W / s.ch.shape[1]
        s.put(pan, s.ch, W * 0.30, H * 0.50 + s.ch.shape[0] * 0.10 * zoom, zoom, 1.0)
        m = s.L0(); x1 = W * 0.62 * e; s.poly(m, [(0, 0), (x1, 0), (x1 - W * 0.14, H), (0, H)], 1.0, 1, True)
        f = f * (1 - m[:, :, None]) + pan * m[:, :, None]
        L = s.L0(); s.line(L, (x1, 0), (x1 - W * 0.14, H), 1.0, 4); s.speed(L, u, 16, 1.5708, 0.2); s.bloom(f, L, gain=1.0)
        for i, T in enumerate(s.vo): s.text(f, T, W * 0.93, H * 0.12 + i * 36 * s.k, clamp(u / 0.3), 1.0)
        s.vname(f, W * 0.80, H * 0.26, u - 0.3)
        if s.txt_en is not None and u > 1.1: s.text(f, s.txt_en, W * 0.5, H * 0.93, clamp((u - 1.1) / 0.3), 1.0)
        return f

    def c_stamp(s, u, d, t):
        W, H = s.W, s.H; n = len(s.ht); k = int(clamp(u / 0.13 / n) * n)
        sh = (math.sin(u * 90) * 6 * s.k) if (u % 0.13) < 0.05 and u < 0.13 * n + 0.1 else 0
        f = s.bg(t, 0.35); s.glow(f, (W / 2, H * 0.40), W * 0.4, 0.5)
        s.char(f, W / 2 + sh, H * 0.40, 0.8, 1.0)
        bar = s.L0(); cv2.rectangle(bar, (0, int(H * 0.64)), (W, int(H * 0.76)), 1.0, -1); f *= (1 - bar * 0.8)[:, :, None]
        L = s.L0(); s.line(L, (0, H * 0.64), (W * clamp(u / 0.2), H * 0.64), 1.0, 2); s.line(L, (W, H * 0.76), (W * (1 - clamp(u / 0.2)), H * 0.76), 1.0, 2); s.bloom(f, L)
        s.text(f, s.txt_ougi, W / 2, H * 0.60, clamp(u / 0.25), 1.0); s.hname(f, H * 0.70, u - 0.1, 0.13, True)
        if s.txt_en is not None: s.text(f, s.txt_en, W / 2, H * 0.79, clamp((u - 0.2 - 0.13 * n) / 0.3), 1.0)
        return f

    def c_panels(s, u, d, t):
        W, H = s.W, s.H; f = s.bg(t, 0.15); L = s.L0()
        for i, (fy, zm) in enumerate(((0.30, 2.0), (0.55, 1.5), (0.82, 1.9))):
            e = ease((u - i * 0.14) / 0.25)
            if e <= 0: continue
            y0 = H * (0.10 + i * 0.165); hh = H * 0.15; pan = np.zeros((H, W, 3), np.float32) + s.col * (0.25 + 0.1 * i)
            zoom = zm * W / s.ch.shape[1]; dx = (1 - e) * W * (1 if i % 2 else -1)
            s.put(pan, s.ch, W / 2 + dx + (u * 10 * s.k * (1 if i % 2 else -1)), y0 + hh / 2 + (s.ch.shape[0] / 2 - s.ch.shape[0] * fy) * zoom, zoom, 1.0)
            m = s.L0(); cv2.rectangle(m, (int(dx if dx > 0 else 0), int(y0)), (int(W + (dx if dx < 0 else 0)), int(y0 + hh)), 1.0, -1)
            f = f * (1 - m[:, :, None]) + pan * m[:, :, None]
            s.line(L, (0, y0), (W, y0), 0.9 * e, 2); s.line(L, (0, y0 + hh), (W, y0 + hh), 0.9 * e, 2)
        s.bloom(f, L)
        s.text(f, s.txt_ougi, W / 2, H * 0.655, clamp((u - 0.4) / 0.25), 1.0); s.hname(f, H * 0.735, u - 0.5, 0.12)
        if s.txt_en is not None: s.text(f, s.txt_en, W / 2, H * 0.80, clamp((u - 0.6 - 0.12 * len(s.ht)) / 0.3), 1.0)
        return f

    # ================= 技 =================
    def a_slash(s, u, d, t):
        W, H = s.W, s.H; f = s.bg(t, 0.6); s.char(f, *s.Cb, 0.42); L = s.L0(); n = 7
        for i in range(n):
            t0 = i * d * 0.75 / n; v = (u - t0) / 0.12
            if v <= 0: continue
            r = s.rnd[i + 20]; a = (r[0] - 0.5) * 1.6 + (1.57 if i % 2 else 0); c = (W * (0.3 + 0.4 * r[1]), H * (0.2 + 0.35 * r[2]))
            ln = W * 1.3 * clamp(v); dx, dy = math.cos(a), math.sin(a)
            s.line(L, (c[0] - dx * W * 0.65, c[1] - dy * W * 0.65), (c[0] - dx * W * 0.65 + dx * ln, c[1] - dy * W * 0.65 + dy * ln), 1.0 if v < 2 else 0.45, 4 if v < 2 else 2)
        fin = u > d * 0.82
        s.bloom(f, L, WHITE if fin else None, 1.5 if fin else 1.0)
        if fin: f += 0.35 * (1 - (u - d * 0.82) / (d * 0.18))
        return f

    def a_meteor(s, u, d, t):
        W, H = s.W, s.H; f = s.bg(t, 0.7); s.char(f, W * 0.5, H * 0.2, 0.36); L = s.L0()
        for i in range(12):
            r = s.rnd[i + 40]; t0 = r[0] * d * 0.7; v = (u - t0) / 0.5
            if v <= 0: continue
            x0, x1 = W * (0.3 + r[1] * 1.0), W * (r[1] * 0.9 - 0.05); yv = H * (0.75 + 0.2 * r[2])
            if v < 1:
                x, y = x0 + (x1 - x0) * v, yv * v
                s.line(L, (x, y), (x + (x0 - x1) * 0.25, y - yv * 0.25), 0.8, 3); cv2.circle(L, (int(x), int(y)), int(W * 0.022), 1.0, -1, cv2.LINE_AA)
            elif v < 2.2: s.ell(L, (x1, yv), (W * 0.22 * (v - 1), W * 0.07 * (v - 1)), 0, max(0, 1 - (v - 1) / 1.2), 3)
        s.bloom(f, L, gain=1.2); return f

    def a_beam(s, u, d, t):
        W, H = s.W, s.H; f = s.bg(t, 0.6); cy = H * 0.46; cx = W * 0.26; ch = d * 0.4; L = s.L0()
        if u < ch:
            p = u / ch; cv2.circle(L, (int(cx + W * 0.18), int(cy)), int(W * 0.09 * p), 1.0, -1, cv2.LINE_AA)
            for i in range(12):
                a = i * 0.5236 + u * 3; rr = W * 0.35 * (1 - (u * 2 + i * 0.13) % 1.0)
                s.line(L, (cx + W * 0.18 + math.cos(a) * rr, cy + math.sin(a) * rr), (cx + W * 0.18 + math.cos(a) * rr * 0.8, cy + math.sin(a) * rr * 0.8), 0.7, 1)
        else:
            p = clamp((u - ch) / 0.15); hh = H * 0.07 * p * (1 + 0.15 * math.sin(u * 60))
            cv2.rectangle(L, (int(cx + W * 0.16), int(cy - hh)), (W, int(cy + hh)), 1.0, -1)
            for i in range(4):
                x = cx + W * 0.2 + ((u * 1.8 + i * 0.25) % 1.0) * W; s.ell(L, (x, cy), (W * 0.03, hh * 1.9), 0, 0.8, 2)
            s.speed(L, u, 26, 0.0, 0.5)
        s.bloom(f, L, gain=1.3); s.char(f, cx, cy, 0.42)
        if ch <= u < ch + 0.12: f += 0.5
        return f

    def a_orb(s, u, d, t):
        W, H = s.W, s.H; f = s.bg(t, 0.8); O = (W / 2, H * 0.29); L = s.L0(); p = clamp(u / (d * 0.6))
        for i in range(9): s.line(L, (s.Cb[0] + (i - 4) * W * 0.015, s.Cb[1] + W * 0.05), (s.Cb[0] + (i - 4) * W * 0.24, H), 0.5, 2)
        b2 = clamp((u - d * 0.5) / (d * 0.4))
        if b2 > 0:
            for a, r0 in s.dots:
                rr = W * (0.3 + 1.5 * ((r0 + u * 0.25) % 1.0)) * b2; x, y = O[0] + math.cos(a) * rr, O[1] + math.sin(a) * rr
                if 0 <= x < W and 0 <= y < H: cv2.circle(L, (int(x), int(y)), max(1, int(2.2 * s.k)), 0.9, -1, cv2.LINE_AA)
        s.bloom(f, L); s.char(f, *s.Cb, 0.44); s.motif(f, O, W * 0.275 * (0.4 + 0.6 * clamp(u / 0.5)), p, t); return f

    def a_vortex(s, u, d, t):
        W, H = s.W, s.H; f = s.bg(t, 0.6); O = (W / 2, H * 0.36); L = s.L0(); sp = 2 + u * 4
        for i in range(5):
            pts = [(O[0] + math.cos(i * 1.2566 + u * sp + j * 0.2) * W * 0.9 * j / 36, O[1] + math.sin(i * 1.2566 + u * sp + j * 0.2) * W * 0.9 * j / 36 * 0.8) for j in range(36)]
            for j in range(35): s.line(L, pts[j], pts[j + 1], 0.25 + 0.7 * (1 - j / 35), 3)
        for i in range(40):
            r = s.rnd[i + 60]; rr = W * 0.9 * (1 - (u * (0.5 + r[2]) + r[0]) % 1.0); a = r[1] * 6.283 + u * 3
            cv2.circle(L, (int(O[0] + math.cos(a) * rr), int(O[1] + math.sin(a) * rr * 0.8)), max(1, int(3 * s.k)), 0.9, -1, cv2.LINE_AA)
        s.bloom(f, L, gain=1.1)
        core = s.L0(); cv2.circle(core, (int(O[0]), int(O[1])), int(W * 0.07 * clamp(u / d * 1.5)), 1.0, -1, cv2.LINE_AA); f *= (1 - core)[:, :, None]
        s.char(f, *s.Cb, 0.42); return f

    def a_pillars(s, u, d, t):
        W, H = s.W, s.H; f = s.bg(t, 0.6); s.char(f, W / 2, H * 0.22, 0.36); L = s.L0(); col = s.L0(); gy = H * 0.86
        s.line(L, (0, gy), (W, gy), 0.6, 2)
        for i in range(5):
            t0 = i * d * 0.6 / 5; v = (u - t0) / 0.5
            if v <= 0: continue
            x = W * (0.12 + i * 0.19); w = W * 0.07 * math.sin(clamp(v / 1.6) * 3.14) + 2
            cv2.rectangle(col, (int(x - w), 0), (int(x + w), int(gy)), 1.0, -1); s.ell(L, (x, gy), (W * 0.14 * clamp(v), W * 0.04 * clamp(v)), 0, 0.9, 2)
        f += cv2.GaussianBlur(col, (0, 0), 9 * s.k)[:, :, None] * (s.col * 0.9 + 0.5)[None, None, :]
        s.bloom(f, L); return f

    def a_rush(s, u, d, t):
        W, H = s.W, s.H; f = s.bg(t, 0.6); L = s.L0(); P = [(0.2, 0.75), (0.8, 0.62), (0.22, 0.5), (0.78, 0.38), (0.25, 0.26), (0.5, 0.42)]; n = len(P); cur = min(n - 1, int(u / (d * 0.8) * n))
        s.speed(L, u, 26, 0.0, 0.3)
        for i in range(cur + 1):
            p = (P[i][0] * W, P[i][1] * H)
            if i > 0: s.line(L, (P[i - 1][0] * W, P[i - 1][1] * H), p, 1.0 if i == cur else 0.4, 4 if i == cur else 2)
        s.bloom(f, L, WHITE, 1.0)
        for i in range(cur + 1): s.char(f, P[i][0] * W, P[i][1] * H, 0.34 if i < n - 1 else 0.6, 1.0 if i == cur else 0.28, 0.0 if i == cur else 0.7)
        return f

    def a_bloom(s, u, d, t):
        W, H = s.W, s.H; f = s.bg(t, 0.6); L = s.L0(); c = s.C
        for k2, (n, rr, wd, sp) in enumerate(((8, 0.75, 0.16, 0.25), (8, 0.52, 0.13, -0.4), (6, 0.32, 0.1, 0.6))):
            q = ease((u - k2 * d * 0.18) / (d * 0.35))
            for i in range(n):
                a = t * sp + i * 6.283 / n + k2 * 0.39
                if q > 0: s.poly(L, s.lens(c, a, W * rr * q, W * wd * q), 0.9, 2, k2 == 2)
        s.circ(L, c, W * 0.82 * ease(u / d), 0.5, 2); s.bloom(f, L, gain=1.1); s.char(f, c[0], c[1], 0.44); return f

    def a_storm(s, u, d, t):
        W, H = s.W, s.H; f = s.bg(t, 0.5); L = s.L0(); k2 = int(u / 0.22); rg = np.random.RandomState(s.cid * 100 + k2)
        if (u % 0.22) < 0.13:
            for b in range(2 + (k2 % 2)):
                x = rg.rand() * W; pts = [(x, 0)]; gy = H * (0.6 + 0.25 * rg.rand())
                for j in range(1, 9): pts.append((x + (rg.rand() - 0.5) * W * 0.28, gy * j / 8))
                for j in range(8): s.line(L, pts[j], pts[j + 1], 1.0, 4)
                s.ell(L, pts[-1], (W * 0.15, W * 0.04), 0, 0.9, 2)
            f += 0.18
        s.bloom(f, L, s.col * 0.6 + 0.4, 1.4); s.char(f, *s.Cb, 0.42); return f

    def a_wave(s, u, d, t):
        W, H = s.W, s.H; f = s.bg(t, 0.6); L = s.L0(); top = H * (1.05 - 0.95 * ease(u / (d * 0.8)))
        for j in range(6):
            pts = [(x, top + j * H * 0.06 + math.sin(x * 0.02 / s.k + u * 6 + j) * H * 0.02) for x in range(0, s.W + 20, 20)]
            for i in range(len(pts) - 1): s.line(L, pts[i], pts[i + 1], 0.9 - j * 0.12, 3)
        for i in range(30):
            r = s.rnd[i + 100]; cv2.circle(L, (int(r[0] * W), int(top - r[1] * H * 0.05 + math.sin(u * 8 + i) * 6)), max(1, int(3 * s.k * r[2] + 1)), 0.9, -1, cv2.LINE_AA)
        fill = s.L0(); cv2.rectangle(fill, (0, int(max(0, top + H * 0.03))), (W, H), 0.35, -1); f += fill[:, :, None] * s.col[None, None, :]
        s.bloom(f, L); s.char(f, W / 2, min(s.C[1], top - H * 0.12) if top < H * 0.7 else s.C[1], 0.42); return f

    def a_stars(s, u, d, t):
        W, H = s.W, s.H; f = s.bg(t, 0.7); L = s.L0(); P = [(W * (0.15 + 0.7 * s.rnd[i + 130][0]), H * (0.1 + 0.48 * s.rnd[i + 130][1])) for i in range(9)]; n = int(clamp(u / (d * 0.6)) * 9)
        for i in range(n):
            cv2.circle(L, (int(P[i][0]), int(P[i][1])), int(5 * s.k), 1.0, -1, cv2.LINE_AA)
            s.line(L, (P[i][0] - 14 * s.k, P[i][1]), (P[i][0] + 14 * s.k, P[i][1]), 0.9, 1); s.line(L, (P[i][0], P[i][1] - 14 * s.k), (P[i][0], P[i][1] + 14 * s.k), 0.9, 1)
            if i > 0: s.line(L, P[i - 1], P[i], 0.7, 2)
        if u > d * 0.7:
            v = (u - d * 0.7) / (d * 0.3)
            for p in P:
                for j in range(8): s.line(L, p, (p[0] + math.cos(j * 0.785) * W * 0.5 * v, p[1] + math.sin(j * 0.785) * W * 0.5 * v), 0.8 * (1 - v * 0.5), 2)
        s.bloom(f, L, gain=1.2); s.char(f, *s.Cb, 0.42); return f

    def a_crush(s, u, d, t):
        W, H = s.W, s.H; f = s.bg(t, 0.6); L = s.L0(); hit = d * 0.5; gy = H * 0.78
        y = -W * 0.4 + (gy - W * 0.34 + W * 0.4) * (clamp(u / hit) ** 3); Rr = W * 0.36
        pts = [(W / 2 + math.cos(i * 1.0472 + 0.52) * Rr, y + math.sin(i * 1.0472 + 0.52) * Rr) for i in range(6)]
        blk = s.L0(); s.poly(blk, pts, 1.0, 1, True); f *= (1 - blk * 0.85)[:, :, None]; s.poly(L, pts, 1.0, 5)
        for i in range(6): s.line(L, (W / 2, y), ((W / 2 + pts[i][0]) / 2, (y + pts[i][1]) / 2), 0.5, 2)
        if u > hit:
            v = (u - hit) / (d - hit)
            for i in range(3): s.ell(L, (W / 2, gy), (W * 1.2 * ((v + i * 0.2) % 1.0), W * 0.3 * ((v + i * 0.2) % 1.0)), 0, 1 - ((v + i * 0.2) % 1.0), 3)
            for a, r0, a2, ln in s.cracks[:24]: s.line(L, (W / 2, gy), (W / 2 + math.cos(a) * W * (0.3 + ln) * clamp(v * 3), gy + math.sin(a) * W * 0.25 * clamp(v * 3)), 0.7, 2)
            if v < 0.12: f += 0.5
        s.bloom(f, L); s.char(f, W / 2, y - Rr * 0.2 if y > 0 else -H, 0.3); return f

    def a_rings(s, u, d, t):
        W, H = s.W, s.H; f = s.bg(t, 0.6); L = s.L0(); c = s.C
        for i in range(5):
            v = (u * 1.3 + i * 0.2) % 1.0; s.circ(L, c, W * 1.1 * v, 1 - v, 5)
        for i in range(36):
            a = i * 0.1745; r0 = W * (0.32 + 0.1 * math.sin(u * 20 + i)); s.line(L, (c[0] + math.cos(a) * r0, c[1] + math.sin(a) * r0), (c[0] + math.cos(a) * (r0 + W * 0.3), c[1] + math.sin(a) * (r0 + W * 0.3)), 0.6, 2)
        s.bloom(f, L, gain=1.1); sh = math.sin(u * 70) * 4 * s.k; s.char(f, c[0] + sh, c[1], 0.56 + 0.04 * math.sin(u * 30)); return f

    def a_blades(s, u, d, t):
        W, H = s.W, s.H; f = s.bg(t, 0.6); L = s.L0(); O = (W / 2, H * 0.34); go = d * 0.55; n = 22
        for i in range(n):
            a = i * 6.283 / n + u * 2.5
            if u < go:
                rr = W * 0.34 * ease(u / (go * 0.6)); p = (O[0] + math.cos(a) * rr, O[1] + math.sin(a) * rr * 0.75); ang = a + 1.57
            else:
                v = clamp((u - go - i * 0.012) / 0.25); p0 = (O[0] + math.cos(a) * W * 0.34, O[1] + math.sin(a) * W * 0.255); tg = (W * (0.2 + 0.6 * s.rnd[i + 150][0]), H * 0.86)
                p = (p0[0] + (tg[0] - p0[0]) * v, p0[1] + (tg[1] - p0[1]) * v); ang = math.atan2(tg[1] - p0[1], tg[0] - p0[0])
                if v >= 1: s.ell(L, tg, (W * 0.08, W * 0.025), 0, 0.7, 2)
            s.poly(L, s.lens((p[0] - math.cos(ang) * W * 0.05, p[1] - math.sin(ang) * W * 0.05), ang, W * 0.1, W * 0.012), 1.0, 2, True)
        s.bloom(f, L, WHITE * 0.6 + s.col * 0.4, 1.1); s.char(f, O[0], O[1], 0.36); return f

    # ================= しめ =================
    def f_white(s, u, d, t, base):
        f = base(); v = clamp(u / (d * 0.55)); s.glow(f, s.C, s.W * (0.2 + v * 3), 1.6 * v, s.col * 0.4 + 0.6)
        if u > d * 0.55: f = np.ones_like(f) * (1 - clamp((u - d * 0.55) / (d * 0.45))) ** 2
        return f

    def f_shatter(s, u, d, t, base):
        W, H = s.W, s.H; f = base() * (1 - 0.5 * clamp(u / d)); L = s.L0(); O = s.C; g = ease(u / (d * 0.35))
        for pts in s.crk:
            for j in range(len(pts) - 1):
                if pts[j][0] > g * 1.7: break
                s.line(L, (O[0] + math.cos(pts[j][1]) * pts[j][0] * W, O[1] + math.sin(pts[j][1]) * pts[j][0] * W), (O[0] + math.cos(pts[j + 1][1]) * pts[j + 1][0] * W, O[1] + math.sin(pts[j + 1][1]) * pts[j + 1][0] * W), 1.0, 2)
        for a, sp, rot, sz in s.shards:
            rr = W * (0.1 + max(0, u - d * 0.3) * sp * 1.3); c = (O[0] + math.cos(a) * rr, O[1] + math.sin(a) * rr)
            if u > d * 0.3 and -50 < c[0] < W + 50 and -50 < c[1] < H + 50:
                s.poly(L, [(c[0] + math.cos(rot + u * 5 * sp + j * 2.094) * W * 0.035 * sz, c[1] + math.sin(rot + u * 5 * sp + j * 2.094) * W * 0.035 * sz) for j in range(3)], 0.9, 2)
        s.bloom(f, L, WHITE)
        if u > d * 0.7: v = clamp((u - d * 0.7) / (d * 0.3)); f = f * (1 - v) ** 2
        return f

    def f_iris(s, u, d, t, base):
        f = base(); v = ease(u / (d * 0.75)); r = s.W * 1.3 * (1 - v)
        m = s.L0(); cv2.circle(m, (int(s.C[0]), int(s.C[1])), int(max(0, r)), 1.0, -1, cv2.LINE_AA); f *= m[:, :, None]
        L = s.L0(); s.circ(L, s.C, r, 1.0, 4)
        if v > 0.95:
            for j in range(4): s.line(L, s.C, (s.C[0] + math.cos(j * 1.57) * s.W * 0.12 * (1 - clamp((u - d * 0.75) / (d * 0.25))), s.C[1] + math.sin(j * 1.57) * s.W * 0.12 * (1 - clamp((u - d * 0.75) / (d * 0.25)))), 1.0, 2)
        s.bloom(f, L, WHITE); return f

    def f_cut(s, u, d, t, base):
        W, H = s.W, s.H; f = base(); v = ease((u - d * 0.25) / (d * 0.5)); m = s.L0()
        s.poly(m, [(0, 0), (W, 0), (W, H * 0.3), (0, H * 0.7)], 1.0, 1, True); m3 = m[:, :, None]
        if v > 0:
            sh = int(W * 0.25 * v); f = np.roll(f * m3, (-sh // 2, -sh), (0, 1)) + np.roll(f * (1 - m3), (sh // 2, sh), (0, 1))
        L = s.L0(); s.line(L, (W, H * 0.3), (W - W * clamp(u / (d * 0.2)), H * 0.3 + H * 0.4 * clamp(u / (d * 0.2))), 1.0, 4); s.bloom(f, L, WHITE, 1.4 * (1 - v * 0.7))
        if u > d * 0.75: f = f * (1 - clamp((u - d * 0.75) / (d * 0.25))) ** 2
        return f

    # ---- 絵コンテを再生 ----
    def setup(s, board):
        s.seq = []; t0 = 0.0
        for name, d in board:
            s.seq.append((getattr(s, name), d, t0, name)); t0 += d
        s.total = t0

    def frame(s, t): return s.post(s.frame0(t))

    def frame0(s, t):
        for i, (fn, d, t0, name) in enumerate(s.seq):
            if t < t0 + d or i == len(s.seq) - 1:
                u = min(t - t0, d)
                if name.startswith("f_"):
                    pf, pd, pt0, pn = s.seq[i - 1]
                    return fn(u, d, t, lambda: pf(pd, pd, pt0 + pd))
                f = fn(u, d, t)
                if i > 0 and u < 0.1 and not name.startswith("c_"): f = f + (1 - u / 0.1) * 0.6        # 場面がわりの白
                return f

    def audio(s, path):
        sr = 44100; D = s.total; n = int(D * sr); t = np.arange(n) / sr; a = np.zeros(n, np.float32)
        base = {"fire": 55, "thunder": 62, "dark": 49, "poison": 49, "light": 65, "star": 73, "ice": 69, "water": 58, "wind": 65, "leaf": 62, "petal": 65, "earth": 46}[s.elem]
        rng = np.random.RandomState(5); nz = rng.randn(n).astype(np.float32)
        def lp(x, kk): return lfilter([kk], [1, kk - 1], x).astype(np.float32)
        def env(t0, at, dec): return np.clip((t - t0) / at, 0, 1) * np.exp(-np.clip(t - t0 - at, 0, None) / dec) * (t >= t0)
        def hit(t0, g=1.0, fq=None): return (np.sin(2 * np.pi * (fq or base * 0.9) * (t - t0) * np.exp(-np.clip(t - t0, 0, None) * 2.5)) * env(t0, 0.004, 0.33) + lp(nz, 0.5) * env(t0, 0.003, 0.2) * 0.5) * g
        def swish(t0, g=0.4): return (nz - lp(nz, 0.3)) * env(t0, 0.02, 0.07) * g
        a += (np.sin(2 * np.pi * base * t) + 0.4 * np.sin(2 * np.pi * base * 2.01 * t) + 0.25 * np.sin(2 * np.pi * base * 1.5 * t)) * 0.17 * np.clip(t / 0.8, 0, 1)
        for fn, d, t0, name in s.seq:
            if name.startswith("i_"):
                a += lp(nz, 0.25) * 0.4 * np.clip((t - t0) / d, 0, 1) ** 2 * ((t >= t0) & (t < t0 + d)); a += hit(t0 + d * 0.6, 0.8)
            elif name.startswith("c_"):
                a += hit(t0, 1.0)
                for i in range(len(s.name)): a += np.sin(2 * np.pi * (base * 8 * (1 + i * 0.06)) * t) * env(t0 + 0.3 + i * 0.13, 0.003, 0.07) * 0.17
            elif name.startswith("a_"):
                a += hit(t0, 0.8)
                reps = {"a_slash": 7, "a_meteor": 8, "a_storm": int(d / 0.22), "a_pillars": 5, "a_rush": 6, "a_blades": 1, "a_stars": 9}.get(name, 0)
                for i in range(reps):
                    tt = t0 + (i + 0.5) * d * 0.8 / max(1, reps)
                    a += swish(tt, 0.5) if name in ("a_slash", "a_rush") else (np.sin(2 * np.pi * base * 12 * t) * env(tt, 0.002, 0.08) * 0.15 if name == "a_stars" else hit(tt, 0.55, base * 1.3))
                if name in ("a_beam", "a_orb", "a_vortex", "a_wave", "a_bloom", "a_rings", "a_crush", "a_blades"):
                    fr = base * 2 * (1 + 1.5 * np.clip((t - t0) / d, 0, 1) ** 2); ph = 2 * np.pi * np.cumsum(fr) / sr
                    a += (np.sin(ph) * 0.14 + lp(nz, 0.2) * 0.22) * np.clip((t - t0) / d, 0, 1) * ((t >= t0) & (t < t0 + d)); a += hit(t0 + d * 0.5, 0.9)
            else:
                a += hit(t0, 1.2, base * 0.8)
                for j, m in enumerate((1, 1.5, 2, 3)): a += np.sin(2 * np.pi * base * 2 * m * t + j) * env(t0 + j * 0.08, 0.2, 0.7) * 0.09
        # ---- キャラごとの曲: 調・テンポ・旋律・音色・リズムを id から決める ----
        mg = np.random.RandomState(s.cid * 97 + 3)
        SC = {"dark": [0, 1, 3, 5, 7, 8, 10], "poison": [0, 1, 4, 5, 7, 8, 10], "fire": [0, 2, 3, 5, 7, 8, 11], "thunder": [0, 2, 3, 6, 7, 8, 11], "earth": [0, 2, 3, 5, 7, 9, 10],
              "light": [0, 2, 4, 6, 7, 9, 11], "star": [0, 2, 4, 6, 7, 9, 11], "leaf": [0, 2, 4, 5, 7, 9, 11], "petal": [0, 2, 4, 7, 9, 12, 14], "wind": [0, 2, 4, 5, 7, 9, 10],
              "water": [0, 2, 3, 5, 7, 9, 10], "ice": [0, 2, 3, 5, 7, 8, 10]}[s.elem]
        bpm = [112, 124, 136, 148, 160, 172][mg.randint(6)] + (s.rr - 3) * 4; st = 60.0 / bpm / 2
        root = base * 4 * 2 ** (mg.randint(-3, 4) / 12.0); tim = mg.randint(4)
        motif = [mg.randint(0, 7) + 7 * mg.randint(0, 2) for _ in range(8)]; motif2 = [(m + mg.randint(-2, 3)) % 14 for m in motif]
        rest = mg.rand(8) < 0.25; kick = mg.rand(8) < 0.42; kick[0] = True; bassd = [0, 0, mg.randint(2, 6), 0, mg.randint(3, 7), 0, mg.randint(1, 5), 0]
        t_in = s.seq[1][2] if len(s.seq) > 1 else 0.0; t_atk = next((q[2] for q in s.seq if q[3].startswith("a_")), t_in); t_fin = s.seq[-1][2]
        def fq(dg): return root * 2 ** ((SC[dg % 7] + 12 * (dg // 7)) / 12.0)
        def voice(f0, t0, dec, g):
            ph = 2 * np.pi * f0 * (t - t0); e = env(t0, 0.004, dec) * g
            if tim == 0: w = np.sin(ph) + 0.5 * np.sin(2 * ph) + 0.33 * np.sin(3 * ph) + 0.25 * np.sin(4 * ph)       # のこぎり風
            elif tim == 1: w = np.sin(ph) + 0.33 * np.sin(3 * ph) + 0.2 * np.sin(5 * ph)                              # 四角風
            elif tim == 2: w = np.sin(ph) + 0.6 * np.sin(2.76 * ph) * np.exp(-np.clip(t - t0, 0, None) * 9)           # 鐘
            else: w = np.sin(ph + 2.5 * np.sin(2 * ph) * np.exp(-np.clip(t - t0, 0, None) * 6))                       # FM
            return w * e
        m = np.zeros(n, np.float32); k = 0; tk = t_in * 0.5
        while tk < t_fin + 0.3:
            j = k % 8; bar = k // 8; hot = tk >= t_atk
            if not rest[j] or hot:
                dg = (motif2 if bar % 2 else motif)[j] + (7 if hot and bar % 2 else 0)
                m += voice(fq(dg), tk, 0.10 + 0.08 * (tim == 2), 0.16 if hot else 0.10)
                if s.rr >= 5 and hot: m += voice(fq(dg + 2), tk, 0.10, 0.07)
            if j % 2 == 0: m += voice(fq(bassd[j]) / 4, tk, 0.16, 0.20)
            if tk >= t_in:
                if kick[j]: m += (np.sin(2 * np.pi * 52 * (t - tk) * np.exp(-np.clip(t - tk, 0, None) * 14)) * env(tk, 0.002, 0.11)) * (0.5 if hot else 0.3)
                if hot or j % 2: m += (nz - lp(nz, 0.6)) * env(tk, 0.001, 0.025) * 0.10
                if j == 4: m += lp(nz, 0.7) * env(tk, 0.001, 0.07) * (0.22 if hot else 0.10)
            k += 1; tk += st
        for j, dgc in enumerate((0, 2, 4)): m += np.sin(2 * np.pi * fq(dgc + 7) * t + j) * env(t_fin, 0.05, 0.9) * 0.10   # しめの和音
        a = a * 0.8 + m * 0.9
        a *= np.clip((D - t) / 0.4, 0, 1); a = np.tanh(a * 1.1) * 0.85
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((a * 32767).astype(np.int16).tobytes())


def render(cid, outdir, W=540):
    import hs_board
    b = hs_board.BOARD[cid]
    r = S(cid, b["elem"], b["name"], b["en"], W); r.rr = b["r"]; r.setup(b["seq"])
    out = "%s/%d.mp4" % (outdir, cid); wav = out + ".wav"; r.audio(wav)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (r.W, r.H), "-r", str(FPS), "-i", "-", "-i", wav,
           "-c:v", "libx264", "-preset", "medium", "-crf", "27", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", "-shortest", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(int(r.total * FPS)):
        p.stdin.write((np.clip(r.frame(i / FPS), 0, 1) * 255).astype(np.uint8).tobytes())
    p.stdin.close(); p.wait(); os.remove(wav); print(cid, b["name"], "%.1fs" % r.total, os.path.getsize(out) // 1024, "KB", flush=True)


if __name__ == "__main__":
    import hs_board
    outdir = sys.argv[2] if len(sys.argv) > 2 else "/home/claude/hissatsu"
    if sys.argv[1] == "all":
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 1; k = int(sys.argv[4]) if len(sys.argv) > 4 else 0
        ids = [i for i in sorted(hs_board.BOARD) if not os.path.exists("%s/%d.mp4" % (outdir, i))]
        for i in ids[k::n]:
            try: render(i, outdir)
            except Exception as e: print("FAIL", i, repr(e), flush=True)
    else:
        for i in sys.argv[1].split(","): render(int(i), outdir)
