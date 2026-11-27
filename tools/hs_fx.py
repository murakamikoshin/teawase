"""高品質用の 絵づくりの道具(オルレア級を めざす): もや(fbm)・細かい光のつぶ・レンズの光・光のすじ(ゴッドレイ)・キャラの ふち光・フィルム仕上げ。
Kit に まぜて使う: class X(FX, Kit)。場面の 中身(話・大道具)は 各台本で 書く。ここは「質感」だけ。"""
import math
import numpy as np, cv2


def sstep(a, b, x):
    t = np.clip((x - a) / (b - a + 1e-9), 0, 1); return t * t * (3 - 2 * t)


class FX:
    def fx_init(s, seed=1):
        rg = np.random.RandomState(seed * 7 + 3); s._tiles = []
        for i in range(4):                                                           # つなぎ目の ない ノイズ
            n = rg.rand(256, 256).astype(np.float32)
            n = cv2.GaussianBlur(np.tile(n, (3, 3)), (0, 0), 3 + 2 * i)[256:512, 256:512]; n = (n - n.min()) / (n.max() - n.min() + 1e-9); s._tiles.append(n)
        s._grain = rg.rand(8, s.H // 2, s.W // 2).astype(np.float32)
        s._rgx = rg
        cx, cy = s.W / 2, s.H / 2; s._rad = np.sqrt(((s.XX - cx) / (s.W * 0.75)) ** 2 + ((s.YY - cy) / (s.H * 0.62)) ** 2)

    # ---- もや・けむり・火・星雲 ----
    def fbm(s, sc=0.0010, ox=0.0, oy=0.0, octs=4, warp=0.0, tile=0):
        """0..1 の もや。sc: こまかさ、ox/oy: 流す量(px)、warp: うねり"""
        X = s.XX; Y = s.YY
        if warp:
            wx = cv2.remap(s._tiles[(tile + 1) % 4], ((X * sc * 0.5 + ox * 0.3) * 256) % 256, ((Y * sc * 0.5 + 37 + oy * 0.3) * 256) % 256, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            X = X + (wx - 0.5) * warp; Y = Y + (wx - 0.5) * warp * 0.7
        out = np.zeros((s.H, s.W), np.float32); amp = 0.5; tot = 0.0; f = sc
        for o in range(octs):
            mx = ((X * f + ox * f * (1 + 0.3 * o)) * 256) % 256; my = ((Y * f + oy * f * (1 + 0.3 * o) + o * 71) * 256) % 256
            out += cv2.remap(s._tiles[(tile + o) % 4], mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP) * amp
            tot += amp; amp *= 0.5; f *= 2.03
        return out / tot

    def ramp(s, v, stops):
        """v(0..1) を 色に。stops: [(位置, (r,g,b)), ...]"""
        out = np.zeros((s.H, s.W, 3), np.float32)
        for (p0, c0), (p1, c1) in zip(stops[:-1], stops[1:]):
            w = np.clip((v - p0) / (p1 - p0 + 1e-9), 0, 1)[:, :, None] * ((v >= p0)[:, :, None]); c0 = np.asarray(c0, np.float32); c1 = np.asarray(c1, np.float32)
            seg = (v >= p0) & (v < p1) if p1 < stops[-1][0] else (v >= p0)
            out[seg] = (c0[None, :] * (1 - w[seg]) + c1[None, :] * w[seg])
        return out

    def nebula(s, t, cols, sc=0.0006, drift=(6.0, 2.0), dens=1.0, tile=0):
        """色つきの もや(星雲・オーロラ・けむり)。cols: 2〜3色"""
        n = s.fbm(sc, t * drift[0], t * drift[1], 4, 60 * s.k, tile); m = s.fbm(sc * 2.2, -t * drift[0] * 0.6, t * drift[1] * 1.3, 3, 0, tile + 2)
        a = sstep(0.38, 0.80, n) * dens; c0 = np.asarray(cols[0], np.float32); c1 = np.asarray(cols[1], np.float32); c2 = np.asarray(cols[-1], np.float32)
        col = c0[None, None, :] * (1 - m)[:, :, None] + c1[None, None, :] * m[:, :, None]
        col = col * (1 - sstep(0.7, 0.95, n))[:, :, None] + c2[None, None, :] * sstep(0.7, 0.95, n)[:, :, None]
        return col * a[:, :, None]

    def flame(s, t, mask, speed=260.0, sc=0.0016, heat=1.0):
        """火: mask(0..1 の 形)の 中を 上へ 流れる 炎。もどり値は 強さ(0..1)"""
        n = s.fbm(sc, 0.0, t * speed, 4, 40 * s.k, 1); n2 = s.fbm(sc * 2.5, 13.0, t * speed * 1.6, 3, 0, 2)
        return np.clip((n * 0.7 + n2 * 0.5 - 0.42) * 2.4 * heat, 0, 1) * mask

    FIRE = [(0.0, (0.0, 0.0, 0.0)), (0.25, (0.55, 0.05, 0.0)), (0.55, (1.0, 0.38, 0.04)), (0.82, (1.0, 0.82, 0.30)), (1.0, (1.0, 1.0, 0.92))]

    # ---- 細かい 光のつぶ ----
    def splat(s, f, xs, ys, br, col, blur=1.0, glow=0.6):
        """点を たくさん 打つ(xs, ys, br は 同じ長さの 配列)"""
        xi = xs.astype(np.int32); yi = ys.astype(np.int32); ok = (xi >= 0) & (xi < s.W) & (yi >= 0) & (yi < s.H)
        buf = np.zeros((s.H, s.W), np.float32); np.add.at(buf, (yi[ok], xi[ok]), np.asarray(br, np.float32)[ok] if np.ndim(br) else float(br))
        g = cv2.GaussianBlur(buf, (0, 0), 0.75 * blur * s.k) * 6.0
        if glow: g += cv2.GaussianBlur(buf, (0, 0), 5 * blur * s.k) * 14 * glow
        f += g[:, :, None] * np.asarray(col, np.float32)[None, None, :]

    def starfield(s, f, t, n=900, col=(0.9, 0.95, 1.0), seed=0, v=1.0, drift=(0.0, 0.0)):
        if not hasattr(s, "_st") or s._st[0] != (n, seed):
            rg = np.random.RandomState(seed + 11); s._st = ((n, seed), rg.rand(n, 4).astype(np.float32))
        p = s._st[1]; x = (p[:, 0] * s.W + t * drift[0] * (0.3 + p[:, 2])) % s.W; y = (p[:, 1] * s.H + t * drift[1] * (0.3 + p[:, 2])) % s.H
        tw = 0.35 + 0.65 * np.sin(t * (1.5 + 3 * p[:, 3]) + p[:, 2] * 40) ** 2
        s.splat(f, x, y, (p[:, 2] ** 3) * tw * 0.9 * v, col, 0.9, 0.35)

    def rushdust(s, f, t, c, n=1400, speed=1.0, col=(1, 1, 1), v=1.0, seed=3, streak=0.0):
        """奥の 1点 c から こちらへ 流れてくる 細かい つぶ(streak>0 で 尾を ひく)"""
        if not hasattr(s, "_rd") or s._rd[0] != (n, seed):
            rg = np.random.RandomState(seed + 5); s._rd = ((n, seed), rg.rand(n, 4).astype(np.float32))
        p = s._rd[1]; a = p[:, 0] * 6.2832; z = (p[:, 1] + t * speed * (0.35 + 0.65 * p[:, 2])) % 1.0; r = (z ** 2.2) * s.H * 0.95 + 4 * s.k
        x = c[0] + np.cos(a) * r; y = c[1] + np.sin(a) * r; br = (0.15 + 0.85 * z) * (0.3 + 0.7 * p[:, 3]) * v
        s.splat(f, x, y, br, col, 0.9 + 0.6 * streak, 0.4)
        if streak > 0:
            for q in (0.96, 0.92, 0.86): s.splat(f, c[0] + np.cos(a) * r * q, c[1] + np.sin(a) * r * q, br * 0.5 * streak * z, col, 0.9, 0.0)

    # ---- レンズの光 ----
    def flare(s, f, c, size, col=(1, 1, 1), v=1.0, streak=1.0, rot=0.0, ghosts=True):
        """強い光: しん・十字の光・横に のびる すじ・うすい輪・レンズの うつりこみ"""
        if v <= 0.01 or size <= 0.5: return
        col = np.asarray(col, np.float32); k = s.k; x, y = float(c[0]), float(c[1])
        L = np.zeros((s.H, s.W), np.float32); cv2.circle(L, (int(x), int(y)), max(1, int(size * 0.16)), 1.0, -1, cv2.LINE_AA)
        core = cv2.GaussianBlur(L, (0, 0), max(1.0, size * 0.10)) * 2.2 + cv2.GaussianBlur(L, (0, 0), max(2.0, min(size * 0.5, 90 * k))) * 3.0
        S = np.zeros((s.H, s.W), np.float32)
        for i in range(4):
            a = rot + i * 1.5708; ln = size * (2.6 if i % 2 == 0 else 1.5)
            cv2.line(S, (int(x), int(y)), (int(x + math.cos(a) * ln), int(y + math.sin(a) * ln)), 1.0, 1, cv2.LINE_AA)
        core += cv2.GaussianBlur(S, (0, 0), max(0.8, size * 0.02)) * 1.6
        if streak > 0:
            A = np.zeros((s.H, s.W), np.float32); cv2.line(A, (int(x - size * 7 * streak), int(y)), (int(x + size * 7 * streak), int(y)), 1.0, max(1, int(size * 0.05)), cv2.LINE_AA)
            core += cv2.GaussianBlur(A, (0, 0), sigmaX=max(4.0, min(size * 2.0, 120 * k)), sigmaY=max(0.8, size * 0.035)) * 1.8
        f += (core * v)[:, :, None] * (col * 0.75 + 0.25)[None, None, :]
        R = np.zeros((s.H, s.W), np.float32); cv2.circle(R, (int(x), int(y)), max(2, int(size * 1.55)), 1.0, max(1, int(size * 0.05)), cv2.LINE_AA)
        f += (cv2.GaussianBlur(R, (0, 0), max(1.0, size * 0.05)) * 0.35 * v)[:, :, None] * (np.roll(col, 1) * 0.6 + col * 0.4)[None, None, :]
        if ghosts:
            cx, cy = s.W / 2, s.H / 2; G = np.zeros((s.H, s.W, 3), np.float32)
            for q, rr, tint in ((-0.45, 0.20, (0.5, 0.8, 1.0)), (-0.9, 0.34, (1.0, 0.6, 0.9)), (0.55, 0.12, (0.6, 1.0, 0.7))):
                p = (int(cx + (x - cx) * q), int(cy + (y - cy) * q)); g1 = np.zeros((s.H, s.W), np.float32); cv2.circle(g1, p, max(2, int(size * rr * 2.2)), 1.0, -1, cv2.LINE_AA)
                G += cv2.GaussianBlur(g1, (0, 0), max(1.5, size * 0.12))[:, :, None] * np.asarray(tint, np.float32)[None, None, :]
            f += G * 0.05 * v

    def godrays(s, f, c, strength=0.6, thr=0.55, n=10, decay=0.86, reach=0.35):
        """明るい所から 光のすじが のびる(c から 外へ)"""
        if strength <= 0.01: return f
        sm = cv2.resize(f, (s.W // 4, s.H // 4), interpolation=cv2.INTER_AREA); b = np.clip(sm - thr, 0, 2.0)
        acc = np.zeros_like(b); w = 1.0; cx, cy = c[0] / 4.0, c[1] / 4.0
        for i in range(1, n + 1):
            z = 1 + reach * i / n; M = np.float32([[z, 0, cx * (1 - z)], [0, z, cy * (1 - z)]])
            acc += cv2.warpAffine(b, M, (b.shape[1], b.shape[0]), flags=cv2.INTER_LINEAR) * w; w *= decay
        acc = cv2.GaussianBlur(acc / n, (0, 0), 1.2)
        return f + cv2.resize(acc, (s.W, s.H), interpolation=cv2.INTER_LINEAR) * strength * 2.2

    def glowline(s, f, pts, col, w=1.5, v=1.0, glow=1.0, closed=False):
        """細く するどい線 + にじみ(軌道・紋・刃の光)"""
        L = np.zeros((s.H, s.W), np.float32); cv2.polylines(L, [np.array(pts, np.int32)], closed, 1.0, max(1, int(w * s.k)), cv2.LINE_AA)
        col = np.asarray(col, np.float32)
        f += ((L * 1.1 + cv2.GaussianBlur(L, (0, 0), 2.5 * s.k) * 1.3 * glow + cv2.GaussianBlur(L, (0, 0), 10 * s.k) * 1.6 * glow) * v)[:, :, None] * col[None, None, :] + (L * 0.5 * v)[:, :, None]

    def glowmask(s, f, m, col, v=1.0, inner=0.9, glow=1.0, r=8.0):
        """形(0..1)を 光らせる"""
        col = np.asarray(col, np.float32)
        f += ((m * inner + cv2.GaussianBlur(m, (0, 0), r * 0.3 * s.k) * 0.6 * glow + cv2.GaussianBlur(m, (0, 0), r * s.k) * 0.9 * glow) * v)[:, :, None] * col[None, None, :]

    # ---- キャラを 場に なじませる ----
    def draw_lit(s, f, img, cx, cy, hfrac, rim=(1, 1, 1), rim_dir=(0.0, -1.0), rim_v=1.0, amb=None, amb_v=0.25, aura=None, aura_v=0.5, alpha=1.0, shade=1.0):
        """キャラを 置く + ふちの光(rim_dir の側から)+ 場の色の うつりこみ + うしろの 光/影"""
        sc = hfrac * s.H / s.ah; im = img
        if amb is not None or shade != 1.0:
            im = img.copy(); a_ = np.asarray(amb if amb is not None else (1, 1, 1), np.float32)
            im[:, :, :3] = im[:, :, :3] * shade * (1 - amb_v) + im[:, :, :3] * a_[None, None, :] * amb_v * 1.6 * shade
        al = img[:, :, 3]
        if aura is not None and aura_v > 0:
            A = np.zeros((s.H, s.W), np.float32)
            big = cv2.resize(al, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA); bh, bw = big.shape; x0 = int(cx - bw / 2); y0 = int(cy - bh / 2)
            xa, ya = max(0, x0), max(0, y0); xb, yb = min(s.W, x0 + bw), min(s.H, y0 + bh)
            if xb > xa and yb > ya:
                A[ya:yb, xa:xb] = big[ya - y0:yb - y0, xa - x0:xb - x0]
                f += (cv2.GaussianBlur(A, (0, 0), 14 * s.k) * aura_v)[:, :, None] * np.asarray(aura, np.float32)[None, None, :]
        s.put(f, im, cx, cy, sc, alpha, 0.0, 0.0); s._pl = (cx, cy, sc)
        if rim_v > 0:
            d = max(2, int(5 * s.k / max(sc, 0.2))); M = np.float32([[1, 0, -rim_dir[0] * d], [0, 1, -rim_dir[1] * d]])
            edge = np.clip(al - cv2.warpAffine(al, M, (al.shape[1], al.shape[0])), 0, 1); edge = cv2.GaussianBlur(edge, (0, 0), 1.2)
            rimg = np.zeros_like(img); rimg[:, :, :3] = np.asarray(rim, np.float32)[None, None, :]; rimg[:, :, 3] = edge
            s.put(f, rimg, cx, cy, sc, min(1.0, rim_v) * alpha, 0.0, 0.0)

    # ---- フィルム仕上げ ----
    def film(s, f, t, shake=0.0, bloom=0.55, thr=0.78, tint=(1, 1, 1), ca=1.0, grain=0.035, vig=0.55, sat=1.12, expo=1.0):
        f = np.maximum(f, 0) * expo
        br = np.clip(f - thr, 0, 4.0); sm = cv2.resize(br, (s.W // 4, s.H // 4), interpolation=cv2.INTER_AREA)
        b = cv2.resize(cv2.GaussianBlur(sm, (0, 0), 2.2) * 0.9 + cv2.GaussianBlur(sm, (0, 0), 7.0) * 1.0 + cv2.GaussianBlur(sm, (0, 0), 18.0) * 0.7, (s.W, s.H), interpolation=cv2.INTER_LINEAR)
        f = f + b * bloom * np.asarray(tint, np.float32)[None, None, :]
        if shake > 0.3:
            M = np.float32([[1, 0, math.sin(t * 113) * shake], [0, 1, math.cos(t * 97) * shake]]); f = cv2.warpAffine(f, M, (s.W, s.H), borderMode=cv2.BORDER_REFLECT)
        if ca > 0:                                                                   # 色の ずれ(ふちほど 大きい)
            z = 1 + 0.0035 * ca; cx, cy = s.W / 2, s.H / 2
            f[:, :, 0] = cv2.warpAffine(f[:, :, 0], np.float32([[z, 0, cx * (1 - z)], [0, z, cy * (1 - z)]]), (s.W, s.H), borderMode=cv2.BORDER_REFLECT)
            z = 1 - 0.0035 * ca; f[:, :, 2] = cv2.warpAffine(f[:, :, 2], np.float32([[z, 0, cx * (1 - z)], [0, z, cy * (1 - z)]]), (s.W, s.H), borderMode=cv2.BORDER_REFLECT)
        f = 1 - np.exp(-f * 1.35)                                                    # 明るい所を やわらかく つぶす
        g = f.mean(2, keepdims=True); f = g + (f - g) * sat
        f = f * (1 - vig * np.clip(s._rad, 0, 1.3) ** 2.2)[:, :, None]
        if grain > 0:
            gr = cv2.resize(s._grain[int(t * 30) % 8], (s.W, s.H), interpolation=cv2.INTER_LINEAR); f = f + (gr - 0.5)[:, :, None] * grain
        return np.clip(f, 0, 1)
