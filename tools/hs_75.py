"""ユグド(★6 世界樹の子) 専用の必殺技「世界樹ノ芽吹キ」— 高品質版(hs_fx)・通し直し。世界樹の 枝が こちらへ のび、目の前で 芽が ひらく
ひびわれた 灰色の 荒れ地。杖を つくと 地の ひびに 緑が 走り、ふた葉が ぽんと 出る(前ふり: 芽吹き) → 年輪の 丸窓のカットイン →
早回し: 日が 何度も 空を わたり、春夏秋冬が めぐる あいだに、ふた葉が 枝分かれして 大樹に なる → 夜。根が ひろがり、光の つぶが 幹を のぼる →
こずえから 1本の 枝が こちらへ のびてきて、目の前で 芽が ふくらみ、葉が ひらいて 視界を おおう。まん中から 木もれ日 → 白 → 暗転
使い方: python3 hs_75.py 出力.mp4 / python3 hs_75.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old75", "/home/claude/tools/pre_fin/hs_75_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = 10.4
LEAF, BARK, SUNC, MOONC, SEASON = old.LEAF, old.BARK, old.SUNC, old.MOONC, old.SEASON
T_PLANT, T_SPROUT, T_CUT, T_GROW = old.T_PLANT, old.T_SPROUT, old.T_CUT, old.T_GROW
T_WORLD, T_REACH, T_OPEN, T_HIT = 7.6, 8.35, 9.0, 9.7


class Yggd(FX, old.Yggd):
    def __init__(s, W=540):
        old.Yggd.__init__(s, W); s.fx_init(75); H = s.H; k = s.k; s.yv = (s.YY / H).astype(np.float32); s._tt = 0.0; s._sun = SUNC
        s._bark = cv2.remap(s._tiles[1], ((s.XX * 2.4 / k) % 256).astype(np.float32), ((s.YY * 0.28 / k) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        s._ltex = cv2.remap(s._tiles[0], ((s.XX * 3.2 / k) % 256).astype(np.float32), ((s.YY * 3.2 / k) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        g0 = s.gnd; dz = (H - g0) / (np.clip(s.YY - g0, 0, None) + 5 * k); mx = (((s.XX - W / 2) * dz * 0.42 / k + 128) % 256).astype(np.float32); my = ((dz * 95) % 256).astype(np.float32)
        s._gt = cv2.remap(s._tiles[2], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP); s._gc = sstep(0.05, 0.0, np.abs(cv2.remap(s._tiles[3], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP) - 0.5))
        hx = np.interp((np.arange(W) * 0.4 / k) % 255, np.arange(256), s._tiles[2][9]); s._ridge = sstep(0, 3 * k, s.YY - (g0 - H * (0.015 + 0.05 * hx))[None, :]); s._gm = sstep(g0 - 1.5, g0 + 1.5, s.YY)

    # ---- 荒れ地(→ 草地) ----
    def land(s, sky, lum=1.0, green=0.0):
        W, H, k = s.W, s.H, s.k; sky = np.asarray(sky, np.float32); n = s.fbm(0.0009, s._tt * 22, 0, 4, 60 * k, 0)
        f = sky[None, None, :] * ((0.22 + 0.78 * s.yv) * (0.65 + 0.6 * n))[:, :, None]
        f += (np.exp(-np.clip(s.gnd - s.YY, 0, None) / (H * 0.10)) * 0.16)[:, :, None] * (sky * 0.5 + 0.5)[None, None, :]
        rc = (sky * 0.22 * (1 - green) + np.array([0.05, 0.16, 0.08], np.float32) * green)[None, None, :] * (0.7 + 0.6 * s._gt)[:, :, None]; f = f * (1 - s._ridge[:, :, None]) + rc * s._ridge[:, :, None]
        dry = np.array([0.17, 0.145, 0.13], np.float32)[None, None, :] * ((0.45 + 0.9 * s._gt) * (1 - 0.65 * s._gc))[:, :, None]
        grs = np.array([0.06, 0.24, 0.09], np.float32)[None, None, :] * (0.45 + 1.1 * s._gt)[:, :, None] + (sstep(0.66, 0.9, s._gt) * 0.22)[:, :, None] * np.array([0.45, 0.8, 0.3], np.float32)[None, None, :]
        gc = dry * (1 - green) + grs * green; f = f * (1 - s._gm[:, :, None]) + gc * s._gm[:, :, None]
        return (f * lum).astype(np.float32)

    def tree(s, f, g, leafc, leaves=1.0, lit=0.0):
        """g: 0..1 どこまで 育ったか。樹皮の すじ・丸み・こけ、葉の かたまりは 上から てらす"""
        tt = g * s.tmax; D = s.L0(); k = s.k; sun = np.asarray(s._sun, np.float32)
        for (p, q, d, t0, dt) in s.seg:
            if t0 > tt: continue
            a = clamp((tt - t0) / dt); e = (p[0] + (q[0] - p[0]) * a, p[1] + (q[1] - p[1]) * a)
            cv2.line(D, (int(p[0]), int(p[1])), (int(e[0]), int(e[1])), 1.0, max(1, int((24 - d * 3.3) * k * (0.35 + 0.65 * g))), cv2.LINE_AA)
        if D.max() > 0:
            rnd = np.clip(cv2.GaussianBlur(D, (0, 0), 4 * k) * 1.5 - 0.35, 0, 1); side = np.clip(D - np.roll(D, int(4 * k), axis=1), 0, 1)
            col = BARK[None, None, :] * ((0.25 + 0.65 * s._bark) * (0.45 + 0.75 * rnd) * (1 + 0.8 * lit))[:, :, None] + (side * 0.30)[:, :, None] * sun[None, None, :] + (sstep(0.62, 0.85, s._ltex) * 0.10 * rnd)[:, :, None] * LEAF[None, None, :]
            f[:] = f * (1 - D[:, :, None]) + col * D[:, :, None]
        if leaves > 0:
            L = s.L0()
            for i, (q, te, _) in enumerate(s.tips):
                if te > tt: continue
                a = back((tt - te) / (s.tmax * 0.08), 1.5); r = s.rd[i % 400]
                cv2.circle(L, (int(q[0] + (r[0] - 0.5) * 10 * k), int(q[1] + (r[1] - 0.5) * 10 * k)), max(1, int((6 + 7 * r[2]) * k * a * leaves)), float(0.55 + 0.45 * r[3]), -1, cv2.LINE_AA)
            al = np.clip(cv2.GaussianBlur(L, (0, 0), 0.8 * k), 0, 1) * (0.55 + 0.45 * sstep(0.25, 0.5, s._ltex)); Lb = cv2.GaussianBlur(L, (0, 0), 5 * k); top = np.clip(Lb - np.roll(Lb, int(8 * k), axis=0), -0.4, 0.4)
            lc = np.asarray(leafc, np.float32)[None, None, :] * ((0.40 + 0.6 * s._ltex) * (0.70 + 1.7 * top) * (0.8 + 0.7 * lit))[:, :, None] + (np.clip(top, 0, 1) * 0.35)[:, :, None] * sun[None, None, :]
            f[:] = f * (1 - al[:, :, None] * 0.94) + lc * al[:, :, None] * 0.94
            if lit > 0: f += cv2.GaussianBlur(L, (0, 0), 14 * k)[:, :, None] * LEAF[None, None, :] * 0.45 * lit

    def gleaf(s, f, c, S, rot, al=1.0, lite=0.0, wide=1.0):
        """葉 1枚(式): 葉脈・ふちの ぎざ・すける 光"""
        W, H = s.W, s.H; x0 = int(max(0, c[0] - S - 2)); x1 = int(min(W, c[0] + S + 2)); y0 = int(max(0, c[1] - S - 2)); y1 = int(min(H, c[1] + S + 2))
        if x1 - x0 < 2 or y1 - y0 < 2 or al <= 0.01 or S < 1.5: return
        dx = s.XX[y0:y1, x0:x1] - c[0]; dy = s.YY[y0:y1, x0:x1] - c[1]; ca, sa = math.cos(rot), math.sin(rot); u = (dx * ca + dy * sa) / S; w = (-dx * sa + dy * ca) / S
        half = 0.40 * wide * np.clip(1 - u * u, 0, 1) ** 0.62 * (1 - 0.30 * u) * (1 + 0.03 * np.sin(u * 60)); aa = min(0.25, 1.4 / S)
        m = sstep(half + aa * 0.5, half - aa * 0.5, np.abs(w)) * (np.abs(u) < 1) * al; rel = np.abs(w) / (half + 1e-3)
        rib = np.exp(-(w / (0.018 + aa * 0.3)) ** 2); vn = np.exp(-((((u * 7 - np.abs(w) * 9) % 1.0) - 0.5) / 0.10) ** 2) * (rel < 0.95); tx = s._ltex[y0:y1, x0:x1]
        base = np.array([0.10, 0.42, 0.12], np.float32) * (1 - lite) + np.array([0.55, 0.95, 0.30], np.float32) * lite
        col = base[None, None, :] * ((0.60 + 0.5 * tx) * (0.75 + 0.45 * (1 - rel)) * (1 - 0.30 * vn) * (0.8 + 0.4 * (0.5 - 0.5 * u)))[:, :, None] + ((rib * 0.35 + vn * 0.10) * (0.4 + 0.9 * lite))[:, :, None] * np.array([0.85, 1.0, 0.55], np.float32)[None, None, :]
        col += (np.exp(-((rel - 1) / 0.10) ** 2) * 0.12 * (1 + 2 * lite))[:, :, None] * np.array([1.0, 1.0, 0.7], np.float32)[None, None, :]
        F = f[y0:y1, x0:x1]; F[:] = F * (1 - m[:, :, None]) + col * m[:, :, None]

    def sprout(s, f, p, a, t):
        if a <= 0.02: return
        k = s.k; h = 30 * k * min(a, 1.3); s.glowline(f, [p, (p[0], p[1] - h)], LEAF * 0.7, 1.6, 0.6, 0.6)
        for sg in (-1, 1): s.gleaf(f, (p[0] + sg * 13 * k * a, p[1] - h - 3 * k), 15 * k * a, (0.45 if sg > 0 else 3.1416 - 0.45) * -1 if sg > 0 else 3.1416 + 0.45, 1.0, 0.8, 1.2)
        f += (np.exp(-np.hypot(s.XX - p[0], s.YY - p[1] + h) / (22 * k)) * 0.35 * min(1.0, a))[:, :, None] * LEAF[None, None, :]

    def gem_glow(s, f, v, t):
        if v <= 0.01: return
        p = s.at(*s.gem); sc = s._pl[2]; s.flare(f, p, (7 + 2 * math.sin(t * 6)) * sc * 2 * min(1.3, v), LEAF * 0.8 + 0.2, 0.55 * min(1.3, v), 0.6, 0.0, False)

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(np.asarray(s._sun, np.float32) * 0.8 + 0.2), rim_dir=(0.5, -0.86), rim_v=0.7 * alpha, amb=(0.7, 1.0, 0.7), amb_v=0.12, alpha=alpha, shade=0.9)

    def add(s, f, L, col, gain=1.0, b1=3, b2=12):
        s.glowmask(f, np.clip(L, 0, 1), col, gain * 0.8, 1.0, 1.0, float(b2))

    def frame(s, t):
        o = {}; s._tt = t; s._sun = SUNC
        if t < T_CUT: f, sh = s.sc_plant(t); f = f * clamp(t / 0.4)
        elif t < T_GROW: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.9)
        elif t < T_WORLD: f, sh = s.sc_grow(t)
        else: f, sh = s.sc_world(t)
        f, sh = s.finale(f, sh, t, DUR, None, (0.45, 1, 0.5), (1, 0.9, 0.5), -1)
        return s.film(f, t, sh * s.k, 0.5, o.get("thr", 0.80), (0.92, 1.0, 0.88), 1.0 + 0.15 * sh, 0.04, o.get("vig", 0.55), 1.12, 1.10)

    # 2) 年輪の 丸窓
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT; C = (W / 2, H * 0.36)
        d = np.hypot(s.XX - C[0], s.YY - C[1]) / k; ang = np.arctan2(s.YY - C[1], s.XX - C[0]); wob = 1 + 0.035 * np.sin(ang * 3 + 1) + 0.02 * np.sin(ang * 7)
        ring = 0.5 + 0.5 * np.sin(d * wob * 0.32 + (s._ltex - 0.5) * 2.0 + 6 * s._tiles[2][7, 7]); rev = sstep(W * 1.5 * ease(u / 0.6) / k + 10, W * 1.5 * ease(u / 0.6) / k - 30, d)        # 年輪が 外へ ひろがる
        f = BARK[None, None, :] * ((0.07 + 0.20 * ring ** 2 + 0.06 * s._bark) * rev)[:, :, None] + (sstep(0.93, 1.0, ring) * rev * 0.06)[:, :, None] * LEAF[None, None, :]
        R = W * 0.34 * back(u / 0.35, 1.2)
        win = s.nebula(t, [(0.05, 0.25, 0.08), (0.20, 0.55, 0.18), (0.85, 1.0, 0.55)], 0.0015, (8.0, 14.0), 0.9, 1) + (LEAF * 0.14 + BARK * 0.12)[None, None, :]
        win = s.godrays(win + (np.exp(-np.hypot(s.XX - C[0], s.YY - (C[1] - R)) / (80 * k)) * 0.6)[:, :, None] * SUNC[None, None, :], (C[0], C[1] - R), 0.5, 0.4)
        ex = (s.eyeL[0] + s.eyeR[0]) / 2; ey = (s.eyeL[1] + s.eyeR[1]) / 2; zf = 1.35; sc = zf * H / s.ah
        img = s.pose(t); s.draw(win, img, C[0] - (ex - s.aw / 2) * sc, C[1] - (ey - s.ah / 2) * sc, zf)
        for e in (s.eyeL, s.eyeR): s.flare(win, s.at(*e), 9 * sc, LEAF, 0.35 + 0.12 * math.sin(u * 9), 0.0, 0.0, False)
        m = s.L0(); cv2.circle(m, (int(C[0]), int(C[1])), max(1, int(R)), 1.0, -1, cv2.LINE_AA)
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        L = s.L0(); s.circ(L, C, R, 1.0, 3); s.glowmask(f, L, LEAF * 0.6 + BARK, 0.8, 0.9, 1.0, 8.0)
        n = len(s.ht); step = W * 0.128; by = H * 0.74
        for i, T in enumerate(s.ht):                                                 # 技名: 下から 芽のように のびる。上に ふた葉
            a = (u - 0.55 - i * 0.11) / 0.25
            if a <= 0: continue
            x = W / 2 + (i - (n - 1) / 2) * step; e = back(a, 1.6)
            T2 = tuple(cv2.resize(q, None, fx=1.0, fy=max(0.08, e)) for q in T); s.text(f, T2, x, by + (1 - e) * 30 * k, clamp(a * 2), 1.0)
            if a > 0.6: s.sprout(f, (x, by - 34 * k), clamp((a - 0.6) / 0.5) * 0.45, t)
        for i, T in enumerate(s.vo):
            if u > 0.45 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, by - 96 * k, 1.0, 1.0)
        if u > 1.5: s.text(f, s.txt_en, W / 2, by + 58 * k, ease((u - 1.5) / 0.3), 1.0)
        if u < 0.08: f += (1 - u / 0.08) * 0.7
        return f, 0

    # 3) 早回し: 日が めぐり、春夏秋冬、そして 大樹
    def sc_grow(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_GROW; dur = T_WORLD - T_GROW
        g = ease2(u / (dur - 0.25)); ph = clamp(u / (dur - 0.2)) * 4.0; si = min(4, int(ph)); fr = ph - int(ph) if si < 4 else 0
        a_, b_ = SEASON[si], SEASON[min(4, si + 1)]; q = ease2(clamp((fr - 0.6) / 0.4))
        sky = np.array(a_[0]) * (1 - q) + np.array(b_[0]) * q; leafc = np.array(a_[1]) * (1 - q) + np.array(b_[1]) * q
        cyc = (u * 1.55) % 1.0; hgt = math.sin(cyc * 3.1416); s._sun = SUNC * (0.6 + 0.4 * hgt) + np.array([1.0, 0.45, 0.25], np.float32) * 0.3 * (1 - hgt)
        f = s.land(sky * (0.62 + 0.38 * hgt), 1.0, g)
        c = (W * (-0.1 + 1.2 * cyc), H * (0.34 - 0.24 * hgt)); f += (np.exp(-(np.hypot(s.XX - c[0], s.YY - c[1]) / (15 * k)) ** 4) * 1.3 + np.exp(-np.hypot(s.XX - c[0], s.YY - c[1]) / (70 * k)) * 0.35)[:, :, None] * np.asarray(s._sun, np.float32)[None, None, :]; s.flare(f, c, 12 * k, s._sun, 0.5, 0.0, 0.0, True)
        s.tree(f, g, leafc, 0.25 if si == 3 else 1.0)
        f = s.godrays(f, c, 0.35, 0.75, 8, 0.86, 0.4)
        if si == 3 or (si == 2 and fr > 0.5):                                        # 秋は 葉が 散り、冬は 雪
            r = s.rd[100:340]; s.splat(f, r[:, 0] * W + np.sin(u * 3 + r[:, 3] * 6) * 12 * k, ((r[:, 1] + u * (0.3 + 0.3 * r[:, 2])) % 1.0) * s.gnd, 0.25 + 0.4 * r[:, 4], np.asarray(leafc, np.float32) * 0.8 + 0.2, 1.2, 0.3)
        img = s.pose(t); s.draw(f, img, W * 0.20, s.gnd - H * 0.125, 0.26); s.gem_glow(f, 0.6, t)
        if u < 0.15: f += (1 - u / 0.15) * 0.6
        return f, 0

    def bough(s, f, e):
        """こずえから こちらへ のびてくる 枝(手前ほど 太い・樹皮の すじ)"""
        W, H, k = s.W, s.H, s.k; P0 = (W * 0.60, H * 0.27); P1 = (W * 0.84, H * 0.50); P2 = (W * 0.50, H * 0.54); M = s.L0(); A = s.L0(); N = 150
        for i in range(N + 1):
            a = e * i / N; x = (1 - a) ** 2 * P0[0] + 2 * a * (1 - a) * P1[0] + a * a * P2[0]; y = (1 - a) ** 2 * P0[1] + 2 * a * (1 - a) * P1[1] + a * a * P2[1]; r = max(1, int((5 + 92 * a ** 2.0) * k))
            cv2.circle(M, (int(x), int(y)), r, 1.0, -1, cv2.LINE_AA); cv2.circle(A, (int(x), int(y)), r, float(a + 0.01), -1)
        dt = cv2.distanceTransform((M > 0.5).astype(np.uint8), cv2.DIST_L2, 3); rad = (5 + 92 * np.clip(A, 0, 1) ** 2.0) * k; nd = np.clip(dt / (rad + 1), 0, 1)
        tx = cv2.remap(s._tiles[1], ((A * 110) % 256).astype(np.float32), ((dt * 4.5 / k) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        col = BARK[None, None, :] * ((0.22 + 0.7 * tx) * (0.30 + 1.0 * nd ** 0.6) * (0.6 + 0.8 * A))[:, :, None] + (np.exp(-dt / (2.5 * k)) * 0.14)[:, :, None] * LEAF[None, None, :] + (sstep(0.65, 0.85, s._ltex) * 0.10 * nd)[:, :, None] * LEAF[None, None, :]
        f[:] = f * (1 - M[:, :, None]) + col * M[:, :, None]; return P2

    def budopen(s, f, C, e2, t):
        """目の前で ふくらむ 芽 → 葉が ひらいて 視界を おおう"""
        W, H, k = s.W, s.H, s.k; R = (16 + 100 * ease(clamp(e2 / 0.40))) * k; o = ease2(clamp((e2 - 0.32) / 0.68)); n = 9
        for ring, (sc_, lite, wd) in enumerate(((1.0, 0.25, 1.25), (0.72, 0.6, 1.1), (0.45, 0.95, 1.0))):
            for i in range(n):
                a_open = i * 6.2832 / n + ring * 0.35 + 0.2; a_cl = -1.5708 + (i - (n - 1) / 2) * 0.10 * (1 + ring); dlt = (a_open - a_cl + 3.1416) % 6.2832 - 3.1416
                oo = clamp(o * (1.25 - 0.2 * ring)); ang = a_cl + dlt * oo; S = R * sc_ * (1.1 + 7.5 * oo ** 1.6)
                c = (C[0] + math.cos(ang) * S * (0.10 + 0.72 * oo), C[1] + math.sin(ang) * S * (0.10 + 0.72 * oo)); s.gleaf(f, c, S, ang, 1.0, min(1.0, lite + 0.5 * o), wd * (1.0 - 0.3 * oo))
        d = np.hypot(s.XX - C[0], s.YY - C[1]); f += (np.exp(-d / (R * (0.25 + 2.6 * o))) * (0.25 + 2.4 * o ** 1.5))[:, :, None] * np.array([1.0, 0.97, 0.72], np.float32)[None, None, :]   # 木もれ日

    # 4) 夜。根と のぼる 光 → 枝が こちらへ → 芽が ひらく
    def sc_world(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_WORLD; rc = t - T_REACH; op = t - T_OPEN; s._sun = MOONC * 0.6 + LEAF * 0.4
        f = s.land((0.04, 0.13, 0.16), 1.0, 1.0); f += s.nebula(t, [(0.02, 0.20, 0.12), (0.10, 0.50, 0.30), (0.5, 1.0, 0.6)], 0.0010, (10.0, 3.0), 0.5) * sstep(0.6, 0.2, s.yv)[:, :, None]
        s.starfield(f, t, 500, (0.8, 1.0, 0.85), 3, 0.6 * 1.0, (0.0, 0.0)); lit = ease(u / 0.8)
        Lr = s.L0(); tt = ease(u / 0.7) * s.rmax                                     # 根(光の すじ)
        for (p, q, d, t0, dt) in s.root:
            if t0 > tt: continue
            a = clamp((tt - t0) / dt); cv2.line(Lr, (int(p[0]), int(p[1])), (int(p[0] + (q[0] - p[0]) * a), int(p[1] + (q[1] - p[1]) * a)), 0.9, max(1, int((5 - d * 1.0) * k)), cv2.LINE_AA)
        s.glowmask(f, Lr, LEAF * 0.7 + SUNC * 0.3, 0.5, 0.9, 1.0, 7.0)
        s.tree(f, 1.0, SEASON[4][1], 1.0, lit * 0.7)
        if u > 0.25:                                                                 # 光の つぶが 幹を のぼる
            xs = []; ys = []
            for i in range(0, len(s.tips), 2):
                q, te, path = s.tips[i]; v = ((u - 0.25) * 0.9 + s.rd[i % 400][0]) % 1.0; x = v * (len(path) - 1); j = int(x); fr = x - j
                if j + 1 < len(path): xs.append(path[j][0] + (path[j + 1][0] - path[j][0]) * fr); ys.append(path[j][1] + (path[j + 1][1] - path[j][1]) * fr)
            s.splat(f, np.array(xs), np.array(ys), 0.5, SUNC, 1.2, 0.5)
        r = s.rd[:260]; s.splat(f, r[:, 0] * W + np.sin(t * 1.3 + r[:, 3] * 6) * 14 * k, ((r[:, 1] - t * (0.03 + 0.05 * r[:, 2])) % 1.0) * H * 0.85, (0.15 + 0.3 * r[:, 4]) * lit, LEAF * 0.6 + SUNC * 0.4, 1.3, 0.5)   # 光の 種
        img = s.pose(t); s.draw(f, img, W * 0.20, s.gnd - H * 0.125, 0.26); s.gem_glow(f, 1.0, t)
        sh = 0
        if rc > 0:
            e = ease2(clamp(rc / (T_OPEN - T_REACH + 0.1))); f *= 1 - 0.35 * e; C = s.bough(f, e)
            if op > -0.25: e2 = clamp((op + 0.25) / (T_HIT - T_OPEN + 0.25)); s.budopen(f, C, e2, t); sh = 2 + 6 * e2 ** 2
            if op > 0:
                e3 = clamp(op / (T_HIT - T_OPEN))
                if e3 > 0.8: f = f + ((e3 - 0.8) / 0.2) ** 2 * 0.9
        if u < 0.25: f = f * (u / 0.25) + (1 - u / 0.25) * (LEAF * 0.3 + 0.7)[None, None, :]
        return f, sh

    # ---- この子だけの曲: 笛と ハープ(ト調の 五音)。早回しの間は 1日ごとに 木琴が 1音、季節ごとに 和音が 変わる。夜の 合唱 → 枝の きしみ → 芽が ひらく 鈴 ----
    def audio(s, path):
        a = Snd(DUR, 75); G = 196.0; pen = [0, 2, 4, 7, 9]
        def fq(d, o=0): return G * 2 ** ((pen[d % 5] + 12 * (d // 5 + o)) / 12.0)
        def flute(d, t0, dur, g=0.08, o=1): a.note(fq(d, o), t0, dur, g, "sine", 0.06); a.note(fq(d, o) * 2, t0, dur * 0.6, g * 0.2, "sine", 0.06); a.a += a.lp(a.nz, 0.3) * a.env(t0, 0.03, 0.08) * g * 0.25
        def harp(d, t0, g=0.09, o=0): a.note(fq(d, o), t0, 0.5, g, "bell", 0.002)
        a.a += a.lp(a.nz, 0.02) * 0.45 * a.gate(0, T_PLANT, 0.3)                                                    # 荒れ地の 風
        a.boom(T_PLANT, 55, 0.9, 0.3)
        for i in range(5): harp(i, T_PLANT + 0.1 + i * 0.09, 0.07, 1)
        a.note(fq(4, 2), T_SPROUT, 0.5, 0.14, "bell"); a.note(fq(7, 2), T_SPROUT + 0.08, 0.5, 0.10, "bell")          # ぽん
        a.boom(T_CUT, 60, 0.7, 0.25)
        for i in range(16): a.hat(T_CUT + i * 0.035, 0.03, 0.02)
        for i in range(7): harp([0, 1, 2, 3, 4, 5, 7][i], T_CUT + 0.55 + i * 0.11, 0.10, 1)
        flute(4, T_CUT + 0.6, 1.0, 0.07)
        day = (T_WORLD - T_GROW) / 12.0; tk = T_GROW; i = 0; chords = [(0, 2, 4), (1, 3, 5), (2, 4, 6), (0, 3, 4), (0, 2, 4)]
        mel = [0, 2, 4, 2, 5, 4, 7, 5, 7, 9, 10, 12]
        while tk < T_WORLD - 0.05:                                                   # 木琴: だんだん 高く
            si = min(4, i // 3)
            a.note(fq(mel[i % len(mel)], 1), tk, 0.18, 0.11, "fm", 0.002); a.hat(tk + day / 2, 0.04)
            if i % 2 == 0:
                for d in chords[si]: a.note(fq(d, 0), tk, day * 2, 0.045, "pad", 0.05)
                a.boom(tk, 62, 0.3, 0.12)
            if i % 4 == 0: flute(mel[(i // 2) % len(mel)] + 2, tk, day * 3.5, 0.06)
            tk += day; i += 1
        a.rise(T_GROW, T_WORLD, 110, 330, 0.06)
        a.boom(T_WORLD, 42, 1.2, 0.8)
        for j, (d, o) in enumerate(((0, -1), (2, -1), (0, 0), (2, 0), (4, 0), (5, 0), (7, 0))): a.note(fq(d, o), T_WORLD + j * 0.05, T_HIT - T_WORLD, 0.05, "pad", 0.4)   # 合唱
        for i in range(8): harp(i % 10, T_WORLD + 0.25 + i * 0.07, 0.06, 1)                                           # のぼる 光
        a.rise(T_REACH, T_OPEN, 70, 140, 0.16); a.a += a.lp((a.nz - a.lp(a.nz, 0.2)), 0.15) * a.gate(T_REACH, T_OPEN, 0.1) * 0.5 * (0.5 + 0.5 * np.sin(a.t * 60))      # 枝が のびる きしみ
        flute(7, T_REACH + 0.05, 0.6, 0.08)
        for i in range(12): a.note(fq(5 + i, 2), T_OPEN + i * 0.05, 0.4, 0.05, "bell")                                # 芽が ひらく
        a.whoosh(T_OPEN, T_HIT, 0.7, 0.3); flute(9, T_OPEN + 0.1, 0.6, 0.09); a.boom(T_HIT, 46, 1.3, 0.6)
        for d in (0, 2, 4, 7): harp(d, T_HIT, 0.08, 1)
        s.fin_snd(a, DUR, 75); a.save(path)


if __name__ == "__main__":
    r = Yggd()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
