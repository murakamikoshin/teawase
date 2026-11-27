"""ユグド(★6 世界樹の子) 専用の必殺技「世界樹ノ芽吹キ」
灰色の荒れ地。杖を つくと 地のひびに 緑が走り、ふた葉が ぽんと出る → 年輪が ひろがる 丸窓のカットイン →
早回し: 日と月が 何度も空をわたり、春夏秋冬が めぐるあいだに、ふた葉が 枝分かれして 大樹になる →
根が 地の下へ ひろがり、光の粒が 根から 幹をのぼって 梢へ → 梢から 光の種が いっせいに 舞い上がる
使い方: python3 hs_75.py 出力.mp4 / python3 hs_75.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 13.8
LEAF = np.array([0.42, 0.95, 0.40], np.float32); BARK = np.array([0.50, 0.33, 0.17], np.float32); SUNC = np.array([1.0, 0.86, 0.45], np.float32); MOONC = np.array([0.70, 0.82, 1.0], np.float32)
T_PLANT, T_SPROUT, T_CUT, T_GROW, T_WORLD, T_SEED = 1.3, 1.95, 2.9, 5.0, 10.0, 11.9
SEASON = [((0.55, 0.80, 0.95), (0.55, 1.00, 0.45)), ((0.25, 0.55, 0.85), (0.16, 0.70, 0.22)), ((0.95, 0.62, 0.35), (1.00, 0.50, 0.15)), ((0.75, 0.82, 0.92), (0.95, 0.97, 1.0)), ((0.50, 0.88, 0.80), (0.60, 1.00, 0.50))]


class Yggd(Kit):
    def __init__(s, W=540):
        super().__init__(75, (0.42, 0.95, 0.40), "世界樹ノ芽吹キ", "Yggdrasil Bloom", W)
        s.gem = s.A(110, 187); s.eyeL = s.A(306, 282); s.eyeR = s.A(416, 272); s.foot = s.A(330, 740)
        s.ht = [s.text_img(c, hz.SERIF, int(58 * s.k), 3) for c in "世界樹ノ芽吹キ"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        rg = np.random.RandomState(75); s.rd = rg.rand(400, 5).astype(np.float32)
        s.gnd = s.H * 0.86; s.base = (s.W * 0.58, s.gnd)
        # 木: 枝を 1本ずつ「いつ のびるか」つきで 作っておく
        s.seg = []; s.tips = []
        def grow(p, ang, ln, d, t0, path):
            q = (p[0] + math.cos(ang) * ln, p[1] + math.sin(ang) * ln); dt = ln / (s.H * 1.9)
            s.seg.append((p, q, d, t0, dt)); path = path + [q]
            if d >= 7 or ln < 9 * s.k: s.tips.append((q, t0 + dt, path)); return
            n = 2 if rg.rand() < 0.8 else 3
            for j in range(n):
                sp = (j - (n - 1) / 2) * (0.50 if d > 0 else 0.42) + (rg.rand() - 0.5) * 0.55
                grow(q, ang + sp - 0.04 * math.cos(ang) * 3, ln * (0.66 + 0.16 * rg.rand()), d + 1, t0 + dt, path)
            if d >= 3: s.tips.append((q, t0 + dt, path))
        grow(s.base, -1.5708, s.H * 0.18, 0, 0.0, [s.base])
        s.tmax = max(t for _, t, _ in s.tips)
        s.root = []
        def rootg(p, ang, ln, d, t0):
            q = (p[0] + math.cos(ang) * ln, p[1] + math.sin(ang) * ln * 0.55); dt = ln / (s.W * 1.2); s.root.append((p, q, d, t0, dt))
            if d >= 4: return
            for j in range(2): rootg(q, ang + (j - 0.5) * 0.9 + (rg.rand() - 0.5) * 0.3, ln * 0.72, d + 1, t0 + dt)
        for a0 in (0.35, 1.2, 1.94, 2.79): rootg(s.base, a0, s.W * 0.20, 0, 0.0)
        s.rmax = max(t0 + dt for _, _, _, t0, dt in s.root)

    # ---- この子だけの絵 ----
    def land(s, sky, lum=1.0, green=0.0):
        y = s.YY / s.H
        f = (np.asarray(sky, np.float32)[None, None, :] * (0.25 + 0.75 * y)[:, :, None]) * lum
        g = int(s.gnd); f[g:, :, :] = (np.array([0.13, 0.11, 0.10], np.float32) * (1 - green) + np.array([0.08, 0.22, 0.10], np.float32) * green) * lum
        return f.astype(np.float32)

    def tree(s, f, g, leafc, leaves=1.0, lit=0.0):
        """g: 0..1 どこまで 育ったか"""
        tt = g * s.tmax; D = s.L0(); k = s.k
        for (p, q, d, t0, dt) in s.seg:
            if t0 > tt: continue
            a = clamp((tt - t0) / dt); e = (p[0] + (q[0] - p[0]) * a, p[1] + (q[1] - p[1]) * a)
            cv2.line(D, (int(p[0]), int(p[1])), (int(e[0]), int(e[1])), 1.0, max(1, int((24 - d * 3.3) * k * (0.35 + 0.65 * g))), cv2.LINE_AA)
        f[:] = f * (1 - D * 0.96)[:, :, None] + D[:, :, None] * (BARK * (0.55 + 0.6 * lit))[None, None, :]
        if leaves > 0:
            L = s.L0()
            for i, (q, te, _) in enumerate(s.tips):
                if te > tt: continue
                a = back((tt - te) / (s.tmax * 0.08), 1.5); r = s.rd[i % 400]
                cv2.circle(L, (int(q[0] + (r[0] - 0.5) * 10 * k), int(q[1] + (r[1] - 0.5) * 10 * k)), max(1, int((6 + 7 * r[2]) * k * a * leaves)), float(0.55 + 0.45 * r[3]), -1, cv2.LINE_AA)
            al = np.clip(L, 0, 1)
            f[:] = f * (1 - al * 0.9)[:, :, None] + al[:, :, None] * np.asarray(leafc, np.float32)[None, None, :] * (0.75 + 0.5 * lit)
            if lit > 0: f += cv2.GaussianBlur(L, (0, 0), 14 * k)[:, :, None] * LEAF[None, None, :] * 0.5 * lit

    def sprout(s, f, p, a, t):
        """ふた葉"""
        if a <= 0: return
        k = s.k; h = 30 * k * a; L = s.L0()
        cv2.line(L, (int(p[0]), int(p[1])), (int(p[0]), int(p[1] - h)), 1.0, max(1, int(3 * k)), cv2.LINE_AA)
        for sg in (-1, 1):
            cv2.ellipse(L, (int(p[0] + sg * 12 * k * a), int(p[1] - h - 2 * k)), (max(1, int(13 * k * a)), max(1, int(6 * k * a))), sg * 25, 0, 360, 1.0, -1, cv2.LINE_AA)
        f[:] = f * (1 - np.clip(L, 0, 1))[:, :, None] + np.clip(L, 0, 1)[:, :, None] * LEAF[None, None, :]
        f += cv2.GaussianBlur(L, (0, 0), 9 * k)[:, :, None] * LEAF[None, None, :] * 0.7

    def gem_glow(s, f, v, t):
        if v <= 0: return
        p = s.at(*s.gem); sc = s._pl[2]; L = s.L0()
        s.star4(L, p, (16 + 5 * math.sin(t * 6)) * sc * v * 1.6, v, 0.0, 0.25); cv2.circle(L, (int(p[0]), int(p[1])), max(1, int(9 * sc)), float(v), -1, cv2.LINE_AA)
        s.add(f, L, LEAF, 1.1, 2, 9)

    def frame(s, t):
        k = s.k
        if t < T_CUT: f, sh = s.sc_plant(t)
        elif t < T_GROW: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_WORLD: f, sh = s.sc_grow(t)
        else: f, sh = s.sc_world(t)
        f, sh = s.finale(f, sh, t, DUR, s.gem, (0.45, 1, 0.5), (1, 0.9, 0.5), 2)
        return s.post(f, LEAF, 0.5, sh * k, t, 0.90)

    # 1) 荒れ地に 杖をつく → ひびに緑 → ふた葉
    def sc_plant(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_PLANT
        f = s.land((0.34, 0.33, 0.36), clamp(t / 0.5))
        L = s.L0()
        for i in range(40):                                                          # 風に 流れる 灰
            r = s.rd[i]; x = ((r[0] + t * (0.12 + 0.2 * r[2])) % 1.2 - 0.1) * W; y = r[1] * H * 0.9
            s.line(L, (x, y), (x + 18 * k, y + 3 * k), 0.25, 1)
        f += L[:, :, None] * 0.5
        cy = s.gnd - H * 0.215
        img = s.pose(t, squash=(0.07 * math.exp(-u * 9) * math.cos(u * 22) if u > 0 else -0.03 * clamp((t - 0.7) / 0.5)))
        s.draw(f, img, W * 0.42, cy, 0.44)
        if u > 0:
            L = s.L0(); c = (s.base[0], s.gnd + 8 * k); g = ease(u / 0.7)
            for i in range(9):                                                       # 地のひびに 緑が 走る
                a = -0.2 + i * 0.44; pts = [c]; rr = 0
                for j in range(8):
                    rr += W * 0.07 * g; r = s.rd[i * 8 + j]; pts.append((c[0] + math.cos(a + (r[0] - 0.5) * 0.5) * rr, c[1] + abs(math.sin(a + (r[1] - 0.5) * 0.5)) * rr * 0.28))
                cv2.polylines(L, [np.array(pts, np.int32)], False, 0.9, 2, cv2.LINE_AA)
            for i in range(2):
                v = (u - i * 0.12) / 0.7
                if 0 < v < 1: s.ell(L, c, (W * 0.8 * v, W * 0.12 * v), 0, 1 - v, 2)
            s.add(f, L, LEAF, 0.9); s.gem_glow(f, clamp(u / 0.3), t)
        sp = t - T_SPROUT
        if sp > 0:
            s.sprout(f, s.base, back(sp / 0.35, 2.2), t)
            L = s.L0()
            for i in range(10):                                                      # ぽん、と はじける光
                a = -3.14 * i / 9.0; rr = (10 + 70 * clamp(sp / 0.4)) * k; s.star4(L, (s.base[0] + math.cos(a) * rr, s.base[1] - 30 * k + math.sin(a) * rr), 6 * k * clamp(1 - sp / 0.6), 1.0, sp * 4)
            s.add(f, L, LEAF * 0.6 + 0.4, 1.0)
        return f, (9 * math.exp(-u * 8) if u > 0 else 0)

    # 2) 年輪の丸窓
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = np.zeros((H, W, 3), np.float32) + (BARK * 0.16)[None, None, :]
        C = (W / 2, H * 0.36); L = s.L0()
        for i in range(16):                                                          # 年輪が 1本ずつ ひろがる
            a = (u - i * 0.035) / 0.3
            if a <= 0: continue
            rr = W * (0.36 + i * 0.05) * ease(a); pts = [(C[0] + math.cos(j * 0.1745) * rr * (1 + 0.035 * math.sin(j * 0.9 + i * 1.7)), C[1] + math.sin(j * 0.1745) * rr * (1 + 0.035 * math.cos(j * 0.7 + i))) for j in range(36)]
            cv2.polylines(L, [np.array(pts, np.int32)], True, 0.5 if i % 3 else 0.9, 2 if i % 3 == 0 else 1, cv2.LINE_AA)
        f += (L + cv2.GaussianBlur(L, (0, 0), 3 * k) * 0.6)[:, :, None] * (BARK * 1.5 + LEAF * 0.15)[None, None, :]
        R = W * 0.34 * back(u / 0.35, 1.2)
        win = np.zeros((H, W, 3), np.float32) + (LEAF * 0.22 + BARK * 0.25)[None, None, :]
        ex = (s.eyeL[0] + s.eyeR[0]) / 2; ey = (s.eyeL[1] + s.eyeR[1]) / 2; zf = 1.35; sc = zf * H / s.ah
        img = s.pose(t); s.draw(win, img, C[0] - (ex - s.aw / 2) * sc, C[1] - (ey - s.ah / 2) * sc, zf)
        Le = s.L0()
        for e in (s.eyeL, s.eyeR):
            p = s.at(*e); cv2.ellipse(Le, (int(p[0]), int(p[1])), (max(1, int(15 * sc)), max(1, int(22 * sc))), 0, 0, 360, 0.40 + 0.15 * math.sin(u * 9), -1, cv2.LINE_AA)
        s.add(win, Le, LEAF, 0.9, 2, 8)
        m = s.L0(); cv2.circle(m, (int(C[0]), int(C[1])), max(1, int(R)), 1.0, -1, cv2.LINE_AA)
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        L = s.L0(); s.circ(L, C, R, 1.0, 4); s.add(f, L, LEAF * 0.5 + BARK, 0.9)
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

    # 3) 早回し: 日と月、春夏秋冬、そして 大樹
    def sc_grow(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_GROW; dur = T_WORLD - T_GROW
        g = ease2(u / (dur - 0.5)); ph = clamp(u / (dur - 0.6)) * 4.0; si = min(4, int(ph)); fr = ph - int(ph) if si < 4 else 0
        a_, b_ = SEASON[si], SEASON[min(4, si + 1)]; q = ease2(clamp((fr - 0.7) / 0.3))
        sky = np.array(a_[0]) * (1 - q) + np.array(b_[0]) * q; leafc = np.array(a_[1]) * (1 - q) + np.array(b_[1]) * q
        cyc = (u * 2.2) % 1.0; night = int(u * 2.2) % 2                              # 日 と 月が かわるがわる わたる
        f = s.land(sky * (0.55 if night else 1.0), 1.0, g)
        c = (W * (-0.1 + 1.2 * cyc), H * (0.30 - 0.22 * math.sin(cyc * 3.1416)))
        s.soft(f, c, 22 * k, 0.9, MOONC if night else SUNC); L = s.L0(); cv2.circle(L, (int(c[0]), int(c[1])), int(13 * k), 1.0, -1, cv2.LINE_AA); f += L[:, :, None] * (MOONC if night else SUNC)[None, None, :]
        L = s.L0()
        for j in range(8):                                                           # 通ったあとの 光の弧
            cq = cyc - j * 0.03
            if cq > 0: cv2.circle(L, (int(W * (-0.1 + 1.2 * cq)), int(H * (0.30 - 0.22 * math.sin(cq * 3.1416)))), int(4 * k), 0.4 * (1 - j / 8), -1)
        f += L[:, :, None] * (MOONC if night else SUNC)[None, None, :] * 0.6
        s.tree(f, g, leafc, 0.25 if si == 3 else 1.0)
        if si == 3 or (si == 2 and fr > 0.6):                                        # 秋は葉が散り、冬は雪
            L = s.L0()
            for i in range(60):
                r = s.rd[i + 100]; y = ((r[1] + u * (0.3 + 0.3 * r[2])) % 1.0) * s.gnd; x = r[0] * W + math.sin(u * 3 + r[3] * 6) * 12 * k
                cv2.circle(L, (int(x), int(y)), max(1, int(2.5 * k)), 0.8, -1)
            f += L[:, :, None] * np.asarray(leafc, np.float32)[None, None, :]
        img = s.pose(t); s.draw(f, img, W * 0.20, s.gnd - H * 0.125, 0.26); s.gem_glow(f, 0.6, t)
        if u < 0.15: f += (1 - u / 0.15) * 0.6
        return f, 0

    # 4) 世界樹: 根・のぼる光・光の種
    def sc_world(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_WORLD; sd = t - T_SEED
        y = s.YY / H
        f = (np.array([0.03, 0.10, 0.12], np.float32)[None, None, :] * (0.5 + 1.2 * y)[:, :, None]).astype(np.float32)
        L = s.L0()
        for i in range(5):                                                           # うすい オーロラ
            xs = np.arange(0, W + 20, 20); pts = [(x, H * (0.10 + 0.05 * i) + math.sin(x * 0.012 + t * 0.8 + i) * H * 0.03) for x in xs]
            cv2.polylines(L, [np.array(pts, np.int32)], False, 0.25, int(10 * k), cv2.LINE_AA)
        f += cv2.GaussianBlur(L, (0, 0), 16 * k)[:, :, None] * LEAF[None, None, :] * 0.7
        g = int(s.gnd); f[g:, :, :] = np.array([0.03, 0.07, 0.05], np.float32)
        lit = ease(u / 1.4)
        Lr = s.L0(); tt = ease(u / 1.1) * s.rmax                                     # 根
        for (p, q, d, t0, dt) in s.root:
            if t0 > tt: continue
            a = clamp((tt - t0) / dt); cv2.line(Lr, (int(p[0]), int(p[1])), (int(p[0] + (q[0] - p[0]) * a), int(p[1] + (q[1] - p[1]) * a)), 0.9, max(1, int((7 - d * 1.4) * k)), cv2.LINE_AA)
        s.add(f, Lr, LEAF * 0.7 + SUNC * 0.3, 0.6, 2, 7)
        s.tree(f, 1.0, SEASON[4][1], 1.0, lit)
        Lp = s.L0()
        for i in range(0, len(s.tips), 2):                                           # 光の粒が 幹を のぼる
            q, te, path = s.tips[i]; v = ((u - 0.5) * 0.55 + s.rd[i % 400][0]) % 1.0
            if u < 0.5: continue
            x = v * (len(path) - 1); j = int(x); fr = x - j
            if j + 1 < len(path): cv2.circle(Lp, (int(path[j][0] + (path[j + 1][0] - path[j][0]) * fr), int(path[j][1] + (path[j + 1][1] - path[j][1]) * fr)), max(1, int(3 * k)), 1.0, -1, cv2.LINE_AA)
        s.add(f, Lp, SUNC, 1.2, 2, 7)
        img = s.pose(t, squash=(-0.03 * math.exp(-sd * 6) if sd > 0 else 0)); s.draw(f, img, W * 0.20, s.gnd - H * 0.125, 0.26); s.gem_glow(f, 1.0, t)
        if sd > 0:                                                                   # 梢から 光の種
            L = s.L0(); n = len(s.tips)
            for i in range(240):
                q = s.tips[(i * 7) % n][0]; r = s.rd[i]; v = sd * (0.25 + 0.45 * r[2]) - r[3] * 0.25
                if v <= 0: continue
                x = q[0] + math.sin(sd * 2 + r[0] * 6.28) * 22 * k * v + (r[1] - 0.5) * W * 0.5 * v; y2 = q[1] - v * H * 0.75
                if -10 < y2 < H: s.star4(L, (x, y2), (3 + 5 * r[4]) * k, 0.9, r[0] * 3, 0.3)
            s.add(f, L, LEAF * 0.5 + SUNC * 0.5, 1.2, 2, 8)
            for i in range(3):
                v = (sd - i * 0.14) / 1.0
                if 0 < v < 1:
                    L2 = s.L0(); s.circ(L2, (s.base[0], H * 0.34), W * 1.2 * v, 1 - v, 3); s.add(f, L2, LEAF, 0.9)
            if sd < 0.12: f += (1 - sd / 0.12) * 0.9
        if u < 0.25: f = f * (u / 0.25) + (1 - u / 0.25) * (LEAF * 0.3 + 0.7)[None, None, :]
        end = DUR - t
        if end < 0.8: v = clamp((0.8 - end) / 0.4); f = f * (1 - v) + v * (LEAF * 0.35 + 0.65)[None, None, :]
        if end < 0.4: f = f * clamp(end / 0.35) ** 2
        return f, (6 * math.exp(-sd * 5) if sd > 0 else 0)

    # ---- この子だけの曲: 笛と ハープ(ト調の 五音)。早回しの間は 1日ごとに 木琴が1音、季節ごとに 和音が変わる。さいごは 合唱 ----
    def audio(s, path):
        a = Snd(DUR, 75); G = 196.0; pen = [0, 2, 4, 7, 9]
        def fq(d, o=0): return G * 2 ** ((pen[d % 5] + 12 * (d // 5 + o)) / 12.0)
        def flute(d, t0, dur, g=0.08, o=1): a.note(fq(d, o), t0, dur, g, "sine", 0.06); a.note(fq(d, o) * 2, t0, dur * 0.6, g * 0.2, "sine", 0.06); a.a += a.lp(a.nz, 0.3) * a.env(t0, 0.03, 0.08) * g * 0.25
        def harp(d, t0, g=0.09, o=0): a.note(fq(d, o), t0, 0.5, g, "bell", 0.002)
        a.a += a.lp(a.nz, 0.02) * 0.45 * a.gate(0, T_PLANT, 0.3)                                                    # 荒れ地の風
        a.boom(T_PLANT, 55, 0.9, 0.3)
        for i in range(5): harp(i, T_PLANT + 0.1 + i * 0.09, 0.07, 1)
        a.note(fq(4, 2), T_SPROUT, 0.5, 0.14, "bell"); a.note(fq(7, 2), T_SPROUT + 0.08, 0.5, 0.10, "bell")          # ぽん
        a.boom(T_CUT, 60, 0.7, 0.25)
        for i in range(16): a.hat(T_CUT + i * 0.035, 0.03, 0.02)
        for i in range(7): harp([0, 1, 2, 3, 4, 5, 7][i], T_CUT + 0.55 + i * 0.11, 0.10, 1)
        flute(4, T_CUT + 0.6, 1.0, 0.07)
        day = 1 / 2.2; tk = T_GROW; i = 0; chords = [(0, 2, 4), (1, 3, 5), (2, 4, 6), (0, 3, 4), (0, 2, 4)]
        mel = [0, 2, 4, 2, 5, 4, 2, 4, 7, 5, 4, 5, 7, 9, 7, 9, 10, 9, 7, 9, 12, 10]
        while tk < T_WORLD - 0.05:                                                   # 1日 = 木琴1音。だんだん 高く
            u = tk - T_GROW; si = min(4, int(u / ((T_WORLD - T_GROW - 0.6) / 4.0)))
            a.note(fq(mel[i % len(mel)], 1), tk, 0.18, 0.11, "fm", 0.002); a.hat(tk + day / 2, 0.04)
            if i % 2 == 0:
                for d in chords[si]: a.note(fq(d, 0), tk, day * 2, 0.045, "pad", 0.05)
                a.boom(tk, 62, 0.3, 0.12)
            if i % 4 == 0: flute(mel[(i // 2) % len(mel)] + 2, tk, day * 3.5, 0.06)
            tk += day; i += 1
        a.rise(T_GROW, T_WORLD, 110, 330, 0.06)
        a.boom(T_WORLD, 42, 1.2, 0.8)
        for j, (d, o) in enumerate(((0, -1), (2, -1), (0, 0), (2, 0), (4, 0), (5, 0), (7, 0))): a.note(fq(d, o), T_WORLD + j * 0.05, 3.2, 0.05, "pad", 0.4)   # 合唱
        for i in range(14): harp(i % 10, T_WORLD + 0.5 + i * 0.1, 0.06, 1)                                            # のぼる光
        a.boom(T_SEED, 50, 1.0, 0.6)
        for i in range(18): a.note(fq(5 + (i * 3) % 10, 2), T_SEED + i * 0.07, 0.4, 0.045, "bell")                     # 種が 舞う
        flute(7, T_SEED + 0.1, 1.5, 0.09); flute(9, T_SEED + 0.9, 0.9, 0.07)
        s.fin_snd(a, DUR, 75); a.save(path)


if __name__ == "__main__":
    r = Yggd()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
