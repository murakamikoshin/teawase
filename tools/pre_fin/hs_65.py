"""ヒミカ(★5 日の巫女王) 専用の必殺技「日輪ノ裁キ」
まっ暗な 古い社(赤い柱・しめなわ)。天井の すき間から 細い日ざしが 1本、床に 落ちている → 鏡を 向けると 日ざしが 鏡に 乗り、
宙に 現れる 銅鏡へ、キィン、キィンと 6回 はね返って 光の あみを 張る(1回ごとに 社が 明るくなる) → 銅鏡の形の 窓のカットイン →
7枚の鏡が いっせいに 1点を 向き、光の玉が ふくらむ → 光の柱が 落ちる → 床に 日の紋が やきつき、光のつぶが のぼる
使い方: python3 hs_65.py 出力.mp4 / python3 hs_65.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 11.5
GOLD = np.array([1.0, 0.86, 0.46], np.float32); SUN = np.array([1.0, 0.95, 0.76], np.float32); VERM = np.array([0.72, 0.16, 0.08], np.float32)
T_CATCH, T_R0, RSTEP, T_CUT, T_AIM, T_ORB, T_FALL, T_END = 1.8, 2.15, 0.2, 3.6, 5.5, 6.1, 7.5, 8.5
MIR = ((0.13, 0.50), (0.86, 0.27), (0.30, 0.09), (0.93, 0.50), (0.10, 0.24), (0.62, 0.07))
TGT = (0.27, 0.20)


class Himika(Kit):
    def __init__(s, W=540):
        super().__init__(65, (1.0, 0.86, 0.46), "日輪ノ裁キ", "Solar Judgment", W)
        H, k = s.H, s.k
        s.eL = s.A(385, 265); s.eR = s.A(465, 270); s.fc = s.A(420, 255); s.mir = s.A(345, 440); s.crn = s.A(380, 70)
        s.ht = [s.text_img(c, hz.SERIF, int(78 * k), 3) for c in "日輪ノ裁キ"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * k), 1) for c in "奥義"]
        s.HF = 0.44; s.flr = H * 0.87; s.CX = W * 0.64; s.CY = s.flr - s.HF * H / 2; s.SRC = (W * 0.30, -10.0); s.FLO = (W * 0.27, s.flr + 14 * k)
        s._pose = dict(t=0, sway=0, breath=0, lean=0, squash=0, stretch=None); s._pl = (s.CX, s.CY, s.HF * H / s.ah); s.M0 = s.at(*s.mir)
        s.mp = [(x * W, y * H) for x, y in MIR]; s.T = (TGT[0] * W, TGT[1] * H)
        rg = np.random.RandomState(65); s.mo = rg.rand(50, 4).astype(np.float32); s.yn = s.YY / H
        # 社: 柱・しめなわ・床板
        col = s.L0()
        for x, w in ((0.03, 34), (0.21, 22), (0.80, 22), (0.97, 34)): cv2.rectangle(col, (int(x * W - w * k / 2), 0), (int(x * W + w * k / 2), int(s.flr)), 1.0, -1)
        cv2.rectangle(col, (0, int(H * 0.035)), (W, int(H * 0.035 + 20 * k)), 1.0, -1)
        s.pil = cv2.GaussianBlur(col, (0, 0), 0.8 * k)
        rope = s.L0(); sh = s.L0(); pts = [(W * i / 30.0, H * (0.105 + 0.03 * math.sin(i / 30.0 * 3.1416))) for i in range(31)]
        cv2.polylines(rope, [np.array(pts, np.int32)], False, 1.0, int(15 * k), cv2.LINE_AA)
        for i in (5, 12, 18, 25):
            x, y = pts[i]; y += 8 * k; z = [(x, y), (x + 12 * k, y + 14 * k), (x - 2 * k, y + 14 * k), (x + 10 * k, y + 30 * k), (x - 4 * k, y + 30 * k), (x + 8 * k, y + 48 * k), (x - 12 * k, y + 28 * k), (x, y + 28 * k), (x - 10 * k, y + 13 * k), (x, y + 13 * k)]
            cv2.fillPoly(sh, [np.array(z, np.int32)], 1.0, cv2.LINE_AA)
        s.rope = rope; s.shide = sh

    # ---- 大道具 ----
    def shrine(s, t, lit=0.0):
        W, H, k = s.W, s.H, s.k; a = 0.10 + 0.90 * lit
        f = (np.array([0.05, 0.035, 0.03], np.float32)[None, None, :] * (0.4 + 0.6 * s.yn)[:, :, None] * (0.6 + 2.2 * lit)).astype(np.float32)
        f = f * (1 - s.pil[:, :, None]) + (VERM * a)[None, None, :] * s.pil[:, :, None]
        f = f * (1 - s.rope[:, :, None]) + (np.array([0.78, 0.64, 0.34], np.float32) * a)[None, None, :] * s.rope[:, :, None]
        f = f * (1 - s.shide[:, :, None]) + (np.array([0.96, 0.94, 0.88], np.float32) * (0.2 + 0.8 * lit))[None, None, :] * s.shide[:, :, None]
        g = int(s.flr); f[g:] = np.array([0.30, 0.19, 0.10], np.float32) * (0.25 + 0.9 * lit)
        for i in range(1, 6): cv2.line(f, (0, int(g + (H - g) * (i / 6.0) ** 1.5)), (W, int(g + (H - g) * (i / 6.0) ** 1.5)), (0.04, 0.025, 0.015), max(1, int(1.5 * k)))
        return f

    def beam(s, L, a, b, v=1.0, th=4):
        s.line(L, a, b, v, max(1, int(th * s.k)))

    def bronze(s, f, c, r, lit=0.0, sq=1.0):
        """銅鏡(光を うける 面)"""
        x, y = int(c[0]), int(c[1]); rx = max(2, int(r * sq)); ry = int(r)
        cv2.ellipse(f, (x, y), (rx, ry), 0, 0, 360, (0.20, 0.13, 0.05), -1, cv2.LINE_AA)
        fc = np.array([0.62, 0.44, 0.18], np.float32) * (1 - lit) + SUN * lit; cv2.ellipse(f, (x, y), (int(rx * 0.86), int(ry * 0.86)), 0, 0, 360, (float(fc[0]), float(fc[1]), float(fc[2])), -1, cv2.LINE_AA)
        cv2.ellipse(f, (x, y), (int(rx * 0.60), int(ry * 0.60)), 0, 0, 360, (0.36, 0.24, 0.09), max(1, int(r * 0.06)), cv2.LINE_AA)
        for j in range(8): a = j * 0.7854; cv2.circle(f, (int(x + math.cos(a) * rx * 0.73), int(y + math.sin(a) * ry * 0.73)), max(1, int(r * 0.06)), (0.36, 0.24, 0.09), -1, cv2.LINE_AA)
        cv2.ellipse(f, (x, y), (int(rx * 0.20), int(ry * 0.20)), 0, 0, 360, (0.40, 0.27, 0.10), -1, cv2.LINE_AA)
        if lit > 0.05: s.soft(f, c, r * 1.3, 0.55 * lit, GOLD)

    def motes(s, f, t, v=1.0, rise=0.0):
        L = s.L0()
        for r in s.mo:
            x = (r[0] + 0.02 * math.sin(t * 0.8 + r[2] * 9)) * s.W; y = ((r[1] - rise * t * 0.06 * (0.5 + r[3])) % 1.0) * s.H
            cv2.circle(L, (int(x), int(y)), max(1, int((1.2 + 1.8 * r[3]) * s.k)), float(v * (0.4 + 0.6 * r[2])), -1, cv2.LINE_AA)
        f += (L + cv2.GaussianBlur(L, (0, 0), 3 * s.k) * 1.5)[:, :, None] * GOLD[None, None, :]

    def hero(s, f, t, light=1.0, **kw):
        img = s.pose(t, **kw)
        if light < 1: img = img.copy(); img[:, :, :3] *= light
        s.draw(f, img, s.CX, s.CY, s.HF)

    def nref(s, t): return sum(1 for i in range(6) if t >= T_R0 + i * RSTEP)

    def web(s, f, t, n, v=1.0):
        """日ざし → 手の鏡 → 銅鏡 n枚 の 光の あみ"""
        L = s.L0(); pts = [s.M0] + s.mp[:n]; s.beam(L, s.SRC, s.M0, v, 5)
        for i in range(len(pts) - 1):
            g = ease((t - (T_R0 + i * RSTEP)) / 0.07); a, b = pts[i], pts[i + 1]; s.beam(L, a, (a[0] + (b[0] - a[0]) * g, a[1] + (b[1] - a[1]) * g), v, 4)
        s.add(f, L, GOLD, 0.55 * v, 2, 9)

    def frame(s, t):
        if t < T_CUT: f, sh = s.sc_relay(t)
        elif t < T_AIM: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_END: f, sh = s.sc_judge(t)
        else: f, sh = s.sc_end(t - T_END)
        return s.post(f, GOLD, 0.30, sh * s.k, t, 0.97)

    # 1) 日ざしを 鏡で うけて、はね返していく
    def sc_relay(s, t):
        W, H, k = s.W, s.H, s.k; c = t - T_CATCH; n = s.nref(t); lit = 0.05 + 0.11 * n + (0.08 * clamp(c / 0.3) if c > 0 else 0)
        f = s.shrine(t, lit)
        for i in range(n):
            v = t - (T_R0 + i * RSTEP); s.bronze(f, s.mp[i], 48 * k * back((v + 0.05) / 0.15, 2.0), 0.35 + 0.65 * math.exp(-v * 5), 1.0)
        s.hero(f, t, 0.35 + 0.65 * clamp(lit * 1.6 + (0.4 if c > 0 else 0)), squash=(-0.05 * math.exp(-c * 10) if c > 0 else 0))
        if c <= 0.25:                                                                # 床に 落ちる 日ざし → 鏡へ
            g = ease2(c / 0.25) if c > 0 else 0.0; e = (s.FLO[0] + (s.M0[0] - s.FLO[0]) * g, s.FLO[1] + (s.M0[1] - s.FLO[1]) * g)
            L = s.L0(); s.beam(L, s.SRC, e, 0.9, 5); s.add(f, L, GOLD, 0.5, 2, 9)
            if c <= 0: cv2.ellipse(f, (int(s.FLO[0]), int(s.FLO[1])), (int(26 * k), int(7 * k)), 0, 0, 360, (1.0, 0.92, 0.66), -1, cv2.LINE_AA)
        else: s.web(f, t, n)
        if c > 0: s.soft(f, s.M0, 40 * k, 0.5 + 0.5 * math.exp(-c * 6), SUN)
        for i in range(n):
            v = t - (T_R0 + i * RSTEP + 0.07)
            if 0 < v < 0.2: L = s.L0(); s.star4(L, s.mp[i], 70 * k * (1 - v / 0.2), 1.0, 0.4, 0.10); s.add(f, L, SUN, 0.9, 2, 6)
        s.motes(f, t, 0.25 + 0.5 * lit)
        return f * clamp(t / 0.4), 0

    # 2) 銅鏡の形の 窓
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = np.zeros((H, W, 3), np.float32) + np.array([0.10, 0.03, 0.02], np.float32)
        C = (W / 2, H * 0.38); g = back(u / 0.3, 1.3); R = W * 0.44 * g; ri = R * 0.74
        L = s.L0()
        for j in range(16): a = j * 0.3927 + t * 0.15; cv2.fillPoly(L, [np.array([C, (C[0] + math.cos(a - 0.05) * H, C[1] + math.sin(a - 0.05) * H), (C[0] + math.cos(a + 0.05) * H, C[1] + math.sin(a + 0.05) * H)], np.int32)], 0.25, cv2.LINE_AA)
        f += L[:, :, None] * VERM[None, None, :]
        cv2.circle(f, (int(C[0]), int(C[1])), int(R), (0.20, 0.13, 0.05), -1, cv2.LINE_AA); cv2.circle(f, (int(C[0]), int(C[1])), int(R * 0.97), (0.62, 0.44, 0.18), -1, cv2.LINE_AA)
        for j in range(28):                                                          # のこぎり歯の もよう
            a = j * 6.2832 / 28 + t * 0.2; P = [(C[0] + math.cos(a - 0.09) * R * 0.94, C[1] + math.sin(a - 0.09) * R * 0.94), (C[0] + math.cos(a + 0.09) * R * 0.94, C[1] + math.sin(a + 0.09) * R * 0.94), (C[0] + math.cos(a) * ri * 1.06, C[1] + math.sin(a) * ri * 1.06)]
            cv2.fillPoly(f, [np.array(P, np.int32)], (0.36, 0.24, 0.09), cv2.LINE_AA)
        m = s.L0(); cv2.circle(m, (int(C[0]), int(C[1])), int(ri), 1.0, -1, cv2.LINE_AA)
        win = np.zeros((H, W, 3), np.float32) + np.array([0.95, 0.74, 0.36], np.float32) * (1.1 - 0.5 * np.hypot(s.XX - C[0], s.YY - C[1]) / max(1.0, ri))[:, :, None]
        zf = 1.02 + 0.05 * ease(u / 1.8); sc = zf * H / s.ah; img = s.pose(t)
        s.draw(win, img, C[0] - (s.fc[0] - s.aw / 2) * sc, C[1] + H * 0.015 - (s.fc[1] - s.ah / 2) * sc, zf)
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        cv2.circle(f, (int(C[0]), int(C[1])), int(ri), (0.30, 0.20, 0.07), max(1, int(5 * k)), cv2.LINE_AA)
        sw = (u - 0.45) / 0.4                                                        # 鏡の面を 光が よぎる
        if 0 < sw < 1:
            L = s.L0(); x = C[0] - ri * 1.3 + ri * 2.6 * sw; cv2.fillPoly(L, [np.array([(x - 30 * k, C[1] - ri), (x + 10 * k, C[1] - ri), (x - 30 * k, C[1] + ri), (x - 70 * k, C[1] + ri)], np.int32)], 0.5, cv2.LINE_AA); f += (L * m)[:, :, None]
        n = len(s.ht); step = W * 0.175
        for i, T in enumerate(s.ht):                                                 # 技名: まん中から 日が さすように
            a = (u - 0.55 - abs(i - 2) * 0.12) / 0.2
            if a > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.835, clamp(a * 2), 1.0)
        for i, T in enumerate(s.vo):
            if u > 0.4 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.755, 1.0, 1.0)
        if u > 1.3: s.text(f, s.txt_en, W / 2, H * 0.835 + 60 * k, ease((u - 1.3) / 0.3), 1.0)
        if u < 0.08: f = f + (1 - u / 0.08) * 0.7
        return f, 0

    # 3) 鏡が 1点を 向く → 光の玉 → 柱が 落ちる
    def sc_judge(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_AIM; o = t - T_ORB; fl = t - T_FALL; T = s.T
        f = s.shrine(t, 0.75 + (0.25 * math.exp(-fl * 2) if fl > 0 else 0))
        aim = [clamp((u - 0.1 - i * 0.07) / 0.12) for i in range(6)]
        if fl < 0:
            L = s.L0(); s.beam(L, s.SRC, s.M0, 1.0, 5)
            if u < 0.5: s.web(f, T_CUT, 6, 1 - u / 0.5)
            for i, p in enumerate([s.M0] + s.mp):                                    # それぞれの鏡から 的へ
                g = ease((u - 0.2 - i * 0.07) / 0.1)
                if g > 0: s.beam(L, p, (p[0] + (T[0] - p[0]) * g, p[1] + (T[1] - p[1]) * g), 1.0, 4)
            s.add(f, L, GOLD, 0.55, 2, 9)
        for i in range(6): s.bronze(f, s.mp[i], 48 * k, 0.5 + 0.5 * aim[i] if fl < 0 else 0.3, 1.0 - 0.35 * math.sin(aim[i] * 3.1416))
        s.hero(f, t, squash=(0.05 * math.exp(-fl * 8) if fl > 0 else 0)); s.soft(f, s.M0, 40 * k, 0.6 if fl < 0 else 0.25, SUN)
        sh = 0
        if o > 0 and fl < 0:                                                         # 光の玉が ふくらみ、落ちる前に きゅっと ちぢむ
            pre = clamp((fl + 0.22) / 0.22); r = (20 + 70 * ease(o / 1.1)) * k * (1 - 0.8 * pre)
            s.soft(f, T, r * 2.2, 0.8, GOLD); cv2.circle(f, (int(T[0]), int(T[1])), max(1, int(r)), (1.0, 0.98, 0.88), -1, cv2.LINE_AA)
            L = s.L0()
            for j in range(12): a = j * 0.5236 + t * 0.8; s.line(L, (T[0] + math.cos(a) * r * 1.25, T[1] + math.sin(a) * r * 1.25), (T[0] + math.cos(a) * r * (1.6 + 0.25 * (j % 2)), T[1] + math.sin(a) * r * (1.6 + 0.25 * (j % 2))), 0.9, 3)
            s.add(f, L, GOLD, 0.8, 2, 6); sh = 2.5 * clamp(o / 1.0)
        if fl > 0:                                                                   # 光の柱
            hw = W * (0.21 * math.exp(-fl * 1.6) + 0.015) * (1 + 0.06 * math.sin(t * 50)); yb = min(H * 1.1, T[1] + H * 2.2 * ease(fl / 0.12))
            m = s.L0(); cv2.rectangle(m, (int(T[0] - hw), -5), (int(T[0] + hw), int(yb)), 1.0, -1); core = s.L0(); cv2.rectangle(core, (int(T[0] - hw * 0.55), -5), (int(T[0] + hw * 0.55), int(yb)), 1.0, -1)
            a = clamp(1.5 - fl * 1.6)
            f += (cv2.GaussianBlur(m, (0, 0), 14 * k) * 0.9 * a)[:, :, None] * GOLD[None, None, :] + (cv2.GaussianBlur(core, (0, 0), 5 * k) * a)[:, :, None]
            L = s.L0()
            for j in range(3):
                v = (fl - j * 0.1) / 0.6
                if 0 < v < 1: s.ell(L, s.FLO, (W * 0.95 * ease(v), W * 0.20 * ease(v)), 0, 1 - v, 4)
            s.add(f, L, GOLD, 0.9, 2, 9)
            if fl < 0.12: f = f + (1 - fl / 0.12) * 0.9
            sh = 18 * math.exp(-fl * 5)
        s.motes(f, t, 0.7)
        if u < 0.1: f = f * (u / 0.1) + (1 - u / 0.1) * 0.6
        return f, sh

    # 4) 床に 日の紋。光のつぶが のぼる
    def sc_end(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_END
        f = s.shrine(t, 0.85)
        L = s.L0(); c = s.FLO; gl = 0.6 + 0.4 * math.exp(-u * 1.5)                       # 日の紋(床に ねた 円)
        s.ell(L, c, (90 * k, 22 * k), 0, gl, 3); s.ell(L, c, (58 * k, 14 * k), 0, gl, 2)
        for j in range(16): a = j * 0.3927; s.line(L, (c[0] + math.cos(a) * 96 * k, c[1] + math.sin(a) * 23.5 * k), (c[0] + math.cos(a) * (116 + 14 * (j % 2)) * k, c[1] + math.sin(a) * (28.5 + 3.5 * (j % 2)) * k), gl, 2)
        s.add(f, L, GOLD, 0.8, 2, 7)
        for i in range(6): s.bronze(f, s.mp[i], 48 * k * (1 - ease((u - 0.3 - i * 0.12) / 0.3)), 0.3, 1.0)
        s.hero(f, t); s.soft(f, s.M0, 36 * k, 0.25 + 0.5 * math.sin(clamp((u - 1.3) / 0.6) * 3.1416), SUN)
        g = u - 1.3
        if 0 < g < 0.6: Ls = s.L0(); s.star4(Ls, s.M0, 56 * k * math.sin(g / 0.6 * 3.1416), 1.0, 0.3, 0.10); s.add(f, Ls, SUN, 0.8, 2, 6)
        s.motes(f, t, 0.9, 1.0)
        end = DUR - t
        if end < 0.45: v = clamp((0.45 - end) / 0.4); f = f * (1 - v) + v
        return f, 0

    # ---- この子だけの曲: 神楽鈴(しゃらん)と 大太鼓、銅の かねの「キィン」が 1回ごとに 上がっていく。光の玉は のぼる きらめき、柱は 大太鼓と ト長調の のびる 和音。さいごは 鈴 ひとふり ----
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
        for i in range(7): a.note(1500 + i * 180, T_AIM + 0.2 + i * 0.07, 0.05, 0.05, "sq", 0.001); a.hat(T_AIM + 0.2 + i * 0.07, 0.12, 0.01)   # 鏡が 向く
        a.rise(T_ORB, T_FALL - 0.2, 300, 2200, 0.08)
        tk = T_ORB; i = 0
        while tk < T_FALL - 0.25: a.note(fq(pent[i % 6] + 12 * (i // 6 % 2)) * 2, tk, 0.15, 0.04, "bell", 0.001); tk += max(0.05, 0.16 - i * 0.006); i += 1
        suzu(T_FALL - 0.9, 0.05, 12)
        odaiko(T_FALL, 1.8); a.a += a.lp(a.nz, 0.3) * a.env(T_FALL, 0.005, 0.5) * 0.6; a.rise(T_FALL - 0.02, T_FALL + 0.12, 4000, 300, 0.12)
        for sm, g in ((0, 0.05), (7, 0.04), (12, 0.04), (16, 0.035), (19, 0.03)): organ(sm, T_FALL, 1.4, g)
        for i in range(8): kane((0, 4, 7, 12, 16, 19, 24, 28)[i] - 12, T_FALL + 0.05 + i * 0.05, 0.05)
        for i, sm in enumerate((16, 14, 12, 7)): kane(sm - 12, T_END + 0.4 + i * 0.4, 0.08)
        organ(0, T_END + 1.6, 1.0, 0.04); organ(7, T_END + 1.6, 1.0, 0.03); suzu(T_END + 1.3, 0.07, 7); odaiko(T_END + 1.6, 0.5)
        a.save(path)


if __name__ == "__main__":
    r = Himika()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
