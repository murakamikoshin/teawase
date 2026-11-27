"""アカツキ(★5 夜明けの王) 専用の必殺技「夜明ケノ王蹴」(絵が とびげり なので 王剣→王蹴 に変更)
夜明け前。雲の海と 山のかげ → 雲を つきやぶって 真上へ。のぼる 大きな日輪を 背に、黒い影で 止まる →
日の出の形(半円)の窓の カットイン → 日輪ごと ななめに 落ちる とびげり(くつ底が せまる) →
雲の海に 当たると、当たった点から 夜が 朝に ぬりかわる → 金色の雲の上に 立つ
使い方: python3 hs_20.py 出力.mp4 / python3 hs_20.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 8.4
SUNC = np.array([1.0, 0.62, 0.20], np.float32); RED = np.array([0.95, 0.22, 0.14], np.float32); GOLD = np.array([1.0, 0.85, 0.45], np.float32)
N_TOP = np.array([0.03, 0.04, 0.14], np.float32); N_LOW = np.array([0.20, 0.10, 0.25], np.float32); D_TOP = np.array([0.30, 0.60, 0.98], np.float32); D_LOW = np.array([1.0, 0.80, 0.55], np.float32)
T_LAUNCH, T_APEX, T_CUT, T_DIVE, T_HIT, T_STAND = 1.2, 1.9, 3.3, 5.3, 7.6, 99.0


class Akatsuki(Kit):
    def __init__(s, W=540):
        super().__init__(20, (1.0, 0.62, 0.20), "夜明ケノ王蹴", "Dawnbreaker", W)
        s.eyeL = s.A(262, 290); s.eyeR = s.A(372, 262); s.emb = s.A(332, 425); s.boot = s.A(80, 470)
        s.ht = [s.text_img(c, hz.SERIF, int(66 * s.k), 3) for c in "夜明ケノ王蹴"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        rg = np.random.RandomState(20); s.rd = rg.rand(200, 5).astype(np.float32)
        s.sil = s.cp.copy(); s.sil[:, :, :3] = 0.02
        s.yn = s.YY / s.H; s.xr = s.XX[0]
        s.peak = [(0, s.H * 0.80), (s.W * 0.18, s.H * 0.70), (s.W * 0.30, s.H * 0.76), (s.W * 0.46, s.H * 0.62), (s.W * 0.60, s.H * 0.74), (s.W * 0.80, s.H * 0.66), (s.W, s.H * 0.78), (s.W, s.H), (0, s.H)]

    # ---- この子だけの絵 ----
    def sky(s, day):
        """day: 0..1 の数 か、画面と同じ大きさの配列(場所ごとに 夜/朝)"""
        d = day if isinstance(day, np.ndarray) else np.full((s.H, s.W), float(day), np.float32)
        y = s.yn[:, :, None]; n = N_TOP[None, None, :] * (1 - y) + N_LOW[None, None, :] * y; dd = D_TOP[None, None, :] * (1 - y) ** 1.2 + D_LOW[None, None, :] * y
        return (n * (1 - d[:, :, None]) + dd * d[:, :, None]).astype(np.float32), d

    def clouds(s, f, t, d, y0=0.72, n=4, drift=1.0):
        """雲の海: もこもこの ふちを 奥から 重ねる"""
        for j in range(n):
            base = s.H * (y0 + j * 0.07); x = s.xr + t * (10 + 8 * j) * s.k * drift + j * 137
            top = base - np.abs(np.sin(x / ((38 + 9 * j) * s.k))) * (16 + 5 * j) * s.k - np.abs(np.sin(x / ((97 + 13 * j) * s.k) + j)) * 14 * s.k
            m = np.clip((s.YY - top[None, :]) / 2.0, 0, 1)[:, :, None]; sh = (0.55 + 0.15 * j)
            col = (np.array([0.16, 0.14, 0.30], np.float32) * sh)[None, None, :] * (1 - d[:, :, None]) + (np.array([1.0, 0.86, 0.62], np.float32) * (0.72 + 0.09 * j))[None, None, :] * d[:, :, None]
            f[:] = f * (1 - m) + col * m
            edge = np.clip(1 - np.abs(s.YY - top[None, :]) / (2.0 * s.k), 0, 1)[:, :, None]; f += edge * (GOLD * 0.25)[None, None, :] * (0.3 + 0.7 * d[:, :, None])

    def sun(s, f, c, r, v=1.0):
        d = np.sqrt((s.XX - c[0]) ** 2 + (s.YY - c[1]) ** 2)
        disc = np.clip((r - d) / 2.0, 0, 1)[:, :, None]; col = (GOLD * 1.05)[None, None, :] * (1 - np.clip(d / r, 0, 1) ** 2 * 0.45)[:, :, None] + (RED * 0.25)[None, None, :] * (np.clip(d / r, 0, 1) ** 2)[:, :, None]
        f[:] = f * (1 - disc * v) + col * disc * v
        f += (np.exp(-np.clip(d - r, 0, None) / (r * 0.35)) * (d > r))[:, :, None] * (SUNC * 0.55)[None, None, :] * v

    def place(s, f, img, cx, cy, hfrac, alpha=1.0): s.put(f, img, cx, cy, hfrac * s.H / s.ah, alpha)

    def emblem(s, f, v, t):
        if v <= 0: return
        p = s.at(*s.emb); sc = s._pl[2]; L = s.L0()
        for i in range(12): a = i * 0.5236 + t; s.line(L, (p[0] + math.cos(a) * 14 * sc, p[1] + math.sin(a) * 14 * sc), (p[0] + math.cos(a) * (30 + 10 * (i % 2)) * sc * v, p[1] + math.sin(a) * (30 + 10 * (i % 2)) * sc * v), v, 2)
        cv2.circle(L, (int(p[0]), int(p[1])), max(1, int(13 * sc)), float(v), -1, cv2.LINE_AA); s.add(f, L, GOLD, 1.0, 2, 8)

    def frame(s, t):
        k = s.k
        if t < T_CUT: f, sh = s.sc_rise(t)
        elif t < T_DIVE: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_HIT: f, sh = s.sc_dive(t)
        else: f, sh = s.sc_dive(T_HIT)
        f, sh = s.finale(f, sh, t, DUR, s.boot, (1, 0.6, 0.2), (1, 0.9, 0.5), -1)
        return s.post(f, SUNC, 0.4, sh * k, t, 0.95)

    # 1) 雲を つきやぶって、日輪を 背に 止まる
    def sc_rise(s, t):
        W, H, k = s.W, s.H, s.k; la = t - T_LAUNCH; ap = t - T_APEX
        f, d = s.sky(0.0)
        L = s.L0()
        for i in range(70): r = s.rd[i]; cv2.circle(L, (int(r[0] * W), int(r[1] * H * 0.6)), 1, float(0.3 + 0.7 * math.sin(t * 2 + r[2] * 30) ** 2), -1)
        f += L[:, :, None] * clamp(1 - ap) if ap > 0 else L[:, :, None]
        hz_ = H * 0.70; glow = ease(t / 1.2)
        f += np.exp(-((s.YY - hz_) / (H * 0.10)) ** 2)[:, :, None] * (SUNC * 0.5 + RED * 0.2)[None, None, :] * glow                 # 地平の 朝やけ
        C = (W / 2, H * 0.34)
        if ap > 0: s.sun(f, (W / 2, hz_ + H * 0.10 - (hz_ + H * 0.10 - C[1]) * ease(ap / 1.1)), W * 0.36 * (0.6 + 0.4 * ease(ap / 1.1)))
        M = s.L0(); cv2.fillPoly(M, [np.array(s.peak, np.int32)], 1.0, cv2.LINE_AA); f = f * (1 - M[:, :, None]) + M[:, :, None] * np.array([0.05, 0.04, 0.10], np.float32)[None, None, :]
        s.clouds(f, t, d, 0.76, 4)
        if la > 0:
            a = ease(la / (T_APEX - T_LAUNCH)); cy = H * 1.05 - (H * 1.05 - C[1]) * a
            L = s.L0()
            if ap < 0.5: s.line(L, (W / 2, H), (W / 2, cy + H * 0.1), clamp(1 - ap * 2 if ap > 0 else 1), int(10 * k))             # 上がる すじ
            for i in range(3):                                                       # 雲の 輪
                v = (la - 0.15 - i * 0.1) / 0.8
                if 0 < v < 1: s.ell(L, (W / 2, H * 0.78), (W * 0.75 * v, W * 0.10 * v), 0, 1 - v, 3)
            s.add(f, L, GOLD * 0.6 + 0.4, 0.9)
            img = s.pose(t, squash=(-0.10 * (1 - a) if ap < 0 else 0.05 * math.exp(-ap * 9) * math.cos(ap * 20)))
            s.draw(f, img, W / 2, cy, 0.42, 0.0); s.place(f, cv2.resize(s.sil, (s.pw, s.ph)), W / 2, cy, 0.42, 1.0)                   # 影のまま
            if ap > 0.75: s.draw(f, img, W / 2, cy, 0.42, ease((ap - 0.75) / 0.4)); s.emblem(f, ease((ap - 0.8) / 0.3), t)
        return f * clamp(t / 0.3), (7 * math.exp(-la * 6) if la > 0 else 0)

    # 2) 日の出の形の 窓
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = np.zeros((H, W, 3), np.float32) + (N_TOP * 0.8)[None, None, :]
        hz_ = H * 0.52; R = W * 0.47 * back(u / 0.4, 1.2); C = (W / 2, hz_)
        m = s.L0(); cv2.ellipse(m, (int(C[0]), int(C[1])), (max(1, int(R)), max(1, int(R))), 0, 180, 360, 1.0, -1, cv2.LINE_AA)
        win = np.zeros((H, W, 3), np.float32) + (GOLD * 0.9)[None, None, :] * (1 - 0.5 * np.clip(np.sqrt((s.XX - C[0]) ** 2 + (s.YY - C[1]) ** 2) / (W * 0.47), 0, 1))[:, :, None] + (RED * 0.25)[None, None, :]
        ex = (s.eyeL[0] + s.eyeR[0]) / 2; ey = (s.eyeL[1] + s.eyeR[1]) / 2; zf = 1.25; sc = zf * H / s.ah
        img = s.pose(t); s.draw(win, img, C[0] - (ex - s.aw / 2) * sc, hz_ - R * 0.42 - (ey - s.ah / 2) * sc, zf)
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        L = s.L0(); cv2.ellipse(L, (int(C[0]), int(C[1])), (max(1, int(R)), max(1, int(R))), 0, 180, 360, 1.0, 3, cv2.LINE_AA); s.line(L, (0, hz_), (W, hz_), 1.0, 3)
        for j in range(7): w = W * (0.42 - j * 0.05) * ease(u / 0.5); s.line(L, (W / 2 - w, hz_ + (14 + j * 13) * k), (W / 2 + w, hz_ + (14 + j * 13) * k), 0.5 - j * 0.05, 2)       # 水にうつる 光のみち
        s.add(f, L, GOLD, 0.6, 2, 6)
        n = len(s.ht); step = W * 0.15
        for i, T in enumerate(s.ht):                                                 # 技名: 地平から のぼる
            a = (u - 0.55 - i * 0.1) / 0.3
            if a > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.76 + (1 - ease(a)) * 40 * k, clamp(a * 1.5), 1.0)
        for i, T in enumerate(s.vo):
            if u > 0.45 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.665, 1.0, 1.0)
        if u > 1.4: s.text(f, s.txt_en, W / 2, H * 0.76 + 62 * k, ease((u - 1.4) / 0.3), 1.0)
        if u < 0.08: f += (1 - u / 0.08) * 0.7
        return f, 0

    def sole(s, f, c, S, rot):
        """こちらへ せまる くつ底(ゴムの みぞ・金の ふち・まん中に 日の紋)"""
        ca, sa = math.cos(rot), math.sin(rot); dg = math.degrees(rot)
        def R(x, y): return (int(c[0] + (x * ca - y * sa) * S), int(c[1] + (x * sa + y * ca) * S))
        K = (0.03, 0.02, 0.02); RB = (0.20, 0.12, 0.08); RB2 = (0.32, 0.20, 0.13); GD = (1.0, 0.84, 0.42)
        for cy_, ax, ay in ((-0.38, 0.52, 0.66), (0.62, 0.40, 0.42)):                # つま先側・かかと
            cv2.ellipse(f, R(0, cy_), (int(S * (ax + 0.06)), int(S * (ay + 0.06))), dg, 0, 360, K, -1, cv2.LINE_AA)
        cv2.fillPoly(f, [np.array([R(-0.42, -0.2), R(0.42, -0.2), R(0.36, 0.7), R(-0.36, 0.7)], np.int32)], K, cv2.LINE_AA)
        for cy_, ax, ay in ((-0.38, 0.52, 0.66), (0.62, 0.40, 0.42)):
            cv2.ellipse(f, R(0, cy_), (int(S * ax), int(S * ay)), dg, 0, 360, GD, -1, cv2.LINE_AA); cv2.ellipse(f, R(0, cy_), (int(S * (ax - 0.05)), int(S * (ay - 0.05))), dg, 0, 360, RB, -1, cv2.LINE_AA)
        cv2.fillPoly(f, [np.array([R(-0.30, -0.1), R(0.30, -0.1), R(0.27, 0.6), R(-0.27, 0.6)], np.int32)], RB, cv2.LINE_AA)
        cv2.ellipse(f, R(-0.12, -0.50), (int(S * 0.30), int(S * 0.42)), dg, 0, 360, RB2, -1, cv2.LINE_AA)                             # てり
        for j in range(6):                                                           # みぞ(つま先)
            y = -0.92 + j * 0.15; w = 0.40 * math.sqrt(max(0.05, 1 - ((y + 0.38) / 0.66) ** 2)); cv2.line(f, R(-w, y), R(w, y + 0.05), K, max(1, int(S * 0.045)), cv2.LINE_AA)
        for j in range(3): y = 0.45 + j * 0.15; w = 0.30 * math.sqrt(max(0.05, 1 - ((y - 0.62) / 0.42) ** 2)); cv2.line(f, R(-w, y), R(w, y + 0.04), K, max(1, int(S * 0.045)), cv2.LINE_AA)
        e = R(0, 0.12)                                                               # 日の紋
        for j in range(12):
            a = rot + j * 0.5236; cv2.line(f, (int(e[0] + math.cos(a) * S * 0.13), int(e[1] + math.sin(a) * S * 0.13)), (int(e[0] + math.cos(a) * S * (0.22 if j % 2 else 0.27)), int(e[1] + math.sin(a) * S * (0.22 if j % 2 else 0.27))), GD, max(1, int(S * 0.035)), cv2.LINE_AA)
        cv2.circle(f, e, max(1, int(S * 0.11)), K, -1, cv2.LINE_AA); cv2.circle(f, e, max(1, int(S * 0.085)), GD, -1, cv2.LINE_AA)

    # 3) 日輪を 背に、とびげりが まっすぐ こちらへ。空は 近づくほど 朝に なり、くつ底が 目の前を ふさぐ
    def sc_dive(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_DIVE; a = ease2(u / (T_HIT - T_DIVE)) ** 1.6
        f, d = s.sky(clamp(a * 1.2) ** 2)
        L = s.L0()
        for i in range(50):                                                          # 奥から こちらへ 流れる 速さの線
            r = s.rd[i]; an = r[0] * 6.283; q = ((r[1] + u * (0.7 + 1.5 * a) * (0.6 + r[2])) % 1.0) ** 2; r0 = q * H * 0.9 + 20 * k
            s.line(L, (W * 0.5 + math.cos(an) * r0, H * 0.34 + math.sin(an) * r0), (W * 0.5 + math.cos(an) * r0 * (1.1 + 0.3 * q), H * 0.34 + math.sin(an) * r0 * (1.1 + 0.3 * q)), 0.25 + 0.5 * q, 1 if q < 0.5 else 2)
        s.add(f, L, GOLD * 0.7 + 0.3, 0.6)
        c = (W * (0.66 - 0.14 * a), H * (0.22 + 0.20 * a)); hf = 0.30 + 0.75 * a
        s.sun(f, (W * 0.5, H * 0.30), W * (0.26 + 0.34 * a))
        s.clouds(f, t, d, 0.86 + 0.20 * a, 3, 2.0)
        img = s.pose(t, jit=1.2 * k * a)
        s.draw(f, img, c[0], c[1], hf); s.emblem(f, 1.0, t)
        bp = s.at(*s.boot); e = clamp((a - 0.30) / 0.70) ** 2.3
        if a > 0.30:
            S = (40 + 1250 * e) * k; sc_ = (bp[0] + (W * 0.50 - bp[0]) * ease2(e * 1.6), bp[1] + (H * 0.50 - bp[1]) * ease2(e * 1.6))
            Lb = s.L0()                                                              # くつ底の 前の 空気の かべ
            for i in range(3): s.circ(Lb, sc_, S * (1.15 + 0.22 * i + 0.1 * math.sin(t * 30 + i)), (0.7 - 0.2 * i) * (1 - e), 3)
            s.add(f, Lb, WHITE, 0.6)
            s.sole(f, sc_, S, 0.5 - 0.35 * e)
        if u < 0.1: f += (1 - u / 0.1) * 0.6
        return f, 2 + 5 * a + 12 * e if a > 0.30 else 2 + 5 * a

    # 4) 夜が 朝に ぬりかわる
    def sc_dawn(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_HIT; st = t - T_STAND
        P = (W * 0.34, H * 0.74); rad = W * 2.2 * ease2(u / 1.6)
        dist = np.sqrt((s.XX - P[0]) ** 2 + (s.YY - P[1]) ** 2)
        dmap = np.clip((rad - dist) / (30 * k) + 0.5, 0, 1).astype(np.float32)
        f, d = s.sky(dmap)
        s.sun(f, (W * 0.5, H * 0.30), W * 0.20, 1.0)
        L = s.L0(); cv2.circle(L, (int(P[0]), int(P[1])), int(rad), 1.0, int(8 * k), cv2.LINE_AA)                                       # 朝の ふち
        for i in range(3):
            v = (u - i * 0.12) / 0.9
            if 0 < v < 1: s.ell(L, P, (W * 1.3 * v, W * 0.22 * v), 0, 1 - v, 4)
        s.clouds(f, t, d, 0.74, 4)
        s.add(f, L, GOLD * 0.5 + 0.5, 1.0, 3, 14)
        Lp = s.L0()
        for i in range(40):                                                          # はねた 雲
            r = s.rd[i]; a = -3.1416 * r[0]; v = u * (0.7 + r[1]); x = P[0] + math.cos(a) * W * 0.6 * v; y = P[1] + math.sin(a) * H * 0.35 * v + H * 0.25 * v * v
            if y < H: cv2.circle(Lp, (int(x), int(y)), max(1, int((16 * r[2] + 5) * k * clamp(1.5 - u))), 0.8, -1, cv2.LINE_AA)
        f = f * (1 - np.clip(Lp, 0, 1)[:, :, None] * 0.9) + np.clip(Lp, 0, 1)[:, :, None] * (np.array([1.0, 0.9, 0.75], np.float32))[None, None, :]
        g = ease(u / 0.9); c = (W * (0.46 + 0.04 * g), H * (0.50 + 0.06 * (1 - g))); hf = 0.89 - 0.45 * g
        img = s.pose(t, squash=(0.06 * math.exp(-st * 8) * math.cos(st * 18) if st > 0 else 0)); s.draw(f, img, c[0], c[1], hf); s.emblem(f, 1.0, t)
        Lr = s.L0()
        for i in range(50):                                                          # 朝の 光の つぶ
            r = s.rd[i + 60]; tw = math.sin(t * 3 + r[2] * 30) ** 6; s.star4(Lr, (r[0] * W, r[1] * H * 0.7), (2 + 6 * r[3]) * k * tw, 1.0, 0.0, 0.2)
        f += Lr[:, :, None] * dmap[:, :, None]
        if u < 0.16: f = f + (1 - u / 0.16) * 1.4
        end = DUR - t
        if end < 0.7: v = clamp((0.7 - end) / 0.4); f = f * (1 - v) + v * (GOLD * 0.3 + 0.7)[None, None, :]
        if end < 0.35: f = f * clamp(end / 0.3) ** 2
        return f, 24 * math.exp(-u * 4)

    # ---- この子だけの曲: 夜明けの ほら貝と 太鼓。のぼる時は 5度ずつ 上がる らっぱ。落ちる間は 太鼓の 連打が 速くなり、当たったあとは 明るい 祝いの ふし(ト長調) ----
    def audio(s, path):
        a = Snd(DUR, 20); G = 196.0; maj = [0, 2, 4, 5, 7, 9, 11]
        def fq(d, o=0): return G * 2 ** ((maj[d % 7] + 12 * (d // 7 + o)) / 12.0)
        def horn(d, t0, dur, g=0.08, o=0): a.note(fq(d, o), t0, dur, g, "saw", 0.05); a.note(fq(d, o) * 1.003, t0, dur, g * 0.5, "saw", 0.05)
        a.a += a.lp(a.nz, 0.012) * 0.4 * a.gate(0, T_LAUNCH, 0.3)
        horn(0, 0.15, 0.9, 0.06, -1)                                                                                    # ほら貝
        a.whoosh(T_LAUNCH - 0.1, T_APEX, 0.7, 0.3); a.rise(T_LAUNCH, T_APEX, 120, 900, 0.14)
        a.boom(T_APEX, 52, 1.0, 0.5)
        for i, d in enumerate([0, 4, 7, 11]): horn(d, T_APEX + 0.05 + i * 0.16, 0.5, 0.07)
        a.boom(T_CUT, 46, 1.0)
        for i in range(6): horn([0, 2, 4, 7, 4, 7][i], T_CUT + 0.55 + i * 0.1, 0.18, 0.07); a.boom(T_CUT + 0.55 + i * 0.1, 80, 0.25, 0.08)
        for d in (0, 4): a.note(fq(d, -1), T_CUT + 0.5, 1.6, 0.05, "pad", 0.2)
        tk = T_DIVE; i = 0
        while tk < T_HIT:                                                            # 連打が 速くなる
            a.boom(tk, 60 + i, 0.4, 0.09); a.hat(tk, 0.05)
            if i % 2 == 0: horn([7, 6, 5, 4, 3, 2, 1, 0][(i // 2) % 8], tk, 0.12, 0.05, 1)
            tk += max(0.055, 0.2 - i * 0.008); i += 1
        a.rise(T_DIVE, T_HIT, 900, 150, 0.16); a.whoosh(T_DIVE, T_HIT, 0.8, 0.25)
        a.boom(T_HIT, 34, 1.7, 0.9); a.boom(T_HIT + 0.15, 60, 0.9, 0.4); a.a += a.lp(a.nz, 0.2) * a.env(T_HIT, 0.01, 0.8) * 0.7
        s.fin_snd(a, DUR, 20); a.save(path); return
        for j, (d, o) in enumerate(((0, -1), (4, -1), (0, 0), (2, 0), (4, 0), (7, 0))): a.note(fq(d, o), T_HIT + 0.2 + j * 0.04, 2.6, 0.045, "pad", 0.3)
        bt = 0.19; mel = [4, 4, 7, 4, 9, 7, 4, 7, 11, 9, 7, 9, 11, 14]
        for i, d in enumerate(mel):                                                  # 祝いの ふし
            tk = T_HIT + 0.6 + i * bt; horn(d, tk, 0.16, 0.07); a.note(fq(d, 1), tk, 0.2, 0.04, "bell")
            if i % 2 == 0: a.boom(tk, 58, 0.4, 0.12)
            else: a.hat(tk, 0.06)
        horn(14, T_HIT + 0.6 + len(mel) * bt, 0.9, 0.08); a.boom(T_HIT + 0.6 + len(mel) * bt, 44, 0.9, 0.5)
        s.fin_snd(a, DUR, 20); a.save(path)


if __name__ == "__main__":
    r = Akatsuki()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
