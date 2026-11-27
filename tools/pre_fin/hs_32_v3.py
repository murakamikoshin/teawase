"""クロノス(★5 時の賢者) 専用の必殺技「時止メノ杖」— 陽気な おじいさん なので 明るく
夜の雨。しずく・木の葉・りんごが ふってくる中へ ぴょんと 出る → 杖を かかげて「チン」。ぜんぶ 空中で 止まり、色が あせる(本人だけ 色つき) →
砂時計の形の 窓のカットイン → 空の 大きな砂時計を ひっくり返す。砂が 上へ 流れ、止まっていた物が 上へ もどっていく。夜→夕やけ→昼へ 空も まきもどる →
ぱちん。時が 動きだし、ぜんぶが いっぺんに 落ちてくる
使い方: python3 hs_32.py 出力.mp4 / python3 hs_32.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 11.6
GOLD = np.array([1.0, 0.84, 0.40], np.float32); TEAL = np.array([0.20, 0.80, 0.66], np.float32); ICE = np.array([0.80, 0.95, 1.0], np.float32)
SKYS = [((0.04, 0.06, 0.18), (0.14, 0.16, 0.34)), ((0.30, 0.16, 0.40), (1.0, 0.52, 0.28)), ((0.30, 0.62, 0.98), (0.85, 0.93, 1.0))]      # 夜 / 夕 / 昼 (上, 下)
T_STOP, T_CUT, T_BACK, T_FLIP, T_REW, T_SNAP, T_LAND = 1.8, 2.6, 4.6, 4.9, 5.4, 8.4, 8.85


class Chronos(Kit):
    def __init__(s, W=540):
        super().__init__(32, (1.0, 0.84, 0.40), "時止メノ杖", "Chrono Halt", W)
        s.gemp = s.A(262, 62); s.face = s.A(352, 290)
        s.ht = [s.text_img(c, hz.SERIF, int(78 * s.k), 3) for c in "時止メノ杖"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        rg = np.random.RandomState(32); s.ob = rg.rand(90, 6).astype(np.float32)      # 0-59 しずく / 60-79 木の葉 / 80-89 りんご
        s.yn = s.YY / s.H; s.gnd = s.H * 0.86

    # ---- 時の流れ ----
    def phase(s, t):
        if t < T_STOP: return t, 1.0
        if t < T_REW: return T_STOP, 0.0
        if t < T_SNAP: u = min(t, T_SNAP - 0.1) - T_REW; return T_STOP - 0.22 * u * u, (-0.44 * u if t < T_SNAP - 0.1 else 0.0)
        u = T_SNAP - 0.1 - T_REW; return T_STOP - 0.22 * u * u + 3.2 * (t - T_SNAP), 3.2

    def tod(s, t):
        """空の時刻: 0=夜 1=夕 2=昼"""
        if t < T_REW: return 0.0
        return 2.0 * ease2((t - T_REW) / (T_SNAP - T_REW - 0.2))

    def sky(s, tod):
        i = min(1, int(tod)); q = tod - i; a, b = SKYS[i], SKYS[i + 1]
        top = np.array(a[0]) * (1 - q) + np.array(b[0]) * q; low = np.array(a[1]) * (1 - q) + np.array(b[1]) * q
        f = (top[None, None, :] * (1 - s.yn)[:, :, None] + low[None, None, :] * s.yn[:, :, None]).astype(np.float32)
        g = int(s.gnd); f[g:, :, :] = np.array([0.10, 0.20, 0.16], np.float32) * (0.5 + 0.5 * tod / 2)
        return f

    def things(s, f, t, v=1.0):
        ph, vel = s.phase(t); H, W, k = s.H, s.W, s.k; L = s.L0()
        for i in range(90):
            r = s.ob[i]; sp = 0.55 + 0.6 * r[2]; y = ((r[1] + ph * sp) % 1.0) * (s.gnd + 40 * k) - 20 * k; x = r[0] * W
            if i < 60:                                                               # しずく: 動けば すじ、止まれば 玉
                ln = abs(vel) * sp * 30 * k
                if ln < 3 * k: cv2.circle(L, (int(x), int(y)), max(1, int((2 + 2 * r[3]) * k)), 0.9 * v, -1, cv2.LINE_AA)
                else: s.line(L, (x, y), (x, y - ln * (1 if vel > 0 else -1)), 0.6 * v, 1 if ln < 40 * k else 2)
            elif i < 80:
                x += math.sin(ph * 4 + r[3] * 9) * 26 * k; a = ph * 200 * (r[4] - 0.5) + r[3] * 180
                cv2.ellipse(f, (int(x), int(y)), (int(11 * k), int(5 * k)), a, 0, 360, tuple(float(q) for q in np.array([0.85, 0.55, 0.18]) * v + np.array([0.3, 0.3, 0.3]) * (1 - v)), -1, cv2.LINE_AA)
            else:
                cv2.circle(f, (int(x), int(y)), int(11 * k), tuple(float(q) for q in np.array([0.90, 0.18, 0.16]) * v + np.array([0.3, 0.3, 0.3]) * (1 - v)), -1, cv2.LINE_AA)
                cv2.line(f, (int(x), int(y - 10 * k)), (int(x + 3 * k), int(y - 17 * k)), (0.25, 0.16, 0.08), max(1, int(2 * k)), cv2.LINE_AA)
        f += (L + cv2.GaussianBlur(L, (0, 0), 2 * k))[:, :, None] * ICE[None, None, :]

    def glass(s, f, c, w, h, rot, fill, t, stream=0):
        """砂時計。fill=上の玉に のこる 砂(0..1)。stream: 1=下へ -1=上へ"""
        ca, sa = math.cos(rot), math.sin(rot)
        def R(x, y): return (c[0] + x * ca - y * sa, c[1] + x * sa + y * ca)
        top = [R(-w, -h), R(w, -h), R(0, 0)]; bot = [R(0, 0), R(w, h), R(-w, h)]
        for tri, lv, up in ((top, fill, True), (bot, 1 - fill, False)):
            if lv > 0.02:
                q = math.sqrt(lv)
                P = [R(-w * q, -h * q), R(w * q, -h * q), R(0, 0)] if up else [R(-w * (1 - (1 - q)), h), R(w, h), R(w * (1 - q) * 0 + 0, h * (1 - q))]
                if not up: P = [R(-w, h), R(w, h), R(w * (1 - q), h * (1 - q)), R(-w * (1 - q), h * (1 - q))]
                cv2.fillPoly(f, [np.array(P, np.int32)], tuple(float(v) for v in GOLD * 0.95), cv2.LINE_AA)
        L = s.L0(); cv2.polylines(L, [np.array(top, np.int32)], True, 1.0, 3, cv2.LINE_AA); cv2.polylines(L, [np.array(bot, np.int32)], True, 1.0, 3, cv2.LINE_AA)
        s.line(L, R(-w * 1.15, -h), R(w * 1.15, -h), 1.0, 5); s.line(L, R(-w * 1.15, h), R(w * 1.15, h), 1.0, 5)
        if stream:
            for j in range(6): v = ((t * 2.5 * stream + j / 6.0) % 1.0); p = R(0, (h * 0.9 * v) if stream > 0 else (-h * 0.9 * (1 - v))); cv2.circle(L, (int(p[0]), int(p[1])), max(1, int(2.5 * s.k)), 1.0, -1)
        s.add(f, L, GOLD, 0.7, 2, 7)

    def gem(s, f, v, t):
        if v <= 0: return
        p = s.at(*s.gemp); sc = s._pl[2]; L = s.L0(); s.star4(L, p, (22 + 6 * math.sin(t * 8)) * sc * 2 * v, min(1.0, v), 0.0, 0.14); cv2.circle(L, (int(p[0]), int(p[1])), max(1, int(12 * sc)), float(min(1.0, v)), -1, cv2.LINE_AA)
        s.add(f, L, ICE * 0.6 + GOLD * 0.4, 1.0, 2, 8)

    def frame(s, t):
        k = s.k
        if t < T_CUT: f, sh = s.sc_stop(t)
        elif t < T_BACK: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_LAND: f, sh = s.sc_rewind(t)
        else: f, sh = s.sc_land(t - T_LAND)
        f, sh = s.finale(f, sh, t, DUR, s.gemp, (1, 0.84, 0.4), (0.3, 0.9, 0.75), 4)
        return s.post(f, GOLD, 0.35, sh * k, t, 0.96)

    def sepia(s, f, a):
        g = f.mean(2, keepdims=True); return f * (1 - a) + g * np.array([1.08, 0.98, 0.82], np.float32)[None, None, :] * a

    # 1) ふってくる中で、時を 止める
    def sc_stop(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_STOP
        f = s.sky(0.0); s.things(f, t, 1.0)
        if u > 0: f = s.sepia(f, 0.9 * ease(u / 0.25))
        hop = math.sin(clamp(t / 1.0) * 3.1416) * H * 0.06 if t < 1.0 else 0
        cx = W * (-0.3 + 0.8 * ease(t / 1.0)); cy = s.gnd - H * 0.24 - hop
        img = s.pose(t, squash=(0.06 * math.exp(-(t - 1.0) * 9) * math.cos((t - 1.0) * 18) if t > 1.0 else 0) + (-0.05 * math.exp(-u * 8) if u > 0 else 0))
        s.draw(f, img, cx, cy, 0.44); s.gem(f, (clamp((t - 1.3) / 0.4) + (1.2 * math.exp(-u * 4) if u > 0 else 0)), t)
        if u > 0:
            p = s.at(*s.gemp); L = s.L0()
            for i in range(3):
                v = (u - i * 0.12) / 0.8
                if 0 < v < 1: s.circ(L, p, W * 1.5 * v, 1 - v, 3)
            s.add(f, L, ICE, 0.9)
            if u < 0.08: f += (1 - u / 0.08) * 0.7
        return f * clamp(t / 0.3), 0

    # 2) 砂時計の形の 窓
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = np.zeros((H, W, 3), np.float32) + np.array([0.03, 0.05, 0.08], np.float32)
        g = back(u / 0.35, 1.2); C = (W / 2, H * 0.38); w = W * 0.40 * g; h = H * 0.27 * g
        top = [(C[0] - w, C[1] - h), (C[0] + w, C[1] - h), C]; bot = [C, (C[0] + w, C[1] + h), (C[0] - w, C[1] + h)]
        m = s.L0(); cv2.fillPoly(m, [np.array(top, np.int32)], 1.0, cv2.LINE_AA)
        win = np.zeros((H, W, 3), np.float32) + (TEAL * 0.35)[None, None, :]
        zf = 1.15; sc = zf * H / s.ah; img = s.pose(t); s.draw(win, img, C[0] - (s.face[0] - s.aw / 2) * sc, C[1] - h * 0.60 - (s.face[1] - s.ah / 2) * sc, zf)
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        lv = clamp(u / 1.9); q = math.sqrt(lv)                                        # 下の玉に 砂が たまる
        P = [(C[0] - w, C[1] + h), (C[0] + w, C[1] + h), (C[0] + w * (1 - q), C[1] + h * (1 - q)), (C[0] - w * (1 - q), C[1] + h * (1 - q))]
        cv2.fillPoly(f, [np.array(P, np.int32)], tuple(float(v) for v in GOLD * 0.9), cv2.LINE_AA)
        L = s.L0(); cv2.polylines(L, [np.array(top, np.int32)], True, 1.0, 4, cv2.LINE_AA); cv2.polylines(L, [np.array(bot, np.int32)], True, 1.0, 4, cv2.LINE_AA)
        s.line(L, (C[0] - w * 1.1, C[1] - h), (C[0] + w * 1.1, C[1] - h), 1.0, 6); s.line(L, (C[0] - w * 1.1, C[1] + h), (C[0] + w * 1.1, C[1] + h), 1.0, 6)
        if lv < 1: s.line(L, C, (C[0], C[1] + h * (1 - q)), 0.9, 2)
        s.add(f, L, GOLD, 0.6, 2, 6)
        n = len(s.ht); step = W * 0.175
        for i, T in enumerate(s.ht):                                                 # 技名: 砂のように 上から 落ちて 止まる
            a = (u - 0.5 - i * 0.13) / 0.25
            if a > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.80 - (1 - ease(a)) * 70 * k, clamp(a * 2), 1.0)
        for i, T in enumerate(s.vo):
            if u > 0.4 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.705, 1.0, 1.0)
        if u > 1.4: s.text(f, s.txt_en, W / 2, H * 0.80 + 60 * k, ease((u - 1.4) / 0.3), 1.0)
        if u < 0.08: f += (1 - u / 0.08) * 0.7
        return f, 0

    # 3) 砂時計を ひっくり返す → まきもどる → ぱちん
    def sc_rewind(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_BACK; rw = t - T_REW; sn = t - T_SNAP
        f = s.sky(s.tod(t))
        td = s.tod(t)
        if td > 0.3:                                                                 # 西から もどってくる 太陽
            a = 3.1416 * (1.0 - 0.45 * (td - 0.3) / 1.7); c = (W / 2 + math.cos(a) * W * 0.55, s.gnd - math.sin(a) * H * 0.62)
            s.soft(f, c, W * 0.10, 0.8, GOLD * 0.9 + 0.2); cv2.circle(f, (int(c[0]), int(c[1])), int(W * 0.055), (1.0, 0.95, 0.75), -1, cv2.LINE_AA)
        s.things(f, t, 1.0 if rw > 0 else 0.4)
        if rw < 0: f = s.sepia(f, 0.9)
        elif sn < 0: f = s.sepia(f, 0.9 * (1 - ease(rw / 0.5)))
        G = (W / 2, H * 0.22); fl = t - T_FLIP
        rot = 3.1416 * ease2(fl / 0.45) if fl > 0 else 0.0
        fill = clamp(rw / (T_SNAP - T_REW - 0.2)) if rw > 0 else 0.0                    # ひっくり返したあと: 下の砂が 上へ のぼる
        s.glass(f, G, W * 0.13, H * 0.105, rot + (3.1416 if fl > 0.45 else 0) * 0, fill if fl > 0.45 else 0.0, t, -1 if (rw > 0 and sn < 0) else 0)
        if rw > 0 and sn < 0:                                                        # まきもどしの 矢印の 輪
            L = s.L0(); a0 = -rw * (3 + 4 * rw / 3.0); R = W * 0.30
            for j in range(3):
                b = a0 + j * 2.094; cv2.ellipse(L, (int(G[0]), int(G[1])), (int(R), int(R)), 0, math.degrees(b), math.degrees(b) + 75, 0.7, 3, cv2.LINE_AA)
                p = (G[0] + math.cos(b) * R, G[1] + math.sin(b) * R); cv2.fillPoly(L, [np.array([(p[0] + math.cos(b - 1.57) * 16 * k, p[1] + math.sin(b - 1.57) * 16 * k), (p[0] + math.cos(b) * 9 * k, p[1] + math.sin(b) * 9 * k), (p[0] - math.cos(b) * 9 * k, p[1] - math.sin(b) * 9 * k)], np.int32)], 0.9, cv2.LINE_AA)
            s.add(f, L, GOLD, 0.8)
        cy = s.gnd - H * 0.24
        img = s.pose(t, squash=(-0.05 * math.exp(-fl * 8) if fl > 0 else 0) + (0.06 * math.exp(-sn * 12) if sn > 0 else 0), jit=(1.0 * k * clamp(rw / 3.0) if rw > 0 and sn < 0 else 0))
        s.draw(f, img, W / 2, cy, 0.44); s.gem(f, 0.8 + (0.6 * math.exp(-fl * 5) if fl > 0 else 0), t)
        if fl > 0 and fl < 0.5:
            p = s.at(*s.gemp); L = s.L0(); s.line(L, p, (p[0] + (G[0] - p[0]) * ease(fl / 0.2), p[1] + (G[1] - p[1]) * ease(fl / 0.2)), 1 - fl / 0.5, 2); s.add(f, L, ICE, 1.0)
        if sn > 0 and sn < 0.1: f = f + (1 - sn / 0.1) * 0.9
        if u < 0.12: f = f * (u / 0.12) + (1 - u / 0.12)
        return f, (3 * clamp(rw / 3.0) if rw > 0 and sn < 0 else 0) + (8 * math.exp(-sn * 10) if sn > 0 else 0)

    # 4) いっぺんに 落ちてくる
    def sc_land(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_LAND
        f = s.sky(2.0); c = (W * 0.72, H * 0.16); s.soft(f, c, W * 0.10, 0.8, GOLD * 0.9 + 0.2); cv2.circle(f, (int(c[0]), int(c[1])), int(W * 0.055), (1.0, 0.95, 0.75), -1, cv2.LINE_AA)
        L = s.L0(); P = (W / 2, s.gnd)
        for i in range(90):                                                          # 地面で はねる
            r = s.ob[i]; a = -3.1416 * r[0]; v = u * (0.5 + r[2]); x = r[0] * W + math.cos(a) * W * 0.12 * v; y = s.gnd - abs(math.sin(v * 5)) * H * 0.22 * math.exp(-v * 2.2) * (0.3 + r[3])
            if i < 60: cv2.circle(L, (int(x), int(y)), max(1, int((2 + 2 * r[3]) * k)), float(clamp(1.6 - u * 0.8)), -1, cv2.LINE_AA)
            elif i < 80: cv2.ellipse(f, (int(x), int(y)), (int(11 * k), int(5 * k)), v * 300 * (r[4] - 0.5), 0, 360, (0.85, 0.55, 0.18), -1, cv2.LINE_AA)
            else: cv2.circle(f, (int(x), int(y - 6 * k)), int(11 * k), (0.90, 0.18, 0.16), -1, cv2.LINE_AA)
        for i in range(3):
            v = (u - i * 0.1) / 0.8
            if 0 < v < 1: s.ell(L, P, (W * 1.2 * v, W * 0.16 * v), 0, 1 - v, 4)
        for i in range(60): x = s.ob[i][0] * W; s.line(L, (x, s.gnd), (x + (s.ob[i][1] - 0.5) * 30 * k, s.gnd - (40 + 120 * s.ob[i][2]) * k * clamp(1 - u / 0.4)), clamp(1 - u / 0.4), 2)     # しぶき
        f += (L + cv2.GaussianBlur(L, (0, 0), 3 * k))[:, :, None] * (ICE * 0.7 + 0.3)[None, None, :]
        s.glass(f, (W / 2, H * 0.22), W * 0.13, H * 0.105, 0.0, 1.0, t, 0)
        hop = abs(math.sin(u * 5.2)) * H * 0.035 * math.exp(-u * 0.9)                # うれしくて はねる
        img = s.pose(t, squash=0.05 * math.exp(-u * 7)); s.draw(f, img, W / 2, s.gnd - H * 0.24 - hop, 0.44); s.gem(f, 0.7, t)
        if u < 0.12: f = f + (1 - u / 0.12) * 1.0
        end = DUR - t
        if end < 0.5: v = clamp((0.5 - end) / 0.45); f = f * (1 - v) + v
        return f, 18 * math.exp(-u * 5)

    # ---- この子だけの曲: 陽気な 3拍子の オルゴール。カチコチが だんだん おそくなって 止まる。まきもどしは 同じ ふしを うしろから、だんだん 速く。ぱちん のあと 元の ふしで しめる ----
    def audio(s, path):
        a = Snd(DUR, 32); F = 349.23; maj = [0, 2, 4, 5, 7, 9, 11]
        def fq(d, o=0): return F * 2 ** ((maj[d % 7] + 12 * (d // 7 + o)) / 12.0)
        def box(d, t0, g=0.09, o=0, dec=0.4): a.note(fq(d, o), t0, dec, g, "bell", 0.001)
        def tick(t0, g=0.14, hi=True): a.note(2400 if hi else 1700, t0, 0.02, g, "sine", 0.0005)
        tune = [0, 2, 4, 7, 4, 2, 5, 4, 2, 1, 2, 4]
        a.a += (a.nz - a.lp(a.nz, 0.5)) * 0.06 * a.gate(0, T_STOP, 0.2)                                                 # 雨
        tk = 0.1; i = 0
        while tk < T_STOP: box(tune[i % 12], tk, 0.08, 1, 0.25); tick(tk, 0.10, i % 2 == 0); a.note(fq([0, 4, 4][i % 3], -1), tk, 0.2, 0.05, "sq"); tk += 0.14 + i * 0.008; i += 1
        a.note(fq(7, 2), T_STOP, 1.4, 0.16, "bell"); a.note(fq(11, 2), T_STOP, 1.2, 0.08, "bell")                        # チン
        a.a += a.tone(58) * 0.07 * a.gate(T_STOP, T_REW, 0.2)
        a.boom(T_CUT, 60, 0.7, 0.2)
        for i in range(5): box([7, 5, 4, 2, 0][i], T_CUT + 0.5 + i * 0.13, 0.10, 1)
        a.a += (a.nz - a.lp(a.nz, 0.6)) * 0.04 * a.gate(T_CUT, T_CUT + 1.9, 0.1)                                        # 砂
        a.whoosh(T_FLIP, T_FLIP + 0.45, 0.4, 0.3); a.boom(T_FLIP + 0.45, 80, 0.5, 0.15)
        tk = T_REW; i = 0
        while tk < T_SNAP - 0.12:                                                    # うしろから、だんだん 速く
            box(tune[(11 - i) % 12] + (i // 12) * 2, tk, 0.08, 1, 0.15); tick(tk, 0.08, i % 2 == 1)
            if i % 3 == 0: a.note(fq([4, 0, 5, 2][(i // 3) % 4], -1), tk, 0.15, 0.06, "sq")
            tk += max(0.05, 0.22 - i * 0.006); i += 1
        a.rise(T_REW, T_SNAP, 200, 1600, 0.10)
        a.note(3000, T_SNAP, 0.05, 0.25, "sine", 0.0005); a.a += (a.nz - a.lp(a.nz, 0.7)) * a.env(T_SNAP, 0.0005, 0.02) * 0.5   # ぱちん
        a.whoosh(T_SNAP, T_LAND, 0.9, 0.4); a.boom(T_LAND, 40, 1.5, 0.7); a.a += a.lp(a.nz, 0.3) * a.env(T_LAND, 0.005, 0.5) * 0.6
        for i in range(14): a.note(700 + (i * 173) % 900, T_LAND + 0.05 + i * 0.05, 0.08, 0.04, "sine")                 # ぽとぽと
        for i in range(9): tk = T_LAND + 0.5 + i * 0.19; box(tune[i] + (7 if i == 8 else 0), tk, 0.10, 1, 0.35); a.note(fq([0, 4, 4][i % 3], -1), tk, 0.22, 0.06, "sq")
        for d in (0, 2, 4, 7): box(d, T_LAND + 0.5 + 9 * 0.19, 0.09, 0, 1.0)
        s.fin_snd(a, DUR, 32); a.save(path)


if __name__ == "__main__":
    r = Chronos()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
