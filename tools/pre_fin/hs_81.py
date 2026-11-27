"""アルテア(★7 はじまりの光) 専用の必殺技「星冠ノ創世」
まっくらな中に 杖の先の一番星が灯る → その光で 姿が浮かび上がる → 翼がひらいて 金の目がひらく → 目もとのカットイン →
杖から 光の矢が空へ飛び、12の星をむすんで 星の冠を編む → 冠が 一点に ちぢむ(しずけさ) → そこから 銀河が生まれる
使い方: python3 hs_81.py 出力.mp4 / python3 hs_81.py still 出力.png"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 14.0
GOLD = np.array([1.0, 0.84, 0.42], np.float32); PALE = np.array([0.72, 0.85, 1.0], np.float32); DEEP = np.array([0.10, 0.13, 0.34], np.float32)
T_EYE, T_CUT, T_CROWN, T_HUSH, T_BANG = 2.7, 4.3, 6.5, 9.9, 10.7


class Altea(Kit):
    def __init__(s, W=540):
        super().__init__(81, (1.0, 0.86, 0.5), "星冠ノ創世", "Genesis Corona", W)
        s.eyeL = s.A(390, 322); s.eyeR = s.A(470, 322); s.staff = s.A(165, 170); s.crown = s.A(415, 150)
        s.wing = (s.A(415, 0)[0], 170 * s.ah / 760, 400 * s.ah / 760, 450 * s.ah / 760)
        s.vt = [s.text_img(c, hz.SERIF, int(66 * s.k), 3) for c in "星冠ノ創世"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        rg = np.random.RandomState(81)
        s.sky = [(rg.rand() * s.W, rg.rand() * s.H, rg.rand(), rg.rand()) for _ in range(150)]
        s.feath = rg.rand(60, 5).astype(np.float32)
        s.gal = rg.rand(900, 4).astype(np.float32)                                    # 銀河のつぶ(腕・半径・ぶれ・明るさ)
        s.dust = rg.rand(140, 4).astype(np.float32)
        ay, ax = np.mgrid[0:s.ph, 0:s.pw].astype(np.float32)
        s.dstaff = np.sqrt((ax - s.pad - s.staff[0]) ** 2 + (ay - s.pad - s.staff[1]) ** 2)   # 杖の星からの距離(絵の中)

    # ---- この子だけの絵 ----
    def night(s, t, lum=1.0, warm=0.0):
        f = np.zeros((s.H, s.W, 3), np.float32)
        off = int((t * 9) % 300); n = s.neb[:, off:off + s.W]
        f += ((n ** 2 * 1.5 + 0.10) * lum)[:, :, None] * (DEEP * (1 - warm) + GOLD * 0.30 * warm + PALE * 0.05)[None, None, :]
        L = s.L0()
        for x, y, p, q in s.sky:
            cv2.circle(L, (int(x), int((y + t * 4 * q) % s.H)), 1, float((0.3 + 0.7 * math.sin(t * 2.5 + p * 30) ** 2) * lum), -1)
        f += (L + cv2.GaussianBlur(L, (0, 0), 2.5 * s.k) * 1.5)[:, :, None] * (PALE * 0.8 + 0.2)[None, None, :]
        return f

    def eyes(s, f, v, streak=1.0):
        """金の目が ひかる。絵の目の位置に合わせる"""
        if v <= 0: return
        L = s.L0(); sc = s._pl[2]
        for e in (s.eyeL, s.eyeR):
            p = s.at(*e)
            cv2.ellipse(L, (int(p[0]), int(p[1])), (max(1, int(15 * sc * s.k)), max(1, int(21 * sc * s.k))), 0, 0, 360, float(v), -1, cv2.LINE_AA)
            s.line(L, (p[0] - 150 * sc * streak * v, p[1]), (p[0] + 150 * sc * streak * v, p[1]), 0.8 * v, 1)
        s.add(f, L, GOLD, 1.3, 2, 9)

    def crown_pts(s, t, O, R, n=12):
        pts = []
        for i in range(n):
            a = -1.5708 + i * 6.2832 / n + t * 0.25; rr = R * (1.0 if i % 2 == 0 else 0.72)
            pts.append((O[0] + math.cos(a) * rr, O[1] + math.sin(a) * rr * 0.42 - (R * 0.30 if i % 2 == 0 else 0)))
        return pts

    def frame(s, t):
        W, H, k = s.W, s.H, s.k
        if t < T_CUT: f, sh, g = s.sc_dawn(t)
        elif t < T_CROWN: f, sh, g = s.sc_cut(t - T_CUT)
        elif t < T_BANG: f, sh, g = s.sc_crown(t)
        else: f, sh, g = s.sc_bang(t - T_BANG)
        return s.post(f, GOLD, g * 0.5, sh * k, t, 0.97)

    # 1) 一番星が灯り、その光で 姿が浮かぶ。翼がひらき、目がひらく
    def sc_dawn(s, t):
        W, H, k = s.W, s.H, s.k
        lit = ease2((t - 0.9) / 1.6)                                                # 光が ひろがる
        f = s.night(t, 0.25 + 0.75 * lit)
        open_ = back((t - T_EYE) / 0.55)                                             # 翼が ばさっと ひらく
        zoom = 0.44 + 0.06 * ease((t - T_EYE) / 0.8)
        img = s.pose(t, sway=0.9 * math.exp(-max(0, t - T_EYE) * 2.5) * (t > T_EYE), breath=0.3, stretch=s.wing + (0.34 * (1 - open_),))
        R = (40 + 1100 * lit) * k
        light = np.clip(1.15 - s.dstaff / R, 0.0, 1.0) ** 0.8 if lit < 1 else None
        if t < 0.9: light = np.zeros_like(s.dstaff)
        cy = H * 0.50
        s.draw(f, img, W / 2, cy, zoom, light=light)
        sp = s.at(*s.staff)
        # 一番星: 0.3秒で ぽっと灯り、脈うつ
        if t > 0.3:
            v = ease((t - 0.3) / 0.5); pulse = 1 + 0.15 * math.sin(t * 9)
            L = s.L0(); s.star4(L, sp, (26 + 60 * v * pulse) * k, 1.0, t * 0.6); s.star4(L, sp, (14 + 30 * v) * k, 1.0, t * 0.6 + 0.7854, 0.3)
            s.add(f, L, GOLD, 1.2); s.soft(f, sp, (60 + 200 * lit) * k, 0.35 * v, GOLD)
        if t > T_EYE:                                                               # 開眼: 目・輪・羽
            u = t - T_EYE
            s.eyes(f, clamp(1.6 - u * 0.7) if u < 1.3 else 0.5, 1.0)
            L = s.L0()
            for i in range(2):
                v = (u - i * 0.16) / 0.9
                if 0 < v < 1: s.circ(L, (W / 2, cy - H * 0.05), W * (0.2 + 1.1 * v), 1 - v, 3)
            for i in range(60):                                                     # 羽が舞う
                r = s.feath[i]; sg = 1 if i % 2 else -1; v = u * (0.5 + r[2])
                x = W / 2 + sg * (W * 0.18 + v * W * (0.25 + 0.5 * r[0])); y = cy - H * 0.12 + (r[1] - 0.5) * H * 0.3 + v * v * H * 0.10 - v * H * 0.06
                a = r[3] * 6.28 + u * 3 * (r[4] - 0.5)
                if 0 < x < W: cv2.ellipse(L, (int(x), int(y)), (int(11 * k), int(3 * k)), math.degrees(a), 0, 360, float(clamp(1.3 - v)), -1, cv2.LINE_AA)
            s.add(f, L, WHITE * 0.9 + GOLD * 0.1, 0.9)
            if u < 0.18: f += (1 - u / 0.18) * 0.7
        return f * clamp(t / 0.3), (9 * clamp(1 - (t - T_EYE) / 0.5) if t > T_EYE else 0), 0.9

    # 2) カットイン: 目もとの帯 と 縦書きの技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.night(t, 0.35)
        op = ease(u / 0.3); cyb = H * 0.40; hh = H * 0.11 * op
        band = np.zeros((H, W, 3), np.float32)
        img = s.pose(t, sway=0.0, breath=0.0, stretch=s.wing + (0.0,))
        zoom = 2.5; ex = (s.eyeL[0] + s.eyeR[0]) / 2; ey = s.eyeL[1]; sc = zoom * H / s.ah * 0.5
        s.draw(band, img, W * 0.46 - u * 7 * k - (ex - s.aw / 2) * sc, cyb - (ey - s.ah / 2) * sc, zoom * 0.5)
        s.eyes(band, 0.75 + 0.25 * math.sin(u * 12), 0.6)
        m = s.L0(); cv2.fillPoly(m, [np.array([(0, cyb - hh + W * 0.06), (W, cyb - hh - W * 0.06), (W, cyb + hh - W * 0.06), (0, cyb + hh + W * 0.06)], np.int32)], 1.0, cv2.LINE_AA)
        back_ = (DEEP * 1.6 + GOLD * 0.25)[None, None, :] * (0.6 + 0.4 * s.XX[:, :, None] / W)
        f = f * (1 - m[:, :, None]) + (back_ * (1 - np.clip(band.sum(2, keepdims=True) * 6, 0, 1)) + band) * m[:, :, None]
        L = s.L0()
        s.line(L, (0, cyb - hh + W * 0.06), (W, cyb - hh - W * 0.06), 1.0, 3); s.line(L, (0, cyb + hh + W * 0.06), (W, cyb + hh - W * 0.06), 1.0, 3)
        for i in range(26):                                                         # 帯の中を 星が流れる
            r = s.dust[i]; x = ((r[0] + u * (0.5 + r[2])) % 1.2 - 0.1) * W; y = cyb + (r[1] - 0.5) * hh * 1.7 - (x / W - 0.5) * W * 0.12
            s.line(L, (x, y), (x - W * 0.09 * r[3], y + W * 0.011 * r[3]), 0.5, 1)
        s.add(f, L, GOLD, 0.9)
        for i, T in enumerate(s.vo):
            if u > 0.15 + i * 0.1: s.text(f, T, W * 0.13, H * 0.17 + i * 36 * k, ease((u - 0.15 - i * 0.1) / 0.2), 1.0)
        for i, T in enumerate(s.vt):                                                # 1字ずつ 星が はじけて出る
            a = (u - 0.45 - i * 0.17) / 0.16
            if a > 0:
                p = (W * 0.84, H * 0.56 + i * 76 * k); s.text(f, T, p[0], p[1], clamp(a), 1.0)
                if a < 2.2:
                    L2 = s.L0(); s.star4(L2, p, (70 * clamp(1 - a / 2.2) + 8) * k, clamp(1.4 - a / 1.6), a); s.add(f, L2, GOLD, 1.0)
        if u > 1.4: s.text(f, s.txt_en, W * 0.42, H * 0.585, ease((u - 1.4) / 0.3), 1.0)
        if u < 0.12: f += (1 - u / 0.12) * 0.8
        return f, 0, 0.7

    # 3) 杖から 光の矢 → 12の星 → 星の冠。そのあと 一点に ちぢむ
    def sc_crown(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_CROWN
        hush = ease2((t - T_HUSH) / 0.55)                                           # ちぢむ
        f = s.night(t, 0.9 * (1 - hush * 0.85), 0.15)
        O = (W / 2, H * 0.27); R = W * 0.36 * (1 - hush)
        raise_ = ease(u / 0.6)                                                      # 杖を かかげる(上体が そる)
        img = s.pose(t, sway=0.0, breath=0.3, lean=-10 * k * raise_, squash=-0.03 * raise_, stretch=s.wing + (0.0,))
        s.soft(f, (W / 2, H * 0.74), W * 0.34, 0.22, GOLD)
        s.draw(f, img, W / 2, H * 0.735, 0.47)
        sp = s.at(*s.staff)
        pts = s.crown_pts(t, O, R)
        L = s.L0(); Ls = s.L0(); n = len(pts); per = 0.2
        for i, p in enumerate(pts):
            a = (u - 0.5 - i * per) / 0.22                                          # i番目の矢
            if a <= 0: continue
            if a < 1 and not hush:                                                  # 飛んでいる矢(曲がって飛ぶ)
                for j in range(9):
                    q = clamp(a - j * 0.05)
                    mx = sp[0] + (p[0] - sp[0]) * q + math.sin(q * 3.14) * (60 if i % 2 else -60) * k; my = sp[1] + (p[1] - sp[1]) * q ** 0.8
                    cv2.circle(L, (int(mx), int(my)), max(1, int((6 - j * 0.55) * k)), float(1 - j / 9), -1, cv2.LINE_AA)
            else:
                tw = 1 + 0.25 * math.sin(t * 7 + i); pop = 1 + 1.6 * clamp(1 - (a - 1) / 0.8) if a >= 1 else 1
                s.star4(Ls, p, (17 if i % 2 == 0 else 11) * k * tw * pop * (1 - hush * 0.6), 1.0, t * 0.8 + i)
                if i > 0 and (u - 0.5 - (i - 1) * per) / 0.22 >= 1:                 # となりと 線でむすぶ
                    q = clamp((a - 1) / 0.5); p0 = pts[i - 1]; s.line(L, p0, (p0[0] + (p[0] - p0[0]) * q, p0[1] + (p[1] - p0[1]) * q), 0.9, 2)
        if (u - 0.5 - (n - 1) * per) / 0.22 >= 1:                                   # 冠が とじる
            q = clamp(((u - 0.5 - (n - 1) * per) / 0.22 - 1) / 0.5); p0 = pts[-1]; p = pts[0]
            s.line(L, p0, (p0[0] + (p[0] - p0[0]) * q, p0[1] + (p[1] - p0[1]) * q), 0.9, 2)
            if q >= 1:
                gl = clamp((u - 0.5 - n * per - 0.3) / 0.5)
                s.ell(L, (O[0], O[1] + R * 0.05), (R * 1.0, R * 0.42), 0, 0.6 * gl, 2); s.ell(L, (O[0], O[1] - R * 0.1), (R * 0.72, R * 0.30), 0, 0.4 * gl, 1)
                for i in range(0, n, 2):                                            # 冠の先から 光がのびる
                    p = pts[i]; s.line(L, p, (p[0], p[1] - (50 + 30 * math.sin(t * 5 + i)) * k * gl * (1 - hush)), 0.8, 1)
                s.soft(f, O, R * 0.9 + 4, 0.25 * gl, GOLD)
        s.add(f, L, GOLD, 1.0); s.add(f, Ls, WHITE * 0.6 + GOLD * 0.4, 1.3, 2, 8)
        L = s.L0(); s.star4(L, sp, (30 + 8 * math.sin(t * 11)) * k, 1.0, t)       # 杖の星
        s.add(f, L, GOLD, 1.1)
        if hush > 0:                                                                # しずけさ: 一点の光と 目だけ
            L = s.L0(); s.star4(L, O, (6 + 26 * (1 - hush) + 5 * math.sin(t * 30)) * k, 1.0, 0.0, 0.1); s.add(f, L, WHITE, 1.4, 2, 7)
            f *= (1 - 0.55 * hush); s.eyes(f, 0.9 * hush, 0.5)
        if u < 0.15: f += (1 - u / 0.15) * 0.8
        return f, 0, 0.8

    # 4) 創世: 一点から 銀河が生まれる
    def sc_bang(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_BANG
        O = (W / 2, H * 0.27)
        g = ease(u / 1.6)
        f = s.night(t, 1.0, 0.5)
        f += np.exp(-(((s.XX - O[0]) ** 2 + (s.YY - O[1]) ** 2) / (W * (0.25 + 0.5 * g)) ** 2))[:, :, None] * (GOLD * 0.5 + PALE * 0.2)[None, None, :] * (0.7 + 0.3 * math.sin(u * 20) * (1 - g))
        L = s.L0(); Lb = s.L0()
        rot = u * 1.1 + g * 2.0
        for i in range(900):                                                        # うずを巻く 銀河(4本の腕)
            r = s.gal[i]; rr = r[1] ** 0.7 * W * 0.95 * g
            a = int(r[0] * 4) * 1.5708 + rr / (W * 0.22) + rot + (r[2] - 0.5) * 0.7
            x = O[0] + math.cos(a) * rr; y = O[1] + math.sin(a) * rr * 0.55 + rr * 0.25
            if 0 <= x < W and 0 <= y < H:
                (L if i % 3 else Lb)[int(y), int(x)] += 0.6 + r[3]
        Lg = cv2.GaussianBlur(L, (0, 0), 1.6 * k) * 9 + cv2.GaussianBlur(L, (0, 0), 9 * k) * 30
        Lg2 = cv2.GaussianBlur(Lb, (0, 0), 1.6 * k) * 9 + cv2.GaussianBlur(Lb, (0, 0), 9 * k) * 30
        f += Lg[:, :, None] * GOLD[None, None, :] + Lg2[:, :, None] * PALE[None, None, :]
        L = s.L0()
        for i in range(4):                                                          # ひろがる輪
            v = (u - i * 0.22) / 1.3
            if 0 < v < 1: s.ell(L, O, (W * 1.5 * v, W * 0.9 * v), 0, (1 - v), 3)
        for i in range(16):                                                         # 光のすじ
            a = i * 0.3927 + u * 0.3; r0 = W * 0.05; r1 = W * (0.3 + 1.2 * clamp(u / 0.5)) * (0.5 + s.dust[i][0])
            s.line(L, (O[0] + math.cos(a) * r0, O[1] + math.sin(a) * r0), (O[0] + math.cos(a) * r1, O[1] + math.sin(a) * r1), 0.5 * clamp(1.5 - u / 1.2), 1)
        s.star4(L, O, (40 + 130 * clamp(u / 0.3) * clamp(1.6 - u / 1.5)) * k, 1.0, u * 0.7)
        s.add(f, L, WHITE * 0.5 + GOLD * 0.5, 1.1)
        # アルテア: 翼を いっぱいに ひろげて 浮かぶ。下から 光を受ける
        img = s.pose(t, sway=1.1 * math.exp(-u * 1.8), breath=0.3, lean=-6 * k, stretch=s.wing + (-0.16 * back(u / 0.5),), jit=2.0 * k * clamp(1 - u / 0.5))
        cy = H * (0.735 - 0.04 * ease(u / 2.0))
        s.soft(f, (W / 2, cy), W * 0.42, 0.3, GOLD)
        s.draw(f, img, W / 2, cy, 0.49 + 0.05 * ease(u / 2.0), add=0.10)
        s.eyes(f, 0.8, 0.8)
        sp = s.at(*s.staff); L = s.L0()
        s.star4(L, sp, 34 * k, 1.0, t * 1.5); s.line(L, sp, O, 0.5 + 0.3 * math.sin(u * 30), 2)     # 杖と 銀河の芯を つなぐ光
        for i in range(60):                                                         # 降ってくる 星くず
            r = s.dust[i + 40]; y = ((r[1] + u * (0.25 + 0.5 * r[2])) % 1.0) * H; x = r[0] * W
            s.line(L, (x, y), (x - 3 * k, y - (10 + 26 * r[3]) * k), 0.55 * clamp(u / 0.8), 1)
        s.add(f, L, GOLD, 0.9)
        if u < 0.35: f = f + (1 - u / 0.35) ** 1.5 * 1.6
        end = DUR - T_BANG
        if u > end - 1.0: v = clamp((u - (end - 1.0)) / 0.45); f = f * (1 - v) + v
        if u > end - 0.55: v = clamp((u - (end - 0.55)) / 0.5); f = np.ones_like(f) * (1 - v) ** 2
        return f, 14 * clamp(1 - u / 0.9), 1.0

    # ---- この子だけの曲: リディアの かがやく調べ。鐘の一音 → ハープの上昇 → 12の星ぶんの音階 → 無音 → 大合唱 ----
    def audio(s, path):
        a = Snd(DUR, 81); D4 = 293.66
        lyd = [0, 2, 4, 6, 7, 9, 11]
        def fq(d, o=0): return D4 * 2 ** ((lyd[d % 7] + 12 * (d // 7 + o)) / 12.0)
        a.a += (a.tone(D4 / 4) + 0.5 * a.tone(D4 / 4 * 1.5)) * 0.10 * np.clip(a.t / 1.5, 0, 1) * a.gate(0, T_HUSH, 0.3)       # 低い持続
        a.note(fq(0, 1), 0.3, 1.6, 0.22, "bell"); a.note(fq(4, 1), 0.3, 1.2, 0.10, "bell")                                    # 一番星
        for i in range(10): a.note(fq(i), 1.0 + i * 0.16, 0.5, 0.07, "bell")                                                   # 光が ひろがる
        a.boom(T_EYE, 58, 0.9); a.whoosh(T_EYE - 0.4, T_EYE, 0.35)
        for d in (0, 2, 4, 6, 8): a.note(fq(d), T_EYE, 1.4, 0.05, "pad", 0.05)
        a.boom(T_CUT, 48, 1.0)
        for i in range(5): a.note(fq(7 - i, 1), T_CUT + 0.45 + i * 0.17, 0.25, 0.14, "bell")                                   # 5文字
        for d in (0, 4): a.note(fq(d, -1), T_CUT, 1.8, 0.06, "pad", 0.2)
        st = 0.2                                                                                                                # 12の星 = 12の音が のぼる
        up = [0, 2, 4, 1, 3, 5, 4, 6, 8, 7, 9, 11]
        for i in range(12):
            tk = T_CROWN + 0.5 + i * st + 0.22
            a.note(fq(up[i]), tk, 0.6, 0.13, "bell"); a.note(fq(up[i], -1), tk, 0.25, 0.06, "fm"); a.hat(tk, 0.05)
            if i % 2 == 0: a.note(fq([0, 3, 4, 3, 0, 4][i // 2], -2), tk, 0.45, 0.16, "saw")
        for d in (0, 2, 4, 6): a.note(fq(d, 1), T_CROWN + 0.5 + 12 * st + 0.4, 1.0, 0.05, "pad", 0.1)
        a.note(fq(7, 2), T_HUSH + 0.25, 0.5, 0.16, "sine")                                                                      # しずけさ: 高い一音だけ
        a.boom(T_BANG, 42, 1.3, 0.7); a.boom(T_BANG + 0.22, 70, 0.7)
        a.a += a.lp(a.nz, 0.12) * a.env(T_BANG, 0.1, 1.3) * 0.4
        for j, d in enumerate((0, 2, 4, 6, 7, 9, 11, 14)): a.note(fq(d, -1 if j < 3 else 0), T_BANG + 0.05 + j * 0.05, 2.4, 0.055, "pad", 0.3)   # 大合唱
        mel = [7, 8, 9, 11, 9, 8, 11, 14]
        for i, d in enumerate(mel): a.note(fq(d), T_BANG + 0.5 + i * 0.3, 0.5, 0.11, "bell"); a.note(fq(d, 1), T_BANG + 0.5 + i * 0.3, 0.3, 0.05, "bell")
        for i in range(16): a.note(fq(i % 7 + 7, 1), T_BANG + 0.5 + i * 0.15, 0.1, 0.025, "sine")
        a.save(path)


if __name__ == "__main__":
    r = Altea()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
