"""エイオン(★8 永劫の調停者) 専用の必殺技「永劫ノ天秤」— 6つのポーズの絵を 場面ごとに使い分ける
(f1 横向き) 何もない白い地平の かなたに立つ。星ぜんぶが この子へ流れこむ → (f5 正面) ふり向いた瞬間、世界が 光と夜に 割れる →
(f5/f3/f4 の目) 三段のカットイン → (f0 立ち) 宇宙の天秤。左の皿に太陽、右の皿に黒い月。ゆれて、ゆれて →
(f2 目を閉じる) ぴたりと釣り合う。色も音も 消える → (f3 開眼) 太陽と月が重なって 金環になる →
(f4 指さす) 一本の線。指した先で 光と夜が うずを巻いて はじける → (f5 正面・大) 金環を背に、光と夜の翼
使い方: python3 hs_83.py 出力.mp4 / python3 hs_83.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from PIL import Image
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 16.4
GOLD = np.array([1.0, 0.82, 0.42], np.float32); VIO = np.array([0.56, 0.40, 1.0], np.float32); ICE = np.array([0.74, 0.86, 1.0], np.float32)
T_TURN, T_CUT, T_SCALE, T_STILL, T_OPEN, T_POINT, T_BURST, T_APEX = 2.5, 3.3, 5.7, 8.4, 10.1, 11.5, 12.0, 13.3
Q = 760.0 / 498.0
EYES = {0: ((245, 216), (305, 216)), 3: ((216, 242), (308, 242)), 4: ((193, 226), (265, 236)), 5: ((233, 198), (318, 198))}
HANDS2 = (242, 282); FINGER4 = (300, 252)


class Aion(Kit):
    def __init__(s, W=540):
        super().__init__(83, (0.74, 0.86, 1.0), "永劫ノ天秤", "Eternal Equilibrium", W)
        side = int(760 * s.k); s.fr = []
        for i in range(6):                                                           # 6ポーズを 同じ大きさの台紙で持つ
            im = Image.open('/home/claude/anim/cut/83_%d.png' % i).resize((side, side), Image.LANCZOS)
            a = np.asarray(im).astype(np.float32) / 255.0; p = s.pad
            s.fr.append(cv2.copyMakeBorder(a, p, p, p, p, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0)))
        s.ah = s.aw = side; s.ph, s.pw = s.fr[0].shape[:2]; s.gy, s.gx = np.mgrid[0:s.ph, 0:s.pw].astype(np.float32); s.cp = s.fr[0]
        s.vt = [s.text_img(c, hz.SERIF, int(72 * s.k), 3) for c in "永劫ノ天秤"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        rg = np.random.RandomState(83)
        s.st = rg.rand(260, 4).astype(np.float32); s.sp = rg.rand(700, 4).astype(np.float32)
        s.xn = s.XX / s.W; s.yn = s.YY / s.H

    def P(s, xy): return (xy[0] * Q * s.k, xy[1] * Q * s.k)

    # ---- この子だけの絵 ----
    def hero(s, f, i, t, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, **kw):
        s.cp = s.fr[i]; img = s.pose(t, **kw); s.draw(f, img, cx, cy, hfrac, alpha, white, add); s._i = i

    def gaze(s, f, v, col=ICE, streak=1.0):
        if v <= 0 or s._i not in EYES: return
        L = s.L0(); sc = s._pl[2]
        for e in EYES[s._i]:
            p = s.at(*s.P(e)); cv2.ellipse(L, (int(p[0]), int(p[1])), (max(1, int(13 * sc)), max(1, int(10 * sc))), 0, 0, 360, float(v), -1, cv2.LINE_AA)
            s.line(L, (p[0] - 170 * sc * streak * v, p[1]), (p[0] + 170 * sc * streak * v, p[1]), 0.7 * v, 1)
        s.add(f, L, col, 1.2, 2, 8)

    def twin(s, t, mix=1.0, lum=1.0, mono=0.0):
        """左は光、右は夜。mix=0 だと まっ白な 無の空"""
        off = int((t * 9) % 300); n = s.neb[:, off:off + s.W]
        b = np.clip((s.xn - 0.5) / 0.03 + 0.5, 0, 1)[:, :, None]
        col = (GOLD * 0.75)[None, None, :] * (1 - b) + (VIO * 0.62)[None, None, :] * b
        f = (0.14 + 0.75 * n ** 2)[:, :, None] * col * lum
        if mono: g = f.mean(2, keepdims=True); f = f * (1 - mono) + g * mono * 1.2
        if mix < 1:
            void = np.ones((s.H, s.W, 3), np.float32) * (0.90 - 0.25 * np.abs(s.yn - 0.62))[:, :, None] * (ICE * 0.2 + 0.8)[None, None, :]
            f = void * (1 - mix) + f * mix
        return f.astype(np.float32)

    def sun(s, f, c, r, t, v=1.0):
        L = s.L0(); cv2.circle(L, (int(c[0]), int(c[1])), max(1, int(r)), 1.0, -1, cv2.LINE_AA)
        for i in range(16):
            a = i * 0.3927 + t * 0.5; s.line(L, (c[0] + math.cos(a) * r * 1.25, c[1] + math.sin(a) * r * 1.25), (c[0] + math.cos(a) * r * (1.6 + 0.25 * (i % 2)), c[1] + math.sin(a) * r * (1.6 + 0.25 * (i % 2))), 0.9, 2)
        s.add(f, L, GOLD, 1.1 * v, 3, 14)

    def moon(s, f, c, r, t, v=1.0):
        L = s.L0(); cv2.circle(L, (int(c[0]), int(c[1])), max(1, int(r * 1.06)), 1.0, -1, cv2.LINE_AA)
        for i in range(30):
            a = i * 0.2094 - t * 0.3; r1 = r * (1.2 + 0.3 * (0.5 + 0.5 * math.sin(i * 2.3 + t * 2))); s.line(L, (c[0] + math.cos(a) * r, c[1] + math.sin(a) * r), (c[0] + math.cos(a) * r1, c[1] + math.sin(a) * r1), 0.6, 1)
        s.add(f, L, VIO, 1.0 * v, 3, 14)
        core = s.L0(); cv2.circle(core, (int(c[0]), int(c[1])), max(1, int(r)), 1.0, -1, cv2.LINE_AA); f *= (1 - core)[:, :, None]

    def scales(s, f, tilt, t, lit=1.0):
        """宇宙の天秤: さお・くさり・2つの皿。左に太陽、右に黒い月"""
        W, H, k = s.W, s.H, s.k; B = (W / 2, H * 0.17); hl = W * 0.36
        L = s.L0(); ca, sa = math.cos(tilt), math.sin(tilt); ends = [(B[0] - ca * hl, B[1] - sa * hl), (B[0] + ca * hl, B[1] + sa * hl)]
        s.line(L, ends[0], ends[1], 1.0, 4); s.line(L, (ends[0][0], ends[0][1] - 5 * k), (ends[1][0], ends[1][1] - 5 * k), 0.5, 1)
        s.line(L, B, (B[0], H * 0.50), 0.8, 3); s.star4(L, (B[0], B[1] - 26 * k), 26 * k, 1.0, 0.0, 0.2); s.circ(L, B, 10 * k, 1.0, 3)
        for i in range(-3, 4):                                                       # さおの 目もり
            p = (B[0] + ca * hl * i / 4.0, B[1] + sa * hl * i / 4.0); s.line(L, p, (p[0], p[1] + 9 * k), 0.6, 1)
        pans = []
        for e in ends:
            py = e[1] + H * 0.15; pw = W * 0.15
            for dx in (-pw, 0, pw): s.line(L, e, (e[0] + dx, py), 0.6, 1)
            s.ell(L, (e[0], py), (pw, pw * 0.22), 0, 1.0, 3, 0, 180); s.line(L, (e[0] - pw, py), (e[0] + pw, py), 0.8, 2); pans.append((e[0], py - W * 0.085))
        s.add(f, L, WHITE * 0.7 + GOLD * 0.3, 0.9 * lit)
        s.sun(f, pans[0], W * 0.060, t, lit); s.moon(f, pans[1], W * 0.060, t, lit)
        return pans

    def frame(s, t):
        k = s.k
        if t < T_CUT: f, sh, g = s.sc_void(t)
        elif t < T_SCALE: f, sh, g = s.sc_cut(t - T_CUT)
        elif t < T_OPEN: f, sh, g = s.sc_scale(t)
        elif t < T_APEX: f, sh, g = s.sc_verdict(t)
        else: f, sh, g = s.sc_apex(t - T_APEX)
        return s.post(f, ICE, g, sh * k, t, 0.97)

    # 1) 無の地平。星が 流れこむ。ふり向くと 世界が割れる
    def sc_void(s, t):
        W, H, k = s.W, s.H, s.k; turn = t - T_TURN
        mix = ease(turn / 0.25) if turn > 0 else 0.0
        f = s.twin(t, mix)
        hy = H * 0.66; a = ease2(t / T_TURN); size = 0.10 + 0.34 * a ** 1.6
        C = (W / 2, hy - size * H * 0.47)
        ink = 1 - mix                                                                # 無の空では 線は 黒
        L = s.L0(); s.line(L, (0, hy), (W, hy), 0.5, 1)
        for i in range(260):                                                         # 世界中の星が この子へ
            r = s.st[i]; v = (t * (0.25 + 0.5 * r[2]) + r[3]) % 1.0; ang = r[0] * 6.283; rr = (1 - v) ** 1.5 * W * 1.3
            p = (C[0] + math.cos(ang) * rr, C[1] + math.sin(ang) * rr * 1.3); q = (C[0] + math.cos(ang) * rr * 0.9, C[1] + math.sin(ang) * rr * 1.3 * 0.9)
            s.line(L, p, q, 0.55 * v, 1)
        if ink > 0: f = f * (1 - np.clip(L + cv2.GaussianBlur(L, (0, 0), 1.5), 0, 1) * 0.75 * ink)[:, :, None]
        if mix > 0: s.add(f, L, WHITE, 0.6 * mix)
        if turn < 0:
            sh = s.L0(); cv2.ellipse(sh, (int(W / 2), int(hy + 4 * k)), (int(size * H * 0.42), int(size * H * 0.04)), 0, 0, 360, 0.35, -1); f *= (1 - cv2.GaussianBlur(sh, (0, 0), 4 * k))[:, :, None]
            s.hero(f, 1, t, W / 2, C[1], size)
        else:
            s.hero(f, 5, t, W / 2, C[1], size + 0.02 * ease(turn / 0.5), squash=-0.04 * math.exp(-turn * 8))
            s.gaze(f, clamp(1.5 - turn), ICE, 1.6)
            L = s.L0(); s.line(L, (W / 2, 0), (W / 2, H), clamp(1.4 - turn * 2), 3)     # 世界を割る 一本の線
            for i in range(2):
                v = (turn - i * 0.12) / 0.7
                if 0 < v < 1: s.circ(L, C, W * (0.25 + 1.0 * v), 1 - v, 3)
            s.add(f, L, WHITE, 1.2)
            if turn < 0.1: f = 1 - np.clip(f, 0, 1) * (1 - turn / 0.1)               # 反転の一瞬
        return f * clamp(t / 0.5), (10 * math.exp(-turn * 6) if turn > 0 else 0), 0.5

    # 2) 三段のカットイン: 3つのポーズの「目」
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.twin(t, 1.0, 0.25)
        rows = [(5, H * 0.20, GOLD, -1), (3, H * 0.385, ICE, 1), (4, H * 0.57, VIO, -1)]; hh = H * 0.082
        for j, (fi, cy, col, dr) in enumerate(rows):
            a = ease((u - j * 0.14) / 0.26)
            if a <= 0: continue
            band = np.zeros((H, W, 3), np.float32) + (col * 0.30)[None, None, :]
            e = EYES[fi]; ex, ey = s.P(((e[0][0] + e[1][0]) / 2, e[0][1])); zf = 1.5; sc = zf * H / s.ah
            s.hero(band, fi, t, W * 0.5 - (ex - s.aw / 2) * sc + dr * (1 - a) * W + dr * u * 5 * k, cy - (ey - s.ah / 2) * sc, zf)
            s.gaze(band, 0.8, col, 0.5)
            m = ((np.abs(s.YY - cy) < hh)).astype(np.float32)[:, :, None]
            f = f * (1 - m) + band * m
            L = s.L0(); s.line(L, (0, cy - hh), (W, cy - hh), 1.0, 2); s.line(L, (0, cy + hh), (W, cy + hh), 1.0, 2); s.add(f, L, col, 0.9)
        n = len(s.vt); step = W * 0.165
        for i, T in enumerate(s.vt):                                                 # 技名: 両はしから まん中へ そろう(釣り合い)
            a = ease((u - 0.75 - abs(i - 2) * 0.12) / 0.3)
            if a > 0:
                x = W / 2 + (i - (n - 1) / 2) * step * (1 + (1 - a) * 0.45); s.text(f, T, min(W - 50 * k, max(50 * k, x)), H * 0.77, a, 1.0)
        if u > 0.75:
            L = s.L0(); g = ease((u - 0.75) / 0.6); s.line(L, (W / 2 - W * 0.45 * g, H * 0.77 + 54 * k), (W / 2 + W * 0.45 * g, H * 0.77 + 54 * k), 0.9, 2)
            s.star4(L, (W / 2, H * 0.77 + 54 * k), 14 * k * g, 1.0); s.add(f, L, WHITE, 0.9)
        for i, T in enumerate(s.vo):
            if u > 0.6 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.77 - 70 * k, 1.0, 1.0)
        if u > 1.5: s.text(f, s.txt_en, W / 2, H * 0.77 + 84 * k, ease((u - 1.5) / 0.3), 1.0)
        if u < 0.1: f += (1 - u / 0.1) * 0.8
        return f, 0, 0.5

    # 3) 宇宙の天秤: ゆれて → ぴたり(色も 消える)
    def sc_scale(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_SCALE; st = t - T_STILL
        if st < 0: tilt = 0.42 * math.exp(-u * 0.75) * math.cos(u * 4.2); mono = 0.0
        else: tilt = 0.42 * math.exp(-(T_STILL - T_SCALE) * 0.75) * math.cos((T_STILL - T_SCALE) * 4.2) * (1 - ease(st / 0.25)); mono = ease(st / 0.5)
        side = -tilt / 0.42                                                          # かたむいた側が 明るくなる
        f = s.twin(t, 1.0, 0.75, mono)
        f *= (1 + 0.5 * np.clip(side * (0.5 - s.xn) * 2, -0.6, 1))[:, :, None]
        g = ease(u / 0.6)
        s.scales(f, tilt, t if st < 0 else T_STILL, g)
        cy = H * 0.72
        if st < 0:
            s.hero(f, 0, t, W / 2, cy, 0.46); s.gaze(f, 0.35, ICE, 0.6)
        else:
            s.hero(f, 2, t, W / 2, cy, 0.46, squash=-0.03 * math.exp(-st * 7))
            hp = s.at(*s.P(HANDS2)); L = s.L0()                                      # 両手の間に 光が 集まる
            for i in range(40):
                r = s.st[i]; v = (st * (0.6 + r[2]) + r[3]) % 1.0; a = r[0] * 6.283; rr = (1 - v) * W * 0.5
                s.line(L, (hp[0] + math.cos(a) * rr, hp[1] + math.sin(a) * rr), (hp[0] + math.cos(a) * rr * 0.85, hp[1] + math.sin(a) * rr * 0.85), 0.6 * v, 1)
            s.star4(L, hp, (10 + 26 * clamp(st / 1.5)) * k, 1.0, st * 0.8); s.add(f, L, GOLD * (1 - mono * 0.5) + WHITE * mono * 0.5, 1.0)
            if st < 0.1: f += (1 - st / 0.1) * 0.5
        if u < 0.2: f = f * (u / 0.2) + (1 - u / 0.2)
        return f, (3 * abs(math.sin(u * 4.2)) * math.exp(-u * 0.9) if st < 0 else 0), 0.5

    # 4) 開眼 → 太陽と月が重なり 金環 → 指さす → 光と夜の うず
    def sc_verdict(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_OPEN; pt = t - T_POINT; br = t - T_BURST
        f = s.twin(t, 1.0, 0.85 + 0.6 * math.exp(-u * 3))
        O = (W / 2, H * 0.25); m = ease2(u / 1.1); R = W * 0.11
        pl = (W / 2 - W * 0.36 * (1 - m), H * 0.17 + H * 0.15 - W * 0.085 + (O[1] - (H * 0.32 - W * 0.085)) * m)
        if m < 1:                                                                    # 2つが 寄っていく
            s.sun(f, pl, R * (0.7 + 0.5 * m), t); s.moon(f, (W - pl[0], pl[1]), R * (0.7 + 0.5 * m), t)
        else:                                                                        # 金環
            v = u - 1.1; rr = R * 1.25 * (1 + 0.03 * math.sin(t * 20))
            L = s.L0(); s.circ(L, O, rr, 1.0, int(7 * k))
            for i in range(24):
                a = i * 0.2618 + t * 0.4; s.line(L, (O[0] + math.cos(a) * rr * 1.15, O[1] + math.sin(a) * rr * 1.15), (O[0] + math.cos(a) * rr * (1.5 + 0.5 * (i % 3 == 0)), O[1] + math.sin(a) * rr * (1.5 + 0.5 * (i % 3 == 0))), 0.8, 1)
            for i in range(3):
                q = (v - i * 0.12) / 0.8
                if 0 < q < 1: s.circ(L, O, rr + W * 1.1 * q, 1 - q, 3)
            s.add(f, L, GOLD * 0.5 + VIO * 0.25 + 0.25, 1.4, 3, 16)
            core = s.L0(); cv2.circle(core, (int(O[0]), int(O[1])), int(rr - 4 * k), 1.0, -1, cv2.LINE_AA); f *= (1 - core)[:, :, None]
            if v < 0.1: f += (1 - v / 0.1) * 1.0
        cy = H * 0.72
        if pt < 0:
            s.hero(f, 3, t, W / 2, cy, 0.47, sway=0.8 * math.exp(-u * 3), squash=-0.05 * math.exp(-u * 7)); s.gaze(f, 1.0, ICE, 1.2)
            if u < 0.12: f += (1 - u / 0.12) * 0.9
        else:
            s.hero(f, 4, t, W / 2, cy, 0.47, squash=-0.04 * math.exp(-pt * 9)); s.gaze(f, 0.9, ICE, 0.8)
            fp = s.at(*s.P(FINGER4)); tg = (W * 0.86, H * 0.10); L = s.L0()
            g = ease(pt / 0.18); s.line(L, fp, (fp[0] + (tg[0] - fp[0]) * g * 1.6, fp[1] + (tg[1] - fp[1]) * g * 1.6), 1.0, 2)     # 指先から 一本の線
            s.star4(L, fp, 12 * k, 1.0, t * 3); s.add(f, L, WHITE, 1.3)
            if br > 0:                                                               # 指した先で 光と夜が うずを巻く
                g2 = ease(br / 1.0); La = s.L0(); Lb = s.L0(); rot = br * 3.2
                for i in range(700):
                    r = s.sp[i]; rr = r[1] ** 0.6 * W * 1.5 * g2; a = (i % 2) * 3.1416 + rr / (W * 0.20) - rot + (r[2] - 0.5) * 0.5
                    x = tg[0] + math.cos(a) * rr; y = tg[1] + math.sin(a) * rr
                    if 0 <= x < W and 0 <= y < H: (La if i % 2 else Lb)[int(y), int(x)] += 0.7 + r[3]
                f += (cv2.GaussianBlur(La, (0, 0), 2 * k) * 22 + cv2.GaussianBlur(La, (0, 0), 12 * k) * 90)[:, :, None] * GOLD[None, None, :]
                f += (cv2.GaussianBlur(Lb, (0, 0), 2 * k) * 22 + cv2.GaussianBlur(Lb, (0, 0), 12 * k) * 90)[:, :, None] * VIO[None, None, :]
                L = s.L0()
                for i in range(3):
                    q = (br - i * 0.1) / 0.8
                    if 0 < q < 1: s.circ(L, tg, W * 1.6 * q, 1 - q, 4)
                s.add(f, L, WHITE, 1.2)
                if br < 0.1: f += (1 - br / 0.1) * 1.2
                wo = clamp((br - 0.85) / 0.4)
                if wo > 0: f = f * (1 - wo) + wo
        return f, 8 * math.exp(-u * 5) + (18 * math.exp(-br * 3) if br > 0 else 0), 0.6

    # 5) 頂点: 金環を背に、光と夜の翼
    def sc_apex(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_APEX
        f = s.twin(t, 1.0, 0.55)
        C = (W / 2, H * 0.44); g = ease(u / 0.7)
        L1 = s.L0(); L2 = s.L0()
        for i in range(11):                                                          # 光の翼(左)と 夜の翼(右): 羽が 1枚ずつ ひらく
            a = ease((u - 0.15 - i * 0.05) / 0.5)
            if a <= 0: continue
            ang = math.radians(-70 + i * 13); ln = W * (0.95 - 0.045 * abs(i - 4)) * a; wd = 0.075
            for sg, L in ((-1, L1), (1, L2)):
                d = (sg * math.cos(ang), math.sin(ang)); b = C
                cv2.fillPoly(L, [np.array([b, (b[0] + (d[0] * math.cos(wd) - d[1] * math.sin(wd) * sg) * ln, b[1] + (d[1] * math.cos(wd) + d[0] * math.sin(wd) * sg) * ln),
                                           (b[0] + d[0] * ln * 1.08, b[1] + d[1] * ln * 1.08),
                                           (b[0] + (d[0] * math.cos(wd) + d[1] * math.sin(wd) * sg) * ln, b[1] + (d[1] * math.cos(wd) - d[0] * math.sin(wd) * sg) * ln)], np.int32)], 0.42, cv2.LINE_AA)
        f += (L1 + cv2.GaussianBlur(L1, (0, 0), 8 * k))[:, :, None] * GOLD[None, None, :] * 0.8 + (L2 + cv2.GaussianBlur(L2, (0, 0), 8 * k))[:, :, None] * VIO[None, None, :] * 0.8
        rr = W * 0.34 * g; L = s.L0(); s.circ(L, C, rr, 1.0, int(6 * k)); s.circ(L, C, rr * 1.12, 0.5, 1)
        for i in range(36):
            a = i * 0.1745 - t * 0.15; s.line(L, (C[0] + math.cos(a) * rr * 1.12, C[1] + math.sin(a) * rr * 1.12), (C[0] + math.cos(a) * rr * (1.2 + 0.1 * (i % 3 == 0)), C[1] + math.sin(a) * rr * (1.2 + 0.1 * (i % 3 == 0))), 0.7, 1)
        s.add(f, L, WHITE * 0.6 + GOLD * 0.4, 1.1, 3, 14)
        s.hero(f, 5, t, W / 2, H * 0.50, 0.40 + 0.20 * g, squash=-0.03 * math.exp(-u * 6))
        s.gaze(f, 0.5 + 0.5 * math.exp(-u * 2), ICE, 1.0)
        L = s.L0()
        for i in range(80):                                                          # しずかに 降る 光と夜のかけら
            r = s.st[i]; y = ((r[1] + u * (0.06 + 0.12 * r[2])) % 1.0) * H; s.star4(L, (r[0] * W, y), (2 + 4 * r[3]) * k, 0.7, r[2] * 3)
        f += L[:, :, None] * np.where(s.xn[:, :, None] < 0.5, GOLD[None, None, :], VIO[None, None, :]) * 0.9
        if u < 0.5: f = f * (u / 0.5) + (1 - u / 0.5)
        end = DUR - t
        if end < 0.7: f = f * clamp(end / 0.65) ** 1.5
        return f, 0, 0.5

    # ---- この子だけの曲: 光の調べ(リディア)と 夜の調べ(フリギア)が 同じ主音の上で 交互に鳴り、釣り合った所で 無音。開眼で 両方が 同時に鳴る ----
    def audio(s, path):
        a = Snd(DUR, 83); D = 146.83; lyd = [0, 2, 4, 6, 7, 9, 11]; phr = [0, 1, 3, 5, 7, 8, 10]
        def fl(d, o=0): return D * 2 ** ((lyd[d % 7] + 12 * (d // 7 + o)) / 12.0)
        def fp(d, o=0): return D * 2 ** ((phr[d % 7] + 12 * (d // 7 + o)) / 12.0)
        a.a += a.tone(D / 2) * 0.08 * np.clip(a.t / 2, 0, 1) * a.gate(0, T_STILL, 0.3)
        a.rise(0.3, T_TURN, 1200, 200, 0.05); a.whoosh(0.5, T_TURN, 0.35, 0.05)                                    # 星が 吸いこまれる
        a.boom(T_TURN, 40, 1.3, 0.7); a.note(fl(7, 1), T_TURN, 1.4, 0.12, "bell"); a.note(fp(7, 0), T_TURN, 1.4, 0.10, "bell")
        for j in range(3): a.boom(T_CUT + j * 0.14, 60 - j * 6, 0.8, 0.2); a.note([fl(4, 1), fl(7, 1), fp(8, 1)][j], T_CUT + j * 0.14, 0.5, 0.10, "bell")
        for i in range(5): a.note(fl([0, 4, 7, 4, 0][i], 1), T_CUT + 0.75 + abs(i - 2) * 0.12 + i * 0.001, 0.4, 0.09, "bell")
        for d in (0, 4): a.note(fl(d, -1), T_CUT + 0.75, 1.6, 0.06, "pad", 0.2)
        k = 0; per = math.pi / 4.2                                                                                   # 天秤が かたむくたび 光と夜が 交互に
        while T_SCALE + k * per < T_STILL - 0.05:
            tk = T_SCALE + k * per; g = math.exp(-k * per * 0.55)
            a.a += (a.nz - a.lp(a.nz, 0.4)) * a.env(tk, 0.001, 0.03) * 0.25 * g                                      # かちり
            if k % 2 == 0:
                for j, d in enumerate((0, 2, 4, 6)): a.note(fl(d + (k // 2) % 3, 0), tk + j * 0.06, 0.5, 0.075 * (0.5 + g), "bell")
            else:
                for j, d in enumerate((7, 5, 3, 1)): a.note(fp(d, -1), tk + j * 0.06, 0.5, 0.085 * (0.5 + g), "fm")
            a.boom(tk, 46, 0.35 * g, 0.2); k += 1
        a.note(fl(14, 1), T_STILL + 0.1, 0.9, 0.07, "sine")                                                          # 釣り合い: 細い一音だけ
        a.rise(T_STILL + 0.5, T_OPEN, 100, 180, 0.07)
        a.boom(T_OPEN, 38, 1.5, 0.9)                                                                                 # 開眼: 両方の和音が 同時に
        for d in (0, 2, 4, 6, 7): a.note(fl(d, 0), T_OPEN, 1.8, 0.045, "pad", 0.05)
        for d in (0, 3, 4): a.note(fp(d, -1), T_OPEN, 1.8, 0.055, "saw", 0.05)
        a.rise(T_OPEN, T_OPEN + 1.1, 200, 800, 0.10); a.boom(T_OPEN + 1.1, 55, 1.0, 0.4); a.note(fl(11, 1), T_OPEN + 1.1, 1.2, 0.12, "bell")
        a.note(fl(14, 2), T_POINT, 0.4, 0.10, "sine"); a.whoosh(T_POINT, T_BURST, 0.4, 0.4)
        a.boom(T_BURST, 34, 1.6, 1.0); a.a += a.lp(a.nz, 0.1) * a.env(T_BURST, 0.05, 1.0) * 0.5
        for i in range(10): a.note(fl(i + 3, 0), T_BURST + i * 0.09, 0.3, 0.07, "bell"); a.note(fp(13 - i, 0), T_BURST + i * 0.09 + 0.045, 0.3, 0.07, "fm")
        for j, (d, o) in enumerate(((0, -2), (0, -1), (4, -1), (0, 0), (2, 0), (4, 0), (6, 0), (7, 0), (11, 0))): a.note(fl(d, o), T_APEX + j * 0.04, 2.6, 0.045, "pad", 0.4)   # 頂点: 大きな和音
        mel = [7, 9, 11, 14, 11, 14, 16, 18]
        for i, d in enumerate(mel): a.note(fl(d, 0), T_APEX + 0.3 + i * 0.3, 0.6, 0.10, "bell")
        a.boom(T_APEX, 36, 1.0, 0.8)
        a.save(path)


if __name__ == "__main__":
    r = Aion()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
