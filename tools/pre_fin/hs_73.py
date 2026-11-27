"""ノア(★6 方舟の旅人) 専用の必殺技「方舟ノ大洪水」
雨の夜。ランタンの明かりだけが この子を照らす → 肩のハトが 飛び立つ → 木の板が 上下から閉じる カットイン(すき間に目) →
水かさが 画面の下から 上がってくる。小さな舟が 大きな方舟になり、窓に 明かりが 2つずつ ともる →
右から 山のような大波 → くだけて まっ白 → なぎ。朝の空に 虹が かかり、ハトが 葉をくわえて 帰ってくる
使い方: python3 hs_73.py 出力.mp4 / python3 hs_73.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 13.6
TEAL = np.array([0.10, 0.42, 0.50], np.float32); DEEP = np.array([0.03, 0.10, 0.16], np.float32); LAMP = np.array([1.0, 0.78, 0.36], np.float32); WOOD = np.array([0.42, 0.26, 0.13], np.float32); FOAM = np.array([0.86, 0.96, 1.0], np.float32)
T_DOVE, T_CUT, T_FLOOD, T_WAVE, T_CRASH, T_CALM = 2.3, 3.1, 5.2, 9.3, 10.55, 10.95
BOW = [(1.0, 0.25, 0.25), (1.0, 0.55, 0.2), (1.0, 0.88, 0.3), (0.4, 0.85, 0.4), (0.3, 0.6, 1.0), (0.35, 0.35, 0.9), (0.6, 0.35, 0.85)]


class Noah(Kit):
    def __init__(s, W=540):
        super().__init__(73, (0.30, 0.78, 0.86), "方舟ノ大洪水", "Ark Deluge", W)
        s.lamp = s.A(48, 240); s.eyeL = s.A(291, 348); s.eyeR = s.A(420, 346); s.dove0 = s.A(535, 392)
        s.ht = [s.text_img(c, hz.SERIF, int(66 * s.k), 3) for c in "方舟ノ大洪水"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        rg = np.random.RandomState(73); s.rn = rg.rand(200, 4).astype(np.float32)
        ay, ax = np.mgrid[0:s.ph, 0:s.pw].astype(np.float32); s.dl = np.sqrt((ax - s.pad - s.lamp[0]) ** 2 + (ay - s.pad - s.lamp[1]) ** 2)
        g = rg.rand(s.H // 6 + 1, 8).astype(np.float32); s.grain = cv2.resize(cv2.GaussianBlur(g, (0, 0), 1.0), (s.W, s.H), interpolation=cv2.INTER_CUBIC)   # 木目(横に流れるすじ)
        s.xr = s.XX[0]

    # ---- この子だけの絵 ----
    def night(s, t, lum=1.0, dawn=0.0):
        y = s.YY / s.H
        f = (DEEP[None, None, :] * (0.6 + 1.6 * y)[:, :, None] + TEAL[None, None, :] * (0.25 * y ** 2)[:, :, None]) * lum * (1 - dawn)
        if dawn > 0: f += dawn * (np.array([1.0, 0.80, 0.62], np.float32)[None, None, :] * (y ** 1.5)[:, :, None] + np.array([0.45, 0.72, 0.95], np.float32)[None, None, :] * (1 - y ** 1.5)[:, :, None])
        return f.astype(np.float32)

    def rain(s, f, t, amt=1.0):
        L = s.L0(); n = int(120 * amt)
        for i in range(n):
            r = s.rn[i % 200]; v = (t * (1.6 + r[2]) + r[1] + i * 0.37) % 1.0; x = ((r[0] + i * 0.618) % 1.0) * s.W * 1.2 - v * s.W * 0.12; y = v * s.H * 1.1 - 30
            s.line(L, (x, y), (x - 5 * s.k, y + (22 + 26 * r[3]) * s.k), 0.30 + 0.25 * r[3], 1)
        f += L[:, :, None] * (FOAM * 0.6)[None, None, :]

    def surf(s, level, t, amp, ph=0.0, sp=1.0):
        x = s.xr / s.W
        return level + amp * (np.sin(x * 9 + t * 1.7 * sp + ph) * 0.6 + np.sin(x * 21 - t * 2.6 * sp + ph * 2) * 0.25 + np.sin(x * 4 + t * 0.9 * sp + ph * 3) * 0.5)

    def water(s, f, level, t, amp, lum=1.0, calm=0.0):
        """3枚の波を 奥から重ねる"""
        for j, (dy, c, ph, sp) in enumerate(((-16, 0.55, 2.0, 0.7), (0, 0.8, 0.0, 1.0), (20, 1.0, 4.0, 1.25))):
            ys = s.surf(level + dy * s.k, t, amp * (0.7 + 0.2 * j), ph, sp)
            m = np.clip((s.YY - ys[None, :]) / 2.0, 0, 1)
            depth = np.clip((s.YY - ys[None, :]) / (s.H * 0.5), 0, 1)
            col = (TEAL * c * (1 - calm) + np.array([0.45, 0.70, 0.85], np.float32) * calm * (0.7 + 0.1 * j))[None, None, :] * (1 - 0.6 * depth)[:, :, None] * lum
            f[:] = f * (1 - m[:, :, None] * 0.93) + col * m[:, :, None] * 0.93
            edge = np.clip(1 - np.abs(s.YY - ys[None, :]) / (2.2 * s.k), 0, 1)
            f += edge[:, :, None] * FOAM[None, None, :] * (0.25 + 0.2 * j) * lum
        return s.surf(level, t, amp, 0.0, 1.0)

    def dove(s, L, p, t, size, d=1):
        """白いハト: からだ・しっぽ・はばたく 2枚の羽"""
        k = size; fl = math.sin(t * 16)
        cv2.ellipse(L, (int(p[0]), int(p[1])), (int(13 * k), int(7 * k)), -8 * d, 0, 360, 1.0, -1, cv2.LINE_AA)
        cv2.circle(L, (int(p[0] + d * 12 * k), int(p[1] - 5 * k)), int(5 * k), 1.0, -1, cv2.LINE_AA)
        cv2.fillPoly(L, [np.array([(p[0] - d * 10 * k, p[1]), (p[0] - d * 26 * k, p[1] - 5 * k), (p[0] - d * 24 * k, p[1] + 6 * k)], np.int32)], 1.0, cv2.LINE_AA)
        for sg in (1, 0.6):
            cv2.fillPoly(L, [np.array([(p[0] - 6 * k, p[1] - 2 * k), (p[0] + 6 * k, p[1] - 2 * k), (p[0] - d * 8 * k * sg, p[1] - (26 * fl + 4) * k * sg)], np.int32)], float(sg), cv2.LINE_AA)

    def ark(s, f, cx, wy, size, t, lit=0, glow=1.0):
        """方舟: 丸い船底・箱形の家・屋根。窓は 2つずつ ともる"""
        k = s.k * size; D = s.L0(); L = s.L0(); Wn = s.L0()
        hull = [(cx - 190 * k, wy - 46 * k), (cx + 190 * k, wy - 46 * k), (cx + 150 * k, wy + 30 * k), (cx - 150 * k, wy + 30 * k)]
        house = [(cx - 130 * k, wy - 46 * k), (cx - 130 * k, wy - 118 * k), (cx + 130 * k, wy - 118 * k), (cx + 130 * k, wy - 46 * k)]
        roof = [(cx - 150 * k, wy - 118 * k), (cx, wy - 170 * k), (cx + 150 * k, wy - 118 * k)]
        for poly in (hull, house, roof): cv2.fillPoly(D, [np.array(poly, np.int32)], 1.0, cv2.LINE_AA); cv2.polylines(L, [np.array(poly, np.int32)], True, 0.8, 2, cv2.LINE_AA)
        for i in range(1, 5): s.line(L, (cx - 190 * k + i * 6 * k, wy - 46 * k + i * 15 * k), (cx + 190 * k - i * 6 * k, wy - 46 * k + i * 15 * k), 0.35, 1)     # 板の すじ
        f[:] = f * (1 - D * 0.95)[:, :, None] + D[:, :, None] * (WOOD * 0.55)[None, None, :]
        for j in range(8):                                                           # 窓(2つで1組)
            if j < lit:
                for q in (-1, 1):
                    wx = cx + (j - 3.5) * 31 * k + q * 6 * k; wyy = wy - (96 if j % 2 else 70) * k
                    cv2.rectangle(Wn, (int(wx - 4 * k), int(wyy - 6 * k)), (int(wx + 4 * k), int(wyy + 6 * k)), 1.0, -1)
        s.add(f, L, WOOD * 1.6 + LAMP * 0.3, 0.6 * glow, 2, 6); s.add(f, Wn, LAMP, 0.9 * glow, 2, 8)

    def lit(s, R, base=0.22):
        return np.clip(base + np.exp(-(s.dl / R) ** 2) * 1.1, 0, 1.0)

    def lantern(s, f, t, v=1.0):
        p = s.at(*s.lamp); fl = 1 + 0.10 * math.sin(t * 13) + 0.06 * math.sin(t * 29)
        s.soft(f, p, 34 * s.k * fl * (s._pl[2] / 0.55), 0.55 * v, LAMP); s.soft(f, p, 120 * s.k * (s._pl[2] / 0.55), 0.20 * v * fl, LAMP)
        return p

    def frame(s, t):
        k = s.k
        if t < T_CUT: f, sh = s.sc_rain(t)
        elif t < T_FLOOD: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_WAVE: f, sh = s.sc_flood(t)
        else: f, sh = s.sc_wave(t)
        return s.post(f, FOAM, 0.5, sh * k, t, 0.90)

    # 1) 雨の夜。ランタンの明かり。ハトが 飛び立つ
    def sc_rain(s, t):
        W, H, k = s.W, s.H, s.k
        f = s.night(t, clamp(t / 0.5))
        R = (60 + 520 * ease2((t - 0.4) / 1.6)) * k
        img = s.pose(t, breath=0.25)
        s.draw(f, img, W / 2, H * 0.56, 0.45, light=s.lit(R))
        s.lantern(f, t, ease(t / 0.6))
        s.water(f, H * 0.84, t, 5 * k)
        L = s.L0()
        for i in range(10):                                                          # 水面の 雨の輪
            r = s.rn[i + 50]; v = (t * 0.9 + r[0]) % 1.0; s.ell(L, (r[1] * W, H * (0.87 + 0.11 * r[2])), (40 * k * v, 8 * k * v), 0, 0.5 * (1 - v), 1)
        f += L[:, :, None] * FOAM[None, None, :] * 0.6
        s.rain(f, t, 1.0)
        fly = t - T_DOVE
        if fly > 0:                                                                  # ハト: 肩から 右上へ
            p0 = s.at(*s.dove0); a = ease2(fly / 0.8); p = (p0[0] + W * 0.55 * a, p0[1] - H * 0.52 * a - math.sin(a * 3.14) * H * 0.05)
            L = s.L0(); s.dove(L, p, t, 1.5 * k + 0.6 * k * a); s.add(f, L, WHITE, 1.0, 1.5, 6)
            for i in range(5):                                                       # 抜けた羽
                v = fly * (0.5 + 0.2 * i); q = (p0[0] + (i - 2) * 14 * k + math.sin(fly * 4 + i) * 12 * k, p0[1] - 20 * k + v * 70 * k)
                cv2.ellipse(f, (int(q[0]), int(q[1])), (int(7 * k), int(2 * k)), 30 * i + fly * 90, 0, 360, (0.9, 0.9, 0.9), -1, cv2.LINE_AA)
        return f, 0

    # 2) 木の板が 上下から閉じる。すき間に 目
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        mid = np.zeros((H, W, 3), np.float32) + (DEEP * 2.0 + LAMP * 0.10)[None, None, :]
        ex = (s.eyeL[0] + s.eyeR[0]) / 2; zf = 1.35; sc = zf * H / s.ah; cyb = H * 0.42
        img = s.pose(t); s.draw(mid, img, W * 0.5 - (ex - s.aw / 2) * sc + u * 6 * k, cyb - (s.eyeL[1] - s.ah / 2) * sc, zf, light=s.lit(380 * k, 0.45))
        L = s.L0()
        for e in (s.eyeL, s.eyeR):                                                   # ひとみに ランタンの灯が うつる
            p = s.at(*e); cv2.circle(L, (int(p[0] - 9 * sc), int(p[1] - 10 * sc)), max(2, int(6 * sc)), 1.0, -1, cv2.LINE_AA)
        s.add(mid, L, LAMP, 1.0, 2, 7)
        a = back(u / 0.35, 1.0); gap = H * 0.085
        top = cyb - gap - (1 - a) * H * 0.5; bot = cyb + gap + (1 - a) * H * 0.5
        plank = (WOOD[None, None, :] * (0.55 + 0.7 * s.grain)[:, :, None]).astype(np.float32)
        for j in range(1, 9): plank[int(j * H / 9) - 2:int(j * H / 9) + 1, :, :] *= 0.45         # 板の つぎ目
        m = ((s.YY < top) | (s.YY > bot)).astype(np.float32)[:, :, None]
        f = mid * (1 - m) + plank * m
        L = s.L0(); s.line(L, (0, top), (W, top), 1.0, 3); s.line(L, (0, bot), (W, bot), 1.0, 3); s.add(f, L, LAMP, 0.6)
        n = len(s.ht); step = W * 0.148
        for i, T in enumerate(s.ht):                                                 # 下の板に しずくが落ちて 1字ずつ にじみ出る
            q = (u - 0.5 - i * 0.14) / 0.22
            if q > 0:
                x = W / 2 + (i - (n - 1) / 2) * step; y = H * 0.70; s.text(f, T, x, y, clamp(q), 1.0)
                if q < 2.5:
                    L2 = s.L0(); s.ell(L2, (x, y + 30 * k), (46 * k * q / 2.5, 12 * k * q / 2.5), 0, clamp(1 - q / 2.5), 2); s.add(f, L2, FOAM, 0.7)
        for i, T in enumerate(s.vo):
            if u > 0.4 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.20, 1.0, 1.0)
        if u > 1.5: s.text(f, s.txt_en, W / 2, H * 0.70 + 62 * k, ease((u - 1.5) / 0.3), 1.0)
        if u < 0.35:
            pass
        return f, (5 * math.exp(-(u - 0.3) * 12) if u > 0.3 else 0)

    # 3) 水かさが 上がる。舟が 方舟になる
    def sc_flood(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_FLOOD
        f = s.night(t, 0.9)
        rise = ease2(u / 2.8); level = H * (0.90 - 0.36 * rise); amp = (6 + 16 * rise) * k
        ys = s.surf(level, t, amp); wy = float(ys[W // 2])
        grow = ease(u / 1.4); size = 0.35 + 0.75 * grow
        s.ark(f, W / 2, wy + 8 * k, size, t, int(clamp((u - 1.5) / 2.2) * 8.99), 1.0)
        img = s.pose(t)
        cyc = wy - (118 + 170) * 0.5 * k * size - H * 0.085
        s.draw(f, img, W / 2, cyc, 0.22 + 0.04 * grow, light=s.lit(420 * k, 0.5)); s.lantern(f, t, 1.0)
        s.water(f, level + 26 * k, t, amp)
        s.rain(f, t, 1.4)
        if u < 0.2: f = f * (u / 0.2)
        return f, 1.5 * rise

    # 4) 大波 → くだける → なぎ と 虹
    def sc_wave(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_WAVE; cr = t - T_CRASH; cm = t - T_CALM
        if cm < 0:
            f = s.night(t, 0.9)
            level = H * 0.54; amp = 22 * k * (1 + u); ys = s.surf(level, t, amp); wy = float(ys[W // 2])
            s.ark(f, W / 2, wy + 8 * k, 1.1, t, 8, 1.0)
            img = s.pose(t); s.draw(f, img, W / 2, wy - 144 * k * 1.1 - H * 0.085, 0.26, light=s.lit(420 * k, 0.5)); s.lantern(f, t, 1.0)
            s.water(f, level + 26 * k, t, amp)
            g = ease2(u / 1.25)                                                      # 右から せり上がる 大波
            x0 = W * (1.25 - 1.05 * g); top = H * (0.62 - 0.60 * g)
            body = [(W + 40, H), (W + 40, top), (x0 + W * 0.25, top - H * 0.02), (x0 + W * 0.04, top + H * 0.05), (x0 - W * 0.10, top + H * 0.16), (x0 - W * 0.06, top + H * 0.24),
                    (x0 + W * 0.02, top + H * 0.19), (x0 + W * 0.08, top + H * 0.30), (x0 + W * 0.02, H * 0.75), (x0 - W * 0.05, H)]
            D = s.L0(); cv2.fillPoly(D, [np.array(body, np.int32)], 1.0, cv2.LINE_AA)
            shade = (TEAL[None, None, :] * (0.55 + 0.8 * (1 - s.YY / H))[:, :, None]).astype(np.float32)
            f = f * (1 - D[:, :, None]) + shade * D[:, :, None]
            L = s.L0(); cv2.polylines(L, [np.array(body[1:8], np.int32)], False, 1.0, int(5 * k), cv2.LINE_AA)
            for i in range(40):                                                      # 波がしらの しぶき
                r = s.rn[i]; v = (u * (1 + r[2]) + r[0]) % 1.0; p = (x0 - W * 0.10 - v * W * 0.25 * r[1], top + H * 0.16 + (v * v * 0.25 - 0.06 * v) * H * (0.5 + r[3]))
                cv2.circle(L, (int(p[0]), int(p[1])), max(1, int(4 * k * (1 - v))), float(1 - v), -1, cv2.LINE_AA)
            for j in range(6): s.line(L, (W + 20, top + H * 0.10 + j * H * 0.07), (x0 + W * 0.12, top + H * 0.14 + j * H * 0.07), 0.25, 1)
            s.add(f, L, FOAM, 0.9, 3, 10)
            s.rain(f, t, 1.6)
            if cr > 0: f = f * (1 - ease(cr / 0.3)) + ease(cr / 0.3) * (FOAM * 0.2 + 0.8)[None, None, :]     # くだけて まっ白
            return f, 4 * g + (22 * math.exp(-cr * 5) if cr > 0 else 0)
        dawn = ease2(cm / 1.2)
        f = s.night(t, 1.0, dawn)
        C = (W / 2, H * 0.60); L = None                                              # 虹
        arc = ease2((cm - 0.5) / 1.3)
        if arc > 0:
            for j, c in enumerate(BOW):
                Lr = s.L0(); rr = W * (0.60 - j * 0.022)
                cv2.ellipse(Lr, (int(C[0]), int(C[1])), (int(rr), int(rr * 0.82)), 0, 180, 180 + 180 * arc, 1.0, int(12 * k), cv2.LINE_AA)
                al = cv2.GaussianBlur(Lr, (0, 0), 1.5 * k)[:, :, None] * 0.8; f[:] = f * (1 - al) + np.array(c, np.float32)[None, None, :] * al
        level = H * 0.60; ys = s.water(f, level, t * 0.4, 3 * k, 1.0, dawn)
        L = s.L0()
        for i in range(50):                                                          # 水面の きらめき
            r = s.rn[i]; tw = math.sin(t * 3 + r[2] * 20) ** 2; s.star4(L, (r[0] * W, level + 10 * k + r[1] * H * 0.36), (2 + 5 * r[3]) * k * tw, 0.8, 0.0, 0.2)
        f += L[:, :, None] * (WHITE * 0.9)[None, None, :] * dawn
        rail = H * 0.86                                                              # 手前: 方舟の 手すり
        f[int(rail):, :, :] = (WOOD[None, None, :] * (0.5 + 0.6 * s.grain[int(rail):, :])[:, :, None]) * (0.6 + 0.4 * dawn)
        for i in range(8): cv2.rectangle(f, (int(i * W / 7 - 6 * k), int(rail - 26 * k)), (int(i * W / 7 + 6 * k), int(rail)), tuple(float(x) for x in WOOD * 0.7), -1)
        cv2.rectangle(f, (0, int(rail - 34 * k)), (W, int(rail - 24 * k)), tuple(float(x) for x in WOOD * 0.9), -1)
        img = s.pose(t, breath=0.25, squash=0.04 * math.exp(-cm * 6) * math.cos(cm * 16))
        s.draw(f, img, W / 2, H * 0.60, 0.42, light=np.clip(s.lit(600 * k, 0.5) + dawn * 0.5, 0, 1)); s.lantern(f, t, 1 - dawn * 0.6)
        back_ = cm - 1.3
        if back_ > 0:                                                                # ハトが 葉をくわえて 帰ってくる
            p1 = s.at(*s.dove0); a = ease2(back_ / 1.1); p = (W * 1.1 + (p1[0] - W * 1.1) * a, H * 0.10 + (p1[1] - 18 * k - H * 0.10) * a + math.sin(a * 3.14) * -H * 0.05)
            if a < 1:
                L = s.L0(); s.dove(L, p, t, 1.6 * k, -1); f[:] = f * (1 - np.clip(L, 0, 1) * 0.95)[:, :, None] + np.clip(L, 0, 1)[:, :, None]
                cv2.ellipse(f, (int(p[0] - 22 * k), int(p[1] - 2 * k)), (int(9 * k), int(4 * k)), 30, 0, 360, (0.35, 0.8, 0.3), -1, cv2.LINE_AA)
            else:
                v = back_ - 1.1; L = s.L0()
                for i in range(2):
                    q = (v - i * 0.15) / 0.8
                    if 0 < q < 1: s.circ(L, p1, W * 0.5 * q, 1 - q, 2)
                s.add(f, L, LAMP, 0.8)
                cv2.ellipse(f, (int(p1[0] + 16 * k), int(p1[1] - 2 * k)), (int(9 * k), int(4 * k)), -30, 0, 360, (0.35, 0.8, 0.3), -1, cv2.LINE_AA)
        if cm < 0.5: f = f * (cm / 0.5) + (1 - cm / 0.5) * (FOAM * 0.2 + 0.8)[None, None, :]
        end = DUR - t
        if end < 0.6: f = f * clamp(end / 0.55) + (1 - clamp(end / 0.55))
        return f, 0

    # ---- この子だけの曲: 雨の音 と、8分の6拍子の 舟歌(ニ調ドリア)。窓がともるたび 1音。大波のあと 長調に ひらく ----
    def audio(s, path):
        a = Snd(DUR, 73); D = 146.83; dor = [0, 2, 3, 5, 7, 9, 10]; maj = [0, 2, 4, 5, 7, 9, 11]
        def fq(d, o=0, sc=dor): return D * 2 ** ((sc[d % 7] + 12 * (d // 7 + o)) / 12.0)
        a.a += (a.nz - a.lp(a.nz, 0.5)) * 0.10 * a.gate(0, T_CRASH, 0.4) * (1 + 0.8 * np.clip((a.t - T_FLOOD) / 3, 0, 1))        # 雨
        a.a += a.lp(a.nz, 0.02) * 0.5 * np.clip((a.t - T_FLOOD) / 3, 0, 1) * a.gate(T_FLOOD, T_CRASH, 0.3)                    # 水の うなり
        for i, d in enumerate([0, 4, 7, 4]): a.note(fq(d, 0), 0.5 + i * 0.45, 0.8, 0.08, "fm")
        for i in range(6): a.hat(T_DOVE + i * 0.06, 0.07, 0.03)                                                                # はばたき
        a.note(fq(9, 1), T_DOVE, 0.6, 0.07, "bell")
        a.boom(T_CUT + 0.33, 70, 1.0, 0.18); a.a += a.lp(a.nz, 0.6) * a.env(T_CUT + 0.33, 0.001, 0.06) * 0.5                   # 板が ばたん
        for i in range(6): a.note(fq([7, 5, 4, 2, 4, 0][i], 0), T_CUT + 0.5 + i * 0.14, 0.4, 0.10, "fm"); a.note(fq(14, 1), T_CUT + 0.5 + i * 0.14, 0.08, 0.04, "sine")
        bt = 0.22; tk = T_FLOOD; i = 0; mel = [0, 2, 4, 7, 4, 2, 3, 4, 5, 7, 9, 7, 5, 4, 2, 0]                                  # 舟歌(強・弱・弱)
        while tk < T_WAVE:
            if i % 3 == 0: a.boom(tk, 52, 0.45, 0.16); a.note(fq([0, 0, 3, 4][(i // 6) % 4], -1), tk, 0.5, 0.14, "saw", 0.01)
            else: a.note(fq([4, 7][i % 2], 0), tk, 0.12, 0.035, "sq")
            if i % 3 != 1: a.note(fq(mel[(i * 2 // 3) % 16], 1), tk, 0.3, 0.085, "fm")
            tk += bt; i += 1
        for j in range(8): a.note(fq(j, 1, maj), T_FLOOD + 1.5 + j * 2.2 / 8, 0.5, 0.07, "bell")                               # 窓が 2つずつ
        a.rise(T_WAVE, T_CRASH, 60, 200, 0.25); a.whoosh(T_WAVE, T_CRASH, 0.8, 0.08)
        a.boom(T_CRASH, 36, 1.5, 0.9); a.a += a.lp(a.nz, 0.25) * a.env(T_CRASH, 0.02, 0.9) * 0.9
        for j, (d, o) in enumerate(((0, -1), (4, -1), (0, 0), (2, 0), (4, 0), (7, 0))): a.note(fq(d, o, maj), T_CALM + 0.3 + j * 0.08, 2.4, 0.05, "pad", 0.5)   # なぎ: 長調
        for j, d in enumerate([0, 2, 4, 7, 9, 11, 14]): a.note(fq(d, 1, maj), T_CALM + 0.5 + j * 1.3 / 7, 0.8, 0.07, "bell")   # 虹の 7音
        a.note(fq(16, 1, maj), T_CALM + 2.4, 1.0, 0.10, "bell")
        a.save(path)


if __name__ == "__main__":
    r = Noah()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
