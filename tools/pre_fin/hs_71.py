"""ゼノン(★6 建国の王) 専用の必殺技「創国ノ王笏」
金の線で描かれた 古い地図の空 → ゼノンが 斜めに飛びこみ、頂点で ぴたりと止まる → 王笏(つえ)の「目」が ひらく(丸い窓のカットイン) →
ふりおろして 地を打つ → 打った所から 金の線が走り、城が 塔ごとに 立ち上がり、窓に灯がともり、旗があがる → 朝日と、国の印(封ろうの はんこ)が どんと押される
使い方: python3 hs_71.py 出力.mp4 / python3 hs_71.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 13.0
GOLD = np.array([1.0, 0.80, 0.34], np.float32); ROYAL = np.array([0.10, 0.18, 0.55], np.float32); SKYB = np.array([0.30, 0.55, 1.0], np.float32); RED = np.array([0.85, 0.16, 0.14], np.float32)
T_STOP, T_CUT, T_RAISE, T_HIT, T_SUN, T_SEAL = 1.3, 2.7, 4.8, 5.55, 9.0, 10.7
TOWERS = [(0.50, 0.17, 0.36, 2), (0.355, 0.10, 0.27, 1), (0.645, 0.10, 0.27, 1), (0.22, 0.09, 0.20, 0), (0.78, 0.09, 0.20, 0), (0.09, 0.08, 0.14, 1), (0.91, 0.08, 0.14, 1)]


class Zenon(Kit):
    def __init__(s, W=540):
        super().__init__(71, (1.0, 0.80, 0.34), "創国ノ王笏", "Founding Scepter", W)
        s.orb = s.A(398, 86); s.eyeL = s.A(150, 352); s.eyeR = s.A(246, 352)
        s.ht = [s.text_img(c, hz.SERIF, int(80 * s.k), 3) for c in "創国ノ王笏"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        rg = np.random.RandomState(71)
        s.cf = rg.rand(80, 6).astype(np.float32); s.win = rg.rand(60, 3).astype(np.float32)
        # 地図の線は 1回だけ描いて ずらして使う
        M = np.zeros((s.H, s.W + 200), np.float32)
        for i in range(-2, 16): cv2.line(M, (i * 70, 0), (i * 70 + 160, s.H), 0.5, 1, cv2.LINE_AA)
        for j in range(14): cv2.ellipse(M, (s.W // 2 + 100, -200), (120 + j * 95, 100 + j * 80), 0, 0, 180, 0.5, 1, cv2.LINE_AA)
        s.map = cv2.GaussianBlur(M, (0, 0), 0.7)

    # ---- この子だけの絵 ----
    def atlas(s, t, lum=1.0, day=0.0):
        y = s.YY / s.H
        f = (ROYAL[None, None, :] * (0.35 + 0.9 * y)[:, :, None] * (1 - day) + (SKYB * 0.55 + GOLD * 0.35)[None, None, :] * (0.4 + 0.8 * y)[:, :, None] * day) * lum
        off = int((t * 12) % 70); m = s.map[:, off:off + s.W]
        f += m[:, :, None] * GOLD[None, None, :] * 0.22 * lum * (1 - day * 0.5)
        return f.astype(np.float32)

    def hero(s, f, t, cx, cy, hfrac, deg=0.0, alpha=1.0, white=0.0, **kw):
        img = s.pose(t, **kw)
        if deg:
            M = cv2.getRotationMatrix2D((s.pw / 2, s.ph / 2), deg, 1.0); img = cv2.warpAffine(img, M, (s.pw, s.ph), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        s.draw(f, img, cx, cy, hfrac, alpha, white); s._deg = deg

    def pt(s, a):
        p = s.at(*a); cx, cy, _ = s._pl; r = math.radians(s._deg); dx, dy = p[0] - cx, p[1] - cy
        return (cx + math.cos(r) * dx + math.sin(r) * dy, cy - math.sin(r) * dx + math.cos(r) * dy)

    def scepter_eye(s, f, p, open_, t, big=1.0):
        """王笏の目: まぶたが ひらいて 青いひとみが光り、輪が ひろがる"""
        if open_ <= 0: return
        L = s.L0(); r = 12 * s._pl[2]                                                # ひとみの大きさは 絵に合わせる
        cv2.circle(L, (int(p[0]), int(p[1])), max(1, int(r * (0.6 + 0.4 * open_))), 1.0, -1, cv2.LINE_AA)
        f += (cv2.GaussianBlur(L, (0, 0), max(1.5, r * 0.5)) * 0.9 * open_ + cv2.GaussianBlur(L, (0, 0), max(3, r * 2.2)) * 0.9 * open_ * big)[:, :, None] * (SKYB * 0.9 + 0.1)[None, None, :]
        r = r * 0.9 * (0.6 + 0.4 * big)
        L = s.L0()
        for i in range(3):
            v = (t * 0.9 + i / 3.0) % 1.0; s.circ(L, p, r * (1.5 + 5 * v), (1 - v) * 0.8 * open_, 2)
        s.add(f, L, SKYB * 0.6 + GOLD * 0.4, 0.8)

    def tower(s, L, Wn, x, w, h, kind, g, gy, t, i):
        W = s.W; H = s.H; x0 = (x - w / 2) * W; x1 = (x + w / 2) * W; top = gy - h * H * g
        s.line(L, (x0, gy), (x0, top), 1.0, 2); s.line(L, (x1, gy), (x1, top), 1.0, 2); s.line(L, (x0, top), (x1, top), 1.0, 2)
        if g < 1: return
        n = max(3, int(w * 40)); cw = (x1 - x0) / (n * 2 - 1)
        for j in range(n):                                                           # 胸壁
            a = x0 + j * 2 * cw; s.poly(L, [(a, top), (a, top - 9 * s.k), (a + cw, top - 9 * s.k), (a + cw, top)], 1.0, 2)
        cx = x * W
        if kind:                                                                     # とんがり屋根 と 旗
            rh = (60 if kind == 2 else 38) * s.k; s.poly(L, [(x0 + 6 * s.k, top - 9 * s.k), (cx, top - 9 * s.k - rh), (x1 - 6 * s.k, top - 9 * s.k)], 1.0, 2)
            fy = top - 9 * s.k - rh; s.line(L, (cx, fy), (cx, fy - 34 * s.k), 1.0, 2)
            wv = math.sin(t * 6 + i) * 5 * s.k
            cv2.fillPoly(Wn, [np.array([(cx, fy - 34 * s.k), (cx + 30 * s.k, fy - 27 * s.k + wv), (cx, fy - 19 * s.k)], np.int32)], 1.0, cv2.LINE_AA)

    def frame(s, t):
        k = s.k
        if t < T_CUT: f, sh = s.sc_leap(t)
        elif t < T_RAISE: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_SUN: f, sh = s.sc_found(t)
        else: f, sh = s.sc_seal(t)
        return s.post(f, GOLD, 0.6, sh * k, t, 0.88)

    # 1) 飛びこんで、頂点で ぴたり
    def sc_leap(s, t):
        W, H, k = s.W, s.H, s.k
        f = s.atlas(t, clamp(t / 0.5))
        a = ease(t / T_STOP); stop = t - T_STOP
        cx = W * (1.25 - 0.75 * a); cy = H * (0.05 + 0.40 * a) - math.sin(a * 3.1416) * H * 0.10; deg = -28 * (1 - a)
        if stop < 0:
            L = s.L0()
            for i in range(26):                                                      # 飛んできた向きの 速さの線
                r = s.cf[i]; x = (r[0] * 1.6 - 0.2 + t * 0.9) % 1.4 * W; y = r[1] * H
                s.line(L, (x, y), (x + W * 0.22 * r[2], y - H * 0.09 * r[2]), 0.35 * (1 - a), 1)
            s.add(f, L, GOLD, 0.7)
            for j in (3, 2, 1):
                b = ease((t - j * 0.07) / T_STOP)
                s.hero(f, t, W * (1.25 - 0.75 * b), H * (0.05 + 0.40 * b) - math.sin(b * 3.1416) * H * 0.10, 0.46, -28 * (1 - b), 0.18 * (1 - a), 0.7)
        s.hero(f, t, cx, cy, 0.46, deg, squash=(-0.05 * math.exp(-stop * 9) if stop > 0 else 0.0))
        if stop > 0:                                                                 # ぴたり: 輪がひとつ、王笏の目に 小さな光
            L = s.L0(); v = stop / 0.6
            if v < 1: s.circ(L, (cx, cy), W * (0.2 + 0.7 * v), 1 - v, 3)
            s.add(f, L, GOLD, 0.9)
            p = s.pt(s.orb); s.scepter_eye(f, p, 0.12 + 0.1 * math.sin(stop * 9), t, 1.0)
            if stop < 0.08: f += (1 - stop / 0.08) * 0.6
        return f, (6 * math.exp(-stop * 8) if stop > 0 else 0)

    # 2) 丸い窓の中で 王笏の目が ひらく
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.atlas(t, 0.35)
        C = (W / 2, H * 0.34); R = W * 0.40 * back(u / 0.35, 1.2)
        win = np.zeros((H, W, 3), np.float32) + (ROYAL * 1.3 + SKYB * 0.15)[None, None, :]
        zf = 1.55; sc = zf * H / s.ah
        s.hero(win, t, C[0] - (s.orb[0] - s.aw / 2) * sc, C[1] - (s.orb[1] - s.ah / 2) * sc, zf)
        op = ease((u - 0.35) / 0.25)
        s.scepter_eye(win, s.pt(s.orb), op * (0.8 + 0.2 * math.sin(u * 10)), t, 1.6)
        m = s.L0(); cv2.circle(m, (int(C[0]), int(C[1])), max(1, int(R)), 1.0, -1, cv2.LINE_AA)
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        L = s.L0(); s.circ(L, C, R, 1.0, 4); s.circ(L, C, R + 12 * k, 0.5, 1)
        for i in range(36):                                                          # 窓のふちの 目もり(回る)
            a = i * 0.1745 + u * 0.5; r0 = R + (20 if i % 3 else 30) * k
            s.line(L, (C[0] + math.cos(a) * (R + 12 * k), C[1] + math.sin(a) * (R + 12 * k)), (C[0] + math.cos(a) * r0, C[1] + math.sin(a) * r0), 0.7, 1)
        s.add(f, L, GOLD, 1.0)
        n = len(s.ht); step = W * 0.17
        for i, T in enumerate(s.ht):                                                 # 下から せり上がる 技名
            a = (u - 0.6 - i * 0.12) / 0.2
            if a > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.72 + (1 - ease(a)) * 50 * k, clamp(a), 1.0)
        if u > 0.6:
            L = s.L0(); g = ease((u - 0.6) / 0.5); s.line(L, (W / 2 - W * 0.44 * g, H * 0.72 + 56 * k), (W / 2 + W * 0.44 * g, H * 0.72 + 56 * k), 0.9, 2); s.add(f, L, GOLD, 0.9)
        for i, T in enumerate(s.vo):
            if u > 0.5 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.72 - 74 * k, 1.0, 1.0)
        if u > 1.3: s.text(f, s.txt_en, W / 2, H * 0.72 + 84 * k, ease((u - 1.3) / 0.3), 1.0)
        if u < 0.1: f += (1 - u / 0.1) * 0.8
        return f, 0

    # 3) ふりおろして 地を打つ → 城が 立ち上がる
    def sc_found(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_RAISE; hit = t - T_HIT; gy = H * 0.80
        day = ease2((hit - 1.0) / 2.4) * 0.35 if hit > 0 else 0
        f = s.atlas(t, 0.8, day)
        if hit > 0:                                                                  # 城(うしろ)
            L = s.L0(); Wn = s.L0(); Fl = s.L0()
            for i, (x, w, h, kind) in enumerate(TOWERS):
                g = back((hit - 0.25 - i * 0.22) / 0.5, 1.4) if hit > 0.25 + i * 0.22 else 0
                if g > 0: s.tower(L, Fl, x, w, h, kind, min(g, 1.0) if g < 1 else 1.0, gy, t, i)
            wl = ease((hit - 0.2) / 0.6); s.line(L, (W / 2 - W * 0.5 * wl, gy - H * 0.07), (W / 2 + W * 0.5 * wl, gy - H * 0.07), 0.8, 2)
            for j in range(60):                                                      # 窓に 灯が ともる
                r = s.win[j]; i = int(r[0] * len(TOWERS)); x, w, h, kind = TOWERS[i]
                if hit > 1.0 + i * 0.22 + r[2] * 1.2:
                    wx = (x + (r[1] - 0.5) * w * 0.6) * W; wy = gy - (0.12 + r[2] * 0.8) * h * H
                    cv2.rectangle(Wn, (int(wx - 3 * k), int(wy - 5 * k)), (int(wx + 3 * k), int(wy + 5 * k)), 1.0, -1)
            s.add(f, L, GOLD, 0.9); s.add(f, Wn, GOLD * 0.8 + 0.2, 0.9, 2, 7); s.add(f, Fl, RED, 0.8, 1, 4); f += Fl[:, :, None] * RED[None, None, :] * 0.7
        f[int(gy):, :, :] = ROYAL * 0.35
        if hit > 0:                                                                  # 打った所から 道が走る
            L = s.L0(); g = ease(hit / 0.5)
            for i in range(9):
                a = (i - 4) * 0.34; s.line(L, (W / 2, gy), (W / 2 + math.sin(a) * W * 1.2 * g, gy + math.cos(a) * (H - gy) * 1.3 * g), 0.8, 2)
            for i in range(3):
                v = (hit - i * 0.1) / 0.8
                if 0 < v < 1: s.ell(L, (W / 2, gy), (W * 1.0 * v, W * 0.16 * v), 0, 1 - v, 3)
            s.add(f, L, GOLD, 1.0)
        # ゼノン: ため(のびて うしろへ) → ふりおろす → 打つ → 立つ
        if hit < 0:
            r = ease(u / 0.5); dv = clamp((u - 0.5) / (T_HIT - T_RAISE - 0.5)) ** 2
            cy = H * (0.36 - 0.05 * r) + (gy - H * 0.17 - H * 0.31) * dv; deg = 22 * r - 60 * dv
            for j in (2, 1):
                if dv > 0.05: s.hero(f, t, W / 2, cy - j * H * 0.07 * dv, 0.40, deg + 14 * j, 0.2, 0.6)
            s.hero(f, t, W / 2, cy, 0.40, deg, squash=-0.06 * r * (1 - dv), jit=1.2 * k * r * (1 - dv))
            s.scepter_eye(f, s.pt(s.orb), 1.0, t, 1.2 + r)
        else:
            s.hero(f, t, W / 2, gy - H * 0.17, 0.40, -38 * math.exp(-hit * 5), squash=0.10 * math.exp(-hit * 8) * math.cos(hit * 22), sway=0.8 * math.exp(-hit * 3))
            s.scepter_eye(f, s.pt(s.orb), 1.0, t, 1.2)
            if hit < 0.12: f = f + (1 - hit / 0.12) * 1.3
        if u < 0.15: f += (1 - u / 0.15) * 0.6
        return f, (18 * math.exp(-hit * 4.5) if hit > 0 else 0)

    # 4) 朝日が昇り、国の印が 押される
    def sc_seal(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_SUN; gy = H * 0.80; se = t - T_SEAL
        f = s.atlas(t, 0.9, 0.35 + 0.55 * ease2(u / 1.4))
        sun = (W / 2, gy - H * 0.07 - H * 0.30 * ease(u / 1.6))
        L = s.L0()
        for i in range(14):                                                          # 朝日の すじ(回る)
            a = i * 0.4488 + t * 0.25
            cv2.fillPoly(L, [np.array([sun, (sun[0] + math.cos(a) * W * 1.6, sun[1] + math.sin(a) * W * 1.6), (sun[0] + math.cos(a + 0.12) * W * 1.6, sun[1] + math.sin(a + 0.12) * W * 1.6)], np.int32)], 0.22, cv2.LINE_AA)
        cv2.circle(L, (int(sun[0]), int(sun[1])), int(W * 0.13), 1.0, -1, cv2.LINE_AA)
        f += (L + cv2.GaussianBlur(L, (0, 0), 18 * k) * 0.8)[:, :, None] * (GOLD * 0.9 + 0.15)[None, None, :] * ease(u / 0.8)
        L = s.L0(); Wn = s.L0(); Fl = s.L0(); D = s.L0()
        for i, (x, w, h, kind) in enumerate(TOWERS):                                 # 城は 逆光の影になる
            x0 = int((x - w / 2) * W); x1 = int((x + w / 2) * W); cv2.rectangle(D, (x0, int(gy - h * H)), (x1, int(gy)), 1.0, -1)
            s.tower(L, Fl, x, w, h, kind, 1.0, gy, t, i)
        cv2.rectangle(D, (0, int(gy - H * 0.07)), (W, int(gy)), 1.0, -1)
        f = f * (1 - D * 0.72)[:, :, None]; s.add(f, L, GOLD, 0.8); f += Fl[:, :, None] * RED[None, None, :] * 0.9
        f[int(gy):, :, :] = ROYAL * 0.45
        s.hero(f, t, W / 2, gy - H * 0.17, 0.40, 0, squash=(0.03 * math.exp(-se * 8) if se > 0 else 0))
        s.scepter_eye(f, s.pt(s.orb), 1.0, t, 1.2)
        L = s.L0()
        for i in range(80):                                                          # 紙ふぶき
            r = s.cf[i]; y = ((r[1] + u * (0.18 + 0.3 * r[2])) % 1.0) * H; x = r[0] * W + math.sin(u * 3 + r[3] * 6) * 16 * k
            cv2.ellipse(L, (int(x), int(y)), (int(5 * k), int(2 * k)), math.degrees(u * 5 * (r[4] - 0.5) + r[5] * 6), 0, 360, 0.7 * clamp(u / 0.6), -1, cv2.LINE_AA)
        s.add(f, L, GOLD * 0.7 + 0.3, 0.7, 1, 5)
        if se > 0:                                                                   # 国の印: 封ろうの はんこが 上から どん
            z = 1 + 2.6 * (1 - ease(se / 0.16)); C = (W / 2, H * 0.30); R = W * 0.25 * z
            wax = s.L0(); pts = [(C[0] + math.cos(i * 0.2618) * R * (1.0 + 0.07 * math.sin(i * 2.3)), C[1] + math.sin(i * 0.2618) * R * (1.0 + 0.07 * math.sin(i * 2.3))) for i in range(24)]
            cv2.fillPoly(wax, [np.array(pts, np.int32)], 1.0, cv2.LINE_AA)
            a = clamp(se / 0.1); f = f * (1 - wax * 0.92 * a)[:, :, None] + wax[:, :, None] * RED[None, None, :] * 0.75 * a
            L = s.L0(); s.circ(L, C, R * 0.80, 1.0, 3); s.circ(L, C, R * 0.72, 0.6, 1)
            cr = [(-0.42, 0.22), (-0.50, -0.20), (-0.25, 0.02), (0.0, -0.36), (0.25, 0.02), (0.50, -0.20), (0.42, 0.22)]      # 王冠のしるし
            s.poly(L, [(C[0] + x * R, C[1] + y * R) for x, y in cr], 1.0, 3); s.line(L, (C[0] - 0.42 * R, C[1] + 0.32 * R), (C[0] + 0.42 * R, C[1] + 0.32 * R), 1.0, 3)
            for i in range(24): b = i * 0.2618; s.line(L, (C[0] + math.cos(b) * R * 0.72, C[1] + math.sin(b) * R * 0.72), (C[0] + math.cos(b) * R * 0.66, C[1] + math.sin(b) * R * 0.66), 0.7, 1)
            s.add(f, L, GOLD, 1.1 * a)
            if se > 0.16:
                v = (se - 0.16); L = s.L0()
                for i in range(2):
                    q = (v - i * 0.12) / 0.7
                    if 0 < q < 1: s.circ(L, C, W * (0.25 + 0.9 * q), 1 - q, 3)
                for i in range(30):
                    r = s.cf[i]; b = r[0] * 6.283; rr = W * (0.26 + v * (0.5 + r[1])); s.line(L, (C[0] + math.cos(b) * rr, C[1] + math.sin(b) * rr), (C[0] + math.cos(b) * (rr + 18 * k), C[1] + math.sin(b) * (rr + 18 * k)), clamp(1 - v), 2)
                s.add(f, L, GOLD, 1.0)
                if v < 0.1: f += (1 - v / 0.1) * 0.9
        if u < 0.2: f += (1 - u / 0.2) * 0.5
        end = DUR - t
        if end < 0.9: v = clamp((0.9 - end) / 0.4); f = f * (1 - v) + v * (GOLD * 0.3 + 0.7)[None, None, :]
        if end < 0.5: f = f * clamp(end / 0.45) ** 2
        return f, (14 * math.exp(-(se - 0.16) * 6) if se > 0.16 else 0)

    # ---- この子だけの曲: 王のファンファーレ(変ロ長調)。風 → 目がひらく音 → ため → ティンパニ → らっぱの和音が 塔ごとに 重なる → 印 ----
    def audio(s, path):
        a = Snd(DUR, 71); Bb = 233.08; maj = [0, 2, 4, 5, 7, 9, 11]
        def fq(d, o=0): return Bb * 2 ** ((maj[d % 7] + 12 * (d // 7 + o)) / 12.0)
        def brass(d, t0, dur, g=0.07, o=0):
            a.note(fq(d, o), t0, dur, g, "saw", 0.03); a.note(fq(d, o) * 1.004, t0, dur, g * 0.6, "saw", 0.03)
        a.whoosh(0.1, T_STOP, 0.5, 0.25); a.boom(T_STOP, 70, 0.6, 0.15)
        a.note(fq(4, 2), T_STOP + 0.05, 0.9, 0.10, "bell")
        a.boom(T_CUT, 50, 1.0); a.rise(T_CUT + 0.3, T_CUT + 0.6, 300, 900, 0.12)
        for d in (0, 2, 4): brass(d, T_CUT + 0.35, 0.9, 0.05)
        for i in range(5): brass([0, 2, 4, 2, 7][i], T_CUT + 0.6 + i * 0.12, 0.16, 0.07)
        a.rise(T_RAISE, T_HIT, 90, 400, 0.2)
        for i in range(14): a.hat(T_RAISE + i * (T_HIT - T_RAISE) / 14, 0.05 + i * 0.008, 0.03)                    # たいこの ロール
        a.boom(T_HIT, 40, 1.4, 0.7); a.boom(T_HIT + 0.02, 90, 0.6, 0.2)
        chord = [(0, -1), (4, -1), (0, 0), (2, 0), (4, 0), (7, 0), (9, 0)]                                         # 塔が立つたび 音が重なる
        for i, (d, o) in enumerate(chord):
            tk = T_HIT + 0.25 + i * 0.22; brass(d, tk, T_SUN - tk + 0.3, 0.035, o); a.boom(tk, 60 + i * 6, 0.35, 0.12)
        fan = [(0, 0.0), (0, 0.18), (0, 0.27), (4, 0.36), (7, 0.72), (4, 0.9), (7, 1.08), (9, 1.26), (11, 1.44), (14, 1.62)]     # ファンファーレ
        for d, dt in fan: brass(d, T_SUN - 1.9 + dt, 0.22, 0.09, 0); a.note(fq(d, 1), T_SUN - 1.9 + dt, 0.2, 0.03, "sq")
        for d in (0, 2, 4, 7, 9): brass(d, T_SUN, 1.6, 0.045)
        bt = 0.34
        for i in range(5): a.boom(T_SUN + i * bt, 52, 0.4, 0.15); a.hat(T_SUN + i * bt + bt / 2, 0.06)
        for i, d in enumerate([7, 9, 11, 14]): brass(d, T_SUN + 0.2 + i * bt, 0.3, 0.07)
        a.whoosh(T_SEAL - 0.25, T_SEAL + 0.16, 0.5, 0.3); a.boom(T_SEAL + 0.16, 38, 1.5, 0.8)
        for d, o in ((0, -1), (0, 0), (2, 0), (4, 0), (7, 0), (14, 0)): brass(d, T_SEAL + 0.18, 1.8, 0.05, o)
        a.note(fq(7, 1), T_SEAL + 0.2, 1.4, 0.10, "bell")
        a.save(path)


if __name__ == "__main__":
    r = Zenon()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
