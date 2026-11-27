"""オーロラ(★6 極光の女神) 専用の必殺技「極光ノ天幕」
しずかな 雪原の夜。流れ星のように おりてくる → 白い光が ひたいの星で 七色に 分かれる(プリズムのカットイン) →
両手を ひろげると、空に 光のカーテンが 一枚ずつ ひらく(雪に うつる) → カーテンが おりてきて、頭上の一点へ 集まり、光の天幕になる →
舞台の幕のように 左右から 閉じる → ひらくと 夜明け。太陽のまわりに 光の輪(ハロ)と 幻日
使い方: python3 hs_78.py 出力.mp4 / python3 hs_78.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 13.8
AG = np.array([0.30, 1.0, 0.62], np.float32); AM = np.array([1.0, 0.42, 0.85], np.float32); AB = np.array([0.45, 0.72, 1.0], np.float32); SNOW = np.array([0.80, 0.88, 1.0], np.float32); NIGHT = np.array([0.02, 0.04, 0.12], np.float32)
BOW = [(1.0, 0.30, 0.30), (1.0, 0.60, 0.25), (1.0, 0.92, 0.35), (0.40, 0.95, 0.50), (0.35, 0.75, 1.0), (0.45, 0.45, 1.0), (0.80, 0.45, 1.0)]
T_LAND, T_CUT, T_VEIL, T_TENT, T_CLOSE, T_OPEN = 1.5, 2.7, 4.8, 6.9, 8.9, 10.5


class Aurora(Kit):
    def __init__(s, W=540):
        super().__init__(78, (0.30, 1.0, 0.62), "極光ノ天幕", "Aurora Veil", W)
        s.eyeL = s.A(306, 226); s.eyeR = s.A(421, 226); s.gem = s.A(360, 132); s.hL = s.A(160, 388); s.hR = s.A(562, 402)
        s.vt = [s.text_img(c, hz.SERIF, int(68 * s.k), 3) for c in "極光ノ天幕"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        rg = np.random.RandomState(78); s.rd = rg.rand(300, 5).astype(np.float32)
        s.yn = s.YY / s.H; s.xn = s.XX / s.W; s.gnd = s.H * 0.80; s.xr = s.XX[0]
        s.mtn = [(i * s.W / 10.0, s.gnd - (0.04 + 0.12 * rg.rand()) * s.H * (1 if i % 2 else 0.45)) for i in range(11)]
        s.ray = cv2.GaussianBlur(rg.rand(1, s.W * 4).astype(np.float32), (0, 0), 2.0)[0]; s.ray = (s.ray - s.ray.min()) / (s.ray.max() - s.ray.min())

    # ---- この子だけの絵 ----
    def snowfield(s, t, dawn=0.0, lum=1.0):
        f = (NIGHT[None, None, :] * (0.6 + 1.6 * s.yn)[:, :, None] * (1 - dawn) + (np.array([1.0, 0.78, 0.66], np.float32)[None, None, :] * (s.yn ** 1.3)[:, :, None] + np.array([0.42, 0.62, 0.95], np.float32)[None, None, :] * (1 - s.yn ** 1.3)[:, :, None]) * dawn) * lum
        if dawn < 1:
            L = s.L0()
            for i in range(110): r = s.rd[i]; cv2.circle(L, (int(r[0] * s.W), int(r[1] * s.gnd)), 1, float((0.3 + 0.7 * math.sin(t * 2 + r[2] * 30) ** 2) * (1 - dawn)), -1)
            f += L[:, :, None]
        M = s.L0(); cv2.fillPoly(M, [np.array([(0, s.gnd)] + s.mtn + [(s.W, s.gnd)], np.int32)], 1.0, cv2.LINE_AA)
        f = f * (1 - M[:, :, None]) + M[:, :, None] * (NIGHT * 1.6 * (1 - dawn) + np.array([0.45, 0.50, 0.70], np.float32) * dawn)[None, None, :] * lum
        g = int(s.gnd); f[g:, :, :] = (SNOW * (0.22 + 0.55 * dawn))[None, None, :] * (1.15 - 0.6 * (s.yn[g:, :, None] - 0.8) / 0.2) * lum
        return f.astype(np.float32)

    def veil(s, f, t, y0, amp, hgt, open_, ph, v=1.0):
        """光のカーテン 1枚。下のふちは 波うち、上へ すじが のびる。open_ で 左から ひらく"""
        if open_ <= 0: return
        x = s.xr / s.W
        yb = y0 + amp * (np.sin(x * 5.5 + t * 0.7 + ph) * 0.6 + np.sin(x * 13 - t * 1.1 + ph * 2) * 0.25)
        off = int((t * 40 + ph * 300) % (s.W * 3)); ray = s.ray[off:off + s.W]
        h = hgt * (0.45 + 0.85 * ray)
        d = yb[None, :] - s.YY
        I = (d > 0) * np.exp(-np.clip(d, 0, None) / h[None, :]) * (0.5 + 0.7 * ray)[None, :] * np.clip((open_ * 1.25 - x) / 0.25, 0, 1)[None, :] * v
        low = np.exp(-np.clip(d, 0, None) / (h[None, :] * 0.30))
        f += (I * low)[:, :, None] * AG[None, None, :] * 1.1 + (I * (1 - low))[:, :, None] * (AM * 0.8 + AB * 0.35)[None, None, :]
        edge = np.clip(1 - np.abs(d) / (2.5 * s.k), 0, 1) * np.clip((open_ * 1.25 - x) / 0.25, 0, 1)[None, :] * v
        f += edge[:, :, None] * (AG * 0.6 + 0.4)[None, None, :] * 0.6
        ref = np.clip((s.YY - s.gnd) / (s.H - s.gnd), 0, 1) * (s.YY > s.gnd)                                      # 雪に うつる
        f += (ref * (0.5 + 0.7 * ray)[None, :] * np.clip((open_ * 1.25 - x) / 0.25, 0, 1)[None, :] * 0.10 * v)[:, :, None] * AG[None, None, :]

    def snow(s, f, t, v=1.0):
        L = s.L0()
        for i in range(70):
            r = s.rd[i + 120]; y = ((r[1] + t * (0.05 + 0.08 * r[2])) % 1.0) * s.H; x = (r[0] + 0.02 * math.sin(t + r[3] * 9)) * s.W
            cv2.circle(L, (int(x), int(y)), 1 if r[4] < 0.6 else 2, 0.7 * v, -1)
        f += L[:, :, None] * SNOW[None, None, :]

    def hands(s, f, v, t):
        if v <= 0: return
        L = s.L0(); sc = s._pl[2]
        for hnd in (s.hL, s.hR): p = s.at(*hnd); s.star4(L, p, (14 + 4 * math.sin(t * 5)) * sc * 2 * v, v, t * 0.6, 0.2)
        g = s.at(*s.gem); s.star4(L, g, 16 * sc * 2 * v, v, 0.0, 0.18)
        s.add(f, L, AG * 0.5 + 0.5, 1.0, 2, 8)

    def frame(s, t):
        k = s.k
        if t < T_CUT: f, sh = s.sc_fall(t)
        elif t < T_VEIL: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_CLOSE: f, sh = s.sc_veil(t)
        else: f, sh = s.sc_curtain(t)
        return s.post(f, AG, 0.4, sh * k, t, 0.97)

    # 1) 雪原に おりる
    def sc_fall(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_LAND
        f = s.snowfield(t, 0, clamp(t / 0.5)); s.snow(f, t)
        cy = s.gnd - H * 0.235
        if u < 0:                                                                    # 流れ星
            a = ease2(t / T_LAND); p = (W * (0.95 - 0.45 * a), H * (0.02 + (cy / H - 0.02) * a)); L = s.L0()
            for j in range(12): q = max(0, a - j * 0.03); cv2.circle(L, (int(W * (0.95 - 0.45 * q)), int(H * (0.02 + (cy / H - 0.02) * q))), max(1, int((9 - j * 0.7) * k)), float(1 - j / 12), -1, cv2.LINE_AA)
            s.add(f, L, WHITE * 0.6 + AG * 0.4, 1.2)
        else:
            img = s.pose(t, squash=0.06 * math.exp(-u * 8) * math.cos(u * 18)); s.draw(f, img, W / 2, cy, 0.46, 1.0, clamp(1 - u / 0.35))
            L = s.L0()
            for i in range(3):                                                       # 雪の上に 光の輪
                v = (u - i * 0.14) / 0.9
                if 0 < v < 1: s.ell(L, (W / 2, s.gnd + 6 * k), (W * 0.8 * v, W * 0.10 * v), 0, 1 - v, 2)
            s.add(f, L, AG * 0.6 + 0.4, 0.9)
            if u < 0.1: f += (1 - u / 0.1) * 0.5
        return f, 0

    # 2) プリズム: 白い光が 七色に 分かれる
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = np.zeros((H, W, 3), np.float32) + (NIGHT * 0.8)[None, None, :]
        zf = 0.80; sc = zf * H / s.ah; G = (W * 0.30, H * 0.20)
        cx = G[0] - (s.gem[0] - s.aw / 2) * sc; cy = G[1] - (s.gem[1] - s.ah / 2) * sc
        g = ease(u / 0.3); fan = ease((u - 0.3) / 0.35)
        img = s.pose(t); s.draw(f, img, cx, cy, zf)
        if fan > 0:                                                                  # 七色の 扇(ひたいの星から 右下へ)
            for j, c in enumerate(BOW):
                a0 = -0.25 + j * 0.17; a1 = a0 + 0.17; ln = W * 1.7 * fan; M = s.L0()
                cv2.fillPoly(M, [np.array([G, (G[0] + math.cos(a0) * ln, G[1] + math.sin(a0) * ln), (G[0] + math.cos(a1) * ln, G[1] + math.sin(a1) * ln)], np.int32)], 1.0, cv2.LINE_AA)
                f = f * (1 - M[:, :, None] * 0.55) + M[:, :, None] * np.array(c, np.float32)[None, None, :] * 0.60
        L = s.L0(); s.line(L, (-10, G[1] + (G[0] + 10) * 0.25), (-10 + (G[0] + 10) * g, G[1] + (G[0] + 10) * 0.25 * (1 - g)), 1.0, 4); s.add(f, L, WHITE, 1.2)     # 入ってくる 白い光
        L = s.L0(); s.star4(L, G, (20 + 30 * fan + 4 * math.sin(u * 9)) * k, 1.0, u * 0.5, 0.15); s.add(f, L, WHITE, 1.2)
        for i, T in enumerate(s.vt):                                                 # 技名: 色の帯の上に 1字ずつ
            a = (u - 0.75 - i * 0.13) / 0.2
            if a > 0: s.text(f, T, W * 0.86, H * 0.38 + i * 78 * k, clamp(a), 1.0)
        for i, T in enumerate(s.vo):
            if u > 0.6 + i * 0.1: s.text(f, T, W * 0.86, H * 0.25 + i * 36 * k, 1.0, 1.0)
        if u > 1.5: s.text(f, s.txt_en, W * 0.40, H * 0.93, ease((u - 1.5) / 0.3), 1.0)
        if u < 0.08: f += (1 - u / 0.08) * 0.7
        return f, 0

    # 3) 光のカーテンが ひらく → 天幕になる
    def sc_veil(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_VEIL; tn = t - T_TENT
        f = s.snowfield(t, 0, 1.0)
        tent = ease2(tn / 1.5) if tn > 0 else 0.0
        for i, (y0, amp, hgt, ph) in enumerate(((0.30, 0.035, 0.11, 0.0), (0.42, 0.05, 0.14, 2.1), (0.54, 0.04, 0.12, 4.0))):
            s.veil(f, t, H * (y0 + 0.10 * tent), H * amp, H * hgt, ease2((u - 0.2 - i * 0.55) / 1.1), ph, 1.0 - 0.5 * tent)
        cy = s.gnd - H * 0.235
        if tent > 0:                                                                 # 天幕: 地の輪から 頭上の一点へ
            A = (W / 2, H * 0.10); L7 = [s.L0() for _ in range(7)]; n = 56
            for i in range(n):
                a = i * 6.2832 / n + t * 0.12; bx = W / 2 + math.cos(a) * W * 0.62; by = s.gnd + 10 * k + math.sin(a) * H * 0.07
                front = math.sin(a) > 0; q = tent
                e = (bx + (A[0] - bx) * q, by + (A[1] - by) * q)
                mid = ((bx + e[0]) / 2 + (bx - W / 2) * 0.18, (by + e[1]) / 2 + H * 0.05)
                pts = [(bx + (mid[0] - bx) * 2 * v * (1 - v) * 2 + (e[0] - bx) * v * v, by + (mid[1] - by) * 2 * v * (1 - v) * 2 + (e[1] - by) * v * v) for v in np.linspace(0, 1, 10)]
                cv2.polylines(L7[i % 7], [np.array(pts, np.int32)], False, 0.75 if front else 0.35, 2, cv2.LINE_AA)
            for j, c in enumerate(BOW): f += (L7[j] + cv2.GaussianBlur(L7[j], (0, 0), 5 * k) * 1.3)[:, :, None] * np.array(c, np.float32)[None, None, :] * 0.8
            L = s.L0(); s.star4(L, A, (10 + 40 * tent) * k, 1.0, t, 0.15); s.ell(L, (W / 2, s.gnd + 10 * k), (W * 0.62, H * 0.07), 0, 0.6 * tent, 2); s.add(f, L, WHITE * 0.7 + AG * 0.3, 1.0)
        s.snow(f, t, 0.7)
        spread = back(u / 0.5, 1.4)
        img = s.pose(t, squash=-0.04 * math.exp(-u * 6), lean=(-4 * k * math.exp(-tn * 5) if tn > 0 else 0))
        s.soft(f, (W / 2, cy), W * 0.30, 0.16, AG); s.draw(f, img, W / 2, cy, 0.46); s.hands(f, clamp(spread), t)
        if tn > 0:                                                                   # 手から 天幕へ のびる光
            L = s.L0()
            for hnd in (s.hL, s.hR): p = s.at(*hnd); q = clamp(tn / 0.4); s.line(L, p, (p[0] + (W / 2 - p[0]) * q, p[1] + (H * 0.10 - p[1]) * q), 0.6 * clamp(1.6 - tn), 2)
            s.add(f, L, WHITE, 0.9)
        if u < 0.15: f += (1 - u / 0.15) * 0.5
        return f, 0

    # 4) 光の幕が 閉じて、ひらくと 夜明け
    def sc_curtain(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_CLOSE; op = t - T_OPEN
        cy = s.gnd - H * 0.235
        if op < 0:
            f = s.snowfield(t, 0, 1.0)
            for i, (y0, amp, hgt, ph) in enumerate(((0.40, 0.035, 0.11, 0.0), (0.52, 0.05, 0.14, 2.1), (0.64, 0.04, 0.12, 4.0))): s.veil(f, t, H * y0, H * amp, H * hgt, 1.0, ph, 0.5)
            img = s.pose(t); s.draw(f, img, W / 2, cy, 0.46); s.hands(f, 1.0, t)
            c = ease2(u / 1.0)
        else:
            dawn = ease2(op / 1.3); f = s.snowfield(t, dawn, 1.0)
            sun = (W / 2, s.gnd - H * 0.02 - H * 0.58 * ease(op / 2.2))                 # 朝日と ハロ(光の輪)・幻日
            s.soft(f, sun, W * 0.10, 0.9 * dawn, WHITE * 0.8 + np.array([1.0, 0.8, 0.5], np.float32) * 0.2); L = s.L0(); cv2.circle(L, (int(sun[0]), int(sun[1])), int(W * 0.05), 1.0, -1, cv2.LINE_AA); f += L[:, :, None] * dawn
            R = W * 0.34
            for j, c in enumerate(((1.0, 0.55, 0.45), (1.0, 0.95, 0.75), (0.6, 0.8, 1.0))):
                Lr = s.L0(); cv2.circle(Lr, (int(sun[0]), int(sun[1])), int(R + (j - 1) * 4 * k), 0.5, max(1, int(3 * k)), cv2.LINE_AA)
                f += cv2.GaussianBlur(Lr, (0, 0), 2.5 * k)[:, :, None] * np.array(c, np.float32)[None, None, :] * dawn * 0.8
            L = s.L0()
            for sg in (-1, 1): p = (sun[0] + sg * R, sun[1]); s.star4(L, p, (18 + 4 * math.sin(t * 3)) * k, 1.0, 0.0, 0.12); s.line(L, p, (p[0] + sg * 70 * k, p[1]), 0.6, 2)
            cv2.ellipse(L, (int(sun[0]), int(sun[1] - R * 1.9)), (int(R * 1.1), int(R * 0.9)), 0, 60, 120, 0.5, 2, cv2.LINE_AA)
            s.add(f, L, WHITE, 0.9 * dawn)
            L = s.L0()
            for i in range(60):                                                      # ダイヤモンドダスト
                r = s.rd[i]; tw = math.sin(t * 4 + r[2] * 30) ** 8; s.star4(L, (r[0] * W, r[1] * H * 0.9), (2 + 5 * r[3]) * k * tw, 1.0, 0.0, 0.2)
            f += L[:, :, None] * dawn
            img = s.pose(t, squash=0.03 * math.exp(-op * 6)); s.draw(f, img, W / 2, cy, 0.46); s.hands(f, 0.6, t)
            c = 1 - ease2(op / 0.9)
        if c > 0:                                                                    # 舞台の幕: 七色の ひだ
            wdt = W * 0.5 * c; fold = 0.75 + 0.25 * np.sin(s.XX / (13 * k) + np.sin(s.YY / (120 * k)) * 1.5)
            hem = H * (0.90 + 0.02 * np.sin(s.XX / (26 * k)))
            for sg in (-1, 1):
                inside = (s.XX < wdt) if sg < 0 else (s.XX > W - wdt)
                rel = (s.XX / np.maximum(wdt, 1)) if sg < 0 else ((W - s.XX) / np.maximum(wdt, 1))
                idx = np.clip((rel * 7).astype(int), 0, 6); col = np.array(BOW, np.float32)[idx]
                m = (inside & (s.YY < hem)).astype(np.float32)[:, :, None]
                f = f * (1 - m * 0.93) + m * col * fold[:, :, None] * 0.80
            L = s.L0()
            if wdt > 2: s.line(L, (wdt, 0), (wdt, H * 0.9), 1.0, 3); s.line(L, (W - wdt, 0), (W - wdt, H * 0.9), 1.0, 3)
            s.add(f, L, WHITE, 0.9)
            if op < 0 and c >= 1:                                                    # 閉じきった 幕の すき間から 光
                q = (u - 1.0) / (T_OPEN - T_CLOSE - 1.0); L = s.L0(); s.line(L, (W / 2, 0), (W / 2, H * 0.9), 1.0, int(2 + 8 * q)); s.add(f, L, WHITE, 1.0 + q, 4, 18)
        if op > 0 and op < 0.15: f += (1 - op / 0.15) * 0.9
        end = DUR - t
        if end < 0.6: v = clamp((0.6 - end) / 0.5); f = f * (1 - v) + v
        return f, (4 * math.exp(-(u - 1.0) * 8) if 1.0 < u < 1.6 else 0) + (6 * math.exp(-op * 5) if op > 0 else 0)

    # ---- この子だけの曲: ガラスのような 高い音(グラスハープ)と 鈴。カーテンが1枚ひらくたび 和音が1つ ふえる。幕が閉じる間は ハープの 下りの グリッサンド、ひらくと 朝の 長い和音 ----
    def audio(s, path):
        a = Snd(DUR, 78); E4 = 329.63; lyd = [0, 2, 4, 6, 7, 9, 11]
        def fq(d, o=0): return E4 * 2 ** ((lyd[d % 7] + 12 * (d // 7 + o)) / 12.0)
        def glass(d, t0, dur, g=0.06, o=1): a.note(fq(d, o), t0, dur, g, "sine", 0.25); a.note(fq(d, o) * 2.003, t0, dur, g * 0.3, "sine", 0.3)
        a.a += a.lp(a.nz, 0.01) * 0.3 * a.gate(0, T_CUT, 0.4)                                                          # しずかな 風
        a.rise(0.1, T_LAND, 2400, 700, 0.05)                                                                           # 流れ星
        a.note(fq(7, 1), T_LAND, 1.0, 0.12, "bell"); a.boom(T_LAND, 70, 0.4, 0.25)
        for i in range(6): a.note(fq(i + 4, 2), T_LAND + 0.1 + i * 0.07, 0.25, 0.03, "bell")
        a.boom(T_CUT, 52, 0.8); a.note(fq(0, 2), T_CUT + 0.02, 0.6, 0.10, "sine", 0.01)
        for j in range(7): a.note(fq(j, 1), T_CUT + 0.32 + j * 0.05, 0.9, 0.07, "bell")                               # 七色
        for i in range(5): a.note(fq([7, 6, 4, 2, 0][i], 1), T_CUT + 0.75 + i * 0.13, 0.4, 0.09, "bell")
        chords = [(0, 4), (0, 2, 4), (0, 2, 4, 6)]
        for i, ch in enumerate(chords):                                                                                # カーテン 1枚ごとに
            tk = T_VEIL + 0.2 + i * 0.55
            for d in ch: glass(d, tk, 2.4 - i * 0.3, 0.05)
            a.whoosh(tk, tk + 0.9, 0.18, 0.08)
            for j in range(8): a.note(fq((j * 2 + i) % 12, 2), tk + j * 0.1, 0.2, 0.025, "bell")
        a.rise(T_TENT, T_TENT + 1.5, 300, 1200, 0.07)                                                                  # 天幕
        for j, d in enumerate([0, 2, 4, 6, 7, 9, 11, 14]): a.note(fq(d, 1), T_TENT + j * 0.17, 0.5, 0.07, "bell")
        a.boom(T_TENT + 1.5, 48, 0.8, 0.5); glass(7, T_TENT + 1.5, 1.2, 0.07, 1)
        for j in range(16): a.note(fq(18 - j, 1), T_CLOSE + j * 0.06, 0.25, 0.06, "bell")                              # 幕が 閉じる
        a.whoosh(T_CLOSE, T_CLOSE + 1.0, 0.4, 0.1); a.boom(T_CLOSE + 1.0, 60, 0.9, 0.25)
        a.note(fq(14, 2), T_CLOSE + 1.1, 0.5, 0.06, "sine", 0.02)
        a.boom(T_OPEN, 42, 1.1, 0.8)
        for j in range(16): a.note(fq(3 + j, 1), T_OPEN + j * 0.05, 0.3, 0.05, "bell")                                 # ひらく
        for j, (d, o) in enumerate(((0, -1), (4, -1), (0, 0), (2, 0), (4, 0), (6, 0), (7, 0))): a.note(fq(d, o), T_OPEN + 0.1 + j * 0.05, 3.0, 0.045, "pad", 0.5)
        for i, d in enumerate([4, 6, 7, 9, 11, 14]): a.note(fq(d, 1), T_OPEN + 0.8 + i * 0.32, 0.8, 0.07, "bell")
        a.save(path)


if __name__ == "__main__":
    r = Aurora()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
