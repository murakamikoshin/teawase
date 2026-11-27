"""バハル(★6 島を背負う海竜) 専用の必殺技「海呑ミ」
深い海へ 沈んでいく(上から さす光・のぼる泡) → うろこが 画面を おおう カットイン(すき間に 竜のかぶとの目) →
まっくらな底から、この子より大きな「目」が ひらく → さらに下で、画面より大きな あごが ひらく(歯の列) →
槍で 指図する → 泡も光も ぜんぶ 口へ 吸いこまれ、上下の歯が かみ合う(まっくら・ごくり) →
海の上。月の下で、巨大な背が 島を のせて 浮かび上がる
使い方: python3 hs_76.py 出力.mp4 / python3 hs_76.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 13.8
AQUA = np.array([0.25, 0.72, 1.0], np.float32); ABYSS = np.array([0.01, 0.04, 0.10], np.float32); GLOW = np.array([0.55, 1.0, 0.92], np.float32); BONE = np.array([0.92, 0.95, 1.0], np.float32)
T_CUT, T_EYE, T_JAW, T_ORDER, T_SNAP, T_SURF = 2.9, 5.6, 7.0, 8.9, 10.45, 11.0


class Bahr(Kit):
    def __init__(s, W=540):
        super().__init__(76, (0.25, 0.72, 1.0), "海呑ミ", "Ocean Devourer", W)
        s.eyeL = s.A(266, 346); s.eyeR = s.A(391, 322); s.hL = s.A(232, 236); s.hR = s.A(352, 202); s.tip = s.A(22, 702); s.hand = s.A(255, 535)
        s.vt = [s.text_img(c, hz.SERIF, int(110 * s.k), 3) for c in "海呑ミ"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        rg = np.random.RandomState(76); s.rd = rg.rand(300, 5).astype(np.float32)
        s.yn = s.YY / s.H; s.xn = s.XX / s.W

    # ---- この子だけの絵 ----
    def deep(s, t, depth=0.0, lum=1.0):
        """海の中。depth 0=浅い 1=底"""
        top = AQUA * (0.75 * (1 - depth) + 0.04); f = (top[None, None, :] * (1 - s.yn)[:, :, None] ** 1.5 + ABYSS[None, None, :] * (0.6 + s.yn)[:, :, None]) * lum
        L = s.L0()
        for i in range(6):                                                           # 上から さす光
            x = s.W * (0.1 + 0.16 * i) + math.sin(t * 0.5 + i * 1.3) * 25 * s.k; w = (18 + 14 * (i % 3)) * s.k
            cv2.fillPoly(L, [np.array([(x - w, 0), (x + w, 0), (x + w * 2.4 + 60 * s.k, s.H * 0.8), (x - w * 0.4 + 60 * s.k, s.H * 0.8)], np.int32)], 0.16, cv2.LINE_AA)
        f += (cv2.GaussianBlur(L, (0, 0), 9 * s.k) * (1 - s.yn) * (1 - depth) * lum)[:, :, None] * (AQUA * 0.8 + 0.2)[None, None, :]
        return f.astype(np.float32)

    def bubbles(s, f, t, n=50, suck=None, v=1.0):
        L = s.L0()
        for i in range(n):
            r = s.rd[i]; y = ((r[1] - t * (0.08 + 0.16 * r[2])) % 1.0) * s.H; x = (r[0] + 0.02 * math.sin(t * 2 + r[3] * 9)) * s.W
            if suck: q = suck[1]; x = x + (suck[0][0] - x) * q; y = y + (suck[0][1] - y) * q
            cv2.circle(L, (int(x), int(y)), max(1, int((2 + 5 * r[4]) * s.k)), float(0.5 * v), 1, cv2.LINE_AA)
        f += (L + cv2.GaussianBlur(L, (0, 0), 2 * s.k))[:, :, None] * (GLOW * 0.5 + 0.5)[None, None, :]

    def helm(s, f, v):
        """竜のかぶとの目が 光る"""
        if v <= 0: return
        L = s.L0(); sc = s._pl[2]
        for e in (s.hL, s.hR):
            p = s.at(*e); cv2.ellipse(L, (int(p[0]), int(p[1])), (max(1, int(15 * sc)), max(1, int(8 * sc))), -18, 0, 360, float(v), -1, cv2.LINE_AA)
        s.add(f, L, GLOW, 1.3, 2, 8)

    def big_eye(s, f, c, R, open_, t, look=(0, 0)):
        """この子より大きな 目"""
        if open_ <= 0: return
        k = s.k; m = s.L0(); cv2.ellipse(m, (int(c[0]), int(c[1])), (int(R), max(1, int(R * 0.62 * open_))), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        iris = s.L0(); ic = (c[0] + look[0], c[1] + look[1])
        for i in range(48): a = i * 0.1309; s.line(iris, (ic[0] + math.cos(a) * R * 0.14, ic[1] + math.sin(a) * R * 0.14), (ic[0] + math.cos(a) * R * 0.50, ic[1] + math.sin(a) * R * 0.50), 0.5 + 0.4 * (i % 2), 2)
        cv2.circle(iris, (int(ic[0]), int(ic[1])), int(R * 0.52), 0.6, 3, cv2.LINE_AA)
        body = (GLOW[None, None, :] * 0.10 * np.ones((s.H, s.W, 1), np.float32) + (iris + cv2.GaussianBlur(iris, (0, 0), 5 * k))[:, :, None] * GLOW[None, None, :])
        pup = s.L0(); cv2.ellipse(pup, (int(ic[0]), int(ic[1])), (max(1, int(R * 0.07)), int(R * 0.40)), 0, 0, 360, 1.0, -1, cv2.LINE_AA); body *= (1 - pup)[:, :, None]
        f[:] = f * (1 - m[:, :, None]) + body * m[:, :, None]
        L = s.L0(); cv2.ellipse(L, (int(c[0]), int(c[1])), (int(R), max(1, int(R * 0.62 * open_))), 0, 0, 360, 1.0, 3, cv2.LINE_AA); s.add(f, L, GLOW, 0.9, 3, 12)

    def jaws(s, f, gap, t, glow=1.0):
        """画面より大きな あご。gap=上下の歯の すき間(画面の高さに対する割合)"""
        W, H, k = s.W, s.H, s.k; mid = H * 0.56; D = s.L0(); T = s.L0(); n = 9
        for sg in (-1, 1):
            edge = mid + sg * gap * H / 2
            pts = [(-20, mid + sg * H)] + [(-20, edge)]
            for i in range(n + 1):
                x = W * i / n; bow = math.sin(i / n * 3.1416) * H * 0.05 * sg
                pts.append((x, edge + bow))
            pts += [(W + 20, edge), (W + 20, mid + sg * H)]
            cv2.fillPoly(D, [np.array(pts, np.int32)], 1.0, cv2.LINE_AA)
            for i in range(n):                                                       # 歯
                x0 = W * i / n; x1 = W * (i + 1) / n; b0 = math.sin(i / n * 3.1416) * H * 0.05 * sg; b1 = math.sin((i + 1) / n * 3.1416) * H * 0.05 * sg
                ln = H * (0.085 if i % 2 == 0 else 0.055)
                cv2.fillPoly(T, [np.array([(x0 + 3 * k, edge + b0), (x1 - 3 * k, edge + b1), ((x0 + x1) / 2, edge + (b0 + b1) / 2 - sg * ln)], np.int32)], 1.0, cv2.LINE_AA)
        f[:] = f * (1 - D * 0.985)[:, :, None]
        f[:] = f * (1 - T)[:, :, None] + T[:, :, None] * (BONE * (0.55 + 0.25 * glow))[None, None, :]
        f += cv2.GaussianBlur(T, (0, 0), 8 * k)[:, :, None] * GLOW[None, None, :] * 0.35 * glow
        return mid

    def frame(s, t):
        k = s.k
        if t < T_CUT: f, sh = s.sc_sink(t)
        elif t < T_CUT + 2.0: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_SURF: f, sh = s.sc_abyss(t)
        else: f, sh = s.sc_surface(t - T_SURF)
        f, sh = s.finale(f, sh, t, DUR, s.tip, (0.3, 0.85, 1), (0.2, 0.5, 1), 0)
        return s.post(f, AQUA, 0.5, sh * k, t, 0.90)

    # 1) 沈んでいく
    def sc_sink(s, t):
        W, H, k = s.W, s.H, s.k
        d = ease2(t / 2.6); f = s.deep(t, d * 0.8, clamp(t / 0.4))
        L = s.L0(); isl = (W * 0.5, -H * 0.02 - d * H * 0.25)                      # 水面の 島の影が 遠ざかる
        cv2.ellipse(L, (int(isl[0]), int(isl[1])), (int(W * 0.22), int(H * 0.05)), 0, 0, 360, 1.0, -1); f *= (1 - cv2.GaussianBlur(L, (0, 0), 6 * k) * 0.7)[:, :, None]
        cy = H * (0.18 + 0.36 * ease(t / 2.4))
        img = s.pose(t, squash=-0.03 * (1 - ease(t / 2.4)))
        s.draw(f, img, W / 2, cy, 0.45); s.helm(f, ease((t - 1.6) / 0.6))
        s.bubbles(f, t * 1.8, 60)
        L = s.L0()
        for i in range(14):                                                          # 体から はなれる 泡のすじ
            r = s.rd[i + 100]; x = W / 2 + (r[0] - 0.5) * W * 0.5; y = cy - (t * (0.5 + r[2]) % 1.0) * H * 0.5
            cv2.circle(L, (int(x), int(y)), max(1, int(3 * k)), 0.6, 1, cv2.LINE_AA)
        f += L[:, :, None] * 0.7
        return f, 0

    # 2) うろこが おおう カットイン
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        base = s.deep(t, 0.9, 0.5); cyb = H * 0.34
        win = np.zeros((H, W, 3), np.float32) + (AQUA * 0.25 + ABYSS)[None, None, :]
        ex = (s.hL[0] + s.eyeR[0]) / 2; ey = (s.hL[1] + s.eyeL[1]) / 2; zf = 1.35; sc = zf * H / s.ah
        img = s.pose(t); s.draw(win, img, W * 0.5 - (ex - s.aw / 2) * sc + u * 5 * k, cyb - (ey - s.ah / 2) * sc, zf); s.helm(win, 0.75 + 0.25 * math.sin(u * 10))
        S = s.L0(); E = s.L0(); R = W * 0.085                                         # うろこ(半円)が 下の段から ぱたぱた かぶさる
        rows = int(H / (R * 0.9)) + 2
        for j in range(rows):
            y = H - j * R * 0.9
            if abs(y - cyb) < H * 0.10: continue
            a = (u - (rows - j) * 0.012) / 0.18
            if a <= 0: continue
            for i in range(-1, int(W / (R * 2)) + 2):
                x = i * R * 2 + (R if j % 2 else 0); rr = R * 1.05 * ease(a)
                cv2.ellipse(S, (int(x), int(y)), (int(rr), int(rr)), 0, 0, 180, 1.0, -1, cv2.LINE_AA); cv2.ellipse(E, (int(x), int(y)), (int(rr), int(rr)), 0, 0, 180, 1.0, 2, cv2.LINE_AA)
        sc_col = (AQUA * 0.16 + ABYSS * 2)[None, None, :] * (0.7 + 0.6 * (1 - s.yn))[:, :, None]
        f = base * 0.3
        slit = (np.abs(s.YY - cyb) < H * 0.10).astype(np.float32)[:, :, None]; f = f * (1 - slit) + win * slit
        f = f * (1 - S[:, :, None]) + sc_col * S[:, :, None]; f += E[:, :, None] * (AQUA * 0.55)[None, None, :]
        L = s.L0(); s.line(L, (0, cyb - H * 0.10), (W, cyb - H * 0.10), 1.0, 2); s.line(L, (0, cyb + H * 0.10), (W, cyb + H * 0.10), 1.0, 2); s.add(f, L, GLOW, 0.8)
        for i, T in enumerate(s.vt):                                                 # 技名: 大きな3文字が 泡といっしょに 浮かび上がる
            a = (u - 0.5 - i * 0.2) / 0.4
            if a > 0:
                p = (W * (0.22 + i * 0.28), H * 0.68 + (1 - ease(a)) * 90 * k); s.text(f, T, p[0], p[1], clamp(a * 1.5), 1.0)
                L2 = s.L0()
                for j in range(5): r = s.rd[i * 5 + j]; cv2.circle(L2, (int(p[0] + (r[0] - 0.5) * 90 * k), int(p[1] - 60 * k - a * 120 * k * (0.5 + r[1]))), max(1, int((3 + 5 * r[2]) * k)), float(clamp(1.5 - a)), 1, cv2.LINE_AA)
                f += L2[:, :, None]
        for i, T in enumerate(s.vo):
            if u > 0.4 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.53, 1.0, 1.0)
        if u > 1.4: s.text(f, s.txt_en, W / 2, H * 0.68 + 96 * k, ease((u - 1.4) / 0.3), 1.0)
        if u < 0.08: f += (1 - u / 0.08) * 0.7
        return f, 0

    # 3) 底の「目」→ あご → 吸いこんで かみ合う
    def sc_abyss(s, t):
        W, H, k = s.W, s.H, s.k; eo = t - T_EYE; jo = t - T_JAW; od = t - T_ORDER; sn = t - T_SNAP
        if sn > 0:                                                                   # かみ合った あと: まっくら。歯の すじだけ
            f = np.zeros((H, W, 3), np.float32); L = s.L0(); mid = H * 0.56
            pts = [(W * i / 18.0, mid + (H * 0.045 if i % 2 else -H * 0.045)) for i in range(19)]
            cv2.polylines(L, [np.array(pts, np.int32)], False, clamp(1 - sn / 0.5), 2, cv2.LINE_AA); s.add(f, L, GLOW, 0.9)
            return f, 26 * math.exp(-sn * 6)
        f = s.deep(t, 1.0, 0.8)
        cy = H * 0.40; C = (W / 2, cy)
        E = (W * 0.76, H * 0.20)
        s.big_eye(f, E, W * 0.30, back(eo / 0.5, 1.2) if eo > 0 else 0, t, ((W * 0.5 - E[0]) * 0.10, (cy - E[1]) * 0.25))
        suck = clamp(od / (T_SNAP - T_ORDER)) ** 2 if od > 0 else 0.0
        if jo > 0:
            gap = 0.62 * ease2(jo / 1.3) * (1 - ease2((od - 0.5) / (T_SNAP - T_ORDER - 0.5)) if od > 0.5 else 1.0)
            mid = s.jaws(f, gap + 0.02, t, 1.0)
        else: mid = H * 0.56
        if od > 0:                                                                   # ぜんぶ 口の奥へ
            L = s.L0(); M = (W / 2, mid)
            for i in range(90):
                r = s.rd[i]; a = r[0] * 6.283 + od * 2.5 * (0.5 + r[2]); rr = W * (1.1 - ((od * (0.5 + r[1])) % 1.0) * 1.05)
                p = (M[0] + math.cos(a) * rr, M[1] + math.sin(a) * rr * 0.55); q = (M[0] + math.cos(a + 0.25) * rr * 0.86, M[1] + math.sin(a + 0.25) * rr * 0.55 * 0.86)
                s.line(L, p, q, 0.5 * suck + 0.2, 1)
            s.add(f, L, GLOW * 0.6 + AQUA * 0.4, 0.9)
        s.bubbles(f, t * 1.2, 50, ((W / 2, mid), suck) if od > 0 else None)
        rec = 0.08 * math.exp(-od * 7) if od > 0 else 0.0
        img = s.pose(t, lean=(-14 * k * ease(od / 0.2) if od > 0 else 0), squash=rec, jit=(1.5 * k * suck))
        s.soft(f, C, W * 0.26, 0.18, AQUA)
        s.draw(f, img, W / 2, cy + suck * H * 0.03, 0.40 * (1 - 0.12 * suck)); s.helm(f, 1.0)
        if od > 0:                                                                   # 槍の先から 指図の光
            tp = s.at(*s.tip); L = s.L0(); g = ease(od / 0.2)
            s.line(L, tp, (tp[0] - W * 0.25 * g, tp[1] + H * 0.14 * g), clamp(1.5 - od), 3); s.star4(L, tp, 16 * k * clamp(1.5 - od), 1.0, od * 4); s.add(f, L, GLOW, 1.2)
            if od < 0.08: f += (1 - od / 0.08) * 0.5
        if t - (T_CUT + 2.0) < 0.2: f *= (t - T_CUT - 2.0) / 0.2
        return f, (7 * math.exp(-eo * 6) if eo > 0 else 0) + (5 * clamp(jo / 1.3) * (jo > 0) * (od < 0)) * 0.5 + 6 * suck

    # 4) 海の上: 巨大な背が 島を のせて 浮かぶ
    def sc_surface(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_SURF
        f = (np.array([0.03, 0.06, 0.16], np.float32)[None, None, :] * (0.5 + 1.0 * s.yn)[:, :, None]).astype(np.float32)
        M = (W * 0.76, H * 0.16); s.soft(f, M, W * 0.16, 0.5, BONE); L = s.L0(); cv2.circle(L, (int(M[0]), int(M[1])), int(W * 0.075), 1.0, -1, cv2.LINE_AA); f += L[:, :, None] * BONE[None, None, :]
        L = s.L0()
        for i in range(60): r = s.rd[i + 200]; cv2.circle(L, (int(r[0] * W), int(r[1] * H * 0.5)), 1, float(0.4 + 0.6 * math.sin(t * 2 + r[2] * 20) ** 2), -1)
        f += L[:, :, None]
        sea = H * 0.70; rise = ease(u / 1.6)
        hump = s.L0(); hc = (W * 0.5, sea + H * 0.30 - rise * H * 0.22)             # 背
        cv2.ellipse(hump, (int(hc[0]), int(hc[1])), (int(W * 0.62), int(H * 0.24)), 0, 180, 360, 1.0, -1, cv2.LINE_AA)
        f = f * (1 - hump[:, :, None]) + hump[:, :, None] * (np.array([0.02, 0.06, 0.12], np.float32) + AQUA * 0.10)[None, None, :]
        L = s.L0(); cv2.ellipse(L, (int(hc[0]), int(hc[1])), (int(W * 0.62), int(H * 0.24)), 0, 180, 360, 1.0, 2, cv2.LINE_AA)
        for i in range(7): a = 3.4 + i * 0.42; cv2.ellipse(L, (int(hc[0] + math.cos(a) * W * 0.45), int(hc[1] + math.sin(a) * H * 0.17)), (int(22 * k), int(22 * k)), 0, 200, 340, 0.5, 2, cv2.LINE_AA)      # うろこ
        ey = (hc[0] - W * 0.40, hc[1] - H * 0.085); cv2.ellipse(L, (int(ey[0]), int(ey[1])), (int(15 * k), int(7 * k)), 20, 0, 360, 1.0, -1, cv2.LINE_AA)                                              # 目
        s.add(f, L, GLOW, 0.6 * rise, 2, 8)
        top = hc[1] - H * 0.24
        isl = s.L0(); cv2.ellipse(isl, (int(W * 0.56), int(top + 6 * k)), (int(W * 0.17), int(H * 0.035)), 0, 180, 360, 1.0, -1, cv2.LINE_AA)                                                          # 島 と やしの木
        f = f * (1 - isl[:, :, None]) + isl[:, :, None] * np.array([0.20, 0.34, 0.16], np.float32)[None, None, :]
        px = W * 0.64; cv2.line(f, (int(px), int(top - 14 * k)), (int(px + 8 * k), int(top - 62 * k)), (0.25, 0.16, 0.09), max(1, int(4 * k)), cv2.LINE_AA)
        for a in (-2.6, -2.0, -1.2, -0.5, 0.2): cv2.ellipse(f, (int(px + 8 * k + math.cos(a) * 15 * k), int(top - 62 * k + math.sin(a) * 9 * k)), (int(17 * k), int(5 * k)), math.degrees(a), 0, 360, (0.15, 0.42, 0.18), -1, cv2.LINE_AA)
        img = s.pose(t, squash=0.05 * math.exp(-max(0, u - 1.6) * 8) * (u > 1.6)); s.draw(f, img, W * 0.44, top - H * 0.105, 0.24); s.helm(f, 0.8)
        wl = np.clip((s.YY - (sea + np.sin(s.XX / W * 16 + t * 1.5) * 4 * k)) / 2.0, 0, 1)                                                                                                                # 海面
        f = f * (1 - wl[:, :, None] * 0.9) + wl[:, :, None] * (AQUA * 0.14 + ABYSS)[None, None, :] * (1.4 - s.yn)[:, :, None] * 0.9
        L = s.L0()
        for i in range(12):                                                          # 背から 流れ落ちる水
            x = hc[0] + (i - 5.5) * W * 0.10; y0 = hc[1] - math.sqrt(max(0, 1 - ((x - hc[0]) / (W * 0.62)) ** 2)) * H * 0.24
            if y0 < sea and i % 2 == 0: q = ((u * 0.9 + s.rd[i][0]) % 1.0); s.line(L, (x + (x - hc[0]) * 0.05 * q, y0 + (sea - y0) * q), (x + (x - hc[0]) * 0.05 * (q + 0.18), y0 + (sea - y0) * min(1, q + 0.18)), 0.4 * clamp(2.4 - u), 2)
        for i in range(3):
            v = (u - 0.3 - i * 0.3) / 1.4
            if 0 < v < 1: s.ell(L, (W / 2, sea), (W * 0.9 * v, H * 0.03 * v), 0, 1 - v, 2)
        mx = M[0]
        for j in range(9): s.line(L, (mx - (30 - j * 2) * k, sea + (8 + j * 9) * k), (mx + (30 - j * 2) * k, sea + (8 + j * 9) * k), 0.35, 1)            # 月のみち
        s.add(f, L, BONE * 0.6 + AQUA * 0.4, 0.8)
        if u < 0.35: f = f * (u / 0.35)
        end = DUR - t
        if end < 0.5: f = f * clamp(end / 0.45)
        return f, 6 * (1 - rise) * (u < 1.6)

    # ---- この子だけの曲: 深い海の うなり。くじらの声(ゆっくり下がる音)。あごが ひらく間は 低い 2音の くり返し。かみ合ったあとは 月の下の しずかな 旋律 ----
    def audio(s, path):
        a = Snd(DUR, 76); A1 = 55.0; aeo = [0, 2, 3, 5, 7, 8, 10]
        def fq(d, o=0): return A1 * 4 * 2 ** ((aeo[d % 7] + 12 * (d // 7 + o)) / 12.0)
        def whale(t0, f0, f1, dur, g=0.12):
            u = np.clip((a.t - t0) / dur, 0, 1); fr = f0 + (f1 - f0) * u; ph = 2 * np.pi * np.cumsum(fr) / a.sr
            a.a += (np.sin(ph) + 0.4 * np.sin(2 * ph)) * np.sin(u * np.pi) ** 1.5 * ((a.t >= t0) & (a.t < t0 + dur)) * g
        a.a += a.lp(a.nz, 0.008) * 1.2 * a.gate(0, T_SNAP, 0.5)                                                      # 水の 重さ
        a.a += a.tone(A1) * 0.10 * a.gate(0.5, T_SNAP, 0.6)
        for i in range(12): a.note(900 + (i * 137) % 700, 0.3 + i * 0.2, 0.05, 0.03, "sine", 0.002)                  # 泡
        whale(0.8, 420, 260, 1.4, 0.07)
        a.boom(T_CUT, 46, 0.9)
        for i in range(10): a.hat(T_CUT + i * 0.03, 0.04, 0.02)
        for i in range(3): a.note(fq([4, 2, 0][i], 0), T_CUT + 0.5 + i * 0.2, 0.9, 0.10, "bell"); a.boom(T_CUT + 0.5 + i * 0.2, 60, 0.4, 0.2)
        a.boom(T_EYE, 40, 1.0, 0.7); whale(T_EYE, 180, 95, 1.3, 0.16)
        tk = T_JAW; i = 0
        while tk < T_ORDER:                                                          # あご: 低い 2音が せまる
            a.note(fq([0, 1][i % 2], -2), tk, 0.3, 0.22, "saw", 0.02); a.boom(tk, 44, 0.45, 0.2); tk += max(0.16, 0.42 - i * 0.035); i += 1
        whale(T_JAW, 110, 60, 1.6, 0.14)
        a.note(fq(7, 1), T_ORDER, 0.5, 0.12, "bell"); a.rise(T_ORDER, T_SNAP, 70, 900, 0.20); a.whoosh(T_ORDER, T_SNAP, 0.9, 0.3)
        a.boom(T_SNAP, 34, 1.7, 0.6); a.a += (a.nz - a.lp(a.nz, 0.5)) * a.env(T_SNAP, 0.001, 0.05) * 0.6               # かみ合う
        a.boom(T_SNAP + 0.3, 30, 0.8, 0.35)                                                                           # ごくり
        a.a += a.lp(a.nz, 0.03) * a.env(T_SURF, 0.3, 1.6) * 0.7                                                       # 波
        for j, (d, o) in enumerate(((0, -1), (4, -1), (0, 0), (2, 0), (4, 0))): a.note(fq(d, o), T_SURF + j * 0.06, 2.6, 0.045, "pad", 0.5)
        for i, d in enumerate([4, 7, 9, 7, 11, 9, 7]): a.note(fq(d, 0), T_SURF + 0.4 + i * 0.3, 0.7, 0.085, "bell")
        whale(T_SURF + 0.3, 300, 380, 1.8, 0.06)
        s.fin_snd(a, DUR, 76); a.save(path)


if __name__ == "__main__":
    r = Bahr()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
