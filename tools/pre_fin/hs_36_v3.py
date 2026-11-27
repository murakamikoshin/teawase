"""ヤマタ(★5 八つ首の蛇姫) 専用の必殺技「八岐ノ顎」
夜の沼(もや・あぶく・あしの影)に 浮かぶ → 蛇の目が ひとつずつ 赤く灯り、「一、二、三…八」と 数字が 出る(だんだん 速く) →
八角形の窓のカットイン → 八つの口が 毒を 弧にして はく → 闇の中、八方から 大蛇が うずを巻いて まん中へ → かみ合って しめ上げ、毒が はじける →
沼から 毒の柱が 8本 ふき上がる
使い方: python3 hs_36.py 出力.mp4 / python3 hs_36.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 11.5
VEN = np.array([0.74, 0.32, 1.0], np.float32); RED = np.array([1.0, 0.16, 0.14], np.float32); MOS = np.array([0.35, 0.85, 0.40], np.float32); GOLD = np.array([1.0, 0.84, 0.40], np.float32)
T_CNT, T_CUT, T_SPIT, T_VOID, T_BITE, T_BURST, T_END = 1.8, 3.6, 5.5, 6.4, 7.3, 7.75, 8.3
CNT = [T_CNT + v for v in np.cumsum([0, .30, .28, .25, .22, .19, .16, .13])]
HEADS = [(128, 160), (163, 78), (322, 47), (462, 22), (558, 97), (620, 178), (580, 280), (148, 240)]


class Yamata(Kit):
    def __init__(s, W=540):
        super().__init__(36, (0.74, 0.32, 1.0), "八岐ノ顎", "Eight-Headed Maw", W)
        H, k = s.H, s.k
        s.eL = s.A(310, 300); s.eR = s.A(430, 290); s.fc = s.A(370, 300); s.hub = s.A(360, 230); s.stp = s.A(30, 410)
        s.hd = [s.A(*p) for p in HEADS]
        s.ht = [s.text_img(c, hz.SERIF, int(84 * k), 3) for c in "八岐ノ顎"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * k), 1) for c in "奥義"]
        s.num = [s.text_img(c, hz.SERIF, int(30 * k), 1) for c in "一二三四五六七八"]
        s.HF = 0.46; s.CX = W / 2; s.CY = H * 0.48; s.hz_y = H * 0.66
        rg = np.random.RandomState(36); s.mist = rg.rand(24, 80).astype(np.float32); s.bub = rg.rand(14, 4).astype(np.float32); s.dr = rg.rand(120, 5).astype(np.float32)
        s.yn = s.YY / H
        m = s.L0()                                                                   # あしの影
        for i in range(26):
            x = W * rg.rand(); x = x * 0.3 if i % 2 else W - x * 0.3; y = H * (0.80 + 0.2 * rg.rand()); ln = H * (0.10 + 0.14 * rg.rand()); a = (rg.rand() - 0.5) * 0.5
            cv2.line(m, (int(x), int(y)), (int(x + math.sin(a) * ln), int(y - math.cos(a) * ln)), 1.0, max(1, int(2.5 * k)), cv2.LINE_AA)
            if i % 3 == 0: cv2.ellipse(m, (int(x + math.sin(a) * ln), int(y - math.cos(a) * ln)), (int(4 * k), int(13 * k)), math.degrees(a), 0, 360, 1.0, -1, cv2.LINE_AA)
        s.reed = m
        s.wang = np.arctan2(s.YY - H * 0.46, s.XX - W / 2)

    # ---- 大道具 ----
    def swamp(s, t, lit=0.0):
        H, W, k = s.H, s.W, s.k
        sky = np.array([0.05, 0.03, 0.09], np.float32)[None, None, :] * (1 - s.yn)[:, :, None] + np.array([0.20, 0.13, 0.28], np.float32)[None, None, :] * s.yn[:, :, None]
        wat = np.array([0.03, 0.07, 0.06], np.float32)[None, None, :] * np.clip(1.6 - s.yn, 0, 1)[:, :, None]
        wm = (s.YY > s.hz_y).astype(np.float32)[:, :, None]; f = (sky * (1 - wm) + wat * wm).astype(np.float32)
        off = int(t * 5) % 80; fr = t * 5 - int(t * 5)
        n = cv2.resize(np.roll(s.mist, -off, 1), (W + int(W / 80.0) + 2, H), interpolation=cv2.INTER_CUBIC)[:, int(fr * W / 80.0):int(fr * W / 80.0) + W]
        band = np.exp(-((s.YY - s.hz_y) / (H * 0.10)) ** 2)
        f += (np.clip(n, 0, 1) * band * 0.22)[:, :, None] * (VEN * 0.6 + 0.3 + lit * 0.3)[None, None, :]
        L = s.L0()
        for r in s.bub:                                                              # あぶくの 輪
            v = (t * (0.35 + 0.3 * r[2]) + r[3]) % 1.0; x = r[0] * W; y = s.hz_y + (0.06 + 0.3 * r[1]) * H
            cv2.ellipse(L, (int(x), int(y)), (max(1, int(30 * k * v)), max(1, int(8 * k * v))), 0, 0, 360, float(0.5 * (1 - v)), 1, cv2.LINE_AA)
        f += L[:, :, None] * (MOS * 0.7 + 0.2)[None, None, :]
        return f

    def reeds(s, f): return f * (1 - s.reed[:, :, None] * 0.93)

    def hero(s, f, t, dx=0.0, **kw):
        img = s.pose(t, **kw); s.draw(f, img, s.CX + dx, s.CY, s.HF)

    def eyes(s, f, n, v=1.0, own=0.0):
        """蛇の目を n個 灯す。own: 本人の目"""
        sc = s._pl[2]; L = s.L0()
        for i in range(min(8, n)):
            p = s.at(*s.hd[i]); cv2.circle(L, (int(p[0]), int(p[1])), max(1, int(7 * sc)), float(v), -1, cv2.LINE_AA)
        if own > 0.01:
            for e in (s.eL, s.eR): p = s.at(*e); cv2.circle(L, (int(p[0]), int(p[1])), max(1, int(13 * sc)), float(min(1.0, own)), -1, cv2.LINE_AA)
        if n > 0 or own > 0.01: s.add(f, L, RED, 0.75, 2, 7)

    def frame(s, t):
        if t < T_CUT: f, sh = s.sc_count(t)
        elif t < T_SPIT: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_VOID: f, sh = s.sc_spit(t - T_SPIT)
        elif t < T_END: f, sh = s.sc_void(t)
        else: f, sh = s.sc_end(t - T_END)
        f, sh = s.finale(f, sh, t, DUR, s.stp, (0.74, 0.32, 1), (0.35, 0.85, 0.4), -1)
        return s.post(f, VEN * 0.6 + 0.4, 0.4, sh * s.k, t, 0.86)

    # 1) 沼に 浮かぶ。目が ひとつずつ 灯る
    def sc_count(s, t):
        W, H, k = s.W, s.H, s.k
        f = s.swamp(t); f = s.reeds(f); s.hero(f, t)
        n = sum(1 for c in CNT if t >= c)
        s.eyes(f, n, 1.0, clamp((t - 1.0) / 0.5) * 0.8)
        hub = s.at(*s.hub)
        for i in range(n):                                                           # 数字
            p = s.at(*s.hd[i]); d = math.hypot(p[0] - hub[0], p[1] - hub[1]) + 1e-3; v = t - CNT[i]
            q = (p[0] + (p[0] - hub[0]) / d * 34 * k, p[1] + (p[1] - hub[1]) / d * 34 * k - 4 * k)
            s.text(f, s.num[i], min(max(q[0], 30 * k), W - 30 * k), max(q[1], 28 * k), clamp(v / 0.08), 1.0)
            if v < 0.15:
                L = s.L0(); s.circ(L, p, (8 + 60 * v / 0.15) * k, 1 - v / 0.15, 2); s.add(f, L, RED, 0.8)
        return f * clamp(t / 0.4), 0

    # 2) 八角形の窓
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        C = (W / 2, H * 0.38); wa = np.arctan2(s.YY - C[1], s.XX - C[0])
        wed = (np.floor((wa + t * 0.25) / 0.7854) % 2).astype(np.float32)
        f = (wed[:, :, None] * np.array([0.10, 0.04, 0.16], np.float32) + (1 - wed[:, :, None]) * np.array([0.03, 0.09, 0.06], np.float32)).astype(np.float32)
        g = back(u / 0.3, 1.3); R = W * 0.44 * g
        pts = [(C[0] + math.cos(0.3927 + i * 0.7854) * R, C[1] + math.sin(0.3927 + i * 0.7854) * R) for i in range(8)]
        m = s.L0(); cv2.fillPoly(m, [np.array(pts, np.int32)], 1.0, cv2.LINE_AA)
        win = np.zeros((H, W, 3), np.float32) + np.array([0.13, 0.06, 0.20], np.float32)
        zf = 0.98 + 0.05 * ease(u / 1.9); sc = zf * H / s.ah; img = s.pose(t)
        s.draw(win, img, C[0] - (s.fc[0] - s.aw / 2) * sc, C[1] + H * 0.035 - (s.fc[1] - s.ah / 2) * sc, zf)
        s.eyes(win, 8, 1.0, 0.9 + 1.0 * math.exp(-max(0.0, u - 0.45) * 5) * (u > 0.45))
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        cv2.polylines(f, [np.array(pts, np.int32)], True, (0.90, 0.74, 0.30), max(1, int(6 * k)), cv2.LINE_AA)
        for i, p in enumerate(pts):                                                  # 八つの かどに 赤い玉が 順に 灯る
            if u > 0.3 + i * 0.07: s.soft(f, p, 12 * k, 0.9, RED); cv2.circle(f, (int(p[0]), int(p[1])), max(1, int(5 * k)), (1.0, 0.5, 0.45), -1, cv2.LINE_AA)
        n = len(s.ht); step = W * 0.20
        for i, T in enumerate(s.ht):                                                 # 技名: かみつくように 上下から とじる
            a = (u - 0.6 - i * 0.12) / 0.18
            if a > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.835, clamp(a * 2), ease(a))
        for i, T in enumerate(s.vo):
            if u > 0.45 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.755, 1.0, 1.0)
        if u > 1.3: s.text(f, s.txt_en, W / 2, H * 0.835 + 62 * k, ease((u - 1.3) / 0.3), 1.0)
        if u < 0.08: f = f + (1 - u / 0.08) * 0.7
        return f, 0

    # 3) 八つの口が 毒を はく
    def sc_spit(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_SPIT
        f = s.swamp(t, 0.5); f = s.reeds(f)
        s.hero(f, t, squash=0.05 * math.exp(-u * 8)); s.eyes(f, 8, 1.0, 1.0)
        hub = s.at(*s.hub); L = s.L0()
        for i in range(8):
            v = u - i * 0.05
            if v <= 0: continue
            p = s.at(*s.hd[i]); d = math.hypot(p[0] - hub[0], p[1] - hub[1]) + 1e-3; vx = (p[0] - hub[0]) / d * W * 1.5; vy = (p[1] - hub[1]) / d * W * 1.5 - W * 0.9
            pts = []
            for j in range(14):
                tau = max(0.0, v - 0.30 * (1 - j / 13.0)); pts.append((p[0] + vx * tau, p[1] + vy * tau + W * 1.6 * tau * tau))
            for j in range(13): cv2.line(L, (int(pts[j][0]), int(pts[j][1])), (int(pts[j + 1][0]), int(pts[j + 1][1])), float(0.35 + 0.65 * j / 13.0), max(1, int((2 + 5 * j / 13.0) * k)), cv2.LINE_AA)
            cv2.circle(L, (int(pts[-1][0]), int(pts[-1][1])), max(1, int(8 * k)), 1.0, -1, cv2.LINE_AA)
            if v < 0.1: cv2.circle(L, (int(p[0]), int(p[1])), max(1, int(16 * k * (1 - v / 0.1))), 1.0, -1, cv2.LINE_AA)
        s.add(f, L, VEN, 0.85, 2, 8)
        if u < 0.07: f = f + (1 - u / 0.07) * 0.4 * VEN[None, None, :]
        w = u - (T_VOID - T_SPIT - 0.15)
        if w > 0: f = f * (1 - w / 0.15)
        return f, 3 * math.exp(-u * 8)

    def serpent(s, f, C, th0, t, g, twist, r_end, shade):
        """うずを巻いて まん中へ 向かう 大蛇 1ぴき"""
        W, H, k = s.W, s.H, s.k; Ro = H * 0.62; rh = r_end + (Ro - r_end) * (1 - g); pts = []
        for j in range(34):
            r = rh + H * 0.95 * j / 33.0; a = th0 + twist * (1 - r / Ro) + 0.16 * math.sin(r / (46 * k) - t * 9) * min(1.0, (r - rh) / (80 * k) + 0.2)
            pts.append((C[0] + math.cos(a) * r, C[1] + math.sin(a) * r))
        P = np.array(pts, np.int32); body = (0.10 * shade, 0.34 * shade, 0.14 * shade)
        cv2.polylines(f, [P], False, (0.0, 0.02, 0.0), int(54 * k), cv2.LINE_AA); cv2.polylines(f, [P], False, body, int(44 * k), cv2.LINE_AA)
        cv2.polylines(f, [P], False, (0.16 * shade, 0.50 * shade, 0.20 * shade), int(14 * k), cv2.LINE_AA)
        h = pts[0]; ta = math.atan2(pts[0][1] - pts[2][1], pts[0][0] - pts[2][0]); ca, sa = math.cos(ta), math.sin(ta)
        def R(x, y): return (h[0] + x * ca - y * sa, h[1] + x * sa + y * ca)
        op = 0.25 + 0.75 * clamp((g - 0.6) / 0.35) * (1 - clamp((g - 0.97) / 0.03))                    # 口: 近づくと ひらき、かむと とじる
        hl = 84 * k; hw = 40 * k
        cv2.fillPoly(f, [np.array([R(-10 * k, -hw), R(hl * 0.5, -hw * 0.9), R(hl, -hw * 0.9 * op - 3 * k), R(4 * k, 0)], np.int32)], (0.0, 0.02, 0.0), cv2.LINE_AA)
        cv2.fillPoly(f, [np.array([R(-10 * k, hw), R(hl * 0.5, hw * 0.9), R(hl, hw * 0.9 * op + 3 * k), R(4 * k, 0)], np.int32)], (0.0, 0.02, 0.0), cv2.LINE_AA)
        cv2.fillPoly(f, [np.array([R(-6 * k, -hw + 4 * k), R(hl * 0.5, -hw * 0.9 + 4 * k), R(hl - 5 * k, -hw * 0.9 * op), R(6 * k, -2 * k)], np.int32)], body, cv2.LINE_AA)
        cv2.fillPoly(f, [np.array([R(-6 * k, hw - 4 * k), R(hl * 0.5, hw * 0.9 - 4 * k), R(hl - 5 * k, hw * 0.9 * op), R(6 * k, 2 * k)], np.int32)], body, cv2.LINE_AA)
        if op > 0.3: cv2.fillPoly(f, [np.array([R(8 * k, 0), R(hl * 0.7, -hw * 0.6 * op), R(hl * 0.7, hw * 0.6 * op)], np.int32)], (0.70, 0.10, 0.12), cv2.LINE_AA)
        for sg in (-1, 1):
            e = R(hl * 0.32, sg * hw * 0.62); cv2.circle(f, (int(e[0]), int(e[1])), max(1, int(8 * k)), (1.0, 0.25, 0.2), -1, cv2.LINE_AA)
            fg = [R(hl * 0.85, sg * hw * 0.8 * op), R(hl * 0.95, sg * hw * 0.8 * op), R(hl * 0.9, sg * (hw * 0.8 * op - 12 * k))]; cv2.fillPoly(f, [np.array(fg, np.int32)], (0.95, 0.95, 0.85), cv2.LINE_AA)
        return h

    # 4) 闇の中、八方から うずを巻いて → かみ合う → しめ上げて はじける
    def sc_void(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_VOID; bt = t - T_BITE; bu = t - T_BURST
        C = (W / 2, H * 0.46)
        wed = (np.floor((s.wang + t * 0.5) / 0.7854) % 2).astype(np.float32)
        f = (np.array([0.04, 0.02, 0.07], np.float32)[None, None, :] + wed[:, :, None] * np.array([0.03, 0.015, 0.05], np.float32)[None, None, :]).astype(np.float32)
        sh = 0
        if bu < 0:
            g = ease2(u / (T_BITE - T_VOID)); tw = 2.3 + (2.6 * ease2(bt / (T_BURST - T_BITE)) if bt > 0 else 0); re = W * 0.24 * (1 - 0.45 * clamp(bt / (T_BURST - T_BITE))) if bt > 0 else W * 0.24
            for i in range(8): s.serpent(f, C, i * 0.7854 + t * 0.9, t, g, tw, re, 1.0 if i % 2 else 0.78)
            s.soft(f, C, W * 0.10 * g, 0.5 * g + (0.5 * clamp(bt / 0.4) if bt > 0 else 0), VEN)
            if 0 <= bt < 0.1: f = f + (1 - bt / 0.1) * 0.7
            if bt > 0: sh = 4 + 6 * clamp(bt / (T_BURST - T_BITE))
        else:                                                                        # 毒が 八つに 裂けて はじける
            L = s.L0()
            for i in range(8):
                a = i * 0.7854 + 0.39; s.line(L, C, (C[0] + math.cos(a) * H * ease(bu / 0.12), C[1] + math.sin(a) * H * ease(bu / 0.12)), max(0.0, 1 - bu / 0.5), max(1, int(5 * k)))
            for j in range(3):
                v = (bu - j * 0.08) / 0.5
                if 0 < v < 1: s.circ(L, C, W * 1.0 * ease(v), 1 - v, 4)
            s.add(f, L, VEN, 1.0, 3, 12)
            for r in s.dr:
                a = r[0] * 6.283; d = (0.2 + 0.8 * r[1]) * W * 1.1 * (1 - math.exp(-bu * 3.2)); p = (C[0] + math.cos(a) * d, C[1] + math.sin(a) * d + bu * bu * 260 * k * r[2])
                col = VEN * (0.6 + 0.4 * r[3]) * clamp(1.6 - bu * 2.2); cv2.circle(f, (int(p[0]), int(p[1])), max(1, int((4 + 12 * r[4]) * k * (1 - 0.5 * bu))), (float(col[0]), float(col[1]), float(col[2])), -1, cv2.LINE_AA)
            if bu < 0.12: f = f + (1 - bu / 0.12) * 1.0
            sh = 18 * math.exp(-bu * 6)
            w = t - (T_END - 0.15)
            if w > 0: f = f * (1 - w / 0.15) + (w / 0.15) * VEN[None, None, :] * 0.5
        if u < 0.12: f = f * (u / 0.12)
        return f, sh

    def bite(s, f, c, S, ang, op, shade=1.0):
        """こちらを かこむ 蛇の頭(ang の向きに 口)。丸い頭・かげ・光る目・長いきば。op: 口の ひらき"""
        ca, sa = math.cos(ang), math.sin(ang); dg = math.degrees(ang)
        def R(x, y): return (int(c[0] + (x * ca - y * sa) * S), int(c[1] + (x * sa + y * ca) * S))
        def col(r, g, b): return (float(r * shade), float(g * shade), float(b * shade))
        K = (0.0, 0.02, 0.0); G0 = col(0.06, 0.26, 0.11); G1 = col(0.11, 0.40, 0.17); G2 = col(0.22, 0.60, 0.27); BEL = col(0.78, 0.80, 0.46)
        cv2.line(f, R(-3.4, 0), R(-0.6, 0), K, max(2, int(S * 1.56)), cv2.LINE_AA); cv2.line(f, R(-3.4, 0), R(-0.6, 0), G1, max(2, int(S * 1.40)), cv2.LINE_AA)       # 首
        cv2.line(f, R(-3.4, -0.36), R(-0.7, -0.36), G2, max(1, int(S * 0.28)), cv2.LINE_AA); cv2.line(f, R(-3.4, 0.46), R(-0.7, 0.46), G0, max(1, int(S * 0.36)), cv2.LINE_AA)
        for j in range(5): cv2.ellipse(f, R(-3.1 + j * 0.55, 0), (max(1, int(S * 0.16)), max(1, int(S * 0.62))), dg, -70, 70, G0, max(1, int(S * 0.05)), cv2.LINE_AA)     # うろこの 段
        cv2.ellipse(f, R(0, 0), (int(S * 1.24), int(S * 1.02)), dg, 0, 360, K, -1, cv2.LINE_AA); cv2.ellipse(f, R(0, 0), (int(S * 1.16), int(S * 0.94)), dg, 0, 360, G1, -1, cv2.LINE_AA)   # 頭
        cv2.ellipse(f, R(-0.15, 0.30), (int(S * 0.95), int(S * 0.55)), dg, 0, 360, G0, -1, cv2.LINE_AA); cv2.ellipse(f, R(-0.2, -0.34), (int(S * 0.78), int(S * 0.36)), dg, 0, 360, G2, -1, cv2.LINE_AA)   # かげ・てり
        mw = S * (0.34 + 0.34 * op); mh = S * (0.20 + 0.60 * op); mc = R(0.52, 0)                                             # 口
        cv2.ellipse(f, mc, (int(mw + S * 0.07), int(mh + S * 0.07)), dg, 0, 360, K, -1, cv2.LINE_AA); cv2.ellipse(f, mc, (int(mw), int(mh)), dg, 0, 360, (0.62, 0.08, 0.12), -1, cv2.LINE_AA)
        cv2.ellipse(f, R(0.40, 0), (int(mw * 0.62), int(mh * 0.66)), dg, 0, 360, (0.30, 0.02, 0.06), -1, cv2.LINE_AA); cv2.ellipse(f, R(0.30, 0), (int(mw * 0.30), int(mh * 0.34)), dg, 0, 360, (0.08, 0.0, 0.02), -1, cv2.LINE_AA)
        if op > 0.25:                                                                # ふたまたの 舌
            cv2.line(f, R(0.3, 0), R(0.95, 0), (0.95, 0.30, 0.36), max(1, int(S * 0.07)), cv2.LINE_AA)
            for sg in (-1, 1): cv2.line(f, R(0.95, 0), R(1.22, sg * 0.14), (0.95, 0.30, 0.36), max(1, int(S * 0.05)), cv2.LINE_AA)
        for sg in (-1, 1):
            yy = mh / S
            cv2.fillPoly(f, [np.array([R(0.62, sg * (yy + 0.02)), R(0.92, sg * (yy * 0.86)), R(0.80, sg * (yy - 0.52 * (0.5 + 0.5 * op)))], np.int32)], K, cv2.LINE_AA)        # きば(ふち)
            cv2.fillPoly(f, [np.array([R(0.66, sg * yy), R(0.88, sg * (yy * 0.88)), R(0.80, sg * (yy - 0.46 * (0.5 + 0.5 * op)))], np.int32)], (0.98, 0.97, 0.88), cv2.LINE_AA)
            cv2.fillPoly(f, [np.array([R(0.34, sg * (yy * 0.98)), R(0.50, sg * (yy * 1.0)), R(0.42, sg * (yy - 0.22))], np.int32)], (0.98, 0.97, 0.88), cv2.LINE_AA)
            e = R(-0.42, sg * 0.60)                                                  # 目: 赤く光る・たての ひとみ・白い点
            cv2.circle(f, e, max(1, int(S * 0.24)), K, -1, cv2.LINE_AA); cv2.circle(f, e, max(1, int(S * 0.19)), (1.0, 0.20, 0.16), -1, cv2.LINE_AA); cv2.circle(f, e, max(1, int(S * 0.12)), (1.0, 0.62, 0.30), -1, cv2.LINE_AA)
            cv2.ellipse(f, e, (max(1, int(S * 0.15)), max(1, int(S * 0.04))), dg, 0, 360, K, -1, cv2.LINE_AA); cv2.circle(f, R(-0.48, sg * 0.60 - 0.07), max(1, int(S * 0.04)), (1.0, 1.0, 1.0), -1, cv2.LINE_AA)
            cv2.line(f, R(-0.20, sg * 0.34), R(-0.02, sg * 0.84), K, max(1, int(S * 0.10)), cv2.LINE_AA)                      # 目の前の うね(にらみ)
            cv2.circle(f, R(0.98, sg * (yy + 0.16)), max(1, int(S * 0.04)), K, -1, cv2.LINE_AA)                               # 鼻の穴

    # 5) 八つの頭が 画面の ふちから こちらを かこみ、いっせいに かみつく。まん中の 1ぴきが 口を あけて のみこむ
    def sc_end(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_END
        f = s.swamp(t, 1.0); f = s.reeds(f)
        lg = math.exp(-u * 4.5); g = clamp((u - 0.25) / 2.2); e = g ** 1.15
        if g < 0.8: s.hero(f, t, dx=-W * 0.06 * lg, lean=-8 * k * lg, squash=0.04 * lg); s.eyes(f, 8, 1.0, 1.0 + 0.8 * lg)
        C = (W / 2, H * 0.47); f *= (1 - 0.45 * e)
        if u > 0.25:
            snap = ease2((g - 0.70) / 0.12); Rr = (W * (0.66 - 0.24 * e) + W * 0.7 * (1 - ease((u - 0.25) / 0.45))) * (1 - 0.35 * snap); S = (44 + 62 * e) * k * (1 + 0.25 * snap); op = (0.30 + 0.70 * clamp(g / 0.5)) * (1 - 0.8 * snap)
            for i in (5, 6, 7, 4, 0, 3, 1, 2):                                       # 上の頭から 先に(下が 手前)
                a = i * 0.7854 + 0.39 + 0.10 * math.sin(t * 2.2 + i * 1.7) * (1 - e); c = (C[0] + math.cos(a) * Rr, C[1] + math.sin(a) * Rr * 1.28)
                s.bite(f, c, S, a + 3.1416 + 0.10 * math.sin(t * 5 + i), op, 1.0 if i % 2 else 0.82)
            sw = clamp((g - 0.80) / 0.20)                                            # まん中: 下から 1ぴきが 口を あけて のみこむ
            if sw > 0:
                S2 = (90 + 1700 * sw ** 2.2) * k; s.bite(f, (C[0], C[1] + H * 0.75 * (1 - ease(sw / 0.5)) + S2 * 0.30), S2, -1.5708, 0.6 + 0.4 * sw, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.6
        return f, 9 * math.exp(-u * 5) + 12 * e

    # ---- この子だけの曲: 都節(ニ・変ホ・ト・イ・変ロ)の 三味線ふうの はじきと 太鼓。目が 灯るたび 「シャッ」と 1音ずつ 上がる。うずは 太鼓の 連打が 速くなり、どら で しめる。あとは 尺八ふうの ひと節 ----
    def audio(s, path):
        a = Snd(DUR, 36); D = 146.83; sc = [0, 1, 5, 7, 8]
        def fq(d, o=0): return D * 2 ** ((sc[d % 5] + 12 * (d // 5 + o)) / 12.0)
        def pluck(d, t0, g=0.12, o=1): a.note(fq(d, o), t0, 0.16, g, "fm", 0.001); a.note(fq(d, o) * 2.01, t0, 0.05, g * 0.3, "sq", 0.001); a.hat(t0, g * 0.5, 0.008)
        def taiko(t0, g=0.8, f0=58): a.boom(t0, f0, g, 0.22)
        def hiss(t0, g=0.14, ln=0.12): a.a += (a.nz - a.lp(a.nz, 0.75)) * a.env(t0, 0.01, ln) * g
        def flute(d, t0, ln, g=0.07, o=2):
            ph = 2 * np.pi * fq(d, o) * (a.t - t0) + 0.35 * np.sin(2 * np.pi * 5 * (a.t - t0)); e = a.env(t0, 0.08, ln)
            a.a += (np.sin(ph) + 0.25 * np.sin(2 * ph)) * e * g + (a.nz - a.lp(a.nz, 0.6)) * e * g * 0.12
        a.a += a.tone(36.7) * 0.07 * a.gate(0, T_CUT, 0.4) + a.lp(a.nz, 0.03) * 0.5 * a.gate(0, T_CUT, 0.4)
        for i in range(5): a.note(500 + 130 * ((i * 3) % 5), 0.3 + i * 0.31, 0.05, 0.03, "sine")                                   # あぶく
        flute(2, 0.5, 0.5, 0.05, 1); taiko(1.0, 0.5)
        for i, c in enumerate(CNT): hiss(c); pluck(i, c, 0.12); taiko(c, 0.25 + 0.05 * i, 70)
        taiko(CNT[-1] + 0.14, 1.0, 50)
        a.boom(T_CUT, 55, 0.9, 0.3)
        for i, (d, dt) in enumerate(((4, 0.2), (3, 0.4), (2, 0.6), (1, 0.75), (0, 0.9), (2, 1.2), (1, 1.35), (0, 1.5))): pluck(d + 5, T_CUT + dt, 0.13)
        for dt in (0.2, 0.6, 0.9, 1.2, 1.5): taiko(T_CUT + dt, 0.5)
        for i in range(8): a.note(1900 + i * 60, T_CUT + 0.3 + i * 0.07, 0.04, 0.03, "sine")
        for i in range(8): hiss(T_SPIT + i * 0.05, 0.16, 0.25); a.rise(T_SPIT + i * 0.05, T_SPIT + i * 0.05 + 0.3, 500 + 60 * i, 1400 + 100 * i, 0.03)   # 毒を はく
        taiko(T_SPIT, 0.9, 50)
        tk = T_VOID + 0.05; i = 0                                                                                                 # うず: 連打が 速くなる
        while tk < T_BURST - 0.03:
            taiko(tk, 0.35 + 0.5 * (tk - T_VOID) / (T_BURST - T_VOID), 64 if i % 2 else 52); tk += max(0.045, 0.2 - i * 0.011); i += 1
        a.a += (a.nz - a.lp(a.nz, 0.7)) * a.gate(T_VOID, T_BURST, 0.15) * 0.06 * (1 + np.sin(a.t * 40))
        a.rise(T_VOID, T_BITE, 90, 300, 0.10); a.hat(T_BITE, 0.6, 0.05); a.note(1300, T_BITE, 0.08, 0.12, "sq", 0.001)             # かむ
        a.rise(T_BITE, T_BURST, 300, 900, 0.10)
        a.boom(T_BURST, 38, 1.7, 0.8); a.a += a.lp(a.nz, 0.3) * a.env(T_BURST, 0.004, 0.45) * 0.6
        for fr, g in ((98, 0.10), (147.5, 0.07), (233, 0.06), (311, 0.05), (467, 0.03)): a.note(fr, T_BURST, 1.6, g, "sine", 0.002)   # どら
        for i in range(8): a.whoosh(T_END + i * 0.045, T_END + i * 0.045 + 0.3, 0.12, 0.3)
        taiko(T_END, 1.0, 46); a.a += a.lp(a.nz, 0.15) * a.gate(T_END + 0.2, T_END + 2.6, 0.3) * 0.18
        for d, dt, ln in ((4, 0.5, 0.35), (3, 0.9, 0.2), (2, 1.15, 0.6), (0, 1.9, 0.9)): flute(d, T_END + dt, ln, 0.07)
        pluck(0, T_END + 1.9, 0.10, 0); pluck(2, T_END + 1.9, 0.08, 0); taiko(T_END + 1.9, 0.6)
        s.fin_snd(a, DUR, 36); a.save(path)


if __name__ == "__main__":
    r = Yamata()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
