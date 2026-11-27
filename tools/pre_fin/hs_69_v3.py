"""ビャッコ(★5 白虎の槍士) 専用の必殺技「白虎烈爪」— 決めは 奥から こちらへ
夜の すすき野原を 風が わたる。ひざをついて 動かない → 槍を にぎると 目が 光り、ずっと奥の 闇に 黄色い目が ふたつ ひらく →
ねこの目の形(横長)の 窓のカットイン → 奥から 白い虎が すすきを かきわけて こちらへ 走ってくる(前足が かわりばんこに 大きく) →
とびかかる: 大きな前足が 画面を ななめに ないで 通りすぎ、口が せまる → finale
使い方: python3 hs_69.py 出力.mp4 / python3 hs_69.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 11.5
WIND = np.array([0.80, 0.95, 0.92], np.float32); EYE = np.array([1.0, 0.90, 0.30], np.float32); SILV = np.array([0.72, 0.78, 0.84], np.float32)
T_GRIP, T_CUT, T_RUN, T_LEAP, T_PAW, T_MOUTH = 1.7, 3.5, 5.4, 7.5, 8.3, 9.0


class Byakko(Kit):
    def __init__(s, W=540):
        super().__init__(69, (0.80, 0.95, 0.92), "白虎烈爪", "White Tiger Claw", W)
        H, k = s.H, s.k
        s.eL = s.A(225, 295); s.eR = s.A(350, 290); s.fc = s.A(285, 290); s.grip = s.A(55, 265); s.prong = s.A(68, 520)
        s.ht = [s.text_img(c, hz.SERIF, int(84 * k), 3) for c in "白虎烈爪"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * k), 1) for c in "奥義"]
        s.HF = 0.38; s.CX = W / 2; s.flr = H * 0.86; s.CY = s.flr - s.HF * H / 2; s.VP = (W / 2, H * 0.40)
        rg = np.random.RandomState(69); s.yn = s.YY / H; g = []
        for row in range(5):                                                         # すすき: 奥(小さく うすい)→手前
            n = 26 + row * 6; base = H * (0.58 + 0.105 * row); hh = H * (0.05 + 0.035 * row)
            for i in range(n): g.append((W * (i + rg.rand()) / n, base + rg.rand() * H * 0.03, hh * (0.7 + 0.6 * rg.rand()), row, rg.rand(), rg.rand() < 0.45))
        s.grass = g; s.sl = rg.rand(50, 3).astype(np.float32)

    # ---- 大道具 ----
    def night(s, t):
        f = (np.array([0.01, 0.04, 0.08], np.float32)[None, None, :] * (1 - s.yn)[:, :, None] + np.array([0.06, 0.20, 0.24], np.float32)[None, None, :] * s.yn[:, :, None]).astype(np.float32)
        f += (np.exp(-((s.yn - 0.58) / 0.07) ** 2) * 0.16)[:, :, None] * WIND[None, None, :]
        g = int(s.H * 0.585); f[g:] *= 0.55
        return f

    def field(s, f, t, rows=(0, 1, 2, 3, 4), part=0.0, gust=0.0):
        """すすき。part: まん中から 左右へ かきわける強さ"""
        W, k = s.W, s.k
        for (x, y, h, row, r, plume) in s.grass:
            if row not in rows: continue
            bend = 0.32 * math.sin(t * 2.0 + x * 0.011 - row * 0.8 + r) + 0.12 * math.sin(t * 5.3 + r * 9) + gust + part * (1 if x > W / 2 else -1) * math.exp(-abs(x - W / 2) / (W * (0.10 + 0.07 * row))) * 1.3
            bend = max(-0.9, min(0.9, bend)); tip = (x + math.sin(bend) * h, y - math.cos(bend) * h); mid = (x + math.sin(bend * 0.4) * h * 0.55, y - math.cos(bend * 0.4) * h * 0.55); sh = 0.30 + 0.14 * row
            col = (float(SILV[0] * sh * 0.7), float(SILV[1] * sh * 0.8), float(SILV[2] * sh * 0.75))
            cv2.polylines(f, [np.array([(x, y), mid, tip], np.int32)], False, col, max(1, int((1 + row * 0.5) * k)), cv2.LINE_AA)
            if plume: cv2.ellipse(f, (int(tip[0] + math.sin(bend) * h * 0.1), int(tip[1])), (max(1, int(h * 0.20)), max(1, int(h * 0.06))), math.degrees(bend) - 90, 0, 360, (float(SILV[0] * sh * 1.5), float(SILV[1] * sh * 1.5), float(SILV[2] * sh * 1.5)), -1, cv2.LINE_AA)

    def rush(s, f, t, v, speed=1.0):
        L = s.L0(); VP = s.VP
        for r in s.sl:
            a = r[0] * 6.283; q = ((r[1] + t * speed * (0.6 + r[2])) % 1.0) ** 2; r0 = q * s.H * 0.9 + 14 * s.k
            s.line(L, (VP[0] + math.cos(a) * r0, VP[1] + math.sin(a) * r0), (VP[0] + math.cos(a) * r0 * (1.1 + 0.3 * q), VP[1] + math.sin(a) * r0 * (1.1 + 0.3 * q)), float(v * (0.2 + 0.8 * q)), 1 if q < 0.5 else 2)
        f += cv2.GaussianBlur(L, (0, 0), 1.0 * s.k)[:, :, None] * WIND[None, None, :] * 0.6

    def paw(s, f, c, S, rot=0.0, claws=0.0):
        """前足(肉球の側が こちら)。claws: つめの 出ぐあい"""
        ca, sa = math.cos(rot), math.sin(rot)
        def R(x, y): return (int(c[0] + (x * ca - y * sa) * S), int(c[1] + (x * sa + y * ca) * S))
        cv2.circle(f, R(0, 0.1), int(S * 0.62), (0.02, 0.03, 0.04), -1, cv2.LINE_AA); cv2.circle(f, R(0, 0.1), int(S * 0.56), (0.93, 0.95, 0.97), -1, cv2.LINE_AA)
        for i, x in enumerate((-0.48, -0.17, 0.17, 0.48)):
            y = -0.42 - (0.12 if i in (1, 2) else 0)
            if claws > 0: cv2.fillPoly(f, [np.array([R(x - 0.07, y - 0.12), R(x + 0.07, y - 0.12), R(x + 0.04 * (i - 1.5), y - 0.25 - 0.42 * claws)], np.int32)], (0.86, 0.88, 0.92), cv2.LINE_AA)
            cv2.circle(f, R(x, y), int(S * 0.22), (0.02, 0.03, 0.04), -1, cv2.LINE_AA); cv2.circle(f, R(x, y), int(S * 0.18), (0.93, 0.95, 0.97), -1, cv2.LINE_AA); cv2.circle(f, R(x, y + 0.03), int(S * 0.10), (0.95, 0.62, 0.66), -1, cv2.LINE_AA)
        cv2.ellipse(f, R(0, 0.16), (int(S * 0.30), int(S * 0.24)), math.degrees(rot), 0, 360, (0.95, 0.62, 0.66), -1, cv2.LINE_AA)
        for x0, x1 in ((-0.56, -0.30), (0.30, 0.56)): cv2.line(f, R(x0, 0.42), R(x1, 0.50), (0.05, 0.06, 0.08), max(1, int(S * 0.06)), cv2.LINE_AA)

    def tiger(s, f, c, S, t, op=0.0, run=0.0):
        """正面の 白い虎(S=頭の半径)。run: 走りの 位相(前足が かわりばんこ)"""
        x, y = c; k = s.k
        cv2.ellipse(f, (int(x), int(y + S * 0.9)), (int(S * 1.25), int(S * 0.9)), 0, 0, 360, (0.02, 0.03, 0.04), -1, cv2.LINE_AA); cv2.ellipse(f, (int(x), int(y + S * 0.9)), (int(S * 1.18), int(S * 0.84)), 0, 0, 360, (0.80, 0.84, 0.88), -1, cv2.LINE_AA)   # 肩
        for sg in (-1, 1):
            for j in range(2): cv2.fillPoly(f, [np.array([(x + sg * S * 1.18, y + S * (0.7 + j * 0.3)), (x + sg * S * 0.62, y + S * (0.85 + j * 0.3)), (x + sg * S * 1.15, y + S * (0.92 + j * 0.3))], np.int32)], (0.05, 0.06, 0.08), cv2.LINE_AA)
        if run is not None:
            for sg in (-1, 1):
                ph = math.sin(run + (0 if sg < 0 else 3.1416)); s.paw(f, (x + sg * S * 0.95, y + S * (1.55 - 0.25 * ph)), S * (0.50 + 0.22 * max(0.0, ph)), sg * 0.15, 0.3 * max(0.0, ph))
        for sg in (-1, 1):                                                           # 耳
            e = [(x + sg * S * 0.45, y - S * 0.70), (x + sg * S * 0.98, y - S * 1.22), (x + sg * S * 0.98, y - S * 0.38)]
            cv2.fillPoly(f, [np.array(e, np.int32)], (0.02, 0.03, 0.04), cv2.LINE_AA); cv2.fillPoly(f, [np.array([(x + sg * S * 0.60, y - S * 0.68), (x + sg * S * 0.90, y - S * 1.02), (x + sg * S * 0.90, y - S * 0.50)], np.int32)], (0.20, 0.22, 0.26), cv2.LINE_AA)
        ruff = [(x + math.cos(a) * S * (1.22 if j % 2 else 1.02), y + S * 0.18 + math.sin(a) * S * (1.05 if j % 2 else 0.9)) for j, a in enumerate([0.15 + i * (2.84 / 12.0) for i in range(13)])]
        cv2.fillPoly(f, [np.array(ruff + [(x - S, y), (x + S, y)], np.int32)], (0.02, 0.03, 0.04), cv2.LINE_AA); cv2.fillPoly(f, [np.array([(x + (px - x) * 0.94, y + (py - y) * 0.94) for px, py in ruff] + [(x - S * 0.9, y), (x + S * 0.9, y)], np.int32)], (0.86, 0.89, 0.92), cv2.LINE_AA)
        cv2.circle(f, (int(x), int(y)), int(S * 1.05), (0.02, 0.03, 0.04), -1, cv2.LINE_AA); cv2.circle(f, (int(x), int(y)), int(S), (0.95, 0.97, 0.98), -1, cv2.LINE_AA)
        for j in (-1, 0, 1): cv2.fillPoly(f, [np.array([(x + j * S * 0.26 - S * 0.09, y - S * 0.98), (x + j * S * 0.26 + S * 0.09, y - S * 0.98), (x + j * S * 0.20, y - S * (0.50 if j == 0 else 0.60))], np.int32)], (0.05, 0.06, 0.08), cv2.LINE_AA)   # ひたいの しま
        for sg in (-1, 1):
            for j in range(2): cv2.fillPoly(f, [np.array([(x + sg * S * 0.99, y + S * (-0.12 + j * 0.30)), (x + sg * S * 0.55, y + S * (0.02 + j * 0.30)), (x + sg * S * 0.97, y + S * (0.10 + j * 0.30))], np.int32)], (0.05, 0.06, 0.08), cv2.LINE_AA)   # ほおの しま
            e = (x + sg * S * 0.42, y - S * 0.22); cv2.ellipse(f, (int(e[0]), int(e[1])), (int(S * 0.27), int(S * 0.10)), sg * 30, 0, 360, (1.0, 0.90, 0.30), -1, cv2.LINE_AA); cv2.ellipse(f, (int(e[0]), int(e[1])), (max(1, int(S * 0.03)), int(S * 0.09)), sg * 30, 0, 360, (0.0, 0.0, 0.0), -1, cv2.LINE_AA)
            cv2.fillPoly(f, [np.array([(x + sg * S * 0.08, y - S * 0.20), (x + sg * S * 0.78, y - S * 0.62), (x + sg * S * 0.80, y - S * 0.42), (x + sg * S * 0.12, y - S * 0.06)], np.int32)], (0.05, 0.06, 0.08), cv2.LINE_AA)
            for j in (-1, 0, 1): cv2.line(f, (int(x + sg * S * 0.34), int(y + S * 0.30)), (int(x + sg * S * 1.35), int(y + S * (0.28 + j * 0.16))), (0.9, 0.92, 0.95), max(1, int(S * 0.02)), cv2.LINE_AA)   # ひげ
            s.soft(f, e, S * 0.22, 0.35, EYE)
        my = y + S * 0.45; mh = S * (0.10 + 0.56 * op); mw = S * (0.40 + 0.20 * op)
        cv2.ellipse(f, (int(x), int(my + mh * 0.5)), (int(mw), int(mh)), 0, 0, 360, (0.50, 0.04, 0.08), -1, cv2.LINE_AA)
        if op > 0.15:
            for sg in (-1, 1):
                cv2.fillPoly(f, [np.array([(x + sg * mw * 0.80, my + mh * 0.5 - mh * 0.7), (x + sg * mw * 0.45, my + mh * 0.5 - mh * 0.9), (x + sg * mw * 0.60, my + mh * 0.5 - mh * 0.9 + S * 0.46 * op)], np.int32)], (1.0, 1.0, 0.95), cv2.LINE_AA)
                cv2.fillPoly(f, [np.array([(x + sg * mw * 0.62, my + mh * 0.5 + mh * 0.75), (x + sg * mw * 0.30, my + mh * 0.5 + mh * 0.93), (x + sg * mw * 0.45, my + mh * 0.5 + mh * 0.9 - S * 0.20 * op)], np.int32)], (1.0, 1.0, 0.95), cv2.LINE_AA)
        for sg in (-1, 1): cv2.circle(f, (int(x + sg * S * 0.17), int(y + S * 0.30)), int(S * 0.21), (0.90, 0.93, 0.95), -1, cv2.LINE_AA)        # 口もと
        cv2.fillPoly(f, [np.array([(x - S * 0.17, y + S * 0.08), (x + S * 0.17, y + S * 0.08), (x, y + S * 0.26)], np.int32)], (0.10, 0.08, 0.10), cv2.LINE_AA)

    def hero(s, f, t, **kw):
        img = s.pose(t, **kw); s.draw(f, img, s.CX, s.CY, s.HF)

    def eyes(s, f, v):
        if v <= 0.01: return
        sc = s._pl[2]; L = s.L0()
        for e in (s.eL, s.eR): p = s.at(*e); cv2.circle(L, (int(p[0]), int(p[1])), max(1, int(12 * sc)), float(min(1.0, v)), -1, cv2.LINE_AA)
        s.add(f, L, EYE, 0.8, 2, 7)

    def frame(s, t):
        if t < T_CUT: f, sh = s.sc_wait(t)
        elif t < T_RUN: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_LEAP: f, sh = s.sc_run(t)
        else: f, sh = s.sc_leap(t)
        f, sh = s.finale(f, sh, t, DUR, None, (0.80, 0.95, 0.92), (1.0, 0.90, 0.30), -1)
        return s.post(f, WIND, 0.28, sh * s.k, t, 0.97)

    # 1) すすき野原。奥に 目が ひらく
    def sc_wait(s, t):
        W, H, k = s.W, s.H, s.k; gp = t - T_GRIP
        gust = 0.5 * math.exp(-((t - 0.9) / 0.35) ** 2) + (0.6 * math.exp(-((gp - 0.3) / 0.3) ** 2) if gp > 0 else 0)
        f = s.night(t); s.field(f, t, (0, 1, 2), 0, gust)
        if gp > 0.5:                                                                 # 奥の 目
            o = ease2((gp - 0.5) / 0.4)
            for sg in (-1, 1):
                e = (s.VP[0] + sg * 16 * k, s.VP[1] + H * 0.10); cv2.ellipse(f, (int(e[0]), int(e[1])), (int(9 * k), max(1, int(5 * k * o))), sg * 20, 0, 360, (1.0, 0.90, 0.30), -1, cv2.LINE_AA); s.soft(f, e, 16 * k, 0.5 * o, EYE)
        s.hero(f, t, squash=(0.05 * math.exp(-gp * 9) * math.cos(gp * 16) if gp > 0 else 0)); s.eyes(f, clamp(gp / 0.3) if gp > 0 else 0)
        s.field(f, t, (3, 4), 0, gust)
        return f * clamp(t / 0.4), (4 * math.exp(-gp * 9) if gp > 0 else 0)

    # 2) ねこの目の形の 窓
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.night(t) * 0.6; s.field(f, t, (2, 3, 4), 0, 0.3)
        C = (W / 2, H * 0.36); g = back(u / 0.3, 1.2); hw, hh = W * 0.47, H * 0.20 * g
        pts = [(C[0] - hw + 2 * hw * i / 24.0, C[1] - hh * math.sin(math.pi * i / 24.0)) for i in range(25)] + [(C[0] + hw - 2 * hw * i / 24.0, C[1] + hh * math.sin(math.pi * i / 24.0)) for i in range(25)]
        m = s.L0(); cv2.fillPoly(m, [np.array(pts, np.int32)], 1.0, cv2.LINE_AA)
        win = np.zeros((H, W, 3), np.float32) + np.array([0.10, 0.30, 0.32], np.float32)
        for j in range(7): cv2.fillPoly(win, [np.array([(W * (0.02 + j * 0.16), 0), (W * (0.10 + j * 0.16), 0), (W * (0.02 + j * 0.16) - W * 0.1, H)], np.int32)], (0.06, 0.20, 0.22), cv2.LINE_AA)   # しま
        zf = 0.98 + 0.05 * ease(u / 1.8); sc = zf * H / s.ah; img = s.pose(t)
        s.draw(win, img, C[0] - (s.fc[0] - s.aw / 2) * sc, C[1] + H * 0.01 - (s.fc[1] - s.ah / 2) * sc, zf); s.eyes(win, 0.8 + 1.2 * math.exp(-max(0.0, u - 0.45) * 5) * (u > 0.45))
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        cv2.polylines(f, [np.array(pts, np.int32)], True, (0.02, 0.03, 0.04), max(1, int(12 * k)), cv2.LINE_AA); cv2.polylines(f, [np.array(pts, np.int32)], True, (1.0, 0.90, 0.30), max(1, int(5 * k)), cv2.LINE_AA)
        n = len(s.ht); step = W * 0.20
        for i, T in enumerate(s.ht):                                                 # 技名: ななめに ないで 出る
            a = (u - 0.55 - i * 0.1) / 0.14
            if a > 0:
                cx = W / 2 + (i - (n - 1) / 2) * step; s.text(f, T, cx, H * 0.80, clamp(a * 2), 1.0)
                if a < 1.6: L = s.L0(); s.line(L, (cx + 44 * k, H * 0.80 - 60 * k), (cx - 44 * k, H * 0.80 + 60 * k), 1 - a / 1.6, 3); s.add(f, L, WIND, 0.9, 1, 5)
        for i, T in enumerate(s.vo):
            if u > 0.45 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.715, 1.0, 1.0)
        if u > 1.3: s.text(f, s.txt_en, W / 2, H * 0.80 + 62 * k, ease((u - 1.3) / 0.3), 1.0)
        if u < 0.08: f = f + (1 - u / 0.08) * 0.7
        return f, 0

    # 3) 奥から 走ってくる
    def sc_run(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_RUN; g = clamp(u / (T_LEAP - T_RUN))
        S = (10 + 120 * g ** 2.2) * k; run = t * 15; bob = abs(math.sin(run)) * S * 0.22
        f = s.night(t); s.rush(f, t, 0.3 + 0.5 * g, 0.5 + g); s.field(f, t, (0, 1, 2), 0.5 + 0.8 * g, 0)
        c = (s.VP[0] + math.sin(run * 0.5) * S * 0.12, s.VP[1] + H * 0.10 + H * 0.06 * g - bob)
        if g < 0.55: s.hero(f, t); s.eyes(f, 1.0)
        s.tiger(f, c, S, t, 0.35 + 0.25 * g, run)
        s.field(f, t, (3, 4), 0.8 + 1.0 * g, 0)
        if u < 0.1: f = f * (u / 0.1) + (1 - u / 0.1) * 0.5
        return f, 1.5 + 4 * g * abs(math.sin(run))

    # 4) とびかかる: 前足が ないで 通りすぎ、口が せまる
    def sc_leap(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_LEAP; pw = t - T_PAW; mo = t - T_MOUTH
        g = clamp(u / (DUR - 1.7 - T_LEAP)); S = (130 + 330 * g ** 1.6) * k
        arc = math.sin(clamp(u / 0.8) * 3.1416) * H * 0.16                             # とび上がって おりてくる
        c = (s.VP[0], s.VP[1] + H * 0.16 - arc + (S - 130 * k) * 0.25)
        f = s.night(t) * (1 - 0.4 * g); s.rush(f, t, 0.9, 1.6 + g)
        if u < 0.5: s.field(f, t, (3, 4), 2.2, 0)
        op = clamp(0.55 + 0.45 * (mo / 0.5 if mo > 0 else 0.0))
        s.tiger(f, c, S, t, op, None)
        sh = 3 + 6 * g
        if 0 < pw < 0.75:                                                            # 右の前足(画面の 右上→左下)
            e = ease2(pw / 0.32); p = (W * (1.15 - 1.25 * e), H * (-0.10 + 1.15 * e)); PS = W * (0.30 + 0.34 * e)
            L = s.L0()
            for i in range(4):                                                       # つめの 風すじ(通ったあとに 残って 消える)
                off = (i - 1.5) * PS * 0.30; a0 = (W * 1.15 + off * 0.7, -H * 0.10 + off * 0.7 - PS * 0.5); q = (p[0] + off * 0.7, p[1] + off * 0.7 - PS * 0.5)
                s.line(L, a0, q, float(clamp(1.4 - pw * 2.2)), max(2, int(7 * k)))
            s.add(f, L, WIND, 0.9, 3, 12)
            if pw < 0.5: s.paw(f, p, PS, 0.6, 1.0)
            if pw < 0.1: f = f + (1 - pw / 0.1) * 0.35
            sh = max(sh, 16 * math.exp(-pw * 5))
        if u < 0.08: f = f + (1 - u / 0.08) * 0.4
        return f, sh

    # ---- この子だけの曲: 風の音と 低い うなり(のど鳴り)、びわ風の 低い はじき(ホ短調 五音)。走る間は 太鼓が 足音の 速さで つまっていく。とびかかりで 咆哮 ----
    def audio(s, path):
        a = Snd(DUR, 69); E = 82.41; pen = [0, 3, 5, 7, 10]
        def fq(d, o=0): return E * 2 ** ((pen[d % 5] + 12 * (d // 5 + o)) / 12.0)
        def biwa(d, t0, g=0.12, o=1): a.note(fq(d, o), t0, 0.3, g, "fm", 0.001); a.note(fq(d, o) * 3.02, t0, 0.04, g * 0.4, "sq", 0.0005); a.hat(t0, g * 0.4, 0.006)
        def purr(t0, t1, g=0.10): a.a += a.lp((np.sign(np.sin(2 * np.pi * 27 * a.t)) * (0.6 + 0.4 * np.sin(2 * np.pi * 55 * a.t))).astype(np.float32), 0.03) * a.gate(t0, t1, 0.2) * g * 6
        def roar(t0, ln, g=0.5):
            u = np.clip((a.t - t0) / ln, 0, 1); fr = 190 - 90 * u + 12 * np.sin(2 * np.pi * 31 * a.t); ph = 2 * np.pi * np.cumsum(fr) / a.sr; e = np.clip(u * 12, 0, 1) * np.clip((1 - u) * 2.5, 0, 1) * ((a.t >= t0) & (a.t < t0 + ln))
            a.a += (np.tanh(3 * (np.sin(ph) + 0.6 * np.sin(2.01 * ph) + 0.4 * np.sin(3.02 * ph))) * 0.35 + a.lp(a.nz, 0.25) * 0.9) * e * g
        def step_(t0, g=0.5): a.boom(t0, 62, g, 0.12); a.a += a.lp(a.nz, 0.5) * a.env(t0, 0.002, 0.04) * g * 0.3
        a.a += a.lp(a.nz, 0.05) * (0.5 + 0.4 * np.sin(a.t * 1.9) + 0.6 * np.exp(-((a.t - 0.9) / 0.35) ** 2)) * 0.7 * a.gate(0, DUR, 0.4)   # 風
        biwa(0, 0.5, 0.08); biwa(3, 1.1, 0.08)
        a.boom(T_GRIP, 60, 0.9, 0.3); biwa(0, T_GRIP, 0.12, 0); purr(T_GRIP + 0.5, T_CUT, 0.10); a.note(fq(9, 2), T_GRIP + 0.9, 0.5, 0.04, "sine", 0.05)
        a.boom(T_CUT, 52, 1.0, 0.35)
        for i, (d, dt) in enumerate(((5, 0.2), (7, 0.38), (8, 0.56), (7, 0.74), (5, 0.92), (4, 1.1), (5, 1.28), (8, 1.46))): biwa(d, T_CUT + dt, 0.12)
        for i in range(4): a.rise(T_CUT + 0.55 + i * 0.1, T_CUT + 0.62 + i * 0.1, 2500, 900, 0.10); a.hat(T_CUT + 0.6 + i * 0.1, 0.25, 0.03)
        for dt in (0.2, 0.56, 0.92, 1.28, 1.46): a.boom(T_CUT + dt, 66, 0.6, 0.15)
        purr(T_RUN, T_LEAP, 0.12); a.rise(T_RUN, T_LEAP, 60, 260, 0.10)
        n = int((T_LEAP - T_RUN) * 15 / math.pi)                                                                                 # 足音: 絵の 走りと 同じ 速さ
        for i in range(n): t0 = T_RUN + (i + 0.5) * math.pi / 15; step_(t0, 0.25 + 0.6 * (t0 - T_RUN) / (T_LEAP - T_RUN))
        for i in range(10): biwa((0, 3, 5, 3)[i % 4] + 5 * (i // 6), T_RUN + 0.1 + i * 0.2, 0.08)
        a.whoosh(T_LEAP, T_LEAP + 0.5, 0.5, 0.3); roar(T_LEAP + 0.05, 1.1, 0.5)
        a.rise(T_PAW, T_PAW + 0.3, 3200, 500, 0.20); a.whoosh(T_PAW, T_PAW + 0.32, 0.9, 0.5); a.boom(T_PAW + 0.3, 44, 1.5, 0.5)
        roar(T_MOUTH - 0.1, DUR - 1.6 - T_MOUTH, 0.6)
        tk = T_LEAP + 0.1; i = 0
        while tk < DUR - 1.8: a.boom(tk, 56, 0.5, 0.12); tk += max(0.07, 0.22 - i * 0.012); i += 1
        s.fin_snd(a, DUR, 69); a.save(path)


if __name__ == "__main__":
    r = Byakko()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
