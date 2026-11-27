"""モルドレッド(★5 叛逆の騎士) 専用の必殺技「叛逆ノ黒剣」
金色の 玉座の間。赤いじゅうたんの先に 玉座と王冠。ひざまずく 黒い騎士(忠誠の姿) → かぶとの目が 赤く灯ると、金の旗が 1枚ずつ 斬り落とされ、広間が 赤く そまる →
かぶとの すき間のカットイン → 床に立てた剣を ひねる。赤い ひびが じゅうたんを 走り、玉座が 二つに割れ、王冠が はね上がる →
落ちてきた王冠が 目の前で 砕ける → 黒い旗が おりる
使い方: python3 hs_31.py 出力.mp4 / python3 hs_31.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 11.4
RED = np.array([1.0, 0.10, 0.12], np.float32); GOLD = np.array([1.0, 0.82, 0.38], np.float32); STONE = np.array([0.10, 0.09, 0.12], np.float32)
T_EYE, T_CUT, T_TWIST, T_SPLIT, T_FALL, T_FLAG = 1.5, 3.1, 5.0, 6.6, 8.5, 8.9
BAN = [(-1, 0.10, 1.0), (1, 0.10, 1.0), (-1, 0.23, 0.72), (1, 0.23, 0.72), (-1, 0.33, 0.52), (1, 0.33, 0.52)]


class Mordred(Kit):
    def __init__(s, W=540):
        super().__init__(31, (1.0, 0.10, 0.12), "叛逆ノ黒剣", "Traitor's Edge", W)
        s.eyeL = s.A(322, 238); s.eyeR = s.A(432, 238); s.guard = s.A(345, 455); s.tip = s.A(345, 745)
        s.ht = [s.text_img(c, hz.SERIF, int(80 * s.k), 3) for c in "叛逆ノ黒剣"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        rg = np.random.RandomState(31); s.rd = rg.rand(200, 6).astype(np.float32)
        s.yn = s.YY / s.H; s.VP = (s.W / 2, s.H * 0.30); s.TH = (s.W / 2, s.H * 0.37)
        s.crk = []
        for i in range(5):                                                           # じゅうたんを 走る ひび(手前→玉座)
            x = s.W / 2 + (i - 2) * s.W * 0.05; pts = [(x, s.H * 0.86)]; v = 0
            while v < 1: v += 0.07 + rg.rand() * 0.06; y = s.H * 0.86 - (s.H * 0.86 - s.H * 0.44) * min(v, 1); pts.append((s.W / 2 + (x - s.W / 2) * (1 - 0.8 * min(v, 1)) + (rg.rand() - 0.5) * 26 * s.k * (1 - v * 0.7), y))
            s.crk.append(pts)

    # ---- この子だけの絵 ----
    def crown(s, f, c, w, rot=0.0, v=1.0, col=GOLD):
        ca, sa = math.cos(rot), math.sin(rot)
        pts = [(-0.5, 0.3), (-0.5, -0.1), (-0.3, 0.12), (-0.18, -0.32), (0.0, 0.1), (0.18, -0.32), (0.3, 0.12), (0.5, -0.1), (0.5, 0.3)]
        P = np.array([(c[0] + (x * ca - y * sa) * w, c[1] + (x * sa + y * ca) * w) for x, y in pts], np.int32)
        M = s.L0(); cv2.fillPoly(M, [P], 1.0, cv2.LINE_AA); f[:] = f * (1 - M[:, :, None] * v) + M[:, :, None] * np.asarray(col, np.float32)[None, None, :] * v
        f += cv2.GaussianBlur(M, (0, 0), max(2, w * 0.2))[:, :, None] * np.asarray(col, np.float32)[None, None, :] * 0.5 * v

    def hall(s, red, cuts, t, split=0.0, crown=True):
        """玉座の間。red=金→赤、cuts[i]=旗iが 斬られてからの時間(負なら まだ)"""
        W, H, k = s.W, s.H, s.k; light = GOLD * (1 - red) + RED * red
        f = np.zeros((H, W, 3), np.float32) + STONE[None, None, :] * (0.5 + 0.9 * s.yn)[:, :, None]
        f += np.exp(-(((s.XX - s.TH[0]) / (W * 0.5)) ** 2 + ((s.YY - s.TH[1]) / (H * 0.3)) ** 2))[:, :, None] * (light * 0.42)[None, None, :]
        L = s.L0()
        for i in range(-5, 6): s.line(L, s.VP, (W / 2 + i * W * 0.30, H), 0.10, 1)                                           # 床の すじ
        for j in range(5): y = H * (0.46 + 0.54 * ((j + 1) / 5.0) ** 1.8); s.line(L, (0, y), (W, y), 0.08, 1)
        for sg in (-1, 1):
            for j in range(3): x = W / 2 + sg * W * (0.47 - j * 0.11); s.line(L, (x, H * (0.02 + j * 0.06)), (x, H * (0.98 - j * 0.14)), 0.22 - j * 0.04, int((12 - j * 3) * k))     # 柱
        f += L[:, :, None] * (light * 0.5 + 0.3)[None, None, :]
        C = s.L0(); cv2.fillPoly(C, [np.array([(W * 0.18, H), (W * 0.82, H), (W / 2 + W * 0.06, H * 0.44), (W / 2 - W * 0.06, H * 0.44)], np.int32)], 1.0, cv2.LINE_AA)       # じゅうたん
        f = f * (1 - C[:, :, None] * 0.9) + C[:, :, None] * (np.array([0.42, 0.05, 0.07], np.float32) * (0.5 + 0.7 * s.yn)[:, :, None])
        tw = W * 0.15; th = H * 0.17                                                                                         # 玉座(割れると 左右に)
        for sg in (-1, 1):
            off = sg * split * W * 0.10; tilt = sg * split * 0.35
            P = [(0, 0.5), (0, -0.5), (sg * 0.5, -0.62), (sg * 0.5, 0.5)]
            Q = np.array([(s.TH[0] + off + (x * tw) * math.cos(tilt) - (y * th) * math.sin(tilt), s.TH[1] + (x * tw) * math.sin(tilt) + (y * th) * math.cos(tilt) + split * H * 0.03) for x, y in P], np.int32)
            cv2.fillPoly(f, [Q], tuple(float(q) for q in (light * 0.5 + STONE)), cv2.LINE_AA); cv2.polylines(f, [Q], True, tuple(float(q) for q in light), 2, cv2.LINE_AA)
        if split > 0:
            L2 = s.L0(); s.line(L2, (s.TH[0], s.TH[1] - th * 0.7), (s.TH[0], s.TH[1] + th * 0.6), 1.0, int(4 + 10 * split)); s.add(f, L2, RED, 1.2, 3, 12)
        if crown: s.crown(f, (s.TH[0], s.TH[1] - th * 0.78), W * 0.085, 0.0, 1.0, GOLD)
        for i, (sg, dx, sc) in enumerate(BAN):                                                                               # 旗
            x = W / 2 + sg * W * (0.50 - dx); bw = W * 0.10 * sc; top = H * (0.06 + (1 - sc) * 0.16); bh = H * 0.30 * sc; c = cuts[i]
            body = [(x - bw / 2, top), (x + bw / 2, top), (x + bw / 2, top + bh), (x, top + bh * 1.14), (x - bw / 2, top + bh)]
            if c < 0: cv2.fillPoly(f, [np.array(body, np.int32)], tuple(float(q) for q in GOLD * 0.75 * (0.5 + 0.5 * sc)), cv2.LINE_AA); cv2.circle(f, (int(x), int(top + bh * 0.45)), max(2, int(bw * 0.22)), tuple(float(q) for q in GOLD * 0.35), -1, cv2.LINE_AA)
            else:
                cy0 = top + bh * 0.42; up = [(x - bw / 2, top), (x + bw / 2, top), (x + bw / 2, cy0 - bw * 0.3), (x - bw / 2, cy0 + bw * 0.3)]
                cv2.fillPoly(f, [np.array(up, np.int32)], tuple(float(q) for q in (GOLD * 0.75 * (1 - clamp(c / 0.5)) + RED * 0.35 * clamp(c / 0.5)) * (0.5 + 0.5 * sc)), cv2.LINE_AA)
                dy = c * c * H * 1.4; rot = c * sg * 1.2
                if top + bh + dy < H + 60:
                    lo = [(-bw / 2, bw * 0.3 - bh * 0.0), (bw / 2, -bw * 0.3), (bw / 2, bh * 0.58), (0, bh * 0.72), (-bw / 2, bh * 0.58)]
                    cv2.fillPoly(f, [np.array([(x + px * math.cos(rot) - py * math.sin(rot), cy0 + dy + px * math.sin(rot) + py * math.cos(rot)) for px, py in lo], np.int32)], tuple(float(q) for q in GOLD * 0.6 * (0.5 + 0.5 * sc)), cv2.LINE_AA)
                if c < 0.18: L3 = s.L0(); s.line(L3, (x - bw, cy0 + bw * 0.6), (x + bw, cy0 - bw * 0.6), 1 - c / 0.18, 3); s.add(f, L3, WHITE * 0.5 + RED * 0.5, 1.4)
        return f

    def eyes(s, f, v, streak=1.0):
        if v <= 0: return
        L = s.L0(); sc = s._pl[2]
        for sg, e in ((-1, s.eyeL), (1, s.eyeR)):
            p = s.at(*e); cv2.ellipse(L, (int(p[0]), int(p[1])), (max(1, int(26 * sc)), max(1, int(9 * sc))), sg * 18, 0, 360, float(v), -1, cv2.LINE_AA)
            s.line(L, p, (p[0] + sg * 130 * sc * streak * v, p[1] - 36 * sc * streak * v), 0.7 * v, 1)
        s.add(f, L, RED, 1.3, 2, 9)

    def frame(s, t):
        k = s.k
        if t < T_CUT: f, sh = s.sc_fealty(t)
        elif t < T_TWIST: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_FALL: f, sh = s.sc_break(t)
        else: f, sh = s.sc_flag(t)
        f, sh = s.finale(f, sh, t, DUR, s.tip, (1, 0.2, 0.2), (1, 0.7, 0.3), -1)
        return s.post(f, RED, 0.6, sh * k, t, 0.86)

    # 1) 忠誠の姿 → 目が灯り、旗が 斬り落とされる
    def sc_fealty(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_EYE
        cuts = [(u - 0.25 - i * 0.18) if u > 0.25 + i * 0.18 else -1 for i in range(6)]; n = sum(1 for c in cuts if c >= 0)
        f = s.hall(n / 6.0, cuts, t)
        L = s.L0()
        for i in range(4):                                                           # 天窓の 光
            x = W * (0.2 + 0.2 * i); cv2.fillPoly(L, [np.array([(x - 14 * k, 0), (x + 14 * k, 0), (x + 60 * k, H * 0.7), (x - 20 * k, H * 0.7)], np.int32)], 0.10, cv2.LINE_AA)
        f += cv2.GaussianBlur(L, (0, 0), 8 * k)[:, :, None] * (GOLD * (1 - n / 6.0) + RED * n / 6.0)[None, None, :]
        img = s.pose(t); s.draw(f, img, W / 2, H * 0.78, 0.40)
        if u > 0:
            s.eyes(f, clamp(u / 0.15), 1.0 if u < 1 else 0.6)
            if u < 0.1: f *= 0.5 + 0.5 * u / 0.1
        last = max([c for c in cuts if c >= 0] or [9])
        return f * clamp(t / 0.4), 4 * math.exp(-last * 14) if last < 9 else 0

    # 2) かぶとの すき間
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = np.zeros((H, W, 3), np.float32) + np.array([0.02, 0.0, 0.0], np.float32)
        cyb = H * 0.35; hh = H * 0.10 * ease(u / 0.22)
        zf = 1.7; sc = zf * H / s.ah; ex = (s.eyeL[0] + s.eyeR[0]) / 2
        win = np.zeros((H, W, 3), np.float32) + (RED * 0.10)[None, None, :]
        img = s.pose(t); s.draw(win, img, W / 2 - (ex - s.aw / 2) * sc, cyb - (s.eyeL[1] - s.ah / 2) * sc, zf); s.eyes(win, 0.8 + 0.2 * math.sin(u * 13), 0.5)
        m = (np.abs(s.YY - cyb - (s.XX - W / 2) * 0.0) < hh * (0.55 + 0.9 * np.abs(s.XX - W / 2) / (W / 2))).astype(np.float32)[:, :, None]      # まん中が せまい すき間
        f = f * (1 - m) + win * m
        L = s.L0()
        for sg in (-1, 1): cv2.polylines(L, [np.array([(x, cyb + sg * hh * (0.55 + 0.9 * abs(x - W / 2) / (W / 2))) for x in range(0, W + 1, 20)], np.int32)], False, 1.0, 2, cv2.LINE_AA)
        s.add(f, L, RED, 0.8)
        n = len(s.ht); step = W * 0.175
        for i, T in enumerate(s.ht):                                                 # 技名: 1字ずつ たたきつけ、下へ 赤い ひび
            a = (u - 0.5 - i * 0.17) / 0.12
            if a <= 0: continue
            x = W / 2 + (i - (n - 1) / 2) * step; y = H * 0.66
            if a < 1: T2 = tuple(cv2.resize(q, None, fx=1 + (1 - a) * 1.5, fy=1 + (1 - a) * 1.5) for q in T); s.text(f, T2, x, y, a, 1.0)
            else:
                s.text(f, T, x, y, 1.0, 1.0); L2 = s.L0(); g = clamp((a - 1) / 2); px, py = x, y + 46 * k
                for j in range(5): r = s.rd[i * 5 + j]; nx, ny = px + (r[0] - 0.5) * 40 * k, py + 26 * k * g; s.line(L2, (px, py), (nx, ny), 0.8, 1); px, py = nx, ny
                s.add(f, L2, RED, 0.8, 1.5, 5)
        for i, T in enumerate(s.vo):
            if u > 0.35 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.545, 1.0, 1.0)
        if u > 1.5: s.text(f, s.txt_en, W / 2, H * 0.90, ease((u - 1.5) / 0.3), 1.0)
        if u < 0.08: f += (1 - u / 0.08) * np.array([1.0, 0.2, 0.2], np.float32)[None, None, :]
        hit = min([abs(u - 0.5 - i * 0.17 - 0.12) for i in range(n)])
        return f, 5 * math.exp(-hit * 25)

    # 3) 剣を ひねる → ひび → 玉座が 割れ、王冠が はね上がる
    def sc_break(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_TWIST; sp = t - T_SPLIT
        split = ease(sp / 0.5) if sp > 0 else 0.0
        f = s.hall(1.0, [9] * 6, t, split, sp < 0)
        g = ease2(u / (T_SPLIT - T_TWIST)); L = s.L0()
        for pts in s.crk:
            n = int(g * (len(pts) - 1))
            for j in range(n): s.line(L, pts[j], pts[j + 1], 1.0, max(1, int(4 * k * (1 - j / len(pts) * 0.7))))
            if n < len(pts) - 1: cv2.circle(L, (int(pts[n][0]), int(pts[n][1])), int(6 * k), 1.0, -1)
        s.add(f, L, RED, 1.1, 3, 10)
        Lf = s.L0()
        for i in range(30):                                                          # はがれる 床の かけら
            r = s.rd[i]; v = (u * (0.6 + r[2]) - r[0] * 1.2)
            if v <= 0 or v > 1: continue
            x = W / 2 + (r[1] - 0.5) * W * 0.5 * (1 - r[0] * 0.6); y = H * (0.86 - 0.4 * r[0]) - v * H * 0.10
            cv2.fillPoly(Lf, [np.array([(x + math.cos(r[3] * 6 + v * 4 + j * 2.1) * 9 * k, y + math.sin(r[3] * 6 + v * 4 + j * 2.1) * 9 * k) for j in range(3)], np.int32)], float(1 - v), cv2.LINE_AA)
        f += Lf[:, :, None] * (RED * 0.6 + 0.2)[None, None, :]
        img = s.pose(t, squash=0.07 * math.exp(-u * 8) * math.cos(u * 20), jit=1.2 * k * g * (sp < 0))
        s.draw(f, img, W / 2, H * 0.78, 0.40); s.eyes(f, 1.0, 0.7)
        gp = s.at(*s.guard); L = s.L0(); s.star4(L, gp, (26 * math.exp(-u * 4) + 8) * k, 1.0, u * 3, 0.15); s.add(f, L, RED, 1.2)
        if sp > 0:                                                                   # 王冠: はね上がり、大きくなって 落ちてくる
            v = sp / (T_FALL - T_SPLIT); y = s.TH[1] - H * 0.13 - math.sin(min(v * 1.9, 3.1416)) * H * 0.42 + (v * v) * H * 0.42; w = W * (0.085 + 0.42 * v * v)
            s.crown(f, (W / 2 + math.sin(v * 5) * 12 * k, y), w, v * 9.0, 1.0, GOLD)
            if sp < 0.1: f += (1 - sp / 0.1) * 0.7
        if u < 0.1: f += (1 - u / 0.1) * 0.4
        return f, 8 * math.exp(-u * 6) + 3 * g * (sp < 0) + (14 * math.exp(-sp * 6) if sp > 0 else 0)

    def spike(s, f, x, base, h, w, lean=0.0):
        """床から つき出る 黒い剣(左の面は 暗く、右の面は にぶく光る。ふちは 赤)"""
        tip = (x + lean * h, base - h); sh_y = base - h * 0.80
        L_ = [(x - w / 2, base), (x - w / 2 + lean * h * 0.8, sh_y), tip, (x + lean * h * 0.4, base)]
        R_ = [(x + lean * h * 0.4, base), tip, (x + w / 2 + lean * h * 0.8, sh_y), (x + w / 2, base)]
        cv2.fillPoly(f, [np.array([(x - w / 2 - w * 0.10, base), (x - w / 2 - w * 0.10 + lean * h * 0.8, sh_y - w * 0.05), (tip[0], tip[1] - w * 0.22), (x + w / 2 + w * 0.10 + lean * h * 0.8, sh_y - w * 0.05), (x + w / 2 + w * 0.10, base)], np.int32)], (0.55, 0.03, 0.05), cv2.LINE_AA)   # 赤い ふち
        cv2.fillPoly(f, [np.array(L_, np.int32)], (0.035, 0.03, 0.045), cv2.LINE_AA); cv2.fillPoly(f, [np.array(R_, np.int32)], (0.13, 0.12, 0.16), cv2.LINE_AA)
        cv2.line(f, (int(x + lean * h * 0.4), int(base)), (int(tip[0]), int(tip[1])), (0.30, 0.28, 0.36), max(1, int(w * 0.04)), cv2.LINE_AA)                # しのぎ
        cv2.line(f, (int(x + w * 0.30 + lean * h * 0.6), int(base - h * 0.15)), (int(x + w * 0.36 + lean * h * 0.75), int(base - h * 0.70)), (0.42, 0.40, 0.50), max(1, int(w * 0.05)), cv2.LINE_AA)   # てり

    # 4) 王冠が 砕ける → 剣を もう一度 ひねると、黒い剣の 山が 床を つきやぶって 列ごとに こちらへ 走ってくる
    def sc_flag(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_FALL
        f = s.hall(1.0, [9] * 6, t, 1.0, False) * 0.6
        P0 = (W / 2, H * 0.60)
        for i in range(26):                                                          # 王冠の かけら
            r = s.rd[i]; a = -3.1416 * r[0]; v = u * (0.5 + r[1]); x = P0[0] + math.cos(a) * W * 0.55 * v; y = P0[1] + math.sin(a) * H * 0.30 * v + H * 0.35 * v * v
            if y < H * 0.97: b = r[3] * 6 + u * 7 * (r[4] - 0.5); cv2.fillPoly(f, [np.array([(x + math.cos(b + j * 2.094) * (8 + 14 * r[2]) * k, y + math.sin(b + j * 2.094) * (8 + 14 * r[2]) * k) for j in range(3)], np.int32)], (float(GOLD[0]), float(GOLD[1]), float(GOLD[2])), cv2.LINE_AA)
        L = s.L0()
        for i in range(3):
            v = (u - i * 0.1) / 0.8
            if 0 < v < 1: s.circ(L, P0, W * 1.2 * v, 1 - v, 4)
        for i in range(50):                                                          # のぼる 火の粉
            r = s.rd[i + 60]; y = ((r[1] - u * (0.08 + 0.15 * r[2])) % 1.0) * H; cv2.circle(L, (int(r[0] * W + math.sin(u * 2 + r[3] * 9) * 10 * k), int(y)), 1 if r[4] < 0.6 else 2, 0.8, -1)
        s.add(f, L, RED, 1.0)
        tw = u - 0.45                                                                # ひねる
        for j in range(2):                                                           # 本人の うしろ(奥)の 2列
            v = (tw - 0.05 - j * 0.16) / 0.16
            if v > 0:
                for i in range(7 - j): r = s.rd[100 + j * 9 + i]; x = W * (0.08 + 0.84 * (i + 0.2 + 0.6 * r[0]) / (7 - j)); s.spike(f, x, H * (0.60 + 0.07 * j), H * (0.10 + 0.05 * j) * back(v, 2.2) * (0.7 + 0.6 * r[1]), W * (0.035 + 0.012 * j), (r[2] - 0.5) * 0.3)
        img = s.pose(t, squash=0.04 * math.exp(-u * 9) + (0.07 * math.exp(-tw * 8) * math.cos(tw * 20) if tw > 0 else 0)); s.draw(f, img, W / 2, H * 0.78, 0.40); s.eyes(f, 1.0, 0.9)
        sh = 20 * math.exp(-u * 5)
        if tw > 0:
            gp = s.at(*s.tip); Lc = s.L0()                                            # 剣先から 手前へ 走る ひび
            for i in range(7): a = 0.45 + i * 0.37; g = ease(tw / 0.25); s.line(Lc, gp, (gp[0] + math.cos(a) * W * 0.9 * g, gp[1] + math.sin(a) * H * 0.3 * g), 1.0, max(1, int(3 * k)))
            s.add(f, Lc, RED, 1.0 * clamp(1.5 - tw), 2, 9)
            for j in range(6):                                                       # 手前へ 来る 列(近いほど 大きく、画面を ふさぐ)
                v = (tw - 0.40 - j * 0.20) / 0.20
                if v <= 0: continue
                n = 6 - j // 2; base = H * (0.96 + 0.10 * j); hh = H * (0.34 + 0.24 * j) * back(v, 1.8); ww = W * (0.10 + 0.045 * j)
                for i in range(n):
                    r = s.rd[130 + j * 8 + i]; x = W * (-0.05 + 1.10 * (i + 0.15 + 0.7 * r[0]) / n); s.spike(f, x, base, hh * (0.72 + 0.5 * r[1]), ww * (0.8 + 0.4 * r[3]), (r[2] - 0.5) * 0.22)
                if v < 1.2: f += (1 - min(1.0, v / 1.2)) * 0.10 * RED[None, None, :]; sh = max(sh, (8 + 3 * j) * math.exp(-max(0.0, v - 1) * 6))
        if u < 0.14: f = f + (1 - u / 0.14) * np.array([1.3, 0.9, 0.6], np.float32)[None, None, :]
        return f, sh

    # ---- この子だけの曲: はじめは おごそかな 王の ファンファーレ(長調)。目が灯ると 同じ ふしが 半音ずつ ゆがんで 短調に 落ちる。ひびは 低い 弦の きざみ、王冠は 高い鈴が 回りながら 落ち、砕ける ----
    def audio(s, path):
        a = Snd(DUR, 31); C = 130.81; maj = [0, 2, 4, 5, 7, 9, 11]; loc = [0, 1, 3, 5, 6, 8, 10]
        def fq(d, o=0, sc=maj): return C * 2 ** ((sc[d % 7] + 12 * (d // 7 + o)) / 12.0)
        def brass(d, t0, dur, g=0.06, o=1, sc=maj): a.note(fq(d, o, sc), t0, dur, g, "saw", 0.03); a.note(fq(d, o, sc) * 1.004, t0, dur, g * 0.6, "saw", 0.03)
        for i, d in enumerate([0, 0, 4, 7]): brass(d, 0.15 + i * 0.3, 0.4, 0.06)                                              # 王の ふし
        for d in (0, 2, 4): brass(d, 0.15, 1.3, 0.025, 0)
        a.boom(T_EYE, 42, 1.1, 0.6); a.note(fq(6, 2, loc), T_EYE, 0.8, 0.07, "sine")
        for i in range(6):                                                                                                    # 旗が 落ちるたび ふしが ゆがむ
            tk = T_EYE + 0.25 + i * 0.18; a.whoosh(tk - 0.05, tk + 0.03, 0.5, 0.6); brass([7, 6, 4, 3, 1, 0][i], tk, 0.3, 0.07, 1, loc); a.boom(tk + 0.12, 70, 0.3, 0.1)
        a.boom(T_CUT, 38, 1.2)
        for i in range(5): a.boom(T_CUT + 0.5 + i * 0.17 + 0.12, 55, 0.7, 0.15); brass([0, 1, 0, 3, 0][i], T_CUT + 0.62 + i * 0.17, 0.2, 0.08, 0, loc)
        a.boom(T_TWIST, 44, 1.2, 0.4); a.a += (a.nz - a.lp(a.nz, 0.4)) * a.env(T_TWIST, 0.001, 0.12) * 0.4
        tk = T_TWIST; i = 0
        while tk < T_SPLIT:                                                                                                   # ひび: 低い きざみ
            a.note(fq([0, 0, 1, 0, 3, 0, 1, 0][i % 8], -1, loc), tk, 0.09, 0.12, "saw", 0.003); a.hat(tk, 0.04)
            if i % 3 == 0: a.a += (a.nz - a.lp(a.nz, 0.5)) * a.env(tk, 0.001, 0.04) * 0.2
            tk += 0.1; i += 1
        a.rise(T_TWIST, T_SPLIT, 60, 240, 0.16)
        a.boom(T_SPLIT, 36, 1.5, 0.7); a.a += a.lp(a.nz, 0.25) * a.env(T_SPLIT, 0.01, 0.5) * 0.6
        for i in range(16): tk = T_SPLIT + 0.1 + i * (T_FALL - T_SPLIT - 0.1) / 16.0; a.note(fq(14 - (i % 7), 2, loc) * (1.2 - i * 0.03), tk, 0.12, 0.06, "bell")       # 回る 王冠
        a.rise(T_SPLIT + 0.9, T_FALL, 1400, 300, 0.10)
        a.boom(T_FALL, 34, 1.6, 0.8)
        for i in range(20): a.note(1800 + (i * 421) % 2200, T_FALL + i * 0.025, 0.1, 0.04, "bell")                             # 砕ける
        a.boom(T_FALL + 0.45, 44, 1.2, 0.4); a.a += (a.nz - a.lp(a.nz, 0.4)) * a.env(T_FALL + 0.45, 0.001, 0.12) * 0.4                    # ひねる
        for j in range(8): t0 = T_FALL + 0.5 + j * 0.2 - (0.1 if j < 2 else -0.15); a.rise(t0, t0 + 0.08, 900 - 60 * j, 2600 - 150 * j, 0.14); a.boom(t0 + 0.06, 60 - 3 * j, 0.7 + 0.12 * j, 0.25); a.hat(t0, 0.3, 0.04)   # 列ごとに つき出る
        for j, (d, o) in enumerate(((0, -1), (0, 0), (2, 0), (4, 0), (0, 1))): brass(d, T_FALL + 0.5 + j * 0.03, 2.2, 0.05, o, loc)
        s.fin_snd(a, DUR, 31); a.save(path)


if __name__ == "__main__":
    r = Mordred()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
