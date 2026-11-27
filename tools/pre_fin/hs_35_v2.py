"""ヴァルキリア(★5 戦乙女) 専用の必殺技「戦乙女ノ槍」
風の吹く 戦場あとの丘(ささった槍の 影)。下から とび上がって 宙で止まる → 盾の 鳥の紋が 光る → 盾の形(まるい金ぶち)の窓のカットイン →
紋の鳥が 盾から ぬけ出して 大きくなり うしろで はばたく → 鳥が 槍に 重なり、ふりかぶって 光の槍を 投げる →
槍を 追いかける画面: 鳥をまとった槍が 敵の 丸盾を 3枚 つぎつぎ ぶちぬく → 空が 晴れて、羽が まう
使い方: python3 hs_35.py 出力.mp4 / python3 hs_35.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 11.5
ICE = np.array([0.72, 0.87, 1.0], np.float32); GOLD = np.array([1.0, 0.82, 0.36], np.float32); BLUE = np.array([0.16, 0.34, 0.82], np.float32)
T_CUT, T_SKY, T_ANT, T_TH, T_FLY, T_END = 2.0, 3.9, 5.15, 5.5, 5.85, 8.5
HITS = (6.45, 7.05, 7.75); SHC = ((0.50, 0.12, 0.10), (0.20, 0.22, 0.26), (0.20, 0.07, 0.28)); SHR = (0.20, 0.26, 0.44)


class Val(Kit):
    def __init__(s, W=540):
        super().__init__(35, (0.72, 0.87, 1.0), "戦乙女ノ槍", "Valkyrie Lance", W)
        H, k = s.H, s.k
        s.eL = s.A(243, 330); s.eR = s.A(330, 350); s.fc = s.A(290, 330); s.hnd = s.A(125, 140); s.stip = s.A(425, 15); s.sbut = s.A(25, 185); s.shd = s.A(400, 505)
        s.ht = [s.text_img(c, hz.SERIF, int(78 * k), 3) for c in "戦乙女ノ槍"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * k), 1) for c in "奥義"]
        d = np.array([s.stip[0] - s.sbut[0], s.stip[1] - s.sbut[1]], np.float32); s.dir = d / np.linalg.norm(d); s.ang = math.degrees(math.atan2(s.dir[1], s.dir[0]))
        s.HF = 0.46; s.CX = W * 0.46; s.CY = H * 0.56
        # 盾の 鳥の紋を 絵から ぬき出す
        q = s.ah / 760.0; ay, ax = np.mgrid[0:s.ah, 0:s.aw]
        m = ((s.ch[:, :, :3].min(2) > 0.78) & (s.ch[:, :, 3] > 0.5) & (np.hypot(ax - s.shd[0], ay - s.shd[1]) < 66 * q)).astype(np.float32)
        ys, xs = np.nonzero(m); s.ebox = (xs.min(), xs.max() + 1, ys.min(), ys.max() + 1)
        s.emb = m[s.ebox[2]:s.ebox[3], s.ebox[0]:s.ebox[1]].copy(); s.ec = ((s.ebox[0] + s.ebox[1]) / 2.0, (s.ebox[2] + s.ebox[3]) / 2.0)
        s.eh = float(s.emb.shape[0])
        rg = np.random.RandomState(35); s.wd = rg.rand(40, 4).astype(np.float32); s.fe = rg.rand(26, 5).astype(np.float32); s.sk = rg.rand(60, 3).astype(np.float32)
        s.yn = s.YY / H
        # 丘と ささった槍の 影
        m = s.L0(); pts = [(0, H * 0.90)] + [(W * i / 10.0, H * (0.905 - 0.035 * math.sin(i * 0.36 + 0.4))) for i in range(11)] + [(W, H), (0, H)]
        cv2.fillPoly(m, [np.array(pts, np.int32)], 1.0, cv2.LINE_AA)
        for i in range(9):
            x = W * (0.05 + 0.115 * i + 0.03 * rg.rand()); y = H * 0.90; a = (rg.rand() - 0.5) * 0.7; ln = H * (0.07 + 0.08 * rg.rand())
            cv2.line(m, (int(x), int(y)), (int(x + math.sin(a) * ln), int(y - math.cos(a) * ln)), 1.0, max(1, int(3 * k)), cv2.LINE_AA)
            if i % 3 == 0: cv2.ellipse(m, (int(x + 16 * k), int(y - 6 * k)), (int(15 * k), int(15 * k)), 0, 180, 360, 1.0, -1, cv2.LINE_AA)
        s.hill = m

    # ---- 大道具 ----
    def storm(s, t, br=0.0):
        top = np.array([0.05, 0.07, 0.16], np.float32) * (1 - br) + np.array([0.36, 0.62, 0.98], np.float32) * br
        low = np.array([0.20, 0.24, 0.38], np.float32) * (1 - br) + np.array([0.90, 0.95, 1.0], np.float32) * br
        f = (top[None, None, :] * (1 - s.yn)[:, :, None] + low[None, None, :] * s.yn[:, :, None]).astype(np.float32)
        L = s.L0()
        for r in s.wd:                                                               # 風の すじ
            x = ((r[0] + t * (0.5 + r[2])) % 1.3 - 0.15) * s.W; y = r[1] * s.H * 0.9; ln = (40 + 90 * r[3]) * s.k
            s.line(L, (x, y), (x + ln, y - ln * 0.12), 0.22 + 0.2 * r[2], 1)
        f += cv2.GaussianBlur(L, (0, 0), 1.2 * s.k)[:, :, None] * ICE[None, None, :]
        return f

    def ground(s, f, br=0.0):
        c = np.array([0.02, 0.025, 0.05], np.float32) * (1 - br) + np.array([0.20, 0.30, 0.45], np.float32) * br
        return f * (1 - s.hill[:, :, None]) + c[None, None, :] * s.hill[:, :, None]

    def bird(s, L, c, size, rot=0.0, v=1.0, sy=1.0):
        """紋の鳥を 画面に置く。size=高さ(px) rot=度(0で 頭が上)"""
        sc = size / s.eh; a = math.radians(rot); ca, sa = math.cos(a), math.sin(a); ex, ey = s.emb.shape[1] / 2.0, s.emb.shape[0] / 2.0
        M = np.float32([[sc * ca, -sc * sy * sa, 0], [sc * sa, sc * sy * ca, 0]]); M[0, 2] = c[0] - (M[0, 0] * ex + M[0, 1] * ey); M[1, 2] = c[1] - (M[1, 0] * ex + M[1, 1] * ey)
        L += cv2.warpAffine(s.emb, M, (s.W, s.H), flags=cv2.INTER_LINEAR) * v

    def hero(s, f, t, cx=None, cy=None, **kw):
        img = s.pose(t, **kw); s.draw(f, img, s.CX if cx is None else cx, s.CY if cy is None else cy, s.HF)

    def crest(s, f, v):
        """盾の上の 紋が 光る"""
        if v <= 0.01: return
        c = s.at(*s.ec); L = s.L0(); s.bird(L, c, s.eh * s._pl[2], 0.0, min(1.0, v)); s.add(f, L, ICE, 0.9 * min(1.6, v), 2, 8)

    def spear_pts(s):
        return s.at(*s.sbut), s.at(*s.stip)

    def frame(s, t):
        if t < T_CUT: f, sh = s.sc_rise(t)
        elif t < T_SKY: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_FLY: f, sh = s.sc_throw(t)
        elif t < T_END: f, sh = s.sc_fly(t)
        else: f, sh = s.sc_end(t - T_END)
        f, sh = s.finale(f, sh, t, DUR, s.stip, (0.72, 0.87, 1), (1, 0.82, 0.36), -1)
        return s.post(f, ICE, 0.32, sh * s.k, t, 0.96)

    # 1) とび上がって 宙で 止まる
    def sc_rise(s, t):
        f = s.storm(t); f = s.ground(f)
        up = ease(t / 0.85); cy = s.H * 1.25 + (s.CY - s.H * 1.25) * up
        s.hero(f, t, cy=cy, squash=-0.06 * (1 - up) + (0.035 * math.exp(-(t - 0.85) * 9) if t > 0.85 else 0))
        if t < 0.85:                                                                 # とび上がる すじ
            L = s.L0()
            for i in range(5): x = s.CX + (i - 2) * 34 * s.k; s.line(L, (x, cy + s.H * 0.20), (x, cy + s.H * (0.30 + 0.12 * (i % 2))), 0.5 * (1 - up), 2)
            s.add(f, L, ICE, 0.6)
        s.crest(f, clamp((t - 1.25) / 0.6) * 1.4)
        return f * clamp(t / 0.3), 0

    # 2) 盾の形の 窓
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = np.zeros((H, W, 3), np.float32) + np.array([0.03, 0.04, 0.09], np.float32)
        L = s.L0()
        for r in s.wd[:24]: x = ((r[0] + t * (0.9 + r[2])) % 1.3 - 0.15) * W; y = r[1] * H; s.line(L, (x, y), (x + 120 * k, y - 14 * k), 0.25, 1)
        f += L[:, :, None] * ICE[None, None, :]
        g = back(u / 0.32, 1.4); C = (W / 2, H * 0.38); R = W * 0.40 * g
        m = s.L0(); cv2.circle(m, (int(C[0]), int(C[1])), int(R), 1.0, -1, cv2.LINE_AA)
        win = np.zeros((H, W, 3), np.float32) + (BLUE * 0.75)[None, None, :]
        rr = np.hypot(s.XX - C[0], s.YY - C[1]) / max(1.0, R); win *= (1.15 - 0.45 * rr)[:, :, None]
        zf = 1.05 + 0.05 * ease(u / 1.9); sc = zf * H / s.ah; img = s.pose(t)
        s.draw(win, img, C[0] - (s.fc[0] - s.aw / 2) * sc, C[1] + H * 0.01 - (s.fc[1] - s.ah / 2) * sc, zf)
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        cv2.circle(f, (int(C[0]), int(C[1])), int(R), (0.95, 0.76, 0.30), max(1, int(11 * k)), cv2.LINE_AA)                # 金の ふち
        cv2.circle(f, (int(C[0]), int(C[1])), int(R + 6 * k), (0.45, 0.32, 0.10), max(1, int(2 * k)), cv2.LINE_AA)
        for i in range(8):                                                           # びょう
            a = i * 0.7854 + 0.39; cv2.circle(f, (int(C[0] + math.cos(a) * R), int(C[1] + math.sin(a) * R)), max(1, int(5 * k)), (1.0, 0.93, 0.62), -1, cv2.LINE_AA)
        sw = (u - 0.45) / 0.35                                                       # ふちを 光が ひとまわり
        if 0 < sw < 1:
            L = s.L0(); cv2.ellipse(L, (int(C[0]), int(C[1])), (int(R), int(R)), 0, -90 + 360 * sw, -50 + 360 * sw, 1.0, max(1, int(9 * k)), cv2.LINE_AA); s.add(f, L, WHITE, 0.7, 2, 6)
        n = len(s.ht); step = W * 0.175
        for i, T in enumerate(s.ht):                                                 # 技名: 槍の向きに そって ささる
            a = (u - 0.55 - i * 0.11) / 0.2
            if a > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step - (1 - ease(a)) * 50 * k, H * 0.835 + (1 - ease(a)) * 30 * k, clamp(a * 2), 1.0)
        for i, T in enumerate(s.vo):
            if u > 0.4 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.755, 1.0, 1.0)
        if u > 1.25: s.text(f, s.txt_en, W / 2, H * 0.835 + 60 * k, ease((u - 1.25) / 0.3), 1.0)
        if u < 0.08: f = f + (1 - u / 0.08) * 0.7
        return f, 0

    # 3) 鳥が ぬけ出し、槍に 重なって、投げる
    def sc_throw(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_SKY; an = t - T_ANT; th = t - T_TH
        f = s.storm(t); f = s.ground(f)
        lean = 0.0; dx = 0.0
        if an > 0 and th < 0: lean = 16 * k * ease(an / 0.3); dx = -W * 0.03 * ease(an / 0.3)                 # ふりかぶる
        if th >= 0: lean = (16 - 44 * (1 - math.exp(-th * 30))) * k * math.exp(-th * 3.5); dx = W * (-0.03 + 0.09 * (1 - math.exp(-th * 25))) * math.exp(-th * 2.5)
        img = s.pose(t, lean=lean); s._pl = (s.CX + dx, s.CY, s.HF * H / s.ah)
        e0 = s.at(*s.ec); a, b = s.spear_pts(); mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        big = (W * 0.5, H * 0.30); L = s.L0()
        if an < 0:                                                                   # 盾から 出て 大きく。はばたきは 2回だけ
            g = ease(u / 0.6); c = (e0[0] + (big[0] - e0[0]) * g, e0[1] + (big[1] - e0[1]) * g); size = s.eh * s._pl[2] + (H * 0.50 - s.eh * s._pl[2]) * g
            fl = math.sin(clamp((u - 0.6) / 0.55) * 6.283) ** 2
            s.bird(L, c, size * (1 + 0.05 * fl), 0.0, 0.85, 1 - 0.22 * fl)
        elif th < 0:                                                                 # 槍へ 重なる
            g = ease2(an / 0.3); c = (big[0] + (mid[0] - big[0]) * g, big[1] + (mid[1] - big[1]) * g)
            s.bird(L, c, H * (0.50 - 0.28 * g), (s.ang + 90) * g, 0.85 + 0.15 * g)
        if an < 0 or th < 0: s.add(f, L, ICE, 0.75, 3, 14)
        s.draw(f, img, s.CX + dx, s.CY, s.HF)
        if an > 0 and th < 0:                                                        # 槍が 光る
            Ls = s.L0(); s.line(Ls, a, b, ease(an / 0.3), max(1, int(4 * k))); s.add(f, Ls, ICE, 1.0, 2, 9)
        sh = 0
        if th >= 0:                                                                  # 光の槍が とぶ
            d = W * 4.2 * th; hd = (mid[0] + s.dir[0] * (d + W * 0.2), mid[1] + s.dir[1] * (d + W * 0.2)); tl = (mid[0] + s.dir[0] * max(0.0, d - W * 0.9), mid[1] + s.dir[1] * max(0.0, d - W * 0.9))
            Ls = s.L0(); s.line(Ls, tl, hd, 1.0, max(1, int(5 * k))); s.bird(Ls, (hd[0] - s.dir[0] * W * 0.1, hd[1] - s.dir[1] * W * 0.1), H * 0.22, s.ang + 90, 0.9)
            s.add(f, Ls, ICE, 1.2, 3, 12)
            if th < 0.08: f = f + (1 - th / 0.08) * 0.8
            sh = 10 * math.exp(-th * 9)
        if u < 0.1: f = f * (u / 0.1) + (1 - u / 0.1) * 0.6
        return f, sh

    def shield(s, f, c, R, col, spin=0.0):
        cv2.circle(f, (int(c[0]), int(c[1])), int(R), (float(col[0]), float(col[1]), float(col[2])), -1, cv2.LINE_AA)
        for i in range(4): a = spin + i * 0.7854; cv2.line(f, (int(c[0] - math.cos(a) * R), int(c[1] - math.sin(a) * R)), (int(c[0] + math.cos(a) * R), int(c[1] + math.sin(a) * R)), (float(col[0] * 0.6), float(col[1] * 0.6), float(col[2] * 0.6)), max(1, int(R * 0.03)), cv2.LINE_AA)
        cv2.circle(f, (int(c[0]), int(c[1])), int(R), (0.42, 0.40, 0.44), max(2, int(R * 0.09)), cv2.LINE_AA)
        cv2.circle(f, (int(c[0]), int(c[1])), int(R * 0.22), (0.50, 0.48, 0.52), -1, cv2.LINE_AA); cv2.circle(f, (int(c[0] - R * 0.05), int(c[1] - R * 0.06)), int(R * 0.08), (0.80, 0.80, 0.86), -1, cv2.LINE_AA)

    # 4) 槍を 追う。丸盾を 3枚 ぶちぬく
    def sc_fly(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_FLY
        br = clamp((t - HITS[0]) / (HITS[2] - HITS[0])) * 0.55
        top = np.array([0.05, 0.07, 0.16], np.float32) * (1 - br) + np.array([0.30, 0.55, 0.95], np.float32) * br
        f = np.zeros((H, W, 3), np.float32) + top[None, None, :]
        C = np.array([W * 0.5, H * 0.46], np.float32); n2 = np.array([-s.dir[1], s.dir[0]], np.float32)
        L = s.L0()
        for r in s.sk:                                                               # 流れる すじ(槍と 逆むき)
            off = (r[0] - 0.5) * H * 1.3; al = ((r[1] - t * (2.2 + 2.5 * r[2])) % 1.0 - 0.5) * H * 1.6; ln = (60 + 200 * r[2]) * k
            p = C + n2 * off + s.dir * al; s.line(L, p, p + s.dir * ln, 0.18 + 0.3 * r[2], 1 if r[2] < 0.6 else 2)
        f += cv2.GaussianBlur(L, (0, 0), 1.0 * k)[:, :, None] * ICE[None, None, :]
        sh = 0; fl = 0.0
        for j in range(3):                                                           # 盾: 右上から せまり、当たると 6つに 割れて うしろへ
            v = t - HITS[j]; R = W * SHR[j]
            if -0.75 < v < 0:
                c = C + s.dir * (-v) * W * 2.6; s.shield(f, c, R, SHC[j], 0.3 * j)
            elif 0 <= v < 0.9:
                for i in range(6):
                    a0 = i * 1.0472 + 0.3 * j; am = a0 + 0.5236; fly = W * (0.5 + 1.4 * v) * (1 - math.exp(-v * 6)) * (0.7 + 0.3 * ((i * 7 + j) % 3))
                    cc = C + np.array([math.cos(am), math.sin(am)], np.float32) * fly - s.dir * W * 1.5 * v; ro = v * (3 + i) * (1 if i % 2 else -1)
                    P = [cc] + [cc + np.array([math.cos(a0 + ro + q * 1.0472 / 5), math.sin(a0 + ro + q * 1.0472 / 5)], np.float32) * R for q in range(6)]
                    cv2.fillPoly(f, [np.array(P, np.int32)], (float(SHC[j][0]), float(SHC[j][1]), float(SHC[j][2])), cv2.LINE_AA)
                    cv2.polylines(f, [np.array(P[1:], np.int32)], False, (0.42, 0.40, 0.44), max(2, int(R * 0.08)), cv2.LINE_AA)
                Lr = s.L0()
                for q in range(2):
                    w = (v - q * 0.08) / 0.5
                    if 0 < w < 1: s.circ(Lr, C, R * (0.6 + 2.2 * ease(w)), 1 - w, 3)
                s.add(f, Lr, ICE, 0.9)
                fl = max(fl, (1 - v / 0.09) * (0.6 + 0.2 * j) if v < 0.09 else 0); sh = max(sh, (9 + 6 * j) * math.exp(-v * 7))
        Ls = s.L0(); jit = math.sin(t * 60) * 1.5 * k                                # 鳥を まとった 槍(画面の まん中)
        s.line(Ls, C - s.dir * W * 0.75, C + s.dir * W * 0.24, 1.0, max(1, int(5 * k)))
        s.bird(Ls, C + s.dir * W * 0.02 + n2 * jit, H * 0.34, s.ang + 90, 0.9)
        cv2.fillPoly(Ls, [np.array([C + s.dir * W * 0.36, C + s.dir * W * 0.22 + n2 * 13 * k, C + s.dir * W * 0.22 - n2 * 13 * k], np.int32)], 1.0, cv2.LINE_AA)
        s.add(f, Ls, ICE, 0.62, 2, 7)
        f = f + fl
        w = t - (HITS[2] + 0.35)
        if w > 0: f = f + ease(w / (T_END - HITS[2] - 0.35)) * 1.1                      # 白く ぬける
        if u < 0.08: f = f + (1 - u / 0.08) * 0.6
        return f, sh

    # 5) 晴れた空の 奥から、紋の鳥を まとった 槍が 穂先を こちらに 向けて 飛んでくる
    def sc_end(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_END
        f = s.storm(t, 1.0); f = s.ground(f, 1.0)
        C = (W / 2, H * 0.36); g = clamp((u - 0.35) / 1.75); e = g ** 2.6
        L = s.L0()                                                                   # 奥から 流れる 風
        for r in s.sk:
            a = r[0] * 6.283; q = ((r[1] + t * (0.6 + 1.6 * g) * (0.6 + r[2])) % 1.0) ** 2; r0 = q * H * 0.9 + 16 * k
            s.line(L, (C[0] + math.cos(a) * r0, C[1] + math.sin(a) * r0), (C[0] + math.cos(a) * r0 * (1.12 + 0.3 * q), C[1] + math.sin(a) * r0 * (1.12 + 0.3 * q)), 0.25 + 0.6 * q, 1 if q < 0.5 else 2)
        f += cv2.GaussianBlur(L, (0, 0), 1.0 * k)[:, :, None] * 0.5
        if g < 0.6: s.hero(f, t, cy=s.CY + H * 0.14 + H * 0.5 * ease2((g - 0.25) / 0.35)); s.crest(f, 1.0 - g)
        if u > 0.35:
            S = (26 + 2400 * e) * k; fl = math.sin(t * 13) ** 2; c = (C[0], C[1] + H * 0.10 * e)
            B = s.L0(); s.bird(B, (c[0], c[1] + S * 0.05), S * (1 + 0.04 * fl), 0.0, 1.0, 1 - 0.20 * fl); B = np.clip(B, 0, 1)
            if S > 120 * k: B = np.clip((cv2.GaussianBlur(B, (0, 0), min(14 * k, S * 0.006)) - 0.5) * 6 + 0.5, 0, 1)
            Bo = cv2.dilate(B, np.ones((3, 3), np.uint8), iterations=max(1, int(3 * k)))
            f = f * (1 - Bo[:, :, None]) + np.array([0.10, 0.24, 0.62], np.float32)[None, None, :] * Bo[:, :, None]; f = f * (1 - B[:, :, None]) + np.array([0.96, 0.98, 1.0], np.float32)[None, None, :] * B[:, :, None]
            r = S * 0.15                                                             # 穂先(こちら向き: 4つの面の とがり)と 金の つば
            for sg in (-1, 1): cv2.fillPoly(f, [np.array([(c[0] + sg * r * 0.3, c[1] - r * 0.16), (c[0] + sg * r * 1.5, c[1] - r * 0.10), (c[0] + sg * r * 1.75, c[1]), (c[0] + sg * r * 1.5, c[1] + r * 0.10), (c[0] + sg * r * 0.3, c[1] + r * 0.16)], np.int32)], (1.0, 0.82, 0.36), cv2.LINE_AA)
            P = [(c[0], c[1] - r * 1.25), (c[0] + r * 0.55, c[1]), (c[0], c[1] + r * 1.25), (c[0] - r * 0.55, c[1])]
            cv2.fillPoly(f, [np.array(P, np.int32)], (0.05, 0.10, 0.25), cv2.LINE_AA)
            Q = [(c[0] + (x - c[0]) * 0.9, c[1] + (y - c[1]) * 0.9) for x, y in P]
            for a_, b_, col in ((0, 1, (0.98, 0.99, 1.0)), (1, 2, (0.70, 0.76, 0.88)), (2, 3, (0.50, 0.56, 0.70)), (3, 0, (0.84, 0.88, 0.96))): cv2.fillPoly(f, [np.array([Q[a_], Q[b_], c], np.int32)], col, cv2.LINE_AA)
            cv2.circle(f, (int(c[0]), int(c[1])), max(1, int(r * 0.07)), (1.0, 1.0, 1.0), -1, cv2.LINE_AA)
        if u < 0.5: f = f * ease(u / 0.5) + (1 - ease(u / 0.5))
        return f, 2 + 10 * e

    # ---- この子だけの曲: イ長調の 早がけ(タタタン タタタン)と 金管の のぼる ファンファーレ。盾を 1枚 ぬくたび 半音ずつ 上がり、さいごは 高い 合唱ふうの 和音 ----
    def audio(s, path):
        a = Snd(DUR, 35); A = 220.0
        def fq(semi): return A * 2 ** (semi / 12.0)
        def brass(semi, t0, ln, g=0.07):
            b = Snd(DUR, 1); b.note(fq(semi), t0, ln, 1.0, "saw", 0.025); b.note(fq(semi) * 1.004, t0, ln, 0.6, "saw", 0.025); a.a += a.lp(b.a.astype(np.float32), 0.16) * g * 2.2
        def gallop(t0, t1, step, g=0.5):
            tk = t0; i = 0
            while tk < t1:
                for q, gg in ((0, 0.6), (step * 0.25, 0.6), (step * 0.5, 1.0)): a.boom(tk + q, 68, g * gg, 0.07)
                a.hat(tk + step * 0.5, 0.10, 0.03); tk += step; i += 1
        a.a += a.lp(a.nz, 0.06) * (0.5 + 0.3 * np.sin(a.t * 2.3)) * a.gate(0, T_FLY, 0.3) * 0.6                               # 風
        a.whoosh(0.0, 0.85, 0.5, 0.25); a.boom(0.85, 70, 0.4, 0.15)
        brass(0, 0.9, 0.7, 0.05); brass(7, 0.9, 0.7, 0.04)
        for i, sm in enumerate((12, 16, 19)): a.note(fq(sm) * 2, 1.3 + i * 0.2, 0.5, 0.05, "bell")                               # 紋が 光る
        a.boom(T_CUT, 58, 0.9, 0.3); gallop(T_CUT + 0.2, T_SKY - 0.1, 0.4, 0.35)
        for sm, dt, ln in ((0, 0.2, 0.15), (4, 0.4, 0.15), (7, 0.6, 0.15), (12, 0.8, 0.5), (11, 1.2, 0.15), (12, 1.4, 0.4)): brass(sm, T_CUT + dt, ln, 0.08); brass(sm - 12, T_CUT + dt, ln, 0.04)
        a.note(fq(24) * 2, T_CUT + 0.5, 0.5, 0.05, "bell")
        for i, sm in enumerate((12, 14, 16, 19, 21, 24)): a.note(fq(sm) * 2, T_SKY + 0.05 + i * 0.1, 0.4, 0.045, "bell")          # 鳥が 出る
        a.whoosh(T_SKY + 0.6, T_SKY + 0.9, 0.25, 0.1); a.whoosh(T_SKY + 0.9, T_SKY + 1.15, 0.25, 0.1)                           # はばたき 2回
        for sm in (0, 7, 12): a.note(fq(sm), T_SKY + 0.2, 0.9, 0.03, "pad", 0.2)
        a.rise(T_ANT, T_TH, 300, 1500, 0.10)
        a.rise(T_TH, T_TH + 0.2, 1500, 5000, 0.18); a.hat(T_TH, 0.5, 0.08); a.boom(T_TH, 60, 1.0, 0.25)                         # 投げる
        gallop(T_FLY, HITS[2], 0.30, 0.45)
        a.a += a.lp(a.nz, 0.25) * a.gate(T_FLY, HITS[2] + 0.3, 0.1) * 0.22
        for j, th in enumerate(HITS):
            for sm in (0, 4, 7): brass(sm + j + 12, th - 0.3, 0.25, 0.05)
            a.boom(th, 46 - 4 * j, 1.1 + 0.3 * j, 0.4 + 0.15 * j); a.a += a.lp(a.nz, 0.45) * a.env(th, 0.003, 0.18) * 0.6
            for q in range(5): a.note(900 + q * 410 + j * 200, th + 0.02 + q * 0.03, 0.08, 0.05, "sine")                        # 木の 割れる音
        a.rise(HITS[2] + 0.2, T_END, 400, 1800, 0.06)
        for sm, g in ((0, 0.05), (4, 0.04), (7, 0.04), (12, 0.04), (16, 0.03)): a.note(fq(sm) * 2, T_END + 0.1, 1.6, g, "pad", 0.5)  # 晴れる
        for sm, dt, ln in ((7, 0.3, 0.2), (12, 0.6, 0.2), (16, 0.9, 0.9)): brass(sm, T_END + dt, ln, 0.06)
        for i, sm in enumerate((24, 28, 31, 36)): a.note(fq(sm), T_END + 1.0 + i * 0.12, 0.8, 0.05, "bell")
        s.fin_snd(a, DUR, 35); a.save(path)


if __name__ == "__main__":
    r = Val()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
