"""クロガネ(★6 千の刃) 専用の必殺技「千刃地獄」
障子に映る 大きな影 → 一筋の赤い線が走り、障子が左右に ひらく → 筆の一はけの中に 鬼の面(カットイン) →
居合の構えのまま 動かない。まわりの刀が 三→九→二十七→八十一→二百四十三→千 と ふえ、的を かこむ(数が 漢字で 出る) →
外の輪から 順に 飛びこむ。赤い切り筋だけが たまっていく → しずかに 納刀「カチン」 → ひと呼吸おいて 切り筋が いっせいに はじける
使い方: python3 hs_74.py 出力.mp4 / python3 hs_74.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 13.4
RED = np.array([0.95, 0.13, 0.12], np.float32); STEEL = np.array([0.82, 0.88, 1.0], np.float32); PAPER = np.array([0.95, 0.88, 0.74], np.float32); WOODC = np.array([0.20, 0.12, 0.07], np.float32)
T_SLASH, T_CUT, T_GATHER, T_FLY, T_SHEATH, T_BURST = 1.7, 3.0, 5.1, 9.1, 11.2, 11.85
RINGS = [3, 9, 18, 27, 36, 48]; KANJI = ["三", "九", "二十七", "八十一", "二百四十三", "千"]
STAGE = [T_GATHER + 0.45 + i * 0.6 for i in range(6)]


class Kurogane(Kit):
    def __init__(s, W=540):
        super().__init__(74, (0.95, 0.13, 0.12), "千刃地獄", "Thousand Blade Hell", W)
        s.eyeL = s.A(336, 302); s.eyeR = s.A(441, 292); s.mask = s.A(545, 175); s.meye = s.A(590, 160); s.hilt = s.A(215, 615)
        s.vt = [s.text_img(c, hz.SERIF, int(84 * s.k), 3) for c in "千刃地獄"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.kj = [s.text_img(c, hz.SERIF, int((120 if len(c) == 1 else (78 if len(c) == 3 else 52)) * s.k), 3) for c in KANJI]
        rg = np.random.RandomState(74); s.rd = rg.rand(200, 6).astype(np.float32)
        s.cuts = [(rg.rand() * 3.1416, (rg.rand() - 0.5) * 0.16, (rg.rand() - 0.5) * 0.16) for _ in range(40)]          # 切り筋(向き・ずれ)
        sil = s.cp.copy(); sil[:, :, :3] = 0; s.shadow = sil

    # ---- この子だけの絵 ----
    def room(s, lum=1.0):
        y = s.YY / s.H
        f = (np.array([0.10, 0.03, 0.04], np.float32)[None, None, :] * (0.5 + 1.2 * y)[:, :, None]) * lum
        gy = int(s.H * 0.86); f[gy:, :, :] = np.array([0.06, 0.035, 0.03], np.float32) * lum
        for i in range(-6, 7):                                                       # たたみの すじ
            cv2.line(f, (int(s.W / 2 + i * s.W * 0.09), gy), (int(s.W / 2 + i * s.W * 0.22), s.H), (0.12 * lum, 0.07 * lum, 0.05 * lum), 1, cv2.LINE_AA)
        return f.astype(np.float32)

    def shoji(s, f, open_, glow=1.0):
        """障子: 木のわくと 紙。open_ 0..1 で 左右にひらく"""
        W, H, k = s.W, s.H, s.k; off = int(W * 0.52 * open_)
        for sg in (-1, 1):
            x0 = (0 if sg < 0 else W // 2) + sg * off; x1 = x0 + W // 2
            a, b = max(0, x0), min(W, x1)
            if a >= b: continue
            f[:, a:b, :] = PAPER * 0.80 * glow
            for i in range(4):
                xx = x0 + int(i * W / 6); cv2.rectangle(f, (xx - int(3 * k), 0), (xx + int(3 * k), H), tuple(float(v) for v in WOODC), -1)
            for j in range(9): yy = int(j * H / 8); cv2.rectangle(f, (a, yy - int(3 * k)), (b, yy + int(3 * k)), tuple(float(v) for v in WOODC), -1)
            ex = x1 if sg < 0 else x0; cv2.rectangle(f, (ex - int(7 * k), 0), (ex + int(7 * k), H), tuple(float(v) for v in WOODC * 0.8), -1)

    def katana(s, Ls, Lr, c, ang, ln, v=1.0):
        """刀1本: はがね(Ls) と 赤い柄(Lr)。c=まん中、ang=切っ先の向き"""
        ca, sa = math.cos(ang), math.sin(ang); nx, ny = -sa, ca; w = ln * 0.035
        tip = (c[0] + ca * ln * 0.55, c[1] + sa * ln * 0.55); gd = (c[0] - ca * ln * 0.2, c[1] - sa * ln * 0.2); end = (c[0] - ca * ln * 0.45, c[1] - sa * ln * 0.45)
        cv2.fillPoly(Ls, [np.array([(gd[0] + nx * w, gd[1] + ny * w), (tip[0] - ca * ln * 0.08 + nx * w, tip[1] - sa * ln * 0.08 + ny * w), tip, (gd[0] - nx * w, gd[1] - ny * w)], np.int32)], float(v), cv2.LINE_AA)
        cv2.line(Lr, (int(gd[0]), int(gd[1])), (int(end[0]), int(end[1])), float(v), max(1, int(w * 2.2)), cv2.LINE_AA)

    def mask_eye(s, f, v):
        if v <= 0: return
        p = s.at(*s.meye); sc = s._pl[2]; L = s.L0()
        cv2.ellipse(L, (int(p[0]), int(p[1])), (max(1, int(16 * sc)), max(1, int(9 * sc))), -20, 0, 360, float(v), -1, cv2.LINE_AA)
        s.line(L, p, (p[0] + 90 * sc * v, p[1] - 40 * sc * v), 0.7 * v, 1)
        s.add(f, L, RED, 1.4, 2, 9)

    def frame(s, t):
        k = s.k
        if t < T_CUT: f, sh = s.sc_shoji(t)
        elif t < T_GATHER: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_FLY: f, sh = s.sc_gather(t)
        else: f, sh = s.sc_release(t)
        return s.post(f, RED, 0.6, sh * k, t, 0.86)

    # 1) 障子の影 → 赤い一筋 → ひらく
    def sc_shoji(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_SLASH
        op = ease2((u - 0.35) / 0.7) if u > 0.35 else 0.0
        f = s.room(1.0)
        s.pose(t); s.draw(f, s.cp, W / 2, H * 0.66, 0.44)
        s.mask_eye(f, op)
        L = s.L0(); s.ell(L, (W / 2, H * 0.86), (W * 0.4, W * 0.05), 0, 0.5 * op, 2); s.add(f, L, RED, 0.8)
        if op < 1:
            pane = np.zeros_like(f); flick = 1 + 0.05 * math.sin(t * 17)
            s.shoji(pane, op, (0.65 + 0.35 * ease(t / 0.6)) * flick)
            sh = np.zeros((H, W, 4), np.float32)                                     # 紙に うつる 大きな影
            sc = 0.80 * H / s.ah; im = cv2.resize(s.shadow, None, fx=sc, fy=sc)
            hh, ww = im.shape[:2]; x0 = int(W / 2 - ww / 2); y0 = int(H * 0.52 - hh / 2)
            xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + ww), min(H, y0 + hh)
            sh[ya:yb, xa:xb] = im[ya - y0:yb - y0, xa - x0:xb - x0]
            al = cv2.GaussianBlur(sh[:, :, 3], (0, 0), 7 * k) * 0.80 * ease((t - 0.3) / 0.8) * (1 - op)
            m = (pane.sum(2) > 0.01).astype(np.float32)
            paper = (pane.max(2) > 0.5).astype(np.float32)
            pane = pane * (1 - (al * paper)[:, :, None])
            f = f * (1 - m[:, :, None]) + pane * m[:, :, None]
        if u > 0:                                                                    # 赤い一筋
            g = ease(u / 0.12); L = s.L0(); a = (W * 1.05, H * 0.12); b = (-W * 0.05, H * 0.80)
            s.line(L, a, (a[0] + (b[0] - a[0]) * g, a[1] + (b[1] - a[1]) * g), clamp(1.6 - u * 1.2), 3 if u < 0.5 else 2)
            s.add(f, L, RED, 1.4)
            if u < 0.07: f = f + (1 - u / 0.07) * 0.8
        return f * clamp(t / 0.3), (8 * math.exp(-u * 9) if u > 0 else 0)

    # 2) 筆の一はけの中に 鬼の面
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = np.zeros((H, W, 3), np.float32) + np.array([0.03, 0.01, 0.012], np.float32)
        g = ease(u / 0.3); cy = H * 0.34
        top = []; bot = []
        for i in range(25):                                                          # かすれた 筆のふち
            x = -20 + (W + 40) * i / 24.0 * g; r = s.rd[i]
            top.append((x, cy - H * 0.115 + (r[0] - 0.5) * 16 * k + i * 1.2 * k)); bot.append((x, cy + H * 0.115 + (r[1] - 0.5) * 22 * k + i * 1.2 * k))
        m = s.L0(); cv2.fillPoly(m, [np.array(top + bot[::-1], np.int32)], 1.0, cv2.LINE_AA)
        for i in range(14):                                                          # はねた 墨
            r = s.rd[i + 30]; cv2.circle(m, (int(r[0] * W * g), int(cy + (r[1] - 0.5) * H * 0.34)), int((2 + 6 * r[2]) * k), 1.0, -1, cv2.LINE_AA)
        win = np.zeros((H, W, 3), np.float32) + (RED * 0.55)[None, None, :]
        zf = 1.55; sc = zf * H / s.ah
        s.pose(t); s.draw(win, s.cp, W * 0.56 - (s.mask[0] - 30 * k - s.aw / 2) * sc - u * 6 * k, cy - (s.mask[1] + 40 * k - s.ah / 2) * sc, zf)
        s.mask_eye(win, 0.7 + 0.3 * math.sin(u * 11))
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        n = len(s.vt)
        for i, T in enumerate(s.vt):                                                 # 技名: 1字ずつ 斬りこまれる(切り筋つき)
            a = (u - 0.5 - i * 0.2) / 0.12
            if a > 0:
                p = (W * (0.20 + i * 0.20), H * 0.66 + (i % 2) * 22 * k); s.text(f, T, p[0], p[1], clamp(a), 1.0)
                if a < 2.5:
                    L = s.L0(); q = clamp(a); s.line(L, (p[0] + 70 * k, p[1] - 80 * k), (p[0] + 70 * k - 140 * k * q, p[1] - 80 * k + 160 * k * q), clamp(1.6 - a / 1.6), 2); s.add(f, L, WHITE * 0.6 + RED * 0.4, 1.2)
        for i, T in enumerate(s.vo):
            if u > 0.3 + i * 0.1: s.text(f, T, W * 0.5 + (i - 0.5) * 44 * k, H * 0.54, 1.0, 1.0)
        if u > 1.5: s.text(f, s.txt_en, W / 2, H * 0.66 + 92 * k, ease((u - 1.5) / 0.3), 1.0)
        if u < 0.08: f += (1 - u / 0.08) * 0.8
        return f, 0

    def rings(s, f, t, upto, fly=None):
        """的を かこむ 刀の輪。fly[i]=その輪が 飛びこんだ割合(0..1)"""
        W, H, k = s.W, s.H, s.k; O = (W / 2, H * 0.24); Ls = s.L0(); Lr = s.L0()
        for i in range(min(upto, 6)):
            n = RINGS[i]; pop = back((t - STAGE[i]) / 0.3, 1.6) if fly is None else 1.0
            q = fly[i] if fly else 0.0
            if q >= 1: continue
            rr = W * (0.15 + 0.105 * i) * (1.35 - 0.35 * pop) * (1 - q); rot = t * (0.35 if i % 2 else -0.35) * (1 - 0.1 * i)
            ln = W * (0.10 - 0.006 * i) * pop
            for j in range(n):
                a = rot + j * 6.2832 / n; c = (O[0] + math.cos(a) * rr, O[1] + math.sin(a) * rr * 0.92)
                if -40 < c[0] < W + 40 and -40 < c[1] < H * 0.62: s.katana(Ls, Lr, c, a + math.pi, ln, 1.0)
        s.add(f, Ls, STEEL, 0.75, 1.5, 6); f += Lr[:, :, None] * RED[None, None, :] * 0.9
        return O

    def target(s, f, O, t, v=1.0):
        L = s.L0(); r = s.W * 0.055
        s.circ(L, O, r, v, 2); s.circ(L, O, r * 0.45, v * 0.8, 1)
        for i in range(4): a = i * 1.5708 + t * 0.6; s.line(L, (O[0] + math.cos(a) * r * 0.7, O[1] + math.sin(a) * r * 0.7), (O[0] + math.cos(a) * r * 1.5, O[1] + math.sin(a) * r * 1.5), v, 2)
        s.add(f, L, RED, 0.9)

    # 3) 動かない。刀だけが ふえていく
    def sc_gather(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_GATHER
        f = s.room(0.9)
        n = len([x for x in STAGE if x <= t]); last = max([x for x in STAGE if x <= t] or [-9]); since = t - last
        O = (W / 2, H * 0.24); s.target(f, O, t, ease(u / 0.4))
        s.rings(f, t, n)
        s.pose(t, breath=0.2); s.draw(f, s.cp, W / 2, H * 0.72, 0.40)
        s.mask_eye(f, 0.4 + 0.1 * n)
        if n > 0:                                                                    # いまの 本数(漢字)
            T = s.kj[n - 1]; a = since / 0.14
            if a < 1: T = tuple(cv2.resize(q, None, fx=1 + (1 - a) * 0.9, fy=1 + (1 - a) * 0.9) for q in T)
            s.text(f, T, W * 0.80 if n < 3 else W * 0.70, H * 0.495, clamp(a * 2), 1.0)
            if since < 0.1: f += (1 - since / 0.1) * 0.18
        if u < 0.15: f += (1 - u / 0.15) * 0.6
        return f, 2.5 * math.exp(-since * 12)

    # 4) 飛びこむ → 納刀 → はじける
    def sc_release(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_FLY; sh = t - T_SHEATH; br = t - T_BURST
        f = s.room(0.9 * (1 - 0.5 * clamp((u - 1.5) / 0.4)) if br < 0 else 0.5)
        O = (W / 2, H * 0.24)
        fly = [clamp((u - (5 - i) * 0.24) / 0.2) for i in range(6)]                  # 外の輪から
        hits = sum(1 for q in fly if q >= 1); ncut = hits * 6
        if br < 0: s.target(f, O, t, 1.0)
        s.rings(f, t, 6, fly)
        L = s.L0(); thick = 1 if br < 0 else int(2 + 6 * math.exp(-br * 3))
        for i in range(min(ncut, len(s.cuts))):                                      # たまっていく 切り筋
            a, dx, dy = s.cuts[i]; c = (O[0] + dx * W, O[1] + dy * H); ln = W * (0.5 + 0.5 * s.rd[i][0])
            s.line(L, (c[0] - math.cos(a) * ln, c[1] - math.sin(a) * ln), (c[0] + math.cos(a) * ln, c[1] + math.sin(a) * ln), 0.75 if br < 0 else 1.0, thick)
        s.add(f, L, RED * (0.75 if br < 0 else 1.0) + (WHITE * 0.3 if br > 0 else 0), 0.8 if br < 0 else 1.6 * clamp(1.5 - br * 0.5), 2, 9 if br < 0 else 16)
        for i in range(6):                                                           # 当たった 一瞬の光
            q = u - (5 - i) * 0.24 - 0.2
            if 0 < q < 0.09: f += (1 - q / 0.09) * 0.3
        sq = 0.05 * math.exp(-sh * 14) if sh > 0 else 0.0
        s.pose(t, breath=0.2 if sh < 0 else 0.0, squash=sq); s.draw(f, s.cp, W / 2, H * 0.72, 0.40)
        s.mask_eye(f, 1.0 if br < 0 else clamp(1.4 - br * 0.3))
        if sh > 0:                                                                   # 納刀: つばもとの 小さな火花
            p = s.at(*s.hilt); L = s.L0(); v = clamp(1 - sh / 0.35)
            s.star4(L, p, (26 * v + 3) * k, v, 0.4, 0.12)
            for i in range(8): a = i * 0.785 + 0.3; s.line(L, (p[0] + math.cos(a) * 10 * k, p[1] + math.sin(a) * 10 * k), (p[0] + math.cos(a) * (10 + 50 * sh / 0.35) * k, p[1] + math.sin(a) * (10 + 50 * sh / 0.35) * k), v, 1)
            s.add(f, L, WHITE, 1.3)
        if br > 0:                                                                   # はじける: 赤い墨
            L = s.L0()
            for i in range(60):
                r = s.rd[i + 60]; a = r[0] * 6.283; rr = W * (0.03 + br * (0.25 + 0.9 * r[1])); c = (O[0] + math.cos(a) * rr, O[1] + math.sin(a) * rr)
                cv2.circle(L, (int(c[0]), int(c[1])), max(1, int((3 + 16 * r[2]) * k * clamp(1.4 - br * 0.5))), 1.0, -1, cv2.LINE_AA)
            for i in range(3):
                q = (br - i * 0.1) / 0.7
                if 0 < q < 1: s.circ(L, O, W * 1.3 * q, 1 - q, 4)
            f += (L + cv2.GaussianBlur(L, (0, 0), 6 * k))[:, :, None] * RED[None, None, :] * 0.9
            if br < 0.12: f = f + (1 - br / 0.12) * np.array([1.0, 0.5, 0.5], np.float32)[None, None, :] * 1.3
            edge = np.clip((np.sqrt(((s.XX - W / 2) / W) ** 2 + ((s.YY - H / 2) / H) ** 2) - (0.75 - 0.6 * clamp((br - 0.7) / 0.8))) / 0.12, 0, 1)      # まわりから 墨が しみる
            f *= (1 - edge)[:, :, None]
        end = DUR - t
        if end < 0.4: f = f * clamp(end / 0.35)
        return f, 3 * hits * (u < 1.6) * 0.4 + (22 * math.exp(-br * 4) if br > 0 else 0) + (3 * math.exp(-sh * 20) if sh > 0 else 0)

    # ---- この子だけの曲: 三味線ふうの はじく音(都節)と 拍子木。刀が ふえるたび 一音 上がる。納刀の「カチン」のあとは 無音、それから 大太鼓 ----
    def audio(s, path):
        a = Snd(DUR, 74); D = 146.83; miy = [0, 1, 5, 7, 8]
        def fq(d, o=0): return D * 2 ** ((miy[d % 5] + 12 * (d // 5 + o)) / 12.0)
        def pluck(d, t0, g=0.14, o=0): a.note(fq(d, o), t0, 0.22, g, "fm", 0.002); a.note(fq(d, o) * 2.01, t0, 0.07, g * 0.4, "sq", 0.001)
        def clack(t0, g=0.3): a.a += (a.nz - a.lp(a.nz, 0.7)) * a.env(t0, 0.0005, 0.02) * g; a.note(1900, t0, 0.03, g * 0.3, "sine", 0.0005)
        def taiko(t0, g=1.0): a.boom(t0, 48, g, 0.5)
        a.a += a.lp(a.nz, 0.01) * 0.4 * a.gate(0, T_SLASH, 0.3)
        for i, d in enumerate([0, 1, 0]): pluck(d, 0.4 + i * 0.45, 0.10)
        a.whoosh(T_SLASH - 0.12, T_SLASH + 0.1, 0.8, 0.5); clack(T_SLASH + 0.12, 0.5); taiko(T_SLASH + 0.35, 0.7)
        a.rise(T_SLASH + 0.35, T_SLASH + 1.0, 90, 50, 0.2)                                                           # 障子が すべる
        taiko(T_CUT, 1.0)
        for i in range(4): a.whoosh(T_CUT + 0.42 + i * 0.2, T_CUT + 0.5 + i * 0.2, 0.5, 0.6); pluck([7, 5, 3, 0][i], T_CUT + 0.5 + i * 0.2, 0.16)
        for i, x in enumerate(STAGE):                                                                                 # ふえるたび 一音 上がる
            pluck(i, x, 0.16, 1); clack(x, 0.25); a.note(fq(i, 2), x, 0.25, 0.04, "bell")
            for j in range(min(6, RINGS[i] // 3)): a.note(2400 + j * 200, x + j * 0.02, 0.05, 0.03, "sine")              # しゃりん
        tk = T_GATHER; i = 0
        while tk < T_FLY - 0.1:
            if i % 4 == 0: taiko(tk, 0.28)
            if i % 4 == 2: clack(tk, 0.12)
            tk += 0.15; i += 1
        a.rise(STAGE[-1], T_FLY, 150, 500, 0.12)
        for i in range(6):
            tk = T_FLY + i * 0.24; a.whoosh(tk, tk + 0.2, 0.6, 0.6); pluck(9 - i, tk + 0.2, 0.15, 1); taiko(tk + 0.2, 0.45)
            for j in range(4): pluck(4 + (i + j) % 5, tk + j * 0.06, 0.06, 1)
        clack(T_SHEATH, 0.7); a.note(2600, T_SHEATH, 0.5, 0.10, "bell")                                               # カチン
        taiko(T_BURST, 1.6); taiko(T_BURST + 0.18, 0.9); a.a += a.lp(a.nz, 0.3) * a.env(T_BURST, 0.005, 0.5) * 0.6
        for d, o in ((0, -1), (1, 0), (3, 0), (0, 1)): a.note(fq(d, o), T_BURST, 1.3, 0.07, "saw", 0.01)
        pluck(0, T_BURST + 0.9, 0.14, 0); pluck(1, T_BURST + 1.15, 0.10, 0)
        a.save(path)


if __name__ == "__main__":
    r = Kurogane()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
