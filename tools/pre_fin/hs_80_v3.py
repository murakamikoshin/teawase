"""ミライ(★6 まだ見ぬ者) 専用の必殺技「未ダ見ヌ明日」— 紙と えんぴつの世界(ほかの子は みんな 光の世界。この子だけ 白い紙)
まっ白な紙。えんぴつの線が 地平を引き、この子の りんかくを 描き、色が のる → 黒い紙の カットイン(フードの闇に 星ひとつ。技名は 1字ずつ 打たれる) →
足もとから 点線の道が 何本も 枝分かれして、それぞれの先に 小さな扉(ありえた明日) → こぶしを にぎると、えらばなかった道が 消しゴムで 消える →
のこった扉が 大きくなり、ひらく。金の光 → ページが めくれる。つぎのページは 色のついた 朝
使い方: python3 hs_80.py 出力.mp4 / python3 hs_80.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 13.6
PAPER = np.array([0.96, 0.94, 0.88], np.float32); INKC = np.array([0.16, 0.15, 0.20], np.float32); GOLD = np.array([1.0, 0.82, 0.38], np.float32)
T_CUT, T_PATH, T_PICK, T_DOOR, T_OPEN, T_PAGE = 2.6, 4.7, 7.2, 8.6, 9.5, 11.0
DOORS = [(0.44, 0.62), (0.57, 0.49), (0.72, 0.58), (0.50, 0.33), (0.80, 0.40), (0.90, 0.55), (0.66, 0.22)]; PICK = 6


class Mirai(Kit):
    def __init__(s, W=540):
        super().__init__(80, (1.0, 0.82, 0.38), "未ダ見ヌ明日", "Unseen Tomorrow", W)
        s.star = s.A(272, 372); s.fist = s.A(232, 360); s.hood = s.A(300, 262)
        s.ht = [s.text_img(c, hz.SERIF, int(66 * s.k), 3) for c in "未ダ見ヌ明日"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        rg = np.random.RandomState(80); s.rd = rg.rand(300, 5).astype(np.float32)
        al = s.cp[:, :, 3]; rim = np.clip(cv2.dilate(al, np.ones((4, 4), np.uint8)) - cv2.erode(al, np.ones((3, 3), np.uint8)), 0, 1)
        dark = (s.cp[:, :, :3].max(2) < 0.25) * al                                    # 絵の中の 黒い線も えんぴつ線として使う
        s.rim = np.zeros_like(s.cp); s.rim[:, :, :3] = INKC; s.rim[:, :, 3] = np.clip(rim + dark, 0, 1) * 0.9
        s.rows = np.arange(s.ph, dtype=np.float32)[:, None]
        s.gnd = s.H * 0.84; s.grain = 0.97 + 0.05 * s.neb[:, :s.W]
        s.paths = []
        S0 = (s.W * 0.34, s.gnd - 8 * s.k)
        for i, (dx, dy) in enumerate(DOORS):
            D = (dx * s.W, dy * s.H + 24 * s.k); C = ((S0[0] + D[0]) / 2 + (rg.rand() - 0.5) * s.W * 0.25, S0[1] - (S0[1] - D[1]) * (0.2 + 0.3 * rg.rand()))
            s.paths.append([((1 - v) ** 2 * S0[0] + 2 * v * (1 - v) * C[0] + v * v * D[0], (1 - v) ** 2 * S0[1] + 2 * v * (1 - v) * C[1] + v * v * D[1]) for v in np.linspace(0, 1, 34)])

    # ---- この子だけの絵 ----
    def page(s, dark=False):
        if dark: return np.zeros((s.H, s.W, 3), np.float32) + np.array([0.07, 0.07, 0.09], np.float32)[None, None, :] * s.grain[:, :, None]
        return (PAPER[None, None, :] * s.grain[:, :, None]).astype(np.float32)

    def ink(s, f, L, a=0.9, col=INKC):
        m = np.clip(L, 0, 1)[:, :, None] * a; f[:] = f * (1 - m) + m * np.asarray(col, np.float32)[None, None, :]

    def hero(s, f, t, cx, feet, hfrac, colour=1.0, sweep=1.0, **kw):
        """えんぴつ線(上から 描かれる) と 色"""
        sc = hfrac * s.H / s.ah; cy = feet - hfrac * s.H / 2
        img = s.pose(t, **kw)
        if colour > 0: s.draw(f, img, cx, cy, hfrac, colour)
        else: s._pl = (cx, cy, sc)
        if colour < 1:
            r = s.rim.copy(); r[:, :, 3] *= np.clip((sweep * (s.ph + 40) - s.rows) / 40.0, 0, 1) * (1 - colour)
            s.put(f, r, cx, cy, sc, 1.0)

    def door(s, L, c, w, h, v=1.0, th=2):
        x0, x1, y1 = c[0] - w / 2, c[0] + w / 2, c[1] + h / 2; y0 = c[1] - h / 2 + w / 2
        s.line(L, (x0, y1), (x0, y0), v, th); s.line(L, (x1, y1), (x1, y0), v, th); s.line(L, (x0, y1), (x1, y1), v, th)
        cv2.ellipse(L, (int(c[0]), int(y0)), (int(w / 2), int(w / 2)), 0, 180, 360, float(v), th, cv2.LINE_AA)
        cv2.circle(L, (int(x1 - w * 0.2), int(c[1] + h * 0.1)), max(1, int(w * 0.05)), float(v), -1, cv2.LINE_AA)

    def glint(s, f, p, r, t, v=1.0):
        L = s.L0(); s.star4(L, p, r * (1 + 0.15 * math.sin(t * 7)), v, 0.0, 0.14); s.star4(L, p, r * 0.5, v, 0.7854, 0.2)
        m = np.clip(L + cv2.GaussianBlur(L, (0, 0), max(2, r * 0.3)), 0, 1)[:, :, None]; f[:] = f * (1 - m * 0.9) + m * GOLD[None, None, :] * 1.05

    def frame(s, t):
        k = s.k
        if t < T_CUT: f, sh = s.sc_sketch(t)
        elif t < T_PATH: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_DOOR: f, sh = s.sc_paths(t)
        elif t < T_PAGE: f, sh = s.sc_door(t)
        else: f, sh = s.sc_page(t - T_PAGE)
        f, sh = s.finale(f, sh, t, DUR, s.star, (1, 0.85, 0.4), (1, 1, 1), 1)
        return s.post(f, GOLD, 0.25, sh * k, t, 0.99)

    # 1) 白い紙に 描かれる
    def sc_sketch(s, t):
        W, H, k = s.W, s.H, s.k
        f = s.page(); L = s.L0()
        s.line(L, (W * 0.06, s.gnd), (W * 0.06 + W * 0.88 * ease(t / 0.6), s.gnd + 2 * k), 0.9, 2)
        g = clamp(1 - (t - 1.5) / 0.6)                                               # 下がきの あたり線(色が のると 消える)
        if t > 0.3 and g > 0:
            c = (W / 2, s.gnd - H * 0.34); a1 = 360 * ease((t - 0.3) / 0.5)
            cv2.ellipse(L, (int(c[0]), int(c[1])), (int(W * 0.17), int(W * 0.17)), 0, -90, -90 + a1, 0.35 * g, 1, cv2.LINE_AA)
            s.line(L, (W / 2, s.gnd - H * 0.47), (W / 2, s.gnd - H * 0.47 + H * 0.47 * ease((t - 0.5) / 0.5)), 0.3 * g, 1); s.line(L, (W * 0.3, c[1]), (W * 0.3 + W * 0.4 * ease((t - 0.6) / 0.4), c[1]), 0.3 * g, 1)
        s.ink(f, L)
        col = ease2((t - 1.5) / 0.7)
        s.hero(f, t, W / 2, s.gnd, 0.46, col, ease2((t - 0.6) / 0.9), squash=(-0.03 * math.exp(-(t - 2.2) * 9) if t > 2.2 else 0))
        if t > 0.6 and t < 1.55:                                                     # えんぴつの先
            y = s.gnd - H * 0.46 + H * 0.46 * ease2((t - 0.6) / 0.9); x = W / 2 + math.sin(t * 38) * W * 0.16
            cv2.line(f, (int(x), int(y)), (int(x + 26 * k), int(y - 40 * k)), (0.85, 0.65, 0.2), max(2, int(5 * k)), cv2.LINE_AA); cv2.circle(f, (int(x), int(y)), max(1, int(2 * k)), (0.1, 0.1, 0.1), -1)
        if t > 2.2: s.glint(f, s.at(*s.star), 16 * k * clamp((t - 2.2) / 0.2), t)
        return f * clamp(t / 0.25) + (1 - clamp(t / 0.25)), 0

    # 2) 黒い紙: やぶれた帯の中に フード。闇に 星ひとつ
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.page(True); cyb = H * 0.33; hh = H * 0.13 * ease(u / 0.25)
        top = [(W * i / 30.0, cyb - hh + (s.rd[i][0] - 0.5) * 16 * k) for i in range(31)]; bot = [(W * i / 30.0, cyb + hh + (s.rd[i + 40][0] - 0.5) * 16 * k) for i in range(31)]
        m = s.L0(); cv2.fillPoly(m, [np.array(top + bot[::-1], np.int32)], 1.0, cv2.LINE_AA)
        win = s.page(); zf = 1.5; sc = zf * H / s.ah
        img = s.pose(t); s.draw(win, img, W * 0.5 - (s.hood[0] - s.aw / 2) * sc - u * 5 * k, cyb - (s.hood[1] - s.ah / 2) * sc, zf)
        if u > 0.45: s.glint(win, s.at(s.hood[0] + 34 * k, s.hood[1] - 12 * k), 18 * k * back((u - 0.45) / 0.25, 2.0), t)       # 目のあたりに 星
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        n = len(s.ht); step = W * 0.148; y = H * 0.66; shown = int((u - 0.6) / 0.14) + 1 if u > 0.6 else 0
        for i, T in enumerate(s.ht):                                                 # 1字ずつ 打たれる
            if i < shown: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, y + (2 * k if int((u - 0.6) / 0.14) == i else 0), 1.0, 1.0)
        if 0 < shown <= n and int(u * 4) % 2 == 0:
            cx = W / 2 + (min(shown, n) - (n - 1) / 2 - 0.5) * step + 6 * k; cv2.rectangle(f, (int(cx), int(y - 34 * k)), (int(cx + 5 * k), int(y + 34 * k)), (0.95, 0.95, 0.95), -1)
        for i, T in enumerate(s.vo):
            if u > 0.4 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.545, 1.0, 1.0)
        if u > 1.6: s.text(f, s.txt_en, W / 2, y + 62 * k, ease((u - 1.6) / 0.3), 1.0)
        if u < 0.1: f = f * (u / 0.1) + (1 - u / 0.1)
        return f, 0

    # 3) 枝分かれする道と 扉 → えらばなかった道が 消える
    def sc_paths(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_PATH; pk = t - T_PICK
        f = s.page(); L = s.L0(); Lg = s.L0(); s.line(L, (W * 0.04, s.gnd), (W * 0.96, s.gnd + 2 * k), 0.9, 2)
        for i, pts in enumerate(s.paths):
            a = clamp((u - 0.2 - i * 0.22) / 0.7); keep = 1.0
            if pk > 0 and i != PICK: keep = clamp(1 - (pk - 0.15 - (i % 3) * 0.12) / 0.35)
            if a <= 0 or keep <= 0: continue
            tgt = Lg if (i == PICK and pk > 0.5) else L
            n = int(a * (len(pts) - 1))
            for j in range(0, n, 2): s.line(tgt, pts[j], pts[j + 1], 0.75 * keep, 2)
            if a >= 1:
                p = back((u - 0.9 - i * 0.22) / 0.25, 2.0); d = DOORS[i]; s.door(tgt, (d[0] * W, d[1] * H), 40 * k * p, 62 * k * p, 0.9 * keep, 2)
            if pk > 0 and i != PICK and keep < 1:                                    # 消しゴムの かす
                for j in range(6): r = s.rd[i * 6 + j]; q = pts[int(r[0] * 33)]; cv2.circle(f, (int(q[0] + (r[1] - 0.5) * 20 * k), int(q[1] + (1 - keep) * 40 * k * r[2])), max(1, int(2 * k)), (0.72, 0.70, 0.66), -1)
        s.ink(f, L); s.ink(f, Lg + cv2.GaussianBlur(Lg, (0, 0), 3 * k), 1.0, GOLD * 0.9)
        s.hero(f, t, W * 0.20, s.gnd, 0.27, 1.0, squash=(0.05 * math.exp(-pk * 9) if pk > 0 else 0))
        if pk > 0:
            s.glint(f, s.at(*s.fist), 22 * k * clamp(1.4 - pk * 0.6), t)
            d = DOORS[PICK]; s.glint(f, (d[0] * W, d[1] * H - 40 * k), 12 * k * clamp((pk - 0.5) / 0.3), t)
        if u < 0.12: f = f * (u / 0.12) + (1 - u / 0.12)
        return f, (3 * math.exp(-pk * 10) if pk > 0 else 0)

    # 4) 扉が ひらく
    def sc_door(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_DOOR; op = t - T_OPEN
        f = s.page(); L = s.L0(); s.line(L, (W * 0.04, s.gnd), (W * 0.96, s.gnd + 2 * k), 0.9, 2)
        g = back(u / 0.6, 1.2); C = (W / 2, s.gnd - H * 0.29 * g); dw = W * 0.46 * g; dh = H * 0.58 * g
        o = ease2(op / 0.9) if op > 0 else 0.0
        if o > 0:                                                                    # 中の光
            M = s.L0(); x0 = C[0] - dw / 2 * o; x1 = C[0] + dw / 2 * o; y0 = C[1] - dh / 2 + dw / 2
            cv2.rectangle(M, (int(x0), int(y0)), (int(x1), int(C[1] + dh / 2)), 1.0, -1); cv2.ellipse(M, (int(C[0]), int(y0)), (max(1, int(dw / 2 * o)), int(dw / 2)), 0, 180, 360, 1.0, -1, cv2.LINE_AA)
            R = s.L0()
            for i in range(14):
                a = -3.1416 * (i + 0.5) / 14; cv2.fillPoly(R, [np.array([(C[0], C[1]), (C[0] + math.cos(a - 0.05) * W * 1.6, C[1] + math.sin(a - 0.05) * W * 1.6), (C[0] + math.cos(a + 0.05) * W * 1.6, C[1] + math.sin(a + 0.05) * W * 1.6)], np.int32)], 0.35 * o, cv2.LINE_AA)
            m = np.clip(M + cv2.GaussianBlur(M, (0, 0), 14 * k) * 0.9 + cv2.GaussianBlur(R, (0, 0), 5 * k), 0, 1)[:, :, None]
            f = f * (1 - m) + m * (GOLD * 0.75 + 0.3)[None, None, :]
            fl = s.L0(); cv2.fillPoly(fl, [np.array([(x0, s.gnd), (x1, s.gnd), (x1 + W * 0.3 * o, H), (x0 - W * 0.3 * o, H)], np.int32)], 0.6, cv2.LINE_AA); f = f * (1 - fl[:, :, None]) + fl[:, :, None] * (GOLD * 0.8 + 0.25)[None, None, :]
        s.door(L, C, dw, dh, 0.9, 3)
        if o < 1: s.line(L, (C[0], C[1] - dh / 2), (C[0], C[1] + dh / 2), 0.8 * (1 - o), 2)
        s.ink(f, L)
        s.hero(f, t, W / 2, s.gnd + H * 0.10, 0.25, 1.0, squash=(-0.03 * math.exp(-op * 7) if op > 0 else 0))
        s.glint(f, s.at(*s.star), 9 * k, t)
        wo = clamp((op - 0.9) / 0.5) if op > 0 else 0
        if wo > 0: f = f * (1 - wo) + wo * (GOLD * 0.25 + 0.80)[None, None, :]
        if u < 0.12: f = f * (u / 0.12) + (1 - u / 0.12)
        return f, (5 * math.exp(-op * 6) if op > 0 else 0)

    # 5) ページが めくれて、色のついた 朝
    def sc_page(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_PAGE
        old = np.ones((H, W, 3), np.float32) * (GOLD * 0.25 + 0.80)[None, None, :]
        f = s.page(); col = ease2((u - 0.5) / 1.2); yn = s.YY / H; hz_ = s.gnd
        wash = np.exp(-((yn - 0.60) / 0.22) ** 2)[:, :, None] * np.array([1.0, 0.62, 0.42], np.float32)[None, None, :] + np.exp(-((yn - 0.25) / 0.25) ** 2)[:, :, None] * np.array([0.50, 0.72, 1.0], np.float32)[None, None, :]
        f = f * (1 - 0.55 * col * (yn < hz_ / H)[:, :, None]) + wash * 0.55 * col * (yn < hz_ / H)[:, :, None]
        sun = (W * 0.60, hz_); R = W * 0.24
        M = s.L0(); cv2.ellipse(M, (int(sun[0]), int(sun[1])), (int(R), int(R)), 0, 180, 360, 1.0, -1, cv2.LINE_AA); f = f * (1 - M[:, :, None] * col) + M[:, :, None] * (GOLD * 1.0)[None, None, :] * col
        L = s.L0(); s.line(L, (W * 0.04, hz_), (W * 0.96, hz_ + 2 * k), 0.9, 2); cv2.ellipse(L, (int(sun[0]), int(sun[1])), (int(R), int(R)), 0, 180, 180 + 180 * ease(u / 0.7), 0.9, 2, cv2.LINE_AA)
        for i in range(11):
            a = -3.1416 * (i + 0.5) / 11; g = ease((u - 0.3 - i * 0.04) / 0.4); s.line(L, (sun[0] + math.cos(a) * R * 1.15, sun[1] + math.sin(a) * R * 1.15), (sun[0] + math.cos(a) * R * (1.15 + 0.5 * g), sun[1] + math.sin(a) * R * (1.15 + 0.5 * g)), 0.8, 2)
        for i in range(5):                                                           # 鳥
            bx = W * (0.15 + 0.17 * i) + u * 14 * k; by = H * (0.20 + 0.05 * (i % 3)); wv = math.sin(u * 7 + i) * 4 * k
            s.line(L, (bx - 9 * k, by - wv), (bx, by + 3 * k), 0.8 * clamp(u - 0.6), 2); s.line(L, (bx, by + 3 * k), (bx + 9 * k, by - wv), 0.8 * clamp(u - 0.6), 2)
        s.ink(f, L)
        s.hero(f, t, W * 0.27, hz_ + H * 0.09, 0.30, 1.0); s.glint(f, s.at(*s.star), 11 * k, t)
        xf = W * (1.15 - 1.45 * ease2(u / 1.0))                                      # めくれる
        if xf > -80 * k:
            m = (s.XX < xf).astype(np.float32)[:, :, None]; f = f * (1 - m) + old * m
            curl = np.clip(1 - np.abs(s.XX - (xf - 26 * k)) / (26 * k), 0, 1)[:, :, None]; f = f * (1 - curl * 0.35 * (s.XX < xf)[:, :, None]) + curl * 0.35 * (s.XX < xf)[:, :, None]
            shd = np.clip(1 - (s.XX - xf) / (46 * k), 0, 1)[:, :, None] * (s.XX >= xf)[:, :, None]; f = f * (1 - shd * 0.45)
        end = DUR - t
        if end < 0.5: v = clamp((0.5 - end) / 0.45); f = f * (1 - v) + v
        return f, 0

    # ---- この子だけの曲: えんぴつの音と オルゴール(ハ長調)。カットインは タイプライター。道が のびる間は 1本ごとに オルゴールが 別の ふしを ためす。えらぶと 1本の ふしに なる ----
    def audio(s, path):
        a = Snd(DUR, 80); C5 = 523.25; maj = [0, 2, 4, 5, 7, 9, 11]
        def fq(d, o=0): return C5 * 2 ** ((maj[d % 7] + 12 * (d // 7 + o)) / 12.0)
        def box(d, t0, g=0.09, o=0): a.note(fq(d, o), t0, 0.5, g, "bell", 0.001)
        def scratch(t0, dur, g=0.25): a.a += (a.nz - a.lp(a.nz, 0.35)) * a.gate(t0, t0 + dur, 0.03) * g * (0.6 + 0.4 * np.sin(a.t * 60))
        def click(t0, g=0.3): a.a += (a.nz - a.lp(a.nz, 0.7)) * a.env(t0, 0.0005, 0.015) * g; a.boom(t0, 180, g * 0.4, 0.03)
        scratch(0.0, 0.6); scratch(0.6, 0.95, 0.3)
        for i, d in enumerate([0, 2, 4, 7]): box(d, 1.5 + i * 0.17, 0.07)
        box(11, 2.2, 0.11, 0)
        a.boom(T_CUT, 60, 0.8, 0.2); box(7, T_CUT + 0.45, 0.10, 1)
        for i in range(6): click(T_CUT + 0.6 + i * 0.14)
        a.note(2100, T_CUT + 0.6 + 6 * 0.14 + 0.05, 0.4, 0.08, "bell")                                                 # チーン
        ph = [[0, 2, 4], [0, 3, 5], [1, 4, 6], [0, 4, 7], [2, 5, 8], [3, 6, 9], [4, 7, 11]]
        for i in range(7):
            tk = T_PATH + 0.2 + i * 0.22; scratch(tk, 0.5, 0.10)
            for j, d in enumerate(ph[i]): box(d, tk + j * 0.2, 0.05 + 0.005 * i)
            box(ph[i][2] + 3, tk + 0.7, 0.05)
        a.boom(T_PICK, 70, 0.7, 0.2); box(11, T_PICK, 0.12)
        for i in range(6): scratch(T_PICK + 0.15 + i * 0.1, 0.12, 0.2)                                                 # 消しゴム
        for i, d in enumerate([4, 7, 11, 14]): box(d, T_PICK + 0.6 + i * 0.15, 0.09)
        scratch(T_DOOR, 0.6, 0.2)
        for d in (0, 4): a.note(fq(d, -2), T_DOOR, 1.2, 0.05, "pad", 0.3)
        a.rise(T_OPEN, T_OPEN + 0.9, 90, 60, 0.18); a.rise(T_OPEN, T_PAGE, 300, 1500, 0.08)                             # 扉
        for j, (d, o) in enumerate(((0, -2), (4, -2), (0, -1), (2, -1), (4, -1))): a.note(fq(d, o), T_OPEN + j * 0.08, 2.0, 0.05, "pad", 0.4)
        a.whoosh(T_PAGE, T_PAGE + 0.9, 0.5, 0.5)                                                                       # ページ
        mel = [4, 2, 0, 2, 4, 7, 9, 7, 11, 14]
        for i, d in enumerate(mel): box(d, T_PAGE + 0.5 + i * 0.19, 0.10); box(d, T_PAGE + 0.5 + i * 0.19, 0.04, -1)
        for j, (d, o) in enumerate(((0, -2), (4, -2), (0, -1), (4, -1))): a.note(fq(d, o), T_PAGE + 0.5, 2.2, 0.05, "pad", 0.4)
        s.fin_snd(a, DUR, 80); a.save(path)


if __name__ == "__main__":
    r = Mirai()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
