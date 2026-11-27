"""アーサー(★5 聖騎士) 専用の必殺技「聖剣ノ誓」
暗い大聖堂。ばら窓のステンドグラスが 1枚ずつ 灯り、色の光が 床に落ちる → 剣の刃に 目が うつる カットイン →
円卓の誓い: 足もとの輪から 光の剣が 12本、1本ずつ 立ち上がる → ばら窓の光が 剣先に 集まり、刃が 天まで のびる →
ふりおろす(止め絵の一閃) → ステンドグラスが 砕けて 色のかけらが ふりそそぐ
使い方: python3 hs_15.py 出力.mp4 / python3 hs_15.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 12.0
GOLD = np.array([1.0, 0.84, 0.42], np.float32); STONE = np.array([0.07, 0.07, 0.11], np.float32); STEEL = np.array([0.70, 0.82, 1.0], np.float32)
PAL = [(0.90, 0.16, 0.18), (0.15, 0.32, 0.92), (1.0, 0.78, 0.22), (0.18, 0.72, 0.36), (0.62, 0.25, 0.85), (0.20, 0.75, 0.90)]
T_CUT, T_OATH, T_BEAM, T_SWING, T_STOP, T_SHAT = 2.4, 4.3, 7.2, 8.55, 8.75, 9.1


class Arthur(Kit):
    def __init__(s, W=540):
        super().__init__(15, (1.0, 0.84, 0.42), "聖剣ノ誓", "Oath of Excalibur", W)
        s.eyeL = s.A(197, 378); s.eyeR = s.A(280, 380); s.tip = s.A(95, 20); s.guard = s.A(72, 292)
        s.vt = [s.text_img(c, hz.SERIF, int(84 * s.k), 3) for c in "聖剣ノ誓"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        rg = np.random.RandomState(15); s.rd = rg.rand(200, 6).astype(np.float32)
        s.gnd = s.H * 0.84; s.O = (s.W / 2, s.H * 0.25); s.R = s.W * 0.37
        # ばら窓: 輪×12 の ガラス
        idx = np.full((s.H, s.W), -1, np.int32); lead = np.zeros((s.H, s.W), np.float32); s.pan = []
        rings = [0.0, 0.22, 0.52, 0.78, 1.0]; n = 0
        for ri in range(4):
            sec = 1 if ri == 0 else 12
            for si in range(sec):
                a0 = si * 6.2832 / sec + (0.26 if ri % 2 else 0); a1 = a0 + 6.2832 / sec; r0, r1 = rings[ri] * s.R, rings[ri + 1] * s.R
                arc = np.linspace(a0, a1, 10 if sec > 1 else 40)
                pts = [(s.O[0] + math.cos(a) * r1, s.O[1] + math.sin(a) * r1) for a in arc] + ([(s.O[0] + math.cos(a) * r0, s.O[1] + math.sin(a) * r0) for a in arc[::-1]] if ri else [])
                P = np.array(pts, np.int32); cv2.fillPoly(idx, [P], n); cv2.polylines(lead, [P], True, 1.0, max(2, int(3 * s.k)), cv2.LINE_AA)
                am = (a0 + a1) / 2; rm = (r0 + r1) / 2 if ri else 0
                s.pan.append(((s.O[0] + math.cos(am) * rm, s.O[1] + math.sin(am) * rm), PAL[(si + ri * 2) % 6] if ri else (1.0, 0.92, 0.6), ri, si, am)); n += 1
        s.idx = idx; s.lead = lead; s.np_ = n; s.pcol = np.array([p[1] for p in s.pan] + [(0, 0, 0)], np.float32)
        s.order = sorted(range(n), key=lambda i: (s.pan[i][2], s.pan[i][3]))
        s.rank = np.zeros(n, np.float32)
        for q, i in enumerate(s.order): s.rank[i] = q / float(n)
        s.yn = s.YY / s.H

    # ---- この子だけの絵 ----
    def hall(s, lit, glow=1.0):
        """大聖堂: 石の柱と アーチ、ばら窓、床に落ちる 色の光。lit=ガラスごとの明るさ"""
        W, H, k = s.W, s.H, s.k
        f = np.zeros((H, W, 3), np.float32) + STONE[None, None, :] * (0.6 + 0.8 * s.yn)[:, :, None]
        L = s.L0()
        for x in (W * 0.07, W * 0.93): s.line(L, (x, 0), (x, s.gnd), 0.35, int(14 * k))
        cv2.ellipse(L, (int(W / 2), int(H * 0.30)), (int(W * 0.43), int(H * 0.30)), 0, 180, 360, 0.3, int(5 * k), cv2.LINE_AA)
        f += L[:, :, None] * (STONE * 2.2)[None, None, :]
        lv = np.concatenate([lit, [0.0]]).astype(np.float32)
        glass = (s.pcol * lv[:, None])[s.idx]
        f += glass * 0.95 * glow; f *= (1 - s.lead * 0.9)[:, :, None]
        f += cv2.GaussianBlur(glass, (0, 0), 22 * k) * 0.55 * glow
        g = int(s.gnd); f[g:, :, :] = STONE * 0.55
        fl = cv2.resize(glass[int(s.O[1] - s.R):int(s.O[1] + s.R), :], (W, H - g))[::-1]                                # 床に うつる
        f[g:, :, :] += cv2.GaussianBlur(fl, (0, 0), 7 * k) * 0.35 * glow
        return f

    def lit_at(s, p): return np.clip((p * 1.25 - s.rank) / 0.12, 0, 1)

    def hero(s, f, t, cx, cy, hfrac, deg=0.0, alpha=1.0, white=0.0, **kw):
        img = s.pose(t, **kw)
        if deg:
            M = cv2.getRotationMatrix2D((s.pw / 2, s.ph * 0.62), deg, 1.0); img = cv2.warpAffine(img, M, (s.pw, s.ph), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        s.draw(f, img, cx, cy, hfrac, alpha, white); s._deg = deg

    def pt(s, a):
        p = s.at(*a); cx, cy, sc = s._pl; cyr = cy + (s.ph * 0.62 - s.ph / 2) * sc; r = math.radians(s._deg); dx, dy = p[0] - cx, p[1] - cyr
        return (cx + math.cos(r) * dx + math.sin(r) * dy, cyr - math.sin(r) * dx + math.cos(r) * dy)

    def blade(s, f, v, ext=0.0):
        """刃が 光る。ext>0 で 切っ先の先へ のびる"""
        if v <= 0: return
        a = s.pt(s.guard); b = s.pt(s.tip); L = s.L0(); d = (b[0] - a[0], b[1] - a[1]); n = math.hypot(*d) + 1e-6
        e = (b[0] + d[0] / n * s.H * 1.2 * ext, b[1] + d[1] / n * s.H * 1.2 * ext)
        cv2.line(L, (int(a[0]), int(a[1])), (int(e[0]), int(e[1])), float(v), max(2, int((5 + 12 * ext) * s.k)), cv2.LINE_AA)
        s.add(f, L, WHITE * 0.7 + GOLD * 0.3, 0.9 + 0.6 * ext, 3, 12 + 10 * ext)

    def swords(s, f, u, front, t):
        """円卓の 12本"""
        W, H, k = s.W, s.H, s.k; L = s.L0()
        for i in range(12):
            a = -1.5708 + i * 0.5236; sn = math.sin(a)
            if (sn > 0.05) != front: continue
            g = back((u - 0.3 - i * 0.19) / 0.3, 1.8)
            if g <= 0: continue
            x = W / 2 + math.cos(a) * W * 0.41; y = s.gnd + 8 * k + sn * H * 0.055; h = H * (0.15 + 0.03 * sn) * g
            s.line(L, (x, y), (x, y - h), 1.0, 3); s.line(L, (x - 11 * k, y - h * 0.28), (x + 11 * k, y - h * 0.28), 1.0, 3)
            cv2.fillPoly(L, [np.array([(x - 3 * k, y - h), (x + 3 * k, y - h), (x, y - h - 12 * k)], np.int32)], 1.0, cv2.LINE_AA)
            s.ell(L, (x, y), (16 * k, 4 * k), 0, 0.7, 1)
        s.add(f, L, GOLD, 0.9, 2, 8)

    def frame(s, t):
        k = s.k
        if t < T_CUT: f, sh = s.sc_window(t)
        elif t < T_OATH: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_STOP: f, sh = s.sc_oath(t)
        else: f, sh = s.sc_shatter(t)
        return s.post(f, GOLD, 0.45, sh * k, t, 0.95)

    # 1) ばら窓が 灯る
    def sc_window(s, t):
        W, H, k = s.W, s.H, s.k
        f = s.hall(s.lit_at(ease2(t / 1.9)))
        cy = s.gnd - H * 0.20
        s.hero(f, t, W / 2, cy, 0.42, 0, 1.0, 0.0)
        dark = 1 - 0.75 * (1 - ease2((t - 0.6) / 1.4)); sc = s._pl[2]                # はじめは 影。光が 入ると 色が のる
        if dark < 1:
            x0, y0 = int(W / 2 - s.pw * sc / 2), int(cy - s.ph * sc / 2); f[max(0, y0):int(cy + s.ph * sc / 2), max(0, x0):int(W / 2 + s.pw * sc / 2)] *= (0.25 + 0.75 * dark)
        s.blade(f, ease((t - 1.9) / 0.4) * 0.7)
        return f * clamp(t / 0.4), 0

    # 2) 刃に 目が うつる
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.hall(np.ones(s.np_, np.float32), 0.35) * 0.5
        g = ease(u / 0.28); bw = W * 0.15
        top = (W * 0.36, -20); bot = (W * 0.26, H * 0.80 * g)
        quad = [(top[0] - bw, top[1]), (top[0] + bw, top[1]), (bot[0] + bw, bot[1]), (bot[0], bot[1] + bw * 1.6), (bot[0] - bw, bot[1])]
        m = s.L0(); cv2.fillPoly(m, [np.array(quad, np.int32)], 1.0, cv2.LINE_AA)
        win = np.zeros((H, W, 3), np.float32) + (STEEL * 0.30)[None, None, :] * (0.7 + 0.4 * np.abs(s.XX - W * 0.31) / bw)[:, :, None]
        ex = (s.eyeL[0] + s.eyeR[0]) / 2; zf = 1.9; sc = zf * H / s.ah
        img = s.pose(t); s.draw(win, img, W * 0.31 - (ex - s.aw / 2) * sc, H * 0.34 - (s.eyeL[1] - s.ah / 2) * sc, zf, 1.0)
        win = win * np.array([0.72, 0.82, 1.0], np.float32)[None, None, :]
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        L = s.L0(); cv2.polylines(L, [np.array(quad, np.int32)], True, 1.0, 3, cv2.LINE_AA); s.line(L, top, (bot[0], bot[1] + bw * 1.6), 0.5, 1)
        gl = (u * 1.3) % 1.6; s.line(L, (top[0] - bw + (bot[0] - top[0]) * gl, H * 0.8 * gl - 30 * k), (top[0] + bw + (bot[0] - top[0]) * gl, H * 0.8 * gl + 30 * k), 0.5, int(3 * k))     # 刃を 走る 光
        s.add(f, L, WHITE * 0.8 + GOLD * 0.2, 0.30, 1.5, 4)
        for i, T in enumerate(s.vt):
            a = (u - 0.5 - i * 0.16) / 0.14
            if a > 0:
                p = (W * 0.74, H * 0.26 + i * 96 * k)
                if a < 1: T2 = tuple(cv2.resize(q, None, fx=1 + (1 - a) * 1.2, fy=1 + (1 - a) * 1.2) for q in T); s.text(f, T2, p[0], p[1], a, 1.0)
                else: s.text(f, T, p[0], p[1], 1.0, 1.0)
        for i, T in enumerate(s.vo):
            if u > 0.35 + i * 0.1: s.text(f, T, W * 0.74, H * 0.11 + i * 36 * k, 1.0, 1.0)
        if u > 1.4: s.text(f, s.txt_en, W * 0.66, H * 0.90, ease((u - 1.4) / 0.3), 1.0)
        if u < 0.08: f += (1 - u / 0.08) * 0.8
        return f, 0

    # 3) 円卓の誓い → 光が 集まる → ふりおろす
    def sc_oath(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_OATH; bm = t - T_BEAM; sw = t - T_SWING
        f = s.hall(np.ones(s.np_, np.float32), 1.0 + (0.5 * ease(bm / 1.0) if bm > 0 else 0))
        L = s.L0(); s.ell(L, (W / 2, s.gnd + 8 * k), (W * 0.41, H * 0.055), 0, 0.7 * ease(u / 0.5), 2); s.add(f, L, GOLD, 0.8)
        s.swords(f, u, False, t)
        cy = s.gnd - H * 0.20
        deg = 0.0
        if bm > 0 and sw < 0: deg = -10 * ease(bm / 1.0)                             # ため(うしろへ そる)
        if sw > 0: deg = -10 + 46 * ease2(sw / (T_STOP - T_SWING))
        if sw > 0:
            for j in (2, 1): s.hero(f, t, W / 2, cy, 0.42, deg - j * 16, 0.22, 0.7)
        s.hero(f, t, W / 2, cy, 0.42, deg, squash=(-0.04 * ease(bm / 1.0) if bm > 0 and sw < 0 else 0), jit=(1.2 * k * clamp((bm - 0.6) / 0.6) if bm > 0 and sw < 0 else 0))
        ext = ease2((bm - 0.35) / 0.8) if bm > 0 else 0.0
        if bm > 0 and sw < 0:                                                        # ばら窓の光が 剣先へ
            tp = s.pt(s.tip); L = s.L0(); q = ease(bm / 0.35)
            for i in range(0, 12, 2):
                a = i * 0.5236 + 0.26; p0 = (s.O[0] + math.cos(a) * s.R * 0.6, s.O[1] + math.sin(a) * s.R * 0.6); s.line(L, p0, (p0[0] + (tp[0] - p0[0]) * q, p0[1] + (tp[1] - p0[1]) * q), 0.5, 2)
            s.add(f, L, WHITE * 0.6 + GOLD * 0.4, 0.9)
            if bm < 0.08: f += (1 - bm / 0.08) * 0.4
        s.blade(f, 0.7 + 0.3 * clamp(ext), ext)
        s.swords(f, u, True, t)
        if u < 0.12: f += (1 - u / 0.12) * 0.6
        return f, (4 * clamp((bm - 0.6) / 0.6) if bm > 0 and sw < 0 else 0)

    # 4) 止め絵の一閃 → ステンドグラスが 砕ける
    def sc_shatter(s, t):
        W, H, k = s.W, s.H, s.k; st = t - T_STOP; sh = t - T_SHAT
        cy = s.gnd - H * 0.20
        if sh < 0:                                                                   # 止め絵: 金の地に 白い 一閃
            f = np.ones((H, W, 3), np.float32) * (GOLD * 0.75)[None, None, :]
            L = s.L0(); a = (W * 0.88, -10); b = (W * 0.10, H + 10); cv2.line(L, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 1.0, int(16 * k), cv2.LINE_AA)
            for i in range(24): r = s.rd[i]; x = a[0] + (b[0] - a[0]) * r[0]; y = a[1] + (b[1] - a[1]) * r[0]; s.line(L, (x, y), (x + (r[1] - 0.5) * 260 * k, y + (r[2] - 0.5) * 120 * k), 1.0, 1)
            f = f * (1 - np.clip(L, 0, 1)[:, :, None]) + np.clip(L, 0, 1)[:, :, None]
            s.hero(f, t, W / 2, cy, 0.42, 36, 1.0, 0.0); sc = s._pl[2]
            return f, 0
        f = s.hall(np.zeros(s.np_, np.float32), 1.0)
        f += np.exp(-(((s.XX - s.O[0]) ** 2 + (s.YY - s.O[1]) ** 2) / (s.R * 1.2) ** 2))[:, :, None] * (WHITE * 0.8 + GOLD * 0.2)[None, None, :] * (0.55 + 0.8 * math.exp(-sh * 2))     # 窓の むこうの 光
        L = s.L0()
        for i in range(12): a = i * 0.5236 + sh * 0.2; cv2.fillPoly(L, [np.array([s.O, (s.O[0] + math.cos(a - 0.06) * W * 1.6, s.O[1] + math.sin(a - 0.06) * W * 1.6), (s.O[0] + math.cos(a + 0.06) * W * 1.6, s.O[1] + math.sin(a + 0.06) * W * 1.6)], np.int32)], 0.22, cv2.LINE_AA)
        f += cv2.GaussianBlur(L, (0, 0), 6 * k)[:, :, None] * GOLD[None, None, :] * clamp(1.4 - sh * 0.3)
        for i, (c, col, ri, si, am) in enumerate(s.pan):                             # かけら
            r = s.rd[i]; v = 0.6 + 1.4 * r[0]; x = c[0] + math.cos(am) * W * 0.35 * v * sh + (r[1] - 0.5) * 40 * k * sh; y = c[1] + math.sin(am) * W * 0.18 * v * sh + H * 0.22 * sh * sh * (0.6 + r[2])
            if y > H + 40: continue
            sz = (14 + 16 * r[3] + 6 * ri) * k; b = r[4] * 6.28 + sh * 5 * (r[5] - 0.5)
            P = np.array([(x + math.cos(b + j * 2.094 + (0.5 if j == 1 else 0)) * sz * (1.4 if j == 0 else 1), y + math.sin(b + j * 2.094) * sz) for j in range(3)], np.int32)
            cv2.fillPoly(f, [P], tuple(float(q) for q in np.array(col) * (0.8 + 0.5 * abs(math.sin(b * 2)))), cv2.LINE_AA)
        deg = 36 * (1 - ease((sh - 0.5) / 0.7)) if sh > 0.5 else 36
        s.hero(f, t, W / 2, cy, 0.42, deg); s.blade(f, 0.6, 0.0)
        if sh < 0.14: f = f + (1 - sh / 0.14) * 1.2
        end = DUR - t
        if end < 0.8: v = clamp((0.8 - end) / 0.4); f = f * (1 - v) + v * (GOLD * 0.3 + 0.7)[None, None, :]
        if end < 0.4: f = f * clamp(end / 0.35) ** 2
        return f, 22 * math.exp(-sh * 4)

    # ---- この子だけの曲: パイプオルガンと 聖歌(ニ長調)。光の剣が 立つたび 鐘が 1音ずつ のぼる。止め絵は 無音、砕ける音のあとに 大きな 終止の和音 ----
    def audio(s, path):
        a = Snd(DUR, 15); D = 146.83; maj = [0, 2, 4, 5, 7, 9, 11]
        def fq(d, o=0): return D * 2 ** ((maj[d % 7] + 12 * (d // 7 + o)) / 12.0)
        def organ(ds, t0, dur, g=0.035):
            for d, o in ds: a.note(fq(d, o), t0, dur, g, "sq", 0.06); a.note(fq(d, o) * 2, t0, dur, g * 0.4, "sine", 0.06)
        organ([(0, -1), (0, 0), (4, 0)], 0.1, 1.2); organ([(3, -1), (0, 0), (5, 0)], 1.0, 1.0); organ([(4, -1), (1, 0), (4, 0), (6, 0)], 1.7, 0.9)
        for i in range(12): a.note(fq(i % 7 + 7, 1), 0.2 + i * 0.14, 0.2, 0.025, "bell")                               # ガラスが 灯る
        a.boom(T_CUT, 50, 1.0); a.rise(T_CUT, T_CUT + 0.3, 800, 2400, 0.08)
        for i in range(4): a.boom(T_CUT + 0.5 + i * 0.16, 64, 0.5, 0.15); a.note(fq([0, 4, 7, 11][i], 1), T_CUT + 0.5 + i * 0.16, 0.5, 0.10, "bell")
        organ([(0, -1), (4, -1), (0, 0), (2, 0)], T_CUT + 0.5, 1.6)
        sc12 = [0, 1, 2, 4, 3, 4, 5, 7, 6, 7, 8, 9]
        for i in range(12): a.note(fq(sc12[i], 1), T_OATH + 0.3 + i * 0.19, 0.6, 0.10, "bell"); a.note(fq(sc12[i], 0), T_OATH + 0.3 + i * 0.19, 0.3, 0.04, "fm")      # 12本
        organ([(0, -1), (0, 0)], T_OATH, 1.4, 0.03); organ([(3, -1), (5, 0)], T_OATH + 1.3, 1.3, 0.03)
        for d, o in ((0, 0), (2, 0), (4, 0), (7, 0)): a.note(fq(d, o), T_OATH + 0.2, 2.6, 0.03, "pad", 0.6)             # 聖歌
        organ([(4, -2), (4, -1), (1, 0), (4, 0), (6, 0)], T_BEAM, 1.3, 0.045); a.rise(T_BEAM, T_SWING, 200, 1600, 0.14)
        for i in range(10): a.hat(T_BEAM + 0.5 + i * 0.08, 0.04 + i * 0.006)
        a.whoosh(T_SWING, T_STOP, 0.8, 0.5); a.note(fq(14, 2), T_STOP, 0.3, 0.08, "sine")
        a.boom(T_SHAT, 38, 1.6, 0.8); a.a += (a.nz - a.lp(a.nz, 0.6)) * a.env(T_SHAT, 0.001, 0.5) * 0.5
        for i in range(30): a.note(1500 + (i * 331) % 2400, T_SHAT + 0.03 * i + 0.02 * (i % 3), 0.12, 0.035, "bell")   # ガラス
        organ([(0, -2), (0, -1), (4, -1), (0, 0), (2, 0), (4, 0), (7, 0)], T_SHAT + 0.15, 2.4, 0.045)
        for i, d in enumerate([7, 9, 11, 14]): a.note(fq(d, 0), T_SHAT + 0.6 + i * 0.3, 0.8, 0.09, "bell")
        a.save(path)


if __name__ == "__main__":
    r = Arthur()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
