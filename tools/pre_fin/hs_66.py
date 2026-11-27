"""ラグナ(★5 終焉の竜騎士) 専用の必殺技「終焉ノ竜剣」(絵は 大剣なので 槍→剣 に直した)
赤むらさきの 夕空と くずれた塔の 街なみ。右上から ななめに とびこんで 宙で 止まる(翼が ばさっと 1回) → 剣先に 黒い点。
景色が 剣先へ 引きのばされて 曲がりはじめ、がれきが うずを巻いて 吸われる → 剣の形の 窓のカットイン →
黒い穴が ふくらみ 街も 空も のみこむ → まっ暗(目と 刃だけ 光る・無音) → 白い線が 1本 走って 上下に ひらく → 何もない 朝の地平
使い方: python3 hs_66.py 出力.mp4 / python3 hs_66.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 11.5
VIO = np.array([0.66, 0.34, 1.0], np.float32); CRIM = np.array([1.0, 0.22, 0.30], np.float32); PALE = np.array([0.90, 0.84, 1.0], np.float32)
T_STOP, T_DOT, T_CUT, T_GROW, T_EAT, T_BLK, T_LINE, T_OPEN, T_END = 0.8, 1.9, 3.7, 5.6, 7.6, 7.95, 8.4, 8.5, 8.85


class Ragna(Kit):
    def __init__(s, W=540):
        super().__init__(66, (0.66, 0.34, 1.0), "終焉ノ竜剣", "Ragnarok Blade", W)
        H, k = s.H, s.k
        s.eL = s.A(345, 295); s.eR = s.A(458, 268); s.fc = s.A(400, 290); s.hlt = s.A(215, 520); s.tip = s.A(32, 722)
        s.ht = [s.text_img(c, hz.SERIF, int(78 * k), 3) for c in "終焉ノ竜剣"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * k), 1) for c in "奥義"]
        s.HF = 0.42; s.CX = W * 0.60; s.CY = H * 0.40
        s._pose = dict(t=0, sway=0, breath=0, lean=0, squash=0, stretch=None); s._pl = (s.CX, s.CY, s.HF * H / s.ah); s.HC = s.at(*s.tip)       # 穴の 中心 = 剣先
        rg = np.random.RandomState(66); s.yn = s.YY / H; s.gnd = H * 0.93
        m = s.L0(); x = -10 * k                                                       # くずれた塔の 街なみ
        while x < W + 10 * k:
            w = (22 + 46 * rg.rand()) * k; h = H * (0.08 + 0.30 * rg.rand() ** 1.6); top = s.gnd - h; br = rg.rand()
            P = [(x, s.gnd), (x, top + h * 0.12 * br), (x + w * 0.3, top), (x + w * 0.45, top + h * 0.10), (x + w * 0.7, top + h * 0.03 * br), (x + w, top + h * 0.2 * (1 - br)), (x + w, s.gnd)]
            cv2.fillPoly(m, [np.array(P, np.int32)], 1.0, cv2.LINE_AA)
            if rg.rand() < 0.35: cv2.fillPoly(m, [np.array([(x + w * 0.4, top), (x + w * 0.5, top - h * 0.35), (x + w * 0.6, top)], np.int32)], 1.0, cv2.LINE_AA)
            x += w * (0.75 + 0.5 * rg.rand())
        cv2.rectangle(m, (0, int(s.gnd)), (W, H), 1.0, -1); s.city = m
        win = s.L0()                                                                 # 塔の まど(小さな 灯)
        for i in range(60):
            x, y = int(rg.rand() * W), int(s.gnd - rg.rand() * H * 0.3)
            if m[y, x] > 0.5 and m[max(0, y - int(8 * k)), x] > 0.5: cv2.rectangle(win, (x, y), (x + int(3 * k), y + int(5 * k)), 1.0, -1)
        s.lamp = win
        s.rk = rg.rand(16, 6).astype(np.float32); s.mo = rg.rand(40, 4).astype(np.float32)
        s.dx = s.XX - s.HC[0]; s.dy = s.YY - s.HC[1]; s.rr = np.sqrt(s.dx ** 2 + s.dy ** 2) + 1e-3

    # ---- 大道具 ----
    def world(s, t, suck=0.0, drain=0.0):
        """夕空・街・がれき(ゆがめる前の 絵)。suck: がれきが 穴へ よる割合"""
        W, H, k = s.W, s.H, s.k
        top = np.array([0.16, 0.05, 0.22], np.float32); low = np.array([0.95, 0.30, 0.22], np.float32)
        q = np.clip(s.yn * 1.15, 0, 1) ** 1.6; f = (top[None, None, :] * (1 - q)[:, :, None] + low[None, None, :] * q[:, :, None]).astype(np.float32)
        for j, (y, w) in enumerate(((0.20, 0.9), (0.34, 0.6), (0.50, 1.1))):           # 横に たなびく 雲の すじ
            cv2.ellipse(f, (int(W * (0.5 + 0.3 * math.sin(j * 2.1 + t * 0.05))), int(H * y)), (int(W * w * 0.5), int(7 * k)), 0, 0, 360, (0.50, 0.14, 0.26), -1, cv2.LINE_AA)
        f = f * (1 - s.city[:, :, None]) + np.array([0.05, 0.02, 0.07], np.float32)[None, None, :] * s.city[:, :, None]
        f += (s.lamp * 0.8)[:, :, None] * np.array([1.0, 0.75, 0.35], np.float32)[None, None, :]
        for r in s.rk:                                                               # 宙の がれき
            x0, y0 = r[0] * W, (0.12 + 0.6 * r[1]) * H; g = clamp(suck * (1.2 + r[2]) - r[3] * 0.5)
            a = math.atan2(y0 - s.HC[1], x0 - s.HC[0]) + g * 3.5; d = math.hypot(x0 - s.HC[0], y0 - s.HC[1]) * (1 - g)
            x, y = s.HC[0] + math.cos(a) * d, s.HC[1] + math.sin(a) * d; sz = (14 + 22 * r[4]) * k * (1 - 0.7 * g); ro = r[5] * 6 + t * (r[4] - 0.5) * 0.6 + g * 5
            P = [(x + math.cos(ro + j * 1.2566) * sz * (0.6 + 0.4 * ((j * 3 + int(r[5] * 9)) % 3) / 2.0), y + math.sin(ro + j * 1.2566) * sz * (0.6 + 0.4 * ((j * 3 + int(r[5] * 9)) % 3) / 2.0)) for j in range(5)]
            if g < 0.98: cv2.fillPoly(f, [np.array(P, np.int32)], (0.09, 0.04, 0.12), cv2.LINE_AA)
        if drain > 0: gy = f.mean(2, keepdims=True); f = f * (1 - drain) + gy * (VIO * 0.9 + 0.1)[None, None, :] * drain
        return f

    def warp(s, f, pull, swirl, R):
        """穴へ 引きのばして 曲げる"""
        if pull <= 0.001 and swirl <= 0.001: return f
        sc = 1 + pull * (R / s.rr) ** 2; a = swirl * np.exp(-s.rr / (R * 2.2)); ca, sa = np.cos(a), np.sin(a)
        mx = s.HC[0] + (s.dx * ca - s.dy * sa) * sc; my = s.HC[1] + (s.dx * sa + s.dy * ca) * sc
        return cv2.remap(f, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    def hole(s, f, rb, t, v=1.0):
        if rb < 1: return
        c = s.HC; k = s.k
        s.soft(f, c, rb * 1.7, 0.55 * v, VIO)
        L = s.L0(); cv2.ellipse(L, (int(c[0]), int(c[1])), (int(rb * 2.1), int(rb * 0.42)), -22, 0, 360, 0.8 * v, max(1, int(rb * 0.07)), cv2.LINE_AA)       # まわる 光の輪
        for j in range(3): a0 = t * 200 + j * 120; cv2.ellipse(L, (int(c[0]), int(c[1])), (int(rb * 1.25), int(rb * 1.25)), 0, a0, a0 + 70, 0.9 * v, max(1, int(rb * 0.06)), cv2.LINE_AA)
        s.add(f, L, VIO * 0.6 + CRIM * 0.4, 0.8, 2, 7)
        cv2.circle(f, (int(c[0]), int(c[1])), int(rb), (0.0, 0.0, 0.0), -1, cv2.LINE_AA)
        L = s.L0(); cv2.circle(L, (int(c[0]), int(c[1])), int(rb), 1.0 * v, max(1, int(2.5 * k)), cv2.LINE_AA); s.add(f, L, PALE, 0.7, 1, 5)

    def hero(s, f, t, cx=None, cy=None, **kw):
        img = s.pose(t, **kw); s.draw(f, img, s.CX if cx is None else cx, s.CY if cy is None else cy, s.HF)

    def eyes(s, f, v):
        if v <= 0.01: return
        sc = s._pl[2]; L = s.L0()
        for e in (s.eL, s.eR): p = s.at(*e); cv2.circle(L, (int(p[0]), int(p[1])), max(1, int(11 * sc)), float(min(1.0, v)), -1, cv2.LINE_AA)
        s.add(f, L, CRIM, 0.8, 2, 7)

    def edge(s, f, v):
        if v <= 0.01: return
        a = s.at(*s.hlt); b = s.at(*s.tip); L = s.L0(); s.line(L, a, b, min(1.0, v), max(1, int(3 * s.k))); s.add(f, L, PALE, 0.8 * min(1.5, v), 2, 7)

    def frame(s, t):
        if t < T_CUT: f, sh = s.sc_open(t)
        elif t < T_GROW: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_END: f, sh = s.sc_eat(t)
        else: f, sh = s.sc_end(t - T_END)
        return s.post(f, VIO * 0.6 + 0.4, 0.35, sh * s.k, t, 0.9)

    # 1) とびこんで 止まる → 剣先に 黒い点 → 景色が 曲がりはじめる
    def sc_open(s, t):
        W, H, k = s.W, s.H, s.k; d = t - T_DOT; st = t - T_STOP
        g = ease2(d / (T_CUT - T_DOT)) if d > 0 else 0.0
        f = s.warp(s.world(t, 0.55 * g), 0.9 * g, 1.3 * g, 60 * k)
        if d > 0: s.hole(f, (4 + 26 * ease(d / 1.2)) * k, t)
        dv = np.array([s.tip[0] - s.hlt[0], s.tip[1] - s.hlt[1]], np.float32); dv /= np.linalg.norm(dv)       # 剣の向き(左下)へ とびこむ
        off = (1 - ease(t / T_STOP)) * H * 0.75
        fl = math.exp(-st * 7) * (st > 0)
        s.hero(f, t, s.CX - dv[0] * off, s.CY - dv[1] * off, squash=-0.05 * fl, stretch=(s.aw * 0.5, s.aw * 0.18, s.aw * 0.5, s.ah * 0.75, 0.16 * fl * math.cos(st * 14)))
        if t < T_STOP:                                                               # とびこむ すじ
            L = s.L0(); c = (s.CX - dv[0] * off, s.CY - dv[1] * off)
            for i in range(5): o = (i - 2) * 30 * k; s.line(L, (c[0] - dv[0] * 120 * k + dv[1] * o, c[1] - dv[1] * 120 * k - dv[0] * o), (c[0] - dv[0] * (300 + 60 * (i % 2)) * k + dv[1] * o, c[1] - dv[1] * (300 + 60 * (i % 2)) * k - dv[0] * o), 0.5, 2)
            s.add(f, L, VIO, 0.6)
        if d > 0: s.edge(f, 0.4 + 0.8 * math.exp(-d * 5)); s.eyes(f, clamp(d / 0.4))
        if 0 < st < 0.06: f = f + (1 - st / 0.06) * 0.3
        return f * clamp(t / 0.3), 6 * fl

    # 2) 剣の形の 窓
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = np.zeros((H, W, 3), np.float32) + np.array([0.05, 0.015, 0.08], np.float32)
        C = (W / 2, H * 0.36); L = s.L0()
        for j in range(14): a = j * 0.449 + 0.2; s.line(L, (C[0] + math.cos(a) * W * 0.3, C[1] + math.sin(a) * W * 0.3), (C[0] + math.cos(a + 0.12 * math.sin(j * 5)) * H, C[1] + math.sin(a + 0.12 * math.sin(j * 5)) * H), 0.25, 1)
        f += L[:, :, None] * VIO[None, None, :]
        g = back(u / 0.3, 1.2); w = W * 0.37; top = H * 0.07; sh_y = top + (H * 0.50 - top) * g; tipy = top + (H * 0.685 - top) * g
        P = [(C[0] - w, top), (C[0] + w, top), (C[0] + w, sh_y), (C[0], tipy), (C[0] - w, sh_y)]
        m = s.L0(); cv2.fillPoly(m, [np.array(P, np.int32)], 1.0, cv2.LINE_AA)
        win = np.zeros((H, W, 3), np.float32) + np.array([0.30, 0.08, 0.16], np.float32) + (np.clip(s.yn * 1.6, 0, 1) * 0.5)[:, :, None] * np.array([0.9, 0.3, 0.2], np.float32)[None, None, :]
        zf = 0.98 + 0.05 * ease(u / 1.8); sc = zf * H / s.ah; img = s.pose(t)
        s.draw(win, img, C[0] - (s.fc[0] - s.aw / 2) * sc, H * 0.30 - (s.fc[1] - s.ah / 2) * sc, zf); s.eyes(win, 0.9 + 1.2 * math.exp(-max(0.0, u - 0.45) * 5) * (u > 0.45))
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        cv2.polylines(f, [np.array(P, np.int32)], True, (0.0, 0.0, 0.0), max(1, int(18 * k)), cv2.LINE_AA); cv2.polylines(f, [np.array(P, np.int32)], True, (0.95, 0.93, 1.0), max(1, int(9 * k)), cv2.LINE_AA)
        sw = (u - 0.45) / 0.3                                                        # 刃を 光が すべる
        if 0 < sw < 1:
            L = s.L0(); y = top + (tipy - top) * sw; s.line(L, (C[0] - w, y), (C[0] + w, y - 40 * k), 1 - sw, max(1, int(6 * k))); f += (L * m)[:, :, None] * 0.8
        n = len(s.ht); step = W * 0.175
        for i, T in enumerate(s.ht):                                                 # 技名: 吸いよせられるように 外から 集まる
            a = (u - 0.55 - i * 0.07) / 0.3
            if a > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step * (1 + 0.15 * (1 - ease(a))), H * 0.835, clamp(a * 1.5), 1.0)
        for i, T in enumerate(s.vo):
            if u > 0.4 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.758, 1.0, 1.0)
        if u > 1.3: s.text(f, s.txt_en, W / 2, H * 0.835 + 60 * k, ease((u - 1.3) / 0.3), 1.0)
        if u < 0.08: f = f + (1 - u / 0.08) * 0.7
        return f, 0

    # 3) のみこむ → まっ暗 → 白い線 → ひらく
    def sc_eat(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_GROW; e = t - T_EAT; b = t - T_BLK; ln = t - T_LINE; op = t - T_OPEN
        if b < 0:
            g = ease2(u / (T_EAT - T_GROW)); rb = (30 + 120 * g) * k
            if e > 0: rb = rb + H * 1.3 * ease2(e / (T_BLK - T_EAT)) ** 1.5
            f = s.warp(s.world(t, 0.55 + 0.6 * g, 0.7 * g), 0.9 + 2.6 * g, 1.3 + 2.4 * g, (60 + 70 * g) * k)
            s.hole(f, rb, t, 1.0 if e < 0 else clamp(1 - e / 0.2))
            s.soft(f, (s.CX, s.CY), W * 0.30, 0.25 * g, VIO)
            s.hero(f, t, jit=0.8 * k * g); s.eyes(f, 1.0); s.edge(f, 0.5 + 0.5 * g)
            if u < 0.1: f = f * (u / 0.1) + (1 - u / 0.1) * 0.5
            return f, 2 + 5 * g
        f = np.zeros((H, W, 3), np.float32)                                          # まっ暗: 目と 刃だけ
        s._pose = dict(t=t, sway=0, breath=0, lean=0, squash=0, stretch=None); s._pl = (s.CX, s.CY, s.HF * H / s.ah)
        s.eyes(f, 1.0); s.edge(f, 1.0 + 0.6 * math.sin(clamp(b / 0.4) * 3.1416))
        for e in (s.eL, s.eR): s.soft(f, s.at(*e), 16 * k, 0.9, CRIM)
        if ln > 0:
            y = H * 0.56; hw = W * 0.6 * ease(ln / 0.1); hh = 1.5 * k + (H * 0.6 * ease2(op / (T_END - T_OPEN)) ** 1.6 if op > 0 else 0)
            m = s.L0(); cv2.rectangle(m, (int(W / 2 - hw), int(y - hh)), (int(W / 2 + hw), int(y + hh)), 1.0, -1)
            f = f * (1 - m[:, :, None]) + m[:, :, None]; f += cv2.GaussianBlur(m, (0, 0), 10 * k)[:, :, None] * PALE[None, None, :] * 0.6
        return f, 0

    # 4) 何もない 朝の地平
    def sc_end(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_END
        top = np.array([0.62, 0.60, 0.80], np.float32); low = np.array([1.0, 0.92, 0.86], np.float32)
        f = (top[None, None, :] * (1 - s.yn)[:, :, None] + low[None, None, :] * s.yn[:, :, None]).astype(np.float32)
        g = int(s.gnd); f[g:] = np.array([0.50, 0.46, 0.60], np.float32)
        cv2.line(f, (0, g), (W, g), (1.0, 0.98, 0.95), max(1, int(2 * k)))
        L = s.L0()
        for r in s.mo:
            x = (r[0] + 0.02 * math.sin(t + r[2] * 9)) * W; y = ((r[1] - u * 0.03 * (0.5 + r[3])) % 1.0) * H; cv2.circle(L, (int(x), int(y)), max(1, int((1.5 + 2 * r[3]) * k)), 0.8, -1, cv2.LINE_AA)
        f += cv2.GaussianBlur(L, (0, 0), 1.5 * k)[:, :, None] * 0.5
        s.hero(f, t)
        gl = u - 1.2
        if 0 < gl < 0.6: Ls = s.L0(); p = s.at(*s.tip); s.star4(Ls, p, 60 * k * math.sin(gl / 0.6 * 3.1416), 1.0, 0.3, 0.10); s.add(f, Ls, WHITE, 0.8, 2, 6)
        if u < 0.5: f = f * ease(u / 0.5) + (1 - ease(u / 0.5))
        end = DUR - t
        if end < 0.45: v = clamp((0.45 - end) / 0.4); f = f * (1 - v) + v
        return f, 0

    # ---- この子だけの曲: 低い うなりと 三全音の 重い金管。穴は 下がりつづける 音。カットインは ゆがんだ 速い ふし。のみこむ間は 上がる うなり → ぷつっと 無音 → 心ぞうの音 2回 → 細い 高い音 → 明るい 5度の 和音。さいごは 高い 鈴の音 ひとつ ----
    def audio(s, path):
        a = Snd(DUR, 66); C = 65.41
        def fq(semi): return C * 2 ** (semi / 12.0)
        def brass(semi, t0, ln, g=0.08):
            b = Snd(DUR, 1); b.note(fq(semi), t0, ln, 1.0, "saw", 0.04); b.note(fq(semi) * 1.006, t0, ln, 0.7, "saw", 0.04); a.a += a.lp(b.a.astype(np.float32), 0.09) * g * 2.5
        def fall(t0, t1, f0, f1, g=0.10):
            u = np.clip((a.t - t0) / (t1 - t0), 0, 1); fr = f0 * (f1 / f0) ** u; ph = 2 * np.pi * np.cumsum(fr) / a.sr
            a.a += (np.sin(ph) + 0.4 * np.sin(2 * ph)) * np.sin(u * np.pi) ** 0.5 * ((a.t >= t0) & (a.t < t1)) * g
        cut = a.gate(0, T_BLK, 0.02)
        a.whoosh(0.0, T_STOP, 0.5, 0.25); a.boom(T_STOP, 55, 1.0, 0.3); a.a += a.lp(a.nz, 0.3) * a.env(T_STOP + 0.05, 0.02, 0.15) * 0.25    # とびこむ・翼
        a.a += a.tone(32.7) * 0.10 * a.gate(0, T_BLK, 0.3)
        brass(0, 1.0, 0.8, 0.07); brass(6, 1.0, 0.8, 0.05)
        a.note(fq(48), T_DOT, 0.3, 0.05, "sine"); fall(T_DOT, T_CUT, 500, 70, 0.09); a.a += a.lp(a.nz, 0.05) * a.gate(T_DOT, T_CUT, 0.4) * 0.5
        a.boom(T_CUT, 48, 1.0, 0.35)
        riff = (12, 13, 12, 18, 12, 13, 15, 12, 18, 19, 18, 13)
        for i, sm in enumerate(riff): t0 = T_CUT + 0.25 + i * 0.12; a.note(fq(sm + 12), t0, 0.07, 0.07, "sq", 0.002); a.note(fq(sm), t0, 0.07, 0.06, "saw", 0.002)
        for dt in (0.25, 0.61, 0.97, 1.33): a.boom(T_CUT + dt, 60, 0.7, 0.16); a.hat(T_CUT + dt + 0.18, 0.12, 0.03)
        brass(0, T_CUT + 0.25, 1.4, 0.06); brass(6, T_CUT + 0.97, 0.7, 0.05)
        seg = (a.lp(a.nz, 0.08) * 0.8 * np.clip((a.t - T_GROW) / (T_BLK - T_GROW), 0, 1) ** 1.5) * a.gate(T_GROW, T_BLK, 0.02); a.a += seg
        a.rise(T_GROW, T_BLK, 40, 220, 0.14); a.rise(T_EAT, T_BLK, 300, 2400, 0.10); fall(T_GROW, T_EAT, 300, 50, 0.08)
        for i in range(10): brass((0, 6)[i % 2] + 12 * (i // 4), T_GROW + 0.1 + i * 0.2, 0.18, 0.05)
        tk = T_GROW; i = 0
        while tk < T_BLK - 0.05: a.boom(tk, 70, 0.4 + 0.5 * (tk - T_GROW) / 2.3, 0.08); tk += max(0.06, 0.3 - i * 0.02); i += 1
        a.a *= (1 - a.gate(T_BLK, T_LINE - 0.02, 0.01))                                                                          # ぷつっと 無音
        a.boom(T_BLK + 0.12, 46, 0.7, 0.10); a.boom(T_BLK + 0.30, 40, 0.5, 0.12)                                                 # 心ぞう
        a.note(1760, T_LINE, 0.5, 0.08, "sine", 0.002); a.note(2640, T_LINE + 0.02, 0.4, 0.04, "sine", 0.002)
        a.whoosh(T_OPEN, T_END, 0.4, 0.5); a.boom(T_END - 0.05, 50, 1.2, 0.6)
        G = 196.0
        for sm, g in ((0, 0.05), (7, 0.045), (12, 0.04), (19, 0.03)): a.note(G * 2 ** (sm / 12.0), T_END - 0.05, 2.0, g, "pad", 0.3)
        for i, sm in enumerate((24, 19, 24, 31)): a.note(G * 2 ** (sm / 12.0), T_END + 0.5 + i * 0.35, 0.8, 0.05, "bell")
        a.save(path)


if __name__ == "__main__":
    r = Ragna()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
