"""アマツ(★6 雷の帝) 専用の必殺技「天裂ク雷帝」— 高品質版(hs_fx)・通し直し。目の前の 空そのものが 斜めに 裂け、むこうの 雷が こちらへ あふれる
流れる 黒雲。遠雷の たびに、雲の 奥で 斜めの 光の すじが かすかに 走る(前ふり) → 大きな 落雷の 中から アマツが 現れる → 稲妻の形に 裂けた 窓のカットイン →
剣を 天へ。うずを巻く 雲から 雷が 剣に 4度 落ち、剣が 白く 焼ける → こちらへ 向かって 一閃。斜めの 光の 線が 画面を わたって、ひと呼吸 →
雲も アマツも いっしょに、目の前の 景色が 斜めに 裂けて ずれる。すき間の むこうは 金色の 光。そこから 雷が こちらへ のび、すき間が ひろがって 目の前を おおう → 白 → 暗転
使い方: python3 hs_72.py 出力.mp4 / python3 hs_72.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old72", "/home/claude/tools/pre_fin/hs_72_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = 9.6
YEL, HOT, SLATE, AMBER = old.YEL, old.HOT, old.SLATE, old.AMBER
T_BOLT, T_CUT, T_CHARGE = old.T_BOLT, old.T_CUT, old.T_CHARGE
T_DASH, T_TEAR, T_HIT = 6.9, 7.2, 8.9
STRIKES = [T_CHARGE + x for x in (0.5, 1.0, 1.4, 1.7)]


class Amatsu(FX, old.Amatsu):
    def __init__(s, W=540):
        old.Amatsu.__init__(s, W); s.fx_init(72); H = s.H; k = s.k
        P0 = (-W * 0.15, H * 0.64 + 0.15 * H * 0.34 + H * 0.0); P1 = (W * 1.15, H * 0.64 - 1.15 * H * 0.34)
        rz = np.random.RandomState(721); jt = [0.0] + [((1 if i % 2 else -1) * (10 + 30 * rz.rand()) * k) for i in range(1, 22)] + [0.0]; ts = np.sort(np.concatenate([[0, 1], (np.arange(1, 22) + (rz.rand(21) - 0.5) * 0.7) / 22.0]))
        s.zz = [(P0[0] + (P1[0] - P0[0]) * ts[i] - jt[i] * 0.34, P0[1] + (P1[1] - P0[1]) * ts[i] + jt[i]) for i in range(23)]
        m = s.L0(); cv2.fillPoly(m, [np.array([(-W, -H)] + s.zz + [(2 * W, -H)], np.int32)], 1.0, cv2.LINE_AA); s.mup = m      # 裂け目より 上
        s._heat = 0.0

    # ---- 黒雲(中から 光る) ----
    def clouds(s, t, lum=1.0, flashes=()):
        W, H, k = s.W, s.H, s.k; n = s.fbm(0.0007, t * 50, t * 6, 4, 110 * k, 0); n2 = s.fbm(0.0019, t * 80, 0, 3, 40 * k, 2)
        d = sstep(0.28, 0.78, n * 0.8 + n2 * 0.3); rel = np.clip((n - np.roll(n, int(30 * k), axis=0)) * 2.2 + (n2 - np.roll(n2, int(12 * k), axis=0)) * 0.5, -0.3, 0.45)
        f = np.array([0.010, 0.014, 0.032], np.float32)[None, None, :] + SLATE[None, None, :] * np.clip(0.04 + 0.55 * d + 0.9 * rel * d, 0, 2)[:, :, None] + (np.clip(rel, 0, 1) * d * 0.45)[:, :, None] * np.array([0.35, 0.42, 0.62], np.float32)[None, None, :]
        for (x, y, v) in flashes:                                                    # 雲の中の 稲光
            if v > 0.01: f += (np.exp(-(((s.XX - x) ** 2 + (s.YY - y) ** 2) / (W * 0.34) ** 2)) * (0.25 + d + 2.5 * np.clip(rel, 0, 1)) * v * 1.5)[:, :, None] * (YEL * 0.6 + 0.4)[None, None, :]
        return (f * lum).astype(np.float32), d

    def seam(s, f, v):
        if v > 0.01: s.glowline(f, s.zz, YEL * 0.7 + 0.3, 1.0, v, 0.9)

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        h = s._heat
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(YEL * 0.7 + 0.3), rim_dir=(-0.3, -0.95), rim_v=(0.55 + 0.45 * h) * alpha, amb=(0.6, 0.7, 1.0), amb_v=0.16 * (1 - h), aura=YEL if alpha > 0.9 else None, aura_v=0.10 + 0.14 * h, alpha=alpha, shade=0.86 + 0.14 * max(h, white))
        if white > 0.5 and alpha > 0.9:
            w = img.copy(); w[:, :, :3] = 1.0; s.put(f, w, cx, cy, hfrac * s.H / s.ah, 0.9 * white, 0.0, 0.0)

    def add(s, f, L, col, gain=1.0, b1=3, b2=12):
        s.glowmask(f, np.clip(L, 0, 1), col, gain * 0.8, 1.0, 1.0, float(b2))

    def frame(s, t):
        s._t = t; o = {}; s._heat = 0.0
        if t < T_CUT:
            f, sh = s.sc_storm(t); s.seam(f, 0.5 * max(math.exp(-abs(t - x) * 14) for x in (0.5, 1.0, 1.35)) + (0.4 * math.exp(-(t - T_BOLT) * 6) if t > T_BOLT else 0))
            s.rushdust(f, t, (s.W / 2, s.H * 0.5), 400, 0.3, YEL, 0.25, 3, 0.0)
        elif t < T_CHARGE: s._heat = 0.5; f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.9)
        elif t < T_DASH: f, sh = s.sc_charge(t)
        else: f, sh = s.sc_rend(t)
        f, sh = s.finale(f, sh, t, DUR, None, (1, 0.95, 0.4), (0.5, 0.7, 1), -1)
        return s.film(f, t, sh * s.k, 0.6, o.get("thr", 0.80), (1.0, 0.92, 0.70), 1.0 + 0.15 * sh, 0.045, o.get("vig", 0.6), 1.12, 1.10)

    # 2) 稲妻の形に 裂けた 窓 と 雷に 打たれて 出る 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f, _ = s.clouds(t, 0.35); E = (W * 0.60, H * 0.36); ex = (s.eyeL[0] + s.eyeR[0]) / 2; ey = s.eyeL[1]
        s.rushdust(f, -t, E, 900, 0.9, YEL, 0.5 * ease(u / 0.3), 3, 1.0)                # 目に あつまる 光の つぶ
        op = ease(u / 0.22); hw = W * 0.21 * op
        zig = [(W * 0.86, -10), (W * 0.52, H * 0.26), (W * 0.74, H * 0.30), (W * 0.36, H * 0.60), (W * 0.58, H * 0.63), (W * 0.20, H + 10)]
        left = [(x - hw, y) for x, y in zig]; right = [(x + hw, y) for x, y in zig]
        m = s.L0(); cv2.fillPoly(m, [np.array(left + right[::-1], np.int32)], 1.0, cv2.LINE_AA)
        win = s.nebula(t, [(0.45, 0.20, 0.03), (0.95, 0.55, 0.10), (1.0, 0.92, 0.55)], 0.0016, (40.0, 90.0), 1.0, 1) + (AMBER * 0.22 + YEL * 0.10)[None, None, :]
        win += (np.exp(-np.hypot(s.XX - E[0], s.YY - E[1]) / (W * 0.30)) * 0.35)[:, :, None] * HOT[None, None, :]
        zf = 1.2; sc = zf * H / s.ah
        s.hero(win, t, E[0] - (ex - s.aw / 2) * sc, E[1] - (ey - s.ah / 2) * sc, zf); s.eyes(win, 0.55)
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        for ln in (left, right): s.glowline(f, ln, YEL, 2.0, 0.9, 1.0)
        for i, T in enumerate(s.vt):                                                 # 1字ずつ 雷に 打たれて 出る
            a = (u - 0.45 - i * 0.16) / 0.1
            if a > 0:
                p = (W * 0.13, H * 0.44 + i * 80 * k); s.text(f, T, p[0], p[1], 1.0, 1.0)
                if a < 1.6: L2 = s.L0(); s.bolt(L2, (p[0] + 30 * k, -10), p, 300 + i * 9 + int(a * 3), 0.2, 1.0, 2, 1); s.glowmask(f, L2, HOT, 1.1 * clamp(1.6 - a), 1.0, 1.0, 10.0)
        for i, T in enumerate(s.vo):
            if u > 0.25 + i * 0.1: s.text(f, T, W * 0.13, H * 0.30 + i * 36 * k, 1.0, 1.0)
        if u > 1.4: s.text(f, s.txt_en, W * 0.62, H * 0.93, ease((u - 1.4) / 0.3), 1.0)
        if u < 0.1: f += (1 - u / 0.1) * 0.9
        return f, 0

    def vortex(s, f, O, t, heat):
        W, H, k = s.W, s.H, s.k; dx = s.XX - O[0]; dy = (s.YY - O[1]) / 0.45; r = np.hypot(dx, dy); a = np.arctan2(dy, dx); lr = np.log(r + 20 * k)
        u = ((a / 6.2832 * 512 + lr * 70 - t * (30 + 60 * heat)) % 256).astype(np.float32); v = ((lr * 95 - t * 22) % 256).astype(np.float32)
        sw = cv2.remap(s._tiles[1], u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP) * 0.6 + cv2.remap(s._tiles[3], (u * 2) % 256, (v * 2) % 256, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP) * 0.4
        wgt = np.exp(-r / (W * 0.60)); sw = sstep(0.30, 0.80, sw)
        f *= (1 - 0.75 * wgt)[:, :, None]; f += (sw * wgt * 0.9)[:, :, None] * (SLATE * 1.8 + YEL * 0.9 * heat * np.exp(-r / (W * 0.25))[:, :, None])
        hole = np.exp(-(r / (W * 0.10)) ** 2); f *= (1 - 0.92 * hole)[:, :, None]
        f += (np.exp(-((r - W * 0.13) / (W * 0.035)) ** 2) * (0.10 + 0.5 * heat))[:, :, None] * (YEL * 0.7 + 0.3)[None, None, :]

    # 3) 剣を 天へ。うず雲から 雷が 4度 落ちる
    def sc_charge(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_CHARGE; O = (W / 2, H * 0.17)
        last = max([x for x in STRIKES if x <= t] or [-9]); since = t - last; hits = len([x for x in STRIKES if x <= t]); heat = hits / len(STRIKES); s._heat = heat * 0.8
        f, d = s.clouds(t * 0.5, 0.7, [(O[0], O[1], 0.15 + 0.9 * math.exp(-since * 10))]); s.vortex(f, O, t, heat)
        s.seam(f, 0.08 + 0.35 * math.exp(-since * 8) * (hits > 0))
        crouch = ease((u - 1.85) / 0.4); cy = H * 0.70 + crouch * H * 0.02
        f += (np.exp(-((s.XX - W / 2) / (W * 0.30)) ** 2 - ((s.YY - cy - H * 0.10) / (H * 0.07)) ** 2) * (0.05 + 0.14 * heat))[:, :, None] * YEL[None, None, :]
        s.hero(f, t, W / 2, cy, 0.42, -24 * ease(u / 0.4) + 16 * crouch, squash=0.05 * math.exp(-since * 12) + 0.07 * crouch, jit=(2.0 * k * crouch))
        tip = s.pt(s.tip)
        if 0 <= since < 0.14:                                                        # 雷が 剣に 落ちる
            L = s.L0(); s.bolt(L, O, tip, int(last * 100) + int(since * 20), 0.3, 1.0, 2 + hits // 2); s.glowmask(f, L, HOT, 1.5, 1.0, 1.2, 14.0)
            s.flare(f, tip, 16 * k, HOT, 1.2 * math.exp(-since * 14), 0.8, 0.4, True); f += math.exp(-since * 30) * 0.35
        s.blade(f, heat, t); s.eyes(f, 0.25 + 0.35 * heat)
        s.rushdust(f, t, (W / 2, cy), 500, 0.4 + heat, YEL, 0.15 + 0.4 * heat, 5, 0.0)
        if u < 0.15: f += (1 - u / 0.15) * 0.7
        return f, 9 * math.exp(-since * 9) * (0.6 + heat) + 3 * crouch

    # 4) こちらへ 一閃 → 目の前の 空が 斜めに 裂ける → むこうの 雷が あふれる
    def sc_rend(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_DASH; v = t - T_TEAR; s._heat = 1.0
        C, d = s.clouds(t * 0.5, 0.7, [(W / 2, H * 0.45, 0.7 * math.exp(-u * 6))])
        s.hero(C, t, W / 2, H * 0.72, 0.42, -8 + 52 * ease(clamp(u / 0.10)), squash=-0.06 * math.exp(-u * 10)); s.blade(C, 1.0, t); s.eyes(C, 0.8)
        if v < 0:                                                                    # 光の 線が わたって、ひと呼吸
            C *= 0.45 + 0.55 * math.exp(-u * 12); g = ease(clamp(u / 0.11)); n = max(2, int(g * 23))
            s.glowline(C, s.zz[:n], HOT, 2.5, 1.0, 1.3)
            if n < 23: s.flare(C, s.zz[n - 1], 22 * k, HOT, 1.0, 1.0, -0.56, False)
            if u < 0.05: C = C + (1 - u / 0.05) * 0.8
            return C, 7 * math.exp(-u * 9)
        e = clamp(v / (T_HIT - T_TEAR)); gap = H * (0.015 + 0.15 * ease(clamp(v / 0.45)) + 1.5 * clamp((v - 0.4) / (T_HIT - T_TEAR - 0.4)) ** 2.3) + H * 0.008 * math.sin(v * 40)
        dx = gap * 0.34; dy = gap
        n = s.fbm(0.0016, t * 160, -t * 60, 3, 30 * k, 1); cl = np.abs(s.diag) / (gap * 0.5 + 1)                    # すき間の むこう: 金色の 光
        f = (AMBER * 0.55 + YEL * 0.45)[None, None, :] * (0.55 + 1.0 * n)[:, :, None] + (np.exp(-(cl / 0.5) ** 2) * (0.9 + 0.6 * e))[:, :, None] * HOT[None, None, :]
        f = f.astype(np.float32); Lb = s.L0(); sd = int(t * 15)
        for i in range(3 + int(7 * e)):                                              # むこうから こちらへ のびる 雷
            q = s.rr[(sd * 7 + i * 13) % 120]; x = W * (0.05 + 0.9 * q[0]); y = H * 0.64 - x / W * H * 0.34
            a = q[1] * 6.2832; ln = W * (0.25 + 1.3 * e) * (0.5 + q[2]); s.bolt(Lb, (x, y), (x + math.cos(a) * ln, y + math.sin(a) * ln * 1.3), sd * 101 + i * 17, 0.26, 1.0, 1 + int(3 * e + 2 * q[3]), 2)
        f2 = f.copy(); sh_ = np.zeros((H, W), np.float32)
        for sgn, msk in ((-1, s.mup), (1, 1 - s.mup)):                               # 上と 下が ずれて はなれる
            M = np.float32([[1, 0, sgn * dx * 0.5], [0, 1, sgn * dy * 0.5]])
            cm = cv2.warpAffine(C, M, (W, H), borderMode=cv2.BORDER_REFLECT); am = cv2.warpAffine(msk, M, (W, H), borderMode=cv2.BORDER_REPLICATE); cm = cm * am[:, :, None]; f2 = f2 * (1 - am[:, :, None]) + cm; sh_ = np.maximum(sh_, am)
        edge = np.clip(cv2.GaussianBlur(1 - sh_, (0, 0), 10 * k) - (1 - sh_), 0, 1); f2 += (edge * 1.6)[:, :, None] * (YEL * 0.8 + 0.2)[None, None, :]              # ふちに もれる 光
        f = f2; s.glowmask(f, Lb, HOT, 1.2 + 0.5 * e, 1.0, 1.2, 12.0)
        for sgn in (-1, 1): s.glowline(f, [(x + sgn * dx * 0.5, y + sgn * dy * 0.5) for x, y in s.zz], HOT, 1.5, 0.9, 1.0)
        s.rushdust(f, t, (W / 2, H * 0.47), 1300, 1.0 + 2.2 * e, HOT, 0.35 + 0.6 * e, 6, 1.0)
        f = s.godrays(f, (W / 2, H * 0.47), 0.35 + 0.5 * e, 0.85, 10, 0.86, 0.5)
        if v < 0.10: f = f + (1 - v / 0.10) * 1.0
        if e > 0.85: f = f + ((e - 0.85) / 0.15) ** 2 * 0.9
        return f, 12 * math.exp(-v * 4) + 3 + 10 * e ** 2

    # ---- この子だけの曲: 太鼓の 乱れ打ちと ひずんだ 旋律(ホ短調の五音)。雷が 落ちる間が つまる → 一閃の 高い 一音 → 裂ける とどろき ----
    def audio(s, path):
        a = Snd(DUR, 72); E = 164.81; pen = [0, 3, 5, 7, 10]
        def fq(d, o=0): return E * 2 ** ((pen[d % 5] + 12 * (d // 5 + o)) / 12.0)
        def thunder(t0, g=1.0): a.a += a.lp(a.nz, 0.06) * a.env(t0, 0.01, 0.9) * 1.6 * g + (a.nz - a.lp(a.nz, 0.4)) * a.env(t0, 0.001, 0.05) * 0.5 * g; a.boom(t0, 45, 0.9 * g, 0.5)
        def lead(d, t0, dur, g=0.07, o=1):
            a.note(fq(d, o), t0, dur, g, "saw", 0.005); a.note(fq(d, o) * 1.5, t0, dur * 0.7, g * 0.35, "sq", 0.005)
        a.a += a.lp(a.nz, 0.015) * 0.6 * a.gate(0, T_BOLT, 0.3)
        for x in (0.5, 1.0, 1.35): a.a += a.lp(a.nz, 0.04) * a.env(x, 0.02, 0.4) * 0.5; a.note(fq(9, 2), x, 0.25, 0.025, "sine", 0.01)       # 遠雷と、かすかな 光の すじ
        thunder(T_BOLT, 1.3)
        a.boom(T_CUT, 50, 1.0)
        for i in range(5): a.a += (a.nz - a.lp(a.nz, 0.5)) * a.env(T_CUT + 0.45 + i * 0.16, 0.001, 0.04) * 0.35; lead([4, 3, 2, 3, 5][i], T_CUT + 0.45 + i * 0.16, 0.14, 0.08)
        bt = 60.0 / 172 / 2; tk = T_CUT + 1.3; i = 0
        riff = [0, 0, 3, 0, 2, 0, 4, 3]
        while tk < T_DASH - 0.05:                                                    # 16分の 太鼓 と 低い刻み
            if i % 4 in (0, 3): a.boom(tk, 62, 0.4, 0.09)
            if i % 4 == 2: a.a += a.lp(a.nz, 0.5) * a.env(tk, 0.001, 0.05) * 0.25
            a.hat(tk, 0.04); a.note(fq(riff[i % 8], -1), tk, 0.09, 0.10, "saw", 0.003)
            if tk > T_CHARGE and i % 2 == 0: lead([5, 7, 5, 4, 5, 7, 8, 9][(i // 2) % 8], tk, 0.15, 0.055)
            tk += bt; i += 1
        for j, x in enumerate(STRIKES): thunder(x, 0.65 + j * 0.1); a.note(fq(5 + j, 1), x, 0.3, 0.07, "bell")
        a.rise(T_CHARGE + 1.85, T_DASH, 200, 1400, 0.16)
        a.whoosh(T_DASH - 0.08, T_DASH + 0.12, 0.8, 0.4); a.note(fq(9, 2), T_DASH + 0.10, 0.35, 0.12, "sine"); a.note(fq(9, 3), T_DASH + 0.10, 0.25, 0.05, "sine")   # 一閃: 高い 一音だけ
        thunder(T_TEAR, 1.5); a.a += (a.nz - a.lp(a.nz, 0.3)) * a.env(T_TEAR, 0.002, 0.25) * 0.5                        # 裂ける
        for d, o in ((0, -1), (2, 0), (4, 0), (5, 0)): a.note(fq(d, o), T_TEAR, T_HIT - T_TEAR, 0.05, "saw", 0.02)
        a.rise(T_TEAR, T_HIT, 50, 110, 0.20); a.whoosh(T_TEAR + 0.3, T_HIT, 0.9, 0.45)
        tk = T_TEAR; i = 0
        while tk < T_HIT - 0.03:
            if i % 2 == 0: a.boom(tk, 58, 0.4, 0.1)
            a.hat(tk, 0.05); lead([9, 8, 7, 5, 7, 8, 9, 10][i % 8] + (i // 8) * 2, tk, 0.12, 0.06)
            if i % 2 == 1: a.a += (a.nz - a.lp(a.nz, 0.4)) * a.env(tk, 0.001, 0.05) * (0.2 + 0.4 * (tk - T_TEAR) / (T_HIT - T_TEAR)); a.a += a.lp(a.nz, 0.1) * a.env(tk, 0.005, 0.2) * 0.35
            tk += bt * (1.0 - 0.35 * (tk - T_TEAR) / (T_HIT - T_TEAR)); i += 1
        thunder(T_HIT, 1.8); thunder(T_HIT + 0.2, 0.9)
        for d, o in ((0, -1), (0, 0), (2, 0), (4, 0), (5, 1)): a.note(fq(d, o), T_HIT + 0.03, 0.6, 0.05, "saw", 0.02)
        s.fin_snd(a, DUR, 72); a.save(path)


if __name__ == "__main__":
    r = Amatsu()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
