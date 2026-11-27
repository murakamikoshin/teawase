"""クロノス(★5 時の賢者) 専用の必殺技「時止メノ杖」— 高品質版(hs_fx)・通し直し。止めた 雨つぶ・木の葉・りんごを、ぜんぶ こちらへ 放つ
夜の雨。りんごの木の下へ ぴょんと 出る → 杖を かかげて「チン」。雨つぶ(ガラス玉)・木の葉・りんごが 空中で 止まり、色が あせる(前ふり) →
砂時計の形の 窓のカットイン → 空に 時計の 輪。カチ、カチ と 針が 進むたび、止まった ものが 少しずつ こちらへ にじりよる →
ぱちん。色が もどり、ぜんぶが いっぺんに こちらへ 飛んでくる。さいごに 杖の前の ひとつぶが 目の前を おおう → 白 → 暗転
使い方: python3 hs_32.py 出力.mp4 / python3 hs_32.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
import hissatsu as hz
_sp = importlib.util.spec_from_file_location("old32", "/home/claude/tools/pre_fin/hs_32_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = 9.2
GOLD, TEAL, ICE = old.GOLD, old.TEAL, old.ICE
T_STOP, T_CUT, T_BACK, T_SNAP, T_HIT = 1.8, 2.6, 4.6, 7.0, 8.5
TICKS = [5.1, 5.4, 5.7, 6.0, 6.25, 6.5, 6.7, 6.85]
ZSTEP = 0.03; ZM_END = 1 - ZSTEP * len(TICKS); ZH = 2.6
ND, NL, NA = 260, 30, 10


class Chronos(FX, old.Chronos):
    def __init__(s, W=540):
        old.Chronos.__init__(s, W); s.fx_init(32); H = s.H; k = s.k
        s.VPc = (W * 0.455, H * 0.432); s.MOc = (W * 0.80, H * 0.13)
        rg = np.random.RandomState(132); N = ND + NL + NA
        s.o_s = np.stack([rg.rand(N) * W * 1.1 - W * 0.05, rg.rand(N) * H * 0.98 - H * 0.04], 1).astype(np.float32)
        s.o_z = (1.2 + 7.8 * rg.rand(N) ** 0.8).astype(np.float32); s.o_z[:22] = 0.5 + 0.7 * rg.rand(22); s.o_z[ND:] = 1.0 + 5.0 * rg.rand(NL + NA)
        s.o_r = rg.rand(N, 4).astype(np.float32)
        s.o_rb = ((10 + 8 * s.o_r[:, 0]) * k).astype(np.float32); s.o_rb[ND:ND + NL] = 26 * k * (0.7 + 0.6 * s.o_r[ND:ND + NL, 0]); s.o_rb[ND + NL:] = 22 * k
        s.o_sp = np.full(N, 1.7, np.float32) * (0.8 + 0.4 * s.o_r[:, 1]); s.o_sp[ND:ND + NL] = 0.22; s.o_sp[ND + NL:] = 0.8
        s.o_ta = (0.22 + 1.22 * (np.argsort(np.argsort(s.o_z)) / float(N)) ** 0.85).astype(np.float32)                      # 近い ものから 順に とどく(とぎれない)
        s.o_s[0] = (s.VPc[0] + 0.3, s.VPc[1] - 0.3); s.o_z[0] = 2.0; s.o_rb[0] = 14 * k                         # さいごに 来る ひとつぶ
        # 動かない 大道具
        hx = np.interp((np.arange(W) * 0.35 / k) % 255, np.arange(256), s._tiles[2][7]); s._hill = sstep(0, 3 * k, s.YY - (H * (0.72 - 0.08 * hx))[None, :])
        cd = 1.15 - np.hypot(s.XX / (W * 0.80), s.YY / (H * 0.24)); cn = s.fbm(0.0028, 0, 0, 3, 0, 0); cv = cd * 0.55 + cn * 0.62
        cv = cv2.GaussianBlur(cv, (0, 0), 5 * k)
        s._can = sstep(0.50, 0.58, cv); s._canr = sstep(0.50, 0.58, cv) * sstep(0.70, 0.56, cv)
        s._lt = cv2.remap(s._tiles[0], ((s.XX * 1.3 / k) % 256).astype(np.float32), ((s.YY * 1.3 / k) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        s._gt = s.fbm(0.004, 0, 0, 3, 0, 2); s._gm = sstep(s.gnd - 2 * k, s.gnd + 2 * k, s.YY)
        s._rip = rg.rand(9, 3).astype(np.float32)
        s.rom = [s.text_img(c, hz.SERIF, int(30 * k), 1) for c in ("XII", "III", "VI", "IX")]

    # ---- 時の流れ ----
    def wt(s, t): return t if t < T_STOP else (T_STOP if t < T_SNAP else T_STOP + (t - T_SNAP))

    def zmul(s, t):
        n = 0.0
        for tk in TICKS:
            if t > tk: n += ease(clamp((t - tk) / 0.08))
        return 1.0 - ZSTEP * n

    def satv(s, t):
        if t < T_STOP: return 1.0
        if t < T_SNAP: return 1.0 - 0.58 * ease(clamp((t - T_STOP) / 0.25))
        return 0.42 + 0.58 * ease(clamp((t - T_SNAP) / 0.2))

    def grey(s, col, sat):
        if sat >= 0.999: return col
        g = col.mean(2, keepdims=True) * np.array([1.06, 0.99, 0.84], np.float32)[None, None, :]; return g * (1 - sat) * 1.25 + col * sat

    # ---- 夜の雨の 丘・りんごの木 ----
    def backdrop(s, t, sat=1.0):
        W, H, k = s.W, s.H, s.k; w = s.wt(t); icy = np.array([0.62, 0.76, 0.96], np.float32)
        f = (np.array([0.012, 0.025, 0.07], np.float32)[None, None, :] * (1 - s.yn)[:, :, None] + np.array([0.06, 0.13, 0.20], np.float32)[None, None, :] * s.yn[:, :, None]).astype(np.float32)
        d = np.hypot(s.XX - s.MOc[0], s.YY - s.MOc[1]) / (40 * k); cl = s.fbm(0.0011, w * 14, w * 3, 4, 50 * k, 1)
        f += (np.exp(-d / 2.6) * 0.60 + np.exp(-d / 9.0) * 0.16)[:, :, None] * icy[None, None, :]                    # 雲の むこうの 月
        f += (sstep(0.40, 0.85, cl) * (0.05 + 0.5 * np.exp(-d / 5.0)) * sstep(0.75, 0.25, s.yn))[:, :, None] * icy[None, None, :]
        f *= (1 - 0.4 * sstep(0.55, 0.30, cl) * sstep(0.7, 0.3, s.yn))[:, :, None]
        hc = np.array([0.014, 0.042, 0.058], np.float32)[None, None, :] * (0.7 + 0.6 * s._gt)[:, :, None]; f = f * (1 - s._hill[:, :, None]) + hc * s._hill[:, :, None]
        mist = s.fbm(0.0018, w * 18, 0, 3, 0, 3) * np.exp(-((s.yn - 0.76) / 0.08) ** 2); f += (mist * 0.22)[:, :, None] * np.array([0.5, 0.7, 0.8], np.float32)[None, None, :]
        gy = np.clip((s.YY - s.gnd) / (H - s.gnd), 0, 1)                                                           # ぬれた 地面
        gc = np.array([0.02, 0.05, 0.06], np.float32)[None, None, :] * (0.5 + 1.0 * s._gt)[:, :, None]
        gc = gc + (np.exp(-((s.XX - s.MOc[0]) / (50 * k * (1 + 2.5 * gy))) ** 2) * 0.20 * (0.4 + 1.2 * s._gt) + np.exp(-((s.XX - W * 0.5) / (90 * k)) ** 2) * 0.05 * getattr(s, '_gl', 0.0))[:, :, None] * icy[None, None, :]
        g0 = int(s.gnd) - 3; Xg = s.XX[g0:]; Yg = s.YY[g0:]; acc = np.zeros_like(Xg)
        for rx, ry, rp in s._rip:
            ph = (w * 0.9 + rp) % 1.0; de = np.hypot(Xg - rx * W, (Yg - (s.gnd + (0.15 + 0.8 * ry) * (H - s.gnd))) * 3.4) / k
            acc += np.exp(-((de - ph * 48) / 2.0) ** 2) * (1 - ph)
        gc[g0:] += (acc * 0.22)[:, :, None] * icy[None, None, :]
        f = f * (1 - s._gm[:, :, None]) + gc * s._gm[:, :, None]
        cc = np.array([0.018, 0.065, 0.035], np.float32)[None, None, :] * (0.3 + 1.6 * s._lt)[:, :, None] + (s._canr * (0.03 + 0.10 * s._lt))[:, :, None] * icy[None, None, :]   # りんごの木の 枝ぶり
        f = f * (1 - s._can[:, :, None]) + cc * s._can[:, :, None]
        f = s.grey(f, sat)
        for ax, ay in ((0.10, 0.215), (0.33, 0.20), (0.52, 0.125)): s.apple(f, (W * ax, H * ay), 13 * k, sat)
        return f

    # ---- 止まる もの(式で 描く) ----
    def bead(s, f, c, r, al=1.0, gold=0.0):
        """ガラス玉の 雨つぶ: うしろの 景色が さかさに うつる・ふちの 光・てり"""
        W, H = s.W, s.H; x0 = int(max(0, c[0] - r - 2)); x1 = int(min(W, c[0] + r + 2)); y0 = int(max(0, c[1] - r - 2)); y1 = int(min(H, c[1] + r + 2))
        if x1 - x0 < 2 or y1 - y0 < 2 or al <= 0.01: return
        X = (s.XX[y0:y1, x0:x1] - c[0]) / r; Y = (s.YY[y0:y1, x0:x1] - c[1]) / r; d = np.hypot(X, Y); aa = min(0.6, 1.6 / r); m = sstep(1.0, 1.0 - aa, d) * al
        lens = 1.5 * (1.0 + 0.9 * np.clip(d, 0, 1) ** 2); mag = min(r, 60 * s.k) + 0.25 * max(0.0, r - 60 * s.k)
        refr = cv2.remap(s._bg, (c[0] - X * mag * lens).astype(np.float32), (c[1] - Y * mag * lens).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        dc = np.clip(d, 0, 1); tint = ICE * (1 - gold) + GOLD * gold
        col = refr * (1.5 - 0.5 * dc ** 2)[:, :, None] * (1 - 0.55 * np.exp(-((dc - 0.86) / 0.08) ** 2))[:, :, None] + (0.05 + 0.40 * dc ** 9)[:, :, None] * tint[None, None, :]
        col += (np.exp(-((dc - 0.70) / 0.13) ** 2) * np.clip(Y * 0.9 + X * 0.3, 0, 1) ** 1.5 * 0.75)[:, :, None] * (tint * 0.8 + 0.2)[None, None, :] + (np.clip(-Y, 0, 1) * (1 - dc) * 0.10)[:, :, None]
        col += (np.exp(-(((X + 0.40) / 0.20) ** 2 + ((Y + 0.44) / 0.14) ** 2)) * 1.1)[:, :, None] + (np.exp(-(((X - 0.34) / 0.30) ** 2 + ((Y - 0.48) / 0.16) ** 2)) * 0.30)[:, :, None] * tint[None, None, :]
        F = f[y0:y1, x0:x1]; F[:] = F * (1 - m[:, :, None]) + col * m[:, :, None]

    def leaf(s, f, c, S, rot, tum, sat, hue=0.5, al=1.0):
        W, H = s.W, s.H; x0 = int(max(0, c[0] - S - 2)); x1 = int(min(W, c[0] + S + 2)); y0 = int(max(0, c[1] - S - 2)); y1 = int(min(H, c[1] + S + 2))
        if x1 - x0 < 2 or y1 - y0 < 2 or al <= 0.01: return
        dx = s.XX[y0:y1, x0:x1] - c[0]; dy = s.YY[y0:y1, x0:x1] - c[1]; ca, sa = math.cos(rot), math.sin(rot); u = (dx * ca + dy * sa) / S; w = (-dx * sa + dy * ca) / S
        wid = 0.42 * (0.28 + 0.72 * abs(math.cos(tum))); half = wid * np.clip(1 - u * u, 0, 1) ** 0.7 * (1 + 0.35 * u); aa = min(0.25, 1.4 / S)
        m = sstep(half + aa * 0.5, half - aa * 0.5, np.abs(w)) * (np.abs(u) < 1) * al
        rib = np.exp(-(w / (0.028 + aa * 0.3)) ** 2); vein = 0.5 + 0.5 * np.sin(u * 22 + np.abs(w) * 34); sh = 0.62 + 0.38 * np.clip(w / (half + 1e-3), -1, 1) * math.sin(tum + 0.6)
        base = np.array([0.86, 0.52, 0.12], np.float32) * (1 - hue) + np.array([0.74, 0.20, 0.07], np.float32) * hue
        col = base[None, None, :] * (sh * (1 - 0.45 * rib) * (0.82 + 0.18 * vein) * (0.75 + 0.5 * (u * 0.5 + 0.5)))[:, :, None] + (np.exp(-((np.abs(w) - half) / 0.03) ** 2) * 0.10)[:, :, None] * ICE[None, None, :]
        col = s.grey(col, sat); F = f[y0:y1, x0:x1]; F[:] = F * (1 - m[:, :, None]) + col * m[:, :, None]

    def apple(s, f, c, S, sat, al=1.0):
        W, H = s.W, s.H; x0 = int(max(0, c[0] - S * 1.1 - 2)); x1 = int(min(W, c[0] + S * 1.1 + 2)); y0 = int(max(0, c[1] - S * 1.1 - 2)); y1 = int(min(H, c[1] + S * 1.1 + 2))
        if x1 - x0 < 2 or y1 - y0 < 2 or al <= 0.01: return
        X = (s.XX[y0:y1, x0:x1] - c[0]) / S; Y = (s.YY[y0:y1, x0:x1] - c[1]) / S
        d = np.hypot(X / 1.05, Y) + 0.13 * np.exp(-(X / 0.30) ** 2) * np.clip(-Y, 0, 1) ** 2; aa = min(0.4, 1.5 / S); m = sstep(1.0, 1.0 - aa, d) * al
        nz = np.sqrt(np.clip(1 - d * d, 0, 1)); lam = np.clip(-0.45 * X - 0.55 * Y + 0.70 * nz, 0, 1)
        sp = cv2.remap(s._tiles[0], ((X * 70 + 128) % 256).astype(np.float32), ((Y * 70 + 128) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        sp2 = cv2.remap(s._tiles[3], ((X * 22 + 60) % 256).astype(np.float32), ((Y * 22 + 90) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        pt = sstep(0.55, 0.85, sp2 + 0.25 * Y)[:, :, None]; base = np.array([0.74, 0.07, 0.08], np.float32)[None, None, :] * (1 - pt) + np.array([0.82, 0.62, 0.16], np.float32)[None, None, :] * pt
        col = base * ((0.13 + 1.0 * lam ** 1.2) * (0.85 + 0.3 * sp))[:, :, None] + (np.exp(-(((X + 0.38) / 0.17) ** 2 + ((Y + 0.42) / 0.12) ** 2)) * 0.8)[:, :, None] + ((1 - nz) ** 3 * 0.22 * (d < 1))[:, :, None] * ICE[None, None, :]
        col = s.grey(col, sat); F = f[y0:y1, x0:x1]; F[:] = F * (1 - m[:, :, None]) + col * m[:, :, None]
        if S > 3: cv2.line(f, (int(c[0]), int(c[1] - S * 0.82)), (int(c[0] + S * 0.14), int(c[1] - S * 1.22)), (0.09 * al, 0.06 * al, 0.03 * al), max(1, int(S * 0.09)), cv2.LINE_AA)

    def place(s, t):
        """いまの 画面位置・奥行き・大きさ"""
        H = s.H; tau = t - T_SNAP; z0 = s.o_z
        z = z0 * s.zmul(t) if tau <= 0 else np.where(tau < s.o_ta, z0 * ZM_END * np.clip(1 - tau / s.o_ta, 0, 1) ** 1.7, -1.0).astype(np.float32)
        if tau > 0: e = clamp(tau / (T_HIT - T_SNAP)); z = z.copy(); z[0] = 2.0 * ZM_END * (1 - e) ** 2.2 + 0.011
        wf = min(t, T_STOP) - T_STOP; pad = H * 0.08; Hw = H * 1.15
        sy = ((s.o_s[:, 1] + wf * s.o_sp * H / np.sqrt(z0) + pad) % Hw) - pad; sx = s.o_s[:, 0].copy()
        sx[ND:ND + NL] += np.sin(s.wt(t) * 3 + s.o_r[ND:ND + NL, 2] * 9) * 22 * s.k / z0[ND:ND + NL]
        if t < T_STOP: sy[0] = -99; 
        zz = np.maximum(z, 1e-3); rt = z0 / zz
        if tau > 0: rt = (1 / ZM_END) * (rt * ZM_END) ** 0.42                          # こちらを ねらって 来る(わきへ それない)
        px = s.VPc[0] + (sx - s.VPc[0]) * rt; py = s.VPc[1] + (sy - s.VPc[1]) * rt
        if t < T_STOP: px = sx; py = sy
        return px, py, z, s.o_rb / zz

    def objs(s, f, t, near, sat):
        W, H, k = s.W, s.H, s.k; tau = t - T_SNAP; w = s.wt(t); px, py, z, R = s.place(t)
        sel = (z > 0.004) & ((z < ZH) if near else (z >= ZH)) & (px > -R - 20) & (px < W + R + 20) & (py > -R - 20) & (py < H + R + 20)
        s._bg = f.copy(); gold = 0.0 if t < T_STOP else 0.35
        if t < T_STOP:                                                               # ふっている: すじ
            L = s.L0(); idx = np.where(sel[:ND])[0]
            for i in idx:
                ln = s.o_sp[i] * H / math.sqrt(s.o_z[i]) * 0.035; cv2.line(L, (int(px[i]), int(py[i])), (int(px[i]), int(py[i] - ln)), float(min(0.55, 0.16 + 0.3 / s.o_z[i])), 1 if s.o_z[i] > 1.5 else 2, cv2.LINE_AA)
            f += (L * 0.8 + cv2.GaussianBlur(L, (0, 0), 1.6 * k) * 0.8)[:, :, None] * ICE[None, None, :]
        else:
            dsel = sel.copy(); dsel[ND:] = False; small = dsel & (R < 2.2)
            if tau > 0.02:                                                           # こちらへ 飛ぶ: 細かい つぶは 尾を ひく
                p0x, p0y, _, _ = s.place(t - 0.035); L = s.L0()
                for i in np.where(dsel & (R < 6))[0]: cv2.line(L, (int(p0x[i]), int(p0y[i])), (int(px[i]), int(py[i])), float(min(1.0, 0.25 + 0.2 * R[i])), 1 if R[i] < 3 else 2, cv2.LINE_AA)
                f += (L * 0.9 + cv2.GaussianBlur(L, (0, 0), 2.0 * k) * 0.9)[:, :, None] * (ICE * 0.8 + 0.2)[None, None, :]
            if small.any(): s.splat(f, px[small], py[small], 0.10 * R[small], ICE * (1 - gold) + GOLD * gold, 0.8, 0.3)
            big = np.where(dsel & (R >= 2.2))[0]
            for i in big[np.argsort(-z[big])]:
                if i == 0: continue
                s.bead(f, (px[i], py[i]), R[i], 1 - float(sstep(W * 0.28, W * 0.62, R[i])), gold)
        oth = np.where(sel & (np.arange(len(z)) >= ND))[0]
        for i in oth[np.argsort(-z[oth])]:
            al = 1 - float(sstep(W * 0.35, W * 0.75, R[i])); r = s.o_r[i]
            if i < ND + NL: s.leaf(f, (px[i], py[i]), R[i], r[2] * 6.28 + w * 5 * (r[3] - 0.5) + max(0.0, tau) * 5 * (r[2] - 0.5), w * 3.2 + r[3] * 9 + max(0.0, tau) * 6, sat, float(r[1]), al)
            else: s.apple(f, (px[i], py[i]), R[i], sat, al)
        if near and t >= T_STOP and sel[0]: s._bg = f.copy(); s.bead(f, (px[0], py[0]), R[0], 1.0, gold)        # 杖の前の ひとつぶ(さいごに 来る)

    # ---- 時計の 輪 ----
    def dial(s, f, t):
        u = t - T_BACK
        if u <= 0: return
        W, H, k = s.W, s.H, s.k; tau = t - T_SNAP; c = s.VPc
        R = W * 0.44 * (0.3 + 0.7 * ease(clamp(u / 0.5))) * (1 + (5 * tau ** 1.5 if tau > 0 else 0)); v = clamp(u / 0.4) * (clamp(1 - tau / 0.45) if tau > 0 else 1.0)
        if v <= 0.01: return
        n = sum(ease(clamp((t - tk) / 0.08)) for tk in TICKS if t > tk); L = s.L0()
        for rr, th in ((1.0, 2), (0.93, 1), (0.60, 1)): cv2.circle(L, (int(c[0]), int(c[1])), int(R * rr), 1.0 if th == 2 else 0.6, max(1, int(th * k)), cv2.LINE_AA)
        for i in range(60):
            a = i * 6.2832 / 60; r0 = 0.86 if i % 5 == 0 else 0.90
            cv2.line(L, (int(c[0] + math.cos(a) * R * r0), int(c[1] + math.sin(a) * R * r0)), (int(c[0] + math.cos(a) * R * 0.93), int(c[1] + math.sin(a) * R * 0.93)), 1.0 if i % 5 == 0 else 0.5, max(1, int((2 if i % 5 == 0 else 1) * k)), cv2.LINE_AA)
        am = -1.5708 - 6.2832 * (len(TICKS) - n) / 8.0; ah = -1.5708 - 0.5236 * (len(TICKS) - n) / len(TICKS)
        for a, ln, th in ((am, 0.80, 2), (ah, 0.52, 3)):
            cv2.line(L, (int(c[0] - math.cos(a) * R * 0.10), int(c[1] - math.sin(a) * R * 0.10)), (int(c[0] + math.cos(a) * R * ln), int(c[1] + math.sin(a) * R * ln)), 1.0, max(1, int(th * k)), cv2.LINE_AA)
        d = np.hypot(s.XX - c[0], s.YY - c[1]); ang = np.arctan2(s.YY - c[1], s.XX - c[0])
        gear = np.exp(-((d - R * 0.965) / (R * 0.014 + 1)) ** 2) * sstep(-0.2, 0.2, np.sin(ang * 48 + n * 0.785)) * 0.6 + np.exp(-((d - R * 0.60) / (R * 0.05 + 1)) ** 2) * sstep(0.0, 0.6, np.sin(ang * 12 - n * 0.785)) * 0.18
        s.glowmask(f, np.clip(L + gear, 0, 1), GOLD, 0.55 * v, 0.9, 1.0, 9.0)
        if tau <= 0:
            for T, (dx, dy) in zip(s.rom, ((0, -1), (1, 0), (0, 1), (-1, 0))): s.text(f, T, c[0] + dx * R * 0.76, c[1] + dy * R * 0.76, 0.75 * v, 1.0)

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        g = getattr(s, '_gl', 0.0)
        s.draw_lit(f, img, cx, cy, hfrac, rim=(0.75, 0.90, 1.0), rim_dir=(0.5, -0.86), rim_v=0.8 * alpha, amb=(1.0, 0.86, 0.55) if g > 0.3 else (0.6, 0.8, 1.0), amb_v=0.14 + 0.08 * g, aura=GOLD if g > 0.05 else None, aura_v=0.12 * g, alpha=alpha, shade=0.86 + 0.12 * g)

    def add(s, f, L, col, gain=1.0, b1=3, b2=12):
        s.glowmask(f, np.clip(L, 0, 1), col, gain, 0.8, 1.0, float(b2))

    def gem(s, f, v, t):
        if v <= 0.01: return
        p = s.at(*s.gemp); sc = s._pl[2]; s.flare(f, p, (9 + 2 * math.sin(t * 8)) * sc * 2 * min(1.6, v), ICE * 0.5 + GOLD * 0.5, 0.6 * min(1.5, v), 0.7, 0.0, v > 0.9)

    def frame(s, t):
        o = {}
        if t < T_CUT: f, sh = s.sc_stop(t)
        elif t < T_BACK: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.9)
        else: f, sh = s.sc_go(t)
        f, sh = s.finale(f, sh, t, DUR, None, (1, 0.84, 0.4), (0.3, 0.9, 0.75), -1)
        return s.film(f, t, sh * s.k, 0.55, o.get("thr", 0.82), (0.9, 0.97, 1.0), 1.0 + 0.15 * sh, 0.04, o.get("vig", 0.6), 1.12, 1.12)

    # 1) 雨の中へ ぴょんと 出て、時を 止める
    def sc_stop(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_STOP; sat = s.satv(t)
        s._gl = clamp((t - 1.3) / 0.4); f = s.backdrop(t, sat); s.objs(f, t, False, sat)
        hop = math.sin(clamp(t / 1.0) * 3.1416) * H * 0.06 if t < 1.0 else 0
        cx = W * (-0.3 + 0.8 * ease(t / 1.0)); cy = s.gnd - H * 0.24 - hop
        img = s.pose(t, squash=(0.06 * math.exp(-(t - 1.0) * 9) * math.cos((t - 1.0) * 18) if t > 1.0 else 0) + (-0.05 * math.exp(-u * 8) if u > 0 else 0))
        s.draw(f, img, cx, cy, 0.44); s.gem(f, clamp((t - 1.3) / 0.4) + (1.2 * math.exp(-u * 4) if u > 0 else 0), t)
        s.objs(f, t, True, sat)
        if u > 0:
            p = s.at(*s.gemp)
            for i in range(3):
                v = (u - i * 0.12) / 0.8
                if 0 < v < 1: s.glowline(f, [(p[0] + math.cos(a * 0.1309) * W * 1.4 * v, p[1] + math.sin(a * 0.1309) * W * 1.4 * v) for a in range(48)], ICE, 1.5, (1 - v) * 0.8, 1.0, True)
            if u < 0.08: f += (1 - u / 0.08) * 0.6
        return f * clamp(t / 0.3), (5 * math.exp(-u * 9) if u > 0 else 0)

    # 2) 砂時計の形の 窓
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT; s._gl = 0.6
        f = s.nebula(t, [(0.02, 0.10, 0.12), (0.05, 0.20, 0.19), (0.45, 0.36, 0.16)], 0.0009, (8.0, 3.0), 0.55) + np.array([0.012, 0.02, 0.035], np.float32)[None, None, :]
        s.starfield(f, t, 500, GOLD, 4, 0.5, (0.0, 30.0))
        g = back(u / 0.35, 1.2); C = (W / 2, H * 0.38); w = max(4.0, W * 0.40 * g); h = max(4.0, H * 0.27 * g)
        top = [(C[0] - w, C[1] - h), (C[0] + w, C[1] - h), C]; bot = [C, (C[0] + w, C[1] + h), (C[0] - w, C[1] + h)]
        m = s.L0(); cv2.fillPoly(m, [np.array(top, np.int32)], 1.0, cv2.LINE_AA); mb = s.L0(); cv2.fillPoly(mb, [np.array(bot, np.int32)], 1.0, cv2.LINE_AA)
        win = s.nebula(t + 9, [(0.04, 0.22, 0.20), (0.10, 0.40, 0.34), (0.75, 0.66, 0.36)], 0.0016, (14.0, 22.0), 0.9, 1) + (TEAL * 0.16)[None, None, :]
        win = s.godrays(win + (np.exp(-np.hypot(s.XX - C[0], s.YY - (C[1] - h)) / (90 * k)) * 0.5)[:, :, None] * GOLD[None, None, :], (C[0], C[1] - h), 0.5, 0.35)
        zf = 1.15; sc = zf * H / s.ah; img = s.pose(t); s.draw(win, img, C[0] - (s.face[0] - s.aw / 2) * sc, C[1] - h * 0.60 - (s.face[1] - s.ah / 2) * sc, zf)
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]; f *= (1 - 0.55 * mb)[:, :, None]
        lv = clamp(u / 1.9); q = math.sqrt(lv)                                        # 下の玉に 砂が たまる(つぶの きらめき)
        P = [(C[0] - w, C[1] + h), (C[0] + w, C[1] + h), (C[0] + w * (1 - q), C[1] + h * (1 - q)), (C[0] - w * (1 - q), C[1] + h * (1 - q))]
        sm = s.L0(); cv2.fillPoly(sm, [np.array(P, np.int32)], 1.0, cv2.LINE_AA)
        gr = cv2.remap(s._tiles[0], ((s.XX * 9 / k) % 256).astype(np.float32), ((s.YY * 9 / k + u * 40) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        dep = np.clip((s.YY - (C[1] + h * (1 - q))) / (h * q + 1), 0, 1); sand = GOLD[None, None, :] * ((0.30 + 0.9 * gr) * (1.0 - 0.45 * dep))[:, :, None] + (sstep(0.80, 0.95, gr) * 0.9)[:, :, None]
        f = f * (1 - sm[:, :, None]) + sand * sm[:, :, None]
        if lv < 1:
            s.glowline(f, [C, (C[0], C[1] + h * (1 - q))], GOLD, 1.5, 0.9, 0.8)
            rg = np.random.RandomState(int(t * 30)); n = 60; s.splat(f, C[0] + (rg.rand(n) - 0.5) * 8 * k, C[1] + rg.rand(n) * h * (1 - q), rg.rand(n) * 0.5, GOLD, 0.8, 0.3)
        sheen = np.exp(-((s.XX - s.YY * 0.45 - (C[0] - w * 0.55 - C[1] * 0.45)) / (22 * k)) ** 2) * 0.10 + np.exp(-((s.XX - s.YY * 0.45 - (C[0] + w * 0.2 - C[1] * 0.45)) / (7 * k)) ** 2) * 0.07
        f += (sheen * np.clip(m + mb, 0, 1))[:, :, None] * ICE[None, None, :]
        for tri in (top, bot): s.glowline(f, tri, GOLD, 2.2, 0.7, 0.8, True)
        for yy in (C[1] - h, C[1] + h): s.glowline(f, [(C[0] - w * 1.1, yy), (C[0] + w * 1.1, yy)], GOLD, 4.0, 0.8, 0.9)
        n = len(s.ht); step = W * 0.175
        for i, T in enumerate(s.ht):                                                 # 技名: 砂のように 上から 落ちて 止まる
            a = (u - 0.5 - i * 0.13) / 0.25
            if a > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.80 - (1 - ease(a)) * 70 * k, clamp(a * 2), 1.0)
        for i, T in enumerate(s.vo):
            if u > 0.4 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.705, 1.0, 1.0)
        if u > 1.4: s.text(f, s.txt_en, W / 2, H * 0.80 + 60 * k, ease((u - 1.4) / 0.3), 1.0)
        if u < 0.08: f += (1 - u / 0.08) * 0.7
        return f, 0

    # 3) カチ、カチ と にじりよる → ぱちん → ぜんぶ こちらへ
    def sc_go(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_BACK; tau = t - T_SNAP; sat = s.satv(t); e = clamp(tau / (T_HIT - T_SNAP)) if tau > 0 else 0.0
        tk = max([0.0] + [math.exp(-(t - x) * 14) for x in TICKS if t > x]); s._gl = 0.7 + 0.3 * tk
        f = s.backdrop(t, sat); s.dial(f, t)
        if tau > 0: s.rushdust(f, t, s.VPc, 1100, 0.6 + 1.6 * e, ICE, 0.25 + 0.6 * e, 3, 1.0)
        s.objs(f, t, False, sat)
        hop = abs(math.sin(tau * 7.0)) * H * 0.03 * math.exp(-tau * 1.2) if tau > 0 else 0
        img = s.pose(t, squash=0.035 * tk + (0.07 * math.exp(-tau * 10) if tau > 0 else 0)); s.draw(f, img, W / 2, s.gnd - H * 0.24 - hop, 0.44)
        s.gem(f, 0.8 + 0.5 * tk + (1.4 * math.exp(-tau * 5) if tau > 0 else 0), t)
        s.objs(f, t, True, sat)
        sh = 2.5 * tk
        if tau > 0:
            p = s.at(*s.gemp)
            for i in range(2):
                v = (tau - i * 0.1) / 0.6
                if 0 < v < 1: s.glowline(f, [(p[0] + math.cos(a * 0.1309) * W * 1.5 * v, p[1] + math.sin(a * 0.1309) * W * 1.5 * v) for a in range(48)], GOLD * 0.6 + 0.4, 2.0, (1 - v) * 0.9, 1.0, True)
            if tau < 0.1: f = f + (1 - tau / 0.1) * 0.8
            sh = 9 * math.exp(-tau * 8) + 2 + 9 * e ** 2
            if e > 0.86: f = f + ((e - 0.86) / 0.14) ** 2 * 0.8
        if u < 0.12: f = f * (u / 0.12) + (1 - u / 0.12)
        return f, sh

    # ---- この子だけの曲: 陽気な 3拍子の オルゴール。カチコチが おそくなって 止まる → 針の カチ、カチに ひと音ずつ → ぱちん → しずくが いっせいに 通りすぎる ----
    def audio(s, path):
        a = Snd(DUR, 32); F = 349.23; maj = [0, 2, 4, 5, 7, 9, 11]
        def fq(d, o=0): return F * 2 ** ((maj[d % 7] + 12 * (d // 7 + o)) / 12.0)
        def box(d, t0, g=0.09, o=0, dec=0.4): a.note(fq(d, o), t0, dec, g, "bell", 0.001)
        def tick(t0, g=0.14, hi=True): a.note(2400 if hi else 1700, t0, 0.02, g, "sine", 0.0005)
        tune = [0, 2, 4, 7, 4, 2, 5, 4, 2, 1, 2, 4]
        a.a += (a.nz - a.lp(a.nz, 0.5)) * 0.06 * a.gate(0, T_STOP, 0.2)                                                 # 雨
        tk = 0.1; i = 0
        while tk < T_STOP: box(tune[i % 12], tk, 0.08, 1, 0.25); tick(tk, 0.10, i % 2 == 0); a.note(fq([0, 4, 4][i % 3], -1), tk, 0.2, 0.05, "sq"); tk += 0.14 + i * 0.008; i += 1
        a.note(fq(7, 2), T_STOP, 1.4, 0.16, "bell"); a.note(fq(11, 2), T_STOP, 1.2, 0.08, "bell")                        # チン
        a.a += a.tone(58) * 0.07 * a.gate(T_STOP, T_SNAP, 0.2)
        a.boom(T_CUT, 60, 0.7, 0.2)
        for i in range(5): box([7, 5, 4, 2, 0][i], T_CUT + 0.5 + i * 0.13, 0.10, 1)
        a.a += (a.nz - a.lp(a.nz, 0.6)) * 0.04 * a.gate(T_CUT, T_CUT + 1.9, 0.1)                                        # 砂
        a.boom(T_BACK, 70, 0.6, 0.2); a.note(fq(0, 0), T_BACK, 1.0, 0.07, "bell")                                       # 時計の 輪
        for i, x in enumerate(TICKS):                                                                                   # カチ、カチ(ひと音ずつ 上がる)
            tick(x, 0.22, i % 2 == 0); a.note(900, x + 0.012, 0.03, 0.10, "sq", 0.0005); box(tune[i] + (i // 4) * 2, x, 0.10, 1, 0.3); a.boom(x, 90, 0.25 + 0.05 * i, 0.06)
        a.rise(T_BACK + 0.4, T_SNAP, 160, 900, 0.07)
        a.note(3000, T_SNAP, 0.05, 0.25, "sine", 0.0005); a.a += (a.nz - a.lp(a.nz, 0.7)) * a.env(T_SNAP, 0.0005, 0.02) * 0.5   # ぱちん
        a.boom(T_SNAP + 0.02, 55, 1.0, 0.3); a.whoosh(T_SNAP, T_HIT, 0.9, 0.4); a.rise(T_SNAP, T_HIT, 300, 1500, 0.10)
        rg = np.random.RandomState(32)
        for i in range(34):                                                                                             # しずくが 耳もとを 通りすぎる
            x = T_SNAP + 0.12 + (T_HIT - T_SNAP - 0.2) * (i / 34.0) ** 0.8; fr = 1500 + rg.rand() * 1400
            a.rise(x, x + 0.07, fr, fr * 0.45, 0.05 + 0.05 * i / 34.0)
        for i in range(8): box(tune[i] + 7, T_SNAP + 0.15 + i * 0.15, 0.09, 1, 0.25); a.note(fq([0, 4, 4][i % 3], -1), T_SNAP + 0.15 + i * 0.15, 0.2, 0.06, "sq")
        a.boom(T_HIT, 40, 1.7, 0.6); a.a += a.lp(a.nz, 0.3) * a.env(T_HIT, 0.005, 0.4) * 0.6
        for d in (0, 2, 4, 7): box(d, T_HIT, 0.08, 1, 0.6)
        s.fin_snd(a, DUR, 32); a.save(path)


if __name__ == "__main__":
    r = Chronos()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
