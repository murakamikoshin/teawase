"""バハル(★6 島を背負う海竜) 専用の必殺技「海呑ミ」— 高品質版(hs_fx)・通し直し。海ごと のまれて、上下の 歯が 目の前で かみ合う(この子だけ 黒で おわる)
深い 海へ 沈んでいく(上から さす 光・のぼる 泡・ただよう ちり) → うろこが 画面を おおう カットイン(すき間に 竜の かぶとの 目) →
まっくらな 底で、この子より 大きな「目」が ひらく(前ふり) → 槍で こちらを 指す。目が こちらを 見て、頭の 上を 通りすぎる →
画面の 上と 下の ふちから 歯の 列が せり出す(こちらは もう 口の 中)。泡も 光も ぜんぶ こちらへ 吸いこまれ、歯が かみ合う → まっくら → ごくり
使い方: python3 hs_76.py 出力.mp4 / python3 hs_76.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old76", "/home/claude/tools/pre_fin/hs_76_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = 9.9
AQUA, ABYSS, GLOW, BONE = old.AQUA, old.ABYSS, old.GLOW, old.BONE
T_CUT = old.T_CUT; T_AB = T_CUT + 2.0
T_EYE, T_ORDER, T_JAW, T_BITE, T_SNAP = 5.2, 6.2, 6.8, 8.75, 9.15


class Bahr(FX, old.Bahr):
    def __init__(s, W=540):
        old.Bahr.__init__(s, W); s.fx_init(76); s.yn = s.yn.astype(np.float32); s.xn = s.xn.astype(np.float32)

    # ---- 深い 海 ----
    def deep(s, t, depth=0.0, lum=1.0):
        W, H, k = s.W, s.H, s.k; top = AQUA * (0.75 * (1 - depth) + 0.04)
        f = top[None, None, :] * ((1 - s.yn) ** 1.5)[:, :, None] + ABYSS[None, None, :] * (0.6 + s.yn)[:, :, None]
        n = s.fbm(0.0011, t * 8, t * 3, 3, 50 * k, 0); f = f * (0.7 + 0.6 * n)[:, :, None]
        sx = s.XX - s.YY * 0.22; ray = cv2.remap(s._tiles[1], ((sx * 0.55 / k + t * 5) % 256).astype(np.float32), ((s.YY * 0.02 / k + t * 2) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)    # 上から さす 光
        f += (sstep(0.52, 0.85, ray) * (1 - s.yn) ** 1.3 * (1 - depth) * 0.45)[:, :, None] * (AQUA * 0.8 + 0.2)[None, None, :]
        s.starfield(f, t, 420, GLOW * 0.5 + 0.4, 6, 0.35 + 0.25 * (1 - depth), (6.0, -18.0))                              # ただよう ちり
        return (f * lum).astype(np.float32)

    def helm(s, f, v):
        if v <= 0.01: return
        sc = s._pl[2]
        for e in (s.hL, s.hR): s.flare(f, s.at(*e), 6 * sc * 2, GLOW, 0.55 * min(1.2, v), 0.6, -0.3, False)

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(AQUA * 0.7 + 0.3), rim_dir=getattr(s, '_rd2', (0.2, -0.98)), rim_v=0.75 * alpha, amb=(0.45, 0.75, 1.0), amb_v=0.22, aura=AQUA, aura_v=0.10, alpha=alpha, shade=0.82)

    def add(s, f, L, col, gain=1.0, b1=3, b2=12):
        s.glowmask(f, np.clip(L, 0, 1), col, gain * 0.8, 1.0, 1.0, float(b2))

    def eye2(s, f, c, R, open_, t, look=(0.0, 0.0)):
        """大きな 目(式): 虹彩の すじ・たての ひとみ・ぬれた てり・まぶたの 影"""
        if open_ <= 0.02: return
        W, H = s.W, s.H; x0 = int(max(0, c[0] - R * 1.25)); x1 = int(min(W, c[0] + R * 1.25)); y0 = int(max(0, c[1] - R)); y1 = int(min(H, c[1] + R))
        if x1 - x0 < 3 or y1 - y0 < 3: return
        X = (s.XX[y0:y1, x0:x1] - c[0]) / R; Y = (s.YY[y0:y1, x0:x1] - c[1]) / R; lid = 0.60 * open_ * np.sqrt(np.clip(1 - (X / 1.12) ** 2, 0, 1)); aa = 2.0 / R
        m = sstep(lid + aa, lid - aa, np.abs(Y)) * (np.abs(X) < 1.12); ix = X - look[0]; iy = Y - look[1]; ri = np.hypot(ix, iy); an = np.arctan2(iy, ix)
        st = cv2.remap(s._tiles[0], ((an / 6.2832 * 768) % 256).astype(np.float32), ((ri * 60) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP); st2 = cv2.remap(s._tiles[2], ((an / 6.2832 * 256 + 40) % 256).astype(np.float32), ((ri * 150 - t * 6) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        ir = sstep(0.54, 0.50, ri); icol = (GLOW * 0.9)[None, None, :] * ((0.25 + 0.9 * st) * (0.35 + 1.0 * np.clip(ri / 0.5, 0, 1) ** 1.5) * (0.8 + 0.4 * st2))[:, :, None] + (np.exp(-((ri - 0.14) / 0.07) ** 2) * 0.5)[:, :, None] * np.array([1.0, 0.95, 0.55], np.float32)[None, None, :]
        icol *= (1 - 0.75 * sstep(0.42, 0.52, ri))[:, :, None]
        scl = np.array([0.03, 0.10, 0.12], np.float32)[None, None, :] * (0.5 + 1.0 * st2)[:, :, None]; col = scl * (1 - ir[:, :, None]) + icol * ir[:, :, None]
        slit = 0.055 + 0.03 * math.sin(t * 1.3); pm = sstep(1.0 + aa * 6, 1.0 - aa * 6, np.hypot(ix / slit, iy / 0.44)); col *= (1 - pm)[:, :, None]
        col += (np.exp(-(((ix + 0.20) / 0.10) ** 2 + ((iy + 0.20) / 0.07) ** 2)) * 0.9 + np.exp(-(((ix - 0.26) / 0.05) ** 2 + ((iy - 0.22) / 0.04) ** 2)) * 0.35)[:, :, None]        # ぬれた てり
        col *= (0.35 + 0.65 * sstep(-0.05, 0.35, (Y + lid) / (lid * 2 + 1e-3)))[:, :, None]                               # 上まぶたの 影
        F = f[y0:y1, x0:x1]; F[:] = F * (1 - m[:, :, None]) + col * m[:, :, None]
        rim = np.exp(-((np.abs(Y) - lid) / (aa * 3 + 0.012)) ** 2) * (np.abs(X) < 1.12); F += (rim * 0.5 + np.exp(-np.clip(np.abs(Y) - lid, 0, None) / 0.16) * (1 - m) * 0.10 * (np.abs(X) < 1.2))[:, :, None] * GLOW[None, None, :]

    def maw(s, f, c, t, lum=1.0):
        """画面の 上下から せり出す 歯の 列(こちらは 口の 中)。c: 0=ひらく 1=かみ合う"""
        W, H, k = s.W, s.H, s.k; arch = H * 0.07 * np.sin(np.pi * s.xn[0]); n = 7
        fl = s.fbm(0.004, t * 6, 0, 3, 20 * k, 1); fl2 = s.fbm(0.012, 0, 0, 2, 0, 3)
        stri = cv2.remap(s._tiles[1], ((s.XX * 2.6 / k) % 256).astype(np.float32), ((s.YY * 0.2 / k) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        for up in (True, False):
            yg = (H * (-0.34 + 0.76 * c) - arch) if up else (H * (1.34 - 0.76 * c) + arch); dv = (s.YY - yg[None, :]) if up else (yg[None, :] - s.YY)
            xi = s.xn * n - (0.0 if up else 0.5); idx = np.floor(xi); fx = xi - idx - 0.5; lt = H * 0.215 * np.where(idx.astype(np.int32) % 2 == 0, 1.0, 0.76); vv = dv / lt
            hw = 0.47 * np.clip(1 - np.clip(vv, 0, 1), 0, 1) ** 0.68; aa = 1.5 * n / W; tm = sstep(hw + aa, hw - aa, np.abs(fx)) * (vv <= 1) * (dv > -6 * k); nx = np.clip(fx / (hw + 1e-3), -1, 1)
            iv = np.array([0.90, 0.86, 0.74], np.float32)[None, None, :] * ((0.16 + 0.66 * np.sqrt(1 - nx * nx)) * (0.8 + 0.35 * stri) * (0.50 + 0.50 * sstep(0.0, 0.16, vv)))[:, :, None]
            rt = np.exp(-np.clip(vv, 0, None) / 0.22)[:, :, None]; iv = iv * (1 - 0.5 * rt) + np.array([0.42, 0.32, 0.20], np.float32)[None, None, :] * 0.5 * rt * (0.4 + 0.6 * np.sqrt(1 - nx * nx))[:, :, None]
            iv += (np.exp(-((nx + 0.42) / 0.13) ** 2) * 0.30 * (1 - np.clip(vv, 0, 1)))[:, :, None] + (np.clip(vv, 0, 1) ** 2 * 0.22)[:, :, None] * GLOW[None, None, :]
            gm = sstep(3 * k, -3 * k, dv); deepv = np.exp(np.clip(dv, None, 0) / (H * 0.22))
            fc = np.array([0.20, 0.035, 0.085], np.float32)[None, None, :] * ((0.25 + 1.1 * fl) * deepv ** 1.6 * (0.80 + 0.35 * np.exp(-(fx / 0.42) ** 2) * np.exp(np.clip(dv, None, 0) / (H * 0.05))))[:, :, None] + (sstep(0.62, 0.8, fl2) * deepv * 0.14)[:, :, None] * np.array([1.0, 0.8, 0.85], np.float32)[None, None, :]
            f[:] = f * (1 - gm[:, :, None]) + fc * lum * gm[:, :, None]; f[:] = f * (1 - tm[:, :, None]) + iv * lum * tm[:, :, None]

    def scales(s, f, u, cyb):
        """うろこ(式)が 下の 段から かぶさる。すき間(cyb の 帯)は のこす"""
        W, H, k = s.W, s.H, s.k; R = W * 0.085; sp = R * 0.9; rows = int(H / sp) + 2; best = np.zeros((H, W), np.float32); dd = np.zeros((H, W), np.float32); lx = np.zeros((H, W), np.float32)
        j0 = np.floor((H - s.YY) / sp)
        for dj in (0, 1):                                                            # 下の 段 → 上の 段(上が 勝つ)
            j = j0 + dj; yc = H - j * sp; a = np.clip((u - (rows - j) * 0.012) / 0.18, 0, 1); rr = R * 1.05 * a * a * (3 - 2 * a)
            xo = np.where(j.astype(np.int32) % 2 == 1, R, 0.0); xc = np.round((s.XX - xo) / (2 * R)) * 2 * R + xo; d = np.hypot(s.XX - xc, s.YY - yc)
            ok = (d < rr) & (s.YY >= yc) & (np.abs(yc - cyb) >= H * 0.10); best = np.where(ok, 1.0, best); dd = np.where(ok, d / (rr + 1e-3), dd); lx = np.where(ok, (s.XX - xc) / R, lx)
        tx = cv2.remap(s._tiles[2], ((s.XX * 1.5 / k) % 256).astype(np.float32), ((s.YY * 1.5 / k) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        col = (AQUA * 0.20 + ABYSS * 2.2)[None, None, :] * ((0.35 + 0.9 * dd ** 2) * (0.7 + 0.6 * tx) * (0.7 + 0.5 * (1 - s.yn)))[:, :, None] + (np.exp(-((dd - 0.96) / 0.05) ** 2) * 0.55 + np.exp(-(lx / 0.06) ** 2) * 0.10 * dd)[:, :, None] * (AQUA * 0.6 + GLOW * 0.4)[None, None, :]
        col += (sstep(0.6, 0.9, tx) * dd * 0.10)[:, :, None] * np.array([0.6, 0.4, 1.0], np.float32)[None, None, :]
        m = cv2.GaussianBlur(best, (0, 0), 0.7)[:, :, None]; f[:] = f * (1 - m) + col * m

    def frame(s, t):
        o = {}; s._rd2 = (0.2, -0.98)
        if t < T_CUT: f, sh = s.sc_sink(t); f = f * clamp(t / 0.35)
        elif t < T_AB: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.4, thr=0.9)
        else: f, sh = s.sc_abyss(t)
        return s.film(f, t, sh * s.k, 0.5, o.get("thr", 0.78), (0.85, 0.97, 1.0), 1.0 + 0.15 * sh, 0.045, o.get("vig", 0.6), 1.12, 1.15)

    # 2) うろこが おおう カットイン
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT; cyb = H * 0.34
        win = s.nebula(t, [(0.02, 0.10, 0.22), (0.06, 0.30, 0.50), (0.4, 0.9, 1.0)], 0.0015, (18.0, 4.0), 0.9, 1) + (AQUA * 0.16 + ABYSS)[None, None, :]
        ex = (s.hL[0] + s.eyeR[0]) / 2; ey = (s.hL[1] + s.eyeL[1]) / 2; zf = 1.35; sc = zf * H / s.ah
        img = s.pose(t); s.draw(win, img, W * 0.5 - (ex - s.aw / 2) * sc + u * 5 * k, cyb - (ey - s.ah / 2) * sc, zf); s.helm(win, 0.75 + 0.25 * math.sin(u * 10))
        f = s.deep(t, 0.9, 0.3); slit = sstep(H * 0.10 + 1, H * 0.10 - 1, np.abs(s.YY - cyb))[:, :, None]; f = f * (1 - slit) + win * slit
        s.scales(f, u, cyb)
        for yy in (cyb - H * 0.10, cyb + H * 0.10): s.glowline(f, [(0, yy), (W, yy)], GLOW, 1.5, 0.7, 0.9)
        for i, T in enumerate(s.vt):                                                 # 技名: 大きな 3文字が 泡と いっしょに 浮かび上がる
            a = (u - 0.5 - i * 0.2) / 0.4
            if a > 0:
                p = (W * (0.22 + i * 0.28), H * 0.68 + (1 - ease(a)) * 90 * k); s.text(f, T, p[0], p[1], clamp(a * 1.5), 1.0)
                L2 = s.L0()
                for j in range(5): r = s.rd[i * 5 + j]; cv2.circle(L2, (int(p[0] + (r[0] - 0.5) * 90 * k), int(p[1] - 60 * k - a * 120 * k * (0.5 + r[1]))), max(1, int((3 + 5 * r[2]) * k)), float(clamp(1.5 - a)), 1, cv2.LINE_AA)
                f += (L2 + cv2.GaussianBlur(L2, (0, 0), 2 * k))[:, :, None] * (GLOW * 0.5 + 0.5)[None, None, :]
        for i, T in enumerate(s.vo):
            if u > 0.4 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.53, 1.0, 1.0)
        if u > 1.4: s.text(f, s.txt_en, W / 2, H * 0.68 + 96 * k, ease((u - 1.4) / 0.3), 1.0)
        if u < 0.08: f += (1 - u / 0.08) * 0.7
        return f, 0

    # 3) 底の「目」→ 指図 → 歯が せり出し、吸いこまれて かみ合う → まっくら
    def sc_abyss(s, t):
        W, H, k = s.W, s.H, s.k; eo = t - T_EYE; od = t - T_ORDER; jo = t - T_JAW; sn = t - T_SNAP
        if sn > 0:                                                                   # かみ合った あと: まっくら。ごくり と、のぼる 泡が すこし
            f = np.zeros((H, W, 3), np.float32); v = clamp((sn - 0.25) / 0.2) * clamp((DUR - 0.15 - t) / 0.3)
            r = s.rd[:40]; s.splat(f, W * (0.3 + 0.4 * r[:, 0]) + np.sin(t * 5 + r[:, 3] * 9) * 6 * k, H * (0.8 - 0.5 * ((r[:, 1] + (sn - 0.25) * (0.5 + r[:, 2])) % 1.0)), 0.25 * v * r[:, 4], GLOW * 0.6 + 0.2, 1.2, 0.4)
            return f, 24 * math.exp(-sn * 7) + 8 * math.exp(-abs(sn - 0.32) * 14)
        f = s.deep(t, 1.0, 0.8); cy = H * 0.40; C = (W / 2, cy); pas = ease2(clamp((jo + 0.35) / 0.9)) if jo > -0.35 else 0.0
        E = (W * 0.76 + (W * 0.5 - W * 0.76) * pas, H * 0.20 - H * 1.05 * pas ** 1.5); ER = W * 0.30 * (1 + 3.2 * pas ** 1.5); lk = clamp(od / 0.35) if od > 0 else 0.0
        look = ((-0.10) * (1 - lk), 0.16 * (1 - lk)); s._rd2 = (0.6, -0.8)
        if eo > 0: s.eye2(f, E, ER, back(eo / 0.5, 1.2), t, look); f += (np.exp(-np.hypot(s.XX - E[0], s.YY - E[1]) / (ER * 1.2)) * 0.10 * clamp(eo / 0.5))[:, :, None] * GLOW[None, None, :]
        suck = clamp(jo / (T_SNAP - T_JAW)) if jo > 0 else 0.0
        if suck > 0: s.rushdust(f, t, (W / 2, H * 0.45), 1400, 0.5 + 2.2 * suck, GLOW * 0.6 + AQUA * 0.4, 0.3 + 0.6 * suck, 4, 1.0)   # ぜんぶ こちらへ(口の 奥へ)
        L = s.L0(); rr = s.rd[:70]                                                    # 泡(吸われると 外へ ひろがって 通りすぎる)
        for i in range(70):
            r = rr[i]; y = ((r[1] - t * 1.2 * (0.08 + 0.16 * r[2])) % 1.0) * H; x = (r[0] + 0.02 * math.sin(t * 2.4 + r[3] * 9)) * W; z = 1 + 6 * ((suck * (0.6 + r[2]) * 1.6 + r[4]) % 1.0) ** 2 * (suck > 0)
            x = W / 2 + (x - W / 2) * z; y = H * 0.45 + (y - H * 0.45) * z
            if -30 < x < W + 30 and -30 < y < H + 30: cv2.circle(L, (int(x), int(y)), max(1, int((2 + 5 * r[4]) * k * z ** 0.7)), 0.5, 1 if z < 2.5 else 2, cv2.LINE_AA)
        f += (L + cv2.GaussianBlur(L, (0, 0), 2 * k))[:, :, None] * (GLOW * 0.5 + 0.5)[None, None, :]
        rec = 0.08 * math.exp(-od * 7) if od > 0 else 0.0
        img = s.pose(t, lean=(-14 * k * ease(od / 0.2) if od > 0 else 0), squash=rec, jit=(1.2 * k * suck))
        f += (np.exp(-np.hypot(s.XX - C[0], s.YY - C[1]) / (W * 0.30)) * 0.10)[:, :, None] * AQUA[None, None, :]
        s.draw(f, img, W / 2, cy, 0.40); s.helm(f, 1.0)
        if od > 0:                                                                   # 槍の 先から こちらへ 指図の 光
            tp = s.at(*s.tip); g = ease(od / 0.2); v = clamp(1.6 - od * 1.2)
            if v > 0: s.glowline(f, [tp, (tp[0] + (W * 0.5 - tp[0]) * g * 0.6, tp[1] + (H * 1.05 - tp[1]) * g)], GLOW, 1.5 + 4 * g, v, 1.0); s.flare(f, tp, 14 * k, GLOW, v, 0.8, 0.5, True)
            if od < 0.08: f += (1 - od / 0.08) * 0.5
        if jo > 0:
            c = 0.58 * ease2(clamp(jo / 1.3)) + 0.42 * ease2(clamp((t - T_BITE) / (T_SNAP - T_BITE))) ** 1.5 + 0.012 * math.sin(t * 9) * (t < T_BITE)
            f *= (1 - 0.55 * ease(clamp(jo / 0.8)) * np.clip(np.abs(s.yn - 0.5) * 2, 0, 1) ** 1.5)[:, :, None]; f *= 1 - 0.75 * clamp((t - T_BITE) / (T_SNAP - T_BITE)) ** 2
            s.maw(f, c, t, 0.55 + 0.45 * (1 - clamp((t - T_BITE) / (T_SNAP - T_BITE))) )
        if t - T_AB < 0.2: f *= (t - T_AB) / 0.2
        return f, (7 * math.exp(-eo * 6) if eo > 0 else 0) + 3 * suck + (5 * math.exp(-abs(jo - 1.3) * 6) if jo > 0 else 0) + 10 * clamp((t - T_BITE) / (T_SNAP - T_BITE)) ** 2

    # ---- この子だけの曲: 深い 海の うなり。くじらの 声(ゆっくり 下がる 音)。歯が せり出す 間は 低い 2音が せまる。かみ合って 無音 → ごくり ----
    def audio(s, path):
        a = Snd(DUR, 76); A1 = 55.0; aeo = [0, 2, 3, 5, 7, 8, 10]
        def fq(d, o=0): return A1 * 4 * 2 ** ((aeo[d % 7] + 12 * (d // 7 + o)) / 12.0)
        def whale(t0, f0, f1, dur, g=0.12):
            u = np.clip((a.t - t0) / dur, 0, 1); fr = f0 + (f1 - f0) * u; ph = 2 * np.pi * np.cumsum(fr) / a.sr
            a.a += (np.sin(ph) + 0.4 * np.sin(2 * ph)) * np.sin(u * np.pi) ** 1.5 * ((a.t >= t0) & (a.t < t0 + dur)) * g
        a.a += a.lp(a.nz, 0.008) * 1.2 * a.gate(0, T_SNAP, 0.5)                                                      # 水の 重さ
        a.a += a.tone(A1) * 0.10 * a.gate(0.5, T_SNAP, 0.6)
        for i in range(12): a.note(900 + (i * 137) % 700, 0.3 + i * 0.2, 0.05, 0.03, "sine", 0.002)                  # 泡
        whale(0.8, 420, 260, 1.4, 0.07)
        a.boom(T_CUT, 46, 0.9)
        for i in range(10): a.hat(T_CUT + i * 0.03, 0.04, 0.02)
        for i in range(3): a.note(fq([4, 2, 0][i], 0), T_CUT + 0.5 + i * 0.2, 0.9, 0.10, "bell"); a.boom(T_CUT + 0.5 + i * 0.2, 60, 0.4, 0.2)
        a.boom(T_EYE, 40, 1.0, 0.7); whale(T_EYE, 180, 95, 1.0, 0.16)
        a.note(fq(7, 1), T_ORDER, 0.5, 0.12, "bell"); a.whoosh(T_ORDER, T_ORDER + 0.3, 0.6, 0.4)                       # 指図
        whale(T_JAW - 0.3, 110, 60, 1.6, 0.16)
        tk = T_JAW; i = 0
        while tk < T_SNAP - 0.1:                                                     # 歯: 低い 2音が せまる
            a.note(fq([0, 1][i % 2], -2), tk, 0.3, 0.22, "saw", 0.02); a.boom(tk, 44, 0.45, 0.2); tk += max(0.13, 0.40 - i * 0.035); i += 1
        a.rise(T_JAW, T_SNAP, 70, 900, 0.18); a.whoosh(T_JAW, T_SNAP, 0.9, 0.35)                                       # 吸いこむ 流れ
        a.a += (a.nz - a.lp(a.nz, 0.3)) * np.clip((a.t - T_JAW) / (T_SNAP - T_JAW), 0, 1) ** 2 * (a.t < T_SNAP) * 0.25
        a.a *= np.where(a.t >= T_SNAP, 0.0, 1.0)                                                                       # かみ合う: ぜんぶ 切れる
        a.boom(T_SNAP, 34, 1.8, 0.5); a.a += (a.nz - a.lp(a.nz, 0.5)) * a.env(T_SNAP, 0.001, 0.05) * 0.7; a.note(1500, T_SNAP, 0.04, 0.12, "sq", 0.0005)
        a.boom(T_SNAP + 0.32, 30, 0.9, 0.3); a.rise(T_SNAP + 0.30, T_SNAP + 0.42, 140, 70, 0.20)                        # ごくり
        for i in range(4): a.note(700 + i * 160, T_SNAP + 0.45 + i * 0.07, 0.05, 0.03, "sine", 0.002)
        s.fin_snd(a, DUR, 76); a.save(path)


if __name__ == "__main__":
    r = Bahr()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
