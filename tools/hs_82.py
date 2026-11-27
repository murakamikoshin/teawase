"""ノクス(★7 おわりの夜) 専用の必殺技「終夜ノ葬列」— 高品質版(hs_fx)・通し直し。影の 葬列が こちらへ 歩いて きて、棺の ふたが とじる(黒で おわる)
夕やけに 立つ 黒い 影 → 槍の 石突きで 地を 打つと 日が くわれ、道ばたの 灯が ひとつずつ 消える(前ふり: 光が 消えて いく) → ななめに 裂ける カットイン →
ふちが むらさきに 燃える 黒い 月の 下、ちょうちんを さげた 頭巾の 影の 列が、きりの 奥から こちらへ 歩いて くる(鐘が 鳴るたび 灯が ゆれる) →
列は 目の前で 分かれ、まわりから こちらを 見おろす(あおむけ: 真上に 黒い 月) → 棺の ふたが 左右から とじる。さいごの 光の すじ → まっくら → くぎを 打つ 音 3つ
使い方: python3 hs_82.py 出力.mp4 / python3 hs_82.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old82", "/home/claude/tools/pre_fin/hs_82_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = 11.8
VIO, EMB, BONE, INK = old.VIO, old.EMB, old.BONE, old.INK
T_STRIKE, T_CUT, T_MARCH = old.T_STRIKE, old.T_CUT, old.T_MARCH
T_LOOK, T_LID, T_SHUT = 8.6, 9.6, 10.6
TOLL = [T_MARCH + 0.25 + i * 0.95 for i in range(4)]; NAILS = [T_SHUT + 0.22, T_SHUT + 0.50, T_SHUT + 0.78]
GOLDD = np.array([0.75, 0.58, 0.25], np.float32)


class Nox(FX, old.Nox):
    def __init__(s, W=540):
        old.Nox.__init__(s, W); s.fx_init(82); H = s.H; k = s.k; s.yn = s.yn.astype(np.float32)
        s._cl = s.fbm(0.0010, 0, 0, 4, 70 * k, 0); s._clr = np.clip((s._cl - np.roll(s._cl, int(22 * k), axis=0)) * 2.2, -0.3, 0.45)
        s._gr = s.fbm(0.004, 0, 0, 3, 0, 2)
        s._wd = cv2.remap(s._tiles[1], ((s.XX * 2.4 / k) % 256).astype(np.float32), ((s.YY * 0.2 / k) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        s._cf = cv2.remap(s._tiles[3], ((s.XX * 1.8 / k) % 256).astype(np.float32), ((s.YY * 0.35 / k) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 夕やけ → おわりの 夜 ----
    def sky(s, night):
        y = s.yn; hz_ = 0.80; g = np.clip(1 - np.abs(y - hz_) / 0.55, 0, 1) ** 2; cl = sstep(0.35, 0.8, s._cl) * sstep(0.85, 0.3, y)
        day = np.array([0.16, 0.07, 0.22], np.float32)[None, None, :] * ((1 - y) * (0.8 + 0.5 * s._cl))[:, :, None] + (g * (0.80 + 0.35 * s._cl))[:, :, None] * EMB[None, None, :] * 0.95
        day = day * (1 - 0.45 * cl)[:, :, None] + (cl * np.clip(s._clr, 0, 1) * 1.6 * g)[:, :, None] * (EMB * 0.9 + 0.1)[None, None, :]                # 雲の ふちが 夕日に 光る
        ngt = INK[None, None, :] * (0.5 + 1.2 * s._cl)[:, :, None] + (g * 0.20 * (0.6 + 0.8 * s._cl))[:, :, None] * VIO[None, None, :] + (cl * np.clip(s._clr, 0, 1) * 0.10)[:, :, None] * VIO[None, None, :]
        return (day * (1 - night) + ngt * night).astype(np.float32)

    def ground(s, f, y0):
        g0 = int(y0); H = s.H; near = np.clip((s.YY[g0:] - y0) / (H - y0), 0, 1); f[g0:] = (INK * 0.5)[None, None, :] * (0.4 + 1.2 * s._gr[g0:])[:, :, None] * (0.7 + 0.6 * near)[:, :, None]
        f[max(0, g0 - 30):min(H, g0 + 40)] += (np.exp(-((s.YY[max(0, g0 - 30):min(H, g0 + 40)] - y0) / (7 * s.k)) ** 2) * 0.05)[:, :, None] * VIO[None, None, :]

    def eyes(s, f, v, streak=1.0):
        if v <= 0.01: return
        old.Nox.eyes(s, f, min(1.0, v), streak); sc = s._pl[2]
        for e in (s.eyeL, s.eyeR): s.flare(f, s.at(*e), 7 * sc * 2, VIO * 0.8 + 0.2, 0.45 * min(1.2, v), 0.7, -0.2, False)

    def flames(s, f, p, t, power=1.0, n=60):
        """槍の 穂先に まとわりつく 黒むらさきの 炎(fbm)"""
        if power <= 0.02: return
        k = s.k; w = 26 * k * (0.6 + 0.4 * power); hh = 170 * k * power; up = np.clip((p[1] + 30 * k - s.YY) / hh, 0, 1)
        m = np.exp(-((s.XX - p[0] - np.sin(s.YY * 0.03 + t * 5) * 8 * k * up) / (w * (1 - 0.6 * up) + 2)) ** 2) * sstep(0.0, 0.08, up) * (1 - up) ** 1.2
        I = s.flame(t, m, 320.0, 0.0045, 1.5)
        f *= (1 - 0.55 * np.clip(I * 1.5, 0, 1))[:, :, None]; f += (I * 1.5)[:, :, None] * VIO[None, None, :] + (I ** 3 * 1.2)[:, :, None] * BONE[None, None, :]

    def moon(s, f, O, R, t, crack=0.0):
        """黒い 月: ふちが むらさきに 燃える"""
        d = np.hypot(s.XX - O[0], s.YY - O[1]); an = np.arctan2(s.YY - O[1], s.XX - O[0])
        st = cv2.remap(s._tiles[0], ((an / 6.2832 * 512) % 256).astype(np.float32), ((d * 0.9 / s.k - t * 22) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        st2 = cv2.remap(s._tiles[2], ((an / 6.2832 * 1024 + t * 6) % 256).astype(np.float32), ((d * 2.0 / s.k - t * 40) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        out = np.clip(d - R, 0, None); rim = np.exp(-out / (R * (0.10 + 0.22 * st))) * (0.35 + 1.1 * st * st2) * (d > R * 0.98) + np.exp(-((d - R) / (R * 0.035)) ** 2) * 1.3
        f += rim[:, :, None] * (VIO * 0.85 + BONE * 0.15)[None, None, :] + (np.exp(-out / (R * 1.6)) * 0.10 * (d > R))[:, :, None] * VIO[None, None, :]
        m = sstep(R + 1.2, R - 1.2, d); f *= (1 - m)[:, :, None]; f += (m * 0.012 * (0.5 + st))[:, :, None] * VIO[None, None, :]

    def hood(s, f, P, dr, h, lampv=0.0, fog=0.0, t=0.0, lamp_side=1, lamp_r=15.0, rimv=0.16):
        """頭巾の 影 ひとり(顔は 描かない)。P=頭巾の 先、dr=体の のびる 向き、h=身の たけ。ちょうちんつき"""
        W, H, k = s.W, s.H, s.k; nx, ny = -dr[1], dr[0]; ex = (P[0] + dr[0] * h, P[1] + dr[1] * h); pad = h * 0.55
        x0 = int(max(0, min(P[0], ex[0]) - pad)); x1 = int(min(W, max(P[0], ex[0]) + pad)); y0 = int(max(0, min(P[1], ex[1]) - pad)); y1 = int(min(H, max(P[1], ex[1]) + pad))
        if x1 - x0 < 3 or y1 - y0 < 3: return None
        dx = s.XX[y0:y1, x0:x1] - P[0]; dy = s.YY[y0:y1, x0:x1] - P[1]; v = (dx * dr[0] + dy * dr[1]) / h; u = (dx * nx + dy * ny) / h
        w = 0.02 + 0.085 * sstep(0.0, 0.13, v) - 0.012 * sstep(0.15, 0.22, v) + 0.085 * sstep(0.22, 0.34, v) + 0.05 * sstep(0.34, 1.0, v) + 0.008 * np.sin(v * 26 + P[0] * 0.05) * sstep(0.5, 1.0, v); aa = 1.3 / h
        m = sstep(w + aa, w - aa, np.abs(u)) * (v >= 0) * sstep(1.02, 0.98, v); cf = s._cf[y0:y1, x0:x1]
        fogc = np.array([0.10, 0.06, 0.17], np.float32); base = INK * 0.35 * (1 - fog) + fogc * fog
        col = base[None, None, :] * (0.6 + 0.7 * cf)[:, :, None] + (np.exp(-(w - np.abs(u)) / 0.02) * (0.35 + 0.65 * sstep(0.6, 0.0, v)) * rimv * (1 - fog))[:, :, None] * VIO[None, None, :]
        col *= (1 - 0.7 * np.exp(-(np.hypot(dx - dr[0] * h * 0.11, dy - dr[1] * h * 0.11) / (h * 0.05)) ** 2))[:, :, None]       # 頭巾の 中の 闇
        F = f[y0:y1, x0:x1]; F[:] = F * (1 - m[:, :, None]) + col * m[:, :, None]
        if lampv > 0.01:                                                             # ちょうちん(和紙の 光・ほね)
            sw = math.sin(t * 2.3 + P[0] * 0.01) * 0.10; lx = P[0] + dr[0] * h * 0.50 + nx * h * 0.30 * lamp_side + sw * h * 0.2; ly = P[1] + dr[1] * h * 0.50 + ny * h * 0.30 * lamp_side + h * 0.02
            rr = min(h * 0.055, lamp_r * k); X2 = (s.XX - lx) / rr; Y2 = (s.YY - ly) / (rr * 1.35); dl = np.hypot(X2, Y2); lm = sstep(1.0 + 2 / rr, 1.0 - 2 / rr, dl)
            pc = (EMB * 0.95 + VIO * 0.20)[None, None, :] * ((0.55 + 0.6 * np.clip(1 - dl, 0, 1)) * (0.90 + 0.10 * np.cos(Y2 * 9)) * lampv * 1.5)[:, :, None]
            f[:] = f * (1 - lm[:, :, None]) + pc * lm[:, :, None]
            f += (np.exp(-np.hypot(s.XX - lx, s.YY - ly) / (rr * 3.0)) * 0.30 * lampv * (1 - 0.5 * fog))[:, :, None] * (EMB * 0.6 + VIO * 0.5)[None, None, :]
            return (lx, ly)
        return None

    def chochin(s, f, lx, ly, rr, v, t):
        """ちょうちん(和紙の 光・ほね・つりひも)"""
        if v <= 0.01: return
        X2 = (s.XX - lx) / rr; Y2 = (s.YY - ly) / (rr * 1.35); dl = np.hypot(X2, Y2); lm = sstep(1.0 + 2 / rr, 1.0 - 2 / rr, dl); fl = 1 + 0.12 * math.sin(t * 11 + lx)
        pc = (EMB * 1.0 + VIO * 0.15)[None, None, :] * ((0.9 + 1.5 * np.clip(1 - dl, 0, 1) ** 1.5) * (0.88 + 0.12 * np.cos(Y2 * 9)) * v * fl)[:, :, None] + (np.clip(1 - dl, 0, 1) ** 3 * 1.2 * v)[:, :, None]
        f[:] = f * (1 - lm[:, :, None]) + pc * lm[:, :, None]
        d = np.hypot(s.XX - lx, s.YY - ly); f += ((np.exp(-d / (rr * 2.2)) * 0.55 + np.exp(-d / (rr * 7)) * 0.16) * v * fl)[:, :, None] * (EMB * 0.8 + VIO * 0.3)[None, None, :]
        s.glowline(f, [(lx, ly - rr * 1.35), (lx, ly - rr * 2.6)], EMB * 0.5, 1.0, 0.25 * v, 0.3)

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(VIO * 0.8 + 0.2), rim_dir=(0.0, -1.0), rim_v=0.75 * alpha, amb=(0.6, 0.5, 1.0), amb_v=0.14, aura=VIO, aura_v=0.10, alpha=alpha, shade=0.86)

    def add(s, f, L, col, gain=1.0, b1=3, b2=12):
        s.glowmask(f, np.clip(L, 0, 1), col, gain * 0.8, 1.0, 1.0, float(b2))

    def frame(s, t):
        o = {}
        if t < T_CUT: f, sh = s.sc_dusk(t)
        elif t < T_MARCH: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.9)
        elif t < T_LOOK + 0.3: f, sh = s.sc_proc(t)
        else: f, sh = s.sc_look(t)
        if T_LOOK <= t < T_LOOK + 0.3:                                               # あおむけに たおれる: 景色が 下へ 流れる
            q = ease2((t - T_LOOK) / 0.3); M = np.float32([[1, 0, 0], [0, 1, s.H * 1.05 * q]]); f = cv2.warpAffine(f, M, (s.W, s.H)); f = cv2.GaussianBlur(f, (0, 0), sigmaX=0.1, sigmaY=1 + 30 * s.k * q * (1 - q) * 4)
        return s.film(f, t, sh * s.k, 0.55, o.get("thr", 0.78), (0.92, 0.88, 1.0), 1.0 + 0.15 * sh, 0.045, o.get("vig", 0.65), 1.12, 1.15)

    # 3) 黒い 月の 下、影の 葬列が こちらへ 歩いて くる
    def sc_proc(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_MARCH; tl = min([abs(t - x) if t >= x else 9 for x in TOLL]); pulse = math.exp(-tl * 3.5)
        f = s.sky(1.0) * (1 + 0.8 * pulse); gy = H * 0.86; O = (W / 2, H * 0.22); s.moon(f, O, W * 0.20, t)
        L = s.L0()
        for x in TOLL:                                                               # 鐘の 波
            v = (t - x) / 1.2
            if 0 < v < 1: cv2.circle(L, (int(O[0]), int(O[1])), int(W * (0.24 + 0.9 * v)), float((1 - v) * 0.6), max(1, int(2 * k)), cv2.LINE_AA)
        s.glowmask(f, L, VIO, 0.6, 1.0, 1.0, 9.0)
        s.ground(f, gy); mist = s.fbm(0.0020, t * 14, 0, 3, 0, 3)
        prog = u / (T_LOOK - T_MARCH); fig = []
        for i in range(12):                                                          # 2列で 奥から 手前へ(とぎれずに 流れる)
            side = 1 if i % 2 else -1; q = ((i // 2) / 6.0 + (0.08 if side > 0 else 0.0) + prog * 0.75) % 1.0; h = H * (0.07 + 1.0 * q ** 2.2)
            x = W / 2 + side * W * (0.11 + 0.58 * q ** 1.5); y = H * 0.635 + H * 0.66 * q ** 1.6 + abs(math.sin(u * 3.2 + i * 1.3)) * h * 0.02
            fig.append((1 - q, x, y, h, side, i))
        img = s.pose(t, sway=0.35 * pulse, breath=0.3, squash=0.012 * pulse); drawn = False
        for z, x, y, h, side, i in sorted(fig, key=lambda q: -q[0]):
            if not drawn and z < 0.62:                                               # ノクスは 列の まん中に 立つ
                f += (np.exp(-np.hypot(s.XX - W / 2, s.YY - H * 0.66) / (W * 0.28)) * (0.10 + 0.10 * pulse))[:, :, None] * VIO[None, None, :]
                s.draw(f, img, W / 2, gy - H * 0.20 + 5 * k, 0.41); s.eyes(f, 0.55 + 0.45 * pulse, 0.8); s.flames(f, s.at(*s.tip), t, 1.0 + 0.6 * pulse); drawn = True
            s.hood(f, (x + pulse * h * 0.03 * side, y - h), (0.0, 1.0), h, 0.0, clamp(z * 1.15) * 0.9, t, side)
        if not drawn: s.draw(f, img, W / 2, gy - H * 0.20 + 5 * k, 0.41); s.eyes(f, 0.55 + 0.45 * pulse, 0.8); s.flames(f, s.at(*s.tip), t, 1.0 + 0.6 * pulse)
        for z, x, y, h, side, i in fig:                                              # ちょうちんは 影の 手前に ともる
            s.chochin(f, x - side * h * 0.26 + math.sin(t * 2.3 + i) * h * 0.02, y - h * 0.50, min(h * 0.055, 15 * k), (0.8 + 0.4 * pulse) * clamp((1 - z) / 0.08) * clamp((z - 0.30) / 0.12), t)
        f += (mist * np.exp(-((s.yn - 0.70) / 0.10) ** 2) * 0.16)[:, :, None] * (VIO * 0.6 + 0.2)[None, None, :]                                              # きり
        if u < 0.2: f = f * (u / 0.2)
        return f, 2.5 * pulse

    def lid(s, f, c, t):
        """棺の ふたが 左右から とじる(黒い 木目・金の 金具・すき間の 光)"""
        W, H, k = s.W, s.H, s.k; half = W * 0.5 * c; wob = 6 * k * np.sin(s.YY * 0.004 + 1.3)
        dl = half + wob - s.XX; drr = s.XX - (W - half - wob); ins = np.maximum(dl, drr); m = sstep(-1.2, 1.2, ins)
        gapw = max(0.0, W - 2 * half); gl = np.exp(-np.clip(-ins, 0, None) / (10 * k + gapw * 0.25)) * (1 - m)
        f *= (1 - 0.6 * c * (1 - m) * (1 - gl))[:, :, None]; f += (gl * (0.25 + 1.6 * c ** 3))[:, :, None] * (VIO * 0.7 + BONE * 0.4)[None, None, :]              # すき間の 光
        seam = 0.55 + 0.45 * sstep(0.0, 0.05, np.abs(((s.XX + np.where(dl > drr, -half, half)) / (W * 0.16)) % 1.0 - 0.5))
        col = np.array([0.060, 0.038, 0.075], np.float32)[None, None, :] * ((0.35 + 1.1 * s._wd) * seam)[:, :, None]
        col += (np.exp(-np.clip(ins, 0, None) / (16 * k)) * (0.55 * (1 - c ** 6)))[:, :, None] * VIO[None, None, :]                                               # ふちに 当たる 光
        for yy in (H * 0.26, H * 0.74):                                              # 金の 帯 と びょう
            bd = np.exp(-((s.YY - yy) / (9 * k)) ** 2); st = np.exp(-(((s.XX % (W * 0.16)) - W * 0.08) / (5 * k)) ** 2 - ((s.YY - yy) / (5 * k)) ** 2)
            col += (bd * (0.10 + 0.25 * s._wd) + st * 0.35)[:, :, None] * GOLDD[None, None, :] * (0.4 + 0.8 * np.exp(-np.clip(ins, 0, None) / (120 * k)))[:, :, None]
        f[:] = f * (1 - m[:, :, None]) + col * m[:, :, None]

    # 4) まわりから 見おろされる(真上に 黒い 月)→ 棺の ふたが とじる → まっくら → くぎ
    def sc_look(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_LOOK; ld = t - T_LID; sn = t - T_SHUT
        if sn > 0:
            f = np.zeros((H, W, 3), np.float32); kn = sum(math.exp(-(t - x) * 14) for x in NAILS if t >= x)
            if kn > 0.02:
                r = s.fl[:60]; x_ = max([x for x in NAILS if t >= x]); v = t - x_
                s.splat(f, W * (0.2 + 0.6 * r[:, 0]), H * (0.15 + 0.7 * r[:, 1]) + v * H * 0.10 * (0.5 + r[:, 2]), 0.10 * kn * r[:, 3], (0.6, 0.5, 0.7), 1.2, 0.3)             # ふたの すき間から 落ちる ちり
            return f, 10 * kn
        f = s.sky(1.0) * 0.9; O = (W / 2, H * 0.36); s.moon(f, O, W * 0.23, t); s.starfield(f, t, 300, (0.8, 0.75, 1.0), 3, 0.5, (0.0, 0.0))
        a = ease(clamp((u - 0.2) / 0.6)); sl = (1 - a)
        figs = [((W * 0.27, H * 0.17), (-0.80, -0.60), 1), ((W * 0.73, H * 0.17), (0.80, -0.60), -1), ((W * 0.20, H * 0.50), (-1.0, 0.0), 1), ((W * 0.80, H * 0.50), (1.0, 0.0), -1), ((W * 0.25, H * 0.78), (-0.75, 0.66), 1), ((W * 0.75, H * 0.78), (0.75, 0.66), -1)]
        for i, (P, d_, sd) in enumerate(figs):                                       # ふちから のぞきこむ 頭巾
            n_ = math.hypot(*d_); d_ = (d_[0] / n_, d_[1] / n_); bob = math.sin(t * 1.6 + i) * 4 * k; Pp = (P[0] + d_[0] * W * 0.5 * sl + bob * d_[1], P[1] + d_[1] * W * 0.5 * sl - bob * d_[0])
            s.hood(f, Pp, d_, H * 0.62, 0.0, 0.0, t, 1, 26.0, 0.55)
        for lx in (W * 0.30, W * 0.70): s.chochin(f, lx + (lx - W / 2) * 1.6 * sl + math.sin(t * 1.9 + lx) * 6 * k, H * 0.62, 20 * k, 0.9 * a, t)
        img = s.pose(t, breath=0.3); s.draw(f, img, W / 2, H * (1.02 + 0.5 * sl), 0.66); s.eyes(f, 0.9, 0.8); s.flames(f, s.at(*s.tip), t, 1.3)                         # 足もとに 立つ ノクス
        f *= clamp((u - 0.12) / 0.25)
        sh = 0.0
        if ld > 0: c = ease2(clamp(ld / (T_SHUT - T_LID))) ** 0.85; s.lid(f, c, t); sh = 1.5 + 6 * c ** 6
        return f, sh

    # ---- この子だけの曲: 葬送の 行進。低い 鐘、付点の たいこ、オルガン、フリギアの 下降。ふたが とじて 無音 → くぎを 打つ 音 3つ ----
    def audio(s, path):
        a = Snd(DUR, 82); C3 = 130.81; phr = [0, 1, 3, 5, 7, 8, 10]
        def fq(d, o=0): return C3 * 2 ** ((phr[d % 7] + 12 * (d // 7 + o)) / 12.0)
        a.a += a.lp(a.nz, 0.02) * 0.5 * a.gate(0, T_STRIKE, 0.4)                                                    # 夕暮れの 風
        for d in (0, 4, 7): a.note(fq(d), 0.2, 1.6, 0.05, "pad", 0.5)
        a.boom(T_STRIKE, 44, 1.3, 0.6)
        for i in range(7): a.note(fq(11 - i, 1), T_STRIKE + 0.25 + i * 0.16, 0.25, 0.08, "bell")                    # 灯が 消えて いく(下降)
        a.rise(T_STRIKE, T_CUT, 60, 30, 0.2)
        a.boom(T_CUT, 40, 1.1)
        for i in range(5): a.note(fq(0, -1), T_CUT + 0.4 + i * 0.2 + 0.16, 0.5, 0.20, "bell"); a.boom(T_CUT + 0.4 + i * 0.2 + 0.16, 70, 0.35, 0.15)
        for tk in TOLL[:3]:                                                                                          # 鐘
            a.note(fq(0, -1), tk, 1.6, 0.30, "bell"); a.note(fq(4, 0), tk, 1.2, 0.10, "bell"); a.boom(tk, 38, 0.6, 0.5)
        bt = 0.475; tk = T_MARCH; i = 0; mel = [7, 8, 7, 5, 4, 5, 3, 1]
        while tk < T_LOOK - 0.05:                                                                                    # 行進: ドン・(タ)ドン。だんだん 近く
            g = 1 + 0.8 * (tk - T_MARCH) / (T_LOOK - T_MARCH)
            a.boom(tk, 55, 0.35 * g, 0.12); a.boom(tk + bt * 0.75, 62, 0.2 * g, 0.08); a.hat(tk + bt * 0.5, 0.04)
            for d in ((0, 2, 4) if (i // 2) % 2 == 0 else (1, 3, 5)): a.note(fq(d), tk, 0.42, 0.045, "sq", 0.03)       # オルガン
            a.note(fq(mel[i % 8], 1), tk, 0.45, 0.085, "fm"); tk += bt; i += 1
        a.whoosh(T_LOOK, T_LOOK + 0.35, 0.7, 0.4); a.boom(T_LOOK + 0.3, 42, 1.2, 0.5)                                  # あおむけに たおれる
        for d in (0, 1, 4): a.note(fq(d, 0), T_LOOK + 0.35, T_LID - T_LOOK, 0.05, "pad", 0.3)
        a.note(fq(0, -1), T_LOOK + 0.45, 1.6, 0.26, "bell")
        a.rise(T_LID, T_SHUT, 160, 45, 0.22); a.a += a.lp((a.nz - a.lp(a.nz, 0.2)), 0.12) * a.gate(T_LID, T_SHUT, 0.1) * 0.6 * (0.5 + 0.5 * np.sin(a.t * 47))     # ふたが すべる
        for d in (0, 1): a.note(fq(d, -1), T_LID, T_SHUT - T_LID, 0.09, "saw", 0.1)
        a.a *= np.where(a.t >= T_SHUT, 0.0, 1.0)                                                                       # とじる: ぜんぶ 切れる
        a.boom(T_SHUT, 40, 1.8, 0.35); a.a += a.lp(a.nz, 0.4) * a.env(T_SHUT, 0.002, 0.06) * 0.6
        for x in NAILS: a.boom(x, 150, 0.9, 0.05); a.a += (a.nz - a.lp(a.nz, 0.6)) * a.env(x, 0.0005, 0.02) * 0.6; a.note(1250, x, 0.05, 0.10, "sq", 0.0005); a.boom(x + 0.01, 60, 0.5, 0.12)   # くぎ
        s.fin_snd(a, DUR, 82); a.save(path)


if __name__ == "__main__":
    r = Nox()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
