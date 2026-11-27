"""カグヤ(★5 月の姫) 専用の必殺技「月光ノ舞」— 高品質版(hs_fx)・通し直し。敵は さいしょから 画面の こちら側。月が こちらへ 落ちてくる
夜の竹やぶ(もや・ほたる)。まん中の 1本の 節が 金色に 光る → カッ。竹が ななめに 切れて 光の中から 姫 → 扇が きらり、竹が 内から外へ 次々 切れて 空が ひらけ、姫の ま上に 満月が 出る(前ふり) →
扇形の窓のカットイン → 五つの宝(石の鉢・玉の枝・火ねずみの衣・竜の珠・子安貝)が ひとつずつ 月の まわりに 灯り、五色の 月の輪に なる →
姫が 扇を こちらへ ひとふり = 合図。五色の輪を まとった 満月が 姫を こえて こちらへ 落ちてくる(海と くぼみが 見えるほど 近く、画面を おおう)→ 白 → 暗転
使い方: python3 hs_37.py 出力.mp4 / python3 hs_37.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old37", "/home/claude/tools/pre_fin/hs_37_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = 9.4
GOLD, KAS, TC = old.GOLD, old.KAS, old.TC
T_GLOW, T_CHOP, T_FAN, T_CUT, T_TR = old.T_GLOW, old.T_CHOP, old.T_FAN, old.T_CUT, old.T_TR
T_SW, T_HIT = 7.2, 8.7
NIGHT = [(0.03, 0.04, 0.14), (0.10, 0.06, 0.22), (0.30, 0.24, 0.50)]
IVORY = np.array([1.0, 0.95, 0.80], np.float32)


class Kaguya(FX, old.Kaguya):
    def __init__(s, W=540):
        old.Kaguya.__init__(s, W); s.fx_init(37); H, k = s.H, s.k
        sh = np.ones((H, W), np.float32)
        for (x, w, _, _, tilt) in s.st + [s.mid]:                                         # 竹の まるみ(左上から 月明かり)
            q = (s.XX - (x + tilt * (H - s.YY))) / (w / 2 + 1.5 * k); ins = np.abs(q) < 1.0
            v = 0.30 + 1.1 * np.sqrt(np.clip(1 - q * q, 0, 1)) * (0.75 - 0.35 * q) + 0.9 * np.exp(-((q + 0.45) / 0.16) ** 2)
            sh = np.where(ins, v * (0.85 + 0.3 * s._tiles[1][(s.YY.astype(np.int32) // 2) % 256, int(x) % 256]), sh)
        s.bsh = sh.astype(np.float32)
        rg = np.random.RandomState(371); P = []
        while len(P) < 16:
            x, y = rg.rand() * 2 - 1, rg.rand() * 2 - 1
            if x * x + y * y < 0.8: P.append((x, y, 0.04 + 0.13 * rg.rand() ** 2))
        while len(P) < 52:
            x, y = rg.rand() * 2 - 1, rg.rand() * 2 - 1
            if x * x + y * y < 0.9: P.append((x, y, 0.012 + 0.03 * rg.rand()))
        s.crat = P; s.MC = (W / 2, H * 0.25); s.MR = W * 0.20

    # ---- 空・月 ----
    def sky(s, t, open_=1.0):
        H, W, k = s.H, s.W, s.k
        f = (np.array([0.012, 0.018, 0.07], np.float32)[None, None, :] * (1 - s.yn)[:, :, None] + np.array([0.07, 0.055, 0.17], np.float32)[None, None, :] * s.yn[:, :, None]).astype(np.float32)
        f += s.nebula(t, NIGHT, 0.0007, (4.0, 1.0), 0.5); s.starfield(f, t, 900, (0.85, 0.9, 1.0), 2, 0.9 * (0.4 + 0.6 * open_), (1.0, 0.0))
        if open_ > 0.01:                                                             # 金の すやり霞(横に たなびく)
            b = cv2.remap(s._tiles[2], ((s.XX * 0.10 + t * 3) % 256).astype(np.float32), ((s.YY * 0.9) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            b2 = s.fbm(0.0012, t * 6, 0, 3, 0, 3); band = sstep(0.58, 0.80, b * 0.7 + b2 * 0.4) * sstep(0.05, 0.25, s.yn) * sstep(0.80, 0.55, s.yn)
            f += (band * 0.30 * open_)[:, :, None] * KAS[None, None, :]
        g = sstep(s.flr + H * 0.012, s.flr + H * 0.03, s.YY); f = f * (1 - 0.8 * g)[:, :, None] + (g * (0.5 + 0.5 * s.fbm(0.004, 3, 1, 3, 0, 1)))[:, :, None] * np.array([0.012, 0.035, 0.022], np.float32)[None, None, :]
        return f

    def luna(s, f, c, R, t, v=1.0, halo=1.0, gain=1.0):
        """満月(式で 描く: 海・くぼみ・ふちの かげ・光の かさ)"""
        if v <= 0.01: return
        k = s.k; X = (s.XX - c[0]) / R; Y = (s.YY - c[1]) / R; d2 = X * X + Y * Y; d = np.sqrt(d2); m = sstep(1.0, 1.0 - 1.5 / R, d) * v; z = np.sqrt(np.clip(1 - d2, 0, 1))
        rm = lambda T, sc_, o: cv2.remap(s._tiles[T], ((X * sc_ + o) % 256).astype(np.float32), ((Y * sc_ + o * 0.7) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        n1 = rm(0, 34.0, 128.0); n2 = rm(2, 80.0, 60.0); n3 = rm(1, 210.0, 10.0)
        alb = 1.0 - 0.42 * sstep(0.46, 0.62, n1 * 0.7 + n2 * 0.3) + 0.14 * (n3 - 0.5) + 0.10 * (n2 - 0.5)
        if R > 200 * k: alb += 0.16 * (rm(3, 620.0, 33.0) - 0.5) + 0.12 * (rm(0, 1500.0, 77.0) - 0.5)
        for (cx_, cy_, rc) in s.crat:
            if rc * R < 2.5: continue
            ex = (X - cx_) / rc; ey = (Y - cy_) / rc; dd = np.sqrt(ex * ex + ey * ey); rim = np.exp(-((dd - 1) / 0.17) ** 2)
            alb += rim * (0.08 + 0.24 * np.clip((ex + ey) * 0.7, -1, 1)) - sstep(1.0, 0.8, dd) * 0.10
            if rc > 0.11: alb += np.exp(-((np.mod(np.arctan2(ey, ex) * 7, 6.2832) - 3.1416) / 0.5) ** 2) * sstep(1.0, 1.6, dd) * sstep(4.5, 1.6, dd) * 0.06       # 光の すじ
        col = IVORY[None, None, :] * (alb * (0.50 + 0.62 * z ** 0.55))[:, :, None] + (np.clip(1 - z, 0, 1) ** 3 * 0.10)[:, :, None] * np.array([1.0, 0.7, 0.3], np.float32)[None, None, :]
        f[:] = f * (1 - m[:, :, None]) + col * gain * m[:, :, None]
        out = np.clip(d - 1, 0, None) * R; hl = (np.exp(-out / (14 * k + R * 0.10)) * 0.55 + np.exp(-out / (60 * k + R * 0.6)) * 0.22) * (1 - sstep(1.0, 1.0 - 1.5 / R, d)) * v * halo
        f += hl[:, :, None] * (IVORY * 0.8 + 0.2 * GOLD)[None, None, :]

    def ring(s, f, c, R, t, n=5, v=1.0, w=1.6):
        """五色の 月の輪(宝が 灯った分だけ)"""
        for j in range(n):
            a0 = t * 0.7 + j * 1.2566; P = [(c[0] + math.cos(a0 + q * 1.2566 / 14) * R, c[1] + math.sin(a0 + q * 1.2566 / 14) * R * 0.34) for q in range(15)]
            s.glowline(f, P, TC[j], w * 1.8, 0.8 * v, 1.4)

    def tpos2(s, i, t, c, R):
        a = t * 0.7 + i * 1.2566 + 0.63; return (c[0] + math.cos(a) * R, c[1] + math.sin(a) * R * 0.34), math.sin(a)

    # ---- 竹やぶ ----
    def grove(s, f, t, wave=None, glow=0.0, mid=True, lit=0.0):
        W, H, k = s.W, s.H, s.k; g = np.zeros_like(f); cuts = []
        for st in s.st:
            tc = None if wave is None else wave + abs(st[0] - W / 2) / W * 1.7
            c = old.Kaguya.stalk(s, g, st, t, tc, 0.35 + lit)
            if c: cuts.append((c, t - tc, st[1]))
        hot = None
        if mid:
            c = old.Kaguya.stalk(s, g, s.mid, t, T_CHOP, 0.35 + lit)
            if c: cuts.append((c, t - T_CHOP, s.mid[1]))
            elif glow > 0: hot = glow
        m = sstep(0.004, 0.02, g.max(2)); f[:] = f * (1 - m[:, :, None]) + g * (s.bsh * 2.3 * getattr(s, '_gb', 1.0))[:, :, None] * m[:, :, None]
        if hot:                                                                      # 光る 節
            x, w, _, hy, _ = s.mid; L = s.L0(); cv2.rectangle(L, (int(x - w / 2), int(hy + 4 * k)), (int(x + w / 2), int(hy + 92 * k)), 1.0, -1)
            s.glowmask(f, L, GOLD, 0.9 * hot, 0.8, 1.2, 16.0); s.flare(f, (x, hy + 48 * k), 9 * k * hot, GOLD * 0.6 + 0.4, 0.6 * hot, 0.6, 0.0, False)
        for (c, v, w) in cuts:                                                       # 切り口が 金に 灯る
            L = s.L0(); cv2.ellipse(L, (int(c[0]), int(c[1])), (int(w * 0.55), int(w * 0.36)), -35, 0, 360, 1.0, -1, cv2.LINE_AA)
            s.glowmask(f, L, GOLD, 0.55 + 0.6 * math.exp(-v * 6), 0.9, 0.8, 8.0)
            if v < 0.12: s.glowline(f, [(c[0] - w * 2.2, c[1] + w * 1.5), (c[0] + w * 2.2, c[1] - w * 1.5)], WHITE, 1.2, 1 - v / 0.12, 0.8)
        mist = s.fbm(0.0018, t * 12, 0, 3, 30 * k, 2) * np.exp(-((s.YY - s.flr) / (H * 0.10)) ** 2)
        f += (mist * 0.22)[:, :, None] * np.array([0.5, 0.7, 0.6], np.float32)[None, None, :]
        return cuts

    def fireflies(s, f, t, v=1.0):
        s.starfield(f, t, 220, (0.7, 1.0, 0.45), 4, 1.5 * v, (4.0, -7.0))

    def hero(s, f, t, alpha=1.0, shade=0.93, rim_v=0.8, **kw):
        img = s.pose(t, **kw); s.draw_lit(f, img, s.CX, s.CY, s.HF, rim=IVORY, rim_dir=(0.0, -1.0), rim_v=rim_v * alpha, amb=(0.85, 0.85, 1.0), amb_v=0.12, alpha=alpha, shade=shade)

    def glint(s, f, v, t):
        if v <= 0.01: return
        p = s.at(*s.fan); s.flare(f, p, 10 * s.k * v, GOLD * 0.5 + 0.5, min(1.2, v), 1.5, 0.2, v > 0.9)

    def frame(s, t):
        s._t = t; o = {}
        if t < T_CUT: f, sh = s.sc_grove(t)
        elif t < T_TR: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.4, thr=0.9)
        else: f, sh = s.sc_moon(t)
        f, sh = s.finale(f, sh, t, DUR, s.moon, (1, 0.84, 0.42), (0.75, 0.45, 1), -1)
        return s.film(f, t, sh * s.k, 0.55, o.get("thr", 0.84), (1.0, 0.93, 0.78), 1.0 + 0.15 * sh, 0.035, o.get("vig", 0.55), 1.12, 1.05)

    # 1) 竹やぶ → 光る竹 → カッ → 姫 → 竹が 次々 切れて 月が 出る
    def sc_grove(s, t):
        W, H, k = s.W, s.H, s.k; c = t - T_CHOP; fn = t - T_FAN; op = ease(fn / 1.2) if fn > 0 else 0.0
        f = s.sky(t, op); s.luna(f, s.MC, s.MR, t, 1.0, 0.35 + 0.65 * op)
        f *= (0.30 + 0.70 * op)                                                      # 竹に かくれて 暗い → ひらけると 月明かり
        if c > 0: s.flare(f, (s.CX, s.CY), 26 * k * math.exp(-c * 3), GOLD * 0.6 + 0.4, 0.8 * math.exp(-c * 2.5), 1.0, 0.0, False)
        s._gb = 0.40 + 0.45 * op
        s.grove(f, t, T_FAN + 0.12 if fn > 0 else None, clamp((t - T_GLOW) / 0.7) * (0.75 + 0.25 * math.sin(t * 9)), True, 0.5 * op)
        s.fireflies(f, t, 1.0 - 0.6 * clamp(fn))
        if c > 0.05:
            a = c - 0.05; s.hero(f, t, clamp(a / 0.12), 0.80 + 0.13 * op, 0.4 + 0.5 * op, squash=-0.08 * math.exp(-a * 9) * math.cos(a * 16))
            s.glint(f, math.sin(clamp((fn + 0.12) / 0.4) * 3.1416) * 1.3 if fn > -0.12 else 0, t)
        if op > 0: f = s.godrays(f, s.MC, 0.35 * op, 0.6, 8, 0.86, 0.35)
        if 0 < c < 0.1: f = f + (1 - c / 0.1) * 0.8
        return f * clamp(t / 0.4), 7 * math.exp(-c * 9) if c > 0 else 0

    # 2) 扇形の窓
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.sky(t, 1.0) * 0.5
        g = back(u / 0.32, 1.2); P = (W / 2, H * 0.60); R = W * 0.72; r0 = W * 0.16; a0, a1 = -90 - 62 * g, -90 + 62 * g
        m = s.L0(); cv2.ellipse(m, (int(P[0]), int(P[1])), (int(R), int(R)), 0, a0, a1, 1.0, -1, cv2.LINE_AA); cv2.circle(m, (int(P[0]), int(P[1])), int(r0), 0.0, -1, cv2.LINE_AA)
        f *= (1 - 0.6 * cv2.GaussianBlur(m, (0, 0), 16 * k))[:, :, None]
        win = np.zeros((H, W, 3), np.float32) + np.array([0.10, 0.07, 0.22], np.float32) + s.nebula(t, [(0.20, 0.12, 0.40), (0.45, 0.32, 0.20), (1.0, 0.85, 0.5)], 0.0013, (8.0, -4.0), 0.9)
        s.luna(win, (W * 0.5, H * 0.20), W * 0.34, t, 0.9, 0.8); s.starfield(win, t, 400, GOLD * 0.6 + 0.4, 6, 0.8, (5.0, -16.0))
        zf = 0.80 + 0.04 * ease(u / 1.9); sc = zf * H / s.ah; img = s.pose(t)
        s.draw_lit(win, img, P[0] - (s.fc[0] - s.aw / 2) * sc, H * 0.33 - (s.fc[1] - s.ah / 2) * sc, zf, rim=IVORY, rim_dir=(0.0, -1.0), rim_v=0.8, amb=(1.0, 0.92, 0.8), amb_v=0.10, shade=0.97)
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        for i in range(9):                                                           # 扇の 骨と ふち
            a = math.radians(a0 + (a1 - a0) * i / 8.0); e_ = R if i in (0, 8) else r0 * 1.9
            s.glowline(f, [(P[0] + math.cos(a) * r0 * 0.3, P[1] + math.sin(a) * r0 * 0.3), (P[0] + math.cos(a) * e_, P[1] + math.sin(a) * e_)], GOLD, 1.6 if i in (0, 8) else 1.0, 0.7, 0.6)
        for rr_, w_ in ((R, 2.2), (r0, 1.6)): s.glowline(f, [(P[0] + math.cos(math.radians(a0 + (a1 - a0) * q / 30.0)) * rr_, P[1] + math.sin(math.radians(a0 + (a1 - a0) * q / 30.0)) * rr_) for q in range(31)], GOLD, w_, 0.75, 0.8)
        s.flare(f, P, 7 * k, GOLD * 0.6 + 0.4, 0.8, 0.8, 0.0, False)
        f *= (1 - 0.7 * sstep(0.70, 0.79, s.yn) * sstep(0.96, 0.90, s.yn))[:, :, None]
        n = len(s.ht); step = W * 0.20
        for i, T in enumerate(s.ht):                                                 # 技名: 扇を ひらくように 左右へ
            a = (u - 0.55) / 0.35
            if a > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step * ease(a), H * 0.835, clamp(a * 2), 1.0)
        for i, T in enumerate(s.vo):
            if u > 0.45 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.755, 1.0, 1.0)
        if u > 1.3: s.text(f, s.txt_en, W / 2, H * 0.835 + 62 * k, ease((u - 1.3) / 0.3), 1.0)
        if u < 0.08: f = f + (1 - u / 0.08) * 0.7
        return f, 0

    # 3) 宝が 月の まわりに 灯る → 扇を ひとふり → 月が こちらへ 落ちてくる
    def sc_moon(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_TR; b = t - T_SW; g = clamp(b / (T_HIT - T_SW)); gg = g ** 1.7
        R = s.MR * (1 + 0.10 * clamp(u / 1.4)) * (1 + 7.5 * gg); c = (s.MC[0], s.MC[1] + (H * 0.44 - s.MC[1]) * ease2(g) + 0.0)
        f = s.sky(t, 1.0)
        nt = sum(1 for i in range(5) if u > 0.1 + i * 0.22)
        s._gb = 0.85 + 0.5 * g
        s.grove(f, t, T_FAN + 0.12, 0.0, True, 0.5 + 0.8 * g)
        sw = math.exp(-b * 4.5) if b > 0 else 0.0; pre = clamp((t - (T_SW - 0.3)) / 0.3)
        front = g < 0.16                                                             # 月が 姫の うしろ → こえて 前へ
        def put_hero(): s.hero(f, t, 1.0, 0.93 - 0.5 * g, 0.8 + 0.6 * g, lean=(8 * k * pre if b < 0 else -16 * k * sw), squash=0.04 * sw); s.glint(f, 1.4 * math.exp(-b * 7) if b > 0 else 0.5 * pre, t)
        def put_moon():
            s.luna(f, c, R, t, 1.0, 1.0 - 0.6 * g, 1.0 + 0.7 * g)
            if nt: s.ring(f, c, R * 1.42, t, nt, 1.0, 1.6 + 3.0 * gg)
            for i in range(nt):
                p, dz = s.tpos2(i, t, c, R * 1.42); v = u - (0.1 + i * 0.22); sz = (12 + 6 * dz) * k * (1 + 2.5 * gg) * back(v / 0.2, 1.6)
                if -80 * k < p[0] < W + 80 * k and -80 * k < p[1] < H + 80 * k:
                    s.flare(f, p, min(40 * k, sz * 0.75), TC[i], 0.95, 1.1, 0.0, False)
        if front: put_moon(); put_hero()
        else: put_hero(); put_moon()
        s.fireflies(f, t, 0.4)
        if b > 0:
            s.rushdust(f, t, c, 1500, 0.8 + 1.8 * g, IVORY, 0.3 + 0.9 * g, 7, 1.0); f = s.godrays(f, c, 0.3 * (1 - g), 0.8, 10, 0.88, 0.5)
            if b < 0.08: f = f + (1 - b / 0.08) * 0.5
        else: f = s.godrays(f, c, 0.3, 0.7, 8, 0.86, 0.35)
        if u < 0.1: f = f * (u / 0.1) + (1 - u / 0.1) * 0.5
        return f, (3 + 11 * gg if b > 0 else 0)

    # ---- この子だけの曲: ニ調の 陽旋法。琴・竹を 打つ カッ・笙。宝は 琴が 1音ずつ 上がり、扇の ひとふりの あと 琴の 流し弾きが くり返し 下りてきて(月が 落ちる)、笙が ふくらむ ----
    def audio(s, path):
        a = Snd(DUR, 37); D = 293.66; sc = [0, 2, 5, 7, 9]
        def fq(d, o=0): return D * 2 ** ((sc[d % 5] + 12 * (d // 5 + o)) / 12.0)
        def koto(d, t0, g=0.11, o=0, dec=0.5): a.note(fq(d, o), t0, dec, g, "bell", 0.001); a.note(fq(d, o), t0, 0.04, g * 0.6, "saw", 0.001)
        def knock(t0, g=0.3, fr=820): a.note(fr, t0, 0.035, g, "sine", 0.0005); a.note(fr * 2.3, t0, 0.02, g * 0.4, "sine", 0.0005); a.hat(t0, g * 0.6, 0.01)
        def sho(t0, t1, g=0.022):
            for sm in (9, 14, 16, 21, 23): a.a += (a.tone(D * 2 * 2 ** (sm / 12.0)) + 0.4 * a.tone(D * 4 * 2 ** (sm / 12.0) * 1.002)) * a.gate(t0, t1, 0.5) * g
        a.a += a.lp(a.nz, 0.02) * 0.4 * a.gate(0, T_FAN, 0.4)                                                                    # 夜の やぶ
        for i in range(4): a.note(3600 + 300 * (i % 2), 0.2 + i * 0.27, 0.05, 0.02, "sine")                                       # 虫
        a.rise(T_GLOW, T_CHOP, 400, 900, 0.05); koto(0, 0.6, 0.07, 0, 0.8); koto(2, 1.0, 0.07, 0, 0.6)
        knock(T_CHOP, 0.5, 700); a.rise(T_CHOP, T_CHOP + 0.06, 2000, 6000, 0.15); a.boom(T_CHOP + 0.05, 70, 0.6, 0.2)             # カッ
        for i, d in enumerate((5, 7, 9)): koto(d, T_CHOP + 0.12 + i * 0.09, 0.09)
        a.note(fq(10, 1), T_FAN, 0.5, 0.06, "bell")                                                                             # 扇
        for st in s.st: tc = T_FAN + 0.12 + abs(st[0] - s.W / 2) / s.W * 1.7; knock(tc, 0.22, 600 + st[1] * 9); a.hat(tc + 0.02, 0.1, 0.05)
        sho(T_FAN + 0.3, T_CUT + 0.1, 0.012)
        a.boom(T_CUT, 60, 0.7, 0.25)
        for i, (d, dt) in enumerate(((9, 0.2), (7, 0.45), (5, 0.7), (7, 0.95), (4, 1.2), (5, 1.45))): koto(d, T_CUT + dt, 0.12); koto(d - 5, T_CUT + dt, 0.06)
        for i in range(8): koto(i + 3, T_CUT + 0.55 + i * 0.035, 0.05, 0, 0.2)                                                   # 扇が ひらく
        for i in range(5): koto(5 + i, T_TR + 0.1 + i * 0.22, 0.13, 0, 0.7); a.note(fq(5 + i, 2), T_TR + 0.1 + i * 0.22, 0.3, 0.04, "sine"); a.boom(T_TR + 0.1 + i * 0.22, 90, 0.2, 0.08)   # 宝
        sho(T_TR, T_SW, 0.014)
        a.note(fq(10, 1), T_SW, 0.5, 0.08, "bell"); a.whoosh(T_SW - 0.15, T_SW + 0.05, 0.3, 0.3); knock(T_SW, 0.4, 700)            # 扇を ひとふり(はじめの 扇・カッ と 同じ音)
        tk = T_SW + 0.1; j = 0                                                                                                   # 月が 落ちる: 流し弾きが くり返し 下りてくる・だんだん 低く 速く
        while tk < T_HIT - 0.1:
            for i in range(7): koto(14 - i - j, tk + i * 0.03, 0.06 + 0.012 * j, 0, 0.3)
            a.boom(tk, 80 - 5 * j, 0.3 + 0.12 * j, 0.15); tk += max(0.16, 0.34 - 0.04 * j); j += 1
        sho(T_SW, T_HIT + 0.2, 0.034); a.rise(T_SW, T_HIT, 55, 210, 0.16); a.a += a.lp(a.nz, 0.06) * np.clip((a.t - T_SW) / (T_HIT - T_SW), 0, 1) ** 2 * (a.t < T_HIT) * 0.9
        a.boom(T_HIT, 40, 1.6, 0.5)
        s.fin_snd(a, DUR, 37); a.save(path)


if __name__ == "__main__":
    r = Kaguya()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
