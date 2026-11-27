"""ビャッコ(★5 白虎の槍士) 専用の必殺技「白虎烈爪」— 高品質版(hs_fx)・通し直し。敵は さいしょから 画面の こちら側。白虎の 前足が ふりおろされる
月夜の すすき野原を 風が わたる(穂が 月に 光る)。ひざをついて 動かない → 槍を にぎると 目が 光り、ずっと奥の 闇に 黄色い目が ふたつ ひらく(前ふり) →
ねこの目の形(横長)の 窓のカットイン → 奥から すすきが V字に かきわけられ、月明かりの 白い 毛なみと しま、ふたつの 目だけが 見える 大きな 影が こちらへ 走ってくる →
とびかかって 月を かくし、白い 毛と 肉球・黒い つめの 前足が 上から こちらへ ふりおろされて 目の前を おおう → 白 → 暗転
使い方: python3 hs_69.py 出力.mp4 / python3 hs_69.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old69", "/home/claude/tools/pre_fin/hs_69_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = 9.25
WIND, EYE, SILV = old.WIND, old.EYE, old.SILV
T_GRIP, T_CUT, T_RUN, T_LEAP = old.T_GRIP, old.T_CUT, old.T_RUN, old.T_LEAP
T_PAW, T_HIT = 8.0, 8.55
FUR = np.array([0.86, 0.90, 0.94], np.float32)


class Byakko(FX, old.Byakko):
    def __init__(s, W=540):
        old.Byakko.__init__(s, W); s.fx_init(69); s.MO = (W * 0.74, s.H * 0.15)

    # ---- 月夜 ----
    def night(s, t, moon_v=1.0):
        H, W, k = s.H, s.W, s.k
        f = (np.array([0.006, 0.024, 0.055], np.float32)[None, None, :] * (1 - s.yn)[:, :, None] + np.array([0.04, 0.14, 0.18], np.float32)[None, None, :] * s.yn[:, :, None]).astype(np.float32)
        s.starfield(f, t, 700, (0.85, 0.95, 1.0), 2, 0.8, (0.6, 0.0))
        d = np.hypot(s.XX - s.MO[0], s.YY - s.MO[1]) / (34 * k)
        f += ((sstep(1.0, 0.92, d) * 1.1 + np.exp(-np.clip(d - 1, 0, None) / 0.9) * 0.35 * (d > 1) + np.exp(-d / 6.0) * 0.16) * moon_v)[:, :, None] * np.array([0.92, 0.98, 1.0], np.float32)[None, None, :]
        cl = sstep(0.52, 0.80, s.fbm(0.0012, t * 10, 0, 3, 40 * k, 2)) * sstep(0.55, 0.15, s.yn); f += (cl * 0.10 * (0.5 + 1.5 * np.exp(-d / 5.0)))[:, :, None] * WIND[None, None, :]       # 月に てる うす雲
        f += (np.exp(-((s.yn - 0.58) / 0.06) ** 2) * 0.13)[:, :, None] * WIND[None, None, :]
        g = sstep(H * 0.58, H * 0.60, s.YY); f *= (1 - 0.5 * g)[:, :, None]
        mist = s.fbm(0.002, t * 20, 0, 3, 0, 3) * np.exp(-((s.yn - 0.62) / 0.07) ** 2); f += (mist * 0.16)[:, :, None] * WIND[None, None, :]
        return f

    def field(s, f, t, rows=(0, 1, 2, 3, 4), part=0.0, gust=0.0):
        """すすき(穂が 月に にじんで 光る)。part: まん中から 左右へ かきわける強さ"""
        W, k = s.W, s.k; P = s.L0()
        for (x, y, h, row, r, plume) in s.grass:
            if row not in rows: continue
            bend = 0.32 * math.sin(t * 2.0 + x * 0.011 - row * 0.8 + r) + 0.12 * math.sin(t * 5.3 + r * 9) + gust + part * (1 if x > W / 2 else -1) * math.exp(-abs(x - W / 2) / (W * (0.07 + 0.04 * row))) * 0.55
            bend = max(-0.9, min(0.9, bend)); tip = (x + math.sin(bend) * h, y - math.cos(bend) * h); mid = (x + math.sin(bend * 0.4) * h * 0.55, y - math.cos(bend * 0.4) * h * 0.55); sh = 0.22 + 0.11 * row
            col = (float(SILV[0] * sh * 0.6), float(SILV[1] * sh * 0.8), float(SILV[2] * sh * 0.8))
            cv2.polylines(f, [np.array([(x, y), mid, tip], np.int32)], False, col, max(1, int((1 + row * 0.5) * k)), cv2.LINE_AA)
            if plume: cv2.ellipse(P, (int(tip[0] + math.sin(bend) * h * 0.1), int(tip[1])), (max(1, int(h * 0.22)), max(1, int(h * 0.055))), math.degrees(bend) - 90, 0, 360, float((0.30 + 0.05 * row) * getattr(s, '_fv', 1.0)), -1, cv2.LINE_AA)
        f += (cv2.GaussianBlur(P, (0, 0), 1.2 * k) * 0.9 + cv2.GaussianBlur(P, (0, 0), 7 * k) * 0.45)[:, :, None] * (SILV * 0.9 + 0.1)[None, None, :]

    def rush(s, f, t, v, speed=1.0):
        s.rushdust(f, t, s.VP, 900, speed * 1.2, WIND, 0.7 * v, 3, 1.0)

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        s.draw_lit(f, img, cx, cy, hfrac, rim=(0.9, 0.98, 1.0), rim_dir=(0.5, -0.86), rim_v=0.85 * alpha, amb=(0.7, 0.9, 0.95), amb_v=0.16, alpha=alpha, shade=0.88)

    def add(s, f, L, col, gain=1.0, b1=3, b2=12):
        s.glowmask(f, np.clip(L, 0, 1), col, gain, 0.8, 1.0, float(b2))

    def eyes(s, f, v):
        if v <= 0.01: return
        sc = s._pl[2]
        for e in (s.eL, s.eR): p = s.at(*e); s.flare(f, p, 5 * sc * 2, EYE, 0.55 * min(1.6, v), 0.8 + 0.8 * max(0.0, v - 1), 0.0, False)

    def fareyes(s, f, o):
        k = s.k
        for sg in (-1, 1):
            e = (s.W * 0.24 + sg * 16 * k, s.H * 0.535); L = s.L0(); cv2.ellipse(L, (int(e[0]), int(e[1])), (int(9 * k), max(1, int(5 * k * o))), sg * 20, 0, 360, 1.0, -1, cv2.LINE_AA)
            s.glowmask(f, L, EYE, 1.0 * o, 1.0, 1.0, 8.0)

    # ---- 白虎(月明かりの 影: 毛なみ・しま・目の 光だけ) ----
    def beast(s, f, c, S, t, v=1.0, dark=0.0):
        W, H, k = s.W, s.H, s.k; x0 = int(max(0, c[0] - S * 1.1)); x1 = int(min(W, c[0] + S * 1.1)); y0 = int(max(0, c[1] - S * 0.9)); y1 = int(min(H, c[1] + S * 1.1))
        if x1 - x0 < 3 or y1 - y0 < 3: return
        x = (s.XX[y0:y1, x0:x1] - c[0]) / S; y = (s.YY[y0:y1, x0:x1] - c[1]) / S; ax = np.abs(x)
        fz = cv2.remap(s._tiles[3], ((x * 90 + 128) % 256).astype(np.float32), ((y * 90 + 128 + t * 3) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        head = np.hypot(x / 0.50, y / 0.40); ear = np.where(y < -0.22, np.abs(ax - 0.33) / 0.17 + np.clip(-(y + 0.26), 0, 9) / 0.34, 2.0); body = np.hypot(x / 0.95, (y - 0.50) / 0.50) * 1.02
        dd = np.minimum(np.minimum(head, ear), body) + (fz - 0.5) * 0.12; m = sstep(1.0, 0.86, dd) * v
        fur = cv2.remap(s._tiles[0], ((np.arctan2(y, x) * 80) % 256).astype(np.float32), ((np.hypot(x, y) * 26) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        stripe = sstep(0.55, 0.85, np.sin(y * 13 + ax * 6 + (fz - 0.5) * 5)) * sstep(0.10, 0.22, ax) * (head > 0.35)
        top = np.clip(0.9 - (y + 0.45) * 0.9, 0.05, 1.0)                               # 上から 月明かり
        col = FUR[None, None, :] * ((0.035 + 0.50 * top ** 2.2) * (0.6 + 0.7 * fur) * (1 - 0.9 * stripe) * (1 - dark))[:, :, None] + (sstep(0.78, 1.0, dd) * top ** 1.5 * 0.55 * (1 - dark))[:, :, None]
        F = f[y0:y1, x0:x1]; F[:] = F * (1 - m[:, :, None]) + col * m[:, :, None]
        for sg in (-1, 1):                                                           # 目
            e = (c[0] + sg * 0.21 * S, c[1] - 0.04 * S); L = s.L0(); cv2.ellipse(L, (int(e[0]), int(e[1])), (max(1, int(0.10 * S)), max(1, int(0.028 * S))), sg * 24, 0, 360, 1.0, -1, cv2.LINE_AA)
            s.glowmask(f, L, EYE, 1.0 * v, 1.0, 1.0, 8.0); s.flare(f, e, min(20 * k, 0.07 * S), EYE, 0.5 * v, 1.0, 0.0, False)

    def paw(s, f, c, S, t):
        """ふりおろされる 前足(うら側): 白い 毛なみ・しま・肉球・黒い つめ(式)"""
        W, H, k = s.W, s.H, s.k; x = (s.XX - c[0]) / S; y = -(s.YY - c[1]) / S; ax = np.abs(x)                # 指先が 下(こちらへ 先に 来る)
        fz = cv2.remap(s._tiles[3], ((x * 150 + 128) % 256).astype(np.float32), ((y * 150 + 128) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        fur = cv2.remap(s._tiles[0], ((np.arctan2(y - 0.2, x) * 120) % 256).astype(np.float32), ((np.hypot(x, y - 0.2) * 30) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        toes = [(-0.50, -0.12), (-0.19, -0.40), (0.19, -0.40), (0.50, -0.12)]
        dd = np.minimum(np.hypot(x / 0.66, (y - 0.22) / 0.56), np.hypot(x / 0.50, (y - 0.9) / 0.9))
        for tx, ty in toes: dd = np.minimum(dd, np.hypot((x - tx) / 0.27, (y - ty) / 0.30))
        dd = dd + (fz - 0.5) * 0.10; aa = min(0.2, 3.0 / S); m = sstep(1.0, 0.90 - aa, dd)
        stripe = sstep(0.5, 0.85, np.sin(y * 9 + ax * 3 + (fz - 0.5) * 4)) * sstep(0.62, 0.8, y) * sstep(0.12, 0.3, ax)
        lit = 0.42 + 0.55 * np.clip(1 - dd, 0, 1) ** 0.5 + 0.25 * np.clip(-(x * 0.5 + y * 0.6), -1, 1)
        col = FUR[None, None, :] * (lit * (0.72 + 0.45 * fur) * (1 - 0.85 * stripe))[:, :, None]
        pads = [(0.0, 0.26, 0.40, 0.28)] + [(tx, ty + 0.02, 0.155, 0.185) for tx, ty in toes]
        for px, py, rx, ry in pads:                                                    # 肉球
            pd = np.hypot((x - px) / rx, (y - py) / ry); pm = sstep(1.0, 1.0 - max(aa, 0.06), pd)
            pc = np.array([0.20, 0.13, 0.15], np.float32)[None, None, :] * (0.55 + 0.6 * np.clip(1 - pd, 0, 1) ** 0.6 + 0.3 * (fz - 0.5))[:, :, None] + (np.exp(-(((x - px + rx * 0.3) / (rx * 0.35)) ** 2 + ((y - py + ry * 0.35) / (ry * 0.3)) ** 2)) * 0.16)[:, :, None]
            col = col * (1 - pm[:, :, None]) + pc * pm[:, :, None]; col *= (1 - 0.35 * sstep(1.25, 1.0, pd) * (1 - pm))[:, :, None]
        f[:] = f * (1 - m[:, :, None]) + col * m[:, :, None]
        for tx, ty in toes:                                                          # つめ(毛の 前に 出る)
            dn = math.hypot(tx, ty + 0.1); ux, uy = tx / dn, (ty - 0.05) / dn; bx, by = tx + ux * 0.20, ty + uy * 0.20
            u = (x - bx) * ux + (y - by) * uy; w = -(x - bx) * uy + (y - by) * ux - 0.10 * np.clip(u, 0, 1) ** 2 * (1 if tx > 0 else -1)
            half = 0.085 * np.clip(1 - u / 0.42, 0, 1) ** 0.8; cm = sstep(half, half - max(aa * 0.4, 0.012), np.abs(w)) * (u > 0) * (u < 0.42)
            cc = np.array([0.10, 0.10, 0.12], np.float32)[None, None, :] * (0.5 + 0.9 * np.clip(0.5 - w / np.maximum(half, 1e-3) * 0.5, 0, 1))[:, :, None] + (np.exp(-((w + half * 0.45) / (half * 0.25 + 1e-3)) ** 2) * 0.45)[:, :, None]
            f[:] = f * (1 - cm[:, :, None]) + cc * cm[:, :, None]

    def frame(s, t):
        s._t = t; o = {}
        if t < T_CUT: f, sh = s.sc_wait(t)
        elif t < T_RUN: s._fv = 0.3; f, sh = s.sc_cut(t - T_CUT); s._fv = 1.0; o = dict(vig=0.45, thr=0.92)
        else: f, sh = s.sc_hunt(t)
        f, sh = s.finale(f, sh, t, DUR, None, (0.80, 0.95, 0.92), (1.0, 0.90, 0.30), -1)
        return s.film(f, t, sh * s.k, 0.55, o.get("thr", 0.84), (0.85, 0.97, 1.0), 1.0 + 0.2 * sh, 0.04, o.get("vig", 0.6), 1.10, 1.08)

    # 1) すすき野原。奥に 目が ひらく
    def sc_wait(s, t):
        W, H, k = s.W, s.H, s.k; gp = t - T_GRIP
        gust = 0.5 * math.exp(-((t - 0.9) / 0.35) ** 2) + (0.6 * math.exp(-((gp - 0.3) / 0.3) ** 2) if gp > 0 else 0)
        f = s.night(t); s.field(f, t, (0, 1, 2), 0, gust)
        if gp > 0.5: s.fareyes(f, ease2((gp - 0.5) / 0.4))
        s.hero(f, t, squash=(0.05 * math.exp(-gp * 9) * math.cos(gp * 16) if gp > 0 else 0)); s.eyes(f, clamp(gp / 0.3) if gp > 0 else 0)
        s.field(f, t, (3, 4), 0, gust)
        return f * clamp(t / 0.4), (4 * math.exp(-gp * 9) if gp > 0 else 0)

    # 3) かきわけて 走ってくる → とびかかる → 前足
    def sc_hunt(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_RUN; g = clamp(u / (T_LEAP - T_RUN)); lp = t - T_LEAP; pw = t - T_PAW
        le = ease2(lp / (T_PAW - T_LEAP)) if lp > 0 else 0.0
        f = s.night(t, 1.0); s.rush(f, t, 0.3 + 0.5 * g, 0.5 + g); s.field(f, t, (0, 1, 2), 0.5 + 0.8 * g, 0.2 * le)
        run = t * 15; S = (12 + 150 * g ** 2.2) * k * (1 + 2.6 * le); bob = abs(math.sin(run)) * S * 0.16 * (1 - le)
        c = (W * (0.24 + 0.26 * ease2(g)) + math.sin(run * 0.5) * S * 0.10 * (1 - le), H * 0.535 + H * 0.07 * g - bob - H * 0.30 * le)
        if g < 0.5: s.hero(f, t); s.eyes(f, 1.0)
        if g < 0.12: s.fareyes(f, 1.0)
        else: s.beast(f, c, S, t, clamp((g - 0.12) / 0.15), 0.75 * le)
        s.field(f, t, (3, 4), 0.8 + 1.0 * g, 0.3 * le)
        if le > 0: f *= 1 - 0.45 * le                                                # 月が かくれる
        sh = 1.5 + 4 * g * abs(math.sin(run)) * (lp <= 0) + 5 * le
        if pw > 0:
            e = clamp(pw / (T_HIT - T_PAW)); ee = ease2(e) ** 1.4; Sp = H * (0.30 + 0.22 * ee + 0.95 * ee ** 4); pc = (W * 0.5 + W * 0.06 * (1 - ee), -H * 0.40 + H * 0.84 * min(1.0, ee / 0.6) ** 0.8)
            sd = np.exp(-(((s.XX - pc[0]) / (Sp * 1.0)) ** 2 + ((s.YY - pc[1] + Sp * 0.25) / (Sp * 1.1)) ** 2)); f *= (1 - 0.7 * sd * ee)[:, :, None]   # 前足の 影
            s.paw(f, pc, Sp, t); sh = max(sh, 6 + 12 * ee)
            if e > 0.75: f = f + ((e - 0.75) / 0.25) ** 2 * 0.7
        if u < 0.1: f = f * (u / 0.1) + (1 - u / 0.1) * 0.5
        return f, sh

    def audio(s, path):
        a = Snd(DUR, 69); E = 82.41; pen = [0, 3, 5, 7, 10]
        def fq(d, o=0): return E * 2 ** ((pen[d % 5] + 12 * (d // 5 + o)) / 12.0)
        def biwa(d, t0, g=0.12, o=1): a.note(fq(d, o), t0, 0.3, g, "fm", 0.001); a.note(fq(d, o) * 3.02, t0, 0.04, g * 0.4, "sq", 0.0005); a.hat(t0, g * 0.4, 0.006)
        def purr(t0, t1, g=0.10): a.a += a.lp((np.sign(np.sin(2 * np.pi * 27 * a.t)) * (0.6 + 0.4 * np.sin(2 * np.pi * 55 * a.t))).astype(np.float32), 0.03) * a.gate(t0, t1, 0.2) * g * 6
        def roar(t0, ln, g=0.5):
            u = np.clip((a.t - t0) / ln, 0, 1); fr = 190 - 90 * u + 12 * np.sin(2 * np.pi * 31 * a.t); ph = 2 * np.pi * np.cumsum(fr) / a.sr; e = np.clip(u * 12, 0, 1) * np.clip((1 - u) * 2.5, 0, 1) * ((a.t >= t0) & (a.t < t0 + ln))
            a.a += (np.tanh(3 * (np.sin(ph) + 0.6 * np.sin(2.01 * ph) + 0.4 * np.sin(3.02 * ph))) * 0.35 + a.lp(a.nz, 0.25) * 0.9) * e * g
        def step_(t0, g=0.5): a.boom(t0, 62, g, 0.12); a.a += a.lp(a.nz, 0.5) * a.env(t0, 0.002, 0.04) * g * 0.3
        a.a += a.lp(a.nz, 0.05) * (0.5 + 0.4 * np.sin(a.t * 1.9) + 0.6 * np.exp(-((a.t - 0.9) / 0.35) ** 2)) * 0.7 * a.gate(0, DUR, 0.4)   # 風
        biwa(0, 0.5, 0.08); biwa(3, 1.1, 0.08)
        a.boom(T_GRIP, 60, 0.9, 0.3); biwa(0, T_GRIP, 0.12, 0); purr(T_GRIP + 0.5, T_CUT, 0.10); a.note(fq(9, 2), T_GRIP + 0.9, 0.5, 0.04, "sine", 0.05)
        a.boom(T_CUT, 52, 1.0, 0.35)
        for i, (d, dt) in enumerate(((5, 0.2), (7, 0.38), (8, 0.56), (7, 0.74), (5, 0.92), (4, 1.1), (5, 1.28), (8, 1.46))): biwa(d, T_CUT + dt, 0.12)
        for i in range(4): a.rise(T_CUT + 0.55 + i * 0.1, T_CUT + 0.62 + i * 0.1, 2500, 900, 0.10); a.hat(T_CUT + 0.6 + i * 0.1, 0.25, 0.03)
        for dt in (0.2, 0.56, 0.92, 1.28, 1.46): a.boom(T_CUT + dt, 66, 0.6, 0.15)
        purr(T_RUN, T_LEAP, 0.12); a.rise(T_RUN, T_LEAP, 60, 260, 0.10)
        n = int((T_LEAP - T_RUN) * 15 / math.pi)                                                                                 # 足音: 絵の 走りと 同じ 速さ
        for i in range(n): t0 = T_RUN + (i + 0.5) * math.pi / 15; step_(t0, 0.25 + 0.6 * (t0 - T_RUN) / (T_LEAP - T_RUN))
        for i in range(10): biwa((0, 3, 5, 3)[i % 4] + 5 * (i // 6), T_RUN + 0.1 + i * 0.2, 0.08)
        a.whoosh(T_LEAP, T_LEAP + 0.5, 0.5, 0.3); roar(T_LEAP + 0.05, 0.9, 0.55)                                                  # とびかかる
        a.rise(T_PAW, T_HIT, 2600, 300, 0.18); a.whoosh(T_PAW, T_HIT, 0.9, 0.4)                                                    # 前足が ふりおろされる
        tk = T_LEAP + 0.1; i = 0
        while tk < T_HIT - 0.04: a.boom(tk, 56, 0.5 + 0.5 * (tk - T_LEAP) / (T_HIT - T_LEAP), 0.12); tk += max(0.06, 0.20 - i * 0.014); i += 1
        for i in range(4): biwa(5 + i, T_PAW + i * 0.12, 0.10)                                                                   # カットインの 琵琶が かけ上がる
        a.boom(T_HIT, 40, 1.8, 0.6); a.a += a.lp(a.nz, 0.3) * a.env(T_HIT, 0.004, 0.3) * 0.6
        s.fin_snd(a, DUR, 69); a.save(path)


if __name__ == "__main__":
    r = Byakko()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
