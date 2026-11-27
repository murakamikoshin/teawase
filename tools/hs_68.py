"""スザク(★5 朱雀の舞姫) 専用の必殺技「紅蓮ノ天翔」— 高品質版(hs_fx)・通し直し。敵は さいしょから 画面の こちら側。炎の 大扇が 翼に なって こちらを つつむ
うるし黒と 金粉の 舞台(床に 火の てり)。とびこんで 宙で 止まる → 扇が きらり、うしろに 炎の 大扇が ぱっと ひらく(前ふり: この 大扇が 武器) → 炎のしずく形の 窓のカットイン →
手の 扇を ふるたび 大扇の 骨の先から 炎の 羽が 1枚ずつ こちらへ 飛んでくる(5枚・画面を かすめて 通りすぎる) →
大扇が まん中から 左右 2枚の 炎の 翼に 分かれて 大きく ひろがり、ひとはばたきで 両がわから こちらを つつんで とじる → 白 → 暗転
使い方: python3 hs_68.py 出力.mp4 / python3 hs_68.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old68", "/home/claude/tools/pre_fin/hs_68_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = 9.1
FIRE, YEL, CRIM, NF = old.FIRE, old.YEL, old.CRIM, old.NF
T_IN, T_FAN, T_CUT, T_FE = old.T_IN, old.T_FAN, old.T_CUT, old.T_FE
FSTEP, NFL, T_WING, T_BEAT, T_SHUT = 0.30, 5, 7.0, 7.45, 8.4
RAMP = [(0.0, (0.0, 0.0, 0.0)), (0.22, (0.50, 0.03, 0.02)), (0.50, (0.95, 0.22, 0.04)), (0.78, (1.0, 0.70, 0.20)), (1.0, (1.0, 0.97, 0.80))]


class Suzaku(FX, old.Suzaku):
    def __init__(s, W=540):
        old.Suzaku.__init__(s, W); s.fx_init(68)

    # ---- 舞台 ----
    def stage(s, t, heat=0.0):
        H, W, k = s.H, s.W, s.k
        f = (np.array([0.030, 0.006, 0.010], np.float32)[None, None, :] * (1 - s.yn)[:, :, None] + np.array([0.16 + 0.22 * heat, 0.02 + 0.04 * heat, 0.02], np.float32)[None, None, :] * s.yn[:, :, None]).astype(np.float32)
        f += s.nebula(t, [(0.20, 0.02, 0.02), (0.45, 0.08, 0.03), (0.9, 0.4, 0.1)], 0.0009, (3.0, -14.0), 0.25 + 0.5 * heat)
        g = sstep(0.86, 0.88, s.yn); f = f * (1 - 0.55 * g)[:, :, None] + (g * np.exp(-((s.XX - W / 2) / (W * 0.45)) ** 2) * (0.10 + 0.35 * heat))[:, :, None] * (FIRE * 0.8 + YEL * 0.2)[None, None, :]   # うるしの 床の てり
        s.starfield(f, t, 900, YEL, 4, 0.9, (3.0, -16.0)); s.starfield(f, t + 4, 160, YEL * 0.7 + 0.3, 7, 2.0, (6.0, -30.0))               # 金粉
        return f

    def flamefan(s, f, P, R, a0, a1, t, v=1.0, r0=0.22, hot=1.0, feather_n=0):
        """炎の 扇/翼: 中心 P・半径 R・角 a0..a1(度)。炎は 外へ 流れ、ふちは ゆらぐ"""
        if v <= 0.01 or a1 - a0 < 0.5: return
        dx = s.XX - P[0]; dy = s.YY - P[1]; rr = np.hypot(dx, dy) / R; ang = np.degrees(np.arctan2(dy, dx)); ang = np.where(ang > 90, ang - 360, ang)       # -270..90
        n = cv2.remap(s._tiles[1], ((ang * 1.6) % 256).astype(np.float32), ((rr * 150 - t * 110) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        n2 = cv2.remap(s._tiles[3], ((ang * 4.5) % 256).astype(np.float32), ((rr * 380 - t * 190) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        edge = 1.0 - 0.16 * n - 0.06 * n2 - (0.16 * (1 - np.abs(np.cos(np.radians(ang) * feather_n))) ** 2 if feather_n else 0.0); am = sstep(a0 - 1.5, a0 + 2.5, ang) * sstep(a1 + 1.5, a1 - 2.5, ang)
        m = sstep(edge, edge - 0.05, rr) * sstep(r0, r0 + 0.03, rr) * am
        heat = (0.38 + 0.50 * n + 0.35 * n2) * (0.55 + 0.6 * sstep(0.55, 1.0, rr)) * hot
        if feather_n: heat = heat * (0.72 + 0.40 * np.abs(np.cos(np.radians(ang) * feather_n))) * (0.80 + 0.25 * np.cos(rr * 22 - t * 4))           # 羽の すじ・段
        col = s.ramp(np.clip(heat * m, 0, 1), RAMP) * 1.25
        mm = np.clip(m * 1.3, 0, 1)[:, :, None] * v; f[:] = f * (1 - 0.9 * mm) + col * v
        f += (cv2.GaussianBlur(m, (0, 0), 16 * s.k) * 0.22 * v)[:, :, None] * FIRE[None, None, :]

    def bigfan(s, f, op, t, lit=None):
        if op <= 0.01: return
        P = s.FP; R = s.FR; s.flamefan(f, P, R, -90 - 66 * op, -90 + 66 * op, t, 1.0, 0.22, 0.95)
        for i, a in enumerate(s.ribs):                                                # 金の 骨
            aa = -1.5708 + (a + 1.5708) * op; v = 0.8 if lit is None or i not in lit else 0.2
            s.glowline(f, [(P[0] + math.cos(aa) * R * 0.22, P[1] + math.sin(aa) * R * 0.22), (P[0] + math.cos(aa) * R * 0.97, P[1] + math.sin(aa) * R * 0.97)], YEL, 1.0, 0.45 * v, 0.5)

    def ffeather(s, f, c, L, ang, t, v=1.0):
        """炎の 羽 1枚(式): c=根もと・ang の向きに 長さ L。しんは 白熱、ふちは ゆらぐ"""
        W, H, k = s.W, s.H, s.k; ca, sa = math.cos(ang), math.sin(ang); mx, my = c[0] + ca * L * 0.5, c[1] + sa * L * 0.5; r = L * 0.62
        x0 = int(max(0, mx - r)); x1 = int(min(W, mx + r)); y0 = int(max(0, my - r)); y1 = int(min(H, my + r))
        if x1 - x0 < 3 or y1 - y0 < 3: return
        dx = s.XX[y0:y1, x0:x1] - c[0]; dy = s.YY[y0:y1, x0:x1] - c[1]; u = (dx * ca + dy * sa) / L; w = (-dx * sa + dy * ca) / L
        n = cv2.remap(s._tiles[2], ((u * 120 - t * 160) % 256).astype(np.float32), ((w * 300 + 128) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        half = 0.20 * np.clip(np.sin(np.clip(u, 0, 1) * 3.1416), 0, 1) ** 0.7 * (1 - 0.3 * u) * (0.85 + 0.35 * n); q = np.abs(w) / np.maximum(half, 1e-3)
        m = sstep(1.0, 0.75, q) * (u > 0) * (u < 1)
        barb = 0.80 + 0.30 * np.cos((u * 26 - np.abs(w) * 60))
        heat = np.clip((1.05 - q * 0.75) * barb * (0.75 + 0.5 * n), 0, 1) * m + np.exp(-(w * L / (1.3 * k + L * 0.012)) ** 2) * (u > 0.02) * (u < 0.92) * 0.9
        F = f[y0:y1, x0:x1]; F[:] = F * (1 - 0.85 * m[:, :, None] * v) + s_ramp(np.clip(heat, 0, 1)) * 1.3 * v
        s.flare(f, (c[0] + ca * L * 0.55, c[1] + sa * L * 0.55), min(18 * k, L * 0.10), FIRE * 0.6 + YEL * 0.4, 0.5 * v, 1.0, 0.0, False)

    def rush(s, f, t, v, speed=1.0):
        s.rushdust(f, t, s.VP, 1100, speed * 1.4, FIRE * 0.6 + YEL * 0.4, 0.9 * v, 3, 1.0)

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        s.draw_lit(f, img, cx, cy, hfrac, rim=FIRE * 0.6 + YEL * 0.4, rim_dir=(0.0, -1.0), rim_v=0.9 * alpha, amb=(1.0, 0.75, 0.55), amb_v=0.14, alpha=alpha, shade=getattr(s, "_hsh", 0.95))

    def add(s, f, L, col, gain=1.0, b1=3, b2=12):
        s.glowmask(f, np.clip(L, 0, 1), col, gain, 0.8, 1.0, float(b2))

    def glint(s, f, v, t):
        if v <= 0.01: return
        p = s.at(*s.fan); s.flare(f, p, 10 * s.k * v, YEL * 0.5 + 0.5, min(1.2, v), 1.4, 0.3, v > 0.9)

    def frame(s, t):
        s._t = t; o = {}; s._hsh = 0.95
        if t < T_CUT: f, sh = s.sc_open(t)
        elif t < T_FE: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.92)
        else: f, sh = s.sc_wings(t)
        f, sh = s.finale(f, sh, t, DUR, None, (1.0, 0.45, 0.12), (1.0, 0.86, 0.36), -1)
        return s.film(f, t, sh * s.k, 0.6, o.get("thr", 0.82), (1.0, 0.8, 0.6), 1.0 + 0.2 * sh, 0.035, o.get("vig", 0.6), 1.15, 1.04)

    # 3) 炎の 羽が 1枚ずつ こちらへ → 大扇が 2枚の 翼に → つつんで とじる
    def sc_wings(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_FE; wv = t - T_WING; bt = t - T_BEAT
        sp = ease(wv / 0.40) if wv > 0 else 0.0; g = clamp(bt / (T_SHUT - T_BEAT)) if bt > 0 else 0.0; gg = g ** 1.9
        f = s.stage(t, 0.6 + 0.4 * g); s.rush(f, t, 0.5 + 0.5 * g, 0.7 + 0.8 * g)
        gone = [s.order[j] for j in range(NFL) if u > j * FSTEP]
        if wv <= 0: s.bigfan(f, 1.0, t, gone)
        else:                                                                        # 翼: まん中が 分かれ、外へ 大きく ひらき、はばたきで こちらへ
            P = (s.FP[0], s.FP[1] + H * 0.10 * gg); R = s.FR * (1 + 0.12 * sp + 2.6 * gg); gap = 16 * sp * (1 - g) ** 3 - 4 * g; spread = 66 + 44 * sp + 68 * g
            s.flamefan(f, P, R, -90 - spread, -90 - gap, t, 1.0, 0.16 * (1 - gg), 1.0 + 0.25 * g, 9); s.flamefan(f, P, R, -90 + gap, -90 + spread, t, 1.0, 0.16 * (1 - gg), 1.0 + 0.25 * g, 9)
        for i in range(NF):
            if wv <= 0 and i not in gone and i in s.order[:NFL]: s.ffeather(f, s.tipof(i), 46 * k, s.ribs[i], t, 0.9)
        s._hsh = 0.95 - 0.45 * g
        if g < 0.55:
            fl = max([math.exp(-(u - j * FSTEP) * 7) for j in range(NFL) if u > j * FSTEP] + [0.0]) if wv <= 0 else math.exp(-wv * 5)
            s.hero(f, t, squash=0.04 * fl, lean=-8 * k * fl); s.glint(f, 0.5 + 0.9 * fl, t)
        sh = 0; fla = 0.0
        for j in range(NFL):                                                         # とんでくる 羽
            v = u - j * FSTEP
            if v <= 0 or v > 0.62: continue
            i = s.order[j]; o_ = s.tipof(i); z = 8.0 * (1 - v / 0.62) ** 2.0 + 0.22; q = 8.0 / z
            d = (o_[0] - s.VP[0], o_[1] - s.VP[1]); p = (s.VP[0] + d[0] * (0.45 + 0.55 * q ** 0.42), s.VP[1] + d[1] * (0.45 + 0.55 * q ** 0.42) + (q - 1) * 9 * k)
            sz = 46 * k * q ** 0.95; ang = math.atan2(d[1], d[0])
            if z < 0.5: fla = max(fla, (0.5 - z) / 0.28 * 0.45); sh = max(sh, 9)
            s.ffeather(f, (p[0] - math.cos(ang) * sz * 0.5, p[1] - math.sin(ang) * sz * 0.5), sz, ang, t, 1.0)
        f = f + fla * (FIRE * 0.7 + 0.3)[None, None, :]
        if wv > 0:
            if wv < 0.08: f = f + (1 - wv / 0.08) * 0.5
            sh = max(sh, 3 * math.exp(-wv * 6) + 12 * gg)
            if g > 0.5: f = f + ((g - 0.5) / 0.5) ** 2 * 0.8 * (YEL * 0.5 + 0.5)[None, None, :]
            f = s.godrays(f, (W / 2, H * 0.45), 0.2 + 0.4 * g, 0.8, 8, 0.86, 0.4)
        if u < 0.1: f = f * (u / 0.1) + (1 - u / 0.1) * 0.5
        return f, sh

    def audio(s, path):
        a = Snd(DUR, 68); D = 293.66; sc = [0, 2, 5, 7, 10]
        def fq(d, o=0): return D * 2 ** ((sc[d % 5] + 12 * (d // 5 + o)) / 12.0)
        def fue(d, t0, ln, g=0.06, o=1):
            ph = 2 * np.pi * fq(d, o) * (a.t - t0) + 0.3 * np.sin(2 * np.pi * 6 * (a.t - t0)); e = a.env(t0, 0.02, ln); a.a += (np.sin(ph) + 0.3 * np.sin(2 * ph)) * e * g + (a.nz - a.lp(a.nz, 0.7)) * e * g * 0.1
        def shime(t0, g=0.4): a.note(330, t0, 0.05, g * 0.4, "sine", 0.0005); a.hat(t0, g * 0.5, 0.012); a.boom(t0, 150, g * 0.5, 0.05)
        def kane(t0, g=0.06): a.note(2217, t0, 0.06, g, "sq", 0.0005); a.note(3322, t0, 0.04, g * 0.6, "sine", 0.0005)
        def odaiko(t0, g=1.0): a.boom(t0, 48, g, 0.4)
        def cry(t0, ln=0.5, g=0.08):
            u = np.clip((a.t - t0) / ln, 0, 1); fr = 1500 + 900 * np.sin(u * np.pi) + 60 * np.sin(2 * np.pi * 14 * a.t); ph = 2 * np.pi * np.cumsum(fr) / a.sr
            a.a += (np.sin(ph) + 0.4 * np.sin(2 * ph)) * np.sin(u * np.pi) * ((a.t >= t0) & (a.t < t0 + ln)) * g
        a.whoosh(0.0, T_IN, 0.4, 0.3); odaiko(T_IN, 0.8); kane(T_IN + 0.02)
        fue(7, 1.0, 0.4, 0.05); a.note(3500, 1.3, 0.4, 0.04, "sine", 0.05)
        a.whoosh(T_FAN - 0.1, T_FAN + 0.3, 0.5, 0.35); odaiko(T_FAN + 0.02, 1.0); a.a += a.lp(a.nz, 0.2) * a.gate(T_FAN, DUR - 0.7, 0.4) * 0.12     # 火の ごうごう
        pat = (0, 2, 4, 5, 4, 2, 4, 7)
        for i in range(8): fue(pat[i], T_FAN + 0.3 + i * 0.18, 0.16, 0.05); shime(T_FAN + 0.3 + i * 0.18, 0.3 if i % 2 else 0.45); kane(T_FAN + 0.39 + i * 0.18, 0.04)
        odaiko(T_CUT, 1.0)
        for i, (d, dt) in enumerate(((7, 0.2), (9, 0.38), (10, 0.56), (9, 0.74), (7, 0.92), (5, 1.1), (7, 1.28), (10, 1.46))): fue(d, T_CUT + dt, 0.16, 0.07)
        for i in range(10): shime(T_CUT + 0.2 + i * 0.18, 0.4 if i % 2 == 0 else 0.25); kane(T_CUT + 0.29 + i * 0.18, 0.04)
        for j in range(NFL):                                                                                                  # 炎の 羽が とんでくる
            t0 = T_FE + j * FSTEP; a.rise(t0, t0 + 0.55, 500, 2600, 0.05); a.whoosh(t0 + 0.25, t0 + 0.62, 0.35, 0.4); a.boom(t0 + 0.6, 70, 0.6, 0.12); shime(t0, 0.35); fue(j % 5 + 5, t0, 0.12, 0.05)
        cry(T_WING, 0.5, 0.09); odaiko(T_WING, 1.0); a.whoosh(T_WING - 0.1, T_WING + 0.35, 0.5, 0.3)                              # 翼が ひらく
        a.whoosh(T_BEAT - 0.15, T_BEAT + 0.25, 0.7, 0.25); odaiko(T_BEAT, 1.2)                                                    # はばたき
        tk = T_BEAT + 0.12; i = 0
        while tk < T_SHUT - 0.04: odaiko(tk, 0.5 + 0.6 * (tk - T_BEAT) / (T_SHUT - T_BEAT)); shime(tk + 0.05, 0.3); tk += max(0.06, 0.22 - i * 0.02); i += 1
        a.rise(T_BEAT, T_SHUT, 120, 760, 0.13); cry(T_SHUT - 0.6, 0.6, 0.10)
        for i in range(6): fue(5 + i, T_BEAT + 0.1 + i * 0.14, 0.12, 0.05)                                                       # カットインの 笛の ふしが かけ上がる
        a.a += a.lp(a.nz, 0.25) * np.clip((a.t - T_BEAT) / (T_SHUT - T_BEAT), 0, 1) ** 2 * ((a.t > T_BEAT) & (a.t < T_SHUT + 0.1)) * 0.8
        odaiko(T_SHUT, 1.8)
        s.fin_snd(a, DUR, 68); a.save(path)


def s_ramp(v):
    out = np.zeros(v.shape + (3,), np.float32)
    for (p0, c0), (p1, c1) in zip(RAMP[:-1], RAMP[1:]):
        seg = (v >= p0) & ((v < p1) if p1 < 1.0 else (v <= 1.0)); w = np.clip((v - p0) / (p1 - p0), 0, 1)[seg][:, None]
        out[seg] = np.asarray(c0, np.float32)[None, :] * (1 - w) + np.asarray(c1, np.float32)[None, :] * w
    return out


if __name__ == "__main__":
    r = Suzaku()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
