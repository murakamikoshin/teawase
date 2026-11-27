"""スザク(★5 朱雀の舞姫) 専用の必殺技「紅蓮ノ天翔」— 決めは 奥から こちらへ
うるし黒と 金粉の 舞台。とびこんで 宙で 止まる → 扇が きらり、うしろに 炎の 大扇が ぱっと ひらく → 炎のしずく形の 窓のカットイン →
大扇の 骨の先から 火の羽が 1枚ずつ こちらへ 飛んでくる(小さい→画面を かすめて 通りすぎる) → 奥で 羽が あつまって 火の鳥に なり、はばたきながら こちらへ 突っこんでくる → finale
使い方: python3 hs_68.py 出力.mp4 / python3 hs_68.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 11.5
FIRE = np.array([1.0, 0.42, 0.10], np.float32); YEL = np.array([1.0, 0.86, 0.36], np.float32); CRIM = np.array([0.90, 0.10, 0.12], np.float32)
T_IN, T_FAN, T_CUT, T_FE, T_PH, T_NEAR = 0.8, 1.6, 3.4, 5.3, 7.3, 9.6
NF = 11


class Suzaku(Kit):
    def __init__(s, W=540):
        super().__init__(68, (1.0, 0.42, 0.10), "紅蓮ノ天翔", "Crimson Ascension", W)
        H, k = s.H, s.k
        s.eL = s.A(350, 265); s.eR = s.A(455, 280); s.fc = s.A(405, 265); s.fan = s.A(135, 225); s.piv = s.A(125, 285)
        s.ht = [s.text_img(c, hz.SERIF, int(78 * k), 3) for c in "紅蓮ノ天翔"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * k), 1) for c in "奥義"]
        s.HF = 0.40; s.CX = W / 2; s.CY = H * 0.62; s.VP = (W / 2, H * 0.34); s.FP = (W / 2, H * 0.60); s.FR = W * 0.78
        rg = np.random.RandomState(68); s.dust = rg.rand(70, 4).astype(np.float32); s.sl = rg.rand(60, 3).astype(np.float32); s.yn = s.YY / H
        s.ribs = [math.radians(-90 - 66 + 132 * i / (NF - 1.0)) for i in range(NF)]
        s.order = [5, 2, 8, 0, 10, 4, 6, 1, 9, 3, 7]

    # ---- 大道具 ----
    def stage(s, t, heat=0.0):
        f = (np.array([0.05, 0.01, 0.02], np.float32)[None, None, :] * (1 - s.yn)[:, :, None] + np.array([0.22 + 0.25 * heat, 0.03 + 0.05 * heat, 0.03], np.float32)[None, None, :] * s.yn[:, :, None]).astype(np.float32)
        L = s.L0()
        for r in s.dust:                                                             # 金粉
            x = (r[0] + 0.015 * math.sin(t * 0.9 + r[2] * 9)) * s.W; y = ((r[1] - t * 0.02 * (0.4 + r[3])) % 1.0) * s.H
            cv2.circle(L, (int(x), int(y)), max(1, int((1 + 1.6 * r[3]) * s.k)), float(0.35 + 0.65 * abs(math.sin(t * 1.7 + r[2] * 20))), -1, cv2.LINE_AA)
        f += (L + cv2.GaussianBlur(L, (0, 0), 2 * s.k))[:, :, None] * YEL[None, None, :] * 0.8
        return f

    def feather(s, f, c, sz, ang, v=1.0):
        """火の羽 1枚(c=根もと、ang の向きに のびる)"""
        ca, sa = math.cos(ang), math.sin(ang)
        def P(sc):
            pts = [(0, 0), (0.25, -0.16), (0.62, -0.20), (1.0, 0), (0.62, 0.20), (0.25, 0.16)]
            return np.array([(c[0] + (x * ca - y * sa) * sz * sc + (1 - sc) * ca * sz * 0.12, c[1] + (x * sa + y * ca) * sz * sc + (1 - sc) * sa * sz * 0.12) for x, y in pts], np.int32)
        if 14 * s.k < sz < 120 * s.k: s.soft(f, (c[0] + ca * sz * 0.5, c[1] + sa * sz * 0.5), sz * 0.6, 0.22 * v, FIRE)
        cv2.fillPoly(f, [P(1.0)], (float(0.86 * v), float(0.10 * v), float(0.06 * v)), cv2.LINE_AA); cv2.fillPoly(f, [P(0.8)], (float(1.0 * v), float(0.45 * v), float(0.10 * v)), cv2.LINE_AA); cv2.fillPoly(f, [P(0.5)], (float(1.0 * v), float(0.88 * v), float(0.40 * v)), cv2.LINE_AA)

    def bigfan(s, f, op, t, lit=None):
        """うしろの 炎の 大扇。op: ひらき"""
        if op <= 0.01: return
        P = s.FP; R = s.FR; k = s.k; m = s.L0(); a0 = -90 - 66 * op; a1 = -90 + 66 * op
        cv2.ellipse(m, (int(P[0]), int(P[1])), (int(R), int(R)), 0, a0, a1, 1.0, -1, cv2.LINE_AA); cv2.circle(m, (int(P[0]), int(P[1])), int(R * 0.22), 0.0, -1, cv2.LINE_AA)
        rr = np.hypot(s.XX - P[0], s.YY - P[1]) / R; band = (np.sin(rr * 26 - t * 3) * 0.5 + 0.5)
        colr = CRIM[None, None, :] * (0.55 + 0.25 * band)[:, :, None] + (YEL * 0.5)[None, None, :] * (np.clip((rr - 0.72) / 0.28, 0, 1) ** 2)[:, :, None]
        f[:] = f * (1 - m[:, :, None] * 0.85) + colr * (m * 0.85)[:, :, None]
        L = s.L0()
        for i, a in enumerate(s.ribs):
            aa = -1.5708 + (a + 1.5708) * op; s.line(L, (P[0] + math.cos(aa) * R * 0.22, P[1] + math.sin(aa) * R * 0.22), (P[0] + math.cos(aa) * R, P[1] + math.sin(aa) * R), 0.8 if lit is None or i not in lit else 0.2, 2)
        cv2.ellipse(L, (int(P[0]), int(P[1])), (int(R), int(R)), 0, a0, a1, 1.0, max(1, int(3 * k)), cv2.LINE_AA)
        s.add(f, L, YEL, 0.55, 2, 7)

    def tipof(s, i): a = s.ribs[i]; return (s.FP[0] + math.cos(a) * s.FR, s.FP[1] + math.sin(a) * s.FR)

    def rush(s, f, t, v, speed=1.0):
        L = s.L0(); VP = s.VP
        for r in s.sl:
            a = r[0] * 6.283; q = ((r[1] + t * speed * (0.6 + r[2])) % 1.0) ** 2; r0 = q * s.H * 0.9 + 14 * s.k
            s.line(L, (VP[0] + math.cos(a) * r0, VP[1] + math.sin(a) * r0), (VP[0] + math.cos(a) * r0 * (1.1 + 0.3 * q), VP[1] + math.sin(a) * r0 * (1.1 + 0.3 * q)), float(v * (0.2 + 0.8 * q)), 1 if q < 0.5 else 2)
        f += cv2.GaussianBlur(L, (0, 0), 1.0 * s.k)[:, :, None] * (FIRE * 0.7 + YEL * 0.3)[None, None, :] * 0.7

    def phoenix(s, f, c, S, t):
        """正面を 向いて はばたく 火の鳥(S=大きさ)"""
        fl = math.sin(t * 11) * 0.28
        for sg in (-1, 1):                                                           # 翼: 羽を 扇に ならべる
            sh = (c[0] + sg * S * 0.22, c[1] - S * 0.05)
            for j in range(7):
                a = (-1.5708 + sg * (0.55 + j * 0.27 + fl * (0.4 + j * 0.12))); s.feather(f, sh, S * (2.3 - 0.12 * abs(j - 2)), a)
        for j in (-1, 0, 1): s.feather(f, (c[0] + j * S * 0.1, c[1] + S * 0.35), S * (1.5 if j == 0 else 1.2), 1.5708 + j * 0.3 + 0.12 * math.sin(t * 6 + j))     # 尾
        s.soft(f, c, min(S * 0.8, 160 * s.k), 0.3, FIRE)
        cv2.ellipse(f, (int(c[0]), int(c[1] + S * 0.12)), (int(S * 0.22), int(S * 0.36)), 0, 0, 360, (1.0, 0.45, 0.10), -1, cv2.LINE_AA); cv2.ellipse(f, (int(c[0]), int(c[1] + S * 0.16)), (int(S * 0.13), int(S * 0.25)), 0, 0, 360, (1.0, 0.86, 0.40), -1, cv2.LINE_AA)
        h = (c[0], c[1] - S * 0.42)
        for j in (-1, 0, 1): s.feather(f, (h[0] + j * S * 0.06, h[1] - S * 0.16), S * 0.5, -1.5708 + j * 0.45)                                                # かんむり羽
        cv2.circle(f, (int(h[0]), int(h[1])), int(S * 0.23), (1.0, 0.50, 0.12), -1, cv2.LINE_AA)
        cv2.fillPoly(f, [np.array([(h[0] - S * 0.08, h[1] + S * 0.04), (h[0] + S * 0.08, h[1] + S * 0.04), (h[0], h[1] + S * 0.24)], np.int32)], (1.0, 0.92, 0.5), cv2.LINE_AA)   # くちばし
        for sg in (-1, 1): cv2.ellipse(f, (int(h[0] + sg * S * 0.11), int(h[1] - S * 0.03)), (max(1, int(S * 0.055)), max(1, int(S * 0.035))), sg * 25, 0, 360, (1.0, 1.0, 0.9), -1, cv2.LINE_AA); cv2.circle(f, (int(h[0] + sg * S * 0.10), int(h[1] - S * 0.03)), max(1, int(S * 0.022)), (0.5, 0.05, 0.0), -1, cv2.LINE_AA); cv2.line(f, (int(h[0] + sg * S * 0.03), int(h[1] - S * 0.045)), (int(h[0] + sg * S * 0.19), int(h[1] - S * 0.115)), (0.75, 0.12, 0.02), max(1, int(S * 0.03)), cv2.LINE_AA)

    def hero(s, f, t, hf=None, cy=None, cx=None, **kw):
        hf = s.HF if hf is None else hf; img = s.pose(t, **kw); s.draw(f, img, s.CX if cx is None else cx, s.CY if cy is None else cy, hf)

    def glint(s, f, v, t):
        if v <= 0.01: return
        p = s.at(*s.fan); L = s.L0(); s.star4(L, p, 46 * s.k * v, min(1.0, v), 0.3, 0.1); s.add(f, L, YEL * 0.5 + 0.5, 0.8, 2, 6)

    def frame(s, t):
        if t < T_CUT: f, sh = s.sc_open(t)
        elif t < T_FE: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_PH: f, sh = s.sc_feathers(t)
        else: f, sh = s.sc_bird(t)
        f, sh = s.finale(f, sh, t, DUR, None, (1.0, 0.45, 0.12), (1.0, 0.86, 0.36), -1)
        return s.post(f, FIRE * 0.6 + 0.4, 0.40, sh * s.k, t, 0.9)

    # 1) とびこむ → 扇きらり → 大扇が ひらく
    def sc_open(s, t):
        W, H, k = s.W, s.H, s.k; fn = t - T_FAN; op = back(fn / 0.35, 1.3) if fn > 0 else 0.0
        f = s.stage(t, 0.5 * clamp(fn)); s.bigfan(f, op, t)
        up = ease(t / T_IN); st = t - T_IN
        s.hero(f, t, cx=s.CX + W * 0.7 * (1 - up), cy=s.CY - H * 0.25 * (1 - up), squash=(-0.06 * math.exp(-st * 9) * math.cos(st * 15) if st > 0 else -0.05))
        s.glint(f, math.sin(clamp((t - 1.15) / 0.5) * 3.1416) * 1.4, t)
        if 0 < fn < 0.1: f = f + (1 - fn / 0.1) * 0.6
        return f * clamp(t / 0.3), (6 * math.exp(-fn * 8) if fn > 0 else 0)

    # 2) 炎のしずく形の 窓
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.stage(t, 0.3) * 0.7
        C = (W / 2, H * 0.40); g = back(u / 0.3, 1.2); R = max(2.0, W * 0.40 * g)
        pts = []
        for i in range(48):                                                          # しずく: 上が とがる
            a = i * 6.2832 / 48; x = math.sin(a) * R * (1 - 0.0); y = -math.cos(a) * R;
            if y < 0: x *= (1 - min(1.0, abs(y) / max(1.0, abs(R))) ** 2.2 * 0.92); y *= 1.55
            pts.append((C[0] + x + (math.sin(t * 9 + i) * 3 * k if y < -R * 0.6 else 0), C[1] + y))
        m = s.L0(); cv2.fillPoly(m, [np.array(pts, np.int32)], 1.0, cv2.LINE_AA)
        win = np.zeros((H, W, 3), np.float32) + (np.array([0.55, 0.08, 0.06], np.float32)[None, None, :] * (1.3 - s.yn)[:, :, None]) + (np.clip(s.yn * 1.4 - 0.3, 0, 1) * 0.6)[:, :, None] * YEL[None, None, :]
        zf = 0.92 + 0.05 * ease(u / 1.8); sc = zf * H / s.ah; img = s.pose(t)
        s.draw(win, img, C[0] - (s.fc[0] - s.aw / 2) * sc, C[1] + H * 0.02 - (s.fc[1] - s.ah / 2) * sc, zf)
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        cv2.polylines(f, [np.array(pts, np.int32)], True, (1.0, 0.86, 0.40), max(1, int(6 * k)), cv2.LINE_AA); L = s.L0(); cv2.polylines(L, [np.array(pts, np.int32)], True, 1.0, max(1, int(10 * k)), cv2.LINE_AA); f += cv2.GaussianBlur(L, (0, 0), 8 * k)[:, :, None] * FIRE[None, None, :] * 0.8
        n = len(s.ht); step = W * 0.175
        for i, T in enumerate(s.ht):                                                 # 技名: 火の粉のように 下から 舞い上がる
            a = (u - 0.55 - i * 0.1) / 0.25
            if a > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.845 + (1 - ease(a)) * 50 * k, clamp(a * 2), 1.0)
        for i, T in enumerate(s.vo):
            if u > 0.45 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.772, 1.0, 1.0)
        if u > 1.3: s.text(f, s.txt_en, W / 2, H * 0.845 + 58 * k, ease((u - 1.3) / 0.3), 1.0)
        if u < 0.08: f = f + (1 - u / 0.08) * 0.7
        return f, 0

    # 3) 骨の先から 火の羽が こちらへ
    def sc_feathers(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_FE; step = (T_PH - T_FE - 0.5) / NF
        gone = [s.order[j] for j in range(NF) if u > j * step]
        f = s.stage(t, 0.6); s.rush(f, t, 0.5, 0.7); s.bigfan(f, 1.0, t, gone)
        for i in range(NF):
            if i not in gone: s.feather(f, s.tipof(i), 40 * k, s.ribs[i])
        s.hero(f, t, jit=0.0); s.glint(f, 0.6, t)
        sh = 0; fl = 0.0
        for j in range(NF):                                                          # とんでくる 羽(近い ものを あとに 描く)
            v = u - j * step
            if v <= 0 or v > 0.75: continue
            i = s.order[j]; o = s.tipof(i); z = 8.0 * (1 - v / 0.75) ** 2.0 + 0.22; q = (8.0 / z)
            d = (o[0] - s.VP[0], o[1] - s.VP[1]); p = (s.VP[0] + d[0] * (0.45 + 0.55 * q ** 0.42), s.VP[1] + d[1] * (0.45 + 0.55 * q ** 0.42) + (q - 1) * 9 * k)
            sz = 40 * k * q ** 0.95; ang = math.atan2(d[1], d[0])
            if z < 0.5: fl = max(fl, (0.5 - z) / 0.28 * 0.55); sh = max(sh, 9)
            s.feather(f, (p[0] - math.cos(ang) * sz * 0.5, p[1] - math.sin(ang) * sz * 0.5), sz, ang)
        f = f + fl * (FIRE * 0.7 + 0.3)[None, None, :]
        if u < 0.1: f = f * (u / 0.1) + (1 - u / 0.1) * 0.5
        return f, sh

    # 4) 羽が あつまって 火の鳥 → こちらへ 突っこむ
    def sc_bird(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_PH; g = clamp(u / (T_NEAR - T_PH))
        f = s.stage(t, 0.8); s.rush(f, t, 0.5 + 0.5 * g, 0.8 + 1.4 * g)
        if g < 0.7: s.hero(f, t, cy=s.CY + H * 0.5 * ease2((g - 0.35) / 0.35))
        S = (26 + 40 * ease(u / 0.5)) * k * (1 + 7 * g ** 3.0); c = (s.VP[0] + math.sin(t * 3.0) * W * 0.06 * (1 - g), s.VP[1] + S * 0.42 * g + math.sin(t * 11) * S * 0.04)
        if u < 0.5:                                                                  # 火の粉が 1点に あつまる
            L = s.L0()
            for r in s.dust[:30]: a = r[0] * 6.283; d = (1 - ease(u / 0.5)) * W * (0.3 + 0.5 * r[1]); cv2.circle(L, (int(c[0] + math.cos(a) * d), int(c[1] + math.sin(a) * d)), max(1, int(3 * k)), 1.0, -1, cv2.LINE_AA)
            s.add(f, L, YEL, 0.8, 1, 5)
        s.phoenix(f, c, S, t)
        if u < 0.1: f = f * (u / 0.1) + (1 - u / 0.1) * 0.6
        return f, 2 + 10 * g ** 3

    # ---- この子だけの曲: 祭りばやし(ニ・ホ・ト・イ・ハ の 笛、しめ太鼓、かね)。羽は 1枚ごとに「ひゅっ」。火の鳥は 高い 鳴き声と せり上がる 太鼓 ----
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
        a.whoosh(T_FAN - 0.1, T_FAN + 0.3, 0.5, 0.35); odaiko(T_FAN + 0.02, 1.0); a.a += a.lp(a.nz, 0.2) * a.gate(T_FAN, DUR - 1.7, 0.4) * 0.12     # 火の ごうごう
        pat = (0, 2, 4, 5, 4, 2, 4, 7)
        for i in range(8): fue(pat[i], T_FAN + 0.3 + i * 0.18, 0.16, 0.05); shime(T_FAN + 0.3 + i * 0.18, 0.3 if i % 2 else 0.45); kane(T_FAN + 0.39 + i * 0.18, 0.04)
        odaiko(T_CUT, 1.0)
        for i, (d, dt) in enumerate(((7, 0.2), (9, 0.38), (10, 0.56), (9, 0.74), (7, 0.92), (5, 1.1), (7, 1.28), (10, 1.46))): fue(d, T_CUT + dt, 0.16, 0.07)
        for i in range(10): shime(T_CUT + 0.2 + i * 0.18, 0.4 if i % 2 == 0 else 0.25); kane(T_CUT + 0.29 + i * 0.18, 0.04)
        step = (T_PH - T_FE - 0.5) / NF
        for j in range(NF):
            t0 = T_FE + j * step; a.rise(t0, t0 + 0.7, 500, 2600, 0.05); a.whoosh(t0 + 0.3, t0 + 0.75, 0.35, 0.4); a.boom(t0 + 0.72, 70, 0.6, 0.12); shime(t0, 0.35); fue(j % 5 + 5, t0, 0.12, 0.04)
        cry(T_PH + 0.45, 0.6, 0.09); odaiko(T_PH + 0.45, 0.9)
        tk = T_PH + 0.6; i = 0
        while tk < DUR - 1.8: odaiko(tk, 0.5 + 0.5 * (tk - T_PH) / 2.3); shime(tk + 0.07, 0.3); tk += max(0.07, 0.3 - i * 0.02); i += 1
        a.rise(T_PH + 0.5, DUR - 1.7, 120, 700, 0.12); cry(T_NEAR - 0.7, 0.7, 0.10)
        for i in range(8): fue(5 + i, T_PH + 0.9 + i * 0.16, 0.14, 0.05)
        s.fin_snd(a, DUR, 68); a.save(path)


if __name__ == "__main__":
    r = Suzaku()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
