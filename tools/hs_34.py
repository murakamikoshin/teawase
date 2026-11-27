"""ロキ(★5 道化) 専用の必殺技「道化ノ千変」— 高品質版(hs_fx)。話は v3 と 同じ: 敵は さいしょから 画面の こちら側(客席)
もやの かかった まっ暗な舞台に スポットライト(光の柱に ほこりが 舞う)。ひざをついた 道化 → カチッ、カチッ。明かりが 消えるたび べつの場所に べつの色の ロキ → ジャン、5人 いっぺんに →
ひし形の窓: 顔の右半分(仮面の側)だけ 色が 反転 → 舞台の壁(ビロードの ひし形・金の ふち)が めくれ、まん中に 本物。本物が 指を 鳴らすたびに
分身が 光の影に なって 1人ずつ 客席(こちら)へ とびかかってくる → さいごに 本物が 逆光の 影に なって とび、仮面の目の 光が 目の前に → 白 → 暗転
使い方: python3 hs_34.py 出力.mp4 / python3 hs_34.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old34", "/home/claude/tools/pre_fin/hs_34_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = old.DUR; GOLD, GRN, PUR, TINT, SPOT = old.GOLD, old.GRN, old.PUR, old.TINT, old.SPOT
T_B, BEAT, T_ALL, T_CUT, T_D = old.T_B, old.BEAT, old.T_ALL, old.T_CUT, old.T_D
HAZE = [(0.05, 0.03, 0.10), (0.10, 0.05, 0.16), (0.20, 0.14, 0.28)]


class Loki(FX, old.Loki):
    def __init__(s, W=540):
        old.Loki.__init__(s, W); s.fx_init(34); s._hz = (None, None)
        s.vel = 0.70 + 0.6 * s.fbm(0.010, 5, 2, 3, 0, 1)                                  # ビロードの むら
        m = np.maximum(np.abs(s.fu), np.abs(s.fv)); s.bev = (0.50 + 0.75 * (1 - 2 * m) ** 0.8 + 0.30 * (s.fv - s.fu)).astype(np.float32)   # ひし形の ふくらみ(左上から 光)
        s.stud = np.exp(-(((0.5 - np.abs(s.fu)) ** 2 + (0.5 - np.abs(s.fv)) ** 2) / 0.0016)).astype(np.float32)                    # 角の 金の びょう

    # ---- 舞台 ----
    def haze(s, t):
        if s._hz[0] != t:
            d = np.zeros((s.H, s.W, 3), np.float32); s.starfield(d, t, 1600, (1, 1, 1), 3, 1.3, (9.0, 14.0))
            s._hz = (t, 0.45 + 0.9 * s.fbm(0.0016, t * 14, t * 5, 3, 30 * s.k, 2), d[:, :, 0])
        return s._hz[1], s._hz[2]

    def stage(s, t, v=1.0):
        f = np.zeros((s.H, s.W, 3), np.float32) + np.array([0.008, 0.007, 0.014], np.float32)
        f += s.nebula(t, HAZE, 0.0008, (5.0, 2.0), 0.35 * v)
        return f

    def spot(s, f, cx, fy, r, v, col=WHITE):
        if v <= 0.01: return
        k = s.k; hz, dust = s.haze(s._t); L = s.L0(); tx = s.W / 2 + (cx - s.W / 2) * 0.45; col = np.asarray(col, np.float32)
        cv2.fillPoly(L, [np.array([(tx - r * 0.10, -10), (tx + r * 0.10, -10), (cx + r, fy), (cx - r, fy)], np.int32)], 1.0, cv2.LINE_AA)
        cv2.ellipse(L, (int(cx), int(fy)), (int(r), int(r * 0.22)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        B = cv2.GaussianBlur(L, (0, 0), 5 * k); fall = 0.55 + 0.45 * np.clip(s.YY / max(1.0, fy), 0, 1)
        G = s.L0(); cv2.ellipse(G, (int(cx), int(fy)), (int(r), int(r * 0.22)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        Rg = s.L0(); cv2.ellipse(Rg, (int(cx), int(fy)), (int(r), int(r * 0.22)), 0, 0, 360, 1.0, max(1, int(2 * k)), cv2.LINE_AA)
        lum = B * hz * fall * 0.17 + cv2.GaussianBlur(G, (0, 0), 3 * k) * 0.36 + cv2.GaussianBlur(Rg, (0, 0), 2 * k) * 0.16 + B * dust * 1.5
        f += (lum * v)[:, :, None] * (col * 0.65 + 0.35)[None, None, :]
        s.flare(f, (tx, 2 * k), 9 * k, col * 0.5 + 0.5, 0.55 * v, 0.9, 0.0, False)

    def figure(s, f, img, cx, cy, hf, col, v=1.0, white=0.0, shade=0.92):
        fy = cy + hf * s.H / 2; L = s.L0(); cv2.ellipse(L, (int(cx), int(fy + 2 * s.k)), (int(hf * s.H * 0.30), int(hf * s.H * 0.035)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        f *= (1 - 0.75 * v * cv2.GaussianBlur(L, (0, 0), 4 * s.k))[:, :, None]
        s.draw_lit(f, img, cx, cy, hf, rim=np.asarray(col, np.float32) * 0.6 + 0.4, rim_dir=(0.0, -1.0), rim_v=0.9 * v, amb=col, amb_v=0.14, alpha=v, shade=shade)
        if white > 0.02: s.put(f, img, cx, cy, hf * s.H / s.ah, 0.0, 0.0, white) if False else s.put(f, img, cx, cy, hf * s.H / s.ah, v, 0.0, white)

    def copy(s, f, i, alpha=1.0, white=0.0):
        x, fy, hf = SPOT[i]; W, H = s.W, s.H
        s.spot(f, x * W, fy * H, hf * H * 0.42, alpha, TINT[i])
        s._pose = dict(t=0.0, sway=0.0, breath=0.0, lean=0.0, squash=0.0, stretch=None)
        s.figure(f, s.cop[i], x * W, fy * H - hf * H / 2, hf, TINT[i], alpha, white)

    def real(s, f, t, v=1.0, **kw):
        s.spot(f, s.CX, s.flr, s.HF * s.H * 0.42, v, GOLD)
        img = s.pose(t, **kw); s.figure(f, img, s.CX, s.CY, s.HF, GOLD, v)

    def gleam(s, f, v, t):
        if v <= 0.01: return
        p = s.at(*s.eR); s.flare(f, p, 9 * s.k * v, GOLD * 0.6 + 0.4, min(1.2, v), 1.4, 0.3 + t * 0.6, v > 0.8)

    def tiles(s, p, gain=0.5):
        wd = np.abs(np.cos(np.pi * p)); vis = sstep(0.0, 0.03, 0.5 * wd - 0.012 - np.abs(s.fu))
        em = np.maximum(np.abs(s.fu) / np.maximum(wd, 0.05), np.abs(s.fv)); edge = sstep(0.43, 0.47, em) * vis
        on = (p > 0.5).astype(np.float32); sh = s.bev * s.vel * (0.6 + 0.4 * wd)
        f = (s.hq * (gain * sh * on)[:, :, None] + np.array([0.030, 0.024, 0.045], np.float32) * ((1 - on) * (0.5 + 0.5 * s.bev))[:, :, None]) * vis[:, :, None]
        g = edge * on * gain; f += (g * 0.55 + cv2.GaussianBlur(g, (0, 0), 3 * s.k) * 0.5)[:, :, None] * GOLD[None, None, :]
        f += (s.stud * on * vis * gain * 1.6)[:, :, None] * (GOLD * 0.6 + 0.4)[None, None, :]
        return f * (0.40 + 0.60 * np.clip(1.25 - s.YY / s.H, 0, 1))[:, :, None]

    def lsil(s, f, img, cx, cy, sc, col, a=1.0, dark=0.0, rimv=1.0):
        """光の影: ふちは くっきり(拡大しても ぼけない)、中は 暗く、ふちが 色の光"""
        if a <= 0.01: return
        ph, pw = img.shape[:2]; M = np.float32([[sc, 0, cx - pw / 2 * sc], [0, sc, cy - ph / 2 * sc]]); col = np.asarray(col, np.float32)
        al = cv2.warpAffine(img[:, :, 3], M, (s.W, s.H), flags=cv2.INTER_LINEAR)
        if sc > 1.0: d = 0.45 / sc; al = sstep(0.5 - d, 0.5 + d, al)
        rgb = cv2.warpAffine(img[:, :, :3], M, (s.W, s.H), flags=cv2.INTER_LINEAR)
        body = rgb * (1 - dark) + col[None, None, :] * (0.05 * dark) + rgb * col[None, None, :] * 0.14 * dark
        m = (al * a)[:, :, None]; f[:] = f * (1 - m) + body * m
        r = max(2.0, min(14.0 * s.k, 3.0 * s.k * sc)); edge = np.clip(al - cv2.GaussianBlur(al, (0, 0), r), 0, 1) * 2.2
        f += ((edge + cv2.GaussianBlur(edge, (0, 0), 6 * s.k) * 0.8) * a * rimv)[:, :, None] * (col * 0.8 + 0.2)[None, None, :]

    def frame(s, t):
        s._t = t
        if t < T_B: f, sh = s.sc_solo(t)
        elif t < T_CUT: f, sh = s.sc_blink(t)
        elif t < T_D: f, sh = s.sc_cut(t - T_CUT)
        else: f, sh = s.sc_strike(t)
        f, sh = s.finale(f, sh, t, DUR, s.eR, (0.7, 0.4, 1), (0.45, 1, 0.4), -1)
        return s.film(f, t, sh * s.k, 0.6, 0.82, (0.95, 0.85, 1.0), 1.0 + 0.2 * sh, 0.04, 0.6, 1.15, 1.05)

    # 1) スポットライト ひとつ
    def sc_solo(s, t):
        f = s.stage(t); on = 1.0 if t > 0.35 else 0.0
        if 0.35 < t < 0.5: on = 0.4 + 0.6 * (int(t * 60) % 2)                            # 点くときの ちらつき
        s.real(f, t, on)
        if on: s.gleam(f, math.sin(clamp((t - 1.2) / 0.45) * 3.1416), t)
        return f, 0

    # 2) カチッ のたび べつの場所に
    def sc_blink(s, t):
        f = s.stage(t)
        if t < T_ALL:
            j = int((t - T_B) / BEAT); v = (t - T_B) - j * BEAT
            if v > 0.07: s.copy(f, j, 1.0, 0.5 * math.exp(-(v - 0.07) * 14))
            return f, 0
        v = t - T_ALL
        if v < 0.07: return f, 0
        for i in (0, 1, 4, 2, 3): s.copy(f, i, 1.0, 0.6 * math.exp(-(v - 0.07) * 9))
        return f, 6 * math.exp(-(v - 0.07) * 9)

    # 3) ひし形の窓。仮面の側は 色が 反転
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.tiles(np.clip((u * 2.2 - s.td / (W * 1.2)), 0, 1), 0.42)
        g = back(u / 0.3, 1.3); C = (W / 2, H * 0.38); hw, hh = W * 0.44 * g, H * 0.30 * g
        pts = [(C[0], C[1] - hh), (C[0] + hw, C[1]), (C[0], C[1] + hh), (C[0] - hw, C[1])]
        m = s.L0(); cv2.fillPoly(m, [np.array(pts, np.int32)], 1.0, cv2.LINE_AA)
        f *= (1 - 0.6 * cv2.GaussianBlur(m, (0, 0), 16 * k))[:, :, None]                 # 窓の 影が 壁に おちる
        win = np.zeros((H, W, 3), np.float32) + np.array([0.05, 0.03, 0.09], np.float32) + s.nebula(t, [(0.20, 0.10, 0.36), (0.10, 0.26, 0.14), (0.75, 0.55, 0.9)], 0.0014, (10.0, -6.0), 0.9)
        s.starfield(win, t, 500, GOLD * 0.7 + 0.3, 7, 0.7, (6.0, -22.0))
        zf = 1.0 + 0.05 * ease(u / 1.9); sc = zf * H / s.ah; img = s.pose(t)
        mx = C[0] - (s.fc[0] - s.aw / 2) * sc
        s.draw_lit(win, img, mx, C[1] + H * 0.03 - (s.fc[1] - s.ah / 2) * sc, zf, rim=GOLD * 0.6 + 0.4, rim_dir=(-0.6, -0.8), rim_v=0.9, amb=(0.9, 0.8, 1.0), amb_v=0.12, shade=0.97)
        sx = int(s.at(s.fc[0] + 8 * s.ah / 760.0, 0)[0])                               # 顔の まん中の線
        fl = clamp((u - 0.5) / 0.12)
        if fl > 0: win[:, sx:] = win[:, sx:] * (1 - fl) + (1.0 - np.clip(win[:, sx:], 0, 1)) * fl
        s.gleam(win, 1.2 * math.exp(-max(0.0, u - 0.5) * 4) * (u > 0.5), t)
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        s.glowline(f, pts, GOLD, 2.2, 0.75, 0.9, True)
        q = 1.045; s.glowline(f, [(C[0] + (x - C[0]) * q, C[1] + (y - C[1]) * q) for x, y in pts], GOLD * 0.8, 1.0, 0.45, 0.5, True)
        if fl > 0: s.glowline(f, [(sx, C[1] - hh * (1 - abs(sx - C[0]) / max(1.0, hw))), (sx, C[1] + hh * (1 - abs(sx - C[0]) / max(1.0, hw)))], WHITE, 1.0, 0.6 * fl, 0.6)
        for px_, py_ in pts: s.flare(f, (px_, py_), 5 * k * g, GOLD * 0.6 + 0.4, 0.5, 0.8, 0.0, False)
        f *= (1 - 0.7 * sstep(0.70, 0.79, s.YY / H) * sstep(0.96, 0.90, s.YY / H))[:, :, None]
        n = len(s.ht); step = W * 0.175
        for i, T in enumerate(s.ht):                                                 # 技名: 上と下から たがいちがいに
            a = (u - 0.6 - i * 0.11) / 0.22
            if a > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.835 + (1 - ease(a)) * 60 * k * (1 if i % 2 else -1), clamp(a * 2), 1.0)
        for i, T in enumerate(s.vo):
            if u > 0.45 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.755, 1.0, 1.0)
        if u > 1.3: s.text(f, s.txt_en, W / 2, H * 0.835 + 60 * k, ease((u - 1.3) / 0.3), 1.0)
        if u < 0.08: f = f + (1 - u / 0.08) * 0.7
        return f, 0

    # 4) 壁が めくれ、本物が まん中に。指を 鳴らすたび 分身が 光の影に なって こちらへ → さいごに 本物
    def sc_strike(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_D
        f = s.tiles(np.clip((u - s.td / (W * 2.6)) / 0.35, 0, 1), 0.36 + 0.2 * clamp((u - s.L0T) / 2.2))
        f += s.nebula(t, HAZE, 0.0008, (5.0, 2.0), 0.25)
        kick = 0.0; tint = None
        for j, i in enumerate(s.LEAP):
            v0 = u - (s.L0T + j * s.LGAP)
            if v0 > 0 and math.exp(-v0 * 12) > kick: kick = math.exp(-v0 * 12); tint = TINT[i]
        for j, i in enumerate(s.LEAP):                                               # まだ 立っている 分身
            if u < s.L0T + j * s.LGAP: s.copy(f, i, 1.0, 0.0)
        r = u - 0.35; rl = (u - (s.L0T + 5 * s.LGAP + 0.15)) / 0.75                      # 本物: 出る → 指を鳴らす → さいごに とぶ
        if r > 0 and rl <= 0:
            s.real(f, t, 1.0, squash=0.07 * math.exp(-r * 10) * math.cos(r * 20) - 0.045 * kick); s.gleam(f, 0.9 * kick, t)
            if kick > 0.05:                                                          # 指先の 火花
                hp = s.at(*s.hilt); rr = s.rnd[:120]; a_ = rr[:, 0] * 6.2832; d_ = (1 - kick) * (20 + 90 * rr[:, 1]) * k
                s.splat(f, hp[0] + np.cos(a_) * d_, hp[1] + np.sin(a_) * d_, kick * rr[:, 2] * 1.5, (tint * 0.6 + 0.4) if tint is not None else GOLD, 0.9, 0.6)
            if r < 0.06: f = f + (1 - r / 0.06) * 0.35
        if tint is not None and kick > 0.3: f = f + (kick - 0.3) * 0.22 * tint[None, None, :]
        for j, i in enumerate(s.LEAP):                                               # とびかかる 分身(のこり影を ひく)
            v = (u - (s.L0T + j * s.LGAP)) / 0.55
            if v <= 0 or v >= 1: continue
            x, fy, hf = SPOT[i]
            for lag, aa in ((0.14, 0.22), (0.07, 0.40), (0.0, 1.0)):
                vv = v - lag
                if vv <= 0: continue
                e = vv ** 2.4; sc = (hf + 3.2 * e) * H / s.ah
                cx = x * W + (W / 2 - x * W) * min(1.0, vv * 2) + (x - 0.5) * W * 2.2 * e; cy = (fy - hf / 2) * H + (H * 0.45 - (fy - hf / 2) * H) * min(1.0, vv * 2) + H * 0.25 * e
                s.lsil(f, s.cop[i], cx, cy, sc, TINT[i], clamp((1 - v) * 4) * aa, clamp(e * 3 + (0.6 if lag else 0)), 0.9 + 0.8 * e)
            s.rushdust(f, t, (W / 2, H * 0.45), 700, 1.6, TINT[i] * 0.6 + 0.4, 0.9 * math.sin(v * 3.1416), 11 + i, 1.0)
        sh = 5 * kick
        if rl > 0:                                                                   # 本物: 逆光の 影に なって、仮面の目の 光が まん中に 来る
            q = clamp(rl); e = q ** 2.2; hf = s.HF + 3.4 * e; sc = hf * H / s.ah; img = s.pose(t)
            tx, ty = (s.eR[0] - s.aw / 2) * sc, (s.eR[1] - s.ah / 2) * sc
            cx = s.CX + (W / 2 - tx - s.CX) * clamp(rl * 1.5); cy = s.CY + (H * 0.45 - ty - s.CY) * clamp(rl * 1.5)
            f *= 1 - 0.5 * e; s.lsil(f, img, cx, cy, sc, GOLD, 1.0, clamp(q * 2.2), 0.8 + 1.2 * e); s._pl = (cx, cy, sc)
            p = s.at(*s.eR); R = (10 + 300 * e ** 1.5) * k                              # 仮面の目: アーモンド形の 金の光
            dx = (s.XX - p[0]) / (R * 1.9); dy = (s.YY - p[1]) / R; rr2 = dx * dx + dy * dy * (1 + 1.2 * np.abs(dx))
            ring = np.exp(-((np.sqrt(rr2) - 0.62) / 0.16) ** 2) + np.exp(-rr2 * 9) * 1.4
            f += (ring * (0.5 + 0.9 * q))[:, :, None] * (GOLD * 0.75 + 0.25 * PUR)[None, None, :]
            s.flare(f, p, (8 + 46 * e) * k, GOLD * 0.6 + 0.4, 0.8 + 0.6 * q, 1.6, 0.3 + t * 0.6, True)
            s.rushdust(f, t, p, 1500, 1.2 + 2.0 * q, GOLD * 0.7 + 0.3, 0.3 + 1.0 * q, 21, 1.0)
            f = s.godrays(f, p, 0.25 + 0.5 * q, 0.7, 10, 0.88, 0.5); sh += 10 * q ** 2
        if u < 0.1: f = f * (u / 0.1) + (1 - u / 0.1) * 0.5
        return f, sh


if __name__ == "__main__":
    r = Loki()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
