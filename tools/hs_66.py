"""ラグナ(★5 終焉の竜騎士) 専用の必殺技「終焉ノ竜剣」— 高品質版(hs_fx)。話は v3 と 同じ: 敵は こちら側。剣先の 黒い穴が ふくらみ、こちらごと のみこむ
赤むらさきの 夕空(雲・星・地平の 残り火)と くずれた塔の 街なみ(窓あかり・屋根の ふち光)。右上から ななめに とびこんで 宙で 止まる → 剣先に 黒い点。
景色が 剣先へ 引きのばされて 曲がりはじめ、がれきが うずを巻いて 吸われる → 剣の形の 窓のカットイン →
黒い穴(光の輪・かたむいた 降着円盤・うしろの 光が 上へ まわりこむ 弧)が ふくらみ、剣先を はなれて こちらへ → 街も 空も こちらも のまれて まっ暗(目と 刃だけ)→ 白 → 暗転
使い方: python3 hs_66.py 出力.mp4 / python3 hs_66.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old66", "/home/claude/tools/pre_fin/hs_66_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = old.DUR; VIO, CRIM, PALE = old.VIO, old.CRIM, old.PALE
T_CUT, T_GROW, T_BLK = old.T_CUT, old.T_GROW, old.T_BLK
DUSK = [(0.30, 0.06, 0.20), (0.75, 0.18, 0.20), (1.0, 0.55, 0.30)]


class Ragna(FX, old.Ragna):
    def __init__(s, W=540):
        old.Ragna.__init__(s, W); s.fx_init(66); H, k = s.H, s.k
        s.crim = np.clip(s.city - cv2.warpAffine(s.city, np.float32([[1, 0, 0], [0, 1, 3 * k]]), (W, H)), 0, 1)                          # 屋根の ふち
        far = cv2.warpAffine(s.city, np.float32([[-0.8, 0, W * 0.93], [0, 0.62, s.gnd * 0.38 - 4 * k]]), (W, H)); s.far = np.clip(far, 0, 1) * (s.YY < s.gnd)
        s.cdet = 0.6 + 0.8 * s.fbm(0.01, 2, 4, 3, 0, 1); s.lampg = cv2.GaussianBlur(s.lamp, (0, 0), 4 * k)

    def world(s, t, suck=0.0, drain=0.0):
        W, H, k = s.W, s.H, s.k
        top = np.array([0.07, 0.02, 0.13], np.float32); low = np.array([0.95, 0.30, 0.20], np.float32)
        q = np.clip(s.yn * 1.12, 0, 1) ** 1.8; f = (top[None, None, :] * (1 - q)[:, :, None] + low[None, None, :] * q[:, :, None]).astype(np.float32)
        s.starfield(f, t, 700, PALE, 2, 0.9, (1.0, 0.0)); f *= 1.0
        b = cv2.remap(s._tiles[2], ((s.XX * 0.07 + t * 2.5) % 256).astype(np.float32), ((s.YY * 0.55) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        cl = sstep(0.50, 0.78, b * 0.7 + s.fbm(0.0012, t * 5, 0, 3, 0, 3) * 0.45) * sstep(0.05, 0.2, s.yn) * sstep(0.88, 0.6, s.yn)       # たなびく 雲(下が 夕日に そまる)
        f = f * (1 - 0.55 * cl)[:, :, None] + (cl * (0.25 + 0.9 * s.yn))[:, :, None] * np.array([0.95, 0.30, 0.34], np.float32)[None, None, :]
        f += (np.exp(-((s.yn - 0.93) / 0.10) ** 2) * np.exp(-((s.XX / W - 0.35) / 0.5) ** 2) * 0.55)[:, :, None] * np.array([1.0, 0.75, 0.40], np.float32)[None, None, :]   # 地平の 残り火
        fa = s.far[:, :, None]; f = f * (1 - fa * 0.85) + np.array([0.26, 0.07, 0.16], np.float32)[None, None, :] * fa * 0.85              # 遠くの 街(かすむ)
        f += s.nebula(t, DUSK, 0.0012, (8.0, 0.0), 0.22) * sstep(0.55, 0.9, s.yn)[:, :, None]
        c = s.city[:, :, None]; f = f * (1 - c) + (np.array([0.035, 0.014, 0.05], np.float32)[None, None, :] * s.cdet[:, :, None]) * c
        f += (cv2.GaussianBlur(s.crim, (0, 0), 0.8 * k) * 0.55)[:, :, None] * np.array([1.0, 0.42, 0.30], np.float32)[None, None, :]
        fl = 0.75 + 0.25 * np.sin(t * 7 + s.XX * 0.3); f += ((s.lamp * 0.9 + s.lampg * 0.8) * fl)[:, :, None] * np.array([1.0, 0.75, 0.35], np.float32)[None, None, :]
        for r in s.rk:                                                               # 宙の がれき(かげ + 夕日の ふち)
            x0, y0 = r[0] * W, (0.12 + 0.6 * r[1]) * H; g = clamp(suck * (1.2 + r[2]) - r[3] * 0.5)
            a = math.atan2(y0 - s.HC[1], x0 - s.HC[0]) + g * 3.5; d = math.hypot(x0 - s.HC[0], y0 - s.HC[1]) * (1 - g)
            x, y = s.HC[0] + math.cos(a) * d, s.HC[1] + math.sin(a) * d; sz = (14 + 22 * r[4]) * k * (1 - 0.7 * g); ro = r[5] * 6 + t * (r[4] - 0.5) * 0.6 + g * 5
            P = [(x + math.cos(ro + j * 1.2566) * sz * (0.6 + 0.4 * ((j * 3 + int(r[5] * 9)) % 3) / 2.0), y + math.sin(ro + j * 1.2566) * sz * (0.6 + 0.4 * ((j * 3 + int(r[5] * 9)) % 3) / 2.0)) for j in range(5)]
            if g < 0.98:
                cv2.fillPoly(f, [np.array(P, np.int32)], (0.05, 0.02, 0.07), cv2.LINE_AA); Q = sorted(P, key=lambda p: -p[1])[:3]
                cv2.polylines(f, [np.array(sorted(Q, key=lambda p: p[0]), np.int32)], False, (0.9, 0.36, 0.28), max(1, int(1.5 * k)), cv2.LINE_AA)
        mist = s.fbm(0.002, t * 10, 0, 3, 0, 0) * np.exp(-((s.YY - s.gnd) / (H * 0.07)) ** 2); f += (mist * 0.25)[:, :, None] * np.array([0.9, 0.35, 0.35], np.float32)[None, None, :]
        if drain > 0: gy_ = f.mean(2, keepdims=True); f = f * (1 - drain) + gy_ * (VIO * 0.9 + 0.1)[None, None, :] * drain
        return f

    def hole(s, f, rb, t, v=1.0, cen=None):
        """黒い穴(式で 描く): かげ・光の輪・かたむいた 円盤(手前は かげの 前を 通る)・うしろの 光の 弧"""
        if rb < 1: return
        c = s.HC if cen is None else cen; k = s.k; dx = s.XX - c[0]; dy = s.YY - c[1]; d = np.hypot(dx, dy) / rb
        ca, sa = math.cos(math.radians(-22)), math.sin(math.radians(-22)); u = (dx * ca + dy * sa) / rb; w = (-dx * sa + dy * ca) / rb
        rd = np.hypot(u, w / 0.20); an = np.arctan2(w / 0.20, u)
        st = cv2.remap(s._tiles[1], (((an / 6.2832 + 0.5) * 256 * 3 + t * 90 + rd * 22) % 256).astype(np.float32), ((rd * 26) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        st2 = cv2.remap(s._tiles[3], (((an / 6.2832 + 0.5) * 256 * 9 + t * 150 + rd * 40) % 256).astype(np.float32), ((rd * 60) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        prof = sstep(1.25, 1.55, rd) * sstep(3.6, 2.0, rd); dop = 0.45 + 0.85 * (np.cos(an - 2.6) * 0.5 + 0.5) ** 1.5
        disk = prof * (0.25 + 1.6 * st * st2 * 2.0) * dop * v; heat = np.clip(disk * (1.5 - 0.28 * rd), 0, 1.6)
        dc = s.ramp(np.clip(heat / 1.6, 0, 1), [(0, (0.0, 0.0, 0.0)), (0.25, (0.45, 0.08, 0.40)), (0.55, (1.0, 0.28, 0.42)), (0.8, (1.0, 0.72, 0.78)), (1.0, (1.0, 1.0, 1.0))]) * 1.6
        arc = np.exp(-((d - 1.22) / 0.13) ** 2) * (0.35 + 0.9 * st) * np.clip(-w * 2 + 0.6, 0, 1) * v                                    # うしろの 円盤の 光が 上へ まわりこむ
        f += (dc * (w < 0)[:, :, None]) + arc[:, :, None] * (PALE * 0.6 + CRIM * 0.4)[None, None, :] * 0.8
        sh = sstep(1.0, 1.0 - 2.5 / rb, d); f *= (1 - sh)[:, :, None]                                                                 # かげ
        ph = (np.exp(-((d - 1.035) / 0.022) ** 2) * 1.5 + np.exp(-((d - 1.0) / 0.10) ** 2) * 0.35) * (d > 0.985) * v                    # 光の輪
        f += ph[:, :, None] * PALE[None, None, :]
        f += dc * (w >= 0)[:, :, None]                                                                                                # 手前の 円盤
        if rb < 120 * k: s.flare(f, (c[0] - ca * rb * 1.9, c[1] - sa * rb * 1.9), max(2.0, rb * 0.12), PALE, 0.5 * v, 1.2, 0.0, False)

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        s.draw_lit(f, img, cx, cy, hfrac, rim=np.array([1.0, 0.45, 0.40], np.float32), rim_dir=(-0.5, 0.86), rim_v=0.8, amb=(0.9, 0.7, 1.0), amb_v=0.14, alpha=alpha, shade=0.9)

    def add(s, f, L, col, gain=1.0, b1=3, b2=12):
        s.glowmask(f, np.clip(L, 0, 1), col, gain, 0.8, 1.0, float(b2))

    def eyes(s, f, v):
        if v <= 0.01: return
        sc = s._pl[2]
        for e in (s.eL, s.eR): p = s.at(*e); s.flare(f, p, 5 * sc * 2, CRIM, 0.6 * min(1.5, v), 0.9, 0.0, False)

    def edge(s, f, v):
        if v <= 0.01: return
        a = s.at(*s.hlt); b = s.at(*s.tip); s.glowline(f, [a, b], PALE, 1.6, 0.7 * min(1.5, v), 1.0)

    def frame(s, t):
        s._t = t; o = {}
        if t < T_CUT: f, sh = s.sc_open(t)
        elif t < T_GROW: f, sh = s.sc_cut(t - T_CUT); s.starfield(f, t, 300, PALE, 5, 0.5, (0.0, -30.0)); o = dict(vig=0.4, thr=0.9)
        else:
            f, sh = s.sc_eat(t)
            if t >= T_BLK: s.starfield(f, t, 120, CRIM * 0.6 + 0.4, 6, 0.5 * clamp((t - T_BLK) / 0.3), (0.0, -12.0)); o = dict(vig=0.3)
            else: f = s.godrays(f, s.HC, 0.25, 0.85, 8, 0.86, 0.3)
        f, sh = s.finale(f, sh, t, DUR, s.tip, (0.66, 0.34, 1), (1, 0.22, 0.3), -1)
        return s.film(f, t, sh * s.k, 0.6, o.get("thr", 0.80), (0.95, 0.75, 1.0), 1.0 + 0.25 * sh, 0.04, o.get("vig", 0.6), 1.15, 1.05)


if __name__ == "__main__":
    r = Ragna()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
