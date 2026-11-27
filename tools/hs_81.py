"""アルテア(★7 はじまりの光) 専用の必殺技「星冠ノ創世」— 高品質版(hs_fx)・通し直し。しずけさの 一点から 銀河が 生まれ、こちらへ ひろがる
まっくらな 中に 杖の 先の 一番星が 灯る → その 光で 姿が 浮かび上がる(星雲の 夜空) → 翼が ひらいて 金の 目が ひらく → 目もとの カットイン →
杖から 光の 矢が 空へ 飛び、12の 星を むすんで 星の 冠を 編む(前ふり) → 冠が 一点に ちぢむ(しずけさ) →
その 一点から 銀河が 生まれる。渦の うで(星の つぶ・ちりの すじ・青い 若い 星と 黄色い しん)が まわりながら 大きく なり、うでが こちらを 通りすぎ、しんの 光が 目の前を おおう → 白 → 暗転
使い方: python3 hs_81.py 出力.mp4 / python3 hs_81.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old81", "/home/claude/tools/pre_fin/hs_81_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = 12.9
GOLD, PALE, DEEP = old.GOLD, old.PALE, old.DEEP
T_EYE, T_CUT, T_CROWN, T_HUSH, T_BANG = old.T_EYE, old.T_CUT, old.T_CROWN, old.T_HUSH, old.T_BANG
T_HIT = 12.2
PINK = np.array([1.0, 0.45, 0.62], np.float32); BLUE = np.array([0.55, 0.75, 1.0], np.float32)


class Altea(FX, old.Altea):
    def __init__(s, W=540):
        old.Altea.__init__(s, W); s.fx_init(81); H = s.H; k = s.k
        rg = np.random.RandomState(811); n = 3600; s.g_r = (rg.rand(n) ** 0.75 * 1.25 + 0.03).astype(np.float32); s.g_a = rg.randint(0, 2, n).astype(np.float32); s.g_j = rg.randn(n).astype(np.float32); s.g_b = rg.rand(n).astype(np.float32); s.g_h = rg.rand(n).astype(np.float32)
        band = np.exp(-(((s.YY - H * 0.45) + (s.XX - W * 0.5) * 0.55) / (H * 0.16)) ** 2); s._mw = (band * s.fbm(0.0016, 3, 9, 4, 50 * k, 2)).astype(np.float32)                    # 天の川

    # ---- 星雲の 夜空 ----
    def night(s, t, lum=1.0, warm=0.0):
        W, H, k = s.W, s.H, s.k
        f = (DEEP * 0.10)[None, None, :] * np.ones((H, W, 1), np.float32) + s.nebula(t, [tuple(DEEP * 0.9), (0.20, 0.10, 0.36), tuple(PALE * 0.55 * (1 - warm) + GOLD * 0.55 * warm)], 0.0008, (5.0, 1.5), 0.9, 0)
        f += (s._mw * 0.30)[:, :, None] * (PALE * 0.7 + GOLD * 0.3 * warm + 0.1)[None, None, :]
        s.starfield(f, t, 1100, (0.85, 0.92, 1.0), 1, 0.9, (0.0, 3.0)); s.starfield(f, t, 160, tuple(GOLD * 0.8 + 0.2), 7, 1.5, (0.0, 2.0)); s.starfield(f, t, 90, tuple(BLUE), 9, 1.6, (0.0, 2.0))
        return (f * lum).astype(np.float32)

    def eyes(s, f, v, streak=1.0):
        if v <= 0.01: return
        old.Altea.eyes(s, f, min(1.0, v), streak); sc = s._pl[2]
        for e in (s.eyeL, s.eyeR): s.flare(f, s.at(*e), 7 * sc * s.k * 2, GOLD, 0.45 * min(1.2, v), 0.9 * streak, 0.0, False)

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(GOLD * 0.6 + 0.4), rim_dir=getattr(s, '_rdir', (-0.5, -0.86)), rim_v=0.7 * alpha, amb=(0.7, 0.8, 1.0), amb_v=0.12, aura=GOLD, aura_v=0.10, alpha=alpha, shade=0.92)

    def add(s, f, L, col, gain=1.0, b1=3, b2=12):
        s.glowmask(f, np.clip(L, 0, 1), col, gain * 0.8, 1.0, 1.0, float(b2))

    def frame(s, t):
        o = {}; s._rdir = (-0.5, -0.86)
        if t < T_CUT:
            f, sh, g = s.sc_dawn(t)
            if t > 0.3: s.flare(f, s.at(*s.staff), (8 + 10 * ease((t - 0.3) / 0.5)) * s.k, GOLD * 0.7 + 0.3, 0.9 * clamp((t - 0.3) / 0.3), 0.9, t * 0.6, t > 1.0)
        elif t < T_CROWN: f, sh, g = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.9)
        elif t < T_BANG: s._rdir = (0.0, -1.0); f, sh, g = s.sc_crown(t)
        else: s._rdir = (0.0, -1.0); f, sh = s.sc_genesis(t - T_BANG)
        f, sh = s.finale(f, sh, t, DUR, None, (1, 0.82, 0.3), (0.5, 0.2, 0.8), -1)
        return s.film(f, t, sh * s.k, 0.55, o.get("thr", 0.80), (1.0, 0.95, 0.85), 1.0 + 0.15 * sh, 0.04, o.get("vig", 0.55), 1.12, 1.10)

    def galaxy(s, f, O, S, tilt, rot, v=1.0):
        """渦の 銀河(式): 対数らせんの うで・ちりの 暗い すじ・もも色の 星雲・黄色い しん + 星の つぶ"""
        W, H, k = s.W, s.H, s.k; gx = (s.XX - O[0]) / S; gy = (s.YY - O[1]) / (S * tilt); r = np.hypot(gx, gy) + 1e-4; th = np.arctan2(gy, gx); lr = np.log(r)
        ph = 2 * (th - rot) - 5.2 * lr; arm = 0.5 + 0.5 * np.cos(ph); lane = 0.5 + 0.5 * np.cos(ph - 1.3)
        n1 = cv2.remap(s._tiles[0], ((th / 6.2832 * 512 - rot * 40) % 256).astype(np.float32), ((lr * 46) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        n2 = cv2.remap(s._tiles[2], ((th / 6.2832 * 1024 - rot * 80) % 256).astype(np.float32), ((lr * 120) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        disk = np.exp(-r / 0.42) * sstep(1.5, 0.9, r); a3 = arm ** 2.2
        I = disk * (0.10 + 1.25 * a3 * (0.45 + 1.0 * n1)) * (1 - 0.70 * lane ** 6 * sstep(0.08, 0.3, r) * (0.4 + 0.9 * n2))
        col = BLUE[None, None, :] * (I * sstep(0.10, 0.45, r))[:, :, None] * 1.3 + (GOLD * 0.9 + 0.1)[None, None, :] * (I * sstep(0.45, 0.05, r))[:, :, None] * 1.5
        col += (disk * a3 * sstep(0.74, 0.92, n2) * sstep(0.15, 0.4, r) * 1.6)[:, :, None] * PINK[None, None, :]                                                             # 星の 生まれる 雲
        col += (np.exp(-(r / 0.085) ** 2) * 2.4 + np.exp(-r / 0.20) * 0.5)[:, :, None] * np.array([1.0, 0.92, 0.70], np.float32)[None, None, :]                               # しん
        f += col * v
        rr = s.g_r; om = rot + 0.9 / np.sqrt(rr + 0.15) * 0.0; a = s.g_a * 3.1416 + 2.6 * np.log(rr) + om + s.g_j * (0.16 + 0.10 / (rr + 0.2)); x = O[0] + np.cos(a) * rr * S; y = O[1] + np.sin(a) * rr * S * tilt
        br = (0.25 + 0.9 * s.g_b ** 3) * np.exp(-rr / 0.7) * v * min(2.2, 0.6 + S / (W * 0.6)); m = s.g_h < 0.62
        s.splat(f, x[m], y[m], br[m], BLUE * 0.8 + 0.2, 0.9, 0.35); s.splat(f, x[~m], y[~m], br[~m] * 1.1, GOLD * 0.8 + 0.2, 0.9, 0.35)

    # 4) 創世: 一点から 銀河が 生まれ、こちらへ ひろがる
    def sc_genesis(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_BANG; g = ease(clamp(u / 1.1)); e = clamp((u - 0.75) / (T_HIT - T_BANG - 0.75)); ee = e ** 2.4
        O = (W / 2, H * (0.27 + 0.20 * ease2(e))); S = W * (0.02 + 0.50 * g) * (1 + 13 * ee); tilt = 0.50 + 0.48 * ease2(e); rot = u * 0.55 + g * 1.6 + 1.4 * ee
        f = s.night(t, 1.0 - 0.5 * e, 0.4)
        s.galaxy(f, O, S, tilt, rot, clamp(u / 0.25))
        L = s.L0()
        for i in range(3):                                                           # 生まれた しゅんかんの 輪
            v = (u - i * 0.18) / 1.1
            if 0 < v < 1: s.ell(L, O, (W * 1.5 * v, W * 0.8 * v), 0, (1 - v) ** 2.5 * 0.6, 2)
        s.glowmask(f, L, WHITE * 0.5 + GOLD * 0.5, 0.7, 1.0, 1.0, 10.0)
        img = s.pose(t, sway=1.1 * math.exp(-u * 1.8), breath=0.3, lean=-6 * k, stretch=s.wing + (-0.16 * back(u / 0.5),), jit=2.0 * k * clamp(1 - u / 0.5))
        cy = H * (0.735 - 0.04 * ease(u / 2.0) + 0.5 * ee); a_ = 1 - clamp((e - 0.55) / 0.3)
        if a_ > 0.02:
            f += (np.exp(-np.hypot(s.XX - W / 2, s.YY - cy) / (W * 0.35)) * 0.14 * a_)[:, :, None] * GOLD[None, None, :]
            s.draw(f, img, W / 2, cy, 0.49 + 0.05 * ease(u / 2.0), a_); s.eyes(f, 0.8 * a_, 0.8)
            sp = s.at(*s.staff); s.flare(f, sp, 12 * k, GOLD * 0.7 + 0.3, 0.9 * a_, 0.8, t * 1.5, False)
            if e < 0.3: s.glowline(f, [sp, O], GOLD * 0.7 + 0.3, 1.2, (0.4 + 0.25 * math.sin(u * 30)) * (1 - e / 0.3), 0.9)                                                   # 杖と 銀河の しんを つなぐ 光
        if e > 0: s.rushdust(f, t, O, 1500, 0.6 + 2.4 * e, PALE * 0.6 + 0.4, 0.2 + 0.7 * e, 4, 1.0)                                                                           # うでの 星が こちらを 通りすぎる
        f = s.godrays(f, O, 0.25 + 0.5 * e, 0.9, 10, 0.88, 0.5)
        if u < 0.35: f = f + (1 - u / 0.35) ** 1.5 * 1.5
        if e > 0.84: f = f + ((e - 0.84) / 0.16) ** 2 * 0.9
        return f, 12 * clamp(1 - u / 0.9) + 2 + 8 * ee

    # ---- この子だけの曲: リディアの かがやく 調べ。鐘の 一音 → ハープの 上昇 → 12の 星ぶんの 音階 → 無音 → 大合唱が ふくらんで こちらへ ----
    def audio(s, path):
        a = Snd(DUR, 81); D4 = 293.66
        lyd = [0, 2, 4, 6, 7, 9, 11]
        def fq(d, o=0): return D4 * 2 ** ((lyd[d % 7] + 12 * (d // 7 + o)) / 12.0)
        a.a += (a.tone(D4 / 4) + 0.5 * a.tone(D4 / 4 * 1.5)) * 0.10 * np.clip(a.t / 1.5, 0, 1) * a.gate(0, T_HUSH, 0.3)       # 低い 持続
        a.note(fq(0, 1), 0.3, 1.6, 0.22, "bell"); a.note(fq(4, 1), 0.3, 1.2, 0.10, "bell")                                    # 一番星
        for i in range(10): a.note(fq(i), 1.0 + i * 0.16, 0.5, 0.07, "bell")                                                   # 光が ひろがる
        a.boom(T_EYE, 58, 0.9); a.whoosh(T_EYE - 0.4, T_EYE, 0.35)
        for d in (0, 2, 4, 6, 8): a.note(fq(d), T_EYE, 1.4, 0.05, "pad", 0.05)
        a.boom(T_CUT, 48, 1.0)
        for i in range(5): a.note(fq(7 - i, 1), T_CUT + 0.45 + i * 0.17, 0.25, 0.14, "bell")                                   # 5文字
        for d in (0, 4): a.note(fq(d, -1), T_CUT, 1.8, 0.06, "pad", 0.2)
        st = 0.2; up = [0, 2, 4, 1, 3, 5, 4, 6, 8, 7, 9, 11]                                                                    # 12の 星 = 12の 音が のぼる
        for i in range(12):
            tk = T_CROWN + 0.5 + i * st + 0.22
            a.note(fq(up[i]), tk, 0.6, 0.13, "bell"); a.note(fq(up[i], -1), tk, 0.25, 0.06, "fm"); a.hat(tk, 0.05)
            if i % 2 == 0: a.note(fq([0, 3, 4, 3, 0, 4][i // 2], -2), tk, 0.45, 0.16, "saw")
        for d in (0, 2, 4, 6): a.note(fq(d, 1), T_CROWN + 0.5 + 12 * st + 0.4, 1.0, 0.05, "pad", 0.1)
        a.note(fq(7, 2), T_HUSH + 0.25, 0.5, 0.16, "sine")                                                                      # しずけさ: 高い 一音だけ
        a.boom(T_BANG, 42, 1.3, 0.7); a.boom(T_BANG + 0.22, 70, 0.7)
        a.a += a.lp(a.nz, 0.12) * a.env(T_BANG, 0.1, 1.3) * 0.4
        for j, d in enumerate((0, 2, 4, 6, 7, 9, 11, 14)): a.note(fq(d, -1 if j < 3 else 0), T_BANG + 0.05 + j * 0.05, T_HIT - T_BANG + 0.3, 0.055, "pad", 0.3)   # 大合唱
        mel = [7, 8, 9, 11, 9, 11, 14]
        for i, d in enumerate(mel): a.note(fq(d), T_BANG + 0.3 + i * 0.17, 0.5, 0.11, "bell"); a.note(fq(d, 1), T_BANG + 0.3 + i * 0.17, 0.3, 0.05, "bell")
        for i in range(14): a.note(fq(i % 7 + 7, 1), T_BANG + 0.3 + i * 0.085, 0.1, 0.025, "sine")
        a.rise(T_BANG + 0.75, T_HIT, 80, 220, 0.14); a.whoosh(T_BANG + 0.75, T_HIT, 0.9, 0.3)                                    # うでが こちらを 通りすぎる
        a.a += a.lp(a.nz, 0.05) * np.clip((a.t - T_BANG - 0.75) / (T_HIT - T_BANG - 0.75), 0, 1) ** 2 * (a.t < T_HIT) * 0.7
        a.boom(T_HIT, 40, 1.6, 0.7)
        for j, d in enumerate((0, 4, 7, 11, 14)): a.note(fq(d, 0), T_HIT + 0.02, 0.6, 0.06, "bell")
        s.fin_snd(a, DUR, 81); a.save(path)


if __name__ == "__main__":
    r = Altea()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
