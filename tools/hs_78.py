"""オーロラ(★6 極光の女神) 専用の必殺技「極光ノ天幕」— 高品質版(hs_fx)・通し直し。光の 幕が 奥から 順に おりてきて、さいごの 1枚が こちらを つつむ
しずかな 雪原の 夜(雪の きめ・星・雪を かぶった 山)。流れ星のように おりてくる → 白い 光が ひたいの 星で 七色に 分かれる(プリズムのカットイン) →
両手を ひろげると、空に 光の カーテンが 1枚ずつ ひらく(雪に うつる。前ふり) → カーテンが 1枚ずつ こちらへ おりてきて、頭の 上を なでて 通りすぎる →
さいごの 1枚は 通りすぎない。たての 光の すじの 中に つつまれ、色が 緑 → もも色 → 白へ うつろう → 白 → 暗転
使い方: python3 hs_78.py 出力.mp4 / python3 hs_78.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old78", "/home/claude/tools/pre_fin/hs_78_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = 10.4
AG, AM, AB, SNOW, NIGHT, BOW = old.AG, old.AM, old.AB, old.SNOW, old.NIGHT, old.BOW
T_LAND, T_CUT, T_VEIL = old.T_LAND, old.T_CUT, old.T_VEIL
T_DOWN, T_LAST, T_HIT = 6.8, 8.5, 9.7


class Aurora(FX, old.Aurora):
    def __init__(s, W=540):
        old.Aurora.__init__(s, W); s.fx_init(78); H = s.H; k = s.k; s.yn = s.yn.astype(np.float32); s._av = 0.0
        hx = np.interp((np.arange(W) * 0.45 / k) % 255, np.arange(256), s._tiles[1][5]); hx2 = np.interp((np.arange(W) * 1.6 / k) % 255, np.arange(256), s._tiles[0][9])
        s._ry = (s.gnd - H * (0.04 + 0.24 * hx ** 1.3 + 0.02 * hx2)).astype(np.float32); s._mm = sstep(0, 2.5 * k, s.YY - s._ry[None, :]) * sstep(s.gnd + 2, s.gnd - 2, s.YY)
        s._cap = sstep(18 * k, 0, s.YY - s._ry[None, :]) * sstep(H * 0.05, H * 0.11, s.gnd - s._ry)[None, :]
        g0 = s.gnd; dz = (H - g0) / (np.clip(s.YY - g0, 0, None) + 5 * k); mx = (((s.XX - W / 2) * dz * 0.42 / k + 128) % 256).astype(np.float32); my = ((dz * 95) % 256).astype(np.float32)
        s._stx = cv2.remap(s._tiles[2], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP); s._sp = cv2.remap(s._tiles[0], (mx * 5) % 256, (my * 5) % 256, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP); s._gm = sstep(g0 - 1.5, g0 + 1.5, s.YY)
        s._mt = cv2.remap(s._tiles[3], ((s.XX * 1.6 / k) % 256).astype(np.float32), ((s.YY * 1.6 / k) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 雪原の 夜 ----
    def snowfield(s, t, dawn=0.0, lum=1.0):
        W, H, k = s.W, s.H, s.k; n = s.fbm(0.0009, t * 6, 0, 3, 50 * k, 0); av = s._av
        f = NIGHT[None, None, :] * ((0.5 + 1.5 * s.yn) * (0.7 + 0.6 * n))[:, :, None] + ((1 - s.yn) ** 2 * 0.05)[:, :, None] * np.array([0.6, 0.3, 1.0], np.float32)[None, None, :]
        s.starfield(f, t, 800, (0.9, 0.95, 1.0), 2, 0.9 - 0.4 * av, (0.0, 0.0))
        mc = (NIGHT * 1.5)[None, None, :] * (0.6 + 0.8 * s._mt)[:, :, None] + (s._cap * (0.10 + 0.14 * s._mt) * (1 + 1.5 * av))[:, :, None] * (SNOW * (1 - 0.5 * av) + AG * 0.5 * av)[None, None, :]
        f = f * (1 - s._mm[:, :, None]) + mc * s._mm[:, :, None]
        near = np.clip((s.YY - s.gnd) / (H - s.gnd), 0, 1); tw = sstep(0.90, 0.98, s._sp) * (0.5 + 0.5 * np.sin(t * 5 + s._stx * 40))
        sc_ = SNOW[None, None, :] * ((0.13 + 0.14 * s._stx) * (1.2 - 0.5 * near))[:, :, None] + (tw * 0.35)[:, :, None] + (av * (0.10 + 0.16 * s._stx) * (1 - 0.5 * near))[:, :, None] * AG[None, None, :]
        f = f * (1 - s._gm[:, :, None]) + sc_ * s._gm[:, :, None]; f += (np.exp(-((s.YY - s.gnd) / (6 * k)) ** 2) * 0.05)[:, :, None] * SNOW[None, None, :]
        return (f * lum).astype(np.float32)

    def veil(s, f, t, y0, amp, hgt, open_, ph, v=1.0):
        if open_ <= 0: return
        g = np.zeros_like(f); old.Aurora.veil(s, g, t, y0, amp, hgt, open_, ph, v)
        st = cv2.remap(s._tiles[1], ((s.XX * 2.2 / s.k + ph * 40 + t * 9) % 256).astype(np.float32), ((s.YY * 0.06 / s.k + ph * 20) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        g *= (0.45 + 1.1 * st)[:, :, None]; f += g * 0.85 + cv2.GaussianBlur(g, (0, 0), 6 * s.k) * 0.5

    def bigveil(s, f, t, p, i, last=False, wh=0.0):
        """こちらへ おりてくる 光の 幕 1枚。p: 0=空の 上 → 1=通りすぎた。last: 通りすぎずに つつむ"""
        if p <= 0: return
        W, H, k = s.W, s.H, s.k; x = s.xr / W
        yb = H * (0.12 + 1.55 * p ** 1.25) + H * 0.05 * (np.sin(x * 5.5 + t * 0.9 + i * 2.0) + 0.4 * np.sin(x * 13 - t * 1.4 + i))
        ray = np.interp((s.xr * 0.55 / k + t * 22 + i * 61) % 255, np.arange(256), s._tiles[i % 4][11 + i * 7]); ray = sstep(0.2, 0.9, ray)
        st = cv2.remap(s._tiles[(i + 1) % 4], ((s.XX * 2.4 / k + i * 30 - t * 14) % 256).astype(np.float32), ((s.YY * 0.05 / k + t * 3) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        d = yb[None, :] - s.YY; hgt = H * (0.45 + 1.0 * p) * (0.45 + 0.85 * ray); dc = np.clip(d, 0, None)
        I = (d > 0) * np.exp(-dc / hgt[None, :]) * (0.40 + 0.85 * ray)[None, :] * (0.35 + 1.2 * st); low = np.exp(-dc / (H * 0.12))
        fade = 1.0 if last else float(sstep(1.0, 0.70, p)); v = clamp(p / 0.12) * fade
        up = (AM * 1.1 + AB * 0.3) * 1.4 if last else ((AM * 0.8 + AB * 0.35) if i % 2 == 0 else (AB * 0.8 + AM * 0.45)); lowc = AG * (1 - 0.6 * wh) + AM * 0.6 * wh
        g = (I * low * 1.25)[:, :, None] * lowc[None, None, :] + (I * (1 - low))[:, :, None] * up[None, None, :] + (np.exp(-(d / (3.5 * k)) ** 2) * (0.5 + 0.6 * ray)[None, :] * 0.9)[:, :, None] * (AG * 0.6 + 0.4)[None, None, :]
        if wh > 0: g = g * (1 + 1.2 * wh) + (I * wh ** 2 * 1.2)[:, :, None]
        f += (g * 0.85 + cv2.GaussianBlur(g, (0, 0), 8 * k) * 0.5) * v

    def snow(s, f, t, v=1.0):
        r = s.rd[120:300]; s.splat(f, (r[:, 0] + 0.02 * np.sin(t + r[:, 3] * 9)) * s.W, ((r[:, 1] + t * (0.05 + 0.08 * r[:, 2])) % 1.0) * s.H, (0.15 + 0.35 * r[:, 4]) * v, SNOW, 1.1, 0.3)

    def hands(s, f, v, t):
        if v <= 0.01: return
        sc = s._pl[2]
        for hnd in (s.hL, s.hR): s.flare(f, s.at(*hnd), (6 + 1.5 * math.sin(t * 5)) * sc * 2 * v, AG * 0.5 + 0.5, 0.5 * v, 0.5, t * 0.6, False)
        s.flare(f, s.at(*s.gem), 6 * sc * 2 * v, (1.0, 1.0, 1.0), 0.5 * v, 0.7, 0.0, False)

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        a = s._av
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(SNOW * (1 - a) + (AG * 0.6 + 0.4) * a), rim_dir=(0.0, -1.0), rim_v=(0.6 + 0.4 * a) * alpha, amb=(0.6, 0.8, 1.0), amb_v=0.16, aura=AG, aura_v=0.06 + 0.10 * a, alpha=alpha, shade=0.88)
        if white > 0.05:
            w = img.copy(); w[:, :, :3] = 1.0; s.put(f, w, cx, cy, hfrac * s.H / s.ah, 0.9 * white * alpha, 0.0, 0.0)

    def add(s, f, L, col, gain=1.0, b1=3, b2=12):
        s.glowmask(f, np.clip(L, 0, 1), col, gain * 0.8, 1.0, 1.0, float(b2))

    def frame(s, t):
        o = {}; s._av = 0.0
        if t < T_CUT: f, sh = s.sc_fall(t); f = f * clamp(t / 0.4)
        elif t < T_VEIL: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.9)
        else: f, sh = s.sc_veil(t)
        f, sh = s.finale(f, sh, t, DUR, None, (0.4, 1, 0.8), (0.8, 0.5, 1), -1)
        return s.film(f, t, sh * s.k, 0.5, o.get("thr", 0.82), (0.9, 1.0, 0.95), 1.0 + 0.15 * sh, 0.04, o.get("vig", 0.55), 1.15, 1.12)

    # 1) 雪原に おりる
    def sc_fall(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_LAND
        f = s.snowfield(t, 0, 1.0); s.snow(f, t); cy = s.gnd - H * 0.235
        if u < 0:                                                                    # 流れ星
            a = ease2(t / T_LAND); q = np.clip(a - np.arange(160) * 0.003, 0, 1); xs = W * (0.95 - 0.45 * q); ys = H * (0.02 + (cy / H - 0.02) * q)
            s.splat(f, xs, ys, (1 - np.arange(160) / 160.0) ** 1.5 * 0.35, (0.8, 1.0, 0.9), 1.3, 0.6); s.flare(f, (xs[0], ys[0]), 12 * k, (0.9, 1.0, 0.95), 1.0, 0.8, 0.6, False)
        else:
            img = s.pose(t, squash=0.06 * math.exp(-u * 8) * math.cos(u * 18)); s.draw(f, img, W / 2, cy, 0.46, 1.0, clamp(1 - u / 0.35))
            L = s.L0()
            for i in range(3):                                                       # 雪の 上に 光の 輪
                v = (u - i * 0.14) / 0.9
                if 0 < v < 1: s.ell(L, (W / 2, s.gnd + 6 * k), (W * 0.8 * v, W * 0.10 * v), 0, 1 - v, 2)
            s.glowmask(f, L, AG * 0.6 + 0.4, 0.7, 1.0, 1.0, 8.0)
            if u < 0.1: f += (1 - u / 0.1) * 0.5
        return f, (4 * math.exp(-u * 8) if u > 0 else 0)

    # 2) プリズム: 白い 光が 七色に 分かれる
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT; s._av = 0.5
        f = (NIGHT * 0.8)[None, None, :] * np.ones((H, W, 1), np.float32) + s.nebula(t, [(0.02, 0.05, 0.14), (0.06, 0.10, 0.26), (0.3, 0.5, 0.6)], 0.0012, (5.0, 2.0), 0.5)
        s.starfield(f, t, 400, (0.9, 0.95, 1.0), 4, 0.5, (0.0, 0.0))
        zf = 0.80; sc = zf * H / s.ah; G = (W * 0.30, H * 0.20); cx = G[0] - (s.gem[0] - s.aw / 2) * sc; cy = G[1] - (s.gem[1] - s.ah / 2) * sc
        g = ease(u / 0.3); fan = ease((u - 0.3) / 0.35)
        if fan > 0:                                                                  # 七色の 光の 帯(ひたいの 星から 右下へ にじんで ひろがる)
            an = np.arctan2(s.YY - G[1], s.XX - G[0]); rr = np.hypot(s.XX - G[0], s.YY - G[1]); q = (an + 0.25) / 1.19 * 7; qi = np.clip(q, 0, 6.999); i0 = np.floor(qi).astype(np.int32); fr = (qi - i0)[:, :, None]
            B = np.array(BOW + [BOW[-1]], np.float32); col = B[i0] * (1 - fr) + B[i0 + 1] * fr
            st = cv2.remap(s._tiles[1], ((an * 260) % 256).astype(np.float32), ((rr * 0.05 / k - t * 6) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
            m = sstep(-0.3, 0.6, q) * sstep(7.3, 6.4, q) * sstep(W * 1.7 * fan, W * 1.7 * fan * 0.6, rr) * sstep(6 * k, 40 * k, rr) * (0.45 + 0.9 * st) * np.exp(-rr / (W * 1.6))
            f += col * (m * 1.25)[:, :, None]
        img = s.pose(t); s.draw(f, img, cx, cy, zf)
        s.glowline(f, [(-10, G[1] + (G[0] + 10) * 0.25), (-10 + (G[0] + 10) * g, G[1] + (G[0] + 10) * 0.25 * (1 - g))], (1.0, 1.0, 1.0), 2.5, 1.0, 1.2)                       # 入ってくる 白い 光
        s.flare(f, G, (10 + 12 * fan + 2 * math.sin(u * 9)) * k, (1.0, 1.0, 1.0), 1.0, 0.8, u * 0.5, True)
        for i, T in enumerate(s.vt):                                                 # 技名: 色の 帯の 上に 1字ずつ
            a = (u - 0.75 - i * 0.13) / 0.2
            if a > 0: s.text(f, T, W * 0.86, H * 0.38 + i * 78 * k, clamp(a), 1.0)
        for i, T in enumerate(s.vo):
            if u > 0.6 + i * 0.1: s.text(f, T, W * 0.86, H * 0.25 + i * 36 * k, 1.0, 1.0)
        if u > 1.5: s.text(f, s.txt_en, W * 0.40, H * 0.93, ease((u - 1.5) / 0.3), 1.0)
        if u < 0.08: f += (1 - u / 0.08) * 0.7
        return f, 0

    # 3) 光の カーテンが ひらく → 1枚ずつ こちらへ おりてくる → さいごの 1枚に つつまれる
    def sc_veil(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_VEIL; dn = t - T_DOWN; ls = t - T_LAST
        s._av = clamp(u / 1.5); f = s.snowfield(t, 0, 1.0); cy = s.gnd - H * 0.235
        for i, (y0, amp, hgt, ph) in enumerate(((0.26, 0.035, 0.12, 0.0), (0.38, 0.05, 0.15, 2.1), (0.50, 0.04, 0.13, 4.0))):
            gone = clamp((dn - i * 0.55) / 0.3) if dn > 0 else 0.0
            s.veil(f, t, H * y0, H * amp, H * hgt, ease2((u - 0.15 - i * 0.5) / 1.0), ph, 1.0 - gone)
        s.snow(f, t, 0.7); spread = back(u / 0.5, 1.4)
        img = s.pose(t, squash=-0.04 * math.exp(-u * 6)); s.draw(f, img, W / 2, cy, 0.46); s.hands(f, clamp(spread), t)
        sh = 0
        if dn > 0:
            for i in range(3): s.bigveil(f, t, clamp((dn - i * 0.55) / 1.15), i)                                      # 1枚ずつ おりて 通りすぎる
            if ls > 0:
                e = clamp(ls / (T_HIT - T_LAST)); s.bigveil(f, t, 0.15 + 0.85 * ease(clamp(ls / 0.8)), 3, True, ease2(clamp((ls - 0.15) / (T_HIT - T_LAST - 0.15))))
                s.rushdust(f, t, (W / 2, H * 0.3), 700, 0.5, (1.0, 0.9, 1.0), 0.3 * e, 5, 0.0); sh = 2 * e
                if e > 0.82: f = f + ((e - 0.82) / 0.18) ** 2 * 0.9
        if u < 0.15: f += (1 - u / 0.15) * 0.5
        return f, sh

    # ---- この子だけの曲: ガラスのような 高い 音(グラスハープ)と 鈴。カーテンが 1枚 ひらくたび 和音が ひとつ ふえる。おりてくる 幕ごとに 風と 和音、さいごは 長い 和音が 上がって いく ----
    def audio(s, path):
        a = Snd(DUR, 78); E4 = 329.63; lyd = [0, 2, 4, 6, 7, 9, 11]
        def fq(d, o=0): return E4 * 2 ** ((lyd[d % 7] + 12 * (d // 7 + o)) / 12.0)
        def glass(d, t0, dur, g=0.06, o=1): a.note(fq(d, o), t0, dur, g, "sine", 0.25); a.note(fq(d, o) * 2.003, t0, dur, g * 0.3, "sine", 0.3)
        a.a += a.lp(a.nz, 0.01) * 0.3 * a.gate(0, T_CUT, 0.4)                                                          # しずかな 風
        a.rise(0.1, T_LAND, 2400, 700, 0.05)                                                                           # 流れ星
        a.note(fq(7, 1), T_LAND, 1.0, 0.12, "bell"); a.boom(T_LAND, 70, 0.4, 0.25)
        for i in range(6): a.note(fq(i + 4, 2), T_LAND + 0.1 + i * 0.07, 0.25, 0.03, "bell")
        a.boom(T_CUT, 52, 0.8); a.note(fq(0, 2), T_CUT + 0.02, 0.6, 0.10, "sine", 0.01)
        for j in range(7): a.note(fq(j, 1), T_CUT + 0.32 + j * 0.05, 0.9, 0.07, "bell")                               # 七色
        for i in range(5): a.note(fq([7, 6, 4, 2, 0][i], 1), T_CUT + 0.75 + i * 0.13, 0.4, 0.09, "bell")
        chords = [(0, 4), (0, 2, 4), (0, 2, 4, 6)]
        for i, ch in enumerate(chords):                                                                                # カーテン 1枚ごとに
            tk = T_VEIL + 0.15 + i * 0.5
            for d in ch: glass(d, tk, 2.0 - i * 0.3, 0.05)
            a.whoosh(tk, tk + 0.9, 0.18, 0.08)
            for j in range(8): a.note(fq((j * 2 + i) % 12, 2), tk + j * 0.1, 0.2, 0.025, "bell")
        for i in range(3):                                                                                             # 1枚ずつ おりてくる
            tk = T_DOWN + i * 0.55; a.whoosh(tk, tk + 1.1, 0.5, 0.22); a.rise(tk, tk + 1.0, 900, 300, 0.05)
            for d in ((0, 4, 7), (2, 6, 9), (4, 7, 11))[i]: glass(d, tk + 0.1, 1.3, 0.055)
            for j in range(6): a.note(fq(12 - j * 2 + i, 2), tk + 0.3 + j * 0.07, 0.25, 0.035, "bell")
        a.whoosh(T_LAST, T_HIT, 0.6, 0.2); a.rise(T_LAST, T_HIT, 300, 1400, 0.07)                                       # さいごの 1枚に つつまれる
        for j, (d, o) in enumerate(((0, -1), (4, -1), (0, 0), (2, 0), (4, 0), (6, 0), (7, 0), (11, 0))): a.note(fq(d, o), T_LAST + j * 0.06, T_HIT - T_LAST + 0.4, 0.04, "pad", 0.4)
        for j, d in enumerate([0, 2, 4, 6, 7, 9, 11, 14, 16, 18]): a.note(fq(d, 1), T_LAST + 0.2 + j * 0.10, 0.5, 0.06, "bell")
        a.boom(T_HIT, 48, 0.9, 0.5); glass(14, T_HIT, 0.6, 0.07, 1)
        s.fin_snd(a, DUR, 78); a.save(path)


if __name__ == "__main__":
    r = Aurora()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
