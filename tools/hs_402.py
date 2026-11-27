"""タウロ(★4 牡牛座の斧戦士) 専用の必殺技「猛牛ノ一撃」— 高品質版(hs_fx)。大斧の 一撃で 地面が 割れ、地割れが こちらへ 走って 足もとが ひらく
赤い 岩の 荒野(台地・ひび割れた 地面)。足を ふみ鳴らし、鼻息の 湯気(前ふり) → 目の カットイン+技名 →
斧を 高く ふり上げ、地面に たたきつける → 赤く 光る 地割れが ジグザグに こちらへ 走る(岩が はね上がる)→
足もとまで 来ると 割れ目が 大きく ひらき、下からの 熱い 光と 岩の かけらが ふき上がって のみこまれる → 白 → 暗転
使い方: python3 hs_402.py 出力.mp4 / python3 hs_402.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 9.9
T_CUT, T_RAISE, T_SLAM, T_OPEN, T_HIT = 2.7, 4.7, 5.6, 6.95, 9.2
ROCK = np.array([0.46, 0.22, 0.13], np.float32); DUST = np.array([0.85, 0.62, 0.42], np.float32); HOT = np.array([1.0, 0.50, 0.12], np.float32); CORE = np.array([1.0, 0.92, 0.66], np.float32); SKY = np.array([0.30, 0.45, 0.70], np.float32)
EYE = (305, 290); AXE = (690, 340); NOSE = (305, 345)


class Tauro(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 402, (1.0, 0.6, 0.25), "猛牛ノ一撃", "Raging Bull", W); s.fx_init(402)
        s._vt = [s.text_img(c, hz.SERIF, int(66 * s.k), 3) for c in "猛牛ノ一撃"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; s.hy = 0.60; s.y0 = 0.795; H = s.H; k = s.k
        rg = np.random.RandomState(4020); s._rk = rg.rand(70, 7).astype(np.float32)
        st = np.floor(s.tex(1, s.XX * 0.10, 3.0) * 5) / 5.0; st2 = np.floor(s.tex(3, s.XX * 0.16 + 60, 9.0) * 4) / 4.0
        s._mesa = [(0.47 + 0.10 * st).astype(np.float32), (0.535 + 0.07 * st2).astype(np.float32)]
        d = np.clip((s.yn - s.y0) / (1 - s.y0), 0, 1.5)                                                               # 地割れの 道すじ(手前ほど 大きく ゆれる)
        s._fx = (W / 2 + (np.sin(d * 9.0 + 1.0) * 26 + np.sin(d * 23.0) * 9 + (s.tex(2, 5.0, s.YY * 0.8) - 0.5) * 30) * k * (0.25 + 1.6 * d)).astype(np.float32)
        s._fd = d.astype(np.float32)

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 赤い 岩の 荒野 ----
    def badland(s, t, lum=1.0, heat=0.0):
        W, H, k = s.W, s.H, s.k; hy = s.hy; v = (s.yn / hy).clip(0, 1)
        f = SKY[None, None, :] * (1.1 - 0.5 * v)[:, :, None] * (1 - v ** 2.0)[:, :, None] + np.array([1.0, 0.72, 0.45], np.float32)[None, None, :] * (v ** 2.0)[:, :, None]
        cl = s.fbm(0.0015, t * 6, 0, 4, 50 * k, 1); f += (sstep(0.5, 0.85, cl) * np.exp(-((s.yn - 0.2) / 0.16) ** 2) * 0.35)[:, :, None]
        for i, r in enumerate(s._mesa):                                              # 台地(ちそうの しま)
            m = sstep(r - 0.002, r + 0.002, s.yn)[:, :, None] * (s.yn < hy + 0.01)[:, :, None]; hz_ = 0.22 - 0.14 * i
            strata = 0.75 + 0.35 * s.tex(2, s.XX * 0.05, s.YY * 6.0) - 0.25 * sstep(0.9, 0.98, np.abs(np.cos(s.YY * 0.12 + i)))
            col = (ROCK * (1.0 - 0.2 * i))[None, None, :] * strata[:, :, None] * (1 - hz_) + np.array([1.0, 0.72, 0.5], np.float32)[None, None, :] * hz_
            f = f * (1 - m) + col * m
        d = np.clip((s.yn - hy) / (1 - hy), 0, 1); z = 1.0 / (d + 0.07); u = (s.xn - 0.5) * z * 60; v_ = z * 34
        n = s.tex(0, u, v_) * 0.6 + s.tex(2, u * 2.4, v_ * 2.4) * 0.4; cr = sstep(0.06, 0.0, np.abs(s.tex(1, u * 0.8, v_ * 0.8) - 0.5)) * (0.4 + 0.6 * d)        # 地面の ひび
        g = (ROCK * 1.15)[None, None, :] * (0.45 + 0.6 * n)[:, :, None] * (0.75 + 0.5 * d)[:, :, None] * (1 - 0.6 * cr)[:, :, None] + (DUST * 0.25)[None, None, :] * np.exp(-d / 0.08)[:, :, None]
        if heat > 0: g += (cr * heat * 0.8)[:, :, None] * HOT[None, None, :]
        gm = sstep(hy - 0.003, hy + 0.003, s.yn)[:, :, None]
        return ((f * (1 - gm) + g * gm) * lum).astype(np.float32)

    # ---- 地割れ: 赤く 光る 割れ目(front: 先の 位置 0..1、wide: ひらき)----
    def fissure(s, f, t, front, wide=0.0):
        if front <= 0: return
        W, H, k = s.W, s.H, s.k; d = s._fd; on = sstep(front + 0.03, front - 0.03, d) * (s.yn >= s.y0 - 0.004)
        w = ((4.0 + 70 * d ** 1.3) * k * (0.7 + 0.5 * s.tex(3, 9.0, s.YY * 1.5)) + wide * W * (0.15 + 1.2 * d)) * np.clip((front - d) * 6 + 0.25, 0.25, 1.0)
        q = np.abs(s.XX - s._fx) / np.maximum(w, 0.5)
        ins = sstep(1.0, 0.85, q) * on
        hotn = s.tex(1, s.XX * 0.5, s.YY * 0.5 - t * 60)
        col = CORE[None, None, :] * np.exp(-(q / 0.35) ** 2)[:, :, None] * 1.5 + HOT[None, None, :] * (np.exp(-(q / 0.8) ** 2) * (0.7 + 0.6 * hotn))[:, :, None] + np.array([0.25, 0.03, 0.0], np.float32)[None, None, :]
        f *= (1 - ins)[:, :, None]; f += col * ins[:, :, None]
        lip = np.exp(-((q - 1.12) / 0.14) ** 2) * on; f *= (1 - 0.5 * lip * (s.XX > s._fx))[:, :, None]; f += (lip * (s.XX < s._fx) * 0.35)[:, :, None] * DUST[None, None, :]      # めくれた ふち
        f += (np.exp(-(q / 3.5) ** 2) * on * (0.35 + 0.5 * wide))[:, :, None] * HOT[None, None, :]

    def rocks(s, f, t, u, n, origin_y, v=1.0, big=1.0):
        """はね上がる 岩の かけら(かげの ある かたまり)"""
        W, H, k = s.W, s.H, s.k; M = s.L0(); S = s.L0()
        for i in range(n):
            r = s._rk[i]; a = u - r[0] * 1.1
            if a <= 0 or a > 1.6: continue
            dd = 0.15 + 0.85 * r[1] ** 0.6; x0 = s._fx[min(H - 1, int(H * (s.y0 + dd * (1 - s.y0)))), 0]; y0 = H * (origin_y + dd * (1 - origin_y))
            x = x0 + (r[2] - 0.5) * W * 0.7 * a * (0.4 + dd) * big; y = y0 - (a * (1.3 + r[3]) - a * a * 1.1) * H * 0.30 * (0.3 + dd) * big
            sz = (4 + 22 * r[4] * (0.3 + dd)) * k * big; rot = r[5] * 6.28 + a * (r[6] - 0.5) * 9
            pts = np.array([(x + math.cos(rot + j * 1.2566 + 0.3 * math.sin(j * 7 + i)) * sz * (0.7 + 0.5 * ((i * 7 + j * 3) % 5) / 5.0), y + math.sin(rot + j * 1.2566) * sz * (0.7 + 0.5 * ((i * 3 + j * 5) % 5) / 5.0)) for j in range(5)], np.int32)
            al = min(1.0, a * 6) * (1 - max(0.0, a - 1.2) / 0.4) * v; cv2.fillPoly(M, [pts], float(al), cv2.LINE_AA); cv2.fillPoly(S, [pts], float(0.35 + 0.5 * r[6]), cv2.LINE_AA)
            cv2.fillPoly(S, [pts[[0, 1, 2]]], float(0.9), cv2.LINE_AA)
        M = cv2.GaussianBlur(M, (0, 0), 0.7 * k); col = (ROCK * 1.2)[None, None, :] * (S * (0.6 + 0.6 * s.tex(2, s.XX * 1.3, s.YY * 1.3)))[:, :, None] + (HOT * 0.25)[None, None, :] * (s.yn[:, :, None] > 0.7)
        f *= (1 - M)[:, :, None]; f += col * M[:, :, None]

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        h_ = getattr(s, '_ht', 0.0)
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(DUST * (1 - h_) + HOT * h_), rim_dir=(0.0, -1.0 + 2.0 * h_), rim_v=(0.45 + 0.4 * h_) * alpha, amb=(1.0, 0.85, 0.7), amb_v=0.12, aura=HOT, aura_v=0.16 * h_, alpha=alpha, shade=0.92 - 0.1 * h_)

    def frame(s, t):
        o = {}; s._ht = 0.0
        if t < T_CUT: f, sh = s.sc_stomp(t)
        elif t < T_RAISE: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.92)
        else: f, sh = s.sc_quake(t)
        f, sh = s.finale(f, sh, t, DUR, None, (1, 0.7, 0.3), (1, 0.4, 0.1), -1)
        return s.film(f, t, sh * s.k, 0.55, o.get("thr", 0.86), (1.0, 0.85, 0.65), 1.0 + 0.15 * min(sh, 8), 0.04, o.get("vig", 0.52), 1.15, 1.0)

    def bull(s, f, t, dy=0.0, alpha=1.0, **kw):
        W, H, k = s.W, s.H, s.k; cy = H * 0.62 + dy; fy = H * 0.62 + 0.40 * H * 0.46
        sh_ = s.L0(); cv2.ellipse(sh_, (int(W / 2), int(fy)), (int(W * 0.24), int(H * 0.013)), 0, 0, 360, 0.55 * alpha, -1, cv2.LINE_AA); f *= (1 - cv2.GaussianBlur(sh_, (0, 0), 4 * k))[:, :, None]
        img = s.pose(t, **kw); s.draw(f, img, W / 2, cy, 0.40, alpha)

    def puff(s, f, c, r, v, col=DUST):
        if v <= 0.01: return
        d = np.hypot(s.XX - c[0], (s.YY - c[1]) * 1.3) / max(r, 1.0); n = s.tex(2, s.XX * 1.1 + c[0], s.YY * 1.1)
        f += (np.exp(-(d / (0.7 + 0.5 * n)) ** 2) * v)[:, :, None] * np.asarray(col, np.float32)[None, None, :]

    # 1) 荒野。足を ふみ鳴らし、鼻息の 湯気
    def sc_stomp(s, t):
        W, H, k = s.W, s.H, s.k; sh = 0.0
        f = s.badland(t)
        st = [0.9, 1.5]; sq = sum(0.06 * math.exp(-max(0, t - a) * 9) * (t > a) for a in st)
        s.bull(f, t, 0.0, 1.0, sway=0.1, breath=0.7, squash=sq)
        for a in st:                                                                 # ふみ鳴らす 土けむり
            q = t - a
            if 0 < q < 1.0: s.puff(f, (W * 0.5 - 70 * k, H * 0.80 - q * 14 * k), (20 + 70 * q) * k, 0.5 * (1 - q)); s.puff(f, (W * 0.5 + 70 * k, H * 0.80 - q * 14 * k), (20 + 70 * q) * k, 0.5 * (1 - q)); sh += 6 * math.exp(-q * 9)
        q = t - 2.0
        if 0 < q < 0.9:                                                              # 鼻息
            np_ = s.at(*s.A(*NOSE))
            for sg in (-1, 1): s.puff(f, (np_[0] + sg * (14 + 90 * q) * k, np_[1] + 10 * k + 20 * k * q), (8 + 34 * q) * k, 0.75 * (1 - q / 0.9), (0.95, 0.95, 1.0))
        return f * clamp(t / 0.5), sh

    # 2) 目の カットイン(角の ように はねた 帯)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.badland(t, 0.30)
        a = ease(u / 0.24); cy = H * 0.33; hh = H * 0.105
        band = (ROCK * 0.6)[None, None, :] + (HOT * 0.6)[None, None, :] * (0.3 + 0.8 * s.tex(1, s.XX * 0.12 - t * 200, s.YY * 1.6))[:, :, None]
        ex, ey = s.A(*EYE); zf = 1.5; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(band, img, W / 2 - (ex - s.aw / 2) * sc + (1 - a) * W * 0.9 - u * 6 * k, cy - (ey - s.ah / 2) * sc, zf)
        bend = ((s.XX - W / 2) / (W / 2)) ** 2 * H * 0.045; dd = np.abs(s.YY - cy + bend); edge = hh * a
        m = (sstep(edge, edge - 2, dd) * (a > 0.02))[:, :, None]; f = f * (1 - m) + band * m
        s.glowmask(f, np.exp(-((dd - edge) / (1.8 * k)) ** 2) * (a > 0.02), DUST * 0.6 + CORE * 0.4, 0.9 * a, 1.0, 0.8, 8.0)
        n = len(s._vt); step = W * 0.165; sh = 5 * math.exp(-u * 6)
        for i, T in enumerate(s._vt):                                                # 技名: 1文字ずつ 大きく ぶつかる
            p = (u - 0.5 - i * 0.14) / 0.12
            if p > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.70 - (1 - ease(p)) * 50 * k, clamp(p * 2), 1.0); sh += 6 * math.exp(-max(0, p - 1) * 2.0) * (p > 1) * (p < 3)
        for i, T in enumerate(s._vo):
            if u > 0.35 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.70 - 70 * k, 1.0, 1.0)
        if u > 1.4:
            g = ease((u - 1.4) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.70 + 52 * k), (W / 2 + W * 0.42 * g, H * 0.70 + 52 * k)], DUST, 1.0, 0.8, 0.8)
            s.text(f, s.txt_en, W / 2, H * 0.70 + 78 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.7
        return f, sh

    # 3) ふり上げる → たたきつける → 地割れが こちらへ → 足もとが ひらく
    def sc_quake(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_RAISE; us = t - T_SLAM; uo = t - T_OPEN; sh = 0.0
        e = clamp(uo / (T_HIT - T_OPEN)); front = clamp(us / (T_OPEN - T_SLAM - 0.1)) ** 1.6 * 1.05 if us > 0 else 0.0; wide = e ** 1.5
        s._ht = clamp(us / 0.6) * (0.5 + 0.5 * e) if us > 0 else 0.0
        f = s.badland(t, 1.0 - 0.3 * e, 0.6 * s._ht)
        if us > 0:
            L = s.L0()                                                               # 衝撃の 輪(地面に そって)
            for i in range(3):
                q = (us - i * 0.08) / 0.7
                if 0 < q < 1: cv2.ellipse(L, (int(W / 2), int(H * s.y0)), (int(W * 1.1 * q), int(H * 0.16 * q)), 0, 0, 360, float((1 - q) ** 1.5), max(1, int(3 * k)), cv2.LINE_AA)
            s.glowmask(f, L, DUST * 0.6 + CORE * 0.4, 0.9, 1.0, 1.0, 9.0)
            s.fissure(f, t, front, wide)
        if us < 0: g = ease(u / 0.55); s.bull(f, t, -H * 0.035 * g, 1.0, sway=0.08, lean=-6 * k * g, squash=-0.09 * g)
        else: s.bull(f, t, H * 0.012, 1 - clamp((e - 0.55) / 0.25), lean=8 * k, squash=0.10 * math.exp(-us * 7), jit=2.5 * k * math.exp(-us * 4))
        ap = s.at(*s.A(*AXE)); s.flare(f, ap, (8 + 18 * ease(u / 0.55) * (us < 0) + 4 * math.sin(t * 9) ** 2) * k, CORE, 0.9 * (1 - clamp((e - 0.5) / 0.3)), 0.8, 0.5, False)
        if us > 0:
            s.rocks(f, t, us, 44, s.y0, 1.0, 1.0); sh += 18 * math.exp(-us * 4) + 3 * (front < 1.05) * (us < 1.4)
            for sg in (-1, 1): s.puff(f, (W / 2 + sg * (40 + 220 * us) * k, H * (s.y0 + 0.01)), (30 + 120 * us) * k, 0.6 * max(0.0, 1 - us / 1.0))
            if us < 0.1: f = f + (1 - us / 0.1) * 1.0
        if uo > 0:                                                                   # 足もとが ひらく: 熱い 光が 下から
            rise = H * (1.05 - 1.25 * e ** 1.6); gl = sstep(rise - H * 0.25, rise + H * 0.15, s.YY); hn = s.tex(1, s.XX * 0.4, s.YY * 0.4 - t * 150)
            f += (gl * (0.5 + 1.6 * e) * (0.6 + 0.7 * hn))[:, :, None] * (HOT * (1 - 0.5 * e) + CORE * 0.5 * e)[None, None, :]
            s.rocks(f, t, uo * 1.1, 70, 0.55, 1.0, 1.0 + 2.2 * e)
            sp = s._rk; x = (sp[:, 0] * W + np.sin(t * 5 + sp[:, 1] * 30) * 20 * k); y = (sp[:, 2] * H - t * H * (0.9 + 1.4 * sp[:, 3])) % H; s.splat(f, x, y, (0.5 + sp[:, 4]) * (0.3 + e), CORE, 1.3, 0.6)        # 火の粉
            f = s.godrays(f, (W / 2, H * 1.05), 0.3 + 0.5 * e, 0.95, 8, 0.88, 0.5); sh += 3 + 9 * e
            if e > 0.80: f = f + ((e - 0.80) / 0.20) ** 2 * 1.4
        return f, sh

    # ---- この子だけの曲: 太鼓の 足ぶみ → 低い 金管の うなり → ためて 一撃 → 岩の 割れる 音が こちらへ 近づく → 地の底の うなり ----
    def audio(s, path):
        a = Snd(DUR, 402); A = 110.0; mn = [0, 2, 3, 5, 7, 8, 10]
        def fq(d, o=0): return A * 2 ** ((mn[d % 7] + 12 * (d // 7 + o)) / 12.0)
        a.a += (a.lp(a.nz, 0.08) - a.lp(a.nz, 0.02)) * a.gate(0, T_SLAM, 0.2) * 0.5
        for t0 in (0.9, 1.5): a.boom(t0, 48, 1.3, 0.35); a.a += a.lp(a.nz, 0.2) * a.env(t0, 0.005, 0.15) * 0.4
        a.a += (a.nz - a.lp(a.nz, 0.4)) * a.env(2.0, 0.05, 0.3) * 0.35                                                 # 鼻息
        for j, d in enumerate((0, 4)): a.note(fq(d, -1), 0.2, 2.4, 0.07, "saw", 0.4)
        a.boom(T_CUT, 44, 1.2, 0.5); a.whoosh(T_CUT - 0.3, T_CUT, 0.3)
        for i in range(5): a.boom(T_CUT + 0.62 + i * 0.14, 58 - i * 3, 1.0, 0.2); a.note(fq([0, 2, 4, 7, 4][i], 0), T_CUT + 0.62 + i * 0.14, 0.4, 0.10, "saw", 0.01)
        for d in (0, 2, 4): a.note(fq(d, -1), T_CUT + 0.4, 1.6, 0.06, "saw", 0.2)
        a.rise(T_RAISE, T_SLAM, 80, 300, 0.14); a.note(fq(7, 1), T_RAISE + 0.5, 0.4, 0.08, "sine"); a.whoosh(T_SLAM - 0.2, T_SLAM, 0.7, 0.5)
        a.boom(T_SLAM, 34, 2.2, 1.0); a.boom(T_SLAM, 80, 1.0, 0.25); a.a += a.lp(a.nz, 0.25) * a.env(T_SLAM, 0.005, 0.4) * 0.7
        n = 16
        for i in range(n):                                                                                             # 割れる 音が 近づく(だんだん 大きく 低く)
            tk = T_SLAM + 0.1 + (T_OPEN - T_SLAM - 0.15) * (i / n) ** 0.75; g = 0.3 + 0.7 * i / n
            a.a += (a.nz - a.lp(a.nz, 0.5)) * a.env(tk, 0.001, 0.04) * 0.5 * g + a.lp(a.nz, 0.12) * a.env(tk, 0.003, 0.12) * 0.6 * g; a.boom(tk, 70 - 2 * i, 0.4 * g, 0.1)
        rum = a.lp(a.nz, 0.02) * (0.6 + 0.4 * np.sin(2 * np.pi * 19 * a.t)); a.a += rum * np.clip((a.t - T_SLAM) / 0.3, 0, 1) * (a.t < T_HIT) * (1.0 + 2.0 * np.clip((a.t - T_OPEN) / (T_HIT - T_OPEN), 0, 1))
        a.boom(T_OPEN, 38, 1.6, 0.8)
        for j, d in enumerate((0, 2, 4, 7, 9)): a.note(fq(d, -1 if j < 2 else 0), T_OPEN + j * 0.06, T_HIT - T_OPEN + 0.2, 0.055, "saw", 0.3)
        a.rise(T_OPEN, T_HIT, 60, 240, 0.14); a.whoosh(T_OPEN + 0.5, T_HIT, 0.8, 0.3)
        a.boom(T_HIT, 34, 1.8, 0.9)
        for j, d in enumerate((0, 4, 7, 11)): a.note(fq(d, 1), T_HIT + 0.02, 0.6, 0.06, "bell")
        s.fin_snd(a, DUR, 402); a.save(path)


if __name__ == "__main__":
    r = Tauro()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
