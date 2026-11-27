"""ショコラ(★4 チョコレートの騎士) 専用の必殺技「板チョコ砕キ」— 高品質版(hs_fx)。地面から 巨大な 板チョコの 壁が せり上がって 画面を ふさぎ、うしろから 二度 たたかれて みぞに ひびが 走り、ひとかけずつ 割れて こちらへ 飛んで くる
お菓子の 国(クリームの 雲・丘・ビスケットの 地面)に 立つ → 板チョコの ひとかけの 形の 窓の カットイン+技名 →
剣を 地面に つく → 板チョコの 壁が せり上がる(つやの ある ます目)→ どん(みぞが 金に 光る)→ どん(ひびが ふえる)→
まん中から ひとかけずつ 外れて 大きく なって 飛びさる・すき間から 金の 光と 本人 → 白 → 暗転
使い方: python3 hs_427.py 出力.mp4 / python3 hs_427.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.4
T_CUT, T_ACT, T_RISE, T_H1, T_H2, T_BRK, T_HIT = 2.7, 4.7, 5.0, 6.5, 7.1, 7.7, 9.5
CREAM = np.array([1.0, 0.93, 0.80], np.float32); PINK = np.array([1.0, 0.72, 0.74], np.float32); MINT = np.array([0.66, 0.90, 0.78], np.float32); BISC = np.array([0.90, 0.72, 0.46], np.float32); WHIP = np.array([1.0, 0.98, 0.94], np.float32)
CHOC = np.array([0.26, 0.13, 0.06], np.float32); CHOL = np.array([0.56, 0.33, 0.17], np.float32); GOLD = np.array([1.0, 0.82, 0.40], np.float32)
EYE = (410, 275); SWORD = (75, 665)
NC, NR = 3, 5


class Chocola(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 427, (0.9, 0.65, 0.35), "板チョコ砕キ", "Cocoa Breaker", W); s.fx_init(427)
        s._vt = [s.text_img(c, hz.SERIF, int(62 * s.k), 3) for c in "板チョコ砕キ"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; s.hy = 0.58
        s._slab0 = s.slabimg(0.0)

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- お菓子の 国 ----
    def candy(s, t, lum=1.0):
        W, H, k = s.W, s.H, s.k; hy = s.hy; v = (s.yn / hy).clip(0, 1)
        f = np.array([0.40, 0.70, 1.0], np.float32)[None, None, :] * (1 - v ** 1.3)[:, :, None] + (CREAM * 1.0)[None, None, :] * (v ** 1.3)[:, :, None] + (PINK * 0.25)[None, None, :] * (v * (1 - v) * 2)[:, :, None]
        n = s.fbm(0.0009, t * 4, 0, 3, 80 * k, 1); n = cv2.GaussianBlur(n, (0, 0), 5 * k); cu = sstep(0.50, 0.60, n + 0.08 * np.exp(-((s.yn - 0.25) / 0.14) ** 2))                # クリームの 雲
        sh_ = np.clip(0.90 + 0.5 * (cv2.GaussianBlur(n, (0, 0), 6 * k) - n) * 8 + 0.10 * (0.4 - s.yn), 0.86, 1.08)
        f = f * (1 - cu * 0.95)[:, :, None] + (cu * 0.95 * sh_ * 1.12)[:, :, None] * WHIP[None, None, :] + (cu * (1 - sh_).clip(0, 1) * 0.5)[:, :, None] * PINK[None, None, :]
        for cxh, wd, hh, colr, ph in ((0.22, 0.42, 0.13, PINK, 0.0), (0.80, 0.50, 0.17, MINT, 2.0), (0.52, 0.34, 0.08, CREAM * 0.97, 4.0)):                              # 丘
            top = hy + 0.01 - hh * np.exp(-((s.xn - cxh) / wd) ** 2 * 2.2); hm = sstep(-0.003, 0.003, s.yn - top)
            hc = colr[None, None, :] * (0.80 + 0.22 * np.clip((top + hh - s.yn) / hh, 0, 1) + 0.10 * s.tex(1, s.XX * 0.5 + ph * 40, s.YY * 0.5))[:, :, None] + (np.exp(-np.clip(s.yn - top, 0, 1) / 0.012) * 0.25)[:, :, None]
            f = f * (1 - hm)[:, :, None] + hc * hm[:, :, None]
        d = np.clip((s.yn - hy) / (1 - hy), 0, 1); pu = (s.XX - W / 2) / (d * 0.9 + 0.25); pz = 1.0 / (d + 0.12)                                                       # ビスケットの 地面
        cu_ = pu / (90 * k); cv_ = pz * 2.2; chk = ((np.floor(cu_) + np.floor(cv_)) % 2); fu = (cu_ % 1.0); fv = (cv_ % 1.0)
        ln = np.maximum(np.exp(-((np.abs(fu - 0.5) - 0.5) / 0.04) ** 2), np.exp(-((np.abs(fv - 0.5) - 0.5) / 0.06) ** 2)); hole = np.exp(-((np.hypot(fu - 0.5, fv - 0.5)) / 0.07) ** 2)
        gc = BISC[None, None, :] * ((0.82 + 0.12 * chk + 0.22 * s.tex(1, pu * 0.4, pz * 60)) * (1 - 0.35 * ln) * (1 - 0.4 * hole) * (0.85 + 0.25 * d))[:, :, None]
        m = sstep(hy - 0.002, hy + 0.003, s.yn)[:, :, None]; f = f * (1 - m) + gc * m
        return (f * lum).astype(np.float32)

    def sugar(s, f, t, v=1.0):
        W, H = s.W, s.H; r = s.rnd[:80]
        x = ((r[:, 0] + 0.02 * np.sin(t * (1 + r[:, 2]) + r[:, 3] * 6.28)) % 1.0) * W; y = ((r[:, 1] + t * (0.02 + 0.03 * r[:, 4])) % 1.0) * H
        s.splat(f, x, y, (0.3 + 0.5 * r[:, 2]) * v, WHIP, 1.1, 0.2)

    # ---- 板チョコ(式): ます目・みぞ・面取りの 光と かげ・つや。crack で みぞが 金に 光り ひびが 入る ----
    def slabimg(s, crack):
        W, H, k = s.W, s.H, s.k; cw = W / NC; chh = H / NR; u = (s.XX / cw) % 1.0; v = (s.YY / chh) % 1.0
        ex = np.minimum(u, 1 - u) * cw; ey = np.minimum(v, 1 - v) * chh; e = np.minimum(ex, ey); g0, b1 = 5 * k, 24 * k
        isx = ex < ey; lightside = np.where(isx, u < 0.5, v < 0.5)
        bev = sstep(g0, g0 + 2 * k, e) * sstep(b1 + 1.5 * k, b1 - 1.5 * k, e); flat = sstep(b1 - 1.5 * k, b1 + 1.5 * k, e)
        tx = s.tex(1, s.XX * 0.35, s.YY * 0.35); gl = np.exp(-(((u + v) - 0.72) / 0.07) ** 2) * 0.30 + np.exp(-(((u + v) - 0.92) / 0.03) ** 2) * 0.16
        val = 0.34 * (1 - bev - flat).clip(0, 1) + bev * np.where(lightside, np.where(isx, 1.25, 1.55), np.where(isx, 0.55, 0.42)) + flat * (0.92 + 0.14 * tx + gl)
        img = CHOC[None, None, :] * (val * 0.9)[:, :, None] + (CHOL - CHOC)[None, None, :] * (np.clip(val - 0.8, 0, 1.2) * 0.9)[:, :, None]
        img += (np.exp(-(np.hypot(u - 0.5, (v - 0.5) * chh / cw) / 0.16) ** 2) * flat * 0.05)[:, :, None]
        if crack > 0.01:
            gr = np.exp(-e / (4.5 * k)) * crack * (0.6 + 0.6 * s.tex(0, s.XX * 0.8, s.YY * 0.8))
            jag = np.exp(-(np.abs(s.tex(2, s.XX * 0.12 + 13, s.YY * 0.12) - 0.5) / (0.0035 * crack + 1e-4)) ** 2) * flat * crack * np.clip(1.5 - np.hypot(s.xn - 0.5, (s.yn - 0.45) * 1.4) * 2.0, 0, 1)
            img = img * (1 - jag * 0.85)[:, :, None] + ((gr * 1.6 + jag * 0.15)[:, :, None]) * GOLD[None, None, :]
        return img.astype(np.float32)

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        s.draw_lit(f, img, cx, cy, hfrac, rim=(1.0, 0.95, 0.85), rim_dir=(-0.5, -0.8), rim_v=0.4 * alpha, amb=(1.0, 0.95, 0.9), amb_v=0.08, aura=GOLD, aura_v=0.05, alpha=alpha, shade=0.98)

    def frame(s, t):
        o = {}
        if t < T_CUT: f, sh = s.sc_land(t)
        elif t < T_ACT: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.97)
        else: f, sh = s.sc_break(t)
        f, sh = s.finale(f, sh, t, DUR, None, (1.0, 0.95, 0.85), (1.0, 0.8, 0.45), -1)
        return s.film(f, t, sh * s.k, 0.35, o.get("thr", 0.96), (1.0, 0.97, 0.92), 1.0 + 0.15 * min(sh, 8), 0.03, o.get("vig", 0.35), 1.15, 0.97)

    def knight(s, f, t, alpha=1.0, **kw):
        W, H, k = s.W, s.H, s.k; cy = H * 0.60; fy = cy + 0.40 * H * 0.47
        sd = np.exp(-(((s.XX - W * 0.5) / (W * 0.24)) ** 2 + ((s.YY - fy) / (H * 0.014)) ** 2)); f *= (1 - 0.4 * alpha * sd)[:, :, None]
        img = s.pose(t, **kw); s.draw(f, img, W / 2, cy, 0.40, alpha); return s.at(*s.A(*SWORD))

    # 1) お菓子の 国に 立つ
    def sc_land(s, t):
        f = s.candy(t); s.knight(f, t, clamp(t / 0.6), sway=0.25, breath=0.5); s.sugar(f, t, 0.6)
        return f * clamp(t / 0.5), 0

    # 2) 目の カットイン(板チョコの ひとかけの 窓)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.candy(t, 0.36)
        a = back(clamp(u / 0.35)); c = (W / 2, H * 0.32); hw = W * 0.40 * a; hh = H * 0.125 * a
        bandc = (CHOC * 1.4)[None, None, :] + (GOLD * 0.7)[None, None, :] * (0.25 + 0.9 * sstep(0.45, 0.9, s.tex(0, s.XX * 0.4 + t * 50, s.YY * 0.4 - t * 20)))[:, :, None]
        ex, ey = s.A(*EYE); zf = 1.45; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(bandc, img, c[0] - (ex - s.aw / 2) * sc, c[1] - (ey - s.ah / 2) * sc, zf)
        if hw > 8:
            qx = np.abs(s.XX - c[0]) - hw; qy = np.abs(s.YY - c[1]) - hh; rr = 10 * k; d = np.hypot(np.maximum(qx + rr, 0), np.maximum(qy + rr, 0)) + np.minimum(np.maximum(qx, qy) + rr, 0) - rr
            m = sstep(0, -2, d)[:, :, None]; f = f * (1 - m) + bandc * m
            bw = 20 * k; fr = sstep(-1, 1, d) * sstep(bw + 1, bw - 1, d); lightside = ((s.YY - c[1]) / (hh + bw) + (s.XX - c[0]) / (hw + bw) * 0.6) < 0                 # 面取りの 枠
            fv = np.where(lightside, 1.5, 0.55) * (0.85 + 0.3 * s.tex(1, s.XX * 0.4, s.YY * 0.4)) + np.exp(-((d - bw * 0.5) / (2 * k)) ** 2) * 0.3
            f = f * (1 - fr)[:, :, None] + (CHOC * 1.3)[None, None, :] * (fv * fr)[:, :, None] + (CHOL - CHOC)[None, None, :] * (np.clip(fv - 0.8, 0, 1) * fr * 0.8)[:, :, None]
            f *= (1 - 0.35 * sstep(bw, bw + 10 * k, d) * sstep(bw + 22 * k, bw + 8 * k, d))[:, :, None]
        n = len(s._vt); step = W * 0.148
        for i, T in enumerate(s._vt):                                                # 技名: ぱきっ と 1文字ずつ(少し かたむいて はまる)
            p = (u - 0.5 - i * 0.11) / 0.16
            if p > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.69 - (1 - min(1.0, p)) ** 2 * 40 * k, clamp(p * 2), 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.35 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.69 - 70 * k, 1.0, 1.0)
        if u > 1.4:
            g = ease((u - 1.4) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.69 + 52 * k), (W / 2 + W * 0.42 * g, H * 0.69 + 52 * k)], GOLD, 1.0, 0.7, 0.7)
            s.text(f, s.txt_en, W / 2, H * 0.69 + 78 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.5
        return f, 3 * math.exp(-u * 6)

    # 3) 剣を つく → 壁が せり上がる → どん・どん → ひとかけずつ 飛んで くる
    def sc_break(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_ACT; sh = 0.0
        e = clamp((t - T_BRK) / (T_HIT - T_BRK))
        f = s.candy(t)
        sq = 0.06 * ease(u / (T_RISE - T_ACT)) if t < T_RISE else -0.03 * math.exp(-(t - T_RISE) * 4)
        sp = s.knight(f, t, 1.0, sway=0.2, breath=0.4, squash=sq)
        s.sugar(f, t, 0.6)
        if e > 0:                                                                    # うしろ: 金の 光
            C = (W * 0.5, H * 0.45); f = f * (1 - 0.15 * e) + (np.exp(-np.hypot(s.XX - C[0], s.YY - C[1]) / (W * (0.25 + 0.9 * e))) * (0.5 + 1.2 * e))[:, :, None] * GOLD[None, None, :]
            f = s.godrays(f.astype(np.float32), C, 0.15 + 0.45 * e, 1.05, 8, 0.88, 0.4)
        rz = t - T_RISE
        if rz >= 0:
            rise = 1 - (1 - clamp(rz / 1.05)) ** 3; top = H * (1.02 - 1.06 * rise)
            crack = 0.0 if t < T_H1 else (0.5 + 0.1 * clamp((t - T_H1) / 0.5) if t < T_H2 else 1.0 + 0.4 * clamp((t - T_H2) / 0.6))
            sl = s._slab0 if crack <= 0 else s.slabimg(crack)
            if t < T_BRK:
                bump = sum(math.exp(-(t - th) * 9) for th in (T_H1, T_H2) if t >= th)                                                    # たたかれて ふくらむ
                C = (W * 0.5, H * 0.45); dx = s.XX - C[0]; dy = s.YY - C[1]; rr2 = (dx * dx + dy * dy) / (W * 0.5) ** 2; sc_ = 1 + 0.07 * bump * np.exp(-rr2 * 1.5)
                mx = (C[0] + dx / sc_).astype(np.float32); my = (C[1] + dy / sc_ - top).astype(np.float32)
                wimg = cv2.remap(sl, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                wm = sstep(-1.0, 1.0, s.YY - top)
                rim = sstep(-16 * k, -14 * k, s.YY - top) * (1 - wm); f = f * (1 - rim)[:, :, None] + (rim * (0.95 + 0.3 * s.tex(1, s.XX * 0.5, 3)))[:, :, None] * CHOL[None, None, :]      # 上の 面
                f = f * (1 - wm)[:, :, None] + wimg * wm[:, :, None]
                f *= (1 + 0.5 * bump * np.exp(-rr2 * 2.0))[:, :, None]
                if rise < 1: sh += 4 * (1 - rise) + 2; 
                if rz < 0.5:                                                         # 地面の くず
                    n = 60; r = s.rnd[150:150 + n]; s.splat(f, W * r[:, 0], top - (rz * (0.5 + r[:, 1]) * 300 - rz * rz * 600) * k, (0.5 + 0.5 * r[:, 2]) * (1 - rz / 0.5), BISC, 1.6, 0.2)
                sh += 12 * bump
                for th in (T_H1, T_H2):
                    if 0 <= t - th < 0.08: f = f + (1 - (t - th) / 0.08) * 0.3
            else:                                                                    # ひとかけずつ 外れて 飛んで くる(まん中から)
                cw = W / NC; chh = H / NR; q = t - T_BRK; order = []
                for iy in range(NR):
                    for ix in range(NC):
                        cx0 = (ix + 0.5) * cw; cy0 = (iy + 0.5) * chh; dc = math.hypot((cx0 - W * 0.5) / W, (cy0 - H * 0.45) / H); order.append((dc, ix, iy, cx0, cy0))
                for dc, ix, iy, cx0, cy0 in sorted(order, reverse=True):
                    r = s.rnd[300 + iy * NC + ix]; p = clamp((q - dc * 1.5 - 0.10 * r[0]) / 1.0); g = p ** 2.2
                    if p >= 1: continue
                    scl = 1 + 3.2 * g; ox = (cx0 - W * 0.5) * 2.4 * g + (r[1] - 0.5) * W * 0.5 * g; oy = (cy0 - H * 0.45) * 2.2 * g + H * 0.25 * g * g; an = (r[2] - 0.5) * 1.6 * g
                    ca, sa = math.cos(an), math.sin(an); lx = (s.XX - cx0 - ox) / scl; ly = (s.YY - cy0 - oy) / scl; sx = cx0 + lx * ca + ly * sa; sy = cy0 - lx * sa + ly * ca
                    inm = sstep(cw / 2 + 0.6, cw / 2 - 0.6, np.abs(sx - cx0)) * sstep(chh / 2 + 0.6, chh / 2 - 0.6, np.abs(sy - cy0)) * clamp((1 - p) * 6)
                    if inm.max() < 0.01: continue
                    timg = cv2.remap(sl, sx.astype(np.float32), sy.astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
                    edge = (np.maximum(np.abs(sx - cx0) / (cw / 2), np.abs(sy - cy0) / (chh / 2)) > 0.94) * (p > 0.01)
                    f = f * (1 - inm)[:, :, None] + (timg * (1.0 + 0.3 * g) * (1 - 0.4 * edge)[:, :, None]) * inm[:, :, None]
                n = 110; r = s.rnd[100:100 + n]; a = r[:, 0] * 6.283; rr = W * (0.05 + 1.0 * np.clip(q * (0.4 + 0.9 * r[:, 1]), 0, 2) ** 0.8)                      # かけら
                s.splat(f, W * 0.5 + np.cos(a) * rr, H * 0.45 + np.sin(a) * rr * 1.3 + q * q * 120 * k * r[:, 2], (0.4 + 0.6 * r[:, 3]) * clamp(1.8 - q), CHOL, 1.8, 0.15)
                if q < 0.14: f = f + (1 - q / 0.14) * 0.7
                sh += 12 * math.exp(-q * 3) + 2 + 6 * e ** 2
                if e > 0.82: f = f + ((e - 0.82) / 0.18) ** 2 * 1.6
        if u < 0.12: f = f + (1 - u / 0.12) * 0.25
        return f, sh

    # ---- この子だけの曲: オルゴールの ような 鈴 → ごごご(せり上がる)→ どん・どん(木を たたく 音)→ ぱきぱき → 鈴が かけ上がる ----
    def audio(s, path):
        a = Snd(DUR, 427); G = 174.6; pn = [0, 2, 4, 7, 9]
        def fq(d, o=0): return G * 2 ** ((pn[d % 5] + 12 * (d // 5 + o)) / 12.0)
        for d in (0, 2): a.note(fq(d, -1), 0.1, 2.6, 0.04, "pad", 0.6)
        for i, d in enumerate((5, 7, 9, 7, 10, 9, 7, 5, 7, 9)): a.note(fq(d, 1), 0.4 + i * 0.22, 0.5, 0.06, "bell")
        a.boom(T_CUT, 60, 0.6, 0.35); a.whoosh(T_CUT - 0.3, T_CUT, 0.2, 0.1)
        hi = a.nz - a.lp(a.nz, 0.6); wn = a.lp(a.nz, 0.2) - a.lp(a.nz, 0.03); lo = a.lp(a.nz, 0.02)
        for i in range(6): tk = T_CUT + 0.5 + i * 0.11; a.a += hi * a.env(tk, 0.001, 0.03) * 0.5; a.note(fq(5 + i, 1), tk, 0.4, 0.07, "bell")
        for d in (0, 2, 4): a.note(fq(d, 0), T_CUT, 1.9, 0.04, "pad", 0.05)
        a.boom(T_RISE, 50, 0.8, 0.6); g = np.clip((a.t - T_RISE) / 0.2, 0, 1) * np.clip((T_RISE + 1.15 - a.t) / 0.2, 0, 1); a.a += lo * g * 3.5 + wn * g * 0.6; a.rise(T_RISE, T_RISE + 1.05, 50, 120, 0.12)
        a.boom(T_RISE + 1.05, 44, 0.9, 0.6)
        for th in (T_H1, T_H2):
            a.boom(th, 42, 1.0, 1.0); a.boom(th, 120, 0.15, 0.5); a.a += hi * a.env(th + 0.03, 0.001, 0.05) * 0.8
            rg = np.random.RandomState(int(th * 10))
            for i in range(7): tk = th + 0.06 + rg.rand() * 0.35; a.a += hi * a.env(tk, 0.0005, 0.015) * 0.6
        a.boom(T_BRK, 38, 1.5, 1.0); a.a += hi * a.env(T_BRK, 0.001, 0.2) * 1.0 + wn * a.env(T_BRK, 0.003, 0.5) * 1.2
        rg = np.random.RandomState(427)
        for i in range(15): tk = T_BRK + 0.05 + i * 0.07 + rg.rand() * 0.04; a.a += hi * a.env(tk, 0.0005, 0.03) * 0.7; a.boom(tk, 150 + rg.rand() * 80, 0.06, 0.25)
        st = (T_HIT - T_BRK - 0.3) / 14
        for i in range(14): a.note(fq(5 + i, 1), T_BRK + 0.2 + i * st, 0.4, 0.07, "bell")
        for j, d in enumerate((0, 2, 4, 7)): a.note(fq(d, 0), T_BRK + j * 0.08, T_HIT - T_BRK + 0.2, 0.05, "pad", 0.5)
        a.rise(T_BRK, T_HIT, 110, 420, 0.10)
        a.boom(T_HIT, 40, 1.5, 0.8)
        for j, d in enumerate((0, 2, 4, 7)): a.note(fq(d, 2), T_HIT + 0.02 + j * 0.03, 0.9, 0.05, "bell")
        s.fin_snd(a, DUR, 427); a.save(path)


if __name__ == "__main__":
    r = Chocola()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
