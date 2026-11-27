"""ジェミナ(★4 双子座の双剣士) 専用の必殺技「双星乱舞」— 高品質版(hs_fx)。青と 金の ふたつ星が からみ合う らせんで こちらへ 来て、目の前で ひとつに なる
夜の 丘。頭上に ふたつ星(青・金)が 灯る(前ふり) → 目の カットイン(青と 金の 半分ずつ)+技名 →
舞う: 二本の 剣の 光の 帯が 体の まわりを まわり、ふたつ星が 剣に おりる → 突き出す:
青い 星と 金の 星が 二重らせんを えがいて こちらへ(尾を ひき、だんだん 大きく)→ 目の前で ぶつかって ひとつに → 白 → 暗転
使い方: python3 hs_403.py 出力.mp4 / python3 hs_403.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.0
T_CUT, T_DANCE, T_FLY, T_MERGE, T_HIT = 2.7, 4.7, 6.9, 8.75, 9.3
BLUE = np.array([0.35, 0.62, 1.0], np.float32); GOLD = np.array([1.0, 0.78, 0.30], np.float32); PALE = np.array([0.92, 0.95, 1.0], np.float32); NIGHT = np.array([0.03, 0.04, 0.13], np.float32)
EYE = (432, 245); TIPB = (15, 590); TIPG = (895, 560); HILB = (175, 440); HILG = (690, 410)
SKB = (0.36, 0.20); SKG = (0.64, 0.17)                                                 # 空の ふたつ星の 位置(画面の 比)


class Gemina(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 403, (0.6, 0.8, 1.0), "双星乱舞", "Twin Star Waltz", W); s.fx_init(403)
        s._vt = [s.text_img(c, hz.SERIF, int(76 * s.k), 3) for c in "双星乱舞"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; s.hy = 0.74

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 夜の 丘 ----
    def hill(s, t, lum=1.0, tw=0.0):
        W, H, k = s.W, s.H, s.k; hy = s.hy
        f = NIGHT[None, None, :] * (0.5 + 1.5 * (s.yn / hy).clip(0, 1) ** 1.5)[:, :, None] + np.array([0.08, 0.13, 0.26], np.float32)[None, None, :] * np.exp(-((s.yn - hy) / 0.14) ** 2)[:, :, None]
        f = f + s.nebula(t, [(0.05, 0.08, 0.26), (0.20, 0.14, 0.20), (0.45, 0.50, 0.75)], 0.0008, (3.0, 0.6), 0.55, 0)
        s.starfield(f, t, 1100, (0.9, 0.93, 1.0), 2, 0.9, (0.0, 0.0))
        if tw > 0:                                                                   # 画面の 左右に うすく 青と 金
            f += (np.exp(-((s.xn - 0.0) / 0.5) ** 2) * 0.10 * tw)[:, :, None] * BLUE[None, None, :] + (np.exp(-((s.xn - 1.0) / 0.5) ** 2) * 0.10 * tw)[:, :, None] * GOLD[None, None, :]
        rd = hy + 0.035 * np.cos((s.xn - 0.5) * 3.0) - 0.03 + 0.006 * np.sin(s.xn * 21); m = sstep(rd - 0.003, rd + 0.003, s.yn); d = np.clip((s.yn - rd) / (1 - hy), 0, 1)
        gr = s.tex(1, s.XX * 1.6 + np.sin(t * 2 + s.YY * 0.03) * 3 * k, s.YY * 0.5) * 0.6 + s.tex(3, s.XX * 0.7, s.YY * 0.9) * 0.4
        col = np.array([0.015, 0.035, 0.05], np.float32)[None, None, :] * (0.5 + 1.5 * gr)[:, :, None] * (1.3 - 0.7 * d)[:, :, None] + (PALE * 0.10)[None, None, :] * (np.exp(-d / 0.04) * (0.4 + gr))[:, :, None]
        return ((f * (1 - m)[:, :, None] + col * m[:, :, None]) * lum).astype(np.float32)

    # ---- 星: しん + ゆらぐ 光の 輪 + 十字の 光 ----
    def star(s, f, c, R, col, t, v=1.0):
        if v <= 0.01 or R < 0.8: return
        col = np.asarray(col, np.float32)
        if R < 14 * s.k: s.flare(f, c, R * 1.6, col * 0.7 + 0.3, v, 0.6, t * 0.7, False); return
        d = np.hypot(s.XX - c[0], s.YY - c[1]) / R + 1e-4; th = np.arctan2(s.YY - c[1], s.XX - c[0]); n = s.tex(1, th / 6.2832 * 512 + t * 9, d * 16 - t * 40)
        f += (np.exp(-(d / 0.30) ** 2) * 1.6 * v)[:, :, None] * (col * 0.6 + 0.4)[None, None, :] + (np.exp(-d / 0.9) * (0.45 + 1.0 * n ** 2) * 1.1 * v)[:, :, None] * col[None, None, :]
        s.flare(f, c, R * 0.9, col * 0.6 + 0.4, 0.7 * v, 0.8, 0.5, False)

    def ribbon(s, f, pts, col, w, v=1.0):
        """光の 帯(先ほど 太く 明るい)。pts: 古い → 新しい"""
        n = len(pts)
        if n < 2 or v <= 0.01: return
        L = s.L0()
        for i in range(n - 1):
            q = (i + 1) / (n - 1); cv2.line(L, (int(pts[i][0]), int(pts[i][1])), (int(pts[i + 1][0]), int(pts[i + 1][1])), float(q ** 1.5), max(1, int(w * (0.3 + 0.7 * q))), cv2.LINE_AA)
        L = cv2.GaussianBlur(L, (0, 0), 0.8 * s.k) * v; col = np.asarray(col, np.float32)
        f += (L * 1.2 + cv2.GaussianBlur(L, (0, 0), 6 * s.k) * 0.9)[:, :, None] * col[None, None, :] + (L ** 2 * 0.7)[:, :, None]

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        g = getattr(s, '_gl', 0.0)
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(PALE), rim_dir=(0.0, -1.0), rim_v=(0.4 + 0.3 * g) * alpha, amb=(0.7, 0.8, 1.0), amb_v=0.16, aura=BLUE * 0.5 + GOLD * 0.5, aura_v=0.14 * g, alpha=alpha, shade=0.80 + 0.12 * g)

    def frame(s, t):
        o = {}; s._gl = 0.0
        if t < T_CUT: f, sh = s.sc_night(t)
        elif t < T_DANCE: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.92)
        else: f, sh = s.sc_waltz(t)
        f, sh = s.finale(f, sh, t, DUR, None, (0.6, 0.8, 1), (1, 0.8, 0.3), -1)
        return s.film(f, t, sh * s.k, 0.55, o.get("thr", 0.84), (0.95, 0.95, 1.0), 1.0 + 0.15 * min(sh, 8), 0.04, o.get("vig", 0.58), 1.15, 1.10)

    HC = 0.66

    # 1) 夜の 丘。頭上に ふたつ星が 灯る
    def sc_night(s, t):
        W, H, k = s.W, s.H, s.k
        f = s.hill(t, 1.0, 0.5 * clamp((t - 1.0) / 1.2))
        for (p, col, t0) in ((SKB, BLUE, 0.8), (SKG, GOLD, 1.35)):
            q = t - t0
            if q > 0: s.star(f, (W * p[0], H * p[1]), (9 + 14 * math.exp(-q * 5) + 1.5 * math.sin(t * 6)) * k, col, t, clamp(q * 5))
        if t > 1.9: s.glowline(f, [(W * SKB[0], H * SKB[1]), (W * (SKB[0] + (SKG[0] - SKB[0]) * ease((t - 1.9) / 0.5)), H * (SKB[1] + (SKG[1] - SKB[1]) * ease((t - 1.9) / 0.5)))], PALE, 0.8, 0.4, 0.6)
        s._gl = 0.3; img = s.pose(t, sway=0.3, breath=0.5); s.draw(f, img, W / 2, H * s.HC, 0.40)
        return f * clamp(t / 0.5), 0

    # 2) 目の カットイン(青と 金の 半分ずつの 帯)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT; s._gl = 0.6
        f = s.hill(t, 0.32, 0.5)
        a = ease(u / 0.26); cy = H * 0.33; hh = H * 0.105
        hb = sstep(0.48, 0.52, s.xn)[:, :, None]; st = (0.35 + 0.8 * s.tex(1, s.XX * 0.1 - t * 180 * (1 - 2 * hb[:, :, 0]), s.YY * 1.6))[:, :, None]
        band = ((BLUE * 0.6)[None, None, :] * (1 - hb) + (GOLD * 0.6)[None, None, :] * hb) * st + NIGHT[None, None, :]
        ex, ey = s.A(*EYE); zf = 1.45; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(band, img, W / 2 - (ex - s.aw / 2) * sc, cy - (ey - s.ah / 2) * sc + (1 - a) * 0, zf)
        edge = hh * a; off = np.where(s.xn < 0.5, -1.0, 1.0) * (1 - a) * H * 0.3; dd = np.abs(s.YY - cy - off)                                # 左右が 上下から かみ合う
        m = (sstep(edge + 1, edge - 1, dd) * (a > 0.02))[:, :, None]; f = f * (1 - m) + band * m
        s.glowmask(f, np.exp(-((dd - edge) / (1.6 * k)) ** 2) * (a > 0.02), PALE, 0.9 * a, 1.0, 0.8, 8.0)
        n = len(s._vt); step = W * 0.20
        for i, T in enumerate(s._vt):                                                # 技名: 外の 2文字と 内の 2文字が 対で 入る
            p = (u - 0.5 - (0 if i in (0, 3) else 0.18)) / 0.2
            if p > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step * (1 + (1 - ease(p)) * 0.6), H * 0.68, clamp(p * 1.5), 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.35 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.68 - 76 * k, 1.0, 1.0)
        if u > 1.3:
            g = ease((u - 1.3) / 0.4); s.glowline(f, [(W / 2 - W * 0.42 * g, H * 0.68 + 58 * k), (W / 2, H * 0.68 + 58 * k)], BLUE * 0.7 + 0.3, 1.0, 0.8, 0.8); s.glowline(f, [(W / 2, H * 0.68 + 58 * k), (W / 2 + W * 0.42 * g, H * 0.68 + 58 * k)], GOLD, 1.0, 0.8, 0.8)
            s.text(f, s.txt_en, W / 2, H * 0.68 + 84 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.7
        return f, 4 * math.exp(-u * 6)

    def helix(s, p, sg, st):
        """らせんの 上の 位置(p: 0 奥 → 1 目の前)。もどり値 (x, y, 大きさ)"""
        W, H, k = s.W, s.H, s.k; g = ease2(p); cx = st[0] + (W * 0.5 - st[0]) * g; cy = st[1] + (H * 0.44 - st[1]) * g
        r = (W * 0.075 / (1.10 - p) + 26 * k) * sstep(1.0, 0.90, p); a = 2.2 + 10.0 * p ** 1.2 + (0 if sg > 0 else math.pi)
        return cx + math.cos(a) * r, cy + math.sin(a) * r * 0.85, k * (9 + 100 * p ** 2.2)

    # 3) 舞う → ふたつ星が らせんで こちらへ → 目の前で ひとつに
    def sc_waltz(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_DANCE; uf = t - T_FLY; um = t - T_MERGE; sh = 0.0
        s._gl = clamp(u / 0.6); p = clamp(uf / (T_MERGE - T_FLY)) if uf > 0 else 0.0; pe = p ** 1.4
        f = s.hill(t, 1.0 - 0.35 * pe, 0.5 + 0.5 * s._gl)
        hc = (W / 2, H * s.HC)
        if uf < 0:                                                                   # 空の 星が 剣へ おりる
            q = ease2(clamp((u - 0.9) / 1.0))
        else: q = 1.0
        spin = u * 2.2 + u * u * 0.9; rb = []
        if uf < 0.5:                                                                 # 体の まわりを まわる 2本の 光の 帯
            vr = 1 - clamp(uf / 0.5) if uf > 0 else clamp(u / 0.4)
            for sg, col in ((1, BLUE), (-1, GOLD)):
                pts = [(hc[0] + math.cos(spin - j * 0.09 + (0 if sg > 0 else math.pi)) * W * 0.36, hc[1] - H * 0.02 + math.sin(spin - j * 0.09 + (0 if sg > 0 else math.pi)) * H * 0.065 + sg * H * 0.03 * math.sin((spin - j * 0.09) * 0.5)) for j in range(22, -1, -1)]
                rb.append((math.sin(spin + (0 if sg > 0 else math.pi)), pts, col, vr))
            for dp, pts, col, vr in rb:
                if dp < 0: s.ribbon(f, pts, col * 0.85 + 0.15, 12 * k, vr * 0.8)
        a_ = 1 - clamp((p - 0.72) / 0.2)
        if a_ > 0.02:
            if uf < 0: img = s.pose(t, sway=0.7, breath=0.5, lean=10 * k * math.sin(spin), squash=-0.02 * abs(math.cos(spin)))
            else: img = s.pose(t, sway=0.3, lean=0.0, squash=-0.06 * math.exp(-uf * 5) + 0.02)
            s.draw(f, img, hc[0] + (8 * k * math.sin(spin) if uf < 0 else 0), hc[1], 0.40, a_)
            tb = s.at(*s.A(*HILB)); tg = s.at(*s.A(*HILG))
        else: tb = (W * 0.36, H * 0.6); tg = (W * 0.62, H * 0.58)
        for dp, pts, col, vr in rb:
            if dp >= 0: s.ribbon(f, pts, col * 0.85 + 0.15, 13 * k, vr)
        if uf < 0:
            for (sp_, tp, col) in ((SKB, tb, BLUE), (SKG, tg, GOLD)):
                c = (W * sp_[0] + (tp[0] - W * sp_[0]) * q, H * sp_[1] + (tp[1] - H * sp_[1]) * q); s.star(f, c, (9 + 3 * math.sin(t * 7)) * k, col, t, 1.0)
                if 0 < q < 1: s.ribbon(f, [(W * sp_[0] + (tp[0] - W * sp_[0]) * max(0, q - j * 0.04), H * sp_[1] + (tp[1] - H * sp_[1]) * max(0, q - j * 0.04)) for j in range(8, -1, -1)], col, 3 * k, 0.7)
        elif um < 0:                                                                 # 二重らせん
            for sg, col, st in ((1, BLUE, tb), (-1, GOLD, tg)):
                pts = [s.helix(max(0.0, p - j * 0.014), sg, st) for j in range(44, -1, -1)]; x, y, R = pts[-1]
                s.ribbon(f, [(a, b) for a, b, c in pts], col * 0.9 + 0.1, max(3 * k, R * 0.55), 1.0); s.star(f, (x, y), R, col, t, 1.0)
            s.rushdust(f, t, (W / 2, H * 0.44), 800, 0.6 + 2.0 * p, PALE, 0.15 + 0.5 * pe, 3, 1.0); sh += 1 + 7 * pe
            if uf < 0.1: f = f + (1 - uf / 0.1) * 0.6
        if um > 0:                                                                   # ひとつに なる
            C = (W / 2, H * 0.44); e = clamp(um / (T_HIT - T_MERGE)); d = np.hypot(s.XX - C[0], s.YY - C[1])
            s.star(f, C, k * 110 * (1 + 3.5 * e), PALE, t, 1.0)
            for i, col in enumerate((BLUE, GOLD, PALE)):
                qq = (um - i * 0.06) / 0.55
                if 0 < qq < 1.3: f += (np.exp(-((d - qq * W * 1.3) / ((8 + 40 * qq) * k)) ** 2) * max(0.0, 1.3 - qq) * 1.2)[:, :, None] * (col * 0.7 + 0.3)[None, None, :]
            s.rushdust(f, t, C, 1200, 3.0, PALE, 0.8, 3, 1.0); f = s.godrays(f, C, 0.6, 0.95, 8, 0.88, 0.5)
            if um < 0.1: f = f + (1 - um / 0.1) * 1.2
            sh += 16 * math.exp(-um * 4) + 4; f = f + e ** 2 * 1.5
        if u < 0.12: f = f + (1 - u / 0.12) * 0.35
        return f, sh

    # ---- この子だけの曲: 2つの 声(高い 鈴=青、あたたかい 弦=金)が 3度で 重なって おどる。まわるほど はやく、さいごは 同じ 音に とけ合う ----
    def audio(s, path):
        a = Snd(DUR, 403); E = 164.81; mj = [0, 2, 4, 5, 7, 9, 11]
        def fq(d, o=0): return E * 2 ** ((mj[d % 7] + 12 * (d // 7 + o)) / 12.0)
        for d in (0, 4): a.note(fq(d, -1), 0.1, 2.6, 0.045, "pad", 0.6)
        a.note(fq(9, 1), 0.8, 1.2, 0.14, "bell"); a.note(fq(7, 0), 1.35, 1.2, 0.10, "fm"); a.note(fq(7, 0) * 1.003, 1.35, 1.0, 0.06, "saw")           # 青い 星・金の 星
        a.note(fq(11, 1), 1.9, 0.6, 0.07, "bell"); a.note(fq(9, 0), 1.9, 0.6, 0.06, "fm")
        a.boom(T_CUT, 55, 0.9, 0.4); a.whoosh(T_CUT - 0.3, T_CUT, 0.3)
        for d in (0, 2, 4, 6): a.note(fq(d, 0), T_CUT, 1.9, 0.04, "pad", 0.05)
        for i, d in enumerate((7, 9, 9, 11)): a.note(fq(d, 1), T_CUT + 0.5 + (0 if i in (0, 3) else 0.18), 0.6, 0.10, "bell"); a.note(fq(d - 2, 0), T_CUT + 0.5 + (0 if i in (0, 3) else 0.18), 0.5, 0.07, "fm")
        mel = [7, 9, 11, 9, 7, 9, 11, 14, 11, 9, 11, 14, 16, 14, 11, 14, 16, 18, 16, 14, 16, 18, 21, 18]
        tk = T_DANCE + 0.1; st = 0.26; i = 0
        while tk < T_MERGE - 0.05:                                                                                    # 3度で おどる 2つの 声(だんだん はやく)
            d = mel[i % len(mel)]; g = 0.7 + 0.5 * (tk - T_DANCE) / (T_MERGE - T_DANCE)
            a.note(fq(d, 1), tk, 0.3, 0.085 * g, "bell"); a.note(fq(d - 2, 0), tk, 0.3, 0.07 * g, "fm"); a.note(fq(d - 2, 0) * 1.003, tk + 0.01, 0.25, 0.04 * g, "saw")
            if i % 3 == 0: a.note(fq([0, 3, 4, 0][(i // 3) % 4], -1), tk, 0.5, 0.11, "saw", 0.01); a.boom(tk, 78, 0.2, 0.08)
            if tk > T_FLY: a.hat(tk + st / 2, 0.04)
            tk += st; st = max(0.085, st * 0.965); i += 1
        a.whoosh(T_FLY, T_FLY + 0.3, 0.5, 0.5); a.rise(T_FLY, T_MERGE, 140, 620, 0.10); a.rise(T_FLY, T_MERGE, 210, 624, 0.07)            # 2本の 上がる 音が 同じ 高さへ
        a.boom(T_MERGE, 36, 1.9, 0.9); a.a += a.lp(a.nz, 0.15) * a.env(T_MERGE, 0.02, 0.5) * 0.6
        for j, d in enumerate((0, 4, 7, 9, 11, 14)): a.note(fq(d, 0), T_MERGE + j * 0.02, 0.9, 0.06, "bell")
        a.note(fq(14, 1), T_MERGE, 1.0, 0.12, "sine")
        a.boom(T_HIT, 40, 1.2, 0.7)
        s.fin_snd(a, DUR, 403); a.save(path)


if __name__ == "__main__":
    r = Gemina()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
