"""ミライ(★6 まだ見ぬ者) 専用の必殺技「未ダ見ヌ明日」— 高品質版・通し直し。紙と えんぴつの 世界(この子だけ)。えらんだ 扉が こちらへ 来て、ひらく
まっ白な 紙(せんい・しみ)。えんぴつの 線(かすれ)が 地平を 引き、この子の りんかくを 描き、色が のる → 黒い 紙の カットイン(フードの 闇に 星 ひとつ。技名は 1字ずつ 打たれる) →
足もとから 点線の 道が 何本も 枝分かれして、それぞれの 先に 小さな 扉(ありえた 明日。前ふり) → こぶしを にぎると、えらばなかった 道が 消しゴムで 消える →
のこった 扉が 紙の 上を こちらへ 来る。木目が かきこまれ、目の前に 立つ → ひらく。むこうは はじめての 色: 水彩の 朝。ミライは もう むこうに いる → 朝の 光に つつまれて 白 → 暗転
使い方: python3 hs_80.py 出力.mp4 / python3 hs_80.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old80", "/home/claude/tools/pre_fin/hs_80_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = 11.2
PAPER, INKC, GOLD, DOORS, PICK = old.PAPER, old.INKC, old.GOLD, old.DOORS, old.PICK
T_CUT, T_PATH, T_PICK, T_DOOR = old.T_CUT, old.T_PATH, old.T_PICK, old.T_DOOR
T_OPEN, T_HIT = 9.5, 10.5


class Mirai(FX, old.Mirai):
    def __init__(s, W=540):
        old.Mirai.__init__(s, W); s.fx_init(80); H = s.H; k = s.k; s.yv = (s.YY / H).astype(np.float32)
        fib = cv2.remap(s._tiles[0], ((s.XX * 0.5 / k) % 256).astype(np.float32), ((s.YY * 4.0 / k) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP) * 0.5 + cv2.remap(s._tiles[2], ((s.XX * 3.5 / k) % 256).astype(np.float32), ((s.YY * 3.5 / k) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP) * 0.5
        stain = s.fbm(0.0010, 0, 0, 3, 40 * k, 1); s._pp = (0.93 + 0.09 * fib) * (0.95 + 0.07 * stain) * (1 - 0.10 * np.clip(s._rad, 0, 1.3) ** 2.5)                                          # 紙: せんい・しみ・四すみの かげ
        s._pg = cv2.remap(s._tiles[0], ((s.XX * 4.5 / k) % 256).astype(np.float32), ((s.YY * 4.5 / k) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)                # えんぴつの かすれ
        s._wd = cv2.remap(s._tiles[1], ((s.XX * 2.6 / k) % 256).astype(np.float32), ((s.YY * 0.22 / k) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)               # 木目
        s._wc = s.fbm(0.0022, 7, 3, 3, 60 * k, 2)                                                                                                                                              # 水彩の にじみ

    # ---- 紙と えんぴつ ----
    def page(s, dark=False):
        if dark: return (np.array([0.075, 0.072, 0.09], np.float32)[None, None, :] * (0.4 + 0.9 * s._pp * s._pg)[:, :, None]).astype(np.float32)
        return (PAPER[None, None, :] * s._pp[:, :, None]).astype(np.float32)

    def ink(s, f, L, a=0.9, col=INKC):
        Lc = np.clip(L, 0, 1); m = (np.clip(Lc * 0.75 + cv2.GaussianBlur(Lc, (0, 0), 0.9 * s.k) * 0.5, 0, 1) * a * (0.50 + 0.75 * s._pg))[:, :, None]; m = np.clip(m, 0, 1)
        f[:] = f * (1 - m) + m * np.asarray(col, np.float32)[None, None, :]

    def frame(s, t):
        if t < T_CUT: f, sh = s.sc_sketch(t); f = f * clamp(t / 0.25)
        elif t < T_PATH: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_DOOR: f, sh = s.sc_paths(t)
        else: f, sh = s.sc_come(t)
        f, sh = s.finale(f, sh, t, DUR, None, (1, 0.85, 0.4), (1, 1, 1), -1)
        f = np.clip(f, 0, 1.2)
        if sh > 0.3: M = np.float32([[1, 0, math.sin(t * 113) * sh * s.k], [0, 1, math.cos(t * 97) * sh * s.k]]); f = cv2.warpAffine(f, M, (s.W, s.H), borderMode=cv2.BORDER_REFLECT)
        gr = cv2.resize(s._grain[int(t * 30) % 8], (s.W, s.H), interpolation=cv2.INTER_LINEAR); f = f + (gr - 0.5)[:, :, None] * 0.03
        return np.clip(f, 0, 1)

    def morning(s, u, t):
        """扉の むこう: 水彩の 朝(この 映像で はじめての 色)"""
        W, H, k = s.W, s.H, s.k; f = s.page(); hz_ = H * 0.66; col = 1.0; wc = s._wc
        sky = sstep(hz_ + 2, hz_ - 2, s.YY)
        wash = np.exp(-((s.yv - 0.52 + (wc - 0.5) * 0.10) / 0.16) ** 2)[:, :, None] * np.array([1.0, 0.60, 0.40], np.float32)[None, None, :] + np.exp(-((s.yv - 0.20 + (wc - 0.5) * 0.14) / 0.22) ** 2)[:, :, None] * np.array([0.46, 0.70, 1.0], np.float32)[None, None, :]
        a = (0.62 * (0.75 + 0.5 * wc) * sky)[:, :, None]; f = f * (1 - a) + wash * a * (0.9 + 0.2 * s._pp[:, :, None])
        gr = (sstep(hz_ - 2, hz_ + 2, s.YY) * 0.55 * (0.7 + 0.6 * wc))[:, :, None]; f = f * (1 - gr) + np.array([0.55, 0.78, 0.45], np.float32)[None, None, :] * (0.8 + 0.3 * s._pg)[:, :, None] * gr
        sun = (W * 0.60, hz_); R = W * 0.24; d = np.hypot(s.XX - sun[0], s.YY - sun[1])
        M = sstep(R + 1.5, R - 1.5, d + (wc - 0.5) * 10 * k) * sky; f = f * (1 - M[:, :, None] * 0.9) + (GOLD[None, None, :] * (0.9 + 0.25 * wc)[:, :, None]) * M[:, :, None] * 0.9
        f += (np.exp(-d / (R * 1.6)) * 0.22 * sky)[:, :, None] * np.array([1.0, 0.85, 0.5], np.float32)[None, None, :]
        L = s.L0(); s.line(L, (W * 0.02, hz_), (W * 0.98, hz_ + 2 * k), 0.9, 2); cv2.ellipse(L, (int(sun[0]), int(sun[1])), (int(R), int(R)), 0, 180, 360, 0.9, 2, cv2.LINE_AA)
        for i in range(11):
            an = -3.1416 * (i + 0.5) / 11; g = ease(clamp((u - 0.1 - i * 0.03) / 0.3)); s.line(L, (sun[0] + math.cos(an) * R * 1.15, sun[1] + math.sin(an) * R * 1.15), (sun[0] + math.cos(an) * R * (1.15 + 0.5 * g), sun[1] + math.sin(an) * R * (1.15 + 0.5 * g)), 0.8, 2)
        for i in range(5):                                                           # 鳥
            bx = W * (0.15 + 0.17 * i) + u * 22 * k; by = H * (0.18 + 0.05 * (i % 3)); wv = math.sin(t * 7 + i) * 4 * k
            s.line(L, (bx - 9 * k, by - wv), (bx, by + 3 * k), 0.8, 2); s.line(L, (bx, by + 3 * k), (bx + 9 * k, by - wv), 0.8, 2)
        s.ink(f, L)
        s.hero(f, t, W * 0.30, hz_ + H * 0.13, 0.30, 1.0, squash=0.03 * math.exp(-u * 6) * math.cos(u * 14)); s.glint(f, s.at(*s.star), 11 * k, t)
        return f

    def pdoor(s, f, C, dw, o, inside, detail):
        """えんぴつで 描いた 扉(アーチ)。o: ひらき 0..1。detail: 木目の かきこみ 0..1"""
        W, H, k = s.W, s.H, s.k; dh = dw * 1.55; x0 = C[0] - dw / 2; x1 = C[0] + dw / 2; y1 = C[1] + dh / 2; ya = C[1] - dh / 2 + dw / 2
        da = np.where(s.YY < ya, np.hypot(s.XX - C[0], s.YY - ya) - dw / 2, np.maximum(np.abs(s.XX - C[0]) - dw / 2, s.YY - y1)); A = sstep(1.2, -1.2, da)                 # アーチの 形(外まで の きょり)
        xo = x0 + dw * (1 - o) * 1.0; op = A * sstep(xo - 1.5, xo + 1.5, s.XX) if o > 0.002 else np.zeros_like(A); leaf = A * (1 - op) if o > 0.002 else A
        if o > 0.002:                                                                # むこう側 と、あふれる 光
            ins = inside * (1 - 0.35 * np.exp(-np.clip(-da, 0, None) / (10 * k + dw * 0.02)) * (1 - o))[:, :, None]; f[:] = f * (1 - op[:, :, None]) + ins * op[:, :, None]
            sp = cv2.GaussianBlur(op, (0, 0), 30 * k + dw * 0.05) * (1 - A) * o; f[:] = f * (1 - 0.5 * sp[:, :, None]) + (GOLD * 0.7 + 0.35)[None, None, :] * 0.5 * sp[:, :, None]
        tone = (0.90 - 0.10 * detail * (0.4 + 0.6 * s._wd) - 0.05 * detail * np.clip((s.XX - x0) / max(dw, 1), 0, 1))                                                        # とびらの 板: 紙より 少し くらく、木目の ハッチ
        f[:] = f * (1 - leaf[:, :, None]) + (PAPER * np.array([1.0, 0.97, 0.90], np.float32))[None, None, :] * (tone * s._pp)[:, :, None] * leaf[:, :, None]
        L = s.L0(); th = max(2, int(2 + dw * 0.006))
        L += np.exp(-(da / (th * 0.6)) ** 2) * 0.95 + np.exp(-((da + th * 3.2) / (th * 0.45)) ** 2) * 0.55 * detail                                                           # わくの 二重線
        if detail > 0.05:
            lw = dw * (1 - o) if o > 0.002 else dw
            hat = sstep(0.60, 0.78, s._wd) * 0.55 + sstep(0.47, 0.5, np.abs(((s.XX - x0) / max(lw, 1) * 4) % 1.0 - 0.5)) * 0.7                                               # 木目 と 板の 合わせ目
            for yy in (C[1] - dh * 0.12, C[1] + dh * 0.30): hat = hat + np.exp(-((s.YY - yy) / (th * 0.7)) ** 2) * 0.7 + np.exp(-((s.YY - yy - dh * 0.035) / (th * 0.7)) ** 2) * 0.7    # 横木
            L += hat * leaf * detail * sstep(-th * 4.0, -th * 6.0, da)
            kc = (x0 + lw * 0.84, C[1] + dh * 0.10); kd = np.hypot(s.XX - kc[0], s.YY - kc[1]); kr = dw * 0.035; L += (np.exp(-((kd - kr) / (th * 0.6)) ** 2) + sstep(kr * 0.5, kr * 0.2, kd) * 0.8) * leaf * detail   # とって
        if 0.002 < o < 0.999: L += np.exp(-((s.XX - xo) / (th * 0.6)) ** 2) * A * 0.9
        s.ink(f, L)

    # 4) えらんだ 扉が こちらへ 来て、ひらく
    def sc_come(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_DOOR; op = t - T_OPEN; g = ease2(clamp(u / (T_OPEN - T_DOOR - 0.05))); o = ease2(clamp(op / 0.75)) if op > 0 else 0.0; e = clamp(op / (T_HIT - T_OPEN)) if op > 0 else 0.0
        f = s.page(); L = s.L0(); Lg = s.L0(); s.line(L, (W * 0.04, s.gnd), (W * 0.96, s.gnd + 2 * k), 0.9, 2)
        pts = s.paths[PICK]
        for j in range(0, len(pts) - 1, 2): s.line(Lg, pts[j], pts[j + 1], 0.75 * (1 - g), 2)
        s.ink(f, L); s.ink(f, Lg + cv2.GaussianBlur(Lg, (0, 0), 3 * k), 1.0, GOLD * 0.9)
        s.hero(f, t, W * 0.20, s.gnd, 0.27, 1.0, squash=0.03 * math.exp(-u * 8)); s.glint(f, s.at(*s.fist), 10 * k * clamp(1 - u / 0.5), t)
        d = DOORS[PICK]; C = (d[0] * W + (W * 0.5 - d[0] * W) * g, d[1] * H + (H * 0.50 - d[1] * H) * g + H * 0.25 * e ** 2); dw = 40 * k + (W * 1.05 - 40 * k) * g ** 1.6 + W * 1.6 * e ** 2
        ins = s.morning(max(0.0, op), t) if op > -0.05 else None
        if g > 0.02 and g < 1: f *= (1 - 0.20 * g * np.exp(-((s.XX - C[0]) / (dw * 0.7 + 1)) ** 2 - ((s.YY - (C[1] + dw * 0.78)) / (dw * 0.10 + 20 * k)) ** 2))[:, :, None]      # 扉の 下の うすい かげ
        s.pdoor(f, C, dw, o, ins, clamp((g - 0.25) / 0.6))
        if op < 0: s.glint(f, (C[0], C[1] - dw * 0.78 - 14 * k), (12 + 10 * g) * k * (1 - g), t)
        sh = 3 * math.exp(-abs(u - (T_OPEN - T_DOOR)) * 9) + (4 * e ** 2 if op > 0 else 0)
        if e > 0.55: f = f + ((e - 0.55) / 0.45) ** 2 * 0.5 * (GOLD * 0.3 + 0.7)[None, None, :]
        if u < 0.12: f = f * (u / 0.12) + (1 - u / 0.12)
        return f, sh

    # ---- この子だけの曲: えんぴつの 音と オルゴール(ハ長調)。カットインは タイプライター。道が のびる間は 1本ごとに 別の ふしを ためす。えらぶと 1本の ふしに。扉が 来る → ひらくと 朝の ふし ----
    def audio(s, path):
        a = Snd(DUR, 80); C5 = 523.25; maj = [0, 2, 4, 5, 7, 9, 11]
        def fq(d, o=0): return C5 * 2 ** ((maj[d % 7] + 12 * (d // 7 + o)) / 12.0)
        def box(d, t0, g=0.09, o=0): a.note(fq(d, o), t0, 0.5, g, "bell", 0.001)
        def scratch(t0, dur, g=0.25): a.a += (a.nz - a.lp(a.nz, 0.35)) * a.gate(t0, t0 + dur, 0.03) * g * (0.6 + 0.4 * np.sin(a.t * 60))
        def click(t0, g=0.3): a.a += (a.nz - a.lp(a.nz, 0.7)) * a.env(t0, 0.0005, 0.015) * g; a.boom(t0, 180, g * 0.4, 0.03)
        scratch(0.0, 0.6); scratch(0.6, 0.95, 0.3)
        for i, d in enumerate([0, 2, 4, 7]): box(d, 1.5 + i * 0.17, 0.07)
        box(11, 2.2, 0.11, 0)
        a.boom(T_CUT, 60, 0.8, 0.2); box(7, T_CUT + 0.45, 0.10, 1)
        for i in range(6): click(T_CUT + 0.6 + i * 0.14)
        a.note(2100, T_CUT + 0.6 + 6 * 0.14 + 0.05, 0.4, 0.08, "bell")                                                 # チーン
        ph = [[0, 2, 4], [0, 3, 5], [1, 4, 6], [0, 4, 7], [2, 5, 8], [3, 6, 9], [4, 7, 11]]
        for i in range(7):
            tk = T_PATH + 0.2 + i * 0.22; scratch(tk, 0.5, 0.10)
            for j, d in enumerate(ph[i]): box(d, tk + j * 0.2, 0.05 + 0.005 * i)
            box(ph[i][2] + 3, tk + 0.7, 0.05)
        a.boom(T_PICK, 70, 0.7, 0.2); box(11, T_PICK, 0.12)
        for i in range(6): scratch(T_PICK + 0.15 + i * 0.1, 0.12, 0.2)                                                 # 消しゴム
        for i, d in enumerate([4, 7, 11, 14]): box(d, T_PICK + 0.6 + i * 0.15, 0.09)
        a.rise(T_DOOR, T_OPEN, 60, 110, 0.14); a.whoosh(T_DOOR + 0.1, T_OPEN, 0.5, 0.2)                                 # 扉が 来る
        for i in range(5): scratch(T_DOOR + 0.3 + i * 0.11, 0.09, 0.16)                                                 # 木目を かきこむ
        for d in (0, 4): a.note(fq(d, -2), T_DOOR, T_OPEN - T_DOOR, 0.05, "pad", 0.3)
        a.boom(T_OPEN - 0.03, 60, 0.9, 0.3); click(T_OPEN, 0.4)                                                         # 目の前に 立つ → かちゃ
        a.rise(T_OPEN, T_OPEN + 0.7, 90, 60, 0.16); a.rise(T_OPEN, T_HIT, 300, 1500, 0.07)                              # ひらく
        for j, (d, o) in enumerate(((0, -2), (4, -2), (0, -1), (2, -1), (4, -1))): a.note(fq(d, o), T_OPEN + j * 0.06, T_HIT - T_OPEN + 0.4, 0.05, "pad", 0.3)
        mel = [4, 2, 0, 2, 4, 7, 9, 7, 11, 14]
        for i, d in enumerate(mel): box(d, T_OPEN + 0.12 + i * 0.09, 0.10); box(d, T_OPEN + 0.12 + i * 0.09, 0.04, -1)   # 朝の ふし
        a.boom(T_HIT, 50, 0.8, 0.5)
        s.fin_snd(a, DUR, 80); a.save(path)


if __name__ == "__main__":
    r = Mirai()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
