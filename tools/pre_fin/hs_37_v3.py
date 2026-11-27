"""カグヤ(★5 月の姫) 専用の必殺技「月光ノ舞」
夜の竹やぶ。まん中の 1本の 節が 金色に 光る → カッ。竹が ななめに 切れて 光の中から 姫 → 扇が きらり、竹が 内から外へ 次々 切れて 空が ひらける(金の すやり霞) →
扇形の窓のカットイン → 五つの宝(石の鉢・玉の枝・火ねずみの衣・竜の珠・子安貝)が ひとつずつ 現れて まわる → 宝が うしろの月へ すいこまれ、五色の 月の輪が 広がる →
切り株の 切り口が 灯り、羽衣が 空を わたる
使い方: python3 hs_37.py 出力.mp4 / python3 hs_37.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 11.5
GOLD = np.array([1.0, 0.84, 0.42], np.float32); KAS = np.array([0.80, 0.63, 0.26], np.float32)
TC = [np.array(c, np.float32) for c in ((0.62, 0.72, 0.95), (0.45, 1.0, 0.55), (1.0, 0.38, 0.18), (0.75, 0.45, 1.0), (1.0, 0.95, 0.85))]
T_GLOW, T_CHOP, T_FAN, T_CUT, T_TR, T_SUCK, T_BURST, T_END = 0.5, 1.4, 2.3, 3.9, 5.8, 7.15, 7.7, 8.6


class Kaguya(Kit):
    def __init__(s, W=540):
        super().__init__(37, (1.0, 0.84, 0.42), "月光ノ舞", "Moonlight Waltz", W)
        H, k = s.H, s.k
        s.eL = s.A(255, 365); s.eR = s.A(395, 350); s.fc = s.A(330, 350); s.fan = s.A(240, 470); s.moon = s.A(370, 235)
        s.ht = [s.text_img(c, hz.SERIF, int(84 * k), 3) for c in "月光ノ舞"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * k), 1) for c in "奥義"]
        s.HF = 0.38; s.CX = W / 2; s.flr = H * 0.84; s.CY = s.flr - s.HF * H / 2
        rg = np.random.RandomState(37); st = []
        for i in range(15):                                                          # 竹: (x, 太さ, 色のこさ, 切れる高さ, かたむき)
            x = W * (0.03 + 0.94 * i / 14.0) + (rg.rand() - 0.5) * 16 * k
            if abs(x - W / 2) < 34 * k: x += 52 * k
            st.append((x, (13 + 15 * rg.rand()) * k, 0.55 + 0.45 * rg.rand(), H * (0.42 + 0.36 * rg.rand()), (rg.rand() - 0.5) * 0.05))
        s.st = st; s.mid = (W / 2, 30 * k, 1.0, H * 0.60, 0.0)
        s.ff = rg.rand(18, 4).astype(np.float32); s.yn = s.YY / H
        s.kb = [(0.15, 0.13, 0.50), (0.78, 0.22, 0.42), (0.30, 0.33, 0.62), (0.85, 0.46, 0.36), (0.10, 0.55, 0.40), (0.55, 0.66, 0.60)]   # 霞: (x, y, 長さ)

    # ---- 大道具 ----
    def sky(s, t, open_=1.0):
        f = (np.array([0.03, 0.04, 0.12], np.float32)[None, None, :] * (1 - s.yn)[:, :, None] + np.array([0.14, 0.11, 0.28], np.float32)[None, None, :] * s.yn[:, :, None]).astype(np.float32)
        if open_ > 0.01:                                                             # 金の すやり霞
            m = s.L0(); W, H, k = s.W, s.H, s.k
            for i, (x, y, ln) in enumerate(s.kb):
                cx = ((x + t * 0.012 * (1 if i % 2 else -1)) % 1.4 - 0.2) * W; hh = 15 * k; hl = ln * W / 2
                cv2.rectangle(m, (int(cx - hl), int(y * H - hh)), (int(cx + hl), int(y * H + hh)), 1.0, -1); cv2.circle(m, (int(cx - hl), int(y * H)), int(hh), 1.0, -1, cv2.LINE_AA); cv2.circle(m, (int(cx + hl), int(y * H)), int(hh), 1.0, -1, cv2.LINE_AA)
                cv2.rectangle(m, (int(cx - hl * 0.5), int(y * H + hh * 0.6)), (int(cx + hl * 1.3), int(y * H + hh * 2.4)), 1.0, -1); cv2.circle(m, (int(cx + hl * 1.3), int(y * H + hh * 1.5)), int(hh * 0.9), 1.0, -1, cv2.LINE_AA)
            m *= 0.5 * open_; f = f * (1 - m[:, :, None]) + KAS[None, None, :] * m[:, :, None]
        g = int(s.flr + s.H * 0.02); f[g:] = f[g:] * 0.25 + np.array([0.01, 0.03, 0.02], np.float32)
        return f

    def stalk(s, f, st, t, tc, lit=0.0):
        """竹 1本。tc: 切れる時刻(None なら 切れない)。切れたら 上が ななめに すべり落ち、切り口が 光る"""
        x, w, sh, hy, tilt = st; H, k = s.H, s.k
        col = (0.03 * sh + 0.05 * lit, 0.11 * sh + 0.10 * lit, 0.06 * sh + 0.04 * lit); edge = (0.01, 0.04, 0.02)
        def seg(y0, y1, dx=0.0, dy=0.0, a=1.0):
            if y1 <= y0: return
            P = np.array([(x - w / 2 + tilt * (H - y0) + dx, y0 + dy + w * 0.35), (x + w / 2 + tilt * (H - y0) + dx, y0 + dy - w * 0.35), (x + w / 2 + tilt * (H - y1) + dx, y1 + dy), (x - w / 2 + tilt * (H - y1) + dx, y1 + dy)], np.int32)
            if a >= 1: cv2.fillPoly(f, [P], col, cv2.LINE_AA)
            else:
                m = s.L0(); cv2.fillPoly(m, [P], float(a), cv2.LINE_AA); f[:] = f * (1 - m[:, :, None]) + np.array(col, np.float32)[None, None, :] * m[:, :, None]
            if a > 0.5:
                yy = y1
                while yy > y0 + 20 * k: cv2.line(f, (int(x - w / 2 + tilt * (H - yy) + dx), int(yy + dy)), (int(x + w / 2 + tilt * (H - yy) + dx), int(yy + dy)), edge, max(1, int(2 * k)), cv2.LINE_AA); yy -= 96 * k
        v = (t - tc) if tc is not None else -1.0
        if v < 0: seg(-20 * k, H); return None
        seg(hy, H)
        if v < 0.6: e = v / 0.6; seg(-20 * k, hy, w * 1.2 * ease(v / 0.15) + 60 * k * e * e, -w * 0.5 * ease(v / 0.15) + H * 0.5 * e * e, 1 - e)
        return (x + tilt * (H - hy), hy)

    def grove(s, f, t, wave=None, glow=0.0, mid=True, lit=0.0):
        """wave: 竹が 外へ 順に 切れはじめる時刻。もどり値: 切り口の 位置"""
        cuts = []; L = s.L0(); W = s.W
        for st in s.st:
            tc = None if wave is None else wave + abs(st[0] - W / 2) / W * 1.7
            c = s.stalk(f, st, t, tc, lit)
            if c: cuts.append((c, t - tc, st[1]))
        if mid:
            c = s.stalk(f, s.mid, t, T_CHOP, lit)
            if glow > 0 and c is None: x, w, _, hy, _ = s.mid; cv2.rectangle(L, (int(x - w / 2), int(hy + 4 * s.k)), (int(x + w / 2), int(hy + 92 * s.k)), float(glow), -1)
            if c: cuts.append((c, t - T_CHOP, s.mid[1]))
        for (c, v, w) in cuts:                                                       # 切り口
            cv2.ellipse(L, (int(c[0]), int(c[1])), (int(w * 0.55), int(w * 0.36)), -35, 0, 360, float(0.55 + 0.45 * math.exp(-v * 6)), -1, cv2.LINE_AA)
            if v < 0.12: s.line(L, (c[0] - w * 2.2, c[1] + w * 1.5), (c[0] + w * 2.2, c[1] - w * 1.5), 1 - v / 0.12, 2)
        s.add(f, L, GOLD, 0.8, 2, 9)
        return cuts

    def fireflies(s, f, t, v=1.0):
        L = s.L0()
        for r in s.ff:
            x = (r[0] + 0.03 * math.sin(t * 0.7 + r[2] * 9)) * s.W; y = (0.25 + 0.6 * r[1] + 0.02 * math.sin(t * 0.9 + r[3] * 7)) * s.H
            cv2.circle(L, (int(x), int(y)), max(1, int(2 * s.k)), float(v * (0.5 + 0.5 * math.sin(t * 2.2 + r[3] * 20))), -1, cv2.LINE_AA)
        f += (L + cv2.GaussianBlur(L, (0, 0), 4 * s.k) * 2)[:, :, None] * np.array([0.7, 1.0, 0.45], np.float32)[None, None, :]

    def hero(s, f, t, alpha=1.0, **kw):
        img = s.pose(t, **kw); s.draw(f, img, s.CX, s.CY, s.HF, alpha)

    def glint(s, f, v, t):
        if v <= 0.01: return
        p = s.at(*s.fan); L = s.L0(); s.star4(L, p, 44 * s.k * v, min(1.0, v), 0.2, 0.10); s.add(f, L, GOLD * 0.5 + 0.5, 0.8, 2, 6)

    def treasure(s, f, i, c, sz, v=1.0):
        k = s.k; x, y = int(c[0]), int(c[1]); r = int(sz)
        s.soft(f, c, min(sz * 1.5, 160 * s.k), 0.45 * v, TC[i])
        if v < 0.3: return
        if i == 0:                                                                   # 仏の御石の鉢
            cv2.ellipse(f, (x, y), (r, int(r * 0.8)), 0, 0, 180, (0.42, 0.46, 0.56), -1, cv2.LINE_AA); cv2.ellipse(f, (x, y), (r, int(r * 0.28)), 0, 0, 360, (0.72, 0.78, 0.92), -1, cv2.LINE_AA); cv2.ellipse(f, (x, y), (int(r * 0.7), int(r * 0.16)), 0, 0, 360, (0.25, 0.28, 0.38), -1, cv2.LINE_AA)
        elif i == 1:                                                                 # 蓬莱の玉の枝
            cv2.line(f, (x - r, y + r), (x + int(r * 0.2), y - int(r * 0.2)), (0.55, 0.42, 0.20), max(2, int(r * 0.14)), cv2.LINE_AA); cv2.line(f, (x - int(r * 0.2), y + int(r * 0.2)), (x + r, y + int(r * 0.1)), (0.55, 0.42, 0.20), max(2, int(r * 0.12)), cv2.LINE_AA); cv2.line(f, (x + int(r * 0.2), y - int(r * 0.2)), (x + int(r * 0.4), y - r), (0.55, 0.42, 0.20), max(2, int(r * 0.12)), cv2.LINE_AA)
            for j, (dx, dy) in enumerate(((0.4, -1.0), (1.0, 0.1), (0.2, -0.2), (-0.5, 0.5), (0.65, -0.55))): cv2.circle(f, (x + int(r * dx), y + int(r * dy)), max(2, int(r * 0.24)), tuple(float(q) for q in TC[(j + 1) % 5] * 0.9 + 0.1), -1, cv2.LINE_AA)
        elif i == 2:                                                                 # 火ねずみの衣
            P = [(x - r, y + r), (x - int(r * 0.9), y - int(r * 0.2)), (x - int(r * 0.45), y + int(r * 0.2)), (x - int(r * 0.2), y - r), (x + int(r * 0.2), y), (x + int(r * 0.6), y - int(r * 0.7)), (x + r, y + r)]
            cv2.fillPoly(f, [np.array(P, np.int32)], (0.95, 0.25, 0.10), cv2.LINE_AA); cv2.fillPoly(f, [np.array([(x - int(r * 0.5), y + r), (x - int(r * 0.1), y - int(r * 0.2)), (x + int(r * 0.5), y + r)], np.int32)], (1.0, 0.72, 0.20), cv2.LINE_AA)
        elif i == 3:                                                                 # 竜の首の珠
            for j in range(5): cv2.ellipse(f, (x, y), (r, r), 0, j * 72 - 90, j * 72 - 18, tuple(float(q) for q in TC[j] * 0.85), -1, cv2.LINE_AA)
            cv2.circle(f, (x, y), int(r * 0.45), (0.95, 0.90, 1.0), -1, cv2.LINE_AA); cv2.circle(f, (x - int(r * 0.35), y - int(r * 0.4)), max(1, int(r * 0.16)), (1.0, 1.0, 1.0), -1, cv2.LINE_AA)
        else:                                                                        # 燕の子安貝
            cv2.ellipse(f, (x, y), (r, int(r * 0.68)), -20, 0, 360, (0.98, 0.94, 0.86), -1, cv2.LINE_AA); cv2.ellipse(f, (x, y), (int(r * 0.8), int(r * 0.10)), -20, 0, 360, (0.45, 0.30, 0.25), -1, cv2.LINE_AA)
            for j in range(-2, 3): cv2.line(f, (x + int(j * r * 0.28), y - int(j * r * 0.10) - int(r * 0.14)), (x + int(j * r * 0.28), y - int(j * r * 0.10) + int(r * 0.14)), (0.45, 0.30, 0.25), max(1, int(1.5 * k)), cv2.LINE_AA)

    def frame(s, t):
        if t < T_CUT: f, sh = s.sc_grove(t)
        elif t < T_TR: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_END: f, sh = s.sc_tre(t)
        else: f, sh = s.sc_end(t - T_END)
        f, sh = s.finale(f, sh, t, DUR, s.moon, (1, 0.84, 0.42), (0.75, 0.45, 1), -1)
        return s.post(f, GOLD, 0.30, sh * s.k, t, 0.97)

    # 1) 竹やぶ → 光る竹 → カッ → 姫 → 竹が 次々 切れる
    def sc_grove(s, t):
        W, H, k = s.W, s.H, s.k; c = t - T_CHOP; fn = t - T_FAN
        f = s.sky(t, ease(fn / 1.2) if fn > 0 else 0.0)
        if c > 0: s.soft(f, (s.CX, s.CY), W * 0.42, 0.35 * math.exp(-c * 1.5) + 0.12, GOLD)
        s.grove(f, t, T_FAN + 0.12 if fn > 0 else None, clamp((t - T_GLOW) / 0.7) * (0.75 + 0.25 * math.sin(t * 9)), True)
        s.fireflies(f, t, 1.0 - 0.6 * clamp(fn))
        if c > 0.05:
            a = c - 0.05; s.hero(f, t, clamp(a / 0.12), squash=-0.08 * math.exp(-a * 9) * math.cos(a * 16))
            s.glint(f, math.sin(clamp((fn + 0.12) / 0.4) * 3.1416) * 1.3 if fn > -0.12 else 0, t)
        if 0 < c < 0.1: f = f + (1 - c / 0.1) * 0.8
        return f * clamp(t / 0.4), 7 * math.exp(-c * 9) if c > 0 else 0

    # 2) 扇形の窓
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.sky(t, 1.0) * 0.6
        g = back(u / 0.32, 1.2); P = (W / 2, H * 0.60); R = W * 0.72; r0 = W * 0.16; a0, a1 = -90 - 62 * g, -90 + 62 * g
        m = s.L0(); cv2.ellipse(m, (int(P[0]), int(P[1])), (int(R), int(R)), 0, a0, a1, 1.0, -1, cv2.LINE_AA); cv2.circle(m, (int(P[0]), int(P[1])), int(r0), 0.0, -1, cv2.LINE_AA)
        win = np.zeros((H, W, 3), np.float32) + np.array([0.20, 0.13, 0.36], np.float32); win += (np.clip(1 - np.hypot(s.XX - P[0], s.YY - H * 0.30) / (W * 0.6), 0, 1) * 0.25)[:, :, None] * GOLD[None, None, :]
        zf = 0.80 + 0.04 * ease(u / 1.9); sc = zf * H / s.ah; img = s.pose(t)
        s.draw(win, img, P[0] - (s.fc[0] - s.aw / 2) * sc, H * 0.33 - (s.fc[1] - s.ah / 2) * sc, zf)
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        L = s.L0()                                                                   # 扇の 骨と ふち
        for i in range(9):
            a = math.radians(a0 + (a1 - a0) * i / 8.0); s.line(L, (P[0] + math.cos(a) * r0 * 0.3, P[1] + math.sin(a) * r0 * 0.3), (P[0] + math.cos(a) * (R if i in (0, 8) else r0 * 1.9), P[1] + math.sin(a) * (R if i in (0, 8) else r0 * 1.9)), 0.9, 3 if i in (0, 8) else 2)
        cv2.ellipse(L, (int(P[0]), int(P[1])), (int(R), int(R)), 0, a0, a1, 1.0, max(1, int(4 * k)), cv2.LINE_AA); cv2.ellipse(L, (int(P[0]), int(P[1])), (int(r0), int(r0)), 0, a0, a1, 1.0, max(1, int(3 * k)), cv2.LINE_AA)
        cv2.circle(L, (int(P[0]), int(P[1])), max(1, int(6 * k)), 1.0, -1, cv2.LINE_AA)
        f += (L + cv2.GaussianBlur(L, (0, 0), 3 * k) * 0.6)[:, :, None] * (GOLD * 0.9)[None, None, :]
        n = len(s.ht); step = W * 0.20
        for i, T in enumerate(s.ht):                                                 # 技名: 扇を ひらくように 左右へ
            a = (u - 0.55) / 0.35
            if a > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step * ease(a), H * 0.835, clamp(a * 2), 1.0)
        for i, T in enumerate(s.vo):
            if u > 0.45 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.755, 1.0, 1.0)
        if u > 1.3: s.text(f, s.txt_en, W / 2, H * 0.835 + 62 * k, ease((u - 1.3) / 0.3), 1.0)
        if u < 0.08: f = f + (1 - u / 0.08) * 0.7
        return f, 0

    def tpos(s, i, t):
        a = i * 1.2566 + (t - T_TR) * 1.5 - 1.2; return (s.CX + math.cos(a) * s.W * 0.39, s.CY - s.H * 0.06 + math.sin(a) * s.H * 0.13), math.sin(a)

    # 3) 五つの宝 → 月へ → 五色の輪
    def sc_tre(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_TR; sk = t - T_SUCK; b = t - T_BURST
        f = s.sky(t, 1.0); s.grove(f, t, T_FAN + 0.12, 0, True, 0.3 * math.exp(-b * 2) if b > 0 else 0)
        s._pose = dict(t=t, sway=0, breath=0, lean=0, squash=0, stretch=None); s._pl = (s.CX, s.CY, s.HF * H / s.ah); mo = s.at(*s.moon)
        items = []
        for i in range(5):
            v = u - 0.1 - i * 0.22
            if v <= 0: continue
            p, dep = s.tpos(i, t); sz = 50 * k * back(v / 0.2, 2.0)
            if sk > 0:
                e = ease2((sk - i * 0.06) / 0.3); p = (p[0] + (mo[0] - p[0]) * e, p[1] + (mo[1] - p[1]) * e - math.sin(e * 3.1416) * 40 * k); sz *= (1 - 0.7 * e); dep = -1 if e > 0.3 else dep
                if e >= 1: continue
            items.append((dep, i, p, sz, v))
        for dep, i, p, sz, v in items:
            if dep < 0: s.treasure(f, i, p, sz)
        if b > 0:                                                                    # 月が かがやく
            L = s.L0()
            for j in range(18): a = j * 0.349 + 0.1; ln = H * (0.9 if j % 2 else 0.55) * ease(b / 0.25); cv2.fillPoly(L, [np.array([mo, (mo[0] + math.cos(a - 0.03) * ln, mo[1] + math.sin(a - 0.03) * ln), (mo[0] + math.cos(a + 0.03) * ln, mo[1] + math.sin(a + 0.03) * ln)], np.int32)], 0.5 * (1 - 0.5 * clamp(b / 0.9)), cv2.LINE_AA)
            f += (cv2.GaussianBlur(L, (0, 0), 5 * k) * 0.55)[:, :, None] * GOLD[None, None, :]
            for j in range(5):                                                       # 五色の 輪
                v = (b - j * 0.09) / 0.7
                if 0 < v < 1: Lr = s.L0(); s.circ(Lr, mo, H * 0.9 * ease(v) + 10 * k, 1.0, max(2, int(12 * k))); f += (cv2.GaussianBlur(Lr, (0, 0), 3 * k) * (1 - v) * 1.1)[:, :, None] * TC[j][None, None, :]
        s.hero(f, t, squash=(-0.05 * math.exp(-b * 8) if b > 0 else 0))
        for dep, i, p, sz, v in items:
            if dep >= 0: s.treasure(f, i, p, sz)
            if v < 0.15: Lr = s.L0(); s.circ(Lr, p, (10 + 50 * v / 0.15) * k, 1 - v / 0.15, 2); s.add(f, Lr, TC[i], 0.8)
        if sk > 0 and b < 0: s.soft(f, mo, W * 0.16, 0.5 * clamp(sk / 0.5), GOLD)
        sh = 0
        if b > 0:
            if b < 0.1: f = f + (1 - b / 0.1) * 0.85
            sh = 14 * math.exp(-b * 6)
        if u < 0.1: f = f * (u / 0.1) + (1 - u / 0.1) * 0.5
        return f, sh

    # 4) 扇を ひとふりするたび、宝が 1つずつ こちらへ 飛んでくる。さいごの 竜の珠が 目の前を うめる
    def sc_end(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_END
        f = s.sky(t, 1.0); s.grove(f, t, T_FAN + 0.12, 0, True)
        last = u - (0.25 + 4 * 0.36); kick = 0.0
        for j in range(5): v = u - (0.25 + j * 0.36); kick = max(kick, math.exp(-v * 12) if v > 0 else 0)
        if last < 0.45: s.hero(f, t, squash=-0.05 * kick); s.glint(f, 1.3 * kick, t)
        p0 = s.at(*s.fan)
        for j, i in enumerate((0, 4, 2, 1, 3)):
            v = (u - (0.25 + j * 0.36)) / (0.5 if j < 4 else 0.75)
            if v <= 0 or (v >= 1 and j < 4): continue
            v = min(v, 1.0); e = v ** 2.3; ex = (W * (0.18, 0.86, 0.30, 0.74, 0.5)[j], H * (0.30, 0.34, 0.70, 0.66, 0.44)[j])
            c = (p0[0] + (ex[0] - p0[0]) * (v ** 0.8) + (ex[0] - W / 2) * 1.6 * e * (j < 4), p0[1] + (ex[1] - p0[1]) * (v ** 0.8) - math.sin(v * 3.1416) * 60 * k + (ex[1] - H * 0.45) * 1.6 * e * (j < 4))
            s.treasure(f, i, c, (22 + (520 if j < 4 else 760) * e) * k)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.4
        return f, 2 + 7 * kick + (8 * clamp(last / 0.75) ** 2 if last > 0 else 0)

    # ---- この子だけの曲: 律の音階(ニ・ホ・ト・イ・ロ)の 箏と、笙ふうの 高い和音。竹は 「カッ」「コン」。宝は 1つごとに 箏が 1音 上がる。月の輪で 笙が ふくらみ、箏の 流し弾き ----
    def audio(s, path):
        a = Snd(DUR, 37); D = 293.66; sc = [0, 2, 5, 7, 9]
        def fq(d, o=0): return D * 2 ** ((sc[d % 5] + 12 * (d // 5 + o)) / 12.0)
        def koto(d, t0, g=0.11, o=0, dec=0.5): a.note(fq(d, o), t0, dec, g, "bell", 0.001); a.note(fq(d, o), t0, 0.04, g * 0.6, "saw", 0.001)
        def knock(t0, g=0.3, fr=820): a.note(fr, t0, 0.035, g, "sine", 0.0005); a.note(fr * 2.3, t0, 0.02, g * 0.4, "sine", 0.0005); a.hat(t0, g * 0.6, 0.01)
        def sho(t0, t1, g=0.022):
            for sm in (9, 14, 16, 21, 23): a.a += (a.tone(D * 2 * 2 ** (sm / 12.0)) + 0.4 * a.tone(D * 4 * 2 ** (sm / 12.0) * 1.002)) * a.gate(t0, t1, 0.5) * g
        a.a += a.lp(a.nz, 0.02) * 0.4 * a.gate(0, T_FAN, 0.4)                                                                    # 夜の やぶ
        for i in range(4): a.note(3600 + 300 * (i % 2), 0.2 + i * 0.27, 0.05, 0.02, "sine")                                       # 虫
        a.rise(T_GLOW, T_CHOP, 400, 900, 0.05); koto(0, 0.6, 0.07, 0, 0.8); koto(2, 1.0, 0.07, 0, 0.6)
        knock(T_CHOP, 0.5, 700); a.rise(T_CHOP, T_CHOP + 0.06, 2000, 6000, 0.15); a.boom(T_CHOP + 0.05, 70, 0.6, 0.2)             # カッ
        for i, d in enumerate((5, 7, 9)): koto(d, T_CHOP + 0.12 + i * 0.09, 0.09)
        a.note(fq(10, 1), T_FAN, 0.5, 0.06, "bell")                                                                             # 扇
        for st in s.st: tc = T_FAN + 0.12 + abs(st[0] - s.W / 2) / s.W * 1.7; knock(tc, 0.22, 600 + st[1] * 9); a.hat(tc + 0.02, 0.1, 0.05)
        sho(T_FAN + 0.3, T_CUT + 0.1, 0.012)
        a.boom(T_CUT, 60, 0.7, 0.25)
        for i, (d, dt) in enumerate(((9, 0.2), (7, 0.45), (5, 0.7), (7, 0.95), (4, 1.2), (5, 1.45))): koto(d, T_CUT + dt, 0.12); koto(d - 5, T_CUT + dt, 0.06)
        for i in range(8): koto(i + 3, T_CUT + 0.55 + i * 0.035, 0.05, 0, 0.2)                                                   # 扇が ひらく
        for i in range(5): koto(5 + i, T_TR + 0.1 + i * 0.22, 0.13, 0, 0.7); a.note(fq(5 + i, 2), T_TR + 0.1 + i * 0.22, 0.3, 0.04, "sine"); a.boom(T_TR + 0.1 + i * 0.22, 90, 0.2, 0.08)
        sho(T_TR, T_SUCK, 0.014)
        a.rise(T_SUCK, T_BURST, 500, 2400, 0.10)
        for i in range(5): a.note(fq(5 + i, 1), T_SUCK + i * 0.06 + 0.3, 0.15, 0.06, "bell")
        a.boom(T_BURST, 44, 1.4, 0.7); sho(T_BURST, T_END + 0.6, 0.03)
        for i in range(14): koto(14 - i, T_BURST + 0.05 + i * 0.04, 0.07, 0, 0.35)                                               # 流し弾き
        for j in range(5): a.note(fq(j, 2), T_BURST + j * 0.09, 0.5, 0.04, "bell")
        for j in range(5): t0 = T_END + 0.25 + j * 0.36; koto(5 + j, t0, 0.12, 0, 0.5); a.rise(t0, t0 + 0.45, 400 + 120 * j, 2200 + 300 * j, 0.09); a.whoosh(t0 + 0.2, t0 + 0.5, 0.35, 0.4); a.boom(t0 + 0.48, 70, 0.6, 0.12)
        a.rise(T_END + 1.7, T_END + 2.45, 200, 1500, 0.10)
        s.fin_snd(a, DUR, 37); a.save(path)


if __name__ == "__main__":
    r = Kaguya()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
