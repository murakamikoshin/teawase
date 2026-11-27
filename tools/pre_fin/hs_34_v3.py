"""ロキ(★5 道化) 専用の必殺技「道化ノ千変」— 敵は さいしょから 画面の こちら側(客席)
まっ暗な舞台に スポットライトが ひとつ。ひざをついた 道化 → カチッ、カチッ。明かりが 消えるたび べつの場所に べつの色の ロキ → ジャン、5人 いっぺんに(ぜんぶ こちらを 見ている) →
ひし形の窓: 顔の右半分(仮面の側)だけ 色が 反転 → 舞台の壁が ひし形もように めくれ、まん中に 本物が 出る。本物が 指を 鳴らす(はじめの カチッ と 同じ音)たびに
分身が 1人ずつ 客席(こちら)へ とびかかってくる → 5人 行ったあと、さいごに 本物が とんで 仮面の目が 目の前に → 白 → 暗転
使い方: python3 hs_34.py 出力.mp4 / python3 hs_34.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 9.9
GOLD = np.array([1.0, 0.82, 0.30], np.float32); GRN = np.array([0.36, 0.72, 0.26], np.float32); PUR = np.array([0.46, 0.26, 0.80], np.float32)
TINT = [np.array(c, np.float32) for c in ((0.70, 0.40, 1.0), (0.45, 1.0, 0.40), (1.0, 0.80, 0.25), (0.30, 0.90, 1.0), (1.0, 0.35, 0.50))]
SPOT = [(0.22, 0.50, 0.21), (0.78, 0.50, 0.21), (0.25, 0.87, 0.27), (0.75, 0.87, 0.27), (0.50, 0.67, 0.22)]      # (x, 足もとy, 大きさ)
T_B, BEAT, T_ALL, T_CUT, T_D, T_S0, T_REAL, T_BURST, T_E = 1.9, 0.32, 3.5, 4.3, 6.2, 6.95, 8.05, 8.45, 9.2


class Loki(Kit):
    def __init__(s, W=540):
        super().__init__(34, (1.0, 0.82, 0.30), "道化ノ千変", "Trickster's Masque", W)
        H, k = s.H, s.k
        s.eL = s.A(300, 385); s.eR = s.A(447, 372); s.fc = s.A(372, 385); s.hilt = s.A(175, 495)
        s.ht = [s.text_img(c, hz.SERIF, int(78 * k), 3) for c in "道化ノ千変"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * k), 1) for c in "奥義"]
        s.flr = H * 0.86; s.HF = 0.42; s.CX = W / 2; s.CY = s.flr - s.HF * H / 2; s.TG = (W / 2, H * 0.30)
        lum = s.cp[:, :, :3].mean(2, keepdims=True) ** 0.8
        s.cop = []
        for i, c in enumerate(TINT):                                                  # 色ちがいの 分身(左右も 入れかえる)
            im = s.cp.copy(); im[:, :, :3] = np.clip(lum * c[None, None, :] * 1.25, 0, 1)
            s.cop.append(im[:, ::-1].copy() if i % 2 == 0 else im)
        # ひし形の タイル
        a = W / 5.0; b = a * 1.5; u = s.XX / a + s.YY / b; v = s.XX / a - s.YY / b
        iu = np.floor(u); iv = np.floor(v); s.fu = u - iu - 0.5; s.fv = v - iv - 0.5
        s.par = ((iu + iv) % 2).astype(np.float32)
        xc = (iu + iv + 1) * a / 2; yc = (iu - iv) * b / 2
        s.td = np.sqrt((xc - s.CX) ** 2 + (yc - s.CY) ** 2).astype(np.float32)
        s.hq = (GRN[None, None, :] * s.par[:, :, None] + PUR[None, None, :] * (1 - s.par[:, :, None])).astype(np.float32)
        s.cf = np.random.RandomState(34).rand(90, 5).astype(np.float32)

    # ---- 大道具 ----
    def tiles(s, p, gain=0.5):
        """p: タイルごとの めくれ具合(0=闇 1=もよう)。めくれる とちゅうは 細くなる"""
        wd = np.abs(np.cos(np.pi * p)); vis = (np.abs(s.fu) < 0.5 * wd - 0.015).astype(np.float32)
        edge = (np.maximum(np.abs(s.fu) / np.maximum(wd, 0.05), np.abs(s.fv)) > 0.455).astype(np.float32) * vis
        on = (p > 0.5).astype(np.float32)
        f = (s.hq * gain * on[:, :, None] + np.array([0.035, 0.03, 0.05], np.float32) * (1 - on[:, :, None])) * vis[:, :, None]
        f += (edge * on * 0.5 * gain)[:, :, None] * GOLD[None, None, :]
        return f * (0.45 + 0.55 * np.clip(1.25 - s.YY / s.H, 0, 1))[:, :, None]

    def spot(s, f, cx, fy, r, v, col=WHITE):
        if v <= 0.01: return
        L = s.L0(); tx = s.W / 2 + (cx - s.W / 2) * 0.45
        cv2.fillPoly(L, [np.array([(tx - r * 0.12, -10), (tx + r * 0.12, -10), (cx + r, fy), (cx - r, fy)], np.int32)], 1.0, cv2.LINE_AA)
        G = s.L0(); cv2.ellipse(G, (int(cx), int(fy)), (int(r), int(r * 0.22)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        f += ((cv2.GaussianBlur(L, (0, 0), 7 * s.k) * 0.13 + cv2.GaussianBlur(G, (0, 0), 3 * s.k) * 0.42) * v)[:, :, None] * (np.asarray(col, np.float32) * 0.6 + 0.4)[None, None, :]

    def copy(s, f, i, alpha=1.0, white=0.0):
        x, fy, hf = SPOT[i]; W, H = s.W, s.H
        s.spot(f, x * W, fy * H, hf * H * 0.42, alpha, TINT[i])
        s.put(f, s.cop[i], x * W, fy * H - hf * H / 2, hf * H / s.ah, alpha, 0.0, white)

    def real(s, f, t, v=1.0, **kw):
        s.spot(f, s.CX, s.flr, s.HF * s.H * 0.42, v, GOLD)
        img = s.pose(t, **kw); s.draw(f, img, s.CX, s.CY, s.HF, v)

    def gleam(s, f, v, t):
        if v <= 0.01: return
        p = s.at(*s.eR); L = s.L0(); s.star4(L, p, 46 * s.k * v, min(1.0, v), 0.3 + t * 0.6, 0.10); s.add(f, L, GOLD * 0.6 + 0.4, 0.8, 2, 6)

    def frame(s, t):
        if t < T_B: f, sh = s.sc_solo(t)
        elif t < T_CUT: f, sh = s.sc_blink(t)
        elif t < T_D: f, sh = s.sc_cut(t - T_CUT)
        else: f, sh = s.sc_strike(t)
        f, sh = s.finale(f, sh, t, DUR, s.eR, (0.7, 0.4, 1), (0.45, 1, 0.4), -1)
        return s.post(f, GOLD, 0.35, sh * s.k, t, 0.93)

    # 1) スポットライト ひとつ
    def sc_solo(s, t):
        f = np.zeros((s.H, s.W, 3), np.float32) + np.array([0.012, 0.01, 0.02], np.float32)
        on = 1.0 if t > 0.35 else 0.0
        if 0.35 < t < 0.5: on = 0.4 + 0.6 * (int(t * 60) % 2)                            # 点くときの ちらつき
        s.real(f, t, on)
        if on: s.gleam(f, math.sin(clamp((t - 1.2) / 0.45) * 3.1416), t)
        return f, 0

    # 2) カチッ のたび べつの場所に
    def sc_blink(s, t):
        f = np.zeros((s.H, s.W, 3), np.float32) + np.array([0.012, 0.01, 0.02], np.float32)
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
        win = np.zeros((H, W, 3), np.float32) + np.array([0.10, 0.07, 0.16], np.float32)
        zf = 1.0 + 0.05 * ease(u / 1.9); sc = zf * H / s.ah; img = s.pose(t)
        mx = C[0] - (s.fc[0] - s.aw / 2) * sc; s.draw(win, img, mx, C[1] + H * 0.03 - (s.fc[1] - s.ah / 2) * sc, zf)
        sx = int(s.at(s.fc[0] + 8 * s.ah / 760.0, 0)[0])                               # 顔の まん中の線
        fl = clamp((u - 0.5) / 0.12)
        if fl > 0: win[:, sx:] = win[:, sx:] * (1 - fl) + (1.0 - win[:, sx:]) * fl
        s.gleam(win, 1.2 * math.exp(-max(0.0, u - 0.5) * 4) * (u > 0.5), t)
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        L = s.L0(); cv2.polylines(L, [np.array(pts, np.int32)], True, 1.0, max(1, int(4 * k)), cv2.LINE_AA)
        if fl > 0: s.line(L, (sx, C[1] - hh), (sx, C[1] + hh), 0.8, 2)
        Lm = L * cv2.dilate(m, np.ones((9, 9), np.uint8)); s.add(f, Lm, GOLD, 0.6, 2, 6)
        n = len(s.ht); step = W * 0.175
        for i, T in enumerate(s.ht):                                                 # 技名: 上と下から たがいちがいに
            a = (u - 0.6 - i * 0.11) / 0.22
            if a > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.835 + (1 - ease(a)) * 60 * k * (1 if i % 2 else -1), clamp(a * 2), 1.0)
        for i, T in enumerate(s.vo):
            if u > 0.45 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.755, 1.0, 1.0)
        if u > 1.3: s.text(f, s.txt_en, W / 2, H * 0.835 + 60 * k, ease((u - 1.3) / 0.3), 1.0)
        if u < 0.08: f = f + (1 - u / 0.08) * 0.7
        return f, 0

    LEAP = (4, 0, 3, 1, 2); L0T = 0.75; LGAP = 0.26

    # 4) 壁が めくれ、本物が まん中に。指を 鳴らすたび 分身が 1人ずつ こちらへ → さいごに 本物
    def sc_strike(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_D
        f = s.tiles(np.clip((u - s.td / (W * 2.6)) / 0.35, 0, 1), 0.36 + 0.2 * clamp((u - s.L0T) / 2.2))
        kick = 0.0; tint = None
        for j, i in enumerate(s.LEAP):
            v0 = u - (s.L0T + j * s.LGAP)
            if v0 > 0 and math.exp(-v0 * 12) > kick: kick = math.exp(-v0 * 12); tint = TINT[i]
        for j, i in enumerate(s.LEAP):                                               # まだ 立っている 分身
            if u < s.L0T + j * s.LGAP: s.copy(f, i, 1.0, 0.0)
        r = u - 0.35; rl = (u - (s.L0T + 5 * s.LGAP + 0.15)) / 0.75                      # 本物: 出る → 指を鳴らす → さいごに とぶ
        if r > 0 and rl <= 0:
            s.real(f, t, 1.0, squash=0.07 * math.exp(-r * 10) * math.cos(r * 20) - 0.045 * kick); s.gleam(f, 0.9 * kick, t)
            if r < 0.06: f = f + (1 - r / 0.06) * 0.35
        if tint is not None and kick > 0.3: f = f + (kick - 0.3) * 0.25 * tint[None, None, :]
        for j, i in enumerate(s.LEAP):                                               # とびかかる 分身
            v = (u - (s.L0T + j * s.LGAP)) / 0.55
            if v <= 0 or v >= 1: continue
            x, fy, hf = SPOT[i]; e = v ** 2.4; sc = (hf + 3.2 * e) * H / s.ah
            cx = x * W + (W / 2 - x * W) * min(1.0, v * 2) + (x - 0.5) * W * 2.2 * e; cy = (fy - hf / 2) * H + (H * 0.45 - (fy - hf / 2) * H) * min(1.0, v * 2) + H * 0.25 * e
            s.put(f, s.cop[i], cx, cy, sc, clamp((1 - v) * 4), 0.0, 0.25 * e)
        if rl > 0:                                                                   # 本物: 仮面の目が まん中に 来るように
            e = clamp(rl) ** 2.2; hf = s.HF + 3.4 * e; sc = hf * H / s.ah; img = s.pose(t)
            tx, ty = (s.eR[0] - s.aw / 2) * sc, (s.eR[1] - s.ah / 2) * sc
            s.draw(f, img, s.CX + (W / 2 - tx - s.CX) * clamp(rl * 1.5), s.CY + (H * 0.45 - ty - s.CY) * clamp(rl * 1.5), hf); s.gleam(f, 0.8 + 2.5 * e, t)
        if u < 0.1: f = f * (u / 0.1) + (1 - u / 0.1) * 0.5
        return f, 5 * kick + (10 * clamp(rl) ** 2 if rl > 0 else 0)

    # ---- この子だけの曲: ホ短調の しのび足(ピチカート風の ベースと 三全音の 口笛)。カチッ のたび 鈴。分身の線は 1本ずつ 音が 上がる。ぱちん のあと 鈴 ひとつ ----
    def audio(s, path):
        a = Snd(DUR, 34); E = 164.81
        def fq(semi): return E * 2 ** (semi / 12.0)
        def pizz(semi, t0, g=0.12): a.note(fq(semi) / 2, t0, 0.13, g, "fm", 0.002); a.note(fq(semi), t0, 0.06, g * 0.4, "sq", 0.002)
        def jingle(t0, g=0.06):
            for fr in (3150, 4210, 5330, 6800): a.note(fr, t0, 0.09, g, "sine", 0.0005)
            a.hat(t0, g * 0.8, 0.03)
        def click(t0): a.hat(t0, 0.35, 0.012); a.boom(t0 + 0.07, 90, 0.35, 0.10)
        def whistle(semi, t0, ln, g=0.07):
            ph = 2 * np.pi * fq(semi) * 4 * (a.t - t0) + 0.25 * np.sin(2 * np.pi * 5.5 * (a.t - t0)); a.a += np.sin(ph) * a.env(t0, 0.03, ln) * g
        a.a += a.tone(41.2) * 0.05 * a.gate(0, T_B, 0.3)
        click(0.35); jingle(0.5, 0.04)
        for i, sm in enumerate((0, 7, 3, 6)): pizz(sm, 0.7 + i * 0.3, 0.10)                     # しのび足
        whistle(12, 1.2, 0.25); whistle(18, 1.5, 0.3)
        seq = (0, 3, 6, 7, 10)
        for j in range(5): t0 = T_B + j * BEAT; click(t0); jingle(t0 + 0.07, 0.05); pizz(seq[j], t0 + 0.07, 0.13); whistle(seq[j] + 12, t0 + 0.07, 0.12, 0.05)
        click(T_ALL); a.boom(T_ALL + 0.07, 55, 0.9, 0.3)                                         # ジャン
        for sm in (0, 3, 6, 10): a.note(fq(sm) * 2, T_ALL + 0.07, 0.5, 0.07, "saw", 0.005); jingle(T_ALL + 0.07, 0.03)
        a.boom(T_CUT, 60, 0.8, 0.25)
        mel = ((12, 0.0), (15, 0.2), (18, 0.4), (19, 0.6), (18, 0.9), (15, 1.1), (13, 1.3), (12, 1.5))
        for sm, dt in mel: whistle(sm, T_CUT + 0.2 + dt, 0.16, 0.07)
        for i in range(8): pizz((0, 7, 3, 7, 1, 7, 6, 0)[i], T_CUT + 0.2 + i * 0.2, 0.11)
        a.note(fq(18) * 4, T_CUT + 0.5, 0.4, 0.06, "bell"); jingle(T_CUT + 0.5, 0.05)
        a.whoosh(T_D, T_D + 0.6, 0.25, 0.15)                                                    # 壁が めくれる(ぱたぱた)
        for i in range(14): a.hat(T_D + 0.05 + i * 0.045, 0.10, 0.012)
        a.boom(T_D + 0.35, 70, 0.4, 0.12); jingle(T_D + 0.4, 0.07); pizz(0, T_D + 0.4, 0.12)                                       # 本物が 出る
        for j in range(5):                                                                                                       # 指ぱっちん(はじめの カチッ と 同じ音)→ 分身が とぶ
            t0 = T_D + 0.75 + j * 0.26; click(t0); pizz(seq[j], t0 + 0.07, 0.13); whistle(seq[j] + 12, t0 + 0.07, 0.12, 0.05)
            a.rise(t0 + 0.05, t0 + 0.5, 500 + j * 150, 2400 + j * 300, 0.09); a.whoosh(t0 + 0.25, t0 + 0.55, 0.3, 0.4); jingle(t0 + 0.5, 0.05)
        tr = T_D + 0.75 + 5 * 0.26 + 0.15                                                                                        # 本物: カットインの 口笛の ふしを 速く
        for q, (sm, dt) in enumerate(mel[:6]): whistle(sm, tr + dt * 0.5, 0.10, 0.07)
        click(tr); a.rise(tr, tr + 0.75, 200, 1800, 0.12); a.boom(tr + 0.72, 42, 1.4, 0.5); a.note(fq(24) * 4, tr + 0.6, 0.6, 0.09, "bell"); jingle(tr + 0.6, 0.07)
        s.fin_snd(a, DUR, 34); a.save(path)


if __name__ == "__main__":
    r = Loki()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
