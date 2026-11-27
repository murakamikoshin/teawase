"""ノクス(★7 おわりの夜) 専用の必殺技「終夜ノ葬列」
夕焼けに立つ 黒い影 → 槍の石突きで 地を打つと 日が食われ、灯が ひとつずつ消える → ななめに裂ける カットイン →
黒い月の下を 提灯をさげた 影の葬列が歩く(鐘が鳴るたび 灯がゆれる) → 槍を かかげて 月を射抜く → 棺のふたが 左右から閉じ、砕ける
使い方: python3 hs_82.py 出力.mp4 / python3 hs_82.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
import hissatsu as hz

DUR = 13.6
VIO = np.array([0.62, 0.36, 1.0], np.float32); EMB = np.array([1.0, 0.45, 0.16], np.float32); BONE = np.array([0.92, 0.9, 1.0], np.float32); INK = np.array([0.05, 0.03, 0.10], np.float32)
T_STRIKE, T_CUT, T_MARCH, T_AIM, T_SHOT, T_LID, T_BREAK = 1.6, 3.5, 5.6, 9.3, 10.4, 11.2, 12.2
TOLL = [T_MARCH + 0.25 + i * 0.95 for i in range(4)]


class Nox(Kit):
    def __init__(s, W=540):
        super().__init__(82, (0.62, 0.36, 1.0), "終夜ノ葬列", "Night's Requiem", W)
        s.eyeL = s.A(345, 347); s.eyeR = s.A(440, 347); s.tip = s.A(110, 62); s.butt = s.A(215, 690); s.hand = s.A(180, 495)
        s.ht = [s.text_img(c, hz.SERIF, int(78 * s.k), 3) for c in "終夜ノ葬列"]
        s.vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        rg = np.random.RandomState(82)
        s.fl = rg.rand(90, 5).astype(np.float32); s.sh = rg.rand(40, 6).astype(np.float32); s.fe = rg.rand(50, 5).astype(np.float32)
        s.crk = []
        for i in range(9):
            a = i * 0.698 + rg.rand() * 0.3; pts = [(0.0, a)]; r = 0.0
            while r < 1.0: r += 0.1 + rg.rand() * 0.15; a += (rg.rand() - 0.5) * 0.7; pts.append((r, a))
            s.crk.append(pts)
        s.yn = (s.YY / s.H)

    # ---- この子だけの絵 ----
    def sky(s, night):
        """夕焼け → 夜。night 0..1"""
        y = s.yn; hz_ = 0.80
        g = np.clip(1 - np.abs(y - hz_) / 0.55, 0, 1) ** 2
        f = (np.array([0.16, 0.07, 0.22], np.float32)[None, None, :] * (1 - y)[:, :, None] + g[:, :, None] * EMB[None, None, :] * 0.95) * (1 - night)
        f += night * (INK[None, None, :] + (g * 0.22)[:, :, None] * VIO[None, None, :])
        return f.astype(np.float32)

    def ground(s, f, y0):
        f[int(y0):, :, :] = INK * 0.5

    def eyes(s, f, v, streak=1.0):
        if v <= 0: return
        L = s.L0(); sc = s._pl[2]
        for e in (s.eyeL, s.eyeR):
            p = s.at(*e)
            cv2.ellipse(L, (int(p[0]), int(p[1])), (max(1, int(17 * sc)), max(1, int(22 * sc))), 0, 0, 360, float(v), -1, cv2.LINE_AA)
            s.line(L, (p[0], p[1]), (p[0] - 120 * sc * streak * v, p[1] - 26 * sc * streak * v), 0.8 * v, 1)     # 目じりから 上へ流れる
        s.add(f, L, VIO, 1.4, 2, 9)

    def flames(s, f, p, t, power=1.0, n=60):
        """槍の穂先に まとわりつく 黒紫の炎"""
        L = s.L0(); D = s.L0()
        for i in range(n):
            r = s.fl[i]; v = (t * (0.7 + r[2]) + r[0]) % 1.0
            x = p[0] + (r[1] - 0.5) * 60 * s.k * (1 - v * 0.5) + math.sin(t * 5 + r[3] * 9) * 10 * s.k * v; y = p[1] + 40 * s.k - v * (130 + 90 * r[4]) * s.k * power
            rad = max(1, int((13 * (1 - v) + 2) * s.k * power))
            cv2.circle(L if i % 3 else D, (int(x), int(y)), rad, float(0.55 * (1 - v)), -1, cv2.LINE_AA)
        f *= (1 - np.clip(cv2.GaussianBlur(D, (0, 0), 5 * s.k) * 1.5, 0, 0.9))[:, :, None]
        s.add(f, L, VIO, 0.8 * power, 4, 14)

    def mourner(s, L, D, x, y, h, lamp, bow=0.0):
        """葬列のひとり: とがった頭巾の影 と 提灯"""
        w = h * 0.36; top = (x + bow * h * 0.25, y - h)
        body = np.array([top, (x + w * 0.55, y - h * 0.62), (x + w, y), (x - w, y), (x - w * 0.55, y - h * 0.62)], np.int32)
        cv2.fillPoly(D, [body], 1.0, cv2.LINE_AA); cv2.polylines(L, [body], True, 0.28, 1, cv2.LINE_AA)
        px = x + w * 1.25; py = y - h * 0.78
        s.line(L, (x + w * 0.3, y - h * 0.5), (px, py), 0.35, 1); s.line(L, (px, py), (px, py + h * 0.16), 0.35, 1)
        cv2.circle(L, (int(px), int(py + h * 0.24)), max(2, int(h * 0.075)), float(lamp), -1, cv2.LINE_AA)
        return (px, py + h * 0.24)

    def moon(s, f, O, R, t, crack=0.0):
        """黒い月(日食): 黒い円 と むらさきの光の輪"""
        L = s.L0(); cv2.circle(L, (int(O[0]), int(O[1])), int(R * 1.04), 1.0, -1, cv2.LINE_AA)
        for i in range(40):
            a = i * 0.157 + t * 0.2; r1 = R * (1.12 + 0.22 * (0.5 + 0.5 * math.sin(i * 2.7 + t * 2)))
            s.line(L, (O[0] + math.cos(a) * R, O[1] + math.sin(a) * R), (O[0] + math.cos(a) * r1, O[1] + math.sin(a) * r1), 0.6, 1)
        s.add(f, L, VIO * 0.8 + BONE * 0.2, 0.9, 4, 16)
        core = s.L0(); cv2.circle(core, (int(O[0]), int(O[1])), int(R), 1.0, -1, cv2.LINE_AA); f *= (1 - core)[:, :, None]
        if crack > 0:
            L = s.L0()
            for pts in s.crk:
                for j in range(len(pts) - 1):
                    if pts[j][0] > crack: break
                    s.line(L, (O[0] + math.cos(pts[j][1]) * pts[j][0] * R, O[1] + math.sin(pts[j][1]) * pts[j][0] * R),
                           (O[0] + math.cos(pts[j + 1][1]) * pts[j + 1][0] * R, O[1] + math.sin(pts[j + 1][1]) * pts[j + 1][0] * R), 1.0, 2)
            s.add(f, L, BONE, 1.2)

    def frame(s, t):
        k = s.k
        if t < T_CUT: f, sh = s.sc_dusk(t)
        elif t < T_MARCH: f, sh = s.sc_cut(t - T_CUT)
        elif t < T_AIM: f, sh = s.sc_march(t)
        else: f, sh = s.sc_end(t)
        return s.post(f, VIO, 0.8, sh * k, t, 0.80)

    # 1) 夕焼けの影。石突きで地を打つと 日が食われ、灯が消えていく
    def sc_dusk(s, t):
        W, H, k = s.W, s.H, s.k
        u = t - T_STRIKE; night = ease2(u / 1.5) if u > 0 else 0.0
        f = s.sky(night); gy = H * 0.82
        sun = (W / 2, H * 0.50); R = W * 0.23
        L = s.L0(); cv2.circle(L, (int(sun[0]), int(sun[1])), int(R), 1.0, -1, cv2.LINE_AA)
        f += (L + cv2.GaussianBlur(L, (0, 0), 30 * k) * 0.8)[:, :, None] * (EMB * 0.9 + 0.2)[None, None, :] * (1 - night * 0.6)
        if u > 0:                                                                    # 黒い円が 日を食う
            m = (sun[0] + R * 2.3 * (1 - ease2(u / 1.4)), sun[1] - R * 0.5 * (1 - ease2(u / 1.4)))
            s.moon(f, m, R * 1.02, t)
        s.ground(f, gy)
        Lc = s.L0()
        for i in range(7):                                                           # 道ばたの灯: 左から ひとつずつ消える
            x = W * (0.08 + i * 0.14); out = u - 0.25 - i * 0.16
            if out < 0:
                fl = 1 + 0.25 * math.sin(t * 13 + i * 2); cv2.ellipse(Lc, (int(x), int(gy - 16 * k)), (int(4 * k), int(9 * k * fl)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
            elif out < 0.7:                                                          # 消えたあとの けむり
                s.line(Lc, (x, gy - 20 * k - out * 50 * k), (x + math.sin(out * 9) * 5 * k, gy - 40 * k - out * 70 * k), 0.4 * (1 - out / 0.7), 1)
            s.line(Lc, (x, gy), (x, gy - 8 * k), 0.3, 2)
        s.add(f, Lc, EMB * (1 - night) + VIO * night, 0.9)
        sq = 0.09 * math.exp(-u * 7) * math.cos(u * 26) if u > 0 else -0.04 * clamp((t - 1.0) / 0.6)     # ため → 打つ → ゆれが おさまる
        img = s.pose(t, sway=(0.9 * math.exp(-u * 3) if u > 0 else 0.0), breath=0.25, squash=sq)
        light = np.full((s.ph, s.pw), 0.04 + 0.96 * ease2((u - 0.5) / 1.2) if u > 0 else 0.04, np.float32)
        s.draw(f, img, W / 2, gy - H * 0.21 + 6 * k, 0.43, light=light)
        if u > 0:
            b = s.at(*s.butt); L = s.L0()
            for i in range(3):
                v = (u - i * 0.12) / 0.8
                if 0 < v < 1: s.ell(L, (b[0], gy), (W * 0.95 * v, W * 0.10 * v), 0, 1 - v, 3)
            s.line(L, (b[0] - W * ease(u / 0.3), gy), (b[0] + W * ease(u / 0.3), gy), clamp(1.4 - u), 2)
            s.add(f, L, VIO, 1.0)
            if u < 0.12: f = f * 0.3 + (1 - u / 0.12) * 0.5
            if u > 1.0: s.eyes(f, clamp((u - 1.0) / 0.4), 1.0)
            if u > 0.9: s.flames(f, s.at(*s.tip), t, clamp((u - 0.9) / 0.6) * 0.8, 40)
        return f * clamp(t / 0.35), (12 * math.exp(-u * 5) if u > 0 else 0)

    # 2) ななめに裂けた画面: 上に目、下に槍。まん中に 技名が 打ちこまれる
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        sl = ease(u / 0.28)
        img = s.pose(t, sway=0.0, breath=0.0)
        A = np.zeros((H, W, 3), np.float32) + (INK * 1.5 + VIO * 0.10)[None, None, :]
        B = np.zeros((H, W, 3), np.float32) + (VIO * 0.16 + INK)[None, None, :]
        zf = 1.25; sc = zf * H / s.ah
        ex = (s.eyeL[0] + s.eyeR[0]) / 2
        s.draw(A, img, W * 0.5 - (ex - s.aw / 2) * sc - (1 - sl) * W + u * 8 * k, H * 0.27 - (s.eyeL[1] - s.ah / 2) * sc, zf)
        s.eyes(A, 0.7 + 0.3 * math.sin(u * 14), 1.4)
        s.draw(B, img, W * 0.42 - (s.tip[0] + 40 * k - s.aw / 2) * sc + (1 - sl) * W - u * 8 * k, H * 0.80 - (s.tip[1] + 150 * k - s.ah / 2) * sc, zf)
        s.flames(B, s.at(*s.tip), t, 2.2, 80)
        ycut = H * 0.56 - (s.XX / W) * H * 0.12
        m = (s.YY < ycut).astype(np.float32)[:, :, None]
        f = A * m + B * (1 - m)
        band = np.clip(1 - np.abs(s.YY - ycut) / (62 * k), 0, 1)[:, :, None]; f = f * (1 - band * 0.92)
        L = s.L0(); s.line(L, (0, H * 0.56 - 62 * k), (W, H * 0.44 - 62 * k), 1.0, 2); s.line(L, (0, H * 0.56 + 62 * k), (W, H * 0.44 + 62 * k), 1.0, 2)
        s.add(f, L, VIO, 1.0)
        n = len(s.ht); step = W * 0.165
        for i, T in enumerate(s.ht):                                                 # 鐘のように 1字ずつ 落ちてくる
            a = (u - 0.4 - i * 0.2) / 0.16
            if a <= 0: continue
            x = W / 2 + (i - (n - 1) / 2) * step; y = H * 0.50 - (x / W - 0.5) * H * 0.12
            if a < 1:
                sc2 = 1 + (1 - a) * 1.8; T2 = tuple(cv2.resize(q, None, fx=sc2, fy=sc2) for q in T); s.text(f, T2, x, y - (1 - a) * 40 * k, a, 1.0)
            else:
                s.text(f, T, x, y, 1.0, 1.0)
                if a < 3:
                    L2 = s.L0(); s.circ(L2, (x, y), (30 + 40 * (a - 1)) * k, clamp(1 - (a - 1) / 2), 2); s.add(f, L2, VIO, 0.8)
        for i, T in enumerate(s.vo):
            if u > 0.2 + i * 0.1: s.text(f, T, W * 0.10 + i * 34 * k, H * 0.08, 1.0, 1.0)
        if u > 1.5: s.text(f, s.txt_en, W * 0.5, H * 0.50 + 70 * k, ease((u - 1.5) / 0.3), 1.0)
        if u < 0.1: f += (1 - u / 0.1) * 0.9
        return f, 0

    # 3) 黒い月の下の 葬列。鐘が鳴るたび 灯がゆれ、影が頭をたれる
    def sc_march(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_MARCH
        tl = min([abs(t - x) if t >= x else 9 for x in TOLL]); pulse = math.exp(-tl * 3.5)
        f = s.sky(1.0) * (1 + 0.8 * pulse); gy = H * 0.86
        O = (W / 2, H * 0.24); s.moon(f, O, W * 0.21, t)
        L = s.L0()
        for x in TOLL:                                                               # 鐘の波
            v = (t - x) / 1.2
            if 0 < v < 1: s.circ(L, O, W * (0.24 + 0.9 * v), (1 - v) * 0.8, 2)
        s.add(f, L, VIO, 0.8)
        L = s.L0(); D = s.L0()
        for i in range(9):                                                           # 遠くの列(右から左へ)
            x = ((i * 0.16 - u * 0.045) % 1.44 - 0.2) * W; y = H * 0.615 + math.sin(i * 1.7) * 4 * k
            s.mourner(L, D, x, y + abs(math.sin(u * 3 + i)) * 2 * k, H * 0.075, 0.5 + 0.5 * pulse, pulse)
        f *= (1 - D * 0.92)[:, :, None]; s.add(f, L, VIO * 0.7, 0.55, 2, 8)
        f[int(H * 0.615):int(gy), :, :] *= 0.75
        s.ground(f, gy)
        img = s.pose(t, sway=0.35 * pulse, breath=0.3, squash=0.012 * pulse)
        s.soft(f, (W / 2, H * 0.66), W * 0.30, 0.20 + 0.15 * pulse, VIO)
        s.draw(f, img, W / 2, gy - H * 0.20 + 5 * k, 0.41)
        s.eyes(f, 0.55 + 0.45 * pulse, 0.8)
        s.flames(f, s.at(*s.tip), t, 1.0 + 0.6 * pulse)
        L = s.L0(); D = s.L0(); lamps = []
        for i in range(5):                                                           # 手前の列(左から右へ、大きく)
            x = ((i * 0.3 + u * 0.085) % 1.5 - 0.25) * W
            lamps.append(s.mourner(L, D, x, H * 0.985 + abs(math.sin(u * 2.4 + i * 2)) * 3 * k, H * 0.19, 0.7 + 0.3 * pulse, pulse))
        f *= (1 - D * 0.97)[:, :, None]; s.add(f, L, VIO * 0.6 + EMB * 0.4, 0.9, 3, 14)
        for p in lamps: s.soft(f, p, 26 * k * (1 + pulse), 0.35, VIO * 0.6 + EMB * 0.4)
        if u < 0.2: f = f * (u / 0.2)
        return f, 2.5 * pulse

    # 4) 槍をかかげ、月を射抜く → 棺のふたが 閉じて、砕ける
    def sc_end(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_AIM; gy = H * 0.86
        O = (W / 2, H * 0.24); R = W * 0.21
        shot = t - T_SHOT
        f = s.sky(1.0) * (1.0 + (2.5 * math.exp(-shot * 2.5) if shot > 0 else 0))
        s.moon(f, O, R, t, clamp(shot / 0.5) if shot > 0 else 0)
        s.ground(f, gy)
        aim = ease(u / 0.9)
        rec = (0.10 * math.exp(-shot * 6) if shot > 0 else 0)
        img = s.pose(t, sway=(1.3 * math.exp(-shot * 2.5) if shot > 0 else 0.0), breath=0.25, lean=(-16 * aim + 30 * rec) * k, squash=-0.05 * aim + rec, jit=(1.0 * k * clamp((u - 0.5) / 0.6) if shot < 0 else 0))
        s.soft(f, (W / 2, H * 0.66), W * 0.32, 0.25, VIO)
        s.draw(f, img, W / 2, gy - H * 0.20 + 5 * k, 0.41)
        s.eyes(f, 1.0, 1.2 + aim)
        tip = s.at(*s.tip)
        if shot < 0:                                                                 # 闇が 穂先に 集まる
            L = s.L0()
            for i in range(50):
                r = s.fl[i]; v = (u * (0.8 + r[2]) + r[0]) % 1.0; a = r[1] * 6.283 + u
                rr = (1 - v) * W * (0.25 + 0.4 * r[3]); s.line(L, (tip[0] + math.cos(a) * rr, tip[1] + math.sin(a) * rr), (tip[0] + math.cos(a) * rr * 0.86, tip[1] + math.sin(a) * rr * 0.86), 0.7 * v * aim, 1)
            cv2.circle(L, (int(tip[0]), int(tip[1])), int((6 + 20 * aim + 3 * math.sin(u * 40)) * k), 1.0, -1, cv2.LINE_AA)
            s.add(f, L, VIO, 1.0 + aim); s.flames(f, tip, t, 1.4 + aim)
        else:                                                                        # 射抜く: 黒い芯の むらさきの光
            w = (30 * math.exp(-shot * 2.2) + 5) * k * (1 + 0.15 * math.sin(shot * 60))
            L = s.L0(); dx = O[0] - tip[0]; dy = O[1] - tip[1]; far = (tip[0] + dx * 4, tip[1] + dy * 4)
            cv2.line(L, (int(tip[0]), int(tip[1])), (int(far[0]), int(far[1])), 1.0, max(1, int(w)), cv2.LINE_AA)
            s.add(f, L, VIO, 1.5, 6, 26)
            C = s.L0(); cv2.line(C, (int(tip[0]), int(tip[1])), (int(far[0]), int(far[1])), 1.0, max(1, int(w * 0.35)), cv2.LINE_AA); f *= (1 - C * 0.95)[:, :, None]
            L = s.L0()
            for i in range(3):
                v = (shot - i * 0.1) / 0.7
                if 0 < v < 1: s.circ(L, O, R * (1 + 2.2 * v), 1 - v, 3)
            for i in range(50):                                                      # 黒い羽が 降る
                r = s.fe[i]; v = shot * (0.3 + 0.5 * r[2]); x = r[0] * W + math.sin(shot * 3 + r[3] * 6) * 20 * k; y = (r[1] * 0.5 - 0.2 + v) * H
                cv2.ellipse(L, (int(x), int(y)), (int(10 * k), int(3 * k)), math.degrees(r[4] * 6 + shot * 2), 0, 360, 0.5, -1, cv2.LINE_AA)
            s.add(f, L, VIO * 0.7 + BONE * 0.3, 0.9)
            if shot < 0.1: f += (1 - shot / 0.1) * 1.2
        lid = t - T_LID
        if lid > 0:                                                                  # 棺のふた: 左右から 黒い板が閉じる
            brk = t - T_BREAK
            if brk < 0:
                c = ease2(lid / 0.7); x0 = int(W / 2 * c); gap = max(0, int(3 * k * (1 - clamp((lid - 0.7) / 0.2))))
                keep = f[:, max(0, W // 2 - gap):W // 2 + gap + 1].copy()
                f[:, :x0] = INK * 0.4; f[:, W - x0:] = INK * 0.4
                L = s.L0()
                if x0 > 2: s.line(L, (x0, 0), (x0, H), 0.9, 2); s.line(L, (W - x0, 0), (W - x0, H), 0.9, 2)
                for sx in (x0 - W * 0.25, W - x0 + W * 0.25):                         # ふたの飾り(十字)
                    s.line(L, (sx, H * 0.30), (sx, H * 0.70), 0.35, 2); s.line(L, (sx - W * 0.09, H * 0.42), (sx + W * 0.09, H * 0.42), 0.35, 2)
                s.add(f, L, VIO, 0.9)
                if c >= 1 and gap > 0: f[:, W // 2 - gap:W // 2 + gap + 1] = keep[:, :2 * gap + 1] + 0.5
                if lid > 0.8: s._pl = (W / 2, gy - H * 0.20 + 5 * k, 0.41 * H / s.ah); s.eyes(f, clamp((lid - 0.8) / 0.15), 1.6)     # 闇の中に 目だけ
            else:                                                                    # 砕ける
                f = np.zeros_like(f) + (VIO * 0.5 + 0.5)[None, None, :] * math.exp(-brk * 4) * 1.6
                L = s.L0()
                for i in range(40):
                    r = s.sh[i]; a = r[0] * 6.283; rr = W * (0.05 + brk * (0.6 + 1.4 * r[1])); c0 = (W / 2 + math.cos(a) * rr, H * 0.5 + math.sin(a) * rr * 1.5)
                    r2 = W * (0.03 + 0.07 * r[2]); b = r[3] * 6.283 + brk * 6 * (r[4] - 0.5)
                    s.poly(L, [(c0[0] + math.cos(b + j * 2.094) * r2 * (1.7 if j == 0 else 1), c0[1] + math.sin(b + j * 2.094) * r2) for j in range(3)], 0.8 * clamp(1.6 - brk), 2, i % 2 == 0)
                s.line(L, (W / 2, 0), (W / 2, H), clamp(1 - brk * 2), 3)
                s.add(f, L, VIO, 1.1)
                s._pl = (W / 2, gy - H * 0.20 + 5 * k, 0.41 * H / s.ah); s.eyes(f, clamp(1.2 - brk * 0.9), 1.6)
                f *= clamp((DUR - t) / 0.4)
        return f, (16 * math.exp(-shot * 3) if shot > 0 else 3 * aim) + (10 * math.exp(-(t - T_BREAK) * 5) if t > T_BREAK else 0)

    # ---- この子だけの曲: 葬送の行進。低い鐘、付点のたいこ、オルガン、フリギアの下降 ----
    def audio(s, path):
        a = Snd(DUR, 82); C3 = 130.81
        phr = [0, 1, 3, 5, 7, 8, 10]
        def fq(d, o=0): return C3 * 2 ** ((phr[d % 7] + 12 * (d // 7 + o)) / 12.0)
        a.a += a.lp(a.nz, 0.02) * 0.5 * a.gate(0, T_STRIKE, 0.4)                                                    # 夕暮れの風
        for d in (0, 4, 7): a.note(fq(d), 0.2, 1.6, 0.05, "pad", 0.5)
        a.boom(T_STRIKE, 44, 1.3, 0.6)
        for i in range(7): a.note(fq(11 - i, 1), T_STRIKE + 0.25 + i * 0.16, 0.25, 0.08, "bell")                    # 灯が 消えていく(下降)
        a.rise(T_STRIKE, T_CUT, 60, 30, 0.2)
        a.boom(T_CUT, 40, 1.1)
        for i in range(5): a.note(fq(0, -1), T_CUT + 0.4 + i * 0.2 + 0.16, 0.5, 0.20, "bell"); a.boom(T_CUT + 0.4 + i * 0.2 + 0.16, 70, 0.35, 0.15)
        for tk in TOLL:                                                                                              # 鐘
            a.note(fq(0, -1), tk, 1.6, 0.30, "bell"); a.note(fq(4, 0), tk, 1.2, 0.10, "bell"); a.boom(tk, 38, 0.6, 0.5)
        bt = 0.475; tk = T_MARCH; i = 0
        mel = [7, 8, 7, 5, 4, 5, 3, 1]
        while tk < T_AIM:                                                                                            # 行進: ドン・(タ)ドン
            a.boom(tk, 55, 0.35, 0.12); a.boom(tk + bt * 0.75, 62, 0.2, 0.08); a.hat(tk + bt * 0.5, 0.04)
            for d in ((0, 2, 4) if (i // 2) % 2 == 0 else (1, 3, 5)): a.note(fq(d), tk, 0.42, 0.045, "sq", 0.03)       # オルガン
            a.note(fq(mel[i % 8], 1), tk, 0.45, 0.085, "fm"); tk += bt; i += 1
        a.rise(T_AIM, T_SHOT, 80, 520, 0.22); a.whoosh(T_AIM, T_SHOT, 0.45, 0.1)
        a.boom(T_SHOT, 36, 1.4, 0.8); a.a += a.lp(a.nz, 0.3) * a.env(T_SHOT, 0.01, 0.5) * 0.5
        for d in (0, 1, 4, 7): a.note(fq(d, 1), T_SHOT, 1.0, 0.06, "saw", 0.02)
        a.rise(T_LID, T_LID + 0.7, 200, 50, 0.2); a.boom(T_LID + 0.7, 48, 1.2, 0.3)                                  # ふたが 閉じる
        a.boom(T_BREAK, 60, 1.2, 0.5); a.a += (a.nz - a.lp(a.nz, 0.5)) * a.env(T_BREAK, 0.002, 0.35) * 0.35          # 砕ける
        a.note(fq(0, -1), T_BREAK + 0.1, 1.6, 0.28, "bell"); a.note(fq(7, 1), T_BREAK + 0.1, 1.2, 0.07, "bell")
        a.save(path)


if __name__ == "__main__":
    r = Nox()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1])
