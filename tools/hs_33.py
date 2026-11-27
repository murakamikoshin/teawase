"""ジークフリート(★5 竜鱗の騎士) 専用の必殺技「竜鱗ノ剣」— 高品質版(hs_fx)。話は v3 と 同じ: 竜は 騎士の 相棒、敵は 画面の こちら側
溶岩の洞くつ(けむり・火の粉・溶岩の てり)。騎士の うしろの闇で 竜の目が ひらく → 竜が 騎士ごしに こちらへ 火を はく(騎士は うろこの盾で 平気) →
竜の ひとみに うつる 騎士のカットイン → 騎士が こちらへ 剣を ふる(まっ暗に 一閃)= 合図 → 竜が 騎士を とびこえて こちらへ かみついてくる → 白 → 暗転
使い方: python3 hs_33.py 出力.mp4 / python3 hs_33.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old33", "/home/claude/tools/pre_fin/hs_33_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = old.DUR; GREEN, AMBER, LAVA, YEL = old.GREEN, old.AMBER, old.LAVA, old.YEL
T_FIRE, T_CUT, T_SL, T_BLK, T_AFT = old.T_FIRE, old.T_CUT, old.T_SL, old.T_BLK, old.T_AFT
SMOKE = [(0.10, 0.025, 0.012), (0.34, 0.07, 0.02), (0.75, 0.26, 0.05)]


class Sieg(FX, old.Sieg):
    def __init__(s, W=540):
        old.Sieg.__init__(s, W); s.fx_init(33); H, k = s.H, s.k
        sh = lambda a, dy: cv2.warpAffine(a, np.float32([[1, 0, 0], [0, 1, dy]]), (W, H))
        s.emb = np.clip(sh(s.scl, -3 * k) - s.scl, 0, 1); s.grv = cv2.GaussianBlur(s.scl, (0, 0), 1.0 * k)      # うろこ: 下からの てり / みぞ
        dd = cv2.distanceTransform((s.dm > 0.5).astype(np.uint8), cv2.DIST_L2, 5); s.dedge = np.exp(-dd / (9 * k)) * (s.dm > 0.3)
        s.dbulge = np.clip(dd / (W * 0.16), 0, 1) ** 0.6
        det = s.fbm(0.004, 3, 7, 4, 0, 1); s.ddet = 0.55 + 0.9 * det
        s.rrim = np.clip(s.rock - sh(s.rock, 4 * k), 0, 1); s.rdet = 0.5 + s.fbm(0.006, 9, 2, 4, 0, 2)
        # ひとみ(カットイン)の 虹彩: 放射の すじ + ふちの 暗がり
        C = (W / 2, H * 0.38); ang = np.arctan2(s.YY - C[1], s.XX - C[0]); rad = np.sqrt((s.XX - C[0]) ** 2 + (s.YY - C[1]) ** 2) / (W * 0.62)
        st = cv2.remap(s._tiles[0], ((ang / 6.2832 + 0.5) * 256 * 3 % 256).astype(np.float32), (rad * 40 % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        st2 = cv2.remap(s._tiles[1], ((ang / 6.2832 + 0.5) * 256 * 9 % 256).astype(np.float32), (rad * 16 % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        v = (0.25 + 1.1 * st * st2 * 2) * sstep(1.02, 0.92, rad) * (0.35 + 0.65 * sstep(0.1, 0.75, rad)) + sstep(0.86, 0.98, rad) * sstep(1.04, 0.98, rad) * 0.5
        s.iris2 = s.ramp(np.clip(v, 0, 1), [(0, (0.03, 0.01, 0.0)), (0.35, (0.55, 0.22, 0.02)), (0.7, (1.0, 0.62, 0.10)), (1.0, (1.0, 0.95, 0.55))])

    # ---- 洞くつ ----
    def cave(s, t, tint=0.0, smoke=1.0):
        H, W = s.H, s.W
        f = np.zeros((H, W, 3), np.float32) + np.array([0.012, 0.010, 0.016], np.float32)
        f += s.nebula(t, SMOKE, 0.0007, (4.0, -26.0), 0.55 * smoke) * np.clip(0.25 + s.yn * 1.1, 0, 1)[:, :, None]
        m = sstep(0.74, 1.02, s.yn).astype(np.float32); fl = s.flame(t, m, -150.0, 0.0022, 1.15)
        fire = s.ramp(np.clip(fl + m * 0.35, 0, 1), s.FIRE) * (0.9 + 0.1 * math.sin(t * 3.1))
        if tint > 0: fire = fire * (1 - tint) + fire[:, :, [1, 0, 2]] * tint * 0.9
        return f + fire

    def rocks(s, f, t=0.0, tint=0.0):
        m = s.rock[:, :, None]; body = np.array([0.050, 0.030, 0.026], np.float32)[None, None, :] * (s.rdet ** 2 * (0.4 + 1.6 * np.clip(1.25 - (s.yn - 0.86) * 9, 0, 1)))[:, :, None]
        f = f * (1 - m) + body * m
        f += (cv2.GaussianBlur(s.rrim, (0, 0), 1.2 * s.k) * 0.55)[:, :, None] * (LAVA * (1 - tint) + GREEN * tint)[None, None, :]
        return f

    def embers(s, f, t, tint=0.0, v=1.0):
        col = (LAVA * (1 - tint) + GREEN * tint) * 0.8 + 0.2
        s.starfield(f, t, 1100, col, 5, 0.9 * v, (14.0, -150.0)); s.starfield(f, t + 9, 160, col, 8, 2.4 * v, (26.0, -230.0))

    # ---- 竜 ----
    def eye(s, c, ex, ey, rot, op, glow, z=1.0):
        W = s.W; a = W * 0.118 * z; b = W * 0.050 * op * z; r_ = math.radians(rot); dx = s.XX - ex; dy = s.YY - ey
        u = (dx * math.cos(r_) + dy * math.sin(r_)) / a; v = (-dx * math.sin(r_) + dy * math.cos(r_)) / max(1.0, b); u = u + 0.35 * np.sign(u) * v * v * 0   # アーモンド形
        r2 = np.abs(u) ** 2.0 + (np.abs(v) * (1 + 0.9 * np.abs(u))) ** 2.0; m = sstep(1.0, 0.86, r2)
        if m.max() <= 0: return
        st = cv2.remap(s._tiles[2], ((np.arctan2(dy, dx) / 6.2832 + 0.5) * 256 * 5 % 256).astype(np.float32), np.zeros_like(dx) + 30, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        val = np.clip((1.05 - np.sqrt(r2) * 0.75) * (0.55 + 0.9 * st), 0, 1)
        col = s.ramp(val, [(0, (0.25, 0.06, 0.0)), (0.4, (0.95, 0.42, 0.03)), (0.75, (1.0, 0.80, 0.18)), (1.0, (1.0, 1.0, 0.75))]) * glow
        pw = 0.085 + 0.05 * (1 - op); pup = sstep(1.0, 0.75, (u / pw) ** 2 + (v / 0.93) ** 2); col = col * (1 - pup)[:, :, None]
        col += (sstep(0.75, 1.0, r2) * 0.0)[:, :, None]
        hl = np.exp(-(((u + 0.28) / 0.10) ** 2 + ((v + 0.42) / 0.22) ** 2)); col += hl[:, :, None] * 0.9 * glow                       # うつりこみ
        lid = sstep(0.80, 1.0, r2)[:, :, None]; col = col * (1 - 0.8 * lid)
        c[:] = c * (1 - m[:, :, None]) + col * m[:, :, None]

    def scales(s, X, Y):
        """うろこ(どの 大きさでも くっきり): 竜の座標 X,Y → (ふちからの 位置 d 0..1, うろこ内の 上下 v, うろこごとの 乱数)"""
        R = 24 * s.k; rs = R * 0.62; cs = R * 1.84; D = np.full(X.shape, 2.0, np.float32); V = np.zeros_like(D); Hh = np.zeros_like(D); j0 = np.floor(Y / rs)
        for dj in (0, 1, 2):                                                         # 下の段から 上の段へ 上書き(上の うろこが 手前)
            j = j0 - dj; yc = j * rs; off = np.where(np.mod(j, 2) == 1, R * 0.92, 0.0) - R
            i = np.round((X - off) / cs); xc = i * cs + off; d = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2) / R; ok = d < 1.0
            D = np.where(ok, d, D); V = np.where(ok, (Y - yc) / R, V); Hh = np.where(ok, np.mod(np.sin(i * 12.9898 + j * 78.233) * 43758.5453, 1.0), Hh)
        return D, V, Hh.astype(np.float32)

    def dragon(s, t, a, op, glow=1.0, tint=0.0, lit=0.0, z=1.0, tx=0.0, ty=0.0, jaw=0.0):
        """竜の層 (色, 不透明度)を 画面の座標で 作る(z 倍・tx,ty ずらし)。lit: こちらからの 明るさ"""
        H, W, k = s.H, s.W, s.k; lc = LAVA * (1 - tint) + GREEN * 0.8 * tint; M = np.float32([[z, 0, tx], [0, z, ty]])
        wp = lambda A: cv2.warpAffine(A, M, (W, H), flags=cv2.INTER_CUBIC)
        X = (s.XX - tx) / z; Y = (s.YY - ty) / z; yn = Y / H
        m = sstep(0.5 - 0.35 / z, 0.5 + 0.35 / z, wp(s.dm)) if z > 1.05 else s.dm
        under = np.clip((yn - 0.16) / 0.36, 0, 1) ** 1.4; bulge = wp(s.dbulge) if z > 1.05 else s.dbulge; edge = wp(s.dedge) if z > 1.05 else s.dedge
        D, V, Hh = s.scales(X, Y); aa = 0.05 / z + 0.02
        grv = sstep(0.90 - aa, 0.97, D); rimhl = sstep(0.62, 0.86, D) * sstep(0.93, 0.88 - aa, D) * np.clip(V * 1.4, 0, 1)             # みぞ / 下ふちの てり
        dome = np.clip(1.0 - D * D, 0, 1) ** 0.5 * (0.55 + 0.45 * np.clip(1 - V, 0, 1))                                               # うろこの ふくらみ
        det = s.fbm(0.004, 3 + tx * 0.01, 7, 4, 0, 1) if z > 1.05 else s.ddet / 0.9 - 0.6
        sock = 0.0; ridge = np.exp(-((X - W / 2) / (W * 0.07)) ** 2) * np.clip((yn - 0.08) / 0.2, 0, 1)
        for ex, ey, rot in s.EY: sock = sock + np.exp(-(((X - ex) / (W * 0.17)) ** 2 + ((Y - ey) / (H * 0.045)) ** 2))
        shade = (0.45 + 0.75 * bulge) * (1 - 0.55 * np.clip(sock, 0, 1)) * (1 + 0.5 * ridge) * (0.75 + 0.5 * Hh) * (0.7 + 0.6 * det)
        body = np.array([0.030, 0.085, 0.052], np.float32)[None, None, :] * (shade * dome * (0.55 + 4.0 * lit))[:, :, None]
        body += (under * shade * dome * (0.28 + 0.2 * Hh))[:, :, None] * (lc * np.array([0.50, 0.34, 0.30], np.float32))[None, None, :]
        body += (rimhl * shade * (0.10 + 0.9 * under + 1.3 * lit))[:, :, None] * (lc * 0.55 + GREEN * (0.25 + 0.6 * lit) + 0.25 * lit)[None, None, :]
        c = body * (1 - 0.88 * grv)[:, :, None]
        c += (edge * (0.10 + 0.5 * under + 0.5 * lit))[:, :, None] * (lc * 0.7 + 0.1)[None, None, :]
        P = lambda x, y: (int(x * z + tx), int(y * z + ty))
        for sx in (-1, 1):                                                           # まゆの 岩角 と 鼻の穴
            cv2.line(c, P(W / 2 + sx * W * 0.10, H * 0.165), P(W / 2 + sx * W * 0.34, H * 0.122), (0.008, 0.016, 0.012), max(1, int(11 * k * z)), cv2.LINE_AA)
            cv2.line(c, P(W / 2 + sx * W * 0.10, H * 0.160), P(W / 2 + sx * W * 0.34, H * 0.117), tuple(float(v) for v in (lc * 0.10 + GREEN * 0.10 * (1 + 3 * lit))), max(1, int(2.5 * k * z)), cv2.LINE_AA)
            cv2.ellipse(c, P(W / 2 + sx * W * 0.055, H * 0.452), (int(9 * k * z), int(5 * k * z)), sx * 25, 0, 360, (0.55 * glow, 0.16 * glow, 0.02), -1, cv2.LINE_AA)
        for i in range(-3, 4):                                                       # 上あごの きば
            if i == 0: continue
            x = W / 2 + i * W * 0.045; yb = H * (0.492 - 0.012 * abs(i)); ln = H * (0.034 if abs(i) == 3 else 0.020)
            cv2.fillPoly(c, [np.array([P(x - W * 0.014, yb - H * 0.012), P(x + W * 0.014, yb - H * 0.012), P(x, yb + ln)], np.int32)], (0.95, 0.72, 0.45), cv2.LINE_AA)
            cv2.fillPoly(c, [np.array([P(x - W * 0.014, yb - H * 0.012), P(x - W * 0.002, yb - H * 0.012), P(x, yb + ln)], np.int32)], (0.50, 0.30, 0.18), cv2.LINE_AA)
        if op > 0.02:
            for ex, ey, rot in s.EY: s.eye(c, ex * z + tx, ey * z + ty, rot, min(1.25, op), glow, z)
        if jaw > 0.01:                                                               # 口が ひらく: のど(火)と 下あご(きばは 上向き)
            o = jaw * 0.115; hx = W / 2; Q = lambda x, y: P(hx + x * hx, y * H)
            up = [(-0.22, 0.468), (-0.12, 0.488), (0, 0.497), (0.12, 0.488), (0.22, 0.468)]
            lo = [(0.25, 0.47 + o * 0.5), (0.21, 0.535 + o), (0.10, 0.575 + o), (0, 0.585 + o), (-0.10, 0.575 + o), (-0.21, 0.535 + o), (-0.25, 0.47 + o * 0.5)]
            li = [(0.20, 0.475 + o * 0.6), (0.15, 0.515 + o), (0, 0.535 + o), (-0.15, 0.515 + o), (-0.20, 0.475 + o * 0.6)]
            mj = s.L0(); cv2.fillPoly(mj, [np.array([Q(*q) for q in up + lo], np.int32)], 1.0, cv2.LINE_AA)
            th = s.L0(); cv2.fillPoly(th, [np.array([Q(*q) for q in up + li], np.int32)], 1.0, cv2.LINE_AA)
            yy = np.clip((yn - 0.47) / (0.07 + o), 0, 1); thr = np.zeros_like(c); thr[:, :, 0] = 0.10 + 0.5 * yy; thr[:, :, 1] = 0.01 + 0.10 * yy ** 2
            c = c * (1 - th[:, :, None]) + thr * th[:, :, None]; m = np.maximum(m, mj)
            for i in range(-3, 4):                                                   # 下の きば
                x = i * 0.052 + (0.026 if i < 0 else -0.026) * 0; yb = 0.535 + o - 0.02 * (abs(i) / 3.0) ** 2 * 1.6 - (0.006 if i == 0 else 0); ln = 0.030 if abs(i) == 2 else 0.018
                if i == 0: continue
                cv2.fillPoly(c, [np.array([Q(x - 0.026, yb + 0.006), Q(x + 0.026, yb + 0.006), Q(x, yb - ln)], np.int32)], (0.95, 0.70, 0.42), cv2.LINE_AA)
                cv2.fillPoly(c, [np.array([Q(x + 0.004, yb + 0.006), Q(x + 0.026, yb + 0.006), Q(x, yb - ln)], np.int32)], (0.50, 0.28, 0.16), cv2.LINE_AA)
        return c, m * a

    def put_dragon(s, f, t, a, op, glow=1.0, tint=0.0):
        c, m = s.dragon(t, a, op, glow, tint); haze = s.fbm(0.0012, t * 5, -t * 22, 3, 0, 3)                                         # 手前の けむりで 半分 かくす
        m = m * np.clip(1.15 - haze * 0.75 * np.clip(1.4 - s.yn * 2.4, 0, 1), 0, 1); f = f * (1 - m[:, :, None]) + c * m[:, :, None]
        if op > 0.02 and glow > 0.05:
            for ex, ey, rot in s.EY: s.flare(f, (ex, ey), 13 * s.k * op, AMBER, 0.42 * op * glow * a, 1.3, 0.0, False)
        return f

    def hero(s, f, t, dx=0.0, dy=0.0, rim=None, rim_dir=(0.0, 1.0), rim_v=0.9, shade=0.80, **kw):
        img = s.pose(t, **kw); L = np.zeros((s.H, s.W), np.float32)                                                                 # 足もとの 影
        cv2.ellipse(L, (int(s.CX + dx), int(s.flr + 4 * s.k)), (int(s.W * 0.17), int(9 * s.k)), 0, 0, 360, 1.0, -1, cv2.LINE_AA); f *= (1 - 0.7 * cv2.GaussianBlur(L, (0, 0), 5 * s.k))[:, :, None]
        s.draw_lit(f, img, s.CX + dx, s.CY + dy, 0.42, rim=(LAVA + 0.25) if rim is None else rim, rim_dir=rim_dir, rim_v=rim_v, amb=(1.0, 0.72, 0.5), amb_v=0.22, aura=None, shade=shade)

    def heyes(s, f, v):
        if v <= 0.01: return
        sc = s._pl[2]
        for e in (s.eL, s.eR):
            p = s.at(*e); s.flare(f, p, 5.5 * sc * (1 + 0.8 * max(0.0, v - 1)) * 2, YEL, 0.5 * min(1.6, v), 0.5 + 1.2 * max(0.0, v - 1), 0.0, False)

    def blade(s, f, v, col=GREEN, th=1.6):
        if v <= 0.01: return
        a = s.at(*s.grd); b = s.at(*s.tip); s.glowline(f, [a, b], col, th, 0.55 * min(1.6, v), 0.9)

    def frame(s, t):
        if t < T_FIRE: f, sh, o = s.sc_cave(t)
        elif t < T_CUT: f, sh, o = s.sc_fire(t - T_FIRE)
        elif t < T_SL: f, sh, o = s.sc_cut(t - T_CUT)
        else: f, sh, o = s.sc_slash(t)
        f, sh = s.finale(f, sh, t, DUR, s.tip, (0.3, 1, 0.5), (1, 0.7, 0.15), -1)
        return s.film(f, t, sh * s.k, o.get("bloom", 0.6), o.get("thr", 0.80), o.get("tint", (1.0, 0.78, 0.55)), o.get("ca", 1.0), 0.04, o.get("vig", 0.6), 1.15, o.get("expo", 1.0))

    # 1) 闇の中で 目が ひらく
    def sc_cave(s, t):
        W, H, k = s.W, s.H, s.k
        f = s.cave(t); f = s.put_dragon(f, t, 0.95 * ease((t - 0.4) / 1.1), ease2((t - 1.0) / 0.45), 1.0)
        mg = clamp((t - 1.7) / 0.6)
        if mg > 0:                                                                   # のどの奥に 火が ともる
            mo = (W / 2, H * 0.478); msk = np.exp(-(((s.XX - mo[0]) / (W * 0.12)) ** 2 + ((s.YY - mo[1]) / (H * 0.022)) ** 2)).astype(np.float32)
            f += s.ramp(np.clip(s.flame(t, msk, -200.0, 0.004, 1.6) * mg + msk * mg * 0.6, 0, 1), s.FIRE); s.flare(f, mo, 10 * k * mg, LAVA + 0.2, 0.7 * mg, 1.6, 0.0, False)
        s.embers(f, t); f = s.rocks(f, t); s.hero(f, t)
        f = s.godrays(f, (W / 2, H * 1.02), 0.35, 0.5, 8, 0.86, 0.25)
        return f * clamp(t / 0.5), 0, dict(expo=0.95 + 0.1 * mg)

    # 2) 竜が 騎士ごしに こちらへ 火を はく。うろこの盾で 火が われる
    def sc_fire(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_FIRE
        f = s.cave(t, 0.0, 0.6); f = s.put_dragon(f, t, 0.95, 1.0, 1.35); f = s.rocks(f, t)
        pw = clamp(u / 0.25) * (1 - ease((u - 1.55) / 0.4))
        front = np.clip((H * 1.5 * ease(u / 0.45) + H * 0.44 - s.YY) / (H * 0.10), 0, 1) * np.clip((s.YY - H * 0.43) / (H * 0.05), 0, 1)
        cone = np.clip(1.25 - np.abs(s.XX - W / 2) / (W * 0.10 + (s.YY - H * 0.44) * 1.3 + 1), 0, 1)
        n = s.fbm(0.0020, 0.0, -t * 900.0, 4, 70 * k, 1) * 0.65 + s.fbm(0.0055, 20.0, -t * 1500.0, 3, 0, 2) * 0.5
        fi = np.clip((n - 0.22) * 2.1, 0, 1) * front * np.clip(cone * 2.2, 0, 1) * pw
        ay = s.CY - H * 0.245; wy = np.minimum(np.sqrt(np.clip(s.YY - ay, 0, None) / H) * W * 0.66, W * 0.36)
        dd = wy - np.abs(s.XX - s.CX); ins = np.clip(dd / (10 * k), 0, 1) * (s.YY > ay)
        fi = fi * (1 - ins) * (0.75 + 0.6 * np.clip(1 - np.abs(dd) / (60 * k), 0, 1))                                                # 盾の ふちで 火が あつく なる
        fc = s.ramp(np.clip(fi * 0.82, 0, 1), s.FIRE) * 1.05
        f = f * (1 - np.clip(fi * 1.7, 0, 1))[:, :, None] + fc
        s.hero(f, t, rim=GREEN * 0.6 + LAVA * 0.6, rim_dir=(0.0, -1.0), rim_v=0.5 + 0.5 * pw, shade=0.85 + 0.25 * pw, jit=0.5 * k * pw)
        rim = np.clip(1 - np.abs(dd) / (5 * k), 0, 1) * (s.YY > ay - 4 * k) * np.clip(1.4 - (s.YY - ay) / (H * 0.34), 0, 1)
        band = np.clip(1 - dd / (70 * k), 0, 1) * ins * np.clip(1.3 - (s.YY - ay) / (H * 0.32), 0, 1)
        hit = pw * (0.6 + 0.6 * n)
        s.glowmask(f, (rim * hit).astype(np.float32), GREEN * 0.8 + 0.2, 1.0, 0.9, 1.0, 9.0)
        f += (band * (s.grv * 0.9 + s.emb * 0.8) * hit)[:, :, None] * (GREEN * 0.9 + 0.1)[None, None, :]
        if pw > 0.05:                                                                # 盾に はじかれる 火花
            r = s.rnd[:300]; ph = (r[:, 0] + t * (1.2 + r[:, 1])) % 1.0; sy = ay + (ph ** 1.3) * H * 0.42; sw = np.minimum(np.sqrt(np.clip(sy - ay, 0, None) / H) * W * 0.66, W * 0.36)
            sx = s.CX + np.where(r[:, 2] > 0.5, 1, -1) * (sw + (4 + 46 * r[:, 3] * ph) * k)
            s.splat(f, sx, sy, (1 - ph) * pw * 1.6, (1.0, 0.85, 0.5), 1.0, 0.7)
        s.heyes(f, 0.9 + 0.4 * clamp((u - 1.6) / 0.3)); s.blade(f, 0.4 + 0.4 * pw, GREEN * 0.5 + 0.5)
        s.rushdust(f, t, (W / 2, H * 0.47), 900, 1.3, (1.0, 0.7, 0.3), 0.9 * pw, 4, 1.0)
        f += (np.clip(s.yn - 0.55, 0, 1) * 0.8 * pw * (0.6 + 0.6 * n))[:, :, None] * LAVA[None, None, :] * (1 - ins)[:, :, None]
        if u > 1.75:                                                                 # 火が やんで うろこが ひとなで 光る
            v = (u - 1.75) / 0.4; f += (np.exp(-((s.YY - (ay + H * 0.5 * v)) / (30 * k)) ** 2) * ins * (s.grv + s.emb) * (1 - v) * 1.4)[:, :, None] * GREEN[None, None, :]
        f = s.godrays(f, (W / 2, H * 0.47), 0.55 * pw, 0.7, 10, 0.88, 0.45)
        if u < 0.1: f += (1 - u / 0.1) * 0.5 * LAVA[None, None, :]
        return f, 2.4 * pw, dict(ca=1.0 + 1.5 * pw, bloom=0.7)

    # 3) 竜の ひとみに うつる 騎士
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        g = back(u / 0.3, 1.3); C = (W / 2, H * 0.38); hw, hh = W * 0.27 * g, H * 0.30
        f = s.iris2 ** 1.5 * (0.55 + 0.15 * ease(u / 0.5)) * (0.92 + 0.08 * math.sin(t * 5))
        pts = [(C[0] + hw * math.sin(math.pi * i / 24.0), C[1] - hh + 2 * hh * i / 24.0) for i in range(25)] + [(C[0] - hw * math.sin(math.pi * i / 24.0), C[1] + hh - 2 * hh * i / 24.0) for i in range(25)]
        m = s.L0(); cv2.fillPoly(m, [np.array(pts, np.int32)], 1.0, cv2.LINE_AA); m = cv2.GaussianBlur(m, (0, 0), 1.5 * k)
        win = np.zeros((H, W, 3), np.float32) + np.array([0.012, 0.012, 0.016], np.float32)
        win += s.nebula(t, SMOKE, 0.0012, (3.0, -40.0), 0.9) * np.clip(1.3 - s.yn * 1.5, 0, 1)[:, :, None]
        zf = 1.02 + 0.05 * ease(u / 1.9); sc = zf * H / s.ah; img = s.pose(t)
        s.draw_lit(win, img, C[0] - (s.fc[0] - s.aw / 2) * sc, C[1] - H * 0.02 - (s.fc[1] - s.ah / 2) * sc, zf, rim=AMBER + 0.2, rim_dir=(0.6, -0.8), rim_v=0.8, amb=(1.0, 0.8, 0.5), amb_v=0.18, shade=0.95)
        s.heyes(win, 0.8 + 0.5 * math.exp(-max(0.0, u - 0.45) * 5) * (u > 0.45))
        win *= (0.55 + 0.45 * sstep(1.0, 0.55, np.abs(s.XX - C[0]) / max(1.0, hw)))[:, :, None]                                         # ひとみの ふちは 暗く
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        s.glowline(f, pts, AMBER, 1.2, 0.5, 0.8, True)
        hl = np.exp(-(((s.XX - W * 0.30) / (W * 0.10)) ** 2 + ((s.YY - H * 0.20) / (H * 0.035)) ** 2)); f += hl[:, :, None] * 0.55                # 目の うつりこみ
        s.starfield(f, t, 500, (1.0, 0.8, 0.4), 6, 0.5, (0.0, -40.0))
        nn = len(s.ht); step = W * 0.20
        f *= (1 - 0.75 * sstep(0.70, 0.80, s.yn) * sstep(0.95, 0.88, s.yn))[:, :, None]
        for i, T in enumerate(s.ht):                                                 # 技名: 右から 斬りこんで 止まる
            a = (u - 0.55 - i * 0.12) / 0.2
            if a > 0: s.text(f, T, W / 2 + (i - (nn - 1) / 2) * step + (1 - ease(a)) * 60 * k, H * 0.835, clamp(a * 2), 1.0)
        for i, T in enumerate(s.vo):
            if u > 0.4 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.762, 1.0, 1.0)
        if u > 1.2: s.text(f, s.txt_en, W / 2, H * 0.835 + 62 * k, ease((u - 1.2) / 0.3), 1.0)
        if u < 0.08: f = f + (1 - u / 0.08) * 0.7
        return f, 0, dict(tint=(1.0, 0.8, 0.45), bloom=0.5, thr=0.9, vig=0.4)

    # 4) 一閃(合図)→ 竜が とびこえて こちらへ
    def sc_slash(s, t):
        W, H, k = s.W, s.H, s.k
        if T_BLK <= t < T_AFT:                                                       # まっ暗に 線 1本
            v = t - T_BLK; f = np.zeros((H, W, 3), np.float32); pr = ease(v / 0.09)
            a = s.P0 - s.dir * H * 0.9; b = a + s.dir * H * 1.8 * pr; s.glowline(f, [a, b], GREEN * 0.7 + 0.3, 1.4, 1.2, 1.3)
            r = s.rnd[:260]; q = r[:, 0] * pr; px = a[0] + s.dir[0] * H * 1.8 * q + s.nrm[0] * (r[:, 1] - 0.5) * 60 * k * v * 4; py = a[1] + s.dir[1] * H * 1.8 * q + s.nrm[1] * (r[:, 1] - 0.5) * 60 * k * v * 4
            s.splat(f, px, py, r[:, 2] * 1.2, GREEN * 0.6 + 0.4, 0.9, 0.5); s.flare(f, b if pr < 1 else s.P0, 16 * k * (1.2 - v * 2), GREEN * 0.5 + 0.5, 0.9, 1.8, 0.6)
            if v < 0.05: f += 0.6
            return f, 0, dict(tint=(0.6, 1.0, 0.75), ca=2.0, vig=0.3)
        f = s.cave(t)
        if t < T_BLK:                                                                # かまえ: 刃に 光が 走る
            v = t - T_SL; f = s.put_dragon(f, t, 0.95, 1.0, 1.2); f = s.rocks(f, t); s.embers(f, t); s.hero(f, t); s.heyes(f, 1.0)
            q = clamp(v / 0.3); a = s.at(*s.grd); b = s.at(*s.tip); p = (a[0] + (b[0] - a[0]) * q, a[1] + (b[1] - a[1]) * q)
            s.blade(f, 0.8, WHITE * 0.7 + GREEN * 0.3); s.flare(f, p, 12 * k * math.sin(q * 3.1416) + 1, WHITE, 0.9, 1.5, 0.4, False)
            if v < 0.1: f = f * (v / 0.1)
            return f, 0, {}
        v = t - T_AFT; g = clamp(v / 1.35); lunge = math.exp(-v * 4.5)                    # 合図のあと: 竜が 騎士を とびこえて こちらへ。口が ひらき、のどの 火に のまれる
        z = 1.0 + 4.6 * g ** 2.4; e = ease2((g - 0.25) / 0.55); px = W / 2; fy = H * (0.24 + 0.29 * e); sy = H * (0.24 + 0.20 * e) - H * 0.05 * math.sin(clamp(g / 0.35) * 3.1416)
        jaw = ease2((g - 0.18) / 0.5); tx, ty = px * (1 - z), sy - fy * z
        cc, mm = s.dragon(t, 1.0, 1.0 + 0.25 * g, 1.3 + 0.5 * g, 0.9 * g, 0.15 + 0.75 * g, z, tx, ty, jaw)
        front = g > 0.22                                                             # はじめは うしろ、とびこえたら 前
        if not front: f = f * (1 - mm[:, :, None]) + cc * mm[:, :, None]
        f = s.rocks(f, t, g); s.embers(f, t, g)
        s.hero(f, t, dx=-W * 0.07 * lunge, dy=H * 0.012 * lunge, rim=GREEN * 0.8 + 0.2, rim_dir=(0.0, -1.0), rim_v=0.4 + 0.6 * g, shade=0.8 - 0.3 * g, lean=-10 * k * lunge, squash=0.04 * lunge)
        s.heyes(f, 1.0); s.blade(f, 0.9 * lunge + 0.3, GREEN * 0.5 + 0.5)
        mo = (px, H * (0.497 + jaw * 0.115 * 0.55) * z + ty)
        if front:
            f = f * (1 - mm[:, :, None]) + cc * mm[:, :, None]
            rr = max(4.0, H * (0.012 + 0.055 * jaw) * z); msk = np.exp(-(((s.XX - mo[0]) / (rr * 1.9)) ** 2 + ((s.YY - mo[1]) / rr) ** 2)).astype(np.float32)
            fl = s.flame(t, msk, -520.0, 0.0034 / max(1.0, z * 0.45), 1.5); f += s.ramp(np.clip((fl * 0.95 + msk * 0.32) * (0.3 + g) * 1.0, 0, 1), s.FIRE) * (0.9 + 0.2 * g)     # のどの 火
        for ex, ey, rot in s.EY:
            ep = (ex * z + tx, ey * z + ty)
            if -W * 0.2 < ep[0] < W * 1.2 and -H * 0.1 < ep[1] < H * 1.1: s.flare(f, ep, min(60 * k, 13 * k * z), AMBER, 0.5, 1.3, 0.0, z < 2.5)
        mc = (px, float(min(H * 0.8, max(H * 0.2, mo[1]))))
        s.rushdust(f, t, mc, 1500, 0.9 + 1.6 * g, (1.0, 0.75, 0.35), 0.4 + 0.9 * g, 6, 1.0)
        f = s.godrays(f, mc, 0.3 + 0.5 * g, 0.75, 10, 0.88, 0.5)
        if v < 0.06: f = f + (1 - v / 0.06) * 0.5
        return f, 3 + 12 * g ** 2, dict(ca=1.0 + 3.0 * g, bloom=0.6 + 0.3 * g, tint=(1.0, 0.85, 0.6))


if __name__ == "__main__":
    r = Sieg()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
