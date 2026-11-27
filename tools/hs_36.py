"""ヤマタ(★5 八つ首の蛇姫) 専用の必殺技「八岐ノ顎」— 高品質版(hs_fx)・通し直し。敵は さいしょから 画面の こちら側
夜の沼(もや・ほたる・あしの影)に 浮かぶ → 髪の蛇の目が ひとつずつ 赤く灯り「一、二、三…八」。灯った蛇は 赤い光に なって 水に 落ち、波を ひいて こちらへ 泳いでくる(前ふり) →
八角形の窓のカットイン → こちらの まわり(画面の ふち)から、さっきの 八匹が 大蛇に なって 同じ「一…八」の 拍で 1匹ずつ 顔を 上げ、こちらを 見下ろす(かこまれた) →
蛇姫が 槍を ふる = 合図。八つの 顎が いっせいに ひらき、まわりから まん中の こちらへ とじてくる(すきまが なくなる)→ 毒の光 → 白 → 暗転
使い方: python3 hs_36.py 出力.mp4 / python3 hs_36.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old36", "/home/claude/tools/pre_fin/hs_36_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = 9.2
VEN, RED, MOS, GOLD = old.VEN, old.RED, old.MOS, old.GOLD
T_CNT, T_CUT, CNT = old.T_CNT, old.T_CUT, old.CNT
T_SUR, T_BITE, T_SHUT = 5.5, 7.55, 8.45
SUR = [T_SUR + 0.12 + v for v in np.cumsum([0, .26, .24, .21, .19, .16, .14, .11])]
ORDER = (1, 6, 0, 7, 2, 5, 3, 4)                                                     # 上の 左右 → 横 → 下(遠い方から 近い方へ)
SWAMP = [(0.05, 0.02, 0.10), (0.12, 0.05, 0.20), (0.30, 0.16, 0.42)]


class Yamata(FX, old.Yamata):
    def __init__(s, W=540):
        old.Yamata.__init__(s, W); s.fx_init(36); H, k = s.H, s.k
        s.rrim = np.clip(s.reed - cv2.warpAffine(s.reed, np.float32([[1, 0, 2 * k], [0, 1, 0]]), (W, H)), 0, 1)
        s.CEN = (W / 2, H * 0.47); s.anc = []
        for i in range(8):
            a = math.radians(-112.5 + i * 45.0); s.anc.append((math.cos(a), math.sin(a)))       # 0,1: 上  2,3: 右  4,5: 下  6,7: 左(時計まわり)

    # ---- 沼 ----
    def swamp(s, t, lit=0.0):
        H, W, k = s.H, s.W, s.k; hy = s.hz_y
        f = np.zeros((H, W, 3), np.float32) + np.array([0.012, 0.008, 0.024], np.float32)
        f += s.nebula(t, SWAMP, 0.0008, (6.0, 1.0), 0.7 + 0.3 * lit) * np.clip(1.15 - s.yn * 0.9, 0, 1)[:, :, None]
        wm = sstep(hy - 2 * k, hy + 2 * k, s.YY); dep = np.clip((s.YY - hy) / (H - hy), 0, 1)
        rip = s.fbm(0.004, t * 9, 0, 3, 0, 1); rip = cv2.remap(s._tiles[1], ((s.XX * 0.25 + t * 6) % 256).astype(np.float32), ((dep * 900 / (0.3 + dep) + t * 10) % 256).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        wat = np.array([0.010, 0.030, 0.028], np.float32)[None, None, :] * (0.5 + 1.2 * (1 - dep))[:, :, None] + (sstep(0.62, 0.85, rip) * (1 - dep) ** 1.5 * 0.22)[:, :, None] * (VEN * 0.6 + MOS * 0.3)[None, None, :]
        f = f * (1 - wm[:, :, None]) + wat * wm[:, :, None]
        mist = s.fbm(0.0016, t * 16, t * 2, 3, 40 * k, 2) * np.exp(-((s.YY - hy) / (H * 0.09)) ** 2)
        f += (mist * (0.30 + 0.25 * lit))[:, :, None] * (VEN * 0.55 + 0.30)[None, None, :]
        for r in s.bub:                                                              # あぶくの 輪
            v = (t * (0.35 + 0.3 * r[2]) + r[3]) % 1.0; x = r[0] * W; y = hy + (0.06 + 0.3 * r[1]) * H
            pts = [(x + math.cos(q * 0.2618) * 30 * k * v, y + math.sin(q * 0.2618) * 8 * k * v) for q in range(24)]
            s.glowline(f, pts, MOS * 0.7 + 0.2, 0.8, 0.35 * (1 - v), 0.5, True)
        s.starfield(f, t, 260, MOS * 0.8 + 0.2, 4, 1.3, (5.0, -9.0)); s.starfield(f, t + 5, 700, VEN * 0.6 + 0.4, 6, 0.5, (8.0, -4.0))    # ほたる・胞子
        return f

    def reeds(s, f):
        f = f * (1 - s.reed[:, :, None] * 0.95); f += (s.rrim * 0.22)[:, :, None] * (VEN * 0.6 + 0.3)[None, None, :]; return f

    def hero(s, f, t, dx=0.0, hf=None, cy=None, shade=0.9, **kw):
        hf = s.HF if hf is None else hf; cy = s.CY if cy is None else cy; img = s.pose(t, **kw)
        s.draw_lit(f, img, s.CX + dx, cy, hf, rim=VEN * 0.7 + 0.3, rim_dir=(0.0, -1.0), rim_v=0.9, amb=(0.8, 0.7, 1.0), amb_v=0.16, shade=shade)
        fy = cy + hf * s.H / 2; L = s.L0(); cv2.ellipse(L, (int(s.CX + dx), int(max(fy + 30 * s.k, s.hz_y + 16 * s.k))), (int(hf * s.H * 0.26), int(5 * s.k)), 0, 0, 360, 1.0, -1, cv2.LINE_AA)
        f += (cv2.GaussianBlur(L, (0, 0), 5 * s.k) * 0.18)[:, :, None] * VEN[None, None, :]                                           # 水面の てり

    def eyes(s, f, n, v=1.0, own=0.0):
        sc = s._pl[2]
        for i in range(min(8, n)):
            p = s.at(*s.hd[i]); s.flare(f, p, 3.2 * sc * 2, RED, 0.55 * v, 0.8, 0.0, False)
        if own > 0.01:
            for e in (s.eL, s.eR): p = s.at(*e); s.flare(f, p, 4.5 * sc * 2, RED, 0.5 * min(1.5, own), 0.6 + 0.8 * max(0.0, own - 1), 0.0, False)

    # ---- 大蛇 ----
    def scl(s, X, Y, R):
        rs = R * 0.62; cs = R * 1.84; D = np.full(X.shape, 2.0, np.float32); V = np.zeros_like(D); Hh = np.zeros_like(D); j0 = np.floor(Y / rs)
        for dj in (0, 1, 2):
            j = j0 - dj; yc = j * rs; off = np.where(np.mod(j, 2) == 1, R * 0.92, 0.0) - R
            i = np.round((X - off) / cs); xc = i * cs + off; d = np.sqrt((X - xc) ** 2 + (Y - yc) ** 2) / R; ok = d < 1.0
            D = np.where(ok, d, D); V = np.where(ok, (Y - yc) / R, V); Hh = np.where(ok, np.mod(np.sin(i * 12.9898 + j * 78.233) * 43758.5453, 1.0), Hh)
        return D, V, Hh.astype(np.float32)

    def neck(s, f, a, b, w, lit=0.0, bend=0.0):
        """首: a(画面の外)→ b(頭)。まるい 筒に 見える かげ と ふち光"""
        k = s.k; n = (-(b[1] - a[1]), b[0] - a[0]); ln = math.hypot(*n) + 1e-3; P = []
        for q in range(13):
            u = q / 12.0; o = math.sin(u * 3.1416) * bend; P.append((a[0] + (b[0] - a[0]) * u + n[0] / ln * o, a[1] + (b[1] - a[1]) * u + n[1] / ln * o))
        L = s.L0(); cv2.polylines(L, [np.array(P, np.int32)], False, 1.0, max(2, int(w)), cv2.LINE_AA); C = s.L0(); cv2.polylines(C, [np.array(P, np.int32)], False, 1.0, max(1, int(w * 0.22)), cv2.LINE_AA)
        m = cv2.GaussianBlur(L, (0, 0), max(0.8, w * 0.02)); core = np.clip(cv2.GaussianBlur(C, (0, 0), max(1.0, w * 0.20)) * 1.9, 0, 1)
        D, V, Hh = s.scl(s.XX, s.YY, max(4.0, w * 0.17)); grv = sstep(0.86, 0.97, D)
        body = np.array([0.035, 0.105, 0.06], np.float32)[None, None, :] * ((0.25 + 1.3 * core) * (0.75 + 0.5 * Hh) * (1 - 0.8 * grv) * (1 + 1.5 * lit))[:, :, None]
        body += ((1 - core) ** 3 * 0.22)[:, :, None] * (VEN * 0.8)[None, None, :]
        f[:] = f * (1 - m[:, :, None]) + body * m[:, :, None]

    def shead(s, f, c, S, ang=0.0, op=0.0, lit=0.0, eye=1.0):
        """こちらを 向いた 大蛇の顔(式で 描く: どの 大きさでも くっきり)。op: 口の ひらき 0..1"""
        W, H, k = s.W, s.H, s.k; x0 = int(max(0, c[0] - S * 1.25)); x1 = int(min(W, c[0] + S * 1.25)); y0 = int(max(0, c[1] - S * 1.15)); y1 = int(min(H, c[1] + S * 1.6))
        if x1 - x0 < 4 or y1 - y0 < 4: return
        ca, sa = math.cos(ang), math.sin(ang); dx = s.XX[y0:y1, x0:x1] - c[0]; dy = s.YY[y0:y1, x0:x1] - c[1]; x = (dx * ca + dy * sa) / S; y = (-dx * sa + dy * ca) / S; aa = min(0.2, 2.5 / S)
        top = np.hypot(x / (0.72 - 0.40 * np.clip(y + 0.15, 0, 1) - 0.30 * np.clip(-y - 0.15, 0, 1)), y / 0.80); mh = sstep(1.0, 1.0 - aa, top)
        jy = 0.50 + 0.30 * op; jr = np.hypot(x / 0.46, (y - jy) / (0.24 + 0.26 * op)); mj = sstep(1.0, 1.0 - aa, jr)
        my = 0.52 + 0.26 * op; mry = 0.02 + 0.36 * op; mo = np.hypot(x / 0.37, (y - my) / mry); mm = sstep(1.0, 1.0 - aa * 2, mo) * (1.0 if op > 0.04 else 0.0)
        m = np.maximum(mh, mj); rr = np.minimum(top, jr); dome = np.sqrt(np.clip(1 - rr * rr, 0, 1))
        D, V, Hh = s.scl(x, y, 0.085); grv = sstep(0.88 - aa, 0.97, D); rim = sstep(0.60, 0.86, D) * sstep(0.95, 0.87, D) * np.clip(V * 1.4, 0, 1)
        mark = np.exp(-(x / 0.13) ** 2) * (0.5 + 0.5 * np.cos(y * 15.0)) * (y < 0.35)
        body = np.array([0.045, 0.135, 0.075], np.float32)[None, None, :] * ((0.30 + 1.0 * dome) * (0.72 + 0.56 * Hh) * (1 - 0.55 * mark) * (1 + 1.6 * lit))[:, :, None]
        body += (np.clip(-y + 0.1, 0, 1) * dome * 0.30)[:, :, None] * VEN[None, None, :]
        body += (rim * (0.35 + 0.8 * lit))[:, :, None] * (MOS * 0.45 + VEN * 0.25)[None, None, :]
        cc = body * (1 - 0.85 * grv)[:, :, None] + (sstep(0.72, 1.0, rr) * 0.55)[:, :, None] * VEN[None, None, :]
        ax = np.abs(x); brow = np.exp(-((y - (-0.47 + 0.38 * (0.5 - ax))) / 0.04) ** 2) * (ax > 0.05) * (ax < 0.58); cc *= (1 - 0.85 * brow)[:, :, None]
        cc *= (1 - 0.8 * np.exp(-(((ax - 0.09) / 0.035) ** 2 + ((y - 0.44) / 0.025) ** 2)))[:, :, None]                               # 鼻の あな
        for sg in (-1, 1):                                                           # 目(つり上がった 赤)
            u = (x - sg * 0.30) / 0.17; v = (y + 0.20 + 0.38 * sg * (x - sg * 0.30)) / 0.085; r2 = u * u + v * v; me = sstep(1.0, 0.80, r2)
            val = np.clip(1.15 - np.sqrt(r2) * 0.8, 0, 1) * eye; ec = RED[None, None, :] * (val * 1.5)[:, :, None] + np.array([1.0, 0.75, 0.3], np.float32)[None, None, :] * (val ** 3)[:, :, None]
            pup = sstep(1.0, 0.6, (u / 0.17) ** 2 + (v / 0.95) ** 2); ec = ec * (1 - pup)[:, :, None] + (np.exp(-(((u + 0.3) / 0.12) ** 2 + ((v + 0.4) / 0.25) ** 2)) * 0.8 * eye)[:, :, None]
            cc = cc * (1 - me[:, :, None]) + ec * me[:, :, None]
        if op > 0.04:                                                                # 口の中
            g = np.clip((y - (my - mry)) / (2 * mry), 0, 1); inn = np.array([0.36, 0.03, 0.12], np.float32)[None, None, :] * (1 - g)[:, :, None] ** 1.5 + np.array([0.03, 0.0, 0.05], np.float32)[None, None, :]
            inn += (np.exp(-mo * mo * 2.5) * (0.15 + 0.6 * lit))[:, :, None] * VEN[None, None, :]
            cc = cc * (1 - mm[:, :, None]) + inn * mm[:, :, None]
        cc = np.ascontiguousarray(cc.astype(np.float32))
        def P(px, py): return (int(c[0] + (px * ca - py * sa) * S - x0), int(c[1] + (px * sa + py * ca) * S - y0))
        if op > 0.04:
            ty = my - mry * 0.82; fl = 0.05 + 0.40 * op
            for sg in (-1, 1):                                                       # 毒きば
                cv2.fillPoly(cc, [np.array([P(sg * 0.17, ty - 0.02), P(sg * 0.28, ty - 0.02), P(sg * 0.235, ty + fl)], np.int32)], (0.96, 0.92, 0.80), cv2.LINE_AA)
                cv2.fillPoly(cc, [np.array([P(sg * 0.235, ty - 0.02), P(sg * 0.28, ty - 0.02), P(sg * 0.235, ty + fl)], np.int32)], (0.55, 0.45, 0.50), cv2.LINE_AA)
            by = my + mry * 0.78
            for q in (-0.20, -0.10, 0.10, 0.20): cv2.fillPoly(cc, [np.array([P(q - 0.035, by + 0.02), P(q + 0.035, by + 0.02), P(q, by - 0.10 * op - 0.02)], np.int32)], (0.85, 0.80, 0.72), cv2.LINE_AA)
        f[y0:y1, x0:x1] = f[y0:y1, x0:x1] * (1 - m[:, :, None]) + cc * m[:, :, None]
        if eye > 0.05:
            for sg in (-1, 1):
                e = (c[0] + (sg * 0.30 * ca + 0.20 * sa) * S, c[1] + (sg * 0.30 * sa - 0.20 * ca) * S)
                if 0 <= e[0] < W and 0 <= e[1] < H: s.flare(f, e, min(26 * k, 0.085 * S), RED, 0.45 * eye, 1.2, 0.0, False)

    def frame(s, t):
        s._t = t; o = {}
        if t < T_CUT: f, sh = s.sc_count(t)
        elif t < T_SUR: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.4, thr=0.9)
        else: f, sh = s.sc_sur(t)
        f, sh = s.finale(f, sh, t, DUR, s.stp, (0.74, 0.32, 1), (0.35, 0.85, 0.4), -1)
        return s.film(f, t, sh * s.k, 0.6, o.get("thr", 0.80), (0.85, 0.7, 1.0), 1.0 + 0.2 * sh, 0.04, o.get("vig", 0.6), 1.15, 1.08)

    # 1) 沼に 浮かぶ。目が ひとつずつ 灯り、灯った蛇は 赤い光に なって 水へ → こちらへ 泳ぐ
    def sc_count(s, t):
        W, H, k = s.W, s.H, s.k
        f = s.swamp(t); s.hero(f, t)
        n = sum(1 for c in CNT if t >= c)
        s.eyes(f, n, 1.0, clamp((t - 1.0) / 0.5) * 0.8)
        hub = s.at(*s.hub)
        for i in range(n):
            p = s.at(*s.hd[i]); d = math.hypot(p[0] - hub[0], p[1] - hub[1]) + 1e-3; v = t - CNT[i]
            q = (p[0] + (p[0] - hub[0]) / d * 34 * k, p[1] + (p[1] - hub[1]) / d * 34 * k - 4 * k)
            s.text(f, s.num[i], min(max(q[0], 30 * k), W - 30 * k), max(q[1], 28 * k), clamp(v / 0.08), 1.0)
            if v < 0.15: pts = [(p[0] + math.cos(a_ * 0.2618) * (8 + 60 * v / 0.15) * k, p[1] + math.sin(a_ * 0.2618) * (8 + 60 * v / 0.15) * k) for a_ in range(24)]; s.glowline(f, pts, RED, 1.0, 1 - v / 0.15, 0.7, True)
            wx = W * (0.08 + 0.84 * ((i * 5) % 8) / 7.0); y0 = s.hz_y + 10 * k                                                        # 赤い光が 水に 落ち、波を ひいて 手前へ
            if v < 0.3:
                g = v / 0.3; dp = (p[0] + (wx - p[0]) * g, p[1] + (y0 - p[1]) * g * g); s.flare(f, dp, 4 * k, RED, 0.8, 0.5, 0.0, False)
            else:
                g = clamp((v - 0.3) / 1.6); yy = y0 + (H * 1.02 - y0) * g ** 1.6; wd = (8 + 60 * g) * k
                s.glowline(f, [(wx - wd, yy + wd * 0.5), (wx, yy), (wx + wd, yy + wd * 0.5)], VEN * 0.6 + 0.4, 0.8 + g, 0.5 * (1 - g * 0.5), 0.6)
                s.flare(f, (wx, yy), (3 + 5 * g) * k, RED, 0.7, 0.6, 0.0, False)
        f = s.reeds(f)
        return f * clamp(t / 0.4), 0

    # 2) 八角形の窓
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        C = (W / 2, H * 0.38)
        D, V, Hh = s.scl(s.XX + t * 14, s.YY - t * 9, 30 * k)
        f = (np.array([0.020, 0.045, 0.030], np.float32)[None, None, :] * ((0.4 + Hh) * (1 - 0.8 * sstep(0.86, 0.97, D)))[:, :, None]).astype(np.float32) + s.nebula(t, SWAMP, 0.001, (8.0, 3.0), 0.5)
        f += (sstep(0.62, 0.86, D) * sstep(0.95, 0.87, D) * np.clip(V, 0, 1) * 0.10)[:, :, None] * VEN[None, None, :]
        g = back(u / 0.3, 1.3); R = W * 0.44 * g
        pts = [(C[0] + math.cos(0.3927 + i * 0.7854) * R, C[1] + math.sin(0.3927 + i * 0.7854) * R) for i in range(8)]
        m = s.L0(); cv2.fillPoly(m, [np.array(pts, np.int32)], 1.0, cv2.LINE_AA)
        f *= (1 - 0.6 * cv2.GaussianBlur(m, (0, 0), 18 * k))[:, :, None]
        win = np.zeros((H, W, 3), np.float32) + np.array([0.07, 0.03, 0.12], np.float32) + s.nebula(t, [(0.25, 0.08, 0.40), (0.08, 0.25, 0.15), (0.8, 0.5, 1.0)], 0.0014, (10.0, -6.0), 0.9)
        s.starfield(win, t, 400, MOS * 0.7 + 0.3, 7, 0.8, (6.0, -20.0))
        zf = 0.98 + 0.05 * ease(u / 1.9); sc = zf * H / s.ah; img = s.pose(t)
        s.draw_lit(win, img, C[0] - (s.fc[0] - s.aw / 2) * sc, C[1] + H * 0.035 - (s.fc[1] - s.ah / 2) * sc, zf, rim=VEN * 0.6 + 0.4, rim_dir=(0.5, -0.85), rim_v=0.8, amb=(0.85, 0.75, 1.0), amb_v=0.12, shade=0.97)
        s.eyes(win, 8, 1.0, 0.9 + 1.0 * math.exp(-max(0.0, u - 0.45) * 5) * (u > 0.45))
        f = f * (1 - m[:, :, None]) + win * m[:, :, None]
        s.glowline(f, pts, GOLD, 2.4, 0.75, 0.8, True); s.glowline(f, [(C[0] + (x - C[0]) * 1.05, C[1] + (y - C[1]) * 1.05) for x, y in pts], GOLD * 0.8, 1.0, 0.4, 0.4, True)
        for i, p in enumerate(pts):                                                  # 八つの かどに 赤い玉が 順に 灯る
            if u > 0.3 + i * 0.07: s.flare(f, p, 7 * k, RED, 0.9, 0.7, 0.0, False)
        f *= (1 - 0.7 * sstep(0.70, 0.79, s.yn) * sstep(0.96, 0.90, s.yn))[:, :, None]
        n = len(s.ht); step = W * 0.20
        for i, T in enumerate(s.ht):                                                 # 技名: かみつくように 上下から とじる
            a = (u - 0.6 - i * 0.12) / 0.18
            if a > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.835, clamp(a * 2), ease(a))
        for i, T in enumerate(s.vo):
            if u > 0.45 + i * 0.1: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.755, 1.0, 1.0)
        if u > 1.3: s.text(f, s.txt_en, W / 2, H * 0.835 + 62 * k, ease((u - 1.3) / 0.3), 1.0)
        if u < 0.08: f = f + (1 - u / 0.08) * 0.7
        return f, 0

    # 3) かこまれる → 合図 → 八つの 顎が まわりから とじてくる
    def sc_sur(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_SUR; bt = t - T_BITE; g = clamp(bt / (T_SHUT - T_BITE)); gg = g ** 1.8
        f = s.swamp(t, 0.4 + 0.6 * g) * (1 - 0.55 * gg)
        sw = math.exp(-max(0.0, bt) * 5) if bt > 0 else 0.0                              # 槍を ふる(合図)
        s.hero(f, t, hf=0.30, cy=H * 0.50, shade=0.9 - 0.4 * gg, lean=(-14 * k * sw if bt > 0 else 6 * k * clamp((t - (T_BITE - 0.35)) / 0.35)), squash=0.04 * sw)
        s.eyes(f, 8, 0.8, 1.0 + (1.2 * sw if bt > 0 else 0)); f = s.reeds(f)
        if 0 < bt < 0.1: f = f + (1 - bt / 0.1) * 0.5 * VEN[None, None, :]
        if g > 0:                                                                    # まん中に 毒の光が たまる
            s.flare(f, s.CEN, (6 + 60 * gg) * k, VEN * 0.6 + 0.4, 0.4 + 0.8 * g, 1.2, t * 0.4, False); s.rushdust(f, t, s.CEN, 1200, 1.0 + 1.5 * g, VEN * 0.6 + MOS * 0.3 + 0.1, 0.3 + 0.9 * g, 5, 1.0)
        for n_, i in enumerate(ORDER):
            v = t - SUR[n_]
            if v <= 0: continue
            ax, ay = s.anc[i]; up = back(v / 0.22, 1.5); ph = i * 1.7
            rho = 1.0 - 0.72 * gg; rx = W * 0.40 * rho; ry = H * 0.37 * rho; S = H * (0.085 + 0.012 * (ay + 1)) * (1 + 3.6 * gg)
            hx = s.CEN[0] + ax * (rx + (1 - up) * W * 0.5) + math.sin(t * 2.1 + ph) * 7 * k * (1 - g); hy = s.CEN[1] + ay * (ry + (1 - up) * H * 0.4) + math.cos(t * 1.7 + ph) * 6 * k * (1 - g)
            op = 0.10 * clamp((t - (T_BITE - 0.5)) / 0.5) + 0.10 * math.exp(-v * 9) + (0.9 * ease(bt / 0.25) if bt > 0 else 0.0)
            tilt = -ax * 0.45 * (1 if ay <= 0.2 else -1)
            a0 = (s.CEN[0] + ax * W * 1.25, s.CEN[1] + ay * H * 1.0 + H * 0.25)
            s.neck(f, a0, (hx - ax * S * 0.3, hy - ay * S * 0.3 + S * 0.2), S * 0.95, 0.3 * g, S * 0.8 * (1 if i % 2 else -1))
            s.shead(f, (hx, hy), S, tilt, min(1.0, op), 0.15 + 0.6 * g, 1.0)
            if v < 0.5 and bt < 0:                                                   # 数字(同じ「一…八」)
                s.text(f, s.num[n_], min(max(hx - ax * S * 1.5, 26 * k), W - 26 * k), min(max(hy - ay * S * 1.5, 26 * k), H - 26 * k), clamp(v / 0.08) * clamp((0.5 - v) / 0.12), 1.0)
        if g > 0.6: f = f + ((g - 0.6) / 0.4) ** 2 * 0.9 * (VEN * 0.5 + 0.5)[None, None, :]
        if u < 0.1: f = f * (u / 0.1)
        return f, (2.5 * math.exp(-min([t - c for c in SUR if t >= c] + [9]) * 12) if bt < 0 else 4 + 12 * gg)

    # ---- この子だけの曲: ニ調の 陰旋法。琴の つまびき・しめ太鼓・蛇の シュッ。数える拍が 後半で もう一度(かこむ ときに 1オクターブ上で)、合図のあと 連打が つまり どら ----
    def audio(s, path):
        a = Snd(DUR, 36); D = 146.83; sc = [0, 1, 5, 7, 8]
        def fq(d, o=0): return D * 2 ** ((sc[d % 5] + 12 * (d // 5 + o)) / 12.0)
        def pluck(d, t0, g=0.12, o=1): a.note(fq(d, o), t0, 0.16, g, "fm", 0.001); a.note(fq(d, o) * 2.01, t0, 0.05, g * 0.3, "sq", 0.001); a.hat(t0, g * 0.5, 0.008)
        def taiko(t0, g=0.8, f0=58): a.boom(t0, f0, g, 0.22)
        def hiss(t0, g=0.14, ln=0.12): a.a += (a.nz - a.lp(a.nz, 0.75)) * a.env(t0, 0.01, ln) * g
        def flute(d, t0, ln, g=0.07, o=2):
            ph = 2 * np.pi * fq(d, o) * (a.t - t0) + 0.35 * np.sin(2 * np.pi * 5 * (a.t - t0)); e = a.env(t0, 0.08, ln)
            a.a += (np.sin(ph) + 0.25 * np.sin(2 * ph)) * e * g + (a.nz - a.lp(a.nz, 0.6)) * e * g * 0.12
        a.a += a.tone(36.7) * 0.07 * a.gate(0, T_BITE, 0.4) + a.lp(a.nz, 0.03) * 0.5 * a.gate(0, T_CUT, 0.4)
        for i in range(5): a.note(500 + 130 * ((i * 3) % 5), 0.3 + i * 0.31, 0.05, 0.03, "sine")                                   # あぶく
        flute(2, 0.5, 0.5, 0.05, 1); taiko(1.0, 0.5)
        for i, c in enumerate(CNT): hiss(c); pluck(i, c, 0.12); taiko(c, 0.25 + 0.05 * i, 70); a.note(700 - i * 30, c + 0.3, 0.06, 0.03, "sine")   # 数える + 水に 落ちる
        taiko(CNT[-1] + 0.14, 1.0, 50)
        a.boom(T_CUT, 55, 0.9, 0.3)
        for i, (d, dt) in enumerate(((4, 0.2), (3, 0.4), (2, 0.6), (1, 0.75), (0, 0.9), (2, 1.2), (1, 1.35), (0, 1.5))): pluck(d + 5, T_CUT + dt, 0.13)
        for dt in (0.2, 0.6, 0.9, 1.2, 1.5): taiko(T_CUT + dt, 0.5)
        for i in range(8): a.note(1900 + i * 60, T_CUT + 0.3 + i * 0.07, 0.04, 0.03, "sine")
        a.a += a.lp(a.nz, 0.04) * 0.6 * a.gate(T_SUR, T_BITE, 0.2)                                                                # かこまれる: 水の うなり
        for i, c in enumerate(SUR): hiss(c, 0.20, 0.22); pluck(i, c, 0.14, 2); pluck(i, c, 0.08, 1); taiko(c, 0.45 + 0.06 * i, 62); a.whoosh(c - 0.12, c, 0.18, 0.3)
        flute(4, SUR[-1] + 0.12, 0.35, 0.06); tk = SUR[-1] + 0.12
        while tk < T_BITE - 0.04: taiko(tk, 0.3, 80); tk += 0.07                                                                  # ため
        a.hat(T_BITE, 0.6, 0.05); a.note(1300, T_BITE, 0.08, 0.12, "sq", 0.001); taiko(T_BITE, 1.1, 48)                            # 合図
        for i in range(8): hiss(T_BITE + 0.02 + i * 0.02, 0.10, 0.5)
        tk = T_BITE + 0.08; i = 0                                                                                                 # とじてくる: 連打が つまる + 琴が かけ上がる
        while tk < T_SHUT - 0.03:
            taiko(tk, 0.4 + 0.6 * (tk - T_BITE) / (T_SHUT - T_BITE), 64 if i % 2 else 52); pluck(i % 10, tk, 0.07 + 0.006 * i, 1 + i // 10); tk += max(0.045, 0.17 - i * 0.012); i += 1
        a.rise(T_BITE, T_SHUT, 90, 700, 0.13); a.a += (a.nz - a.lp(a.nz, 0.7)) * a.gate(T_BITE, T_SHUT, 0.1) * 0.07 * (1 + np.sin(a.t * 40))
        a.boom(T_SHUT, 38, 1.7, 0.6); a.a += a.lp(a.nz, 0.3) * a.env(T_SHUT, 0.004, 0.35) * 0.6
        for fr, g_ in ((98, 0.10), (147.5, 0.07), (233, 0.06), (311, 0.05), (467, 0.03)): a.note(fr, T_SHUT, 1.2, g_, "sine", 0.002)   # どら
        s.fin_snd(a, DUR, 36); a.save(path)


if __name__ == "__main__":
    r = Yamata()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
