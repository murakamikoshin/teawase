"""スコルピ(★4 蠍座の暗殺者) 専用の必殺技「蠍ノ一刺シ」— 高品質版(hs_fx)。赤い 月の 砂漠、頭上から さそりの 尾が 弧を えがいて 降りてきて、毒針の 一しずくが 視界を そめる
赤い 月の 夜の 砂丘。空に さそり座(赤い 心臓の 星)が 灯る(前ふり) → 目の カットイン+技名 →
姿が 砂に しずんで 消える(しずけさ) → 画面の 上から 節の ある 大きな 尾が 弧を えがいて 降りて くる →
毒針の 先が 目の前で 止まり、紫の しずくが ふくらんで 落ちる → むらさきの 波紋が ひろがって 視界が そまる → 白 → 暗転
使い方: python3 hs_406.py 出力.mp4 / python3 hs_406.py still 出力.png 時刻,時刻"""
import sys, math
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep

DUR = 10.3
T_CUT, T_SINK, T_TAIL, T_STING, T_DROP, T_HIT = 2.7, 4.7, 5.9, 7.3, 8.25, 9.6
PUR = np.array([0.66, 0.30, 1.0], np.float32); VEN = np.array([0.85, 0.35, 1.0], np.float32); REDM = np.array([1.0, 0.30, 0.22], np.float32); NIGHT = np.array([0.04, 0.02, 0.08], np.float32); SAND = np.array([0.30, 0.17, 0.20], np.float32)
EYE = (305, 345); DAG = (20, 595)
SCO = [(0.30, -0.55), (0.18, -0.40), (0.10, -0.22), (0.00, -0.05), (-0.06, 0.14), (-0.02, 0.33), (0.12, 0.48), (0.30, 0.52), (0.45, 0.42), (0.50, 0.26), (0.42, 0.16)]      # さそり座(上が 頭、下で 尾が まがる)


class Scorpi(FX, Kit):
    def __init__(s, W=540):
        Kit.__init__(s, 406, (0.7, 0.35, 1.0), "蠍ノ一刺シ", "Scorpion Sting", W); s.fx_init(406)
        s._vt = [s.text_img(c, hz.SERIF, int(66 * s.k), 3) for c in "蠍ノ一刺シ"]
        s._vo = [s.text_img(c, hz.SERIF, int(26 * s.k), 1) for c in "奥義"]
        s.xn = s.XX / s.W; s.yn = s.YY / s.H; s.hy = 0.66
        s._dn = [(0.60 + 0.06 * i + 0.035 * np.sin(s.xn * (3.0 + 1.7 * i) + i * 2.1) + 0.012 * np.sin(s.xn * (9.0 + 3 * i) + i)).astype(np.float32) for i in range(3)]

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    # ---- 赤い 月の 砂丘 ----
    def dunes(s, t, lum=1.0, stars=0.0):
        W, H, k = s.W, s.H, s.k
        f = NIGHT[None, None, :] * (0.7 + 1.5 * (s.yn / 0.62).clip(0, 1) ** 1.5)[:, :, None] + np.array([0.30, 0.07, 0.10], np.float32)[None, None, :] * np.exp(-((s.yn - 0.58) / 0.12) ** 2)[:, :, None]
        f = f + s.nebula(t, [(0.08, 0.03, 0.14), (0.16, 0.04, 0.12), (0.4, 0.2, 0.4)], 0.0008, (3.0, 0.6), 0.5, 0)
        s.starfield(f, t, 800, (1.0, 0.9, 0.9), 2, 0.8, (0.0, 0.0))
        d = np.hypot(s.xn - 0.76, (s.yn - 0.20) * H / W) / 0.085; mn = s.tex(2, s.XX * 1.1, s.YY * 1.1)                    # 赤い 月
        f += (sstep(1.0, 0.95, d) * (0.9 + 0.3 * mn))[:, :, None] * REDM[None, None, :] * 1.1 + (np.exp(-np.clip(d - 1, 0, None) / 1.1) * 0.35)[:, :, None] * REDM[None, None, :]
        if stars > 0:
            C = (W * 0.34, H * 0.30); S = H * 0.30; P = [(C[0] + x * S, C[1] + y * S) for x, y in SCO]; n = stars * (len(P) + 2)
            for i in range(len(P) - 1):
                q = clamp(n - i - 1.5)
                if q > 0: s.glowline(f, [P[i], (P[i][0] + (P[i + 1][0] - P[i][0]) * q, P[i][1] + (P[i + 1][1] - P[i][1]) * q)], (1.0, 0.8, 0.9), 0.8, 0.35, 0.6)
            for i, p in enumerate(P):
                q = n - i
                if q > 0: s.flare(f, p, (4 + (5 if i == 3 else 0) + 6 * math.exp(-q * 5)) * k, REDM if i == 3 else (1.0, 0.85, 0.95), 0.9 * clamp(q * 4), 0.3, 0.0, False)
        for i, r in enumerate(s._dn):                                                # 砂丘(3かさね: 月の がわの 面が 明るい)
            m = sstep(r - 0.003, r + 0.003, s.yn)[:, :, None]; dr = np.clip((s.yn - r) / 0.25, 0, 1)
            slope = np.gradient(r, axis=1) * W; lit = np.clip(0.60 - slope * 3.2, 0.25, 1.1)
            rip = s.tex(1, s.XX * 0.4 + s.YY * 0.6, s.YY * (2.0 + i) ) * 0.6 + s.tex(3, s.XX * 1.3, s.YY * 1.3) * 0.4
            lit = 0.62 + (lit - 0.62) * np.exp(-dr / 0.22); col = (SAND * (0.55 + 0.25 * i))[None, None, :] * (lit * (0.6 + 0.6 * rip) * (1.1 - 0.5 * dr))[:, :, None] + (REDM * 0.16)[None, None, :] * np.exp(-dr / 0.04)[:, :, None]
            f = f * (1 - m) + col * m
        return (f * lum).astype(np.float32)

    # ---- さそりの 尾(式): 節の ある 弧 + 毒針 ----
    def tail(s, f, grow, t, v=1.0, bob=0.0):
        """grow: 0..1(上から のびて くる)。もどり値: 針の 先の 位置"""
        W, H, k = s.W, s.H, s.k; n = 9; pts = []
        for i in range(n + 1):                                                       # 上の 外 → 右へ ふくらんで → 目の前へ まわりこむ
            q = i / n; a = -1.9 + 3.9 * q; R = W * (0.62 - 0.20 * q)
            pts.append((W * 0.40 + math.cos(a) * R * 0.80, H * 0.20 + math.sin(a) * R * 0.86 + bob * q * q))
        m = int(grow * n + 0.999); fr = grow * n - int(grow * n) if grow < 1 else 1.0
        sh_ = np.zeros((H, W), np.float32); M = s.L0()
        for i in range(min(m, n)):
            a = pts[i]; b = pts[i + 1]; last = (i == m - 1)
            if last and grow < 1 and fr > 0: b = (a[0] + (b[0] - a[0]) * fr, a[1] + (b[1] - a[1]) * fr)
            c = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2); ln = math.hypot(b[0] - a[0], b[1] - a[1]); ang = math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])); wd = (64 - 3.4 * i) * k
            if ln < 2: continue
            cv2.ellipse(M, (int(c[0]), int(c[1])), (int(ln * 0.60), int(wd * 0.5)), ang, 0, 360, 1.0, -1, cv2.LINE_AA)
            cv2.ellipse(sh_, (int(c[0]), int(c[1])), (int(ln * 0.60), int(wd * 0.5)), ang, 0, 360, 0.40, -1, cv2.LINE_AA)
            nx = -math.sin(math.radians(ang)); ny = math.cos(math.radians(ang)); sg = -1 if nx * 1 + ny * -1 > 0 else 1                 # 上・左の がわに てり
            cv2.ellipse(sh_, (int(c[0] - sg * nx * wd * 0.20), int(c[1] - sg * ny * wd * 0.20)), (int(ln * 0.42), max(1, int(wd * 0.13))), ang, 0, 360, 1.0, -1, cv2.LINE_AA)
            cv2.ellipse(sh_, (int(c[0] + sg * nx * wd * 0.26), int(c[1] + sg * ny * wd * 0.26)), (int(ln * 0.50), max(1, int(wd * 0.14))), ang, 0, 360, 0.12, -1, cv2.LINE_AA)
        tip = pts[n]
        if grow >= 0.999:                                                            # 毒針(ふくらんだ 球 + とがった 針)
            a = pts[n]; d0 = (pts[n][0] - pts[n - 1][0], pts[n][1] - pts[n - 1][1]); dl = math.hypot(*d0); d0 = (d0[0] / dl, d0[1] / dl); bc = (a[0] + d0[0] * 26 * k, a[1] + d0[1] * 26 * k)
            cv2.circle(M, (int(bc[0]), int(bc[1])), int(30 * k), 1.0, -1, cv2.LINE_AA); cv2.circle(sh_, (int(bc[0]), int(bc[1])), int(30 * k), 0.45, -1, cv2.LINE_AA); cv2.circle(sh_, (int(bc[0] - 9 * k), int(bc[1] - 10 * k)), int(11 * k), 1.0, -1, cv2.LINE_AA)
            nd = (-0.24, 0.97); pn = (-nd[1], nd[0]); tip = (bc[0] + nd[0] * 92 * k, bc[1] + nd[1] * 92 * k)
            poly = np.array([(bc[0] + pn[0] * 17 * k, bc[1] + pn[1] * 17 * k), (bc[0] + nd[0] * 50 * k + pn[0] * 9 * k - d0[0] * 12 * k, bc[1] + nd[1] * 50 * k + pn[1] * 9 * k - d0[1] * 12 * k), tip, (bc[0] + nd[0] * 46 * k - pn[0] * 5 * k, bc[1] + nd[1] * 46 * k - pn[1] * 5 * k), (bc[0] - pn[0] * 17 * k, bc[1] - pn[1] * 17 * k)], np.int32)
            cv2.fillPoly(M, [poly], 1.0, cv2.LINE_AA); cv2.fillPoly(sh_, [poly], 0.75, cv2.LINE_AA)
        M = cv2.GaussianBlur(M, (0, 0), 0.8 * k) * v; sh_ = cv2.GaussianBlur(sh_, (0, 0), 2.2 * k) * (0.8 + 0.4 * s.tex(2, s.XX * 1.5, s.YY * 1.5))
        dk = np.array([0.06, 0.02, 0.10], np.float32); md = np.array([0.34, 0.14, 0.52], np.float32); hi = np.array([1.0, 0.72, 1.0], np.float32)
        a1 = sstep(0.0, 0.5, sh_)[:, :, None]; a2 = sstep(0.55, 1.0, sh_)[:, :, None]; col = dk + (md - dk) * a1 + (hi - md) * a2
        edge = np.clip(M - cv2.erode(M, np.ones((5, 5), np.uint8)), 0, 1)[:, :, None]; col = col + (REDM * 0.5)[None, None, :] * edge * (s.xn > 0.5)[:, :, None]            # 月の 赤い ふち光
        f *= (1 - M)[:, :, None]; f += col * M[:, :, None]
        return tip

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        if light is not None: return Kit.draw(s, f, img, cx, cy, hfrac, alpha, white, add, light)
        s.draw_lit(f, img, cx, cy, hfrac, rim=tuple(REDM * 0.8 + 0.2), rim_dir=(0.7, -0.7), rim_v=0.6 * alpha, amb=(0.8, 0.6, 1.0), amb_v=0.20, aura=None, aura_v=0.0, alpha=alpha, shade=0.72)

    def frame(s, t):
        o = {}
        if t < T_CUT: f, sh = s.sc_night(t)
        elif t < T_SINK: f, sh = s.sc_cut(t - T_CUT); o = dict(vig=0.45, thr=0.92)
        else: f, sh = s.sc_sting(t)
        f, sh = s.finale(f, sh, t, DUR, None, (0.8, 0.5, 1), (0.6, 0.2, 1.0), -1)
        return s.film(f, t, sh * s.k, 0.55, o.get("thr", 0.82), (0.95, 0.85, 1.0), 1.0 + 0.15 * min(sh, 8), 0.04, o.get("vig", 0.60), 1.15, 1.12)

    def hero(s, f, t, sink=0.0, **kw):
        W, H, k = s.W, s.H, s.k; size = 0.40; cy = H * 0.635; fy = cy + size * H * 0.46
        if sink <= 0:
            sh_ = s.L0(); cv2.ellipse(sh_, (int(W / 2 - 30 * k), int(fy)), (int(W * 0.22), int(H * 0.012)), 0, 0, 360, 0.5, -1, cv2.LINE_AA); f *= (1 - cv2.GaussianBlur(sh_, (0, 0), 4 * k))[:, :, None]
            img = s.pose(t, **kw); s.draw(f, img, W / 2, cy, size); return
        g = f.copy(); img = s.pose(t, **kw); s.draw(g, img, W / 2, cy + sink * size * H * 1.0, size)                                                    # 砂に しずむ(地面より 下は 見えない)
        m = sstep(fy + 2, fy - 2, s.YY)[:, :, None]; f[:] = f * (1 - m) + g * m
        L = s.L0(); cv2.ellipse(L, (int(W / 2), int(fy)), (int(W * (0.20 + 0.10 * sink)), int(H * 0.012)), 0, 0, 360, float(1 - sink), max(1, int(2 * k)), cv2.LINE_AA); s.glowmask(f, L, SAND * 2.2, 0.6, 1.0, 0.6, 6.0)
        rg = s.rnd[:120]; s.splat(f, W / 2 + (rg[:, 0] - 0.5) * W * 0.5, fy - rg[:, 1] * H * 0.05 * math.sin(sink * 3.14) * (1 + 2 * rg[:, 2]), rg[:, 3] * 0.4 * math.sin(sink * 3.14), SAND * 2.5, 2.0, 0.6)

    # 1) 赤い 月の 砂丘。さそり座が 灯る
    def sc_night(s, t):
        W, H, k = s.W, s.H, s.k
        f = s.dunes(t, 1.0, clamp((t - 0.6) / 1.7)); s.hero(f, t, 0.0, sway=0.3, breath=0.4)
        if t > 2.0: s.flare(f, s.at(*s.A(*DAG)), (6 + 10 * math.exp(-(t - 2.0) * 4)) * k, VEN, clamp((t - 2.0) * 5), 0.8, 0.5, False)
        return f * clamp(t / 0.5), 0

    # 2) 目の カットイン(毒針の 形の 帯)+ 技名
    def sc_cut(s, u):
        W, H, k = s.W, s.H, s.k; t = u + T_CUT
        f = s.dunes(t, 0.30, 1.0)
        a = ease(u / 0.22); cy = H * 0.33; hh = H * 0.095
        band = np.array([0.04, 0.01, 0.08], np.float32)[None, None, :] + (PUR * 0.5)[None, None, :] * (0.3 + 0.9 * s.tex(1, s.XX * 0.08 + t * 200, s.YY * 2.0))[:, :, None]
        ex, ey = s.A(*EYE); zf = 1.5; sc = zf * H / s.ah
        img = s.pose(t, breath=0.2); s.draw(band, img, W / 2 - (ex - s.aw / 2) * sc + (1 - a) * W * 0.9 - u * 6 * k, cy - (ey - s.ah / 2) * sc, zf)
        tipx = W * (1 - 1.02 * a); edge = hh * np.clip((s.XX - tipx) / (W * 0.22), 0, 1) ** 0.7; dd = np.abs(s.YY - cy)                                   # 左が とがった 針の 形
        m = (sstep(edge, edge - 2, dd) * (edge > 1))[:, :, None]; f = f * (1 - m) + band * m
        s.glowmask(f, np.exp(-((dd - edge) / (1.6 * k)) ** 2) * (edge > 1), VEN, 0.9 * a, 1.0, 0.8, 8.0)
        n = len(s._vt); step = W * 0.165
        for i, T in enumerate(s._vt):                                                # 技名: すっと 出て、さいごの 1文字だけ 刺さる
            p = (u - 0.45 - i * 0.10) / (0.25 if i < n - 1 else 0.10)
            if p > 0: s.text(f, T, W / 2 + (i - (n - 1) / 2) * step, H * 0.69 - ((1 - ease(p)) * 60 * k if i == n - 1 else 0), clamp(p * (1.2 if i < n - 1 else 2)), 1.0)
        for i, T in enumerate(s._vo):
            if u > 0.3 + i * 0.08: s.text(f, T, W / 2 + (i - 0.5) * 44 * k, H * 0.69 - 70 * k, 1.0, 1.0)
        if u > 1.3:
            g = ease((u - 1.3) / 0.35); s.glowline(f, [(W * 0.92, H * 0.69 + 52 * k), (W * (0.92 - 0.84 * g), H * 0.69 + 52 * k)], VEN, 1.0, 0.8, 0.8)
            s.text(f, s.txt_en, W / 2, H * 0.69 + 78 * k, g, 1.0)
        if u < 0.1: f = f + (1 - u / 0.1) * 0.6
        return f, 4 * math.exp(-u * 6) + 7 * math.exp(-max(0.0, u - 0.95) * 12) * (u > 0.95)

    # 3) 砂に 消える → 尾が 降りて くる → 毒の 一しずく
    def sc_sting(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_SINK; ut = t - T_TAIL; us = t - T_STING; ud = t - T_DROP; sh = 0.0
        f = s.dunes(t, 1.0 - 0.25 * clamp(ut / 1.2), 1.0 - 0.6 * clamp(ut / 0.8))
        sink = ease2(u / 0.85)
        if sink < 1: s.hero(f, t, sink, sway=0.2, squash=0.05 * sink)
        if ut > 0:
            grow = ease(clamp(ut / (T_STING - T_TAIL))) ; f *= (1 - 0.35 * np.exp(-((s.yn - 0.1) / 0.5) ** 2) * grow)[:, :, None]                        # 上から 影が さす
            tip = s.tail(f, grow, t, 1.0, 6 * k * math.sin(t * 2.6) * (1 if us < 0 else math.exp(-us * 2)))
            sh += 2.5 * (grow < 1) * abs(math.sin(ut * 9)) + (9 * math.exp(-us * 7) if us > 0 else 0)
            if us > 0:
                g = clamp(us / (T_DROP - T_STING)); fall = clamp(ud / 0.30) if ud > 0 else 0.0
                R = (3 + 17 * ease(g)) * k * (1 - 0.3 * fall); c = (tip[0], tip[1] + R * 0.9 + fall ** 2 * H * 0.22)                                    # しずく(ガラス玉)
                x = (s.XX - c[0]) / R; y = (s.YY - c[1]) / R; r = np.hypot(x, y * (1 - 0.25 * (y < 0))); ins = sstep(1.0, 0.85, r)
                col = (VEN * 0.9)[None, None, :] * (0.35 + 0.9 * r ** 2)[:, :, None] + (np.exp(-(np.hypot(x + 0.35, y + 0.4) / 0.22) ** 2) * 1.6)[:, :, None] + (np.exp(-(np.hypot(x - 0.3, y - 0.45) / 0.3) ** 2) * 0.5)[:, :, None] * PUR[None, None, :]
                if fall < 1: f *= (1 - ins)[:, :, None]; f += col * ins[:, :, None]; f += (np.exp(-(r / 2.4) ** 2) * 0.35)[:, :, None] * VEN[None, None, :]
                s.flare(f, tip, (7 + 5 * math.sin(t * 11) ** 2) * k, VEN, 0.8 * (1 - fall), 0.7, 0.3, False)
                if us < 0.08: f = f + (1 - us / 0.08) * 0.5
        uw = ud - 0.30                                                               # 波紋が ひろがって 視界が そまる
        if uw > 0:
            e = clamp(uw / (T_HIT - T_DROP - 0.30)); C = (W * 0.5, H * 0.60); d = np.hypot(s.XX - C[0], (s.YY - C[1]) * 1.0); fr = []
            for i in range(4):
                q = uw - i * 0.16
                if q > 0: fr.append((q ** 1.2 * W * 1.3 + 8 * k, 22 * k * math.exp(-q * 1.6), (18 + 40 * q) * k))
            disp = np.zeros_like(d); edge = np.zeros_like(d)
            for rf, amp, wd in fr: disp += amp * np.exp(-((d - rf) / wd) ** 2); edge += np.exp(-((d - rf) / (wd * 0.22)) ** 2) * min(1.0, amp / (16 * k))
            dx = (s.XX - C[0]) / (d + 1e-3); dy = (s.YY - C[1]) / (d + 1e-3); f = cv2.remap(f, (s.XX - dx * disp).astype(np.float32), (s.YY - dy * disp).astype(np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            stain = sstep(e * 1.7 + 0.05, e * 1.7 - 0.25, d / W + (s.tex(1, s.XX * 0.5, s.YY * 0.5) - 0.5) * 0.25)
            n = s.tex(0, s.XX * 0.3 + t * 12, s.YY * 0.3 - t * 20) * 0.6 + s.tex(2, s.XX * 0.8, s.YY * 0.8 + t * 30) * 0.4
            f = f * (1 - 0.82 * stain)[:, :, None] + ((PUR * 0.9)[None, None, :] * (0.25 + 1.1 * n ** 1.5)[:, :, None] + (np.exp(-(d / (W * (0.1 + 0.7 * e))) ** 2) * (0.5 + 1.8 * e))[:, :, None] * (VEN * 0.6 + 0.4)[None, None, :]) * stain[:, :, None]
            f += (edge * 0.9)[:, :, None] * (VEN * 0.7 + 0.3)[None, None, :]
            if uw < 0.1: f = f + (1 - uw / 0.1) * 0.9
            sh += 12 * math.exp(-uw * 4) + 2 + 6 * e
            if e > 0.80: f = f + ((e - 0.80) / 0.20) ** 2 * 1.5
        return f, sh

    # ---- この子だけの曲: 低い 持続と かさかさ いう 足音 → しずけさ → きしむ 節の 音が 9つ → 無音 → しずくの 一音 → にじんで ひろがる 不協和音 ----
    def audio(s, path):
        a = Snd(DUR, 406); B = 123.47; ph = [0, 1, 3, 5, 6, 8, 10]
        def fq(d, o=0): return B * 2 ** ((ph[d % 7] + 12 * (d // 7 + o)) / 12.0)
        a.a += (a.tone(B / 2) + 0.5 * a.tone(B / 2 * 1.06)) * 0.09 * np.clip(a.t / 1.5, 0, 1) * a.gate(0, T_STING, 0.3)
        a.a += (a.lp(a.nz, 0.08) - a.lp(a.nz, 0.03)) * a.gate(0, T_SINK, 0.3) * 0.4
        for i in range(11): a.note(fq([7, 5, 4, 3, 2, 3, 4, 5, 6, 7, 8][i], 1), 0.6 + i * 0.155, 0.4, 0.07 + (0.06 if i == 3 else 0), "bell")
        a.boom(T_CUT, 48, 1.0, 0.45); a.whoosh(T_CUT - 0.25, T_CUT, 0.4, 0.5)
        for i in range(4): a.note(fq([4, 3, 1, 0][i], 1), T_CUT + 0.45 + i * 0.10, 0.4, 0.08, "fm")
        a.boom(T_CUT + 0.95, 70, 1.0, 0.15); a.a += (a.nz - a.lp(a.nz, 0.7)) * a.env(T_CUT + 0.95, 0.001, 0.04) * 0.6
        for d in (0, 1, 4): a.note(fq(d, 0), T_CUT, 1.9, 0.04, "pad", 0.1)
        a.a += (a.nz - a.lp(a.nz, 0.3)) * a.gate(T_SINK, T_SINK + 0.85, 0.2) * 0.12                                           # 砂に しずむ
        rg = np.random.RandomState(406)
        for i in range(9):                                                                                               # 尾の 節が 降りて くる: きしみ
            tk = T_TAIL + (T_STING - T_TAIL) * (1 - (1 - (i + 0.5) / 9) ** (1 / 3.0)) if False else T_TAIL + (T_STING - T_TAIL) * (i + 0.3) / 9.5
            a.a += (a.nz - a.lp(a.nz, 0.55)) * a.env(tk, 0.002, 0.035) * 0.45; a.boom(tk, 62 - 3 * i, 0.5 + 0.06 * i, 0.12); a.note(fq(8 - i, 0), tk, 0.2, 0.05, "fm")
        a.rise(T_TAIL, T_STING, 50, 120, 0.13)
        a.boom(T_STING, 40, 1.2, 0.4); a.note(fq(14, 2), T_STING + 0.05, 0.9, 0.06, "sine")
        a.note(fq(11, 2), T_DROP + 0.30, 1.2, 0.16, "bell"); a.note(1800, T_DROP + 0.30, 0.08, 0.10, "sine", 0.001)               # しずくが 落ちる
        tw = T_DROP + 0.30; a.boom(tw + 0.02, 38, 1.3, 0.7)
        for j, d in enumerate((0, 1, 4, 6, 8, 11)): a.a += np.sin(2 * np.pi * fq(d, 0) * a.t * (1 + 0.004 * np.sin(2 * np.pi * (3 + j) * a.t)) + 2 * np.sin(2 * np.pi * 4 * a.t)) * a.env(tw + j * 0.1, 0.3, 1.2) * 0.035
        for i in range(4): a.boom(tw + i * 0.16, 60, 0.5, 0.2)
        a.rise(tw, T_HIT, 70, 260, 0.12); a.a += a.lp(a.nz, 0.06) * np.clip((a.t - tw) / (T_HIT - tw), 0, 1) ** 2 * (a.t < T_HIT) * (a.t > tw) * 0.7
        a.boom(T_HIT, 36, 1.5, 0.8)
        for j, d in enumerate((0, 3, 4, 7)): a.note(fq(d, 1), T_HIT + 0.02, 0.6, 0.05, "bell")
        s.fin_snd(a, DUR, 406); a.save(path)


if __name__ == "__main__":
    r = Scorpi()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
