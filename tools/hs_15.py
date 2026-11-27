"""アーサー(★5 聖騎士) 専用の必殺技「聖剣ノ誓」— 高品質版(hs_fx)・通し直し。敵は さいしょから 画面の こちら側。ばら窓の 12色が 12本の 光の剣に なって こちらへ
暗い大聖堂(石の柱・ほこりの 舞う 色つきの 光の すじ)。ばら窓のステンドグラスが 1枚ずつ 灯り、色の光が 床に落ちる → 剣の刃に 目が うつる カットイン →
円卓の誓い: 足もとの輪から、ばら窓の 外の輪と 同じ 色の 光の剣が 1本ずつ 立ち上がる(灯った ガラスから 色の 糸が おりてくる) →
聖剣を かかげて 号令。12本が いっせいに 切っ先を こちらへ 向け、時計まわりに 1本ずつ 色の 尾を ひいて 射ち出される → さいごに 聖剣の 金の光が まん中を つらぬく → 白 → 暗転
使い方: python3 hs_15.py 出力.mp4 / python3 hs_15.py still 出力.png 時刻,時刻"""
import sys, math, importlib.util
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_kit import Kit, Snd, encode, stills, ease, ease2, clamp, back, WHITE, FPS
from hs_fx import FX, sstep
_sp = importlib.util.spec_from_file_location("old15", "/home/claude/tools/pre_fin/hs_15_v3.py"); old = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(old)
DUR = 9.6
GOLD, STONE, STEEL, PAL = old.GOLD, old.STONE, old.STEEL, old.PAL
T_CUT, T_OATH = old.T_CUT, old.T_OATH
RSTEP, T_AIM, T_FIRE, FSTEP, T_LAST = 0.17, 6.9, 7.35, 0.095, 8.45
SC = [np.array(PAL[(i + 6) % 6], np.float32) for i in range(12)]                       # 剣の色 = ばら窓の 外の輪(si + ri*2 = i + 6)


class Arthur(FX, old.Arthur):
    def __init__(s, W=540):
        old.Arthur.__init__(s, W); s.fx_init(15); H, k = s.H, s.k
        s.wall = 0.6 + 0.8 * s.fbm(0.004, 3, 1, 4, 0, 1)
        bx = (np.mod(s.XX + (np.floor(s.YY / (46 * k)) % 2) * 40 * k, 80 * k) < 2 * k) | (np.mod(s.YY, 46 * k) < 2 * k); s.brick = (1 - 0.45 * bx).astype(np.float32)      # 石の 目地
        sh = np.zeros((H, W), np.float32); pm = np.zeros((H, W), np.float32)
        for x, w in ((0.07, 30), (0.93, 30)):
            q = (s.XX - x * W) / (w * k / 2); ins = (np.abs(q) < 1.0) & (s.YY < s.gnd)
            pm = np.where(ins, 1.0, pm); sh = np.where(ins, 0.25 + 0.9 * np.sqrt(np.clip(1 - q * q, 0, 1)) * (0.8 + 0.3 * q * (1 if x < 0.5 else -1)), sh)
        s.pmask = pm.astype(np.float32); s.pshade = sh.astype(np.float32)
        s.gmot = 0.55 + 0.9 * s.fbm(0.012, 7, 2, 3, 0, 2)                                # ガラスの むら
        s.leadhi = np.clip(s.lead - cv2.warpAffine(s.lead, np.float32([[1, 0, 1.5 * k], [0, 1, 1.5 * k]]), (W, H)), 0, 1)
        dep = np.clip((s.YY - s.gnd) / (H - s.gnd), 0, 1); tx = (s.XX - W / 2) / ((70 + 260 * dep) * k); ty = dep * 6 / (0.5 + dep)
        s.tile = (1 - 0.5 * ((np.abs(tx - np.round(tx)) < 0.025) | (np.abs(ty - np.round(ty)) < 0.03))).astype(np.float32)
        s.VC = (W / 2, H * 0.44)
        s.sp = [(W / 2 + math.cos(-1.5708 + i * 0.5236) * W * 0.41, s.gnd + 8 * k + math.sin(-1.5708 + i * 0.5236) * H * 0.055) for i in range(12)]
        s.outer = [j for j, p in enumerate(s.pan) if p[2] == 3]                         # 外の輪の ガラス 12枚

    # ---- 大聖堂 ----
    def hall(s, lit, glow=1.0):
        W, H, k = s.W, s.H, s.k; t = getattr(s, "_t", 0.0)
        f = (STONE[None, None, :] * (s.wall * s.brick * (0.45 + 0.8 * s.yn))[:, :, None]).astype(np.float32)
        f = f * (1 - s.pmask[:, :, None]) + (STONE * 2.3)[None, None, :] * (s.pshade * s.wall)[:, :, None] * s.pmask[:, :, None]
        for q, v in ((1.0, 0.5), (0.90, 0.3), (0.80, 0.2)):                           # アーチの すじ
            P = [(W / 2 + math.cos(math.radians(180 + a * 6)) * W * 0.43 * q, H * 0.30 + math.sin(math.radians(180 + a * 6)) * H * 0.30 * q) for a in range(31)]
            s.glowline(f, P, STONE * 5 + 0.05, 2.0, v * 0.5, 0.3)
        lv = np.concatenate([lit, [0.0]]).astype(np.float32)
        glass = (s.pcol * lv[:, None])[s.idx]
        rr = np.hypot(s.XX - s.O[0], s.YY - s.O[1]) / s.R
        gl = glass * (s.gmot * (0.75 + 0.45 * np.clip(1 - rr, 0, 1)))[:, :, None] * 1.15 * glow
        f += gl; f *= (1 - s.lead * 0.92)[:, :, None]; f += (s.leadhi * 0.10 * lv.max())[:, :, None]
        f += cv2.GaussianBlur(glass, (0, 0), 20 * k) * 0.5 * glow
        sh = cv2.warpAffine(glass, np.float32([[1, 0, 0], [0, 1.25, H * 0.16]]), (W, H)); sh = cv2.GaussianBlur(sh, (0, 0), sigmaX=16 * k, sigmaY=95 * k)       # 色の 光の すじ(ばら窓から 床へ)
        hz_ = 0.45 + 0.9 * s.fbm(0.0018, t * 9, t * 6, 3, 0, 3); d = np.zeros_like(f); s.starfield(d, t, 1300, (1, 1, 1), 3, 1.2, (5.0, 9.0))
        f += sh * (hz_ * 0.55 + d[:, :, 0] * 3.0)[:, :, None] * glow * sstep(s.O[1] - s.R * 0.3, s.O[1] + s.R, s.YY)[:, :, None]
        g = sstep(s.gnd - 1, s.gnd + 1, s.YY); fl_ = (STONE * 0.9)[None, None, :] * (s.tile * s.wall)[:, :, None]
        fl = cv2.resize(glass[int(s.O[1] - s.R):int(s.O[1] + s.R), :], (W, H - int(s.gnd)))[::-1]                          # 床に うつる
        fl_[int(s.gnd):] += cv2.GaussianBlur(fl, (0, 0), 7 * k) * 0.42 * glow
        return f * (1 - g[:, :, None]) + fl_ * g[:, :, None]

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        s.draw_lit(f, img, cx, cy, hfrac, rim=(1.0, 0.95, 0.85), rim_dir=(0.0, -1.0), rim_v=0.75 * alpha, amb=(0.9, 0.9, 1.0), amb_v=0.10, alpha=alpha, shade=0.95 * getattr(s, '_hsh', 1.0))
        if white > 0.02: s.put(f, img, cx, cy, hfrac * s.H / s.ah, alpha, 0.0, white)

    def add(s, f, L, col, gain=1.0, b1=3, b2=12):
        s.glowmask(f, np.clip(L, 0, 1), col, gain, 0.8, 1.0, float(b2))

    def blade(s, f, v, ext=0.0):
        if v <= 0: return
        a = s.pt(s.guard); b = s.pt(s.tip); s.glowline(f, [a, b], WHITE * 0.7 + GOLD * 0.3, 1.8 + 3 * ext, 0.7 * v, 1.0 + ext)
        s.flare(f, b, (5 + 16 * ext) * s.k * v, WHITE * 0.7 + GOLD * 0.3, 0.6 * v, 1.0 + ext, 0.0, False)

    # ---- 光の剣 ----
    def lsword(s, f, i, h, a=1.0):
        """立っている 光の剣(色つき)"""
        if a <= 0.01 or h < 2: return
        k = s.k; x, y = s.sp[i]; col = SC[i] * 0.75 + 0.25
        s.glowline(f, [(x, y), (x, y - h)], col, 2.2, 0.85 * a, 1.0); s.glowline(f, [(x - 11 * k, y - h * 0.28), (x + 11 * k, y - h * 0.28)], GOLD, 1.8, 0.7 * a, 0.6)
        s.glowline(f, [(x - 3 * k, y - h), (x, y - h - 12 * k), (x + 3 * k, y - h)], col, 1.4, 0.8 * a, 0.6)
        P = [(x + math.cos(q * 0.2618) * 16 * k, y + math.sin(q * 0.2618) * 4 * k) for q in range(24)]; s.glowline(f, P, col, 0.9, 0.5 * a, 0.5, True)

    def vsword(s, f, c, S, col, a=1.0, trail=0.0):
        """こちらを 向いた 光の剣(式): 切っ先の ひし形(色ガラスの 4面)・金の つば・色の 尾"""
        if a <= 0.01: return
        k = s.k; W, H = s.W, s.H; col = np.asarray(col, np.float32)
        if trail > 0.02:
            P = [s.VC, c]; s.glowline(f, P, col * 0.8 + 0.2, max(1.0, S * 0.05 / k), 0.55 * trail * a, 1.2)
        r = S; x0 = int(max(0, c[0] - r * 2.2)); x1 = int(min(W, c[0] + r * 2.2)); y0 = int(max(0, c[1] - r * 1.5)); y1 = int(min(H, c[1] + r * 1.5))
        if x1 - x0 < 3 or y1 - y0 < 3: return
        X = s.XX[y0:y1, x0:x1] - c[0]; Y = s.YY[y0:y1, x0:x1] - c[1]; F = f[y0:y1, x0:x1]
        gm = sstep(r * 0.11, r * 0.08, np.abs(Y)) * sstep(r * 1.15, r * 1.08, np.abs(X)) * sstep(r * 0.3, r * 0.4, np.abs(X))                   # つば
        gy_ = np.clip(Y / (r * 0.10), -1, 1); gc = GOLD[None, None, :] * (0.85 - 0.45 * gy_ + 0.5 * np.exp(-((gy_ + 0.45) / 0.18) ** 2))[:, :, None]
        F[:] = F * (1 - gm[:, :, None] * a) + gc * gm[:, :, None] * a
        dx = X / (r * 0.55); dy = Y / (r * 1.25); d = np.abs(dx) + np.abs(dy); aa = min(0.25, 2.5 / max(3.0, r * 0.55)); m = sstep(1.0, 1.0 - aa, d) * a
        base = np.where(dx < 0, np.where(dy < 0, 1.25, 0.55), np.where(dy < 0, 0.85, 0.30)).astype(np.float32)
        ridge = np.exp(-(X / (1.2 * k + r * 0.012)) ** 2) + np.exp(-(Y / (1.2 * k + r * 0.012)) ** 2)
        val = base * (0.60 + 0.40 * (1 - d)) + ridge * 0.5
        body = val[:, :, None] * (col * 1.25 + 0.15)[None, None, :] * (1 - 0.5 * sstep(0.90, 1.0, d))[:, :, None] + (np.clip(1 - d, 0, 1) ** 2 * 0.7)[:, :, None]
        F += (np.exp(-np.clip(d - 1, 0, None) * 3.0) * (d > 1) * 0.55 * a)[:, :, None] * col[None, None, :]
        F[:] = F * (1 - m[:, :, None]) + body * m[:, :, None]
        s.flare(f, c, max(2.5, min(26 * k, r * 0.25)), col * 0.5 + 0.5, 0.8 * a, 1.3, 0.0, False)

    def frame(s, t):
        s._t = t; o = {}
        if t < T_CUT: f, sh = s.sc_window(t)
        elif t < T_OATH: f, sh = s.sc_cut(t - T_CUT); s.starfield(f, t, 300, GOLD, 5, 0.5, (0.0, -20.0)); o = dict(vig=0.45, thr=0.9)
        else: f, sh = s.sc_oath(t)
        f, sh = s.finale(f, sh, t, DUR, s.tip, (1, 0.85, 0.4), (0.6, 0.8, 1), -1)
        return s.film(f, t, sh * s.k, 0.6, o.get("thr", 0.82), (1.0, 0.92, 0.75), 1.0 + 0.2 * sh, 0.035, o.get("vig", 0.6), 1.15, 1.05)

    # 1) ばら窓が 灯る(はじめは 影。光が 入ると 色が のる)
    def sc_window(s, t):
        W, H, k = s.W, s.H, s.k
        f = s.hall(s.lit_at(ease2(t / 1.9)))
        s._hsh = 0.25 + 0.75 * ease2((t - 0.6) / 1.4); s.hero(f, t, W / 2, s.gnd - H * 0.20, 0.42, 0, 1.0, 0.0); s._hsh = 1.0
        s.blade(f, ease((t - 1.9) / 0.4) * 0.7)
        return f * clamp(t / 0.4), 0

    # 3) 円卓の誓い → 号令 → 12本が こちらへ → 聖剣
    def sc_oath(s, t):
        W, H, k = s.W, s.H, s.k; u = t - T_OATH; am = clamp((t - T_AIM) / 0.35); la = t - T_LAST; gl_ = clamp(la / (DUR - 0.7 - T_LAST)) if la > 0 else 0.0
        lit = np.ones(s.np_, np.float32)
        for i in range(12):                                                          # 剣が 立つ とき、同じ色の ガラスが 強く 光る
            v = u - (0.3 + i * RSTEP)
            if 0 < v < 0.6: lit[s.outer[i]] = 1.0 + 1.6 * math.exp(-v * 5)
            if t > T_FIRE + i * FSTEP: lit[s.outer[i]] = 0.25                          # 射ち出すと その色が ぬける
        f = s.hall(lit, 1.0 + 0.3 * am)
        P = [(W / 2 + math.cos(q * 0.1309) * W * 0.41, s.gnd + 8 * k + math.sin(q * 0.1309) * H * 0.055) for q in range(48)]; s.glowline(f, P, GOLD, 1.4, 0.6 * ease(u / 0.5), 0.8, True)
        def put_swords(front):
            for i in range(12):
                sn = math.sin(-1.5708 + i * 0.5236)
                if (sn > 0.05) != front: continue
                v = u - (0.3 + i * RSTEP); g = back(v / 0.3, 1.8)
                if g <= 0: continue
                x, y = s.sp[i]; h = H * (0.15 + 0.03 * sn) * g; fv = t - (T_FIRE + i * FSTEP)
                if 0 < v < 0.35:                                                     # ガラスから 色の 糸が おりる
                    pc = s.pan[s.outer[i]][0]; s.glowline(f, [pc, (x, y - h)], SC[i] * 0.8 + 0.2, 0.9, 0.7 * (1 - v / 0.35), 0.8)
                if fv <= 0:
                    s.lsword(f, i, h * (1 - 0.8 * am), 1.0); 
                    if am > 0: s.vsword(f, (x, y - h * (1 - 0.8 * am)), (5 + 6 * am) * k, SC[i], am, 0.0)
        put_swords(False)
        cy = s.gnd - H * 0.20; cmd = math.exp(-max(0.0, t - T_AIM) * 5) if t > T_AIM else 0.0; pre = clamp((t - (T_AIM - 0.4)) / 0.4)
        s.hero(f, t, W / 2, cy, 0.42, -4 * pre + 4 * cmd, 1.0, 0.0, squash=(-0.03 * pre * (t < T_AIM) + 0.03 * cmd))
        s.blade(f, 0.7 + 0.5 * cmd + 0.8 * gl_, 0.3 * cmd + 0.6 * gl_)
        put_swords(True)
        sh = 3 * cmd
        for i in range(12):                                                          # 時計まわりに 1本ずつ 射ち出される
            fv = t - (T_FIRE + i * FSTEP)
            if fv <= 0 or fv > 0.55: continue
            e = clamp(fv / 0.45) ** 2.2; x, y = s.sp[i]; p0 = (x, y - H * 0.03); a_ = -1.5708 + i * 0.5236
            c = (s.VC[0] + (p0[0] - s.VC[0]) * (1 + 0.6 * e) + math.cos(a_) * W * 0.55 * e, s.VC[1] + (p0[1] - s.VC[1]) * (1 - e) + math.sin(a_) * H * 0.50 * e)
            s.vsword(f, c, (11 + 250 * e) * k, SC[i], 1.0, min(1.0, fv * 6))
            if fv < 0.06: f += (1 - fv / 0.06) * 0.12 * SC[i][None, None, :]
            sh = max(sh, 4 + 5 * e)
        if t > T_FIRE: s.rushdust(f, t, s.VC, 1300, 1.0 + 1.5 * clamp((t - T_FIRE) / 1.2), WHITE * 0.6 + GOLD * 0.4, 0.4 + 0.6 * clamp((t - T_FIRE) / 1.0), 4, 1.0)
        if la > 0:                                                                   # 聖剣の 金の光が まん中を つらぬく
            tp = s.pt(s.tip); c = (tp[0] + (s.VC[0] - tp[0]) * ease2(gl_), tp[1] + (s.VC[1] - tp[1]) * ease2(gl_)); S = (12 + 900 * gl_ ** 2.2) * k
            s.vsword(f, c, S, GOLD * 0.5 + 0.5, 1.0, 0.0); s.flare(f, c, min(70 * k, 14 * k + S * 0.3), WHITE * 0.7 + GOLD * 0.3, 1.0, 2.2, t * 0.4, gl_ < 0.5)
            f = s.godrays(f, c, 0.4 + 0.5 * gl_, 0.75, 10, 0.88, 0.5); sh = max(sh, 5 + 10 * gl_)
        if u < 0.12: f += (1 - u / 0.12) * 0.6
        return f, sh

    def audio(s, path):
        a = Snd(DUR, 15); D = 146.83; maj = [0, 2, 4, 5, 7, 9, 11]
        def fq(d, o=0): return D * 2 ** ((maj[d % 7] + 12 * (d // 7 + o)) / 12.0)
        def organ(ds, t0, dur, g=0.035):
            for d, o in ds: a.note(fq(d, o), t0, dur, g, "sq", 0.06); a.note(fq(d, o) * 2, t0, dur, g * 0.4, "sine", 0.06)
        organ([(0, -1), (0, 0), (4, 0)], 0.1, 1.2); organ([(3, -1), (0, 0), (5, 0)], 1.0, 1.0); organ([(4, -1), (1, 0), (4, 0), (6, 0)], 1.7, 0.9)
        for i in range(12): a.note(fq(i % 7 + 7, 1), 0.2 + i * 0.14, 0.2, 0.025, "bell")                               # ガラスが 灯る
        a.boom(T_CUT, 50, 1.0); a.rise(T_CUT, T_CUT + 0.3, 800, 2400, 0.08)
        for i in range(4): a.boom(T_CUT + 0.5 + i * 0.16, 64, 0.5, 0.15); a.note(fq([0, 4, 7, 11][i], 1), T_CUT + 0.5 + i * 0.16, 0.5, 0.10, "bell")
        organ([(0, -1), (4, -1), (0, 0), (2, 0)], T_CUT + 0.5, 1.6)
        sc12 = [0, 1, 2, 4, 3, 4, 5, 7, 6, 7, 8, 9]
        for i in range(12): a.note(fq(sc12[i], 1), T_OATH + 0.3 + i * RSTEP, 0.6, 0.10, "bell"); a.note(fq(sc12[i], 0), T_OATH + 0.3 + i * RSTEP, 0.3, 0.04, "fm")      # 12本が 立つ
        organ([(0, -1), (0, 0)], T_OATH, 1.4, 0.03); organ([(3, -1), (5, 0)], T_OATH + 1.3, 1.3, 0.03)
        for d, o in ((0, 0), (2, 0), (4, 0), (7, 0)): a.note(fq(d, o), T_OATH + 0.2, 2.6, 0.03, "pad", 0.6)             # 聖歌
        organ([(4, -2), (4, -1), (1, 0), (4, 0), (6, 0)], T_AIM, 0.6, 0.045); a.boom(T_AIM, 50, 1.0, 0.3); a.rise(T_AIM, T_FIRE, 400, 1600, 0.10)        # 号令
        for i in range(12): a.hat(T_AIM + 0.05 + i * 0.03, 0.05, 0.01)                                                  # 切っ先が そろう
        for i in range(12):                                                                                            # 同じ 12音が 速く もう一度(射ち出す)
            t0 = T_FIRE + i * FSTEP; a.note(fq(sc12[i], 1), t0, 0.3, 0.09, "bell"); a.rise(t0, t0 + 0.4, 700 + 60 * i, 2800 + 150 * i, 0.07); a.whoosh(t0 + 0.1, t0 + 0.42, 0.35, 0.4); a.boom(t0, 70, 0.3 + 0.03 * i, 0.10)
        organ([(0, -2), (0, -1), (4, -1), (0, 0), (2, 0), (4, 0)], T_FIRE, 1.2, 0.04)
        organ([(0, -2), (0, -1), (4, -1), (0, 0), (2, 0), (4, 0), (7, 0)], T_LAST, DUR - T_LAST, 0.05)                    # 聖剣: 長調の 和音
        a.rise(T_LAST, DUR - 0.7, 150, 1100, 0.14); a.a += a.lp(a.nz, 0.3) * np.clip((a.t - T_LAST) / (DUR - 0.7 - T_LAST), 0, 1) ** 2 * ((a.t > T_LAST) & (a.t < DUR - 0.5)) * 0.8
        a.boom(DUR - 0.72, 38, 1.6, 0.6)
        s.fin_snd(a, DUR, 15); a.save(path)


if __name__ == "__main__":
    r = Arthur()
    if sys.argv[1] == "still": stills(r, [float(x) for x in sys.argv[3].split(",")], sys.argv[2], 270)
    else: encode(r, DUR, sys.argv[1], 22)
