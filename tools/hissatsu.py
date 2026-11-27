"""必殺技の動画を キャラごとに作る。
使い方: python3 hissatsu.py <id> <属性> <技名> <英語名> <出力.mp4> [幅=540]
流れ: 影から登場 → 魔法陣と ため → 奥義カットイン(技名) → 白フラッシュ → 属性ごとの大技 → 白とび
"""
import sys, os, re, io, math, subprocess, wave
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

FPS = 30
DUR = 10.8
COL = {"fire": (255, 122, 42), "water": (79, 195, 255), "leaf": (123, 224, 90), "earth": (208, 165, 106), "wind": (168, 247, 228),
       "thunder": (255, 225, 74), "ice": (201, 239, 255), "dark": (164, 92, 255), "light": (255, 233, 160), "star": (255, 208, 255),
       "petal": (255, 154, 200), "poison": (197, 107, 255)}
SERIF = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
SANS = "/usr/share/fonts/opentype/noto/NotoSansCJK-Black.ttc"


def load_char(i, H=760):
    """元の絵を 背景ぬきで読む(hiart.py の cut を借りる)"""
    src = open('/home/claude/tools/hiart.py').read()
    ns = {}
    head = src[:src.index("F='/home/claude/teawase/index.html'")]
    exec(head, ns)
    exec(src[src.index("def src(i):"):src.index("def cutAnim")], ns)
    f = ns['src'](i)
    g = '/home/claude/anim/%d_f0.png' % i
    if os.path.exists(g):
        f = g
    im = ns['cut'](f, H, True)
    im = im.crop(im.getbbox())
    sc = H / im.height
    im = im.resize((max(1, round(im.width * sc)), H), Image.LANCZOS)
    return np.asarray(im).astype(np.float32) / 255.0


class R:
    def __init__(s, cid, elem, name, en, W=540):
        s.W = W; s.H = W * 16 // 9; s.k = W / 540.0
        s.elem = elem; s.col = np.array(COL[elem], np.float32) / 255.0
        s.name = name; s.en = en
        s.rng = np.random.RandomState(cid * 7 + 3)
        s.ch = load_char(cid, int(760 * s.k))
        s.sil = s.ch.copy(); s.sil[:, :, :3] = 0
        # 星雲: ぼかした乱数を2段
        n = s.rng.rand(s.H // 8 + 2, s.W // 8 + 40).astype(np.float32)
        n = cv2.GaussianBlur(n, (0, 0), 3)
        n = (n - n.min()) / (n.max() - n.min() + 1e-6)
        s.neb = cv2.resize(n, (s.W + 320, s.H), interpolation=cv2.INTER_CUBIC)
        s.stars = [(s.rng.randint(0, s.W), s.rng.randint(0, s.H), s.rng.rand()) for _ in range(90)]
        s.sparks = [(s.rng.rand() * 6.283, 0.2 + s.rng.rand() * 0.8, s.rng.rand(), 0.5 + s.rng.rand()) for _ in range(70)]
        s.cracks = [(s.rng.rand() * 6.283, s.rng.rand(), s.rng.rand() * 6.283, 0.2 + s.rng.rand() * 0.5) for _ in range(46)]
        s.dots = [(s.rng.rand() * 6.283, s.rng.rand()) for _ in range(160)]
        s.txt_main = s.text_img(name, SERIF, int(74 * s.k) if len(name) <= 5 else int(60 * s.k), 3)
        s.txt_ougi = s.text_img("奥 義", SERIF, int(26 * s.k), 1)
        s.txt_en = s.text_img(" ".join(en.upper()), SANS, int(17 * s.k), 0) if en else None

    # ---- 部品 ----
    def text_img(s, t, font, size, stroke):
        f = ImageFont.truetype(font, size, index=0)
        d = ImageDraw.Draw(Image.new("L", (8, 8)))
        b = d.textbbox((0, 0), t, font=f, stroke_width=stroke)
        w, h = b[2] - b[0] + 60, b[3] - b[1] + 60
        m = Image.new("L", (w, h), 0)
        ImageDraw.Draw(m).text((30 - b[0], 30 - b[1]), t, font=f, fill=255, stroke_width=stroke, stroke_fill=255)
        core = Image.new("L", (w, h), 0)
        ImageDraw.Draw(core).text((30 - b[0], 30 - b[1]), t, font=f, fill=255)
        glow = m.filter(ImageFilter.GaussianBlur(9 * s.k))
        return (np.asarray(core, np.float32) / 255, np.asarray(m, np.float32) / 255, np.asarray(glow, np.float32) / 255)

    def bg(s, t, dim=1.0):
        off = int((t * 14) % 300)
        n = s.neb[:, off:off + s.W]
        f = np.zeros((s.H, s.W, 3), np.float32)
        yy = np.linspace(0, 1, s.H, dtype=np.float32)[:, None]
        v = (0.16 + 0.55 * n ** 2) * (0.55 + 0.45 * np.sin(yy * 3.14)) * dim
        f += v[:, :, None] * (s.col * 0.55 + np.array([0.12, 0.05, 0.25], np.float32))[None, None, :] * 0.8
        for x, y, p in s.stars:
            b = (0.35 + 0.65 * math.sin(t * 3 + p * 20)) * 0.6 * dim
            if b > 0: f[y, x] += b
        return f

    def put(s, f, img, cx, cy, sc, alpha=1.0, add=0.0, white=0.0):
        h, w = img.shape[:2]
        nw, nh = max(2, int(w * sc)), max(2, int(h * sc))
        im = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if sc < 1 else cv2.INTER_LINEAR)
        x0, y0 = int(cx - nw / 2), int(cy - nh / 2)
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(s.W, x0 + nw), min(s.H, y0 + nh)
        if xa >= xb or ya >= yb: return
        c = im[ya - y0:yb - y0, xa - x0:xb - x0]
        a = c[:, :, 3:4] * alpha
        rgb = c[:, :, :3] * (1 - white) + white
        f[ya:yb, xa:xb] = f[ya:yb, xa:xb] * (1 - a) + (rgb + add) * a

    def bloom(s, f, L, col=None, gain=1.0):
        """L: 1ch の線画。にじませて 色つきで足す"""
        col = s.col if col is None else col
        b1 = cv2.GaussianBlur(L, (0, 0), 3 * s.k); b2 = cv2.GaussianBlur(L, (0, 0), 12 * s.k)
        g = (L * 0.9 + b1 * 0.9 + b2 * 1.1) * gain
        f += g[:, :, None] * col[None, None, :] + (np.clip(L, 0, 1) ** 2 * 0.8 * gain)[:, :, None]

    def L0(s): return np.zeros((s.H, s.W), np.float32)

    def circ(s, L, c, r, v=1.0, th=2):
        if r > 1: cv2.circle(L, (int(c[0]), int(c[1])), int(r), float(v), max(1, int(th * s.k)), cv2.LINE_AA)

    def line(s, L, a, b, v=1.0, th=2):
        cv2.line(L, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), float(v), max(1, int(th * s.k)), cv2.LINE_AA)

    def ell(s, L, c, ax, ang, v=1.0, th=2, a0=0, a1=360):
        cv2.ellipse(L, (int(c[0]), int(c[1])), (int(ax[0]), int(ax[1])), ang, a0, a1, float(v), max(1, int(th * s.k)), cv2.LINE_AA)

    def poly(s, L, pts, v=1.0, th=2, fill=False):
        p = np.array(pts, np.int32).reshape(-1, 1, 2)
        if fill: cv2.fillPoly(L, [p], float(v), cv2.LINE_AA)
        else: cv2.polylines(L, [p], True, float(v), max(1, int(th * s.k)), cv2.LINE_AA)

    def lens(s, c, ang, ln, wd, n=14):
        """葉っぱ・花びら・目の形"""
        pts = []
        for i in range(n + 1):
            u = i / n; pts.append((u * ln, math.sin(u * math.pi) * wd))
        for i in range(n, -1, -1):
            u = i / n; pts.append((u * ln, -math.sin(u * math.pi) * wd))
        ca, sa = math.cos(ang), math.sin(ang)
        return [(c[0] + x * ca - y * sa, c[1] + x * sa + y * ca) for x, y in pts]

    def circle_magic(s, L, c, r, t, v=1.0):
        s.circ(L, c, r, v, 2); s.circ(L, c, r * 0.9, v * 0.6, 1); s.circ(L, c, r * 0.55, v * 0.5, 1)
        for i in range(3):
            a = t * 0.6 + i * 2.094
            pts = [(c[0] + math.cos(a + j * 2.094) * r * 0.9, c[1] + math.sin(a + j * 2.094) * r * 0.9) for j in range(3)]
            s.poly(L, pts, v * 0.45, 1)
        for i in range(24):
            a = -t * 0.4 + i * 0.2618
            s.line(L, (c[0] + math.cos(a) * r * 0.9, c[1] + math.sin(a) * r * 0.9), (c[0] + math.cos(a) * r * 0.96, c[1] + math.sin(a) * r * 0.96), v * 0.7, 1)

    # ---- 属性ごとの大技のしるし ----
    def motif(s, f, c, Rr, p, t):
        e = s.elem; L = s.L0()
        # 黒い核 + ふちの光
        core = s.L0(); cv2.circle(core, (int(c[0]), int(c[1])), int(Rr), 1.0, -1, cv2.LINE_AA)
        f *= (1 - core * 0.93 * min(1, p * 3))[:, :, None]
        s.circ(L, c, Rr, 1.0, 5)
        q = min(1, max(0, (p - 0.25) / 0.45))        # 中の模様が出てくる度合い
        if e in ("dark", "poison", "earth"):
            for a, r0, a2, ln in s.cracks:
                if r0 > q * 1.2: continue
                x, y = c[0] + math.cos(a) * Rr * r0 * 0.92, c[1] + math.sin(a) * Rr * r0 * 0.92
                x2, y2 = x + math.cos(a2) * Rr * ln, y + math.sin(a2) * Rr * ln
                if (x2 - c[0]) ** 2 + (y2 - c[1]) ** 2 < (Rr * 0.97) ** 2: s.line(L, (x, y), (x2, y2), 0.55, 1)
        if e == "dark":
            o = min(1, max(0, (p - 0.45) / 0.25))
            if o > 0:
                for wd, v in ((0.36, 1.0), (0.2, 0.8)):
                    s.poly(L, s.lens((c[0], c[1] - Rr * 0.72), math.pi / 2, Rr * 1.44, Rr * wd * o), v, 3)
                s.poly(L, s.lens((c[0], c[1] - Rr * 0.6), math.pi / 2, Rr * 1.2, Rr * 0.07 * o), 1.0, 2, True)
        elif e == "fire":
            for i in range(16):
                a = i * 0.3927 + t * 0.5; ln = Rr * (0.35 + 0.25 * math.sin(t * 7 + i * 1.7)) * q
                s.poly(L, s.lens((c[0] + math.cos(a) * Rr, c[1] + math.sin(a) * Rr), a + 0.35, ln, Rr * 0.09), 0.9, 2, True)
            for i in range(3):
                s.ell(L, c, (Rr * (0.75 - i * 0.2), Rr * (0.75 - i * 0.2)), math.degrees(t * (2 + i)), 0.8, 3, 0, int(250 * q))
            s.poly(L, s.lens((c[0], c[1] + Rr * 0.55), -math.pi / 2, Rr * 1.1 * q, Rr * 0.28 * q), 1.0, 3)
        elif e == "thunder":
            rg = np.random.RandomState(int(t * 12))
            for i in range(int(7 * q)):
                a = i * 0.9 + rg.rand(); pts = [c]
                for j in range(1, 7):
                    rr = Rr * 1.6 * j / 6; pts.append((c[0] + math.cos(a + (rg.rand() - 0.5) * 0.5) * rr, c[1] + math.sin(a + (rg.rand() - 0.5) * 0.5) * rr))
                for j in range(6): s.line(L, pts[j], pts[j + 1], 1.0, 3)
            s.circ(L, c, Rr * 0.3 * q, 1.0, 4)
        elif e == "ice":
            for i in range(6):
                a = i * 1.0472 + t * 0.15; ex = (c[0] + math.cos(a) * Rr * 0.92 * q, c[1] + math.sin(a) * Rr * 0.92 * q)
                s.line(L, c, ex, 1.0, 3)
                for u in (0.45, 0.7):
                    bx, by = c[0] + math.cos(a) * Rr * u * q, c[1] + math.sin(a) * Rr * u * q
                    for sg in (-1, 1):
                        s.line(L, (bx, by), (bx + math.cos(a + sg * 1.0) * Rr * 0.22 * q, by + math.sin(a + sg * 1.0) * Rr * 0.22 * q), 0.9, 2)
            s.poly(L, [(c[0] + math.cos(i * 1.0472 + 0.52) * Rr * 0.3 * q, c[1] + math.sin(i * 1.0472 + 0.52) * Rr * 0.3 * q) for i in range(6)], 0.9, 2)
        elif e == "light":
            for i in range(8):
                a = i * 0.7854 + t * 0.2; ln = Rr * (0.9 if i % 2 == 0 else 0.55) * q
                s.poly(L, s.lens(c, a, ln, Rr * 0.07), 1.0, 2, True)
            s.circ(L, c, Rr * 0.62 * q, 0.8, 2); s.circ(L, c, Rr * 0.3 * q, 1.0, 3)
        elif e == "star":
            for k, (n, rr, sp) in enumerate(((5, 0.9, 0.5), (6, 0.55, -0.8))):
                pts = [(c[0] + math.cos(t * sp + i * 6.283 / n) * Rr * rr * q, c[1] + math.sin(t * sp + i * 6.283 / n) * Rr * rr * q) for i in range(n)]
                st = 2 if n == 5 else 2
                for i in range(n): s.line(L, pts[i], pts[(i + st) % n], 0.95, 2)
            for i in range(10):
                a = t * 1.2 + i * 0.6283; s.circ(L, (c[0] + math.cos(a) * Rr * 0.75, c[1] + math.sin(a) * Rr * 0.75), 2 * s.k, 1.0, 2)
        elif e == "water":
            for i in range(5):
                rr = ((t * 0.5 + i * 0.2) % 1.0) * Rr * 0.95 * q
                s.circ(L, c, rr, 1.0 - rr / (Rr + 1) * 0.6, 3)
            s.poly(L, s.lens((c[0], c[1] - Rr * 0.5 * q), math.pi / 2, Rr * 0.9 * q, Rr * 0.26 * q), 1.0, 3)
        elif e == "wind":
            for i in range(3):
                pts = []
                for j in range(40):
                    u = j / 39; a = i * 2.094 + t * 3 + u * 4.2; pts.append((c[0] + math.cos(a) * Rr * 0.93 * u * q, c[1] + math.sin(a) * Rr * 0.93 * u * q))
                for j in range(39): s.line(L, pts[j], pts[j + 1], 0.4 + 0.6 * j / 39, 3)
        elif e in ("leaf", "petal"):
            n = 6 if e == "leaf" else 8
            for k, (rr, wd, sp) in enumerate(((0.9, 0.2, 0.3), (0.55, 0.16, -0.5))):
                for i in range(n):
                    a = t * sp + i * 6.283 / n + k * 3.1416 / n
                    s.poly(L, s.lens(c, a, Rr * rr * q, Rr * wd * q), 0.95, 2)
            s.circ(L, c, Rr * 0.12 * q, 1.0, 3)
        elif e == "earth":
            s.poly(L, [(c[0] + math.cos(i * 1.0472 + t * 0.1) * Rr * 0.7 * q, c[1] + math.sin(i * 1.0472 + t * 0.1) * Rr * 0.7 * q) for i in range(6)], 1.0, 4)
            s.poly(L, [(c[0] + math.cos(i * 1.0472 + 0.52 - t * 0.1) * Rr * 0.38 * q, c[1] + math.sin(i * 1.0472 + 0.52 - t * 0.1) * Rr * 0.38 * q) for i in range(6)], 0.9, 3)
        elif e == "poison":
            for i in range(14):
                a, r0, a2, ln = s.cracks[i]; rr = Rr * 0.1 * (0.5 + ln) * q * (0.7 + 0.3 * math.sin(t * 5 + i))
                s.circ(L, (c[0] + math.cos(a + t * 0.3) * Rr * r0 * 0.75, c[1] + math.sin(a + t * 0.3) * Rr * r0 * 0.75), rr, 0.9, 2)
        # まわりを回る輪と、輪の上の光
        for k, (ang, sp) in enumerate(((-14, 1.1), (12, -0.8))):
            ax = (Rr * 1.55, Rr * 0.42)
            s.ell(L, c, ax, ang, 0.8 * min(1, p * 2.5), 2)
            for i in range(3):
                a = t * sp + i * 2.094 + k; ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
                x, y = math.cos(a) * ax[0], math.sin(a) * ax[1]
                px, py = c[0] + x * ca - y * sa, c[1] + x * sa + y * ca
                s.line(L, (px - 7 * s.k, py), (px + 7 * s.k, py), 1.0, 2); s.line(L, (px, py - 9 * s.k), (px, py + 9 * s.k), 1.0, 2)
        s.bloom(f, L, gain=min(1, p * 2.5))

    # ---- 1コマ ----
    def frame(s, t):
        W, H, k = s.W, s.H, s.k
        C = (W / 2, H * 0.40)
        if t < 3.75:                                   # 登場 → 魔法陣 → ため
            f = s.bg(t, min(1, t / 0.9))
            rev = min(1, max(0, (t - 0.5) / 1.3))      # 影 → 色
            sc = (0.62 + t * 0.012) * W / max(s.ch.shape[1], s.ch.shape[0] * 0.62)
            L = s.L0()
            if t > 1.5:
                gr = min(1, (t - 1.5) / 0.6)
                s.circle_magic(L, C, W * 0.47 * gr, t, 0.75)
                if t > 1.9: s.ell(L, C, (W * 0.62, W * 0.62), 0, 0.5, 1, int(t * 80) % 360, int(t * 80) % 360 + 200)
            s.bloom(f, L, gain=0.8)
            glow = s.L0(); cv2.circle(glow, (int(C[0]), int(C[1])), int(W * 0.36), 0.5 * min(1, t), -1)
            f += cv2.GaussianBlur(glow, (0, 0), 50 * k)[:, :, None] * s.col[None, None, :] * 0.8
            img = s.ch.copy(); img[:, :, :3] *= rev
            s.put(f, img, C[0], C[1], sc, min(1, t / 0.6))
            L = s.L0()
            if t > 2.0:                                # 火花と ための光
                pw = min(1, (t - 2.0) / 1.5)
                for a, r0, ph, sp in s.sparks:
                    u = ((t * sp * 0.9 + ph) % 1.0); rr = W * (0.15 + 0.9 * u) * (0.6 + r0 * 0.6)
                    ln = W * 0.06 * (0.4 + r0) * pw
                    x, y = C[0] + math.cos(a) * rr, C[1] + math.sin(a) * rr * 1.4
                    s.line(L, (x, y), (x + math.cos(a) * ln, y + math.sin(a) * ln * 1.4), 0.75 * pw * (1 - u), 1)
                cx, cy = C[0] - W * 0.03, C[1] + W * 0.03
                cv2.circle(L, (int(cx), int(cy)), int(W * 0.035 * pw), 1.0, -1, cv2.LINE_AA)
                for i in range(10):
                    a = i * 0.628 + t; ln = W * (0.1 + 0.25 * pw) * (0.5 + 0.5 * math.sin(t * 9 + i * 2))
                    s.line(L, (cx, cy), (cx + math.cos(a) * ln, cy + math.sin(a) * ln), 0.8 * pw, 1)
                s.line(L, (0, cy), (W, cy), 0.5 * pw, 1)
            s.bloom(f, L)
            return f
        if t < 6.05:                                   # 奥義カットイン
            u = t - 3.75
            f = s.bg(t, 0.25)
            ghost = s.ch.copy()
            s.put(f, ghost, W / 2, H * 0.62, 0.9 * W / s.ch.shape[1], 0.14)
            # 帯: 顔のあたりを大きく
            band = np.zeros((H, W, 3), np.float32); bm = s.L0()
            zoom = 1.55 * W / s.ch.shape[1]
            fy = s.ch.shape[0] * 0.33
            s.put(band, s.ch, W / 2 + (u - 1.1) * 14 * k, H * 0.385 + (s.ch.shape[0] / 2 - fy) * zoom, zoom, 1.0)
            bh = H * 0.105 * min(1, u / 0.18)
            sl = math.tan(math.radians(-6)) * W / 2
            pts = [(0, H * 0.385 - bh - sl), (W, H * 0.385 - bh + sl), (W, H * 0.385 + bh + sl), (0, H * 0.385 + bh - sl)]
            s.poly(bm, pts, 1.0, 1, True)
            tint = np.zeros((H, W, 3), np.float32) + s.col[None, None, :] * 0.42
            f = f * (1 - bm[:, :, None]) + (tint * (1 - np.clip(band.sum(2, keepdims=True) * 9, 0, 1)) + band * 0.96 + s.col[None, None, :] * 0.05) * bm[:, :, None]
            L = s.L0()
            s.line(L, pts[0], pts[1], 1.0, 3); s.line(L, pts[3], pts[2], 1.0, 3)
            for i in range(26):                        # 流れる線
                a, r0, ph, sp = s.sparks[i]; x = ((u * 1.6 * sp + ph) % 1.2 - 0.1) * W; y = r0 * H
                s.line(L, (x, y), (x + W * 0.08, y - W * 0.03), 0.3, 1)
            s.bloom(f, L, gain=0.9)
            s.text(f, s.txt_ougi, W / 2, H * 0.215, min(1, u / 0.25), 1.0)
            n = len(s.name); rv = min(1, max(0, (u - 0.25) / (0.14 * n)))
            s.text(f, s.txt_main, W / 2, H * 0.60, 1.0, rv)
            if s.txt_en is not None: s.text(f, s.txt_en, W / 2, H * 0.665, min(1, max(0, (u - 0.25 - 0.14 * n) / 0.3)), 1.0)
            return f
        if t < 6.45:                                   # 白い影のフラッシュ
            u = (t - 6.05) / 0.4
            f = np.zeros((H, W, 3), np.float32)
            L = s.L0()
            for i in range(60):
                a = i * 0.1047 + s.sparks[i][2]
                s.line(L, (C[0] + math.cos(a) * W * 0.5, C[1] + math.sin(a) * W * 0.5), (C[0] + math.cos(a) * W * 2, C[1] + math.sin(a) * W * 2), 0.8, 1 + i % 3)
            s.bloom(f, L, np.array([1, 1, 1], np.float32))
            w = s.ch.copy(); w[:, :, :3] = 1
            s.put(f, w, C[0], C[1], (0.95 + u * 0.25) * W / max(s.ch.shape[1], s.ch.shape[0] * 0.62), 1.0, 0, 0)
            f += s.col[None, None, :] * 0.25 * (1 - u)
            return f + max(0, u - 0.7) * 3
        # ---- 大技 ----
        u = t - 6.45; p = min(1, u / 1.6)
        f = s.bg(t, 0.8)
        O = (W / 2, H * 0.29); Rr = W * 0.275
        Cb = (W / 2, H * 0.775)
        L = s.L0()
        for i in range(9):                             # 下から上への光
            x = Cb[0] + (i - 4) * W * 0.13
            s.line(L, (Cb[0] + (i - 4) * W * 0.015, Cb[1] + W * 0.05), (Cb[0] + (i - 4) * W * 0.24, H), 0.5, 2)
        s.ell(L, Cb, (W * 0.62, W * 0.14), 0, 0.6, 1)
        b2 = min(1, max(0, (u - 1.3) / 1.2))           # 点の波
        if b2 > 0:
            for a, r0 in s.dots:
                rr = W * (0.3 + 1.5 * ((r0 + u * 0.25) % 1.0)) * b2
                x, y = O[0] + math.cos(a) * rr, O[1] + math.sin(a) * rr
                if 0 <= x < W and 0 <= y < H: cv2.circle(L, (int(x), int(y)), max(1, int(2.2 * k)), 0.9, -1, cv2.LINE_AA)
        b3 = min(1, max(0, (u - 1.9) / 0.5))           # 光の十字と輪
        if b3 > 0:
            for i in range(6):
                a = i * 1.047 + 0.3 + u * 0.25
                s.line(L, (O[0] - math.cos(a) * W * 2, O[1] - math.sin(a) * W * 2), (O[0] + math.cos(a) * W * 2, O[1] + math.sin(a) * W * 2), 0.95 * b3 * (0.6 + 0.4 * math.sin(u * 20 + i)), 3)
            for i in range(3):
                rr = ((u * 0.7 + i * 0.33) % 1.0) * W * 1.3
                s.circ(L, O, rr, 0.8 * (1 - rr / (W * 1.3)), 3)
        s.bloom(f, L)
        s.put(f, s.ch, Cb[0], Cb[1], 0.46 * W / max(s.ch.shape[1], s.ch.shape[0] * 0.8), 1.0)
        s.motif(f, O, Rr * (0.4 + 0.6 * min(1, u / 0.5)), p, t)
        if u < 0.35: f = f * (u / 0.35) + 0.9 * (1 - u / 0.35)
        if u > 2.9:                                    # ふくらんで 白とび
            v = min(1, (u - 2.9) / 0.55)
            glow = s.L0(); cv2.circle(glow, (int(O[0]), int(O[1])), int(Rr * (0.3 + v * 5)), 1.0, -1)
            f += cv2.GaussianBlur(glow, (0, 0), 30 * k)[:, :, None] * (0.6 + 0.4 * s.col[None, None, :]) * v * 1.4
        if u > 3.5:
            v = min(1, (u - 3.5) / 0.75)
            f = f * (1 - v) ** 2
        return f

    def text(s, f, T, cx, cy, alpha, reveal):
        core, m, glow = T
        h, w = core.shape; x0, y0 = int(cx - w / 2), int(cy - h / 2)
        xa, ya, xb, yb = max(0, x0), max(0, y0), min(s.W, x0 + w), min(s.H, y0 + h)
        sl = (slice(ya - y0, yb - y0), slice(xa - x0, xb - x0))
        mask = np.zeros((1, w), np.float32); mask[:, :int(w * reveal)] = 1
        mask = mask[:, sl[1]]
        g = glow[sl] * mask * alpha; st = m[sl] * mask * alpha; c = core[sl] * mask * alpha
        reg = f[ya:yb, xa:xb]
        reg += g[:, :, None] * s.col[None, None, :] * 1.3
        reg *= (1 - st * 0.85)[:, :, None]
        reg += c[:, :, None]

    # ---- 音 ----
    def audio(s, path):
        sr = 44100; n = int(DUR * sr); t = np.arange(n) / sr; a = np.zeros(n, np.float32)
        base = {"fire": 55, "thunder": 62, "dark": 49, "poison": 49, "light": 65, "star": 73, "ice": 69, "water": 58, "wind": 65, "leaf": 62, "petal": 65, "earth": 46}[s.elem]
        def env(t0, at, dec): return np.clip((t - t0) / at, 0, 1) * np.exp(-np.clip(t - t0 - at, 0, None) / dec) * (t >= t0)
        rng = np.random.RandomState(5); nz = rng.randn(n).astype(np.float32)
        def lp(x, kk):
            from scipy.signal import lfilter
            return lfilter([kk], [1, kk - 1], x).astype(np.float32)
        # 低いうなり(だんだん上がる)
        fr = base * (1 + 0.5 * np.clip(t / 6, 0, 1) ** 2)
        ph = 2 * np.pi * np.cumsum(fr) / sr
        a += (np.sin(ph) + 0.4 * np.sin(ph * 2.01) + 0.25 * np.sin(ph * 3.02)) * 0.22 * np.clip(t / 1.2, 0, 1) * (t < 6.05)
        # ため: 上がっていく風
        a += lp(nz, 0.25) * 0.5 * np.clip((t - 1.8) / 1.9, 0, 1) ** 2 * (t < 3.75)
        # カットインの一撃 と 文字
        for t0, g in ((3.75, 0.9), (6.05, 1.0), (6.45, 0.9), (8.35, 0.8), (9.35, 1.0)):
            a += np.sin(2 * np.pi * (base * 0.9) * (t - t0) * np.exp(-(np.clip(t - t0, 0, None)) * 2.5)) * env(t0, 0.004, 0.35) * g
            a += lp(nz, 0.5) * env(t0, 0.003, 0.22) * 0.5 * g
        for i in range(len(s.name)):
            t0 = 4.0 + i * 0.14; a += np.sin(2 * np.pi * (base * 8 * (1 + i * 0.06)) * t) * env(t0, 0.003, 0.07) * 0.18
        # 大技: 和音が広がる
        for j, m in enumerate((1, 1.5, 2, 3, 4)):
            a += np.sin(2 * np.pi * base * 2 * m * t + j) * env(6.5 + j * 0.25, 0.5, 2.2) * 0.1
        a += lp(nz, 0.12) * 0.35 * np.clip((t - 9.2) / 0.5, 0, 1) * np.exp(-np.clip(t - 9.9, 0, None) * 3)
        a *= np.clip((DUR - t) / 0.5, 0, 1)
        a = np.tanh(a * 1.1) * 0.85
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr); w.writeframes((a * 32767).astype(np.int16).tobytes())


def main():
    cid, elem, name, en, out = int(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
    W = int(sys.argv[6]) if len(sys.argv) > 6 else 540
    r = R(cid, elem, name, en, W)
    wav = out + ".wav"; r.audio(wav)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (r.W, r.H), "-r", str(FPS), "-i", "-",
           "-i", wav, "-c:v", "libx264", "-preset", "medium", "-crf", "27", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k", "-movflags", "+faststart", "-shortest", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(int(DUR * FPS)):
        f = r.frame(i / FPS)
        p.stdin.write((np.clip(f, 0, 1) * 255).astype(np.uint8).tobytes())
    p.stdin.close(); p.wait(); os.remove(wav)
    print(out, os.path.getsize(out) // 1024, "KB")


if __name__ == "__main__":
    main()
