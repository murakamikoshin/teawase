"""タイトル画面の 背景ムービー(つなぎ目なしで くり返す 8秒)。星雲・金の 光の すじ・まわる 紋章の 輪・星・立ちのぼる 光の つぶ・舞台の 床。
使い方: python3 title_bg.py 出力.mp4 / python3 title_bg.py still 出力.png 時刻,時刻"""
import sys, math, subprocess
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
from hs_fx import FX, sstep

P = 8.0; FPS = 30
GOLD = np.array([1.0, 0.78, 0.30], np.float32); VIO = np.array([0.45, 0.28, 1.0], np.float32); PINK = np.array([1.0, 0.45, 0.75], np.float32); CYAN = np.array([0.4, 0.85, 1.0], np.float32); CREAM = np.array([1.0, 0.94, 0.78], np.float32)


class TitleBG(FX):
    def __init__(s, W=540):
        s.W = W; s.H = W * 16 // 9; s.k = W / 540.0
        s.YY, s.XX = np.mgrid[0:s.H, 0:s.W].astype(np.float32); s.fx_init(77)
        s.xn = s.XX / s.W; s.yn = s.YY / s.H
        rg = np.random.RandomState(5); s.st = rg.rand(520, 5).astype(np.float32); s.sp = rg.rand(260, 6).astype(np.float32); s.bk = rg.rand(34, 6).astype(np.float32)

    def tex(s, i, u, v):
        u = np.broadcast_to(np.asarray(u, np.float32), (s.H, s.W)); v = np.broadcast_to(np.asarray(v, np.float32), (s.H, s.W))
        return cv2.remap(s._tiles[i], np.ascontiguousarray(u % 256, np.float32), np.ascontiguousarray(v % 256, np.float32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

    def frame(s, t):
        W, H, k = s.W, s.H, s.k; ph = 2 * math.pi * t / P; q = t / P; C = (W * 0.5, H * 0.585); hy = 0.800
        # 空: 深い むらさきの 星雲
        f = np.array([0.020, 0.015, 0.070], np.float32)[None, None, :] * (0.6 + 1.2 * np.exp(-((s.yn - 0.55) / 0.35) ** 2))[:, :, None]
        n = s.fbm(0.0007, 150 * math.cos(ph), 100 * math.sin(ph), 4, 60 * k, 0); m = s.fbm(0.0016, -90 * math.cos(ph), 120 * math.sin(ph), 3, 0, 2)
        a = sstep(0.36, 0.80, n); col = np.array([0.10, 0.06, 0.34], np.float32)[None, None, :] * (1 - m)[:, :, None] + np.array([0.36, 0.14, 0.62], np.float32)[None, None, :] * m[:, :, None]
        hot = sstep(0.68, 0.95, n)[:, :, None]; col = col * (1 - hot) + np.array([0.95, 0.50, 0.70], np.float32)[None, None, :] * hot
        f += col * a[:, :, None] * 0.9
        # 星(またたく)
        st = s.st; tw = 0.35 + 0.65 * np.sin(ph * np.floor(1 + st[:, 3] * 4) + st[:, 4] * 6.28) ** 2
        s.splat(f, st[:, 0] * W, st[:, 1] * H * hy, st[:, 2] ** 3 * tw * 0.9, (0.85, 0.9, 1.0), 0.9, 0.35)
        # まん中の 光と まわる 光の すじ
        dx = s.XX - C[0]; dy = s.YY - C[1]; r = np.hypot(dx, dy) + 1e-3; th = np.arctan2(dy, dx)
        r1 = s.tex(1, th / 6.2832 * 768 + 256 * q, r * 0.05) ** 2; r2 = s.tex(3, th / 6.2832 * 512 - 256 * q, r * 0.03 + 40) ** 2
        ray = (r1 * 0.75 + r2 * 0.6) * np.exp(-r / (H * 0.50)) * sstep(20 * k, 140 * k, r)
        f += (ray * 1.05)[:, :, None] * (GOLD * 0.8 + PINK * 0.2)[None, None, :]
        f += (np.exp(-(r / (W * 0.42)) ** 2) * 0.34)[:, :, None] * (VIO * 0.6 + PINK * 0.4)[None, None, :] + (np.exp(-(r / (W * 0.22)) ** 2) * 1.0)[:, :, None] * (GOLD * 0.8 + 0.2)[None, None, :]
        # 紋章の 輪(12の 目もり・六つ星・まわる ひし形)
        L = np.zeros((H, W), np.float32); R = W * 0.455; ip = lambda x, y: (int(round(C[0] + x)), int(round(C[1] + y))); rot = 2 * math.pi / 12 * q
        cv2.circle(L, ip(0, 0), int(R), 1.0, max(1, int(2 * k)), cv2.LINE_AA); cv2.circle(L, ip(0, 0), int(R * 0.94), 0.6, 1, cv2.LINE_AA); cv2.circle(L, ip(0, 0), int(R * 0.60), 0.7, max(1, int(1.5 * k)), cv2.LINE_AA); cv2.circle(L, ip(0, 0), int(R * 1.12), 0.35, 1, cv2.LINE_AA)
        for i in range(48):
            a_ = rot + i * 6.2832 / 48; big = i % 4 == 0; r0 = R * (0.86 if big else 0.90)
            cv2.line(L, ip(math.cos(a_) * r0, math.sin(a_) * r0), ip(math.cos(a_) * R * 0.94, math.sin(a_) * R * 0.94), 0.9 if big else 0.5, max(1, int((2 if big else 1) * k)), cv2.LINE_AA)
        for i in range(12):                                                          # 外を まわる ひし形
            a_ = -rot * 2 + i * 6.2832 / 12; c = (math.cos(a_) * R * 1.03, math.sin(a_) * R * 1.03); d = 7 * k
            cv2.fillPoly(L, [np.array([ip(c[0] + math.cos(a_) * d, c[1] + math.sin(a_) * d), ip(c[0] - math.sin(a_) * d * 0.5, c[1] + math.cos(a_) * d * 0.5), ip(c[0] - math.cos(a_) * d, c[1] - math.sin(a_) * d), ip(c[0] + math.sin(a_) * d * 0.5, c[1] - math.cos(a_) * d * 0.5)], np.int32)], 1.0, cv2.LINE_AA)
        for sg, r3 in ((1, 0.86), (-1, 0.86)):                                       # 六つ星(三角 2つが 逆に まわる)
            pts = [ip(math.cos(sg * 2 * math.pi / 3 * q + j * 2.0944 + (0 if sg > 0 else 1.0472)) * R * r3, math.sin(sg * 2 * math.pi / 3 * q + j * 2.0944 + (0 if sg > 0 else 1.0472)) * R * r3) for j in range(3)]
            cv2.polylines(L, [np.array(pts, np.int32)], True, 0.55, max(1, int(1.5 * k)), cv2.LINE_AA)
        L *= (0.75 + 0.25 * np.sin(th * 6 + ph * 2)) * sstep(H * 0.86, H * 0.74, s.YY)
        s.glowmask(f, L, GOLD * 0.85 + 0.15, 0.50, 1.0, 1.0, 9.0)
        # ななめに 走る 光
        for off, v in ((0.0, 0.07), (0.5, 0.05)):
            b = (s.xn * 0.8 + s.yn * 0.45 + q * 2 + off) % 1.0; f += (np.exp(-((b - 0.5) / 0.035) ** 2) * v)[:, :, None] * CREAM[None, None, :]
        # 床: 舞台の てり返し
        up = f[int(H * (2 * hy - 1)):int(H * hy)][::-1]; fl = cv2.GaussianBlur(up, (0, 0), 9 * k); y0 = int(H * hy); hh = min(H - y0, fl.shape[0])
        d = np.clip((s.yn[y0:y0 + hh] - hy) / (1 - hy), 0, 1); fx_ = dx[y0:y0 + hh] / (dy[y0:y0 + hh] + H * 0.03)
        lines = sstep(0.95, 0.995, np.abs(np.cos(fx_ * 9.0))) * 0.16 * (1 - d) + sstep(0.93, 0.995, np.abs(np.cos(18.0 / (d + 0.12) + ph))) * 0.14 * (1 - d)
        f[y0:y0 + hh] = fl[:hh] * (0.50 * (1 - d) ** 1.2)[:, :, None] + np.array([0.03, 0.02, 0.08], np.float32)[None, None, :] + lines[:, :, None] * (GOLD * 0.7 + VIO * 0.3)[None, None, :]
        f += (np.exp(-((s.yn - hy) / 0.004) ** 2) * (0.25 + 0.9 * np.exp(-((s.xn - 0.5) / 0.30) ** 2)))[:, :, None] * (GOLD * 0.8 + 0.2)[None, None, :]
        # 立ちのぼる 光の つぶ(小さい つぶ + 大きい ぼけ玉)
        sp = s.sp; x = sp[:, 0] * W + np.sin(ph * np.floor(1 + sp[:, 3] * 3) + sp[:, 4] * 6.28) * 16 * k; y = (sp[:, 1] * H - q * H * np.floor(1 + sp[:, 2] * 2)) % H
        br = (0.3 + sp[:, 5]) * (0.5 + 0.5 * np.sin(ph * np.floor(2 + sp[:, 4] * 4) + sp[:, 3] * 9) ** 2) * np.clip(1.6 - np.hypot(x - C[0], (y - C[1]) * 0.7) / (W * 0.75), 0.15, 1.3)
        s.splat(f, x, y, br * 1.1, GOLD * 0.8 + 0.2, 1.0, 0.6)
        B = np.zeros((H, W, 3), np.float32)
        for i in range(len(s.bk)):
            b = s.bk[i]; bx = b[0] * W + math.sin(ph * (1 + int(b[3] * 2)) + b[4] * 6.28) * 24 * k; by = (b[1] * H - q * H * (1 + int(b[2] * 1.99))) % H; rr = int((7 + 18 * b[5]) * k)
            c_ = (GOLD, PINK, CYAN, CREAM)[i % 4] * (0.10 + 0.12 * math.sin(ph * (1 + i % 3) + i) ** 2)
            for dy_ in (0, -H, H): cv2.circle(B, (int(bx), int(by + dy_)), rr, tuple(float(v_) for v_ in c_), -1, cv2.LINE_AA)
        f += cv2.GaussianBlur(B, (0, 0), 2.5 * k)
        f = s.godrays(f, C, 0.30, 0.9, 10, 0.88, 0.45)
        f = f * (0.50 + 0.50 * sstep(0.03, 0.34, s.yn))[:, :, None]
        return s.film(f, (int(round(t * FPS)) % 8) / FPS, 0.0, 0.55, 0.80, (1.0, 0.9, 0.8), 1.0, 0.022, 0.55, 1.15, 1.05)


if __name__ == "__main__":
    r = TitleBG()
    if sys.argv[1] == "still":
        fr = [(np.clip(r.frame(float(x)), 0, 1) * 255).astype(np.uint8) for x in sys.argv[3].split(",")]; cv2.imwrite(sys.argv[2], np.concatenate(fr, 1)[:, :, ::-1])
    else:
        p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', '%dx%d' % (r.W, r.H), '-r', str(FPS), '-i', '-', '-an', '-c:v', 'libx264', '-crf', '23', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', sys.argv[1]], stdin=subprocess.PIPE)
        for i in range(int(P * FPS)): p.stdin.write((np.clip(r.frame(i / FPS), 0, 1) * 255).astype(np.uint8).tobytes())
        p.stdin.close(); p.wait(); print('done')
