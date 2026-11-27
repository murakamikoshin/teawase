"""専用の必殺技動画のための「道具」だけ。見た目の演出(場面・大道具・流れ・曲)は 各キャラの hs_<id>.py に その子専用で書く。
ここにあるのは: キャラの絵を動かす(ゆれ・ため・ふるえ・部分のび)、絵の中の位置(目・手・武器)を画面の位置に直す、にじみ、音の素、書き出し。"""
import sys, os, math, subprocess, wave
sys.path.insert(0, '/home/claude/tools')
import numpy as np, cv2
import hissatsu as hz
from scipy.signal import lfilter

FPS = 30
WHITE = np.array([1, 1, 1], np.float32)


def ease(x): x = min(1.0, max(0.0, x)); return 1 - (1 - x) ** 3
def ease2(x): x = min(1.0, max(0.0, x)); return x * x * (3 - 2 * x)
def clamp(x, a=0.0, b=1.0): return min(b, max(a, x))
def back(x, s=1.7): x = min(1.0, max(0.0, x)) - 1; return x * x * ((s + 1) * x + s) + 1      # 行きすぎて もどる


class Kit(hz.R):
    PAD = 70

    def __init__(s, cid, col, name, en, W=540):
        hz.COL["_k"] = tuple(int(c * 255) for c in col)
        super().__init__(cid, "_k", name, en, W)
        p = int(s.PAD * s.k); s.pad = p
        s.cp = cv2.copyMakeBorder(s.ch, p, p, p, p, cv2.BORDER_CONSTANT, value=(0, 0, 0, 0))
        s.ah, s.aw = s.ch.shape[:2]                      # 絵の大きさ(位置あわせの基準)
        s.ph, s.pw = s.cp.shape[:2]
        s.gy, s.gx = np.mgrid[0:s.ph, 0:s.pw].astype(np.float32)
        s.rnd = np.random.RandomState(cid * 131 + 9).rand(400, 6).astype(np.float32)
        yy, xx = np.mgrid[0:s.H, 0:s.W].astype(np.float32)
        e = np.sqrt(((xx - s.W / 2) / (s.W * 0.75)) ** 2 + ((yy - s.H / 2) / (s.H * 0.62)) ** 2)
        s.vig = np.clip(1.08 - 0.5 * e ** 2.4, 0.2, 1.08).astype(np.float32)
        s.XX, s.YY = xx, yy

    def A(s, x, y):
        """760px高の絵で測った位置 → いまの絵の位置"""
        return (x * s.ah / 760.0, y * s.ah / 760.0)

    # ---- キャラの絵を動かす ----
    def pose(s, t, sway=0.0, breath=0.0, lean=0.0, squash=0.0, stretch=None, jit=0.0):
        """sway: 上ほど大きく ゆらゆら / breath: 息 / lean: 上体を かたむける(px) / squash: つぶれ(+)のび(-) /
        stretch: (中心x, 内側, 外側, 上限y, 量) 中心から外の部分だけ 横にのばす(翼・マントなど) / jit: 細かいふるえ(px)"""
        p = s.pad; v = np.clip(1 - (s.gy - p) / s.ah, 0, 1.2)
        mx = s.gx - (lean * v + sway * np.sin(t * 2.3 + s.gy * 0.011) * v * 13 * s.k)
        my = s.gy + breath * math.sin(t * 2.7) * v * 7 * s.k
        if squash:
            foot = p + s.ah; my = foot + (my - foot) / (1 - squash); cxm = s.pw / 2; mx = cxm + (mx - cxm) / (1 + squash * 0.6)
        if stretch:
            cx0, r0, r1, ymax, amt = stretch; cx0 += p
            d = mx - cx0; w = np.clip((np.abs(d) - r0) / (r1 - r0), 0, 1) * np.clip((ymax + p - s.gy) / 80.0, 0, 1)
            mx = cx0 + d / (1 + amt * w)
        if jit:
            mx = mx + math.sin(t * 91) * jit; my = my + math.cos(t * 77) * jit
        s._pose = dict(t=t, sway=sway, breath=breath, lean=lean, squash=squash, stretch=stretch)
        return cv2.remap(s.cp, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    def draw(s, f, img, cx, cy, hfrac, alpha=1.0, white=0.0, add=0.0, light=None):
        """絵を 画面の(cx,cy)中心・高さ hfrac*H で置く。light: (1ch の明るさマスク 絵と同じ大きさ) で絵に光を当てる"""
        sc = hfrac * s.H / s.ah
        if light is not None:
            img = img.copy(); img[:, :, :3] *= light[:, :, None]
        s.put(f, img, cx, cy, sc, alpha, add, white)
        s._pl = (cx, cy, sc)

    def at(s, ax, ay):
        """絵の中の点(いまの絵の座標) → 画面の位置(直前の pose/draw に合わせる)"""
        cx, cy, sc = s._pl; q = s._pose; v = clamp(1 - ay / s.ah, 0, 1.2)
        x = ax + q["lean"] * v + q["sway"] * math.sin(q["t"] * 2.3 + (ay + s.pad) * 0.011) * v * 13 * s.k
        y = ay - q["breath"] * math.sin(q["t"] * 2.7) * v * 7 * s.k
        if q["squash"]: y = s.ah + (y - s.ah) * (1 - q["squash"]); x = s.aw / 2 + (x - s.aw / 2) * (1 + q["squash"] * 0.6)
        if q["stretch"]:
            cx0, r0, r1, ymax, amt = q["stretch"]; d = x - cx0
            w = clamp((abs(d) - r0) / (r1 - r0)) * clamp((ymax - ay) / 80.0); x = cx0 + d * (1 + amt * w)
        return (cx + (x - s.aw / 2) * sc, cy + (y - s.ah / 2) * sc)

    # ---- 光 ----
    def soft(s, f, c, r, v, col):
        g = s.L0(); cv2.circle(g, (int(c[0]), int(c[1])), max(1, int(r)), float(v), -1)
        f += cv2.GaussianBlur(g, (0, 0), max(2, r * 0.5))[:, :, None] * np.asarray(col, np.float32)[None, None, :]

    def add(s, f, L, col, gain=1.0, b1=3, b2=12):
        L1 = cv2.GaussianBlur(L, (0, 0), b1 * s.k); L2 = cv2.GaussianBlur(L, (0, 0), b2 * s.k)
        f += ((L * 0.9 + L1 * 0.9 + L2 * 1.1) * gain)[:, :, None] * np.asarray(col, np.float32)[None, None, :] + (np.clip(L, 0, 1) ** 2 * 0.8 * gain)[:, :, None]

    def star4(s, L, c, r, v=1.0, rot=0.0, thin=0.16):
        pts = []
        for i in range(8):
            a = rot + i * 0.7854; rr = r if i % 2 == 0 else r * thin
            pts.append((c[0] + math.cos(a) * rr, c[1] + math.sin(a) * rr))
        cv2.fillPoly(L, [np.array(pts, np.int32)], float(v), cv2.LINE_AA)

    def post(s, f, col, gain=1.0, shake=0.0, t=0.0, thr=0.72):
        br = np.clip(f.max(2) - thr, 0, 1.5)
        b = cv2.GaussianBlur(br, (0, 0), 20 * s.k) + cv2.GaussianBlur(br, (0, 0), 6 * s.k) * 0.5
        f = (f + b[:, :, None] * (np.asarray(col, np.float32) * 0.85 + 0.15)[None, None, :] * gain) * s.vig[:, :, None]
        if shake > 0.3:
            M = np.float32([[1, 0, math.sin(t * 113) * shake], [0, 1, math.cos(t * 97) * shake]])
            f = cv2.warpAffine(f, M, (s.W, s.H), borderMode=cv2.BORDER_REFLECT)
        return f


class Snd:
    """音の素。曲そのものは 各キャラの台本で書く"""
    def __init__(s, dur, seed=1):
        s.sr = 44100; s.D = dur; s.n = int(dur * s.sr); s.t = np.arange(s.n) / s.sr; s.a = np.zeros(s.n, np.float32)
        s.nz = np.random.RandomState(seed).randn(s.n).astype(np.float32)
    def lp(s, x, k): return lfilter([k], [1, k - 1], x).astype(np.float32)
    def env(s, t0, at, dec): t = s.t; return np.clip((t - t0) / at, 0, 1) * np.exp(-np.clip(t - t0 - at, 0, None) / dec) * (t >= t0)
    def gate(s, a, b, fade=0.05): t = s.t; return np.clip((t - a) / fade, 0, 1) * np.clip((b - t) / fade, 0, 1)
    def tone(s, f, ph=0.0): return np.sin(2 * np.pi * f * s.t + ph)
    def boom(s, t0, f=50, g=1.0, dec=0.4):
        t = s.t; s.a += (np.sin(2 * np.pi * f * (t - t0) * np.exp(-np.clip(t - t0, 0, None) * 2.4)) * s.env(t0, 0.004, dec) + s.lp(s.nz, 0.5) * s.env(t0, 0.003, dec * 0.6) * 0.5) * g
    def note(s, f, t0, dec=0.2, g=0.1, kind="bell", at=0.004):
        t = s.t; ph = 2 * np.pi * f * (t - t0); e = s.env(t0, at, dec) * g
        if kind == "bell": w = np.sin(ph) + 0.55 * np.sin(2.76 * ph) * np.exp(-np.clip(t - t0, 0, None) * 8) + 0.25 * np.sin(5.4 * ph) * np.exp(-np.clip(t - t0, 0, None) * 14)
        elif kind == "saw": w = sum(np.sin(k * ph) / k for k in range(1, 7))
        elif kind == "sq": w = sum(np.sin(k * ph) / k for k in (1, 3, 5, 7))
        elif kind == "fm": w = np.sin(ph + 2.6 * np.sin(2 * ph) * np.exp(-np.clip(t - t0, 0, None) * 5))
        elif kind == "pad": w = np.sin(ph) + 0.5 * np.sin(ph * 1.005 + 1) + 0.3 * np.sin(2 * ph)
        else: w = np.sin(ph)
        s.a += w * e
    def hat(s, t0, g=0.1, dec=0.025): s.a += (s.nz - s.lp(s.nz, 0.6)) * s.env(t0, 0.001, dec) * g
    def whoosh(s, a, b, g=0.3, k=0.2): s.a += s.lp(s.nz, k) * np.clip((s.t - a) / max(0.01, b - a), 0, 1) ** 2 * (s.t < b) * g
    def rise(s, a, b, f0, f1, g=0.15):
        u = np.clip((s.t - a) / (b - a), 0, 1); fr = f0 + (f1 - f0) * u ** 2; ph = 2 * np.pi * np.cumsum(fr) / s.sr
        s.a += np.sin(ph) * u * ((s.t >= a) & (s.t < b)) * g
    def save(s, path):
        a = s.a * np.clip((s.D - s.t) / 0.5, 0, 1); a = np.tanh(a * 1.1) * 0.85
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(s.sr); w.writeframes((a * 32767).astype(np.int16).tobytes())


def encode(r, dur, out, crf=25):
    """r.frame(t) と r.audio(path) を持つものを mp4 にする"""
    wav = out + ".wav"; r.audio(wav)
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "%dx%d" % (r.W, r.H), "-r", str(FPS), "-i", "-", "-i", wav,
           "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "80k", "-movflags", "+faststart", "-shortest", out]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for i in range(int(dur * FPS)):
        p.stdin.write((np.clip(r.frame(i / FPS), 0, 1) * 255).astype(np.uint8).tobytes())
    p.stdin.close(); p.wait(); os.remove(wav); print(out, os.path.getsize(out) // 1024, "KB", flush=True)


def stills(r, times, out, w=180):
    """確認用: 指定の時刻のコマを 横に並べた画像"""
    fr = [cv2.resize((np.clip(r.frame(t), 0, 1) * 255).astype(np.uint8), (w, w * 16 // 9)) for t in times]
    cv2.imwrite(out, np.concatenate(fr, 1)[:, :, ::-1])
