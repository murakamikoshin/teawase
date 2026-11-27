"""元絵の色を数えて(16刻み)、素材ごとの色の表(settings.json の palette)を作る。ボスの README 2-b の書き方どおり"""
import sys, json, colorsys
from collections import Counter, defaultdict, deque
from PIL import Image
src, out, H = sys.argv[1], sys.argv[2], int(sys.argv[3])
im = Image.open(src).convert('RGB'); im.thumbnail((600, 600), Image.NEAREST); w, h = im.size; p = im.load()
def white(c): return min(c) > 236 and max(c) - min(c) < 18
bg = bytearray(w * h); q = deque()
for x in range(w): q.extend(((x, 0), (x, h - 1)))
for y in range(h): q.extend(((0, y), (w - 1, y)))
while q:
    x, y = q.popleft()
    if not (0 <= x < w and 0 <= y < h) or bg[y * w + x] or not white(p[x, y]): continue
    bg[y * w + x] = 1; q.extend(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))
cnt = Counter()
for y in range(h):
    for x in range(w):
        if not bg[y * w + x]:
            r, g, b = p[x, y]; cnt[(r // 16 * 16 + 8, g // 16 * 16 + 8, b // 16 * 16 + 8)] += 1
tot = sum(cnt.values())
bins = [(c, n) for c, n in cnt.most_common() if n >= max(3, tot * 0.0006)]
def hsv(c): return colorsys.rgb_to_hsv(c[0] / 255, c[1] / 255, c[2] / 255)
def group(c):
    hh, s, v = hsv(c); d = hh * 360
    if v < 0.27 or (v < 0.36 and s < 0.35): return 'black'
    if s < 0.13: return 'white' if v > 0.62 else 'steel'
    if 10 <= d <= 45 and s < 0.36 and v > 0.78: return 'skin'
    if s < 0.24 and v > 0.86: return 'white'
    names = ['red', 'orange', 'yellow', 'lime', 'green', 'teal', 'cyan', 'azure', 'blue', 'violet', 'magenta', 'rose']
    return names[int(((d + 15) % 360) // 30)]
G = defaultdict(list)
for c, n in bins: G[group(c)].append((c, n))
def mean(lst):
    t = sum(n for _, n in lst); return tuple(sum(c[i] * n for c, n in lst) / t for i in range(3))
# 小さい素材は、色が近い素材にまとめる(黒・白・肌は残す)
keep = {'black', 'white', 'skin'}
big = [k for k in G if k in keep or sum(n for _, n in G[k]) >= tot * 0.012]
for k in list(G):
    if k in big: continue
    m = mean(G[k]); cand = [b for b in big if b not in ('black', 'white')] or big
    tgt = min(cand, key=lambda b: sum((mean(G[b])[i] - m[i]) ** 2 for i in range(3)))
    G[tgt] += G.pop(k)
def boost(c, ds=1.12):
    hh, s, v = colorsys.rgb_to_hsv(*[t / 255 for t in c]); return [round(t * 255) for t in colorsys.hsv_to_rgb(hh, min(1, s * ds), v)]
pal = {}
for k, lst in G.items():
    if k == 'black':
        pal[k] = {'weight': 0.25, 'edge': 2, 'tones': [[70, 52, 60], [46, 34, 44], [100, 80, 86]],
                  'anchors': [list(c) + [1 if hsv(c)[2] < 0.12 else (0 if hsv(c)[2] < 0.22 else 2)] for c, _ in lst]}
        continue
    two = k in ('skin', 'white')
    lst = sorted(lst, key=lambda t: hsv(t[0])[2]); t = sum(n for _, n in lst); acc = 0; med = hsv(lst[-1][0])[2]
    for c, n in lst:
        acc += n
        if acc >= t / 2: med = hsv(c)[2]; break
    lv = {}
    for c, n in lst:
        v = hsv(c)[2]
        lv[c] = (1 if v < med - (0.05 if two else 0.09) else (2 if (v > med + 0.09 and not two) else 0))
    L = [[(c, n) for c, n in lst if lv[c] == i] for i in range(3)]
    base = mean(L[0]) if L[0] else mean(lst)
    dark = mean(L[1]) if L[1] else tuple(t * 0.72 for t in base)
    light = mean(L[2]) if L[2] else tuple(min(255, t * 1.14 + 14) for t in base)
    if k == 'white': tones = [[250, 250, 246], [190, 198, 222]]
    elif k == 'steel': tones = [[150, 160, 184], [96, 104, 132], [206, 214, 232]]
    elif two: tones = [boost(base, 1.0), boost(dark, 1.05)]
    else: tones = [boost(base), boost(dark), boost(light)]
    hh, s, v = hsv(base)
    wgt = 1.3 if k == 'skin' else (1.4 if (k in ('yellow', 'orange') and s > 0.5) else 1.0)
    e = {'weight': wgt, 'tones': tones, 'anchors': [list(c) + [lv[c]] for c, _ in lst]}
    if k == 'white': e['edge'] = 1
    pal[k] = e
pal['_fringe'] = [k for k in ('skin', 'white') if k in pal]
json.dump({'height': H, 'hole_min': 0, 'palette': pal}, open(out, 'w'), ensure_ascii=False)
print({k: round(sum(n for _, n in G[k]) / tot, 3) for k in G})
