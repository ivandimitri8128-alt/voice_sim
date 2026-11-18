import numpy as np
from scipy.io import wavfile
from score import parse_score
from sing_v2 import plan_song, build_segments, estimate_f0, timing_rng, cons_rng

fs, y = wavfile.read("output/bohongi_hati_v27.wav")
y = y.astype(float) / 32767
if y.ndim == 2:
    y = y.mean(axis=1)
notes, cfg = parse_score(open("songs/bohongi_hati.txt", encoding="utf-8").read())
events, marks, phrases, total = plan_song(notes, cfg["tempo"], rng=timing_rng(1),
                                          beats_per_bar=cfg["meter"], pickup=cfg["pickup"],
                                          bar_reset=bool(cfg["barreset"]))
tt, ff = estimate_f0(y, fs, win=0.045, hop=0.01, fmin=70, fmax=900)
rows = []
for (ta, ma), nxt in zip(marks, marks[1:] + [(total, None)]):
    a, b = ta + 0.12, nxt[0] - 0.06
    if b - a < 0.08:
        continue
    sel = (tt > a) & (tt < b)
    f = ff[sel]
    f = f[~np.isnan(f)]
    if len(f):
        tgt = 440 * 2 ** ((ma - 69) / 12)
        rows.append((abs(1200 * np.log2(np.median(f) / tgt)), ta, ma, np.median(f), len(f)))
rows.sort(reverse=True)
for r in rows[:6]:
    print(f"err {r[0]:7.0f} cents  t={r[1]:6.2f}s midi={r[2]:.0f} f_med={r[3]:.1f} frames={r[4]}")
