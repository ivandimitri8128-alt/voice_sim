"""Diagnosa lengkap: ug osilasi vs ps/q/x0/vokal aktual."""
import numpy as np

from score import load_score
from timeline import synthesize_song
import timeline
from vowels import VOWELS

FS = 44100
notes, cfg = load_score("songs/burung_kakatua.txt")
p, ug, ctl, f0 = synthesize_song(notes, tempo=104, q_base=1.25)
ps, q, x0, f0 = timeline._dbg

# identifikasi vokal dominan: bandingkan area rata-rata segmen dgn tabel vokal
w = int(0.5 * FS)
ug = np.asarray(ug)
print(" t  | osc_ug | ps  | q    | x0mm | f0t | vokal")
for s in range(0, len(ug) - w, w):
    osc = np.std(ug[s:s + w])
    seg_v = ctl.voiced[s:s + w].mean()
    if seg_v < 0.3:
        continue
    i = s + w // 2
    a = ctl.a_cur[i]
    best, bv = None, 1e9
    for k, tab in VOWELS.items():
        d = float(np.mean((a - np.asarray(tab)) ** 2))
        if d < bv:
            bv, best = d, k
    print(f"{s/FS:4.1f} | {osc:.1e} | {ps[i]:5.0f} | {q[i]:.2f} | {x0[i]*1e4:.2f} | {f0[i]:3.0f} | {best}")
