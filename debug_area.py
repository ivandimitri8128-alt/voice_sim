"""Bandingkan profil luas saluran di titik mati vs hidup."""
import numpy as np

from score import load_score
from timeline import synthesize_song
import timeline
from vowels import VOWELS

FS = 44100
notes, cfg = load_score("songs/burung_kakatua.txt")
p, ug, ctl, f0 = synthesize_song(notes, tempo=104, q_base=1.25)
ps, q, x0, f0t = timeline._dbg
ug = np.asarray(ug)

for tsec in (13.5, 25.0, 9.0, 8.5):
    i = int(tsec * FS)
    a = ctl.a_cur[i]
    osc = np.std(ug[i:i + int(0.4 * FS)])
    print(f"t={tsec} osc={osc:.1e} q={q[i]:.2f} ps={ps[i]:.0f} x0={x0[i]*1e4:.2f}mm")
    print("  area:", np.round(a, 2))
