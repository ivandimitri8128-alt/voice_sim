"""Verifikasi akurasi pitch lagu hasil render."""
import numpy as np
from scipy.io import wavfile

from score import load_score
from timeline import plan_phrases, _pitch_track, midi_to_freq, estimate_f0
from sing_song import FS

sr, y = wavfile.read("output/burung_kakatua.wav")
y = y.astype(float) / 32767
n, cfg = load_score("songs/burung_kakatua.txt")
phrases, marks = plan_phrases(n, 60 / 104)
tt = np.arange(len(y)) / FS
f0 = midi_to_freq(_pitch_track(marks, tt, 22.0, 5.4))
te, fe = estimate_f0(y, FS, win=0.05, hop=0.01, fmin=70, fmax=500)
idx = np.clip((te * FS).astype(int), 0, len(f0) - 1)
ok = ~np.isnan(fe) & (f0[idx] > 60)
c = 1200 * np.log2(fe[ok] / f0[idx][ok])
core = np.abs(c) < 200   # abaikan transisi konsonan
print("frame pitch:", int(ok.sum()), " akurasi inti (<200c):", round(float(core.mean()) * 100), "%")
print("error median (inti):", round(float(np.median(np.abs(c[core])))), "cents")
