"""Verifikasi kualitas lagu hasil render."""
import numpy as np
from scipy.io import wavfile

sr, y = wavfile.read("output/burung_kakatua.wav")
y = y.astype(float) / 32767
w = int(0.1 * sr)
rms = np.array([np.sqrt(np.mean(y[i:i + w] ** 2)) for i in range(0, len(y) - w, int(0.5 * sr))])
tv = np.arange(len(rms)) * 0.5
body = rms[(tv > 1) & (tv < len(y) / sr - 2)]
med = np.median(body)
print("RMS median", round(float(med), 3), " CV:", round(float(body.std() / med), 2), "(awal 0.87)")
print("detik lemah (<20% median):", int((body < 0.2 * med).sum()), "/", len(body))

w2 = int(0.02 * sr)
e = np.sqrt(np.convolve(np.nan_to_num(y) ** 2, np.ones(w2) / w2, "same") + 1e-12)
mask = np.abs(y) > 4.0 * e
idx = np.flatnonzero(mask)
clicks = 0
if len(idx):
    sp = np.flatnonzero(np.diff(idx) > 3)
    starts = np.r_[idx[0], idx[sp + 1]]
    ends = np.r_[idx[sp], idx[-1]]
    dur = (ends - starts + 1) / sr * 1000
    clicks = int((dur <= 0.15).sum())
print("klik artifak:", clicks, "(awal 1603)")
print("max", round(float(np.max(np.abs(y))), 3), "NaN", int(np.isnan(y).sum()))
prof = [round(float(np.sqrt(np.mean(y[i:i + w] ** 2))), 2) for i in range(0, len(y) - w, int(1.0 * sr))]
print("rms/detik:", prof)
