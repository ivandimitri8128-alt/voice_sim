"""
Tahap 2 - Vokal: pita suara + saluran suara terhubung

Menghasilkan (folder output/):
  tahap2_vokal.wav      -> urutan a - e - i - o - u dengan nada tetap
  tahap2_spektrogram.png -> spektrogram + garis formant teoretis

Jalankan:  python tahap2.py
"""
import os, time
import numpy as np
from scipy.io import wavfile
import scipy.signal as ss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from voice import synthesize
from vowels import VOWELS

FS = 44100
OUT = "output"
os.makedirs(OUT, exist_ok=True)

SEQ = "aeiou"
HOLD, GLIDE = 0.55, 0.25          # detik: tahan vokal, peralihan antar vokal
LEAD, TAIL = 0.75, 0.35           # detik: onset dan offset suara


def area_at(t):
    """Bentuk saluran pada waktu t: tahan vokal, lalu peralihan mulus (interpolasi log-area)."""
    t = t - LEAD
    period = HOLD + GLIDE
    i = int(max(t, 0) // period)
    if i >= len(SEQ) - 1:
        return VOWELS[SEQ[-1]]
    ph = max(t, 0) - i * period
    if ph < HOLD:
        return VOWELS[SEQ[i]]
    u = (ph - HOLD) / GLIDE
    u = 0.5 - 0.5 * np.cos(np.pi * u)             # easing halus
    a, b = np.log(VOWELS[SEQ[i]]), np.log(VOWELS[SEQ[i + 1]])
    return np.exp(a + (b - a) * u)


def main():
    total = LEAD + len(SEQ) * HOLD + (len(SEQ) - 1) * GLIDE + TAIL
    n = int(FS * total)
    t = np.arange(n) / FS
    ps = np.interp(t, [0, 0.1, 0.3, total - TAIL, total], [0, 150, 800, 800, 0])
    q = np.full(n, 1.0)                      # ~120 Hz
    x0 = np.full(n, 0.02e-2)

    print("Mensimulasikan %.1f detik suara (tunggu, ini butuh waktu)..." % total)
    t0 = time.time()
    p_lip, ug = synthesize(n, FS, ps, q, x0, area_at)
    print("selesai dalam %.1f detik" % (time.time() - t0))

    y = np.append(p_lip[0], p_lip[1:] - 0.97 * p_lip[:-1])       # radiasi bibir (high-pass)
    y /= np.max(np.abs(y)) + 1e-12
    wavfile.write(f"{OUT}/tahap2_vokal.wav", FS, (y * 0.8 * 32767).astype(np.int16))

    f, tt, S = ss.spectrogram(y, FS, nperseg=2048, noverlap=1792)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.pcolormesh(tt, f, 10 * np.log10(S + 1e-12), shading="auto", vmin=-110, vmax=-30, cmap="magma")
    ax.set_ylim(0, 4500); ax.set_xlabel("detik"); ax.set_ylabel("Hz")
    ax.set_title("Spektrogram a-e-i-o-u (pita suara + saluran suara)")
    plt.tight_layout(); plt.savefig(f"{OUT}/tahap2_spektrogram.png", dpi=110)
    print("Selesai ->", OUT)


if __name__ == "__main__":
    main()
