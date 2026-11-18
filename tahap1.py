"""
Tahap 1 - Sumber suara: paru-paru + pita suara (two-mass model)

Menghasilkan:
  output/tahap1_buzz.wav      -> suara "dengung" glotis (belum ada saluran suara, jadi
                                 terdengar seperti buzzer; vokal datang di Tahap 2)
  output/tahap1_grafik.png    -> grafik aliran udara glotis, tekanan, dan f0

Jalankan:  python tahap1.py
"""
import os
import numpy as np
from scipy.io import wavfile
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from vocal_fold import simulate

FS = 44100
OUT = "output"
os.makedirs(OUT, exist_ok=True)


def envelope(n, points):
    """Interpolasi linear: points = [(detik, nilai), ...]"""
    t = np.arange(n) / FS
    ts, vs = zip(*points)
    return np.interp(t, ts, vs)


def estimate_f0(sig, fs, win=0.08, hop=0.01):
    """f0 per jendela dengan autokorelasi sederhana."""
    w, h = int(win * fs), int(hop * fs)
    f0, tt = [], []
    for i in range(0, len(sig) - w, h):
        seg = sig[i:i + w] - np.mean(sig[i:i + w])
        if np.ptp(seg) < 20e-6:   # belum ada osilasi (< 20 cm3/s)
            f0.append(np.nan); tt.append((i + w / 2) / fs); continue
        ac = np.correlate(seg, seg, "full")[w - 1:]
        lo, hi = int(fs / 400), int(fs / 60)
        lag = lo + np.argmax(ac[lo:hi])
        f0.append(fs / lag if ac[lag] > 0.3 * ac[0] else np.nan)
        tt.append((i + w / 2) / fs)
    return np.array(tt), np.array(f0)


def main():
    dur = 3.0
    n = int(FS * dur)

    # Skenario: tarik napas -> mulai bersuara -> nada naik -> lepas
    #  - tekanan subglotal naik pelan (onset), stabil, turun (offset)
    #  - ketegangan q naik (nada naik ~ 1 oktaf), lalu turun lagi
    ps = envelope(n, [(0, 0), (0.2, 150), (0.5, 800), (2.4, 800), (3.0, 0)])
    q = envelope(n, [(0, 0.9), (1.0, 0.9), (2.0, 1.5), (3.0, 1.0)])
    x0 = envelope(n, [(0, 0.02e-2), (3.0, 0.02e-2)])

    print("Mensimulasikan...")
    ug, amin = simulate(n, FS, ps, q, x0, 4)
    ug = np.array(ug)

    # Radiasi di bibir ~ turunan aliran (high-pass +6 dB/oktaf)
    sound = np.diff(ug, prepend=ug[0]) * FS
    sound /= np.max(np.abs(sound)) + 1e-12
    wavfile.write(f"{OUT}/tahap1_buzz.wav", FS, (sound * 0.7 * 32767).astype(np.int16))

    tt, f0 = estimate_f0(ug, FS)

    fig, ax = plt.subplots(4, 1, figsize=(10, 10))
    t = np.arange(n) / FS
    ax[0].plot(t, ps); ax[0].set_ylabel("Ps (Pa)"); ax[0].set_title("Tekanan subglotal (paru-paru)")
    ax[1].plot(t, ug * 1e6, lw=0.5); ax[1].set_ylabel("Ug (cm³/s)"); ax[1].set_title("Aliran udara glotis")
    m = (t > 1.0) & (t < 1.03)
    ax[2].plot(t[m], ug[m] * 1e6); ax[2].set_ylabel("Ug (cm³/s)")
    ax[2].set_title("Zoom 30 ms: bentuk gelombang glotis (fase buka–tutup)")
    ax[3].plot(tt, f0, "o", ms=2); ax[3].set_ylabel("f0 (Hz)"); ax[3].set_xlabel("detik")
    ax[3].set_title("Frekuensi dasar (nada) hasil simulasi")
    plt.tight_layout(); plt.savefig(f"{OUT}/tahap1_grafik.png", dpi=110)
    print("Selesai ->", OUT)


if __name__ == "__main__":
    main()
