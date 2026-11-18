"""
Tahap 3 - Nyanyi: baca partitur teks, hasilkan suara.

Contoh pakai:
    py sing_song.py songs/burung_kakatua.txt
    py sing_song.py songs/burung_kakatua.txt -o output/kakatua.wav

Output default:
    output/<nama_lagu>.wav          -> lagu yang dinyanyikan
    output/<nama_lagu>_spektro.png  -> spektrogram + pitch (cek melodi)
"""
import argparse
import os
import sys
import time

import numpy as np
from scipy.io import wavfile
import scipy.signal as ss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from score import load_score, parse_score, score_duration
from timeline import synthesize_song

FS = 44100
OUT = "output"


def post_process(p_lip, fs):
    """
    Tekanan di bibir -> sinyal audio.
    1) radiasi bibir (high-pass)
    2) kompresor lembut (ratakan beda level, TANPA memicu distorsi)
    3) reverb ruang kecil
    4) limiter tanh ringan (tanpa flat-top yang dalam)
    """
    y = np.nan_to_num(np.asarray(p_lip, dtype=np.float64))
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])
    sos = ss.butter(2, 80.0 / (fs / 2.0), btype="high", output="sos")
    y = ss.sosfilt(sos, y)
    y -= np.mean(y)

    # --- kompresor lembut: rasio rendah, ceiling rendah -> tidak distorsi
    env_fast = np.abs(ss.hilbert(y))
    sos_lp = ss.butter(2, 25.0 / (fs / 2.0), btype="low", output="sos")
    env = np.maximum(ss.sosfilt(sos_lp, env_fast), 0.0) + 1e-6
    ref = np.percentile(env, 85)
    gain = np.clip((ref / env) ** 0.45, 0.8, 2.4)   # sebelumnya 6.0 -> flat-top!
    y = y * gain

    # --- reverb ruang kecil
    ir_len = int(0.35 * fs)
    t_ir = np.arange(ir_len) / fs
    ir = np.random.default_rng(3).standard_normal(ir_len) * np.exp(-t_ir / 0.10)
    ir *= np.linspace(1.0, 0.0, ir_len) ** 2
    ir[0] = 1.0
    wet = ss.fftconvolve(y, ir / np.max(np.abs(ir)))
    y = y + 0.10 * wet[:len(y)] / (np.max(np.abs(wet[:len(y)])) + 1e-12)

    # --- normalisasi berbasis persentil + limiter dengan "knee" lembut.
    # JANGAN pakai max: spike letupan plosif (amplitudo puluhan kali normal)
    # akan menyerap seluruh gain dan membuat audio nyaris hening.
    # Knee: linier di bawah 0.65, saturasi mulus di atasnya (tanpa langkah tajam).
    m = np.percentile(np.abs(y), 99.95) + 1e-12
    a = y / m
    aa = np.abs(a)
    sat = np.tanh((aa - 0.65) / 0.35) * 0.35 + 0.65
    a = np.sign(a) * np.where(aa <= 0.65, aa, sat)
    return a * 0.9


def save_plot(y, f0, good, fs, path, title):
    f, tt, S = ss.spectrogram(y, fs, nperseg=2048, noverlap=1792)
    fig, ax = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1]})
    ax[0].pcolormesh(tt, f, 10 * np.log10(S + 1e-12), shading="auto",
                     vmin=-110, vmax=-35, cmap="magma")
    ax[0].set_ylim(0, 4200)
    ax[0].set_ylabel("Hz")
    ax[0].set_title(title)
    if f0 is not None:
        tt2 = np.arange(len(f0)) / fs
        ax[1].plot(tt2, f0, lw=0.7, color="tab:green", label="f0 target")
        ax[1].set_ylabel("f0 (Hz)")
        # overlay f0 terukur dari audio
        try:
            from timeline import estimate_f0
            te, fe = estimate_f0(y, fs, win=0.045, hop=0.01, fmin=70, fmax=900)
            ax[1].plot(te, fe, ".", ms=2.5, color="tab:red", label="f0 terukur")
        except Exception:
            pass
        ax[1].set_ylim(0, 600)
        ax[1].legend(loc="upper right", fontsize=8)
    ax[1].set_xlabel("detik")
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Nyanyikan partitur teks dengan simulasi fisika pita suara.")
    ap.add_argument("score", help="file partitur .txt")
    ap.add_argument("-o", "--out", default=None, help="file WAV keluaran")
    ap.add_argument("--tempo", type=float, default=None, help="override BPM")
    ap.add_argument("--q", type=float, default=None, help="override ketegangan pita suara")
    ap.add_argument("--vibrato", type=float, default=22.0, help="kedalaman vibrato (cents)")
    ap.add_argument("--noise", type=float, default=0.10, help="amplitudo derau aspirasi")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args(argv)

    with open(args.score, "r", encoding="utf-8") as f:
        text = f.read()
    notes, cfg = parse_score(text)
    if not notes:
        sys.exit("Partitur kosong atau tidak terbaca.")

    tempo = args.tempo if args.tempo is not None else cfg["tempo"]
    q_base = args.q if args.q is not None else cfg["q"]
    name = os.path.splitext(os.path.basename(args.score))[0]
    os.makedirs(OUT, exist_ok=True)
    out_wav = args.out or os.path.join(OUT, name + ".wav")

    dur = score_duration(notes, tempo)
    print(f"Lagu: {name} | {len(notes)} nada | tempo {tempo:.0f} BPM | ~{dur:.1f} detik")
    print(f"Range pitch: {min(n.midi for n in notes if n.midi)}"
          f"..{max(n.midi for n in notes if n.midi)} (MIDI)")

    t0 = time.time()
    p_lip, ug, ctl, f0 = synthesize_song(notes, fs=FS, tempo=tempo, q_base=q_base,
                                         vibrato_cents=args.vibrato,
                                         noise_amp=args.noise)
    print(f"Simulasi selesai dalam {time.time() - t0:.1f} detik")

    if np.max(np.abs(p_lip)) < 1e-9:
        sys.exit("PERINGATAN: simulasi menghasilkan sinyal kosong (model tidak "
                 "osilasi). Turunkan 'q' di partitur atau pitch lagu.")

    y = post_process(p_lip, FS)
    wavfile.write(out_wav, FS, (y * 0.95 * 32767).astype(np.int16))
    print("WAV  ->", out_wav)

    if not args.no_plot:
        png = os.path.join(OUT, name + "_spektro.png")
        save_plot(y, f0, None, FS, png,
                  f"Spektrogram: {name} ({tempo:.0f} BPM, simulasi fisika)")
        print("Plot ->", png)


if __name__ == "__main__":
    main()
