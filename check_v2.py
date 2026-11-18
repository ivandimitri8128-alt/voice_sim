"""
check_v2.py - Verifikasi objektif hasil sing_v2 (tanpa perlu mendengar).

    py check_v2.py songs/burung_kakatua.txt output/burung_kakatua_v2.wav

Yang diukur:
  1. Pitch  : error terhadap NOTA PARTITUR (bukan terhadap kurva yang sudah meluncur),
              hanya di inti vokal; plus fraksi waktu pitch datar.
  2. Level  : kestabilan RMS antar suku kata (lagu nyata tidak boleh naik-turun >6 dB).
  3. Fonasi : jumlah jendela vokal yang "mati" (harus 0).
  4. Formant: F1-F3 tiap vokal terhadap target pria dewasa.
  5. HNR    : rasio harmonik/derau di vokal (suara manusia bernyanyi ~ 15-30 dB).
  6. Klik   : lonjakan impuls.

Opsi --seed N / --no-human / --bar B / --pickup P / --barreset: samakan dengan opsi saat render (rencana waktu harus identik).
Opsi --no-falsetto: secara bawaan falsetto=True, artinya vokal pada nada >= Bb4 (register falset,
amplitudo sengaja -3 dB) dikoreksi +3 dB sebelum diuji "mati" (< -15 dB), jadi level yang turun
karena falset tidak dianggap vokal mati.
"""
import sys

import numpy as np
import scipy.signal as ss
from scipy.io import wavfile
from scipy.linalg import toeplitz, solve

from score import parse_score
from sing_v2 import plan_song, build_segments, estimate_f0, fal_weight, FAL_DB, timing_rng, cons_rng

TARGET = {"a": (730, 1090, 2440), "e": (530, 1840, 2480), "i": (270, 2290, 3010),
          "o": (570, 840, 2410), "u": (300, 870, 2240)}


def lpc_formants(x, fs, order=14):
    x = ss.resample_poly(x, 1, 4)
    fs2 = fs / 4
    x = x * np.hanning(len(x))
    x = np.append(x[0], x[1:] - 0.97 * x[:-1])
    r = np.correlate(x, x, "full")[len(x) - 1:len(x) + order]
    a = solve(toeplitz(r[:order]) + 1e-9 * np.eye(order), r[1:order + 1])
    z = np.roots(np.r_[1, -a])
    z = z[np.imag(z) > 0]
    fr = np.angle(z) * fs2 / (2 * np.pi)
    bw = -np.log(np.abs(z)) * fs2 / np.pi
    o = np.argsort(fr)
    return [(f, b) for f, b in zip(fr[o], bw[o]) if f > 150 and b < 500][:4]


def hnr_db(x, fs, f0):
    """HNR sisir: energi di sekitar harmonik (<= 3 kHz, toleransi vibrato) vs di antaranya."""
    w = np.hanning(len(x))
    S = np.abs(np.fft.rfft((x - np.mean(x)) * w)) ** 2
    fr = np.fft.rfftfreq(len(x), 1 / fs)
    hm = np.zeros(len(fr), bool)
    nh = int(min(12, 3000 / f0))
    for h in range(1, nh + 1):
        hm |= np.abs(fr - h * f0) < 0.02 * h * f0 + 20
    band = (fr > 0.7 * f0) & (fr < (nh + 0.5) * f0)
    return 10 * np.log10(S[hm & band].sum() / (S[~hm & band].sum() + 1e-20))


def main(score_path, wav_path, falsetto=True, seed=1, human=True, bar=None, pickup=None, barreset=False):
    fs, y = wavfile.read(wav_path)
    y = y.astype(float) / 32767
    if y.ndim == 2:
        y = y.mean(axis=1)
    with open(score_path, encoding="utf-8") as f:
        notes, cfg = parse_score(f.read())
    # harus sama persis dengan synthesize(): seed & human sama -> rencana waktu identik
    events, marks, phrases, total = plan_song(notes, cfg["tempo"], rng=timing_rng(seed) if human else None,
                                              beats_per_bar=bar if bar else cfg.get("meter", 4.0),
                                              pickup=pickup if pickup is not None else cfg.get("pickup", 0.0),
                                              bar_reset=barreset or bool(cfg.get("barreset", 0.0)))
    segs = build_segments(events, cons_rng(seed))
    vow = [s for s in segs if s[2] == "vowel"]

    # ---- 1. pitch vs nota partitur
    tt, ff = estimate_f0(y, fs, win=0.045, hop=0.01, fmin=70, fmax=900)
    errs, covered, total_core = [], 0, 0
    for (ta, ma), nxt in zip(marks, marks[1:] + [(total, None)]):
        a, b = ta + 0.12, nxt[0] - 0.06           # inti nada (lewati luncuran & onset)
        if b - a < 0.08:
            continue
        sel = (tt > a) & (tt < b)
        total_core += sel.sum()
        f = ff[sel]
        f = f[~np.isnan(f)]
        covered += len(f)
        if len(f):
            tgt = 440 * 2 ** ((ma - 69) / 12)
            errs.append(1200 * np.log2(np.median(f) / tgt))
    errs = np.array(errs)
    print(f"[PITCH]  nada terukur {len(errs)}/{len(marks)}  frame terdeteksi {covered}/{total_core}"
          f" ({100 * covered / max(total_core, 1):.0f}%)")
    print(f"         error median per nada: {np.median(np.abs(errs)):.0f} cents,"
          f" terburuk {np.max(np.abs(errs)):.0f} cents  (manusia terlatih < 30)")

    # ---- 2. level antar suku kata + 3. mati
    rms = []
    dead = 0
    for t0, t1, kind, name, info in vow:
        a, b = int((t0 + 0.05) * fs), int((t1 - 0.03) * fs)
        if b - a < int(0.04 * fs):
            continue
        r = np.sqrt(np.mean(y[a:b] ** 2))
        rms.append(r)
    rms = np.array(rms)
    db = 20 * np.log10(rms / np.median(rms))
    db_dead = db.copy()
    n_fal = 0
    if falsetto:                      # koreksi level yang memang sengaja turun di register falset
        mt = np.array([m for _, m in marks])
        tm = np.array([t for t, _ in marks])
        wv = []
        for t0, t1, kind, name, info in vow:
            if int((t1 - 0.03) * fs) - int((t0 + 0.05) * fs) < int(0.04 * fs):
                continue
            k = max(0, np.searchsorted(tm, t0 + 0.05, side="right") - 1)
            wv.append(fal_weight(mt[k]))
        wv = np.array(wv)
        n_fal = int((wv > 0.5).sum())
        db_dead = db - FAL_DB * wv
    print(f"[LEVEL]  {len(rms)} vokal: p10/p50/p90 = {np.percentile(db, 10):+.1f}/0/{np.percentile(db, 90):+.1f} dB"
          f"  (rentang {db.max() - db.min():.1f} dB)  vokal 'mati' (<-15 dB): {(db_dead < -15).sum()}"
          + (f"  [falsetto: {n_fal} vokal dikoreksi +3 dB]" if falsetto else ""))

    # ---- 4. formant + 5. HNR per vokal
    print("[FORMANT] vokal  F1/F2/F3 terukur (BW1)  | target")
    seen = {}
    hn = []
    for t0, t1, kind, name, info in vow:
        if name not in TARGET or t1 - t0 < 0.25:
            continue
        mid = 0.5 * (t0 + t1)
        a, b = int((mid - 0.03) * fs), int((mid + 0.03) * fs)
        fm = lpc_formants(y[a:b], fs)
        # f0 lokal
        sel = (tt > mid - 0.06) & (tt < mid + 0.06)
        f0 = np.nanmedian(ff[sel]) if np.any(~np.isnan(ff[sel])) else np.nan
        if not np.isnan(f0):
            hn.append(hnr_db(y[int((mid - 0.06) * fs):int((mid + 0.06) * fs)], fs, f0))
        if name not in seen and len(fm) >= 3:
            seen[name] = fm
    for k, fm in seen.items():
        print(f"          /{k}/    " + " ".join(f"{int(f)}" for f, _ in fm[:3]) +
              f"  (BW1 {int(fm[0][1])})   | " + " ".join(str(v) for v in TARGET[k]))
    if hn:
        print(f"[HNR]    median {np.median(hn):.1f} dB  (p10 {np.percentile(hn, 10):.1f})")

    # ---- 6. klik
    w2 = int(0.02 * fs)
    e = np.sqrt(np.convolve(y ** 2, np.ones(w2) / w2, "same") + 1e-12)
    ok = np.ones(len(y), bool)
    for t0, t1, kind, name, info in segs:        # letupan plosif = transien yang diharapkan
        if kind == "cons" and name in ("p", "b", "t", "d", "k", "g", "c", "j"):
            ok[int((t1 - 0.005) * fs):int((t1 + 0.03) * fs)] = False
    idx = np.flatnonzero((np.abs(y) > 6.0 * e) & ok)
    clicks = 0
    if len(idx):
        sp = np.flatnonzero(np.diff(idx) > 3)
        clicks = len(sp) + 1
    print(f"[KLIK]   lonjakan impuls di luar letupan plosif: {clicks}   puncak {np.max(np.abs(y)):.2f}   NaN {int(np.isnan(y).sum())}")

    # ---- spektrum rata-rata vokal (kemiringan)
    f, P = ss.welch(y, fs, nperseg=8192)
    band = lambda lo, hi: 10 * np.log10(P[(f >= lo) & (f < hi)].sum() + 1e-20)
    b0 = band(100, 500)
    print("[SPEKTRUM] rel. 100-500 Hz:", "  ".join(f"{lo}-{hi}: {band(lo, hi) - b0:+.0f}dB"
          for lo, hi in [(500, 1000), (1000, 2000), (2000, 4000), (4000, 8000), (8000, 16000)]))
    print(f"[SINGER]  pita 2,4-3,4 kHz vs 0,3-1 kHz: {band(2400, 3400) - band(300, 1000):+.1f} dB "
          f"(bandingkan dengan --singer 0)")


if __name__ == "__main__":
    argv = sys.argv[1:]
    opt = lambda k, d: float(argv[argv.index(k) + 1]) if k in argv else d
    skip = {argv.index(k) + 1 for k in ("--seed", "--bar", "--pickup") if k in argv}
    args = [a for i, a in enumerate(argv, 0) if not a.startswith("--") and i not in skip]
    main(args[0], args[1], falsetto="--no-falsetto" not in argv, seed=int(opt("--seed", 1)),
         human="--no-human" not in argv, bar=opt("--bar", 0.0) or None,
         pickup=opt("--pickup", None), barreset="--barreset" in argv)
