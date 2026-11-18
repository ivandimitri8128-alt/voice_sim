"""Eksperimen cepat: render potongan pendek & ukur akurasi pitch.

Jalankan:  py test_selftest.py
"""
import numpy as np

from score import parse_score
from timeline import synthesize_song, estimate_f0

DEMO = """
tempo 120
q 1.5
C4:2=ma C4:1=ma A4:2=ma A4:1=ma G4:2=sa C4:2=la
"""

if __name__ == "__main__":
    notes, cfg = parse_score(DEMO)
    print("not:", len(notes), cfg)
    p_lip, ug, ctl, f0 = synthesize_song(notes, tempo=cfg["tempo"], q_base=cfg["q"])
    tt, ff = estimate_f0(np.asarray(p_lip, float), 44100, win=0.05, hop=0.01,
                         fmin=70, fmax=900)
    sel = ~np.isnan(ff)
    idx = np.clip((tt * 44100).astype(int), 0, len(f0) - 1)
    ok = sel & (f0[idx] > 100) & (ctl.voiced[idx] > 0.5)
    cent = 1200 * np.log2(np.maximum(ff[ok], 1) / f0[idx][ok])
    print(f"frame ber-pitch: {ok.sum()} / {len(ff)}")
    print(f"error median: {np.median(np.abs(cent)):.0f} cents  "
          f"p90: {np.percentile(np.abs(cent), 90):.0f} cents")
    print(f"deteksi salah-oktaf: {np.mean(np.abs(cent) > 600) * 100:.0f}%")
    print("\nRincian per bagian:")
    for tsec in np.arange(0.2, len(f0) / 44100 - 0.1, 0.5):
        k = int(tsec * 44100)
        j = np.argmin(np.abs(tt - tsec))
        meas = "---" if np.isnan(ff[j]) else f"{ff[j]:6.1f}"
        print(f"  t={tsec:4.1f}s target={f0[k]:6.1f}  ukur={meas}")
