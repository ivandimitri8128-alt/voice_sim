"""Eksperimen: relasi f0 vs q & respons amplitudo (untuk kalibrasi nyanyi).

Jalankan:  py tune_pitch.py
"""
import numpy as np

import song as songmod
from timeline import _f0_direct, Control
from vowels import VOWELS

FS = 44100
N = int(FS * 0.6)


def run_fixed(q, ps, x0=0.02e-2, area=None):
    ctl = Control(N, FS)
    a = np.tile(np.asarray(area if area is not None else VOWELS["a"], float), (N, 1))
    ctl.a_cur[:] = a
    ctl.a_next[:] = a
    ctl.mix[:] = 1.0
    ctl.voiced[:] = 1.0
    z = np.zeros(N)
    p_lip, ug = songmod.run_song(N, FS, np.full(N, float(ps)), np.full(N, float(q)),
                                 np.full(N, float(x0)), ctl.a_cur, ctl.a_next,
                                 ctl.mix, ctl.voiced, z, ctl.nas, z, z, z, 4, 0.0)
    return np.asarray(p_lip, float), np.asarray(ug, float)


def report(q, ps, x0=0.02e-2):
    p, ug = run_fixed(q, ps, x0)
    m = _f0_direct(ug, FS, FS, N)
    ok = ~np.isnan(m)
    f = np.median(m[ok]) if ok.sum() > 100 else np.nan
    rms = np.sqrt(np.mean(p[int(0.3 * FS):] ** 2))
    return f, rms, ok.mean()


if __name__ == "__main__":
    print("q      ps     f0      rms      valid")
    for q in [0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]:
        for ps in (800, 1600):
            f, r, v = report(q, ps)
            print(f"{q:4.1f} {ps:6.0f} {f:8.1f} {r:8.4f} {v:5.2f}")
