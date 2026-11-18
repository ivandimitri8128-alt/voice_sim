"""Uji kestabilan osilasi per vokal vs ps & q (mencari syarat fonasi)."""
import numpy as np

import song as songmod
from timeline import Control, _f0_direct
from vowels import VOWELS

FS = 44100
N = int(FS * 0.8)


def run(q, ps, x0, vowel):
    ctl = Control(N, FS)
    a = np.tile(np.asarray(VOWELS[vowel], float), (N, 1))
    ctl.a_cur[:] = a
    ctl.a_next[:] = a
    ctl.mix[:] = 1.0
    ctl.voiced[:] = 1.0
    z = np.zeros(N)
    p, ug = songmod.run_song(N, FS, np.full(N, float(ps)), np.full(N, float(q)),
                             np.full(N, float(x0)), ctl.a_cur, ctl.a_next, ctl.mix,
                             ctl.voiced, z, ctl.nas, z, z, z, 4, 0.0)
    ug = np.asarray(ug)[int(0.3 * FS):]
    # osilasi = std dari ug setelah buang DC
    osc = np.std(ug - np.mean(ug))
    m = _f0_direct(np.asarray(ug), FS, FS, len(ug))
    ok = ~np.isnan(m)
    f0 = float(np.median(m[ok])) if ok.sum() > 100 else np.nan
    return osc, f0


if __name__ == "__main__":
    print("vokal  q    ps     x0(mm)  osc_ug     f0")
    for v in "aiu":
        for q in (0.9, 1.2):
            for ps in (680, 1000, 1400, 2000):
                for x0 in (0.02e-2, 0.03e-2):
                    o, f = run(q, ps, x0, v)
                    print(f"{v}    {q:.1f} {ps:5.0f}  {x0*1e4:4.1f}   {o:.2e}  {f:6.1f}")
