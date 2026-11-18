"""Apakah titik setimbang glotis stabil? (osilasi butuh kick awal?)
Usage: python test_eq.py [ramp0/1] [kick0/1]
"""
import sys

import numpy as np

import song as songmod
from timeline import Control
from vowels import VOWELS

songmod.KICK = bool(int(sys.argv[2])) if len(sys.argv) > 2 else True
songmod.RAMP = bool(int(sys.argv[1])) if len(sys.argv) > 1 else True

FS = 44100
N = int(FS * 0.8)
a = np.tile(np.array(VOWELS["a"]), (N, 1))
z = np.zeros(N)

for label, vo in [
    ("voiced=1 dari sampel 0", np.ones(N)),
    ("voiced=1 mulai 10ms", np.clip((np.arange(N) - int(0.01 * FS)) / int(0.001 * FS), 0, 1)),
    ("voiced=1 mulai 50ms", np.clip((np.arange(N) - int(0.05 * FS)) / int(0.001 * FS), 0, 1)),
]:
    ctl = Control(N, FS)
    ctl.a_cur[:] = a
    ctl.a_next[:] = a
    ctl.mix[:] = 1.0
    ctl.voiced[:] = vo
    p, ug = songmod.run_song(N, FS, np.full(N, 1042.0), np.full(N, 1.54),
                             np.full(N, 2.07e-4), ctl.a_cur, ctl.a_next, ctl.mix,
                             ctl.voiced, z, ctl.nas, z, z, z, 4, 0.0)
    ug = np.asarray(ug)
    o1 = np.std(ug[int(0.05 * FS):int(0.15 * FS)])
    o2 = np.std(ug[int(0.5 * FS):])
    print(f"{label:26s} osc@0.05-0.15s {o1:.1e}   osc@0.5s {o2:.1e}")
