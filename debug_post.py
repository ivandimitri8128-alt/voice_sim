"""Debug post_process: tahap mana yang menghancurkan sinyal."""
import numpy as np
from scipy import signal as ss
from score import parse_score
from timeline import synthesize_song

notes, cfg = parse_score('tempo 104\nq 1.25\nC3:1=Bu_rung C3:1 A3:2=Ka A3:1=ka B3:1=tu B3:1=a')
p, ug, ctl, f0 = synthesize_song(notes, tempo=104, q_base=1.25)

y = np.nan_to_num(np.asarray(p, dtype=np.float64))
y = np.append(y[0], y[1:] - 0.97 * y[:-1])
sos = ss.butter(2, 80.0 / (44100 / 2), btype="high", output="sos")
y = ss.sosfilt(sos, y)
y -= np.mean(y)
env_fast = np.abs(ss.hilbert(y))
sos_lp = ss.butter(2, 25.0 / (44100 / 2), btype="low", output="sos")
env = np.maximum(ss.sosfilt(sos_lp, env_fast), 0.0) + 1e-6
ref = np.percentile(env, 85)
gain = np.clip((ref / env) ** 0.45, 0.8, 2.4)
y2 = y * gain
m = np.percentile(np.abs(y2), 99.95) + 1e-12
a = y2 / m
aa = np.abs(a)
sat = np.tanh((aa - 0.65) / 0.35) * 0.35 + 0.65
a = np.sign(a) * np.where(aa <= 0.65, aa, sat)
out = a * 0.9

for name, sig in [("raw-p", p), ("hp", y), ("comp", y2), ("final", out)]:
    w = int(0.25 * 44100)
    prof = [f"{np.sqrt(np.mean(sig[s:s+w]**2)):.3f}" for s in range(0, len(sig) - w, int(0.5 * 44100))]
    print(name, prof)

import song as songmod
from timeline import Control, plan_phrases, _paint_control, _pitch_track, midi_to_freq, calibrate_q
# jalankan ulang dengan on_gain terekam
phrases, marks = plan_phrases(notes, 60 / 104.0)
N = int((phrases[-1]["t1"] + 0.05) * 44100)
ctl2 = Control(N, 44100)
for ph in phrases:
    _paint_control(ctl2, ph["segs"], 44100)
t = np.arange(N) / 44100
f0 = midi_to_freq(_pitch_track(marks, t, 22.0, 5.4))
q = np.clip(f0 / 136.0, 0.5, 4.0) * 1.0
ps = np.zeros(N)
for ph in phrases:
    i0, i1 = int(ph["t0"] * 44100), min(int(ph["t1"] * 44100), N)
    if i1 > i0:
        ps[i0:i1] = np.interp(np.linspace(0, 1, i1 - i0), [0, 0.08, 0.25, 0.9, 1.0],
                              [0, 620, 790, 780, 0])
ps *= np.clip(0.9 + 0.12 * (f0 / 220.0), 0.85, 1.15)
x0 = np.interp(f0, [80, 130, 260, 520], [0.034e-2, 0.029e-2, 0.020e-2, 0.016e-2])
x0 *= np.clip(1.0 + 0.15 * (1.0 - ctl2.voiced), 1.0, 1.2)
q, ps, x0 = calibrate_q(q, f0, ctl2.voiced, ctl2, ps, x0, 44100)
vs = np.convolve(ctl2.voiced, np.hanning(int(0.08 * 44100)) /
                 np.hanning(int(0.08 * 44100)).sum(), mode="same")
hi_reg = np.clip((f0 - 200.0) / 100.0, 0.0, 1.0)
ps = np.maximum(ps, (680.0 + 700.0 * hi_reg) * vs)
k = int(0.012 * 44100) | 1
ker = np.hanning(k) / np.hanning(k).sum()
ps = np.convolve(ps, ker, mode="same")
x0 = np.convolve(x0, ker, mode="same")
q = np.convolve(q, ker, mode="same")
onset_env = np.clip(vs - np.roll(vs, int(0.03 * 44100)), 0, None)
onset_env = np.convolve(onset_env, ker, mode="same")
x0 = x0 + 0.012e-2 * np.clip(onset_env * 8.0, 0, 1)
rng = np.random.default_rng(0)
uv = songmod.filter_noise(rng, ctl2, 44100, 0.10)
pl, ugg = songmod.run_song(N, 44100, ps.astype(np.float64), q, x0, ctl2.a_cur,
                           ctl2.a_next, ctl2.mix, ctl2.voiced, uv, ctl2.nas,
                           ctl2.burst, ctl2.burst_lo, ctl2.burst_hi, 4, 0.10)
og = songmod._last_on_gain
w = int(0.25 * 44100)
print("on_gain min/max per 0.5s:",
      [f"{og[s:s+w].min():.2f}/{og[s:s+w].max():.2f}" for s in range(0, N - w, int(0.5 * 44100))])
print("p rms per 0.5s:",
      [f"{np.sqrt(np.mean(pl[s:s+w]**2)):.1f}" for s in range(0, N - w, int(0.5 * 44100))])

ug = np.asarray(ug)
w = int(0.25 * 44100)
prof = [f"{np.sqrt(np.mean(ug[s:s+w]**2))*1e4:.2f}" for s in range(0, len(ug) - w, int(0.5 * 44100))]
print("ug x1e4:", prof)
prof = [f"{ctl.voiced[s:s+w].mean():.2f}" for s in range(0, len(ug) - w, int(0.5 * 44100))]
print("voiced:", prof)
prof = [f"{np.mean(ctl.a_cur[s:s+w, 20]):.2f}" for s in range(0, len(ug) - w, int(0.5 * 44100))]
print("area[20]:", prof)
print("m(99.95pct) =", round(float(m), 2))
print("p99.95 dari |y2| =", round(float(np.percentile(np.abs(y2), 99.95)), 2),
      " p99 =", round(float(np.percentile(np.abs(y2), 99)), 2),
      " median =", round(float(np.median(np.abs(y2))), 2))
