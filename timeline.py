"""
Tahap 3 - Mesin nyanyi: partitur teks -> kontrol pita suara + saluran suara -> audio.

Alur:
  score.py    : teks lagu -> not (midi, durasi, lirik)
  phonemes.py : suku kata -> fonem -> kontrol (luas saluran, derau, nasalisasi)
  timeline.py : rangkai kontrol per-sampel (pitch, Ps, x0, area saluran, noise)
  voice.py    : two-mass glotis + waveguide Kelly-Lochbaum -> tekanan di bibir

Hasil: WAV yang mengikuti melodi & lirik partitur.
"""
import numpy as np

import song as songmod
from phonemes import PH, phone_duration
from score import lyric_syllables
from vowels import VOWELS

NT = 44

# vokal -> indeks di tabel luas saluran (dipakai run_block di voice.py)
VOW_TABLE = np.array([VOWELS[k] for k in ("a", "e", "i", "o", "u")], dtype=float)
VOW_IDX = {k: i for i, k in enumerate(("a", "e", "i", "o", "u"))}


# ------------------------------------------------------------------ helpers
def _smoothstep(n, a=0.0, b=1.0):
    if n <= 0:
        return np.zeros(0)
    x = np.linspace(0.0, 1.0, n)
    s = x * x * (3.0 - 2.0 * x)
    return a + (b - a) * s


def _pitch_track(marks, t, vib_c, vib_hz):
    ts = np.array([m[0] for m in marks], dtype=float)
    vs = np.array([m[1] for m in marks], dtype=float)
    semis = np.interp(t, ts, vs)
    onset = np.clip(t / 0.25, 0, 1)          # vibrato muncul setelah nada stabil
    semis += (vib_c / 100.0) * onset * np.sin(2.0 * np.pi * vib_hz * t)
    return semis


def midi_to_freq(m):
    return 440.0 * 2.0 ** ((np.asarray(m, dtype=float) - 69.0) / 12.0)


# ------------------------------------------------------------------ diagnostik pitch
def estimate_f0(sig, fs, win=0.05, hop=0.01, fmin=60, fmax=700):
    """f0 per jendela (autokorelasi + cepstral-lite). Return (t, f0) dengan NaN bila senyap."""
    w, h = int(win * fs), int(hop * fs)
    lo, hi = int(fs / fmax), int(fs / fmin)
    tt, f0 = [], []
    rms_all = np.sqrt(np.mean(np.asarray(sig, float) ** 2)) + 1e-12
    for i in range(0, len(sig) - w, h):
        seg = sig[i:i + w].astype(float)
        lvl = np.sqrt(np.mean(seg ** 2))
        tt.append((i + w / 2) / fs)
        if lvl < 0.06 * rms_all:
            f0.append(np.nan)
            continue
        seg = seg - np.mean(seg)
        seg = seg * np.hanning(len(seg))
        ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
        ac /= ac[0] + 1e-12
        lo2 = min(lo, len(ac) - 2)
        hi2 = min(hi, len(ac) - 1)
        seg_ac = ac[lo2:hi2]
        if len(seg_ac) < 2:
            f0.append(np.nan)
            continue
        lag = lo2 + int(np.argmax(seg_ac))
        if ac[lag] < 0.35:
            f0.append(np.nan)
        else:
            f0.append(fs / lag)
    return np.array(tt), np.array(f0)


# ------------------------------------------------------------------ jadwal fonem
def plan_phrases(notes, spb, lead=0.12, tail=0.30, breath=0.55, legato=1.02):
    """
    notes -> (phrases, marks)
    phrases: list dict berisi segmen fonem [(t0,t1,fonem,jenis)] + t_awal/t_akhir
    marks  : [(detik, semitone)] untuk kurva pitch
    """
    phrases = []
    marks = []
    cur = lead
    prev_midi = None
    segs = []

    def close_phrase(end_t):
        nonlocal segs
        if segs:
            phrases.append({"t0": segs[0][0], "t1": end_t, "segs": segs})
        segs = []

    for nt in notes:
        dur = nt.beats * spb
        if nt.midi is None:
            if segs:
                close_phrase(cur)
                cur += breath
            else:
                cur += dur
            prev_midi = None
            continue

        semi = float(nt.midi)
        marks.append((cur, semi))
        marks.append((cur + min(0.08, 0.25 * dur), semi))
        # sedikit drift pada nada panjang
        if dur > 0.7:
            marks.append((cur + dur * 0.85, semi))

        syls = lyric_syllables(nt.lyric) if nt.lyric else []
        if not syls:
            # slur / tanpa lirik: ikuti vokal nada sebelumnya bila ada (setengah durasi nya)
            if segs:
                prev = segs[-1]
                segs[-1] = (prev[0], cur + dur, prev[2], prev[3])
            else:
                segs.append((cur, cur + dur, "v:a", "vowel"))
            prev_midi = nt.midi
            cur += dur
            continue

        # bagi durasi nada menjadi blok konsonan + vokal
        n_syl = len(syls)
        cons_total = 0.0
        for (on, nu, co) in syls:
            cons_total += sum(phone_duration(p) for p in on)
            if co:
                cons_total += sum(phone_duration(p) for p in co) * 0.8
        cons_total = min(cons_total, dur * 0.5)
        v_total = max(dur * 0.45, dur - cons_total)
        per_v = v_total / n_syl
        t = cur
        for k, (on, nu, co) in enumerate(syls):
            # onset
            for p in on:
                d = phone_duration(p) * (0.85 if k else 1.0)
                segs.append((t, t + min(d, max(0.012, per_v * 0.5)), "c:" + p, "cons"))
                t = segs[-1][1]
            # inti vokal
            vkey = "v:" + nu if nu in PH else "v:e"
            segs.append((t, t + per_v * 0.92, vkey, "vowel"))
            t = segs[-1][1]
            # coda (hanya suku terakhir yang biasanya membawa coda)
            if k == n_syl - 1 and co:
                for p in co:
                    d = phone_duration(p) * 0.8
                    segs.append((t, t + d, "c:" + p, "cons"))
                    t = segs[-1][1]
        prev_midi = nt.midi
        cur += dur

    if segs:
        close_phrase(cur + tail)
    return phrases, marks


# ------------------------------------------------------------------ kontrol per sampel
class Control:
    def __init__(self, n, fs):
        self.n = n
        self.fs = fs + 0
        self.voiced = np.zeros(n)
        self.a_cur = np.full((n, NT), 4.0)
        self.a_next = np.full((n, NT), 4.0)
        self.mix = np.zeros(n)
        self.nz = np.zeros(n)
        self.nz_lo = np.full(n, 1000.0)
        self.nz_hi = np.full(n, 4000.0)
        self.burst = np.zeros(n)
        self.burst_lo = np.full(n, 1000.0)
        self.burst_hi = np.full(n, 3000.0)
        self.nas = np.zeros(n)


def _paint_control(ctl, segs, fs):
    n = ctl.n
    prev_area = np.full(NT, 4.0)
    for (t0, t1, tag, kind) in segs:
        i0 = int(round(t0 * fs))
        i1 = min(n, int(round(t1 * fs)))
        if i1 <= i0:
            continue
        m = i1 - i0
        name = tag.split(":", 1)[1]
        info = PH.get(name, PH["e"])

        if kind == "cons":
            target = info["A"] * 1.0
            frac = np.linspace(0.0, 1.0, m) ** 0.55
            ctl.a_cur[i0:i1] = prev_area[None, :] + (target - prev_area)[None, :] * frac[:, None]
            ctl.a_next[i0:i1] = target
            ctl.mix[i0:i1] = 1.0
            ctl.voiced[i0:i1] = 1.0 if info["voiced"] else 0.0
            if info["nz"] > 0:
                env = np.sin(np.linspace(0, np.pi, m)) if m > 2 else np.ones(m)
                ctl.nz[i0:i1] = np.maximum(ctl.nz[i0:i1], info["nz"] * env)
                ctl.nz_lo[i0:i1] = info["nz_bp"][0]
                ctl.nz_hi[i0:i1] = info["nz_bp"][1]
            if info["burst"] > 0:
                nb = max(8, min(m, int(info["burst"] * fs)))
                ctl.burst[i0:i0 + nb] = np.exp(-np.linspace(0, 5, nb))
                ctl.burst_lo[i0:i0 + nb] = info["burst_bp"][0]
                ctl.burst_hi[i0:i0 + nb] = info["burst_bp"][1]
            if info["nas"]:
                ramp = np.sin(np.linspace(0, np.pi, m)) if m > 2 else np.ones(m)
                ctl.nas[i0:i1] = np.maximum(ctl.nas[i0:i1], ramp)
            prev_area = target
        else:
            target = info["A"]
            frac = np.clip(np.linspace(0.0, 0.3, m), 0, 1)
            ctl.a_cur[i0:i1] = prev_area[None, :] + (target - prev_area)[None, :] * frac[:, None]
            ctl.a_next[i0:i1] = target
            ctl.mix[i0:i1] = np.clip(np.linspace(0, 1.0, m) * 2.2, 0, 1)
            ctl.voiced[i0:i1] = 1.0
            prev_area = target


# ------------------------------------------------------------------ sintesis
def _simulate_q(q_arr, ps, x0, ctl, fs, oversample=4, chunk=None):
    """Jalankan glotis saja (tanpa derau) untuk mengukur f0 hasil simulasi."""
    n = len(q_arr)
    uv = np.zeros(n)
    zig = np.zeros(n)
    out = np.zeros(n)
    if chunk is None:
        chunk = int(fs * 0.5)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        _, ug = songmod.run_song(e - s, fs, ps[s:e].astype(np.float64), q_arr[s:e], x0[s:e],
                                 ctl.a_cur[s:e], ctl.a_next[s:e], ctl.mix[s:e],
                                 ctl.voiced[s:e], uv[s:e], ctl.nas[s:e],
                                 zig[s:e], zig[s:e], zig[s:e], oversample, 0.0)
        out[s:e] = ug
    return out


def _measure_vot(ug, fs, voiced):
    """
    Ukur f0 hanya pada sampel yang stabil (VOT > 120 ms setelah voiced mulai).
    Return (t, f).
    """
    tt, ff = estimate_f0(np.asarray(ug, dtype=float), fs, win=0.04, hop=0.02,
                         fmin=70, fmax=1100)
    vs = np.asarray(voiced, float) > 0.5
    ok = ~np.isnan(ff)
    if len(tt) == 0:
        return tt, ff
    # buang frame di dekat onset (kurang dari 120 ms dari awal voiced)
    idx = np.clip((tt * fs).astype(int), 0, len(vs) - 1)
    for j in range(len(tt)):
        k = idx[j]
        lo = max(0, k - int(0.12 * fs))
        if not vs[lo:k + 1].all():
            ok[j] = False
        # frame di akhir voiced (menuju senyap)
        hi = min(len(vs), k + int(0.03 * fs))
        if not vs[k:hi].any():
            ok[j] = False
    return tt[ok], ff[ok]


def _f0_direct(ug, fs, fs_ctrl, n):
    """
    Estimasi f0 langsung dari aliran glotis: jarak antar puncak positif.
    Lebih andal daripada autokorelasi untuk sinyal pulsa glotis.
    Return array f0 per sampel kontrol (NaN bila tidak valid).
    """
    x = np.asarray(ug, dtype=float)
    out = np.full(n, np.nan)
    xmax = np.max(x) if len(x) else 0.0
    if xmax <= 0:
        return out
    thr = 0.10 * xmax
    w = int(0.0035 * fs)         # jendela lokal 7 ms
    peaks = []
    i = w
    while i < len(x) - w:
        if x[i] > thr and x[i] == np.max(x[i - w:i + w]):
            peaks.append(i)
            i += w
        else:
            i += 1
    if len(peaks) < 3:
        return out
    # median interval adaptif -> tolak deteksi palsu (jarak <55% median)
    diffs = np.diff(peaks)
    med = np.median(diffs)
    keep = [peaks[0]]
    for p, d in zip(peaks[1:], diffs):
        if d >= 0.55 * med:
            keep.append(p)
    if len(keep) < 2:
        return out
    for a, b in zip(keep[:-1], keep[1:]):
        f = fs / (b - a)
        if 40.0 < f < 1500.0:
            k0 = int(a / fs * fs_ctrl)
            k1 = int(b / fs * fs_ctrl)
            out[k0:k1 + 1] = f
    return out


def calibrate_q(q_guess, f0_target, voiced, ctl, ps, x0, fs,
                iterations=3, f_lo=70.0, f_hi=1100.0):
    """
    Model two-mass: f0 ~ sqrt(k) -> koreksi q = q * f_target / f_ukur.
    Setelah q ditemukan, Ps dinaikkan agar getaran stabil pada nada tinggi
    (rms model turun tajam di register atas).
    """
    q = np.array(q_guess, dtype=float)
    target = np.array(f0_target, dtype=float)
    ps = np.array(ps, dtype=float)
    x0 = np.array(x0, dtype=float)
    sel = voiced > 0.5
    if not np.any(sel):
        return q, ps, x0

    for it in range(iterations):
        ug = _simulate_q(q, ps, x0, ctl, fs)
        meas = _f0_direct(ug, fs, fs, len(q))
        ok = ~np.isnan(meas) & sel
        if ok.sum() < int(0.02 * fs):
            break
        first, last = np.flatnonzero(ok)[0], np.flatnonzero(ok)[-1]
        meas[:first] = meas[first]
        meas[last + 1:] = meas[last]
        meas = np.interp(np.arange(len(q)), np.flatnonzero(ok), meas[ok])
        ratio = np.clip(target / np.maximum(meas, 1.0), 0.6, 1.7)
        ratio = np.where(sel, ratio, 1.0)
        k = int(0.05 * fs) | 1
        ker = np.hanning(k)
        ker /= ker.sum()
        ratio_s = np.convolve(ratio, ker, mode="same")
        edge = k // 2
        ratio_s[:edge] = ratio_s[edge]
        ratio_s[-edge:] = ratio_s[-edge - 1]
        q = np.clip(q * ratio_s, 0.3, 6.0)
        cerr = np.median(np.abs(12 * np.log2(np.maximum(meas[sel], 1) / target[sel])))
        if cerr < 0.4:      # < 40 cents
            break

    # level: naikkan Ps untuk nada tinggi agar osilasi kuat & stabil.
    # PENTING: jaga parameter di wilayah stabil model (q<=2.3, ps<=1900).
    # Di luar itu osilasi two-mass mati total -> suara kosong.
    hi = np.clip((target - 200.0) / 300.0, 0.0, 1.0)
    ps = ps * (1.0 + 2.2 * hi)
    x0 = x0 * (1.0 - 0.30 * hi)
    q = np.clip(q, 0.4, 2.3)
    ps = np.clip(ps, 0.0, 1900.0)
    x0 = np.clip(x0, 0.012e-2, 0.05e-2)
    return q, ps, x0


def synthesize_song(notes, fs=44100, tempo=100.0, q_base=1.25, vibrato_cents=22.0,
                    vibrato_hz=5.4, noise_amp=0.10, seed=0, **kw):
    spb = 60.0 / float(tempo)
    phrases, marks = plan_phrases(notes, spb, **kw)
    end_t = phrases[-1]["t1"] if phrases else 0.0
    n = int((end_t + 0.05) * fs)

    ctl = Control(n, fs)
    prev_area = np.full(NT, 4.0)
    for ph in phrases:
        prev_area = np.full(NT, 4.0)
        _paint_control(ctl, ph["segs"], fs)

    t = np.arange(n) / fs
    if marks:
        semis = _pitch_track(marks, t, vibrato_cents, vibrato_hz)
    else:
        semis = np.full(n, 60.0)
    f0 = midi_to_freq(semis)
    # tebakan awal q dari relasi empiris model (q=1.2 -> ~163 Hz), lalu dikalibrasi
    q = np.clip(f0 / 136.0, 0.5, 4.0) * (q_base / 1.25)

    # tekanan paru: naik awal frase, stabil, turun akhir frase
    ps = np.zeros(n)
    for ph in phrases:
        i0, i1 = int(ph["t0"] * fs), int(ph["t1"] * fs)
        i1 = min(i1, n)
        if i1 <= i0:
            continue
        ps[i0:i1] = np.interp(np.linspace(0, 1, i1 - i0),
                              [0, 0.08, 0.25, 0.9, 1.0],
                              [0, 620, 790, 780, 0])
    # nada tinggi butuh tekanan lebih
    ps *= np.clip(0.9 + 0.12 * (f0 / 220.0), 0.85, 1.15)

    # --- kalibrasi q terhadap pitch target (penting: model linear -> f0 ~ sqrt(k))
    # celah istirahat: JANGAN terlalu terbuka - x0 > ~2.6 mm pada vokal 'u'
    # membuat glotis berhenti bergetar (lihat tune_phonation.py)
    x0 = np.interp(f0, [80, 130, 260, 520], [0.024e-2, 0.022e-2, 0.018e-2, 0.015e-2])
    x0 = np.minimum(x0, 0.024e-2)
    x0 *= np.clip(1.0 + 0.10 * (1.0 - ctl.voiced), 1.0, 1.15)

    q, ps, x0 = calibrate_q(q, f0, ctl.voiced, ctl, ps, x0, fs)

    # ------------------------------------------------- stabilkan onset & level
    # 1) Ps minimal saat pita suara seharusnya bergetar (anti "nada mati").
    #    CATATAN FISIKA (tune_pitch.py / tune_phonation.py):
    #      - amplitudo osilasi ~ Ps/q^2 -> nada tinggi (q besar) menyusut & mudah mati
    #      - q=1.5 stabil di Ps>=900 ; q=2.0 butuh ~1500 ; q>=2.2 butuh ~1700+
    #      - vokal 'u' (beban F1 rendah) mati bila Ps<900 & x0>2.6 mm
    vs = np.convolve(ctl.voiced, np.hanning(int(0.08 * fs)) /
                     np.hanning(int(0.08 * fs)).sum(), mode="same")
    k = int(0.012 * fs) | 1
    ker = np.hanning(k)
    ker /= ker.sum()
    q_sm = np.convolve(q, ker, mode="same")
    q_norm = np.clip((q_sm - 1.5) / 0.8, 0.0, 1.2)
    ps_floor = (1000.0 + 900.0 * q_norm) * vs
    ps = np.maximum(ps, ps_floor)
    # 2) haluskan ps & x0 (buang tangga blok yang menyebabkan letupan kecil)
    ps = np.convolve(ps, ker, mode="same")
    x0 = np.convolve(x0, ker, mode="same")
    q = q_sm
    # 3) pre-phonation: buka celah sedikit saat mulai bersuara (hembusan awal),
    #    tetapi tetap di bawah batas mati osilasi (~2.6 mm pada vokal 'u')
    onset_env = np.clip(vs - np.roll(vs, int(0.03 * fs)), 0, None)
    onset_env = np.convolve(onset_env, ker, mode="same")
    x0 = x0 + 0.004e-2 * np.clip(onset_env * 8.0, 0, 1)
    x0 = np.minimum(x0, 0.026e-2)

    # ---- jalankan mesin fisika
    rng = np.random.default_rng(seed)
    uv = songmod.filter_noise(rng, ctl, fs, noise_amp)
    p_lip, ug = songmod.run_song(n, fs, ps.astype(np.float64), q, x0, ctl.a_cur, ctl.a_next,
                                 ctl.mix, ctl.voiced, uv, ctl.nas, ctl.burst,
                                 ctl.burst_lo, ctl.burst_hi, 4, noise_amp)
    globals()["_dbg"] = (ps.copy(), q.copy(), x0.copy(), f0.copy())
    return p_lip, ug, ctl, f0


def render_wav(path, y, fs=44100, gain=0.85):
    from scipy.io import wavfile
    y = np.asarray(y, dtype=np.float64)
    m = np.max(np.abs(y)) + 1e-12
    y = y / m * gain
    wavfile.write(path, fs, (y * 32767).astype(np.int16))
