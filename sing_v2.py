"""
sing_v2.py - Mesin nyanyi yang diperbaiki (Tahap 3, v2).

Perbaikan utama dibanding sing_song.py + timeline.py (lihat ANALISIS.md):
  1. Pitch: tiap nada DATAR, meluncur hanya ~80 ms di batas nada (dulu meluncur sepanjang nada).
  2. Lirik: satu suku kata per nada ("Bu_rung" -> Bu | rung); nada sambungan memperpanjang
     VOKAL, bukan konsonan penutup. Konsonan awal jatuh tepat sebelum ketukan.
  3. Fonasi: sumber glotis parametris (selalu berosilasi) + jitter/shimmer, menggantikan
     two-mass yang sering "mati" (osilasi hilang) di tengah nada.
  4. Vibrato muncul per nada panjang (tunda ~0,3 s), bukan terus-menerus dari detik 0,25.
  5. Konsonan: titik penyempitan yang benar (bibir=42, gusi=37, langit2 lunak=27),
     derau frikatif/ledakan diinjeksi DI titik penyempitan, bukan di glotis.
  6. Dinamika: amplitudo per suku kata (attack/release) + redup kecil saat nada diulang.
  7. Portamento: lompatan >= 2 semiton pada nada legato memakai peluruhan eksponensial
     (bukan smoothstep kaku), lompatan >= 5 semiton ditambah scoop 25-40 cents.
  8. Register falset (>= Bb4): oq lebih kecil, jitter turun, amplitudo -3 dB, dicampur 100 ms.
  9. Penutupan glotis lebih tajam: oq default 0,56 (rasio tp tetap 0,74).
 10. Vibrato organik (v2.1): laju 5,2-5,8 Hz diundi per frasa; tunda 0,35-0,45 s; naik 0,3 s;
     10-20% nada panjang lurus; kedalaman 0 / 60% / 100% menurut panjang nada (<0,45 / <0,9 / >=0,9 s);
     menyusut di akhir frasa.
 11. Dinamika ekspresif (v2.1): aksen +1,5-2 dB di ketukan 1, messa di voce (nada > 0,5 s, puncak 60%),
     busur frasa -3 dB di kedua ujung; level_vowels memakai ctl["dyn"] agar tidak meratakannya kembali.
 12. Timing manusiawi (v2.1): onset +-10-20 ms, napas 0,35-0,8 s acak, akhir frasa +5-8%
     (matikan dengan --no-human).
Saluran suara tetap waveguide Kelly-Lochbaum 44 segmen (vowels.py) -> tetap simulasi fisika.

Pakai:  py sing_v2.py songs/burung_kakatua.txt
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import scipy.signal as ss
from scipy.io import wavfile

try:
    from numba import njit
except ImportError:          # numba opsional (tanpa numba jauh lebih lambat)
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

from score import parse_score, lyric_syllables
from vowels import VOWELS

FS = 44100
NT = 44
BLOCK = 32
OUT = "output"

# ------------------------------------------------------------------ konstanta ekspresi
OQ_DEFAULT = 0.56            # open quotient dasar (lebih kecil = penutupan glotis lebih tajam)
TP_RATIO = 0.74              # tp = oq * TP_RATIO (tetap)
# portamento
PORT_MIN = 2.0               # semiton: di bawah ini pakai luncur kecil 80 ms
PORT_LEAD = 0.175            # detik: luncur mulai sebelum ketukan nada tujuan
PORT_SMALL_GL = 0.08         # detik: luncur untuk langkah kecil
SCOOP_MIN = 5.0              # semiton: lompatan >= ini diberi scoop
# register falset (nomor MIDI standar: Bb4 = 70, A4 = 69 = 440 Hz)
FAL_MIDI = 70                # nada penuh-falset mulai di sini (Bb4)
FAL_BLEND = 3                # nada FAL_MIDI-3 .. FAL_MIDI dicampur bertahap
FAL_TIME = 0.10              # detik: waktu campur antar register
FAL_OQ = 0.48                # oq falset
FAL_JITTER = 0.002           # jitter falset
FAL_DB = -3.0                # amplitudo falset (dB)

# --- ekspresi manusiawi (v2.1) -------------------------------------------------------------
VIB_HZ_RANGE = (5.2, 5.8)    # Hz: laju vibrato DIUNDI SEKALI PER FRASA (bukan per nada)
VIB_DELAY = (0.35, 0.45)     # detik: vibrato mulai setelah onset nada (acak per nada)
VIB_DELAY_FRAC = 0.40        # tunda juga dibatasi 40% panjang nada -> nada 0,5-0,9 s tetap bisa bervibrato
VIB_MIN_SPAN = 0.15          # detik: vibrato yang lebih pendek dari ini dibuang
VIB_RAMP = 0.30              # detik: naik bertahap sampai kedalaman penuh
VIB_STRAIGHT = (0.10, 0.20)  # peluang nada panjang dibiarkan lurus (diundi per lagu)
VIB_PHRASE_END = 0.40        # kedalaman menyusut sampai -40% di ujung frasa (mulai 70% frasa)
ACCENT_DB = (1.5, 2.0)       # aksen ketukan 1
MESSA_DB = 2.5               # messa di voce: crescendo-decrescendo, puncak di 60% nada (> 0,5 s)
ARCH_DB = 3.0                # busur frasa: -3 dB di kedua ujung, 0 dB di tengah
PHRASE_DYN_DB = 5.0          # v2.6: tiap frasa keras/pelan sendiri (+-5 dB, jarak pp-ff lebih lebar)
ONSET_JITTER = (0.010, 0.020)   # detik: onset tiap nada digeser +-10..20 ms
BREATH_RANGE = (0.35, 0.80)     # detik: panjang istirahat napas (acak)
PHRASE_STRETCH = (0.05, 0.08)   # nada terakhir frasa dipanjangkan 5-8%
RUBATO_RANGE = (0.92, 1.10)     # v2.6: skala tempo per frasa (rubato) 

# --- suara bernyanyi lebih hidup (v2.2) ----------------------------------------------------
DETUNE_CENTS = 6.0           # simpangan baku salah-nada tetap per nada (cents), dibatasi +-2x
SAG_CENTS = (0.0, 8.0)       # nada panjang (> 0,9 s) perlahan turun, dikoreksi di nada berikut
OVERSHOOT_CENTS = (8.0, 18.0)   # melampaui target di langkah legato 1-5 semiton, lalu mengendap
SCOOP_IN_CENTS = (20.0, 40.0)   # awal frasa: mulai di bawah nada, naik ~70 ms
BREATH_PHRASE_START = 3.0    # hembusan di awal frasa (kali lipat dasar, meluruh 80 ms)
BREATH_NOTE_START = 0.6      # hembusan di awal tiap nada (meluruh 50 ms)
BREATH_PHRASE_END = 3.0      # hembusan naik di ~0,1 s terakhir frasa
OQ_PHRASE_SPREAD = 0.03      # oq berbeda per frasa (+-), dan sedikit mengikuti dinamika
SINGER_DEFAULT = 1.0         # 0..1: kekuatan singer's formant (puncak resonansi ~2,9 kHz, hanya di vokal)

# v2.3 timbre: spektrum vokal manusia turun lebih cepat di atas 2 kHz (+1/-6/-12 dB per oktaf band)
TILT_LOW_HZ = 200.0          # low shelf: sedikit hangat di 200 Hz
TILT_LOW_DB = 1.5
TILT_HIGH_HZ = 1500.0        # high shelf: meredam "logam" 2-4 kHz
TILT_HIGH_DB = -8.0
TILT_F0_LO = 230.0           # di bawah f0 ini spektrum model sudah gelap -> shelf dikurangi
TILT_F0_HI = 420.0           # di atas f0 ini shelf penuh
TILT_DIP_HZ = 3200.0         # cekungan tambahan: pita 2-4 kHz masih terlalu terang di model
TILT_DIP_DB = -12.0
TILT_DIP_Q = 2.2
TILT_LP_HZ = 18000.0         # batas atas: 18 kHz (dulu 9 kHz -> sibilan /s/ /c/ tumpul)
BRANCH_IDX = 36              # v2.5 cabang samping: segmen faring (efek ke formant kecil)
BRANCH_LEN = 8               # panjang cabang (slot sub-langkah): f = fs/(2*len) ~ 2,76 kHz
BR_OPEN = 1.0                # ujung cabang: +1 tertutup (puncak resonansi), -1 terbuka (lekuk)
BRANCH_ALPHA = 0.0           # kekuatan gandengan cabang (0 = mati)
ROUGH_AMP = 0.03             # v2.4: modulasi amplitudo 20-40 Hz (3% -> getaran otot)
ROUGH_CENTS = 4.0            # v2.4: goyangan pitch 8 Hz (cents) pada skala penuh
COUPLE_DEFAULT = 0.10        # v2.7: kopel sumber-saluran (riak formant). 0 = sumber bebas (v2.6)
COUPLE_OQ = 0.05             # v2.7: lebar pulsa ikut berubah oleh tekanan balik (skala dgn couple/0.10; 0.18 membuat /i/ derau, HNR turun ~10 dB)
# v2.3 rahang: nada tinggi membuka vokal (F1 naik) supaya /i/ /u/ tidak menggelembung
JAW_REF_MIDI = 60.0
JAW_PER_SEMI = 0.028         # per semiton di atas JAW_REF_MIDI
JAW_MAX = 1.75               # batas pembukaan
SINGER_HZ = 2900.0
SINGER_Q = 2.5
SINGER_DB = 6.0              # gain puncak pada singer=1
# Catatan: penyempitan epilaring pada waveguide (3 cm, 0,45 cm^2) sudah dicoba -> pita 2,4-3,4 kHz
# tak berubah, F3 turun dan /u/ rusak. Jadi dipakai EQ puncak ber-gerbang vokal (lihat post_process).


def fal_weight(midi):
    """0 = register penuh, 1 = falset penuh; naik linear di FAL_BLEND semiton terakhir."""
    return float(np.clip((midi - (FAL_MIDI - FAL_BLEND)) / float(FAL_BLEND), 0.0, 1.0))

# ------------------------------------------------------------------ tabel fonem
# indeks tempat artikulasi pada 44 segmen (0 = glotis, 43 = bibir; 1 segmen = 0,4 cm)
BILAB, LABDEN, ALV, POSTALV, PAL, VEL, GLOT = 42.5, 41.5, 37.5, 35.0, 32.0, 27.0, 2.0

# v: bersuara | pl: tempat | ac: luas penyempitan cm^2 | dur: detik | kind
# nz: level derau frikatif | burst: level ledakan | asp: aspirasi setelah ledakan
CONS = {
    "p": dict(v=0, pl=BILAB, ac=0.02, dur=0.060, kind="stop", burst=1.0, asp=0.35),
    "b": dict(v=1, pl=BILAB, ac=0.02, dur=0.050, kind="stop", burst=0.45, asp=0.0),
    "t": dict(v=0, pl=ALV, ac=0.02, dur=0.060, kind="stop", burst=1.0, asp=0.35),
    "d": dict(v=1, pl=ALV, ac=0.02, dur=0.050, kind="stop", burst=0.45, asp=0.0),
    "k": dict(v=0, pl=VEL, ac=0.02, dur=0.065, kind="stop", burst=1.0, asp=0.40),
    "g": dict(v=1, pl=VEL, ac=0.02, dur=0.050, kind="stop", burst=0.45, asp=0.0),
    "c": dict(v=0, pl=PAL, ac=0.02, dur=0.090, kind="affr", burst=0.8, nz=0.9),
    "j": dict(v=1, pl=PAL, ac=0.02, dur=0.080, kind="affr", burst=0.4, nz=0.6),
    "s": dict(v=0, pl=ALV, ac=0.12, dur=0.110, kind="fric", nz=1.0),
    "z": dict(v=1, pl=ALV, ac=0.12, dur=0.090, kind="fric", nz=0.6),
    "sh": dict(v=0, pl=POSTALV, ac=0.25, dur=0.110, kind="fric", nz=1.0),
    "f": dict(v=0, pl=LABDEN, ac=0.15, dur=0.095, kind="fric", nz=0.55),
    "v": dict(v=1, pl=LABDEN, ac=0.15, dur=0.080, kind="fric", nz=0.40),
    "dh": dict(v=1, pl=LABDEN, ac=0.20, dur=0.080, kind="fric", nz=0.35),
    "x": dict(v=0, pl=VEL, ac=0.20, dur=0.100, kind="fric", nz=0.8),
    "h": dict(v=0, pl=GLOT, ac=2.5, dur=0.080, kind="asp", nz=0.010),
    "m": dict(v=1, pl=BILAB, ac=0.02, dur=0.075, kind="nasal"),
    "n": dict(v=1, pl=ALV, ac=0.02, dur=0.070, kind="nasal"),
    "ng": dict(v=1, pl=VEL, ac=0.02, dur=0.080, kind="nasal"),
    "ny": dict(v=1, pl=PAL, ac=0.02, dur=0.075, kind="nasal"),
    "l": dict(v=1, pl=ALV, ac=0.60, dur=0.060, kind="approx"),
    "r": dict(v=1, pl=ALV, ac=0.12, dur=0.035, kind="approx"),
    "rr": dict(v=1, pl=ALV, ac=0.12, dur=0.035, kind="approx"),
    "w": dict(v=1, pl=0, ac=0, dur=0.060, kind="glide", like="u"),
    "y": dict(v=1, pl=0, ac=0, dur=0.060, kind="glide", like="i"),
}
VOWEL_KEYS = ("a", "e", "i", "o", "u")


def vshape(k):
    return np.maximum(np.array(VOWELS[k], dtype=float), 0.25)


def narrow(base, pl, ac, width=1.25):
    idx = np.arange(NT)
    w = np.exp(-0.5 * ((idx - pl) / width) ** 2)
    return np.exp(np.log(base) * (1.0 - w) + np.log(ac) * w)


def smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def jaw_gain(midi):
    """v2.3: 1.0 di bawah JAW_REF_MIDI, naik sampai JAW_MAX untuk nada tinggi."""
    return 1.0 + np.clip((midi - JAW_REF_MIDI) * JAW_PER_SEMI, 0.0, JAW_MAX - 1.0)


def apply_jaw(a, g):
    """Buka vokal: lebar penyempitan dinaikkan dan digeser ke depan (rahang turun).

    a = luas 44 segmen, g >= 1 (1 = tanpa perubahan). Penyempitan dideteksi sendiri
    (indeks luas minimum), jadi aman untuk semua vokal.
    """
    if g <= 1.001:
        return a
    k = np.arange(a.shape[-1])
    ic = int(np.argmin(a))
    centre = ic + min(5.0, 2.0 * (g - 1.0))
    w = np.exp(-0.5 * ((k - centre) / (2.5 + 1.5 * (g - 1.0))) ** 2)
    return np.maximum(a * (1.0 + w * (g - 1.0)), 0.02)


# ------------------------------------------------------------------ 1. lirik -> suku kata
def assign_syllables(notes):
    """
    Kembalikan list per nada: None=rest, []=sambungan (tie), [syl,...]=suku kata nada ini.
    Suku kata ekstra dari token seperti 'Bu_rung' dipindah ke nada-nada berikutnya yang
    tidak punya lirik (dulu semuanya dijejalkan ke satu nada).
    """
    out = []
    pending = []
    last = None
    for i, nt in enumerate(notes):
        if nt.midi is None:
            if pending and last is not None:
                out[last].extend(pending)
            pending = []
            last = None
            out.append(None)
            continue
        lyr = nt.lyric if nt.lyric and nt.lyric not in ("-", "~", "_") else ""
        syl = lyric_syllables(lyr) if lyr else []
        if syl:
            if pending and last is not None:      # sisa suku kata tak kebagian nada
                out[last].extend(pending)
            pending = list(syl[1:])
            out.append([syl[0]])
            last = i
        elif pending:
            out.append([pending.pop(0)])
            last = i
        else:
            out.append([])                         # tie -> sambung vokal
    if pending and last is not None:
        out[last].extend(pending)
    return out


# ------------------------------------------------------------------ 2. rencana waktu
class Mark(tuple):
    """
    (t, midi) -- tetap bisa di-unpack seperti tuple lama -- ditambah metadata nada:
      beat  : posisi ketuk (jumlah beat partitur sebelum nada ini, istirahat ikut dihitung)
      beats : panjang partitur (beat)
      dur   : panjang NYATA nada (detik, sudah termasuk rubato)
      down  : True bila nada jatuh tepat di ketukan 1 birama (pickup = beat anakrusis sebelum ketukan 1)
      half  : True bila nada jatuh di tengah birama (ketukan 3 pada 4/4): aksen setengah kuat
      last  : True bila nada terakhir sebelum istirahat/akhir lagu (akhir frasa)
    """
    def __new__(cls, t, midi, beat=0.0, beats=0.0, dur=0.0, down=False, last=False, half=False):
        m = super().__new__(cls, (t, midi))
        m.beat, m.beats, m.dur, m.down, m.last, m.half = beat, beats, dur, down, last, half
        return m


def timing_rng(seed):
    """RNG khusus timing manusiawi (terpisah dari RNG kontrol; dipakai juga oleh check_v2)."""
    return np.random.default_rng(int(seed) + 7919)


def plan_song(notes, tempo, lead=0.30, tail=0.45, breath=0.35, rng=None, beats_per_bar=4.0,
              pickup=0.0, bar_reset=False):
    """
    rng=None -> timing kaku seperti v2 lama. rng diberikan -> timing manusiawi:
      * onset tiap nada digeser +-10..20 ms (acak, tidak menumpuk: grid nominal tetap)
      * istirahat = max(panjang partitur, acak 0,35-0,8 s)
      * nada terakhir frasa (sebelum istirahat) dipanjangkan 5-8%
    """
    spb = 60.0 / tempo
    assign = assign_syllables(notes)
    human = rng is not None
    # v2.6 rubato: tiap frasa punya skala tempo sendiri (manusia tidak metronomis)
    rubato_done = set()
    events = []        # tiap suku kata: dict
    marks = []         # Mark(t, midi, ...) tiap nada
    phrases = []       # (t_awal, t_akhir) bersuara
    cur = lead         # waktu nominal (akhir nada sebelumnya)
    beat = 0.0
    bar_origin = 0.0   # beat tempat hitungan birama dimulai (bar_reset: tiap awal frasa)
    ph_start = None
    ev_open = None
    prev_mark = None
    ph_rate = 1.0
    for i, (nt, sy) in enumerate(zip(notes, assign)):
        dur = nt.beats * spb * ph_rate
        if nt.midi is None:
            if ev_open is not None:
                ev_open["t1"] = cur
                ev_open = None
            if prev_mark is not None:
                prev_mark.dur = cur - prev_mark[0]
                prev_mark = None
            if ph_start is not None:
                phrases.append((ph_start, cur))
                ph_start = None
            cur += max(dur, float(rng.uniform(*BREATH_RANGE)) if human else breath)
            beat += nt.beats
            continue
        nxt_rest = (i + 1 >= len(notes)) or (notes[i + 1].midi is None)
        on = cur
        if human:
            off = float(rng.uniform(*ONSET_JITTER)) * (1.0 if rng.random() < 0.5 else -1.0)
            on = cur + off
            if prev_mark is not None:                    # jaga nada sebelumnya tidak < 0,10 s
                on = max(on, prev_mark[0] + 0.10)
        if ph_start is None:
            ph_start = on
            if human:
                ph_rate = float(rng.uniform(*RUBATO_RANGE))   # v2.6 rubato per frasa
            if bar_reset:
                bar_origin = beat
        if prev_mark is not None:
            prev_mark.dur = on - prev_mark[0]
        bpb = float(beats_per_bar)
        pos = ((beat - bar_origin) + (bpb - float(pickup) % bpb)) % bpb          # posisi di dalam birama (0 = ketukan 1)
        down = pos < 1e-3 or bpb - pos < 1e-3
        half = abs(pos - bpb / 2.0) < 1e-3
        prev_mark = Mark(on, float(nt.midi), beat=beat, beats=nt.beats, down=down, last=nxt_rest,
                         half=half)
        marks.append(prev_mark)
        if human and nxt_rest:
            dur *= 1.0 + float(rng.uniform(*PHRASE_STRETCH))
        if sy:
            if ev_open is not None:
                ev_open["t1"] = on
            ev_open = dict(t0=on, t1=max(cur + dur, on + 0.10), sylls=sy)
            events.append(ev_open)
        elif ev_open is not None:
            ev_open["t1"] = cur + dur               # tie: perpanjang vokal
        else:
            ev_open = dict(t0=on, t1=max(cur + dur, on + 0.10), sylls=[([], "a", [])])
            events.append(ev_open)
        cur += dur
        beat += nt.beats
    if ev_open is not None:
        ev_open["t1"] = cur
    if prev_mark is not None:
        prev_mark.dur = cur - prev_mark[0]
    if ph_start is not None:
        phrases.append((ph_start, cur))
    return events, marks, phrases, cur + tail


def _cons_dur(p):
    return CONS[p]["dur"] if p in CONS else 0.05


def cons_rng(seed):
    """RNG khusus durasi konsonan (dipakai juga oleh check_v2)."""
    return np.random.default_rng(int(seed) + 104729)


def rgb_segments(*a, **k):
    return None


def build_segments(events, rng=None):
    """
    events -> daftar segmen [(t0, t1, jenis, nama, info)] berurutan.
    jenis: 'cons' | 'vowel' ; vokal dengan diftong: nama 'ai' dll.
    Konsonan awal ditempatkan SEBELUM batas nada (vokal jatuh tepat di ketukan).
    rng: (v2.6) mengacak durasi konsonan agar tidak seragam.
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    # pecah event multi-suku-kata menjadi suku kata berurutan
    syl_list = []
    for ev in events:
        S = len(ev["sylls"])
        dt = (ev["t1"] - ev["t0"]) / S
        for j, (on, nu, co) in enumerate(ev["sylls"]):
            syl_list.append(dict(s0=ev["t0"] + j * dt, s1=ev["t0"] + (j + 1) * dt,
                                 on=[p for p in on if p in CONS], nu=nu,
                                 co=[p for p in co if p in CONS], ev=ev, first=(j == 0),
                                 rng=rng))
    # durasi onset/koda
    for s in syl_list:
        L = s["s1"] - s["s0"]
        # v2.6 variasi konsonan: penutupan plosif 35-90 ms & frikatif 60-140 ms, tidak seragam
        d_on = 0.0
        for p in s["on"]:
            c = CONS.get(p)
            if c and c["kind"] in ("stop", "affr"):
                d_on += 0.060 * s["rng"].uniform(0.58, 1.5)
            elif c and c["kind"] == "fric":
                d_on += 0.090 * s["rng"].uniform(0.7, 1.5)
            else:
                d_on += _cons_dur(p)
        d_co = sum(_cons_dur(p) for p in s["co"]) * 0.85
        cap_on = min(0.16, 0.45 * L)
        cap_co = min(0.13, 0.30 * L)
        s["k_on"] = min(1.0, cap_on / d_on) if d_on > 0 else 1.0
        s["k_co"] = min(1.0, cap_co / d_co) if d_co > 0 else 1.0
        s["d_on"] = d_on * s["k_on"]
        s["d_co"] = d_co * s["k_co"]
        # v2.6: pastikan vokal selalu tersisa >= 0,08 s walau konsonan diacak panjang
        room = max(0.0, L - 0.08)
        if s["d_on"] > 0.55 * room:
            s["k_on"] *= (0.55 * room) / s["d_on"]
            s["d_on"] = 0.55 * room
        if s["d_co"] > 0.45 * room:
            s["k_co"] *= (0.45 * room) / s["d_co"]
            s["d_co"] = 0.45 * room
    segs = []
    for i, s in enumerate(syl_list):
        nxt = syl_list[i + 1] if i + 1 < len(syl_list) else None
        nxt_on = nxt["d_on"] if (nxt and abs(nxt["s0"] - s["s1"]) < 1e-6) else 0.0
        v0 = s["s0"]
        v1 = s["s1"] - nxt_on - s["d_co"]
        if v1 - v0 < 0.08:                         # jaga vokal tidak menghilang
            v1 = v0 + 0.08
        # onset (tepat sebelum v0)
        t = v0 - s["d_on"]
        for p in s["on"]:
            d = _cons_dur(p) * s["k_on"]
            segs.append((t, t + d, "cons", p, dict(side="on", vowel=s["nu"])))
            t += d
        segs.append((v0, v1, "vowel", s["nu"], dict(ev=s["ev"], s0=s["s0"], first=s["first"],
                                                   has_on=bool(s["on"]))))
        t = v1
        for p in s["co"]:
            d = _cons_dur(p) * 0.85 * s["k_co"]
            segs.append((t, t + d, "cons", p, dict(side="co", vowel=s["nu"])))
            t += d
    segs.sort(key=lambda x: x[0])
    return segs


# ------------------------------------------------------------------ 3. kontrol per-sampel
def _vowel_target(name):
    """nama vokal (mungkin diftong 'ai') -> (bentuk awal, bentuk akhir)"""
    if len(name) == 2 and name[0] in VOWEL_KEYS and name[1] in VOWEL_KEYS:
        return vshape(name[0]), vshape(name[1])
    k = name if name in VOWEL_KEYS else "e"
    return vshape(k), vshape(k)


def vib_scale(L):
    """Fraksi kedalaman vibrato menurut panjang nada L (detik): 0 | 0,6 | 1,0."""
    if L < 0.45:
        return 0.0
    return 0.6 if L < 0.90 else 1.0


def build_controls(segs, marks, phrases, total_t, fs, rng, vib_depth=28.0, vib_hz=5.5,
                   jitter=0.004, shimmer=0.03, falsetto=True, detune=DETUNE_CENTS, rough=1.0):
    n = int(total_t * fs) + 1
    t = np.arange(n) / fs

    # ------------------------------------------------ pitch: datar + portamento
    # langkah kecil (< 2 semiton)            : smoothstep 80 ms (seperti dulu)
    # lompatan >= 2 semiton, nada legato     : peluruhan eksponensial dari pitch sebelumnya,
    #     cents = target - d*exp(-(t-t0)/tau), tau 40-60 ms, mulai PORT_LEAD sebelum ketukan
    # lompatan >= 5 semiton                  : + scoop 25-40 cents (tiba sedikit kurang, lalu koreksi)
    # nada tujuan berkonsonan awal           : bukan legato -> smoothstep lama
    cents = np.zeros(n)
    m0 = marks[0][1]
    cents[:] = m0 * 100.0
    cons_onsets = np.array([info["s0"] for (_a, _b, k_, _n, info) in segs
                            if k_ == "vowel" and info.get("first") and info.get("has_on")])

    def add_exp(t_start, amp_c, tau):
        i0 = max(0, int(t_start * fs))
        i1 = min(n, int((t_start + 12.0 * tau) * fs))
        if i0 >= n:
            return
        cents[i0:i1] += amp_c * (1.0 - np.exp(-(t[i0:i1] - t_start) / tau))
        cents[i1:] += amp_c

    for (ta, ma), (tb, mb) in zip(marks[:-1], marks[1:]):
        dsemi = mb - ma
        d = dsemi * 100.0
        if d == 0:
            continue
        adj = abs(dsemi)
        legato = not (len(cons_onsets) and np.any(np.abs(cons_onsets - tb) < 1e-3))
        if adj >= PORT_MIN and legato:
            t_start = tb - min(PORT_LEAD, 0.6 * (tb - ta))
            tau = float(np.clip(0.040 + 0.002 * adj, 0.040, 0.060))
            if adj >= SCOOP_MIN:
                sc = float(np.clip(25.0 + 3.0 * (adj - SCOOP_MIN), 25.0, 40.0))
                sg_ = np.sign(d)
                add_exp(t_start, d - sg_ * sc, tau)           # tiba sedikit di bawah/atas target
                add_exp(tb, sg_ * sc, 0.050)                  # lalu koreksi ke target
            else:
                add_exp(t_start, d, tau)
        else:
            if adj >= PORT_MIN:
                gl = min(0.14, 0.06 + 0.008 * adj)            # berkonsonan: luncur lama
            else:
                gl = PORT_SMALL_GL
            tstart = tb - 0.04
            cents += d * smoothstep((t - tstart) / gl)
    # ---- intonasi manusiawi (v2.2): tidak ada penyanyi yang tepat 0 cent
    #  (a) overshoot kecil di langkah legato 1-5 semiton, lalu mengendap (~60 ms)
    #  (b) awal frasa: mulai 20-40 cents di bawah nada dan naik (~70 ms)
    #  (c) salah-nada tetap per nada (sigma = detune cents) + nada panjang perlahan turun (sag)
    for (ta, ma), (tb, mb) in zip(marks[:-1], marks[1:]):
        adj = abs(mb - ma)
        if 1.0 <= adj < SCOOP_MIN and not (len(cons_onsets) and np.any(np.abs(cons_onsets - tb) < 1e-3)):
            ov = float(np.sign(mb - ma) * rng.uniform(*OVERSHOOT_CENTS))
            j0 = max(0, int((tb - 0.03) * fs))
            j1 = min(n, int((tb + 0.40) * fs))
            tt_ = t[j0:j1] - tb
            cents[j0:j1] += ov * smoothstep((tt_ + 0.03) / 0.05) * np.exp(-np.maximum(tt_, 0.0) / 0.07)
    for (pa, pb) in phrases:
        j0, j1 = int(pa * fs), min(n, int((pa + 0.5) * fs))
        cents[j0:j1] -= float(rng.uniform(*SCOOP_IN_CENTS)) * np.exp(-(t[j0:j1] - pa) / 0.07)
    if detune > 0:
        step = np.zeros(n)
        for k, m in enumerate(marks):
            tk = m[0]
            tn = marks[k + 1][0] if k + 1 < len(marks) else total_t
            j0, j1 = int(tk * fs), min(n, int(tn * fs))
            if j1 - j0 < 2:
                continue
            o = float(np.clip(rng.normal(0.0, detune), -2 * detune, 2 * detune))
            seg_ = np.full(j1 - j0, o)
            if tn - tk > 0.9:
                seg_ -= float(rng.uniform(*SAG_CENTS)) * smoothstep(np.arange(j1 - j0) / (j1 - j0))
            step[j0:j1] = seg_
        ks = int(0.07 * fs) | 1
        ker_s = np.hanning(ks)
        ker_s /= ker_s.sum()
        cents += ss.fftconvolve(np.pad(step, ks // 2, mode="edge"), ker_s, "valid")[:n]
    # drift lambat + vibrato per nada panjang
    drift = np.convolve(rng.standard_normal(n), np.hanning(int(0.35 * fs)), "same")
    drift = drift / (np.std(drift) + 1e-12) * 5.0
    cents += drift
    # ---- vibrato organik: laju diundi per FRASA; kedalaman fungsi panjang nada
    #      (0 di bawah 0,45 s | 60% pada 0,45-0,9 s | 100% di atas 0,9 s); tunda 0,35-0,45 s,
    #      naik 0,3 s; 10-20% nada panjang sengaja lurus; menyusut di akhir frasa.
    vib = np.zeros(n)
    vib_env_all = np.zeros(n)
    evs, seen_ev = [], set()
    for sg in segs:
        if sg[2] == "vowel" and sg[4].get("first", False) and id(sg[4]["ev"]) not in seen_ev:
            seen_ev.add(id(sg[4]["ev"]))
            evs.append(sg[4]["ev"])
    p_straight = float(rng.uniform(*VIB_STRAIGHT))
    for (pa, pb) in phrases:
        i0p, i1p = int(pa * fs), min(n, int(pb * fs))
        Lp = i1p - i0p
        if Lp < int(0.45 * fs):
            continue
        rate = float(rng.uniform(*VIB_HZ_RANGE))                      # satu laju per frasa
        wander = np.cumsum(0.4 * rng.standard_normal(Lp)) / fs
        wander = wander - np.linspace(0, wander[-1], Lp)
        ph = 2 * np.pi * (rate * np.arange(Lp) / fs + 0.3 * wander)   # fase mengalir sepanjang frasa
        for ev in evs:
            if not (pa - 1e-6 <= ev["t0"] < pb):
                continue
            frac = vib_scale(ev["t1"] - ev["t0"])
            if frac <= 0.0 or rng.random() < p_straight:
                continue
            a = ev["t0"] + min(float(rng.uniform(*VIB_DELAY)), VIB_DELAY_FRAC * (ev["t1"] - ev["t0"]))
            b = min(ev["t1"] - 0.10, pb)                              # berhenti sebelum akhir nada
            if b - a < VIB_MIN_SPAN:
                continue
            i0, i1 = max(int(a * fs), i0p), min(int(b * fs), i1p)
            L = i1 - i0
            if L <= 1:
                continue
            r = np.arange(L)
            ramp = max(1.0, min(VIB_RAMP * fs, L / 2))
            env = smoothstep(np.minimum(1.0, np.minimum(r / ramp, (L - r) / (0.1 * fs))))
            x = (i0 + r - i0p) / Lp                                   # progres frasa 0..1
            shrink = 1.0 - VIB_PHRASE_END * smoothstep((x - 0.70) / 0.30)
            v = env * shrink * frac * np.sin(ph[i0 - i0p:i1 - i0p])
            vib[i0:i1] = vib_depth * v
            vib_env_all[i0:i1] = v
    cents += vib
    f0 = 440.0 * 2.0 ** ((cents / 100.0 - 69.0) / 12.0)

    # ------------------------------------------------ bentuk saluran (keyframe)
    # tentukan bentuk untuk tiap segmen; konsonan meminjam bentuk vokal tetangga
    shapes = []
    vowel_idx = [i for i, s in enumerate(segs) if s[2] == "vowel"]
    for i, sg in enumerate(segs):
        t0, t1, kind, name, info = sg
        if kind == "vowel":
            a, b = _vowel_target(name)
            shapes.append((a, b))
        else:
            c = CONS[name]
            vn = info["vowel"]
            va, vb = _vowel_target(vn)
            base = va if info["side"] == "on" else vb
            if c["kind"] == "glide":
                sh = vshape(c["like"]) * np.where(np.arange(NT) > 36, 0.55, 1.0) if c["like"] == "u" \
                    else narrow(vshape("i"), 31, 0.8, 2.0)
                shapes.append((sh, sh))
            else:
                shapes.append((narrow(base, c["pl"], c["ac"]),) * 2)

    kt, kA = [], []
    nseg = len(segs)
    for i, sg in enumerate(segs):
        t0, t1, kind, name, info = sg
        a, b = shapes[i]
        # tr antar segmen
        def tr_with(j):
            if j < 0 or j >= nseg:
                return 0.05
            k1, k2 = segs[i][2], segs[j][2]
            return 0.085 if (k1 == "vowel" and k2 == "vowel") else 0.028
        tin = min(tr_with(i - 1), 0.8 * (t1 - t0))
        tout = min(tr_with(i + 1), 0.8 * (t1 - t0))
        if kind == "vowel" and a is not b:            # diftong: tahan vokal-1 lalu geser ke vokal-2
            mid = t0 + 0.55 * (t1 - t0)
            kt += [t0 + tin / 2, mid, t1 - tout / 2]
            kA += [np.log(a), np.log(a), np.log(b)]
        else:
            kt += [t0 + tin / 2, t1 - tout / 2]
            kA += [np.log(a), np.log(a)]
    kt = np.array(kt)
    kA = np.array(kA)
    # jaga urutan waktu menaik
    for i in range(1, len(kt)):
        if kt[i] <= kt[i - 1]:
            kt[i] = kt[i - 1] + 1e-4
    nb = (n + BLOCK - 1) // BLOCK + 1
    tb = np.arange(nb) * BLOCK / fs
    areas = np.empty((nb, NT))
    for k in range(NT):
        areas[:, k] = np.exp(np.interp(tb, kt, kA[:, k]))
    # haluskan sedikit antar blok (kontinu, bebas klik)
    kern = np.hanning(9)
    kern /= kern.sum()
    for k in range(NT):
        areas[:, k] = np.convolve(np.pad(areas[:, k], 4, mode="edge"), kern, "valid")

    # v2.3 rahang: nada tinggi -> vokal dibuka (F1 naik), per blok mengikuti pitch
    if len(marks):
        mt = np.array([m[0] for m in marks], dtype=float)
        mm = np.array([m[1] for m in marks], dtype=float)
        midi_b = np.interp(tb, mt, mm)
        for b in np.flatnonzero(jaw_gain(midi_b) > 1.001):
            areas[b] = apply_jaw(areas[b], float(jaw_gain(midi_b[b])))

    # ------------------------------------------------ amplitudo bersuara, derau, tap hidung
    amp = np.zeros(n)
    tap = np.zeros(n)
    nz = np.zeros(n)
    nzidx = np.full(nb, 37, dtype=np.int32)
    asp = np.zeros(n)
    voiced_gate = np.zeros(n)

    amp_level = {"stop": 0.30, "affr": 0.30, "fric": 0.45, "asp": 0.0, "nasal": 0.80,
                 "approx": 0.85, "glide": 0.90}
    vgate = np.zeros(n)                            # v2.5: 1 di vokal (gerbang cabang samping)
    for sg in segs:
        t0, t1, kind, name, info = sg
        i0, i1 = int(round(t0 * fs)), min(n, int(round(t1 * fs)))
        if i1 <= i0:
            continue
        m = i1 - i0
        if kind == "vowel":
            amp[i0:i1] = 1.0
            vgate[i0:i1] = 1.0
            continue
        c = CONS[name]
        tap_level = {"nasal": 1.0, "stop": 0.55, "approx": 0.0, "fric": 0.25}.get(c["kind"], 0.0)
        if c["v"]:
            amp[i0:i1] = amp_level.get(c["kind"], 0.4)
            tap[i0:i1] = tap_level
        b0, b1 = int(t0 * fs / BLOCK), int(math.ceil(t1 * fs / BLOCK))
        nzidx[b0:b1 + 1] = int(round(min(max(c["pl"], 1), NT - 2))) if c["kind"] != "glide" else 37
        if c["kind"] in ("fric", "asp", "affr"):
            env = np.sin(np.linspace(0, np.pi, m)) ** 0.6
            if c["kind"] == "affr":
                env = np.linspace(0.3, 1.0, m) * (np.linspace(1, 0.4, m))
                env *= np.r_[np.zeros(m // 3), np.ones(m - m // 3)]
            nz[i0:i1] += c.get("nz", 0.5) * env * rng.standard_normal(m)
        if c["kind"] in ("stop", "affr"):          # ledakan di AKHIR penutupan (pelepasan)
            nbur = int(0.012 * fs)
            e = np.exp(-np.arange(nbur) / (0.0035 * fs))
            j0 = i1
            j1 = min(n, j0 + nbur)
            nz[j0:j1] += c["burst"] * 0.12 * e[:j1 - j0] * rng.standard_normal(j1 - j0)
            if c.get("asp", 0) > 0:               # aspirasi (VOT) setelah ledakan tak bersuara
                na = int(0.022 * fs * rng.uniform(0.6, 1.7))   # v2.6 VOT 13-37 ms
                a1 = min(n, j0 + na)
                asp[j0:a1] += 0.4 * c["asp"] * np.hanning(a1 - j0) * rng.standard_normal(a1 - j0)
            b0, b1 = int(j0 / BLOCK), int(j1 / BLOCK) + 1
            nzidx[b0:b1 + 1] = int(round(c["pl"]))
        if c["kind"] == "asp":
            nzidx[b0:b1 + 1] = 3

    # envelope suku kata: attack/release halus + redup kecil di batas nada
    k = int(0.018 * fs) | 1
    ker = np.hanning(k)
    ker /= ker.sum()
    amp = np.convolve(amp, ker, "same")
    for sg in segs:
        t0, t1, kind, name, info = sg
        if kind == "vowel" and info["first"] and not info["has_on"] and t0 > 0.35:
            g = 1.0 - 0.45 * np.exp(-0.5 * ((t - t0) / 0.022) ** 2)
            amp *= g
    # awal/akhir frasa
    for (pa, pb) in phrases:
        i0, i1 = int(pa * fs), min(n, int(pb * fs))
        fi = min(int(0.05 * fs), (i1 - i0) // 2)
        fo = min(int(0.12 * fs), (i1 - i0) // 2)
        amp[i0:i0 + fi] *= smoothstep(np.arange(fi) / fi)
        amp[i1 - fo:i1] *= smoothstep(np.arange(fo)[::-1] / fo)
    # gerbang: di luar frasa diam
    gate = np.zeros(n)
    for (pa, pb) in phrases:
        gate[int(pa * fs):min(n, int(pb * fs))] = 1.0
    amp *= gate
    # tremolo kecil mengikuti vibrato (±1 dB)
    amp *= 1.0 + 0.10 * vib_env_all
    # dinamika ekspresif (dB): aksen ketukan 1 + messa di voce (nada > 0,5 s) + busur frasa.
    # Disimpan di ctl["dyn"] supaya level_vowels tahu ini disengaja (tidak diratakan kembali).
    dyn_db = np.zeros(n)
    for m in marks:
        on_, dur_ = m[0], getattr(m, "dur", 0.0)
        j0, j1 = int(on_ * fs), min(n, int((on_ + dur_) * fs))
        if j1 - j0 < 2:
            continue
        if getattr(m, "down", False):
            dyn_db[j0:j1] += float(rng.uniform(*ACCENT_DB))
        elif getattr(m, "half", False):
            dyn_db[j0:j1] += 0.5 * float(rng.uniform(*ACCENT_DB))     # ketukan 3: setengah aksen
        if dur_ > 0.5:
            depth_m = MESSA_DB * float(np.clip((dur_ - 0.5) / 0.3, 0.0, 1.0))   # naik halus dari 0,5 s
            xm = np.arange(j1 - j0) / (j1 - j0)
            wm = np.where(xm < 0.6, smoothstep(xm / 0.6), smoothstep((1.0 - xm) / 0.4))
            dyn_db[j0:j1] += depth_m * wm
    for (pa, pb) in phrases:
        j0, j1 = int(pa * fs), min(n, int(pb * fs))
        if j1 - j0 < 2:
            continue
        xp = np.arange(j1 - j0) / (j1 - j0)
        dyn_db[j0:j1] += -ARCH_DB * (2.0 * xp - 1.0) ** 2
        # v2.6 kekuatan frasa: tiap frasa punya tingkat kenyaringan sendiri
        lvl_p = float(rng.normal(0.0, PHRASE_DYN_DB / 2.0))
        dyn_db[j0:j1] += np.clip(lvl_p, -PHRASE_DYN_DB, PHRASE_DYN_DB) * np.ones(j1 - j0)
    kd = int(0.04 * fs) | 1
    ker_d = np.hanning(kd)
    ker_d /= ker_d.sum()
    dyn_db = ss.fftconvolve(np.pad(dyn_db, kd // 2, mode="edge"), ker_d, "valid")[:n]
    dyn = 10.0 ** (dyn_db / 20.0)
    amp *= dyn
    # v2.4 kekasaran sumber: getaran otot tak sempurna (roughness 20-40 Hz) di amplitudo,
    # plus goyangan pitch lambat. Manusia tidak pernah benar-benar halus.
    if rough > 0.0:
        kq = int(0.008 * fs) | 1                      # ~40 Hz
        ker_q = np.hanning(kq)
        ker_q /= ker_q.sum()
        rq = ss.fftconvolve(np.pad(rng.standard_normal(n), kq // 2, mode="edge"), ker_q, "valid")[:n]
        rq /= (np.std(rq) + 1e-12)
        amp *= 1.0 + ROUGH_AMP * rough * rq * gate
        kq2 = int(0.035 * fs) | 1                     # ~8 Hz
        ker_q2 = np.hanning(kq2)
        ker_q2 /= ker_q2.sum()
        rp = ss.fftconvolve(np.pad(rng.standard_normal(n), kq2 // 2, mode="edge"), ker_q2, "valid")[:n]
        rp /= (np.std(rp) + 1e-12)
        cents += ROUGH_CENTS * rough * rp * gate
        f0 = 440.0 * 2.0 ** ((cents / 100.0 - 69.0) / 12.0)
    # hembusan (breathiness) naik di awal nada/frasa dan di akhir frasa; oq berbeda per frasa
    breath_env = np.ones(n)
    for (pa, pb) in phrases:
        j0, j1 = int(pa * fs), min(n, int((pa + 0.5) * fs))
        breath_env[j0:j1] += BREATH_PHRASE_START * np.exp(-(t[j0:j1] - pa) / 0.08)
        e0, e1 = max(0, int((pb - 0.5) * fs)), min(n, int(pb * fs))
        breath_env[e0:e1] += BREATH_PHRASE_END * np.exp(-(pb - t[e0:e1]) / 0.10)
    for m in marks:
        j0, j1 = int(m[0] * fs), min(n, int((m[0] + 0.3) * fs))
        breath_env[j0:j1] += BREATH_NOTE_START * np.exp(-(t[j0:j1] - m[0]) / 0.05)
    oq_off = np.zeros(n)
    for (pa, pb) in phrases:
        oq_off[int(pa * fs):min(n, int(pb * fs))] = float(rng.uniform(-OQ_PHRASE_SPREAD, OQ_PHRASE_SPREAD))
    oq_off -= 0.004 * dyn_db                                      # lebih keras -> glotis menutup lebih tajam
    ko = int(0.3 * fs) | 1
    ker_o = np.hanning(ko)
    ker_o /= ker_o.sum()
    oq_off = ss.fftconvolve(np.pad(oq_off, ko // 2, mode="edge"), ker_o, "valid")[:n]

    # register falset: bobot 0..1 per sampel, dicampur FAL_TIME (100 ms) antar register
    fal = np.zeros(n)
    if falsetto:
        step = np.zeros(n)
        for kk, (tk, mk) in enumerate(marks):
            tn = marks[kk + 1][0] if kk + 1 < len(marks) else total_t
            step[int(tk * fs):min(n, int(tn * fs))] = fal_weight(mk)
        if step.any():
            kf = int(FAL_TIME * fs) | 1
            ker_f = np.hanning(kf)
            ker_f /= ker_f.sum()
            fal = ss.fftconvolve(np.pad(step, kf // 2, mode="edge"), ker_f, "valid")[:n]
            fal = np.clip(fal, 0.0, 1.0)
        amp *= 10.0 ** (FAL_DB * fal / 20.0)

    return dict(n=n, f0=f0, amp=amp, areas=areas, nz=nz, nzidx=nzidx, tap=tap, asp=asp,
                jitter=jitter, shimmer=shimmer, cents=cents, fal=fal, dyn=dyn, breath=breath_env,
                oq_off=oq_off, vgate=vgate)


# ------------------------------------------------------------------ 4. sumber glotis
@njit(cache=True)
def glottal_flow(f0, amp, asp, fs, jit, shim, oq, seed):
    """
    Aliran glotis tipe Rosenberg/LF disederhanakan: fase membuka sinusoid, menutup cosinus
    (penutupan tajam -> spektrum ~ -12 dB/oct), jitter & shimmer per periode.
    jit dan oq berupa ARRAY per sampel (register falset mengubahnya mulus); tp = oq * 0,74.
    """
    np.random.seed(seed)
    n = f0.shape[0]
    out = np.zeros(n)
    flow = np.zeros(n)
    phase = 0.0
    jf = 1.0
    sh = 1.0
    oq_noise = 0.0
    for i in range(n):
        phase += f0[i] * jf / fs
        if phase >= 1.0:
            phase -= 1.0
            jf = 1.0 + jit[i] * np.random.randn()
            sh = 1.0 + shim * np.random.randn()
            if sh < 0.5:
                sh = 0.5
            oq_noise = 0.004 * np.random.randn()
        oqv = oq[i] + oq_noise
        tp = oqv * 0.74
        if phase < tp:
            u = 0.5 * (1.0 - math.cos(math.pi * phase / tp))
        elif phase < oqv:
            u = math.cos(0.5 * math.pi * (phase - tp) / (oqv - tp))
        else:
            u = 0.0
        out[i] = amp[i] * sh * u + asp[i]
        flow[i] = u
    return out, flow


# ------------------------------------------------------------------ 5. waveguide
@njit(cache=True)
def tract_run(n, src, nz, nzidx, tap, refl, loss, lip_refl, rg, tap_g, tap_a,
              br_idx, br_gain, br_len, alpha, br_open):
    nseg = refl.shape[1]
    R = np.zeros(nseg)
    Lw = np.zeros(nseg)
    JR = np.empty(nseg)
    JL = np.empty(nseg + 1)
    rc = np.empty(nseg)
    out = np.zeros(n)
    lp = 0.0
    bd = np.zeros(br_len)          # v2.5 cabang samping: gelombang di dalam cabang
    bi = 0
    wbi = 0.0
    for i in range(n):
        b = i // BLOCK
        u = (i - b * BLOCK) / BLOCK
        w = 0.5 - 0.5 * math.cos(math.pi * u)
        for k in range(nseg):
            rc[k] = refl[b, k] * (1.0 - w) + refl[b + 1, k] * w
        inj = src[i]
        z = nz[i]
        kz = nzidx[b]
        lip = 0.0
        gam = 0.5 * alpha * br_gain[i]         # setengah: dua sub-langkah per sampel
        g1 = br_gain[i]
        bo = 0.0
        for _t in range(2):
            JR[0] = rg * Lw[0] + inj
            for k in range(1, nseg):
                d = R[k - 1] - Lw[k]
                wu = R[k - 1] + rc[k] * d          # percabangan klasik
                wd = Lw[k] + rc[k] * d
                if k == br_idx and gam > 1e-4:
                    pj = 2.0 * (R[k - 1] + Lw[k] + gam * wbi) / (2.0 + gam)
                    wd = (1.0 - g1) * wd + g1 * (pj - Lw[k])
                    wu = (1.0 - g1) * wu + g1 * (pj - R[k - 1])
                    bo = gam * (pj - wbi)
                JR[k] = wu
                JL[k] = wd
            JL[nseg] = lip_refl * R[nseg - 1]
            if z != 0.0:
                JR[kz] += 0.5 * z
                JL[kz] += 0.5 * z
            for k in range(nseg):
                R[k] = JR[k] * loss
                Lw[k] = JL[k + 1] * loss
            lip += R[nseg - 1] * (1.0 + lip_refl)
            # cabang maju satu slot TIAP sub-langkah (konsisten dengan laju 2x saluran)
            if gam > 1e-4:
                wbi = br_open * bd[bi]
                bd[bi] = bo * 0.995
                bi += 1
                if bi >= br_len:
                    bi = 0
        pn = R[12] + Lw[12]                       # tekanan dekat velum -> murmur hidung / bar suara
        lp += tap_a * (pn - lp)
        out[i] = 0.5 * lip + tap_g * tap[i] * lp
    return out


def reflections(areas):
    r = np.zeros_like(areas)
    r[:, 1:] = (areas[:, :-1] - areas[:, 1:]) / (areas[:, :-1] + areas[:, 1:])
    return r


@njit(cache=True)
def tract_run_coupled(n, f0, amp, asp, oq, jit, shim, seed, zs, k_coup, k_oq, fs,
                      nz, nzidx, tap, refl, loss, lip_refl, rg, tap_g, tap_a,
                      br_idx, br_gain, br_len, alpha, br_open):
    """v2.7: sumber glotis dibangkitkan DI DALAM loop waveguide, sehingga gelombang balik
    dari saluran (riak formant) memodulasi pulsa berikutnya -> interaksi sumber-filter.

    Pulsa = Rosenberg/LF seperti `glottal_flow`, tetapi amplitudo dan lebar pulsa dikoreksi
    oleh tekanan balik ternormalisasi di glotis (Lw[0]): u -> u * (1 + k_coup * fb),
    oq -> oq * (1 + k_oq * fb). fb dinormalkan lambat sehingga selalu di [-1, 1] (stabil).
    """
    nseg = refl.shape[1]
    R = np.zeros(nseg)
    Lw = np.zeros(nseg)
    JR = np.empty(nseg)
    JL = np.empty(nseg + 1)
    rc = np.empty(nseg)
    out = np.zeros(n)
    lp = 0.0
    bd = np.zeros(br_len)
    bi = 0
    wbi = 0.0
    np.random.seed(seed)
    phase = 0.0
    jf = 1.0
    sh = 1.0
    oq_noise = 0.0
    lw0 = 0.0
    fbe = 0.0
    lp_hp = 0.0
    lp_hp2 = 0.0
    for i in range(n):
        # ---- sumber glotis (terkopel)
        phase += f0[i] * jf / fs
        if phase >= 1.0:
            phase -= 1.0
            jf = 1.0 + jit[i] * np.random.randn()
            sh = 1.0 + shim * np.random.randn()
            if sh < 0.5:
                sh = 0.5
            oq_noise = 0.004 * np.random.randn()
        fb = 0.0
        if fbe > 1e-12:
            fb = lw0 / fbe
            if fb > 1.0:
                fb = 1.0
            elif fb < -1.0:
                fb = -1.0
        oqv = oq[i] * (1.0 + k_oq * fb) + oq_noise
        if oqv < 0.20:
            oqv = 0.20
        elif oqv > 0.90:
            oqv = 0.90
        tp = oqv * 0.74
        if phase < tp:
            uu = 0.5 * (1.0 - math.cos(math.pi * phase / tp))
        elif phase < oqv:
            uu = math.cos(0.5 * math.pi * (phase - tp) / (oqv - tp))
        else:
            uu = 0.0
        mod = 1.0 + k_coup * fb
        if mod < 0.25:
            mod = 0.25
        inj = (amp[i] * sh * uu * mod + asp[i]) * zs

        # ---- saluran
        b = i // BLOCK
        u = (i - b * BLOCK) / BLOCK
        w = 0.5 - 0.5 * math.cos(math.pi * u)
        for k in range(nseg):
            rc[k] = refl[b, k] * (1.0 - w) + refl[b + 1, k] * w
        z = nz[i]
        kz = nzidx[b]
        lip = 0.0
        gam = 0.5 * alpha * br_gain[i]
        g1 = br_gain[i]
        bo = 0.0
        for _t in range(2):
            JR[0] = rg * Lw[0] + inj
            for k in range(1, nseg):
                d = R[k - 1] - Lw[k]
                wu = R[k - 1] + rc[k] * d
                wd = Lw[k] + rc[k] * d
                if k == br_idx and gam > 1e-4:
                    pj = 2.0 * (R[k - 1] + Lw[k] + gam * wbi) / (2.0 + gam)
                    wd = (1.0 - g1) * wd + g1 * (pj - Lw[k])
                    wu = (1.0 - g1) * wu + g1 * (pj - R[k - 1])
                    bo = gam * (pj - wbi)
                JR[k] = wu
                JL[k] = wd
            JL[nseg] = lip_refl * R[nseg - 1]
            if z != 0.0:
                JR[kz] += 0.5 * z
                JL[kz] += 0.5 * z
            for k in range(nseg):
                R[k] = JR[k] * loss
                Lw[k] = JL[k + 1] * loss
            lip += R[nseg - 1] * (1.0 + lip_refl)
            if gam > 1e-4:
                wbi = br_open * bd[bi]
                bd[bi] = bo * 0.995
                bi += 1
                if bi >= br_len:
                    bi = 0
        # highpass 2 tahap dgn cutoff mengikuti f0 (>= 600 Hz dan >= 3*f0): buang f0 & harmonik
        # rendah dari umpan balik. Cutoff tetap 600 Hz tidak membuang f0 di nada tinggi -> loop
        # umpan balik memicu subharmonik (suara turun oktaf / mati pada /i/ nada tinggi).
        fc = 3.0 * f0[i]
        if fc < 600.0:
            fc = 600.0
        a_hp = 1.0 - math.exp(-2.0 * math.pi * fc / fs)
        lp_hp += a_hp * (Lw[0] - lp_hp)
        h1 = Lw[0] - lp_hp
        lp_hp2 += a_hp * (h1 - lp_hp2)
        lw0 = h1 - lp_hp2                        # hanya riak frekuensi tinggi (formant)
        fbe += 0.0005 * (abs(Lw[0]) - fbe)       # skala = amplitudo total (bukan pita sempit)
        pn = R[12] + Lw[12]
        lp += tap_a * (pn - lp)
        out[i] = 0.5 * lip + tap_g * tap[i] * lp
    return out


def level_vowels(p, segs, fs, f0=None, db_per_semi=0.25, fal=None, dyn=None):
    """
    Penyanyi menjaga kenyaringan relatif rata walau vokalnya beda (/u/ jauh lebih keras
    daripada /a/ di model ini). Hitung RMS tiap vokal, lalu beri gain KONSTAN per vokal
    (konsonan memakai rata-rata gain tetangganya) dengan sambungan halus ~40 ms.
    Sistem linear -> sama dengan menskala eksitasi.
    dyn (kurva dinamika ekspresif dari build_controls): yang diratakan hanya perbedaan karena
    JENIS vokal; aksen/busur frasa yang disengaja dipertahankan (target tiap vokal dikali RMS dyn
    di vokal itu). Bentuk di DALAM satu vokal (messa di voce) tidak disentuh: gain konstan.
    """
    vs = [(t0, t1) for t0, t1, kind, name, info in segs if kind == "vowel"]
    rs = []
    dm = []
    for t0, t1 in vs:
        a, b = int((t0 + 0.04) * fs), int((t1 - 0.02) * fs)
        rs.append(np.sqrt(np.mean(p[a:b] ** 2)) + 1e-12 if b - a > int(0.03 * fs) else np.nan)
        if dyn is not None:
            aa, bb = int(t0 * fs), min(len(dyn), int(t1 * fs))
            if b - a > int(0.03 * fs):
                aa, bb = a, min(len(dyn), b)
            dm.append(float(np.sqrt(np.mean(dyn[aa:bb] ** 2))) if bb > aa else 1.0)
        else:
            dm.append(1.0)
    rs = np.array(rs)
    dm = np.array(dm)
    tgt = np.nanmedian(rs / dm)                # level netral (dinamika ekspresif dikeluarkan)
    if f0 is not None:
        fm = np.median([f0[int(0.5 * (t0 + t1) * fs)] for t0, t1 in vs])
        semi = np.array([12 * np.log2(f0[int(0.5 * (t0 + t1) * fs)] / fm) for t0, t1 in vs])
        tgt_v = tgt * 10.0 ** (db_per_semi * semi / 20.0)
    else:
        tgt_v = np.full(len(rs), tgt)
    tgt_v = tgt_v * dm                         # kembalikan aksen / busur frasa yang disengaja
    if fal is not None:      # falset sengaja -3 dB: jangan diratakan kembali
        wv = np.array([fal[min(len(fal) - 1, int(0.5 * (t0 + t1) * fs))] for t0, t1 in vs])
        tgt_v = tgt_v * 10.0 ** (FAL_DB * wv / 20.0)
    gv = np.clip(tgt_v / np.where(np.isnan(rs), tgt, rs), 0.1, 8.0)
    g = np.ones(len(p))
    vi = -1
    gl = 1.0
    for t0, t1, kind, name, info in segs:
        i0, i1 = int(t0 * fs), min(len(p), int(t1 * fs))
        if kind == "vowel":
            vi += 1
            gl = gv[vi]
            g[i0:i1] = gl
        else:
            nxt = gv[vi + 1] if vi + 1 < len(gv) else gl
            g[i0:i1] = gl if info.get("side") == "co" else nxt
    k = int(0.04 * fs) | 1
    ker = np.hanning(k)
    ker /= ker.sum()
    g = np.convolve(np.pad(g, k // 2, mode="edge"), ker, "valid")
    return p * g


# ------------------------------------------------------------------ 6. pasca-proses
def singer_formant(y, fs, segs, singer):
    """Puncak resonansi ~2,9 kHz (RBJ peaking) yang dicampur hanya di vokal (gerbang halus 30 ms)."""
    if singer <= 0 or not segs:
        return y
    A = 10.0 ** (SINGER_DB * singer / 40.0)
    w0 = 2.0 * np.pi * SINGER_HZ / fs
    al = np.sin(w0) / (2.0 * SINGER_Q)
    b = np.array([1.0 + al * A, -2.0 * np.cos(w0), 1.0 - al * A])
    a = np.array([1.0 + al / A, -2.0 * np.cos(w0), 1.0 - al / A])
    yp = ss.lfilter(b / a[0], a / a[0], y)
    mask = np.zeros(len(y))
    for t0, t1, kind, name, info in segs:
        if kind == "vowel":
            mask[int(t0 * fs):min(len(y), int(t1 * fs))] = 1.0
    k = int(0.03 * fs) | 1
    ker = np.hanning(k)
    ker /= ker.sum()
    mask = ss.fftconvolve(np.pad(mask, k // 2, mode="edge"), ker, "valid")[:len(y)]
    return y + mask * (yp - y)


def timbre_eq(y, fs, amount=1.0, f0=None):
    """v2.3: kemiringan spektrum vokal manusia (hangat di bawah, redup di 2-4 kHz).

    amount 0 = mati, 1 = penuh. Low shelf +1,5 dB @200 Hz, high shelf -10 dB @1,5 kHz,
    lowpass 18 kHz supaya sibilan tidak hilang. Bila f0 diberikan, shelf high dikurangi
    pada suara rendah (< ~230 Hz) karena spektrum model di sana sudah gelap.
    """
    if amount <= 0.0:
        return y
    g_hi = 1.0
    if f0 is not None and len(f0):
        med = float(np.median(f0[f0 > 0])) if np.any(f0 > 0) else TILT_F0_HI
        g_hi = float(smoothstep((med - TILT_F0_LO) / (TILT_F0_HI - TILT_F0_LO)))
    # RBJ shelves: A = 10^(dB/40); A < 1 otomatis menjadi potongan
    A_lo = 10.0 ** (TILT_LOW_DB * amount / 40.0)
    A_hi = 10.0 ** (TILT_HIGH_DB * amount * g_hi / 40.0)

    w0 = 2.0 * np.pi * TILT_LOW_HZ / fs
    al = np.sin(w0) / (2.0 * 0.707)
    c = np.cos(w0)
    r = 2.0 * np.sqrt(A_lo)
    b = np.array([A_lo * ((A_lo + 1) - (A_lo - 1) * c + r * al),
                  2 * A_lo * ((A_lo - 1) - (A_lo + 1) * c),
                  A_lo * ((A_lo + 1) - (A_lo - 1) * c - r * al)])
    a = np.array([(A_lo + 1) + (A_lo - 1) * c + r * al,
                  -2 * ((A_lo - 1) + (A_lo + 1) * c),
                  (A_lo + 1) + (A_lo - 1) * c - r * al])
    y = ss.lfilter(b / a[0], a / a[0], y)

    w0 = 2.0 * np.pi * TILT_HIGH_HZ / fs
    al = np.sin(w0) / (2.0 * 0.707)
    c = np.cos(w0)
    r = 2.0 * np.sqrt(A_hi)
    b = np.array([A_hi * ((A_hi + 1) + (A_hi - 1) * c + r * al),
                  -2 * A_hi * ((A_hi - 1) + (A_hi + 1) * c),
                  A_hi * ((A_hi + 1) + (A_hi - 1) * c - r * al)])
    a = np.array([(A_hi + 1) - (A_hi - 1) * c + r * al,
                  2 * ((A_hi - 1) - (A_hi + 1) * c),
                  (A_hi + 1) - (A_hi - 1) * c - r * al])
    y = ss.lfilter(b / a[0], a / a[0], y)

    # cekungan pita 2-4 kHz (dip) – meredam sisa "logam" tanpa menyentuh sibilan tinggi
    Ad = 10.0 ** (TILT_DIP_DB * amount * g_hi / 40.0)
    w0 = 2.0 * np.pi * TILT_DIP_HZ / fs
    al = np.sin(w0) / (2.0 * TILT_DIP_Q)
    b = np.array([1.0 + al * Ad, -2.0 * np.cos(w0), 1.0 - al * Ad])
    a = np.array([1.0 + al / Ad, -2.0 * np.cos(w0), 1.0 - al / Ad])
    y = ss.lfilter(b / a[0], a / a[0], y)

    sos_lp = ss.butter(2, TILT_LP_HZ / (fs / 2.0), btype="low", output="sos")
    return ss.sosfilt(sos_lp, y)


def room_ir(fs, seed=3, predelay=0.018, decay=0.75, damp=3500.0, early=6):
    """v2.4: ruang lebih nyata: pre-delay, 6 pantulan awal, ekor yang diredam HF.

    Dipakai mono (satu IR) atau stereo (dua IR berbeda seed -> citra lebar).
    """
    n = int((predelay + decay) * fs)
    t = np.arange(n) / fs
    ir = np.random.default_rng(seed).standard_normal(n)
    # redam HF di ekor: lowpass makin berat seiring waktu
    for d in (damp, damp * 0.5):
        sos_d = ss.butter(2, min(d, 0.45 * fs) / (fs / 2.0), btype="low", output="sos")
        ir = ss.sosfilt(sos_d, ir)
    ir *= np.exp(-np.maximum(t - predelay, 0.0) / (decay / 4.6))
    ir[:int(predelay * fs)] = 0.0
    # pantulan awal (diskret, tidak terlalu keras)
    rng_e = np.random.default_rng(seed + 50)
    for _ in range(early):
        k = int(rng_e.uniform(predelay + 0.004, predelay + 0.075) * fs)
        if k < n:
            ir[k] += rng_e.uniform(2.0, 4.5)
    ir[0] = 0.0
    return ir


def apply_space(y, fs, wet=0.16, stereo=False):
    """v2.4: reverb ruang. Mono tetap mono; stereo memakai dua IR berbeda."""
    if wet <= 0.0:
        return y
    scale = np.std(y) + 1e-12
    if stereo:
        out = []
        for sd in (3, 7):
            w = ss.fftconvolve(y, room_ir(fs, seed=sd))[:len(y)]
            out.append(y + wet * w / (np.std(w) + 1e-12) * scale)
        return out
    w = ss.fftconvolve(y, room_ir(fs))[:len(y)]
    return y + wet * w / (np.std(w) + 1e-12) * scale


def post_process(p, fs, segs=None, f0=None, reverb=0.16, fal=None, dyn=None, singer=0.0,
                 timbre=1.0, stereo=False):
    y = np.nan_to_num(np.asarray(p, dtype=np.float64))
    y = np.append(y[0], y[1:] - 0.97 * y[:-1])                    # radiasi bibir
    sos = ss.butter(2, 70.0 / (fs / 2.0), btype="high", output="sos")
    y = ss.sosfilt(sos, y)
    y = timbre_eq(y, fs, timbre, f0)
    if segs is not None:
        y = singer_formant(y, fs, segs, singer)
        y = level_vowels(y, segs, fs, f0, fal=fal, dyn=dyn)
    chans = apply_space(y, fs, reverb, stereo)
    if stereo:
        peak = max(np.percentile(np.abs(c), 99.9) for c in chans) + 1e-12
    else:
        chans, peak = [chans], np.percentile(np.abs(chans), 99.9) + 1e-12
    out = []
    for c in chans:
        a = c / peak
        aa = np.abs(a)
        sat = np.tanh((aa - 0.7) / 0.3) * 0.3 + 0.7
        out.append(np.sign(a) * np.where(aa <= 0.7, aa, sat) * 0.9)
    return out if stereo else out[0]


# ------------------------------------------------------------------ 7. sintesis penuh
def synthesize(notes, tempo, fs=FS, seed=1, vibrato=28.0, oq=OQ_DEFAULT, noise_gain=1.0,
               nasal_gain=0.008, loss=0.9992, rg=0.92, jitter=0.004, shimmer=0.03,
               breath_gain=0.0015, level=True, falsetto=True, human=True, bar=4.0, pickup=0.0,
               bar_reset=False,
               detune=DETUNE_CENTS, breath=1.0, rough=1.0, branch=1.0, couple=COUPLE_DEFAULT):
    rng = np.random.default_rng(seed)
    events, marks, phrases, total_t = plan_song(notes, tempo, rng=timing_rng(seed) if human else None,
                                                beats_per_bar=bar, pickup=pickup, bar_reset=bar_reset)
    segs = build_segments(events, cons_rng(seed))
    ctl = build_controls(segs, marks, phrases, total_t, fs, rng, vib_depth=vibrato,
                         jitter=jitter, shimmer=shimmer, falsetto=falsetto,
                         detune=detune if human else 0.0)
    n = ctl["n"]
    ZS = 1000.0
    asp = ctl["asp"] * 0.10 * noise_gain
    fal = ctl["fal"]
    oq_arr = np.clip(oq + fal * (FAL_OQ - oq) + ctl["oq_off"], 0.40, 0.72)   # falset: oq kecil -> harmonik atas naik
    jit_arr = jitter + fal * (FAL_JITTER - jitter)     # falset: jitter turun
    src, flow = glottal_flow(ctl["f0"], ctl["amp"], asp, float(fs), jit_arr, shimmer, oq_arr, seed)
    src = src * ZS
    # hembusan halus: derau tersaring (<5 kHz), hanya saat glotis terbuka (termodulasi aliran)
    sos_b = ss.butter(2, 5000.0 / (fs / 2.0), btype="low", output="sos")
    breath = ss.sosfilt(sos_b, rng.standard_normal(n)) * ctl["amp"] * flow * ZS * breath_gain * breath * ctl["breath"]
    refl = reflections(ctl["areas"])
    nz = ctl["nz"] * ZS * 0.9 * noise_gain + breath
    # v2.5 cabang samping (side branch) epilaring: resonator sempit, terbuka di ujung,
    # memberi puncak ~3 kHz tanpa menggeser F1-F3. Hanya di vokal (digabung dengan gerbang vokal).
    br_idx = int(np.clip(BRANCH_IDX, 1, ctl["areas"].shape[1] - 2))
    br_open = BR_OPEN
    # cabang selalu aktif (konstan) supaya tetap invarian-waktu & stabil secara energi;
    # menggerbangnya per vokal membuat sambungan berubah-ubah dan bisa memompa energi.
    if branch > 0:
        br_gain = np.full(n, 1.0)
    else:
        br_gain = np.zeros(n)
    if couple <= 0.0:
        p = tract_run(n, src, nz, ctl["nzidx"], ctl["tap"], refl, loss, -0.85, rg,
                      nasal_gain, 1.0 - math.exp(-2.0 * math.pi * 700.0 / fs),
                      br_idx, br_gain, int(BRANCH_LEN), float(BRANCH_ALPHA), float(br_open))
    else:
        # v2.7 kopel sumber-saluran: pulsa glotis dimodulasi tekanan balik dari saluran
        p = tract_run_coupled(n, ctl["f0"], ctl["amp"], asp, oq_arr, jit_arr, shimmer,
                              int(seed), ZS, float(couple), float(COUPLE_OQ) * couple / 0.10, float(fs),
                              nz, ctl["nzidx"], ctl["tap"], refl, loss, -0.85, rg,
                              nasal_gain, 1.0 - math.exp(-2.0 * math.pi * 700.0 / fs),
                              br_idx, br_gain, int(BRANCH_LEN), float(BRANCH_ALPHA), float(br_open))
    return p, ctl, segs, marks, phrases


# ------------------------------------------------------------------ 8. plot
def estimate_f0(sig, fs, win=0.045, hop=0.01, fmin=70, fmax=500):
    w, h = int(win * fs), int(hop * fs)
    lo, hi = int(fs / fmax), int(fs / fmin)
    tt, f0 = [], []
    rms_all = np.sqrt(np.mean(sig ** 2)) + 1e-12
    for i in range(0, len(sig) - w, h):
        seg = sig[i:i + w] - np.mean(sig[i:i + w])
        tt.append((i + w / 2) / fs)
        if np.sqrt(np.mean(seg ** 2)) < 0.15 * rms_all:
            f0.append(np.nan)
            continue
        seg = seg * np.hanning(len(seg))
        ac = np.correlate(seg, seg, "full")[len(seg) - 1:]
        ac /= ac[0] + 1e-12
        lag = lo + int(np.argmax(ac[lo:hi]))
        f0.append(fs / lag if ac[lag] > 0.5 else np.nan)
    return np.array(tt), np.array(f0)


def save_plot(y, ctl, marks, fs, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, tt, S = ss.spectrogram(y, fs, nperseg=2048, noverlap=1792)
    fig, ax = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True,
                           gridspec_kw={"height_ratios": [3, 1]})
    ax[0].pcolormesh(tt, f, 10 * np.log10(S + 1e-12), shading="auto", vmin=-110, vmax=-35, cmap="magma")
    ax[0].set_ylim(0, 5000)
    ax[0].set_ylabel("Hz")
    ax[0].set_title(title)
    t2 = np.arange(len(ctl["f0"])) / fs
    ax[1].plot(t2, ctl["f0"], lw=0.8, color="tab:green", label="f0 target")
    te, fe = estimate_f0(y, fs)
    ax[1].plot(te, fe, ".", ms=2.5, color="tab:red", label="f0 terukur")
    ax[1].set_ylim(80, 320)
    ax[1].set_ylabel("f0 (Hz)")
    ax[1].set_xlabel("detik")
    ax[1].legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Nyanyi v2 - partitur teks -> suara (simulasi saluran suara).")
    ap.add_argument("score")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--tempo", type=float, default=None)
    ap.add_argument("--vibrato", type=float, default=28.0, help="kedalaman vibrato (cents)")
    ap.add_argument("--oq", type=float, default=OQ_DEFAULT,
                    help=f"open quotient dasar (default {OQ_DEFAULT}; lebih kecil = glotis menutup lebih tajam)")
    ap.add_argument("--no-falsetto", action="store_true", help="matikan register falset (>= Bb4)")
    ap.add_argument("--no-human", action="store_true",
                    help="matikan timing manusiawi (onset acak, napas acak, pemanjangan akhir frasa)")
    ap.add_argument("--bar", type=float, default=None, help="beat per birama (default: 'meter' di partitur, atau 4)")
    ap.add_argument("--pickup", type=float, default=None,
                    help="beat anakrusis sebelum ketukan 1 (default: 'pickup' di partitur, atau 0)")
    ap.add_argument("--barreset", action="store_true",
                    help="hitung birama ulang di awal tiap frasa (atau 'barreset 1' di partitur)")
    ap.add_argument("--detune", type=float, default=DETUNE_CENTS,
                    help=f"salah-nada tetap per nada, sigma cents (default {DETUNE_CENTS}; 0 = mati)")
    ap.add_argument("--singer", type=float, default=SINGER_DEFAULT,
                    help=f"0..1 kekuatan singer's formant ~2,9 kHz (default {SINGER_DEFAULT}; 0 = mati)")
    ap.add_argument("--breath", type=float, default=1.0, help="pengali hembusan (default 1; 0 = mati)")
    ap.add_argument("--timbre", type=float, default=1.0,
                    help="kemiringan spektrum vokal manusia (default 1; 0 = datar/lama)")
    ap.add_argument("--rough", type=float, default=1.0,
                    help="kekasaran sumber/getaran otot (default 1; 0 = halus)")
    ap.add_argument("--space", type=float, default=0.16, help="basah reverb ruang (default 0,16)")
    ap.add_argument("--stereo", action="store_true", help="tulis WAV stereo (reverb dua kanal)")
    ap.add_argument("--branch", type=float, default=1.0,
                    help="kekuatan cabang samping 2,9 kHz (default 1; 0 = mati)")
    ap.add_argument("--couple", type=float, default=COUPLE_DEFAULT,
                    help="kopel sumber-saluran / riak formant (default 0,22; 0 = sumber bebas)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-plot", action="store_true")
    a = ap.parse_args(argv)

    with open(a.score, "r", encoding="utf-8") as f:
        notes, cfg = parse_score(f.read())
    if not notes:
        sys.exit("Partitur kosong.")
    tempo = a.tempo if a.tempo else cfg["tempo"]
    name = os.path.splitext(os.path.basename(a.score))[0]
    os.makedirs(OUT, exist_ok=True)
    out_wav = a.out or os.path.join(OUT, name + "_v2.wav")

    t0 = time.time()
    p, ctl, segs, marks, phrases = synthesize(
        notes, tempo, seed=a.seed, vibrato=a.vibrato, oq=a.oq,
        falsetto=not a.no_falsetto, human=not a.no_human,
        bar=a.bar if a.bar else cfg.get("meter", 4.0),
        pickup=a.pickup if a.pickup is not None else cfg.get("pickup", 0.0),
        bar_reset=a.barreset or bool(cfg.get("barreset", 0.0)),
        detune=a.detune, breath=a.breath, rough=a.rough,
        branch=a.branch, couple=a.couple)
    print(f"Simulasi {len(p) / FS:.1f} detik audio dalam {time.time() - t0:.1f} detik")
    y = post_process(p, FS, segs, ctl["f0"], fal=ctl["fal"], dyn=ctl["dyn"], singer=a.singer,
                     timbre=a.timbre, stereo=a.stereo, reverb=a.space)
    if a.stereo:
        wavfile.write(out_wav, FS, (np.stack(y, axis=1) * 0.95 * 32767).astype(np.int16))
        y = y[0]
    else:
        wavfile.write(out_wav, FS, (y * 0.95 * 32767).astype(np.int16))
    print("WAV  ->", out_wav)
    if not a.no_plot:
        png = os.path.join(OUT, name + "_v2_spektro.png")
        save_plot(y, ctl, marks, FS, png, f"{name} v2 ({tempo:.0f} BPM)")
        print("Plot ->", png)


if __name__ == "__main__":
    main()
