"""
Aransemen a cappella "The Sound of Silence" (Simon & Garfunkel) - 4 suara.

Semua bagian disintesis sebagai SUARA MANUSIA (mesin sing_v2), bukan instrumen:
  1. LEAD     - melodi utama + lirik asli
  2. HARMONI  - terts di bawah lead (gaya Simon & Garfunkel), vokal "ah"
  3. ARPEGGIO - melodi/gitar lagu (pola nada akor) dinyanyikan "oo"
  4. BASS     - akar akor + kuint, "dum"

Semua bagian memakai grid birama yang sama (4/4, 112 ketukan = 28 birama),
dan `human=False` supaya onset semua bagian terkunci (tanpa geseran acak).

Pakai:
    python acappella.py            # render penuh (~63 detik)
    python acappella.py --bars 6   # uji cepat 6 birama
"""
import argparse
import os
import time

import numpy as np
from scipy import ndimage
from scipy.io import wavfile

import sing_v2 as S
from score import parse_score

FS = 44100
TEMPO = 106.0
OUT = "output"

# ------------------------------------------------------------------ akor
# pcs  : pitch class anggota akor
# bass : (akar, kuint) di nada rendah
# arp  : pola arpeggio 4 ketukan (terts, kuint, oktaf, kuint)
CHORDS = {
    "Dm": dict(pcs={2, 5, 9},   bass=(50, 57), arp=(53, 57, 62, 57)),
    "C":  dict(pcs={0, 4, 7},   bass=(48, 55), arp=(52, 55, 60, 55)),
    "Bb": dict(pcs={10, 2, 5},  bass=(46, 53), arp=(50, 53, 58, 53)),
    "F":  dict(pcs={5, 9, 0},   bass=(53, 60), arp=(57, 60, 65, 60)),
    "A":  dict(pcs={9, 1, 4},   bass=(45, 52), arp=(49, 52, 57, 52)),
}

# ------------------------------------------------------------------ melodi (lead)
# tiap birama = 4 ketukan: (midi, ketukan, suku kata)
BARS = [
    # ---------- Verse 1: "Hello darkness, my old friend"
    ("Dm", [(69, .5, "Hel"), (69, .5, "lo"), (72, .5, "dark"), (72, .5, "nes"), (69, 1, "mai"), (69, 1, "old")]),
    ("Dm", [(69, 4, "frend")]),
    ("Dm", [(72, .5, "Aiv"), (70, .5, "kam"), (69, 1, "tu"), (70, .5, "tok"), (69, .5, "with"), (67, 1, "yu")]),
    ("C",  [(69, .5, "a"), (67, 3.5, "gen")]),
    ("Bb", [(69, .5, "Bik"), (69, .5, "kos"), (72, 1, "a"), (74, .5, "vi"), (74, .5, "syen"), (72, 1, "soft")]),
    ("F",  [(72, .5, "li"), (69, 1, "kri"), (69, 2.5, "ping")]),
    ("Bb", [(69, .5, "Left"), (69, .5, "its"), (72, 1, "sidz"), (72, .5, "wail"), (69, .5, "ai"), (69, .5, "woz"), (69, .5, "sli")]),
    ("F",  [(69, 4, "ping")]),
    ("Bb", [(72, .5, "And"), (72, .5, "de"), (74, .5, "vi"), (74, .5, "syen"), (72, .5, "dat"), (72, .5, "woz"), (69, 1, "plan")]),
    ("F",  [(69, .5, "ted"), (69, .5, "in"), (67, 1, "mai"), (69, 2, "brein")]),
    ("C",  [(62, 1, "Stil"), (65, 1, "ri"), (69, 2, "meins")]),
    ("Dm", [(69, .5, "Wi"), (69, .5, "thin"), (67, 1, "de"), (65, 1, "saund"), (69, 1, "of")]),
    ("C",  [(67, 1, "sai"), (65, 3, "lens")]),
    ("Dm", [(None, 4, "")]),
    # ---------- Verse 4: "And the people bowed and prayed"
    ("Dm", [(69, .5, "And"), (69, .5, "de"), (72, .5, "pi"), (72, .5, "pol"), (72, 1, "baud"), (69, 1, "and")]),
    ("A",  [(69, 4, "preid")]),
    ("Dm", [(72, .5, "Tu"), (70, .5, "de"), (69, 1, "ni"), (70, .5, "on"), (69, .5, "god"), (67, 1, "dei")]),
    ("C",  [(67, 4, "meid")]),
    ("Bb", [(69, .5, "And"), (69, .5, "de"), (72, 1, "sain"), (74, .5, "flas"), (74, .5, "aut"), (72, 1, "its")]),
    ("F",  [(72, .5, "wor"), (69, 3.5, "ning")]),
    ("Bb", [(69, .5, "In"), (69, .5, "de"), (72, 1, "words"), (72, .5, "dat"), (69, .5, "it"), (69, .5, "woz"), (69, .5, "for")]),
    ("F",  [(69, 4, "ming")]),
    ("C",  [(72, .5, "And"), (72, .5, "de"), (74, 1, "sain"), (72, .5, "sed"), (72, .5, "de"), (72, .5, "words"), (69, .5, "of")]),
    ("Dm", [(69, .5, "de"), (69, .5, "pro"), (69, .5, "fets"), (69, .5, "ar"), (67, .5, "ri"), (67, .5, "ten"), (69, .5, "on"), (69, .5, "de")]),
    ("Bb", [(69, .5, "sab"), (69, .5, "wei"), (69, 3, "wols")]),
    ("F",  [(62, 1, "And"), (65, 1, "te"), (65, .5, "ne"), (65, .5, "ment"), (69, 1, "hols")]),
    ("C",  [(69, .5, "And"), (69, .5, "wis"), (67, 1, "perd"), (65, 1, "in"), (69, 1, "de")]),
    ("Dm", [(69, .5, "saunds"), (69, .5, "of"), (67, .5, "sai"), (65, 2.5, "lens")]),
]

FLAT = {0: "C", 1: "Db", 2: "D", 3: "Eb", 4: "E", 5: "F",
        6: "Gb", 7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B"}


def midi_name(m):
    return f"{FLAT[m % 12]}{m // 12 - 1}"


def harmony_of(lead, pcs, prev=None):
    """Terts di bawah lead, dibatasi nada akor.

    Memilih nada akor 3-11 semiton di bawah lead, dengan prioritas interval
    terts/sekst (3,4,8,9), menghindari sekon/unison, dan menghaluskan lompatan
    dari nada harmoni sebelumnya.
    """
    best, best_score = None, 1e9
    for n in range(lead - 11, lead - 2):
        if n < 55:                      # jangan terlalu rendah (di bawah G3)
            continue
        if n % 12 not in pcs:
            continue
        iv = lead - n
        score = 0.0 if iv in (3, 4, 8, 9) else (3.0 if iv in (5, 6, 7) else 1.0)
        if prev is not None:
            score += abs(n - prev) * 0.25
            if n == prev:
                score += 0.8            # hindari nada harmoni yang mendatar terus
        if score < best_score:
            best_score, best = score, n
    if best is None:
        return prev if prev is not None else lead - 4
    return best


def build_parts(bars):
    """Hasil: dict nama -> list birama berisi (midi|None, ketukan, lirik)."""
    lead, harm, arp, bass = [], [], [], []
    prev_h = None
    for chord, notes in bars:
        C = CHORDS[chord]
        bar_lead, bar_harm = [], []
        for midi, beats, lyr in notes:
            bar_lead.append((midi, beats, lyr))
            if midi is None:
                bar_harm.append((None, beats, ""))
            else:
                h = harmony_of(midi, C["pcs"], prev_h)
                prev_h = h
                bar_harm.append((h, beats, "ah"))
        # arpeggio: 4 ketukan
        a = C["arp"]
        bar_arp = [(a[0], 1, "oo"), (a[1], 1, "oo"), (a[2], 1, "oo"), (a[3], 1, "oo")]
        # bass: akar 2 ketukan + kuint 2 ketukan
        b = C["bass"]
        bar_bass = [(b[0], 2, "dum"), (b[1], 2, "dum")]
        lead.append(bar_lead)
        harm.append(bar_harm)
        arp.append(bar_arp)
        bass.append(bar_bass)
    return dict(lead=lead, harm=harm, arp=arp, bass=bass)


def write_score(path, title, bars):
    lines = [f"# {title}", f"tempo {TEMPO:g}", "meter 4", "barreset 1", ""]
    for bar in bars:
        toks = []
        for midi, beats, lyr in bar:
            b = ("%g" % beats)
            if midi is None:
                toks.append(f"R:{b}")
            else:
                toks.append(f"{midi_name(midi)}:{b}={lyr}")
        lines.append(" ".join(toks))
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def render(path, seed, vibrato=28.0, oq=S.OQ_DEFAULT):
    with open(path, "r", encoding="utf-8") as f:
        notes, cfg = parse_score(f.read())
    p, ctl, segs, marks, phrases = S.synthesize(
        notes, TEMPO, fs=FS, seed=seed, vibrato=vibrato, oq=oq,
        falsetto=True, human=False, bar=4.0, pickup=0.0, bar_reset=False)
    y = S.post_process(p, FS, segs, ctl["f0"], fal=ctl["fal"], dyn=ctl["dyn"],
                       singer=1.0, timbre=1.0, stereo=False, reverb=0.0)
    return np.nan_to_num(y)


def even_levels(y, fs, ratio=2.2, thresh_db=2.0, win=0.10):
    """Ratakan kenyaringan mix: bagian yang menyanyi bersamaan (4 suara menahan nada)
    keluar 6-9 dB lebih keras dari rata-rata. Kompresi lembut di atas median."""
    hop = max(1, int(0.005 * fs))
    n = len(y) // hop
    if n < 20:
        return y
    e = np.sqrt((y[:n * hop].reshape(n, hop) ** 2).mean(axis=1)) + 1e-12
    db = 20.0 * np.log10(e)
    sm = ndimage.uniform_filter1d(db, size=max(3, int(win / 0.005)) | 1, mode="nearest")
    act = sm[sm > sm.max() - 35.0]
    ref = float(np.median(act)) if len(act) else float(np.median(sm))
    over = np.maximum(0.0, sm - (ref + thresh_db))
    gr = -over * (1.0 - 1.0 / ratio)
    g = ndimage.uniform_filter1d(10.0 ** (gr / 20.0), size=max(3, int(0.05 / 0.005)) | 1,
                                 mode="nearest")
    gi = np.interp(np.arange(len(y)), np.arange(n) * hop, g)
    return y * gi


def pan_gains(pan):
    """pan -1..1 -> (gain kiri, gain kanan) dengan daya konstan."""
    th = (pan + 1.0) * np.pi / 4.0
    return np.cos(th), np.sin(th)


def active_level(y, fs, win=0.05, pct=75.0):
    """Kenyaringan 'saat bernyanyi': persentil ke-75 dari RMS jendela pendek.

    Dipakai untuk menyeimbangkan suara: lead punya banyak jeda, pad terus-menerus,
    jadi RMS total akan menyesatkan.
    """
    w = max(1, int(win * fs))
    n = len(y) // w
    r = np.sqrt((y[:n * w].reshape(n, w) ** 2).mean(axis=1))
    return float(np.percentile(r, pct)) + 1e-12


def tame_bursts(y, fs, max_over_db=7.0, win=0.30):
    """Jinakkan ledakan konsonan (p, t, k) yang keluar jauh lebih keras dari vokal.

    Mesin sing_v2 memberi gain vokal ke konsonan tetangganya, sehingga plosif pada
    vokal yang butuh penguatan besar bisa +25 dB di atas vokal. Di sini puncak lokal
    dibatasi `max_over_db` di atas median amplitudo 120 ms sekitarnya.
    """
    hop = int(0.002 * fs)
    n = len(y) // hop
    if n < 8:
        return y
    e = np.sqrt((y[:n * hop].reshape(n, hop) ** 2).mean(axis=1)) + 1e-12
    ref = ndimage.median_filter(e, size=int(win / 0.002) | 1, mode="nearest")
    cap = ref * 10.0 ** (max_over_db / 20.0)
    g = np.minimum(1.0, cap / e)
    g = ndimage.median_filter(g, size=3, mode="nearest")
    gi = np.interp(np.arange(len(y)), np.arange(n) * hop, g)
    return y * gi


def main(argv=None):
    ap = argparse.ArgumentParser(description="Aransemen a cappella 4 suara.")
    ap.add_argument("--bars", type=int, default=0, help="batasi jumlah birama (uji cepat)")
    ap.add_argument("--out", default=os.path.join(OUT, "sound_of_silence_acappella.wav"))
    ap.add_argument("--no-stems", action="store_true")
    a = ap.parse_args(argv)

    bars = BARS[:a.bars] if a.bars else BARS
    parts = build_parts(bars)
    os.makedirs(OUT, exist_ok=True)

    # tulis partitur tiap suara
    for name, title in (("lead", "A cappella - LEAD (melodi utama)"),
                        ("harm", "A cappella - HARMONI (terts bawah lead)"),
                        ("arp", "A cappella - ARPEGGIO (melodi/gitar, 'oo')"),
                        ("bass", "A cappella - BASS (akar akor, 'dum')")):
        write_score(os.path.join(OUT, f"acap_{name}.txt"), title, parts[name])

    # gain & panci tiap suara: target kenyaringan "saat bernyanyi" (dBFS) + panci
    MIX = {"lead": (-15.0, 0.00, 28.0), "harm": (-20.0, -0.30, 24.0),
           "arp": (-21.0, 0.35, 18.0), "bass": (-20.5, 0.00, 12.0)}

    cache = os.path.join(OUT, "cache")
    os.makedirs(cache, exist_ok=True)
    tag = f"v2_{len(bars)}"

    t_all = time.time()
    L = None
    R = None
    stems = {}
    for i, name in enumerate(("lead", "harm", "arp", "bass")):
        cpath = os.path.join(cache, f"acap_{name}_{tag}.npy")
        t0 = time.time()
        if os.path.exists(cpath):
            y = np.load(cpath)
            src = "cache"
        else:
            y = render(os.path.join(OUT, f"acap_{name}.txt"), seed=11 + i,
                       vibrato=MIX[name][2]).astype(np.float32)
            y = tame_bursts(y, FS).astype(np.float32)
            np.save(cpath, y)
            src = f"{time.time() - t0:.0f}s render"
        lvl = active_level(y, FS)
        target_db, pan, _ = MIX[name]
        g = 10.0 ** (target_db / 20.0) / lvl
        st = S.apply_space(y, FS, wet=0.15 if name == "lead" else 0.13, stereo=True)
        gl, gr = pan_gains(pan)
        l = st[0] * g * gl
        r = st[1] * g * gr
        L = l if L is None else L + l
        R = r if R is None else R + r
        stems[name] = y * g
        print(f"  {name:5s} {len(y) / FS:6.1f}s  aktif={20 * np.log10(lvl):6.1f} dBFS"
              f"  -> {target_db:5.1f} dB  ({src})")

    # ratakan kenyaringan, lalu normalisasi + limiter lembut
    L = even_levels(L, FS)
    R = even_levels(R, FS)
    peak = max(np.percentile(np.abs(L), 99.95), np.percentile(np.abs(R), 99.95)) + 1e-12
    L, R = L / peak, R / peak
    for ch in (L, R):
        aa = np.abs(ch)
        sat = np.tanh((aa - 0.75) / 0.25) * 0.25 + 0.75
        ch[:] = np.sign(ch) * np.where(aa <= 0.75, aa, sat) * 0.95

    wavfile.write(a.out, FS, (np.stack([L, R], axis=1) * 32767).astype(np.int16))
    print(f"Mix  -> {a.out}  ({len(L) / FS:.1f} detik, total {time.time() - t_all:.0f}s)")

    if not a.no_stems:
        for name, y in stems.items():
            p = os.path.join(OUT, f"acap_{name}.wav")
            wavfile.write(p, FS, (y / (np.abs(y).max() + 1e-12) * 0.9 * 32767).astype(np.int16))
        print("Stem ->", ", ".join(f"acap_{n}.wav" for n in stems))


if __name__ == "__main__":
    main()
