"""
Fonem -> kontrol akustik untuk pita suara + saluran suara.

Setiap fonem dideskripsikan oleh KONTROL yang dipakai mesin sintesis:
  voiced : pita suara bergetar (True/False)
  A      : luas (cm^2) di titik penyempitan (indeks 0..43, 0 = glotis, 43 = bibir)
  A_val  : luas penyempitan (cm^2) -> konsonan biasanya mengecilkan luas
  nz     : amplitudo derau (noise) relatif, 0..1 (frikatif, aspirasi)
  nz_bp  : (f_lo, f_hi) Hz pita derau
  burst  : durasi ledakan singkat di awal (detik) untuk plosif
  burst_bp: pita frekuensi ledakan
  nas    : True = saluran hidung dibuka (m, n, ng) -> pole/zero tambahan
  affr   : True = frikatif diikuti plosif (c, j)
"""
import numpy as np

from vowels import VOWELS, N_SECTIONS

NT = N_SECTIONS          # 44 titik tabung
NASAL_DEPTH = 0.85       # seberapa besar sisi hidung menyerap energi

VOCAL_KEYS = ("a", "e", "i", "o", "u")


def _glide(**kw):
    d = dict(voiced=True, A=np.full(NT, 4.0), A_val=4.0, nz=0.0, nz_bp=(1000, 4000),
             burst=0.0, burst_bp=(1000, 3000), nas=False, affr=False)
    d.update(kw)
    return d


def _narrow(base, idx, val):
    """Bentuk dasar vokal lalu ciutkan luas di sekitar titik idx."""
    a = np.array(base, dtype=float)
    w = np.exp(-0.5 * ((np.arange(NT) - idx) / 2.2) ** 2)
    return np.maximum(a * (1.0 - w + w * val / a[idx]), 0.35)


def _vowel_ph(k):
    return _glide(A=np.array(VOWELS[k], dtype=float), A_val=float(VOWELS[k].min()))


PH = {}

# --- vokal (monoftong) + variasi open/closed untuk diftong ---
for _k in VOCAL_KEYS:
    PH[_k] = _vowel_ph(_k)

# diftong: vokal >= 8 = vokal pertama, < 8 = vokal kedua ("ai", "au", "oi", "ei")
for _a in VOCAL_KEYS:
    for _b in VOCAL_KEYS:
        if _a != _b:
            PH[_a + _b] = _vowel_ph(_a)

# glides
PH["w"] = _glide(A=np.array(VOWELS["u"]), A_val=1.4)   # semi-vokal u
PH["y"] = _glide(A=np.array(VOWELS["i"]), A_val=1.2)   # semi-vokal i
PH["r"] = _glide(A=_narrow(VOWELS["e"], 30, 0.9), A_val=0.9)

# --- nasal voiced (m, n, ng, ny) ---
PH["m"] = _glide(A=np.full(NT, 1.2), A_val=1.2, nas=True)
PH["n"] = _glide(A=np.full(NT, 1.3), A_val=1.3, nas=True)
PH["ng"] = _glide(A=_narrow(VOWELS["e"], 34, 1.0), A_val=1.0, nas=True)
PH["ny"] = _glide(A=_narrow(VOWELS["i"], 12, 0.8), A_val=0.8, nas=True)

# --- lateral & trill ---
PH["l"] = _glide(A=_narrow(VOWELS["e"], 24, 0.8), A_val=0.8)
PH["rr"] = _glide(A=np.full(NT, 3.0), A_val=3.0)

# --- frikatif unvoiced ---
PH["s"] = _glide(voiced=False, A=_narrow(VOWELS["i"], 20, 0.4), A_val=0.4,
                 nz=1.0, nz_bp=(3600, 9000))
PH["sh"] = _glide(voiced=False, A=_narrow(VOWELS["u"], 24, 0.6), A_val=0.6,
                  nz=1.0, nz_bp=(1800, 6500))
PH["f"] = _glide(voiced=False, A=_narrow(VOWELS["u"], 40, 0.35), A_val=0.35,
                 nz=0.9, nz_bp=(1800, 9000))
PH["h"] = _glide(voiced=True, A=np.full(NT, 5.0), A_val=5.0, nz=0.75, nz_bp=(500, 5000))

# --- frikatif voiced ---
PH["z"] = _glide(A=_narrow(VOWELS["i"], 20, 0.5), A_val=0.5, nz=0.7, nz_bp=(3000, 8000))
PH["v"] = _glide(A=_narrow(VOWELS["u"], 40, 0.4), A_val=0.4, nz=0.6, nz_bp=(1500, 8000))
PH["dh"] = _glide(A=_narrow(VOWELS["e"], 34, 0.45), A_val=0.45, nz=0.55, nz_bp=(2000, 6000))

# --- plosif ---
PH["p"] = _glide(voiced=False, A=np.full(NT, 0.4), A_val=0.4, burst=0.018, burst_bp=(1200, 6000))
PH["b"] = _glide(A=np.full(NT, 0.4), A_val=0.4, burst=0.010, burst_bp=(600, 3500))
PH["t"] = _glide(voiced=False, A=_narrow(VOWELS["i"], 20, 0.35), A_val=0.35,
                 burst=0.016, burst_bp=(2500, 9000))
PH["d"] = _glide(A=_narrow(VOWELS["i"], 20, 0.4), A_val=0.4, burst=0.009, burst_bp=(1500, 6000))
PH["k"] = _glide(voiced=False, A=_narrow(VOWELS["e"], 32, 0.4), A_val=0.4,
                 burst=0.020, burst_bp=(1500, 7000))
PH["g"] = _glide(A=_narrow(VOWELS["e"], 32, 0.4), A_val=0.4, burst=0.011, burst_bp=(800, 4000))
PH["c"] = _glide(voiced=False, A=_narrow(VOWELS["i"], 20, 0.35), A_val=0.35,
                 nz=0.8, nz_bp=(3500, 9000), burst=0.02, burst_bp=(2500, 9000), affr=True)
PH["j"] = _glide(A=_narrow(VOWELS["i"], 20, 0.4), A_val=0.4,
                 nz=0.6, nz_bp=(3000, 8000), burst=0.012, burst_bp=(1500, 6000), affr=True)
PH["q"] = _glide(voiced=False, A=_narrow(VOWELS["e"], 32, 0.4), A_val=0.4, burst=0.018, burst_bp=(1200, 6000))
PH["x"] = _glide(voiced=False, A=np.full(NT, 2.0), A_val=2.0, nz=0.8, nz_bp=(1500, 8000))

# simbol yang tidak didukung -> diam
PH["-"] = _glide(voiced=False, nz=0.0)


# ---------- pemetaan ejaan -> fonem ----------

# tabel digraf (urutan penting)
DIGRAPHS = {
    "ng": "ng", "ny": "ny", "sy": "sh", "kh": "x", "ch": "c", "dj": "j", "tj": "c",
    "dh": "dh", "th": "t", "sh": "sh", "ph": "f", "ck": "k", "wh": "w",
    "ee": "i", "oo": "u", "ea": "i", "ou": "o", "oa": "o", "ai": "ai", "au": "au",
    "oi": "oi", "oy": "oi", "ei": "ei", "ay": "ai", "ey": "ei",
}

SINGLE = {
    "a": "a", "e": "e", "i": "i", "o": "o", "u": "u",
    "b": "b", "c": "c", "d": "d", "f": "f", "g": "g", "h": "h",
    "j": "j", "k": "k", "l": "l", "m": "m", "n": "n", "p": "p",
    "q": "k", "r": "r", "s": "s", "t": "t", "v": "v", "w": "w",
    "x": "x", "y": "y", "z": "z",
}

VOWEL_SET = set(VOCAL_KEYS)


def spell_to_phonemes(word):
    """'ma-kan' / 'makan' / 'sa-yang' -> ['m','a','k','a','n']"""
    w = word.lower().strip()
    out = []
    i = 0
    while i < len(w):
        two = w[i:i + 2]
        if two in DIGRAPHS:
            out.append(DIGRAPHS[two])
            i += 2
            continue
        ch = w[i]
        if ch in SINGLE:
            out.append(SINGLE[ch])
        # huruf asing lain (mis. '' atau angka) -> diabaikan
        i += 1
    return out


def _is_vowel(p):
    return p in VOWEL_SET or (len(p) == 2 and p[0] in VOWEL_SET and p[1] in VOWEL_SET)


def syllable_split(ph):
    """
    Pisahkan satu suku kata -> (onset, nucleus, coda).
    's','t','r' tunggal di antara vokal ikut onset berikutnya (bukan coda).
    """
    if not ph:
        return [], "e", []
    vi = [i for i, p in enumerate(ph) if _is_vowel(p)]
    if vi:
        i0 = vi[0]
        nucleus = ph[i0]
        onset = ph[:i0]
        coda = ph[i0 + 1:]
    else:
        # tanpa vokal: seluruhnya konsonan
        return ph, "e", []
    # konsonan akhir yang berupa sonoran+plosif ("-ng", "-rt") dipisah normal;
    # konsonan tunggal '-r','-l','-n','-s','-k','-t','-m','-p' tetap coda.
    return onset, nucleus, coda


def parse_lyric(text):
    """
    Lirik -> daftar suku kata [(onset, nucleus, coda), ...]
    Pisah per kata (spasi), lalu per suku kata (tanda '-'), lalu konsonan+vokal.
    Tanpa tanda '-', suku dibagi otomatis: KV(KV)*.
    """
    syllables = []
    for token in text.replace("|", " ").split():
        # boleh ditulis "ma-kan" atau "makan"
        parts = token.split("-")
        for part in parts:
            if not part:
                continue
            ph = spell_to_phonemes(part)
            if not ph:
                continue
            if "-" not in token:
                # pecah otomatis: ona..VOn set, K V, dst.
                cur_onset = []
                k = 0
                vidx = [i for i, p in enumerate(ph) if _is_vowel(p)]
                if not vidx:
                    syllables.append((ph, "e", []))
                    continue
                # vokal terakhir suku kata: V diikuti 0-2 konsonan, sisanya ke suku berikut
                last_v = vidx[-1]
                nucleus = ph[last_v]
                onset = ph[:last_v]
                tail = ph[last_v + 1:]
                # pindahkan sisa konsonan yang lebih dari 2 ke berikutnya? sederhana:
                if len(tail) > 2:
                    syllables.append((onset, nucleus, tail[:2]))
                    rest = tail[2:]
                    if rest:
                        syllables.append((rest, "e", []))
                else:
                    syllables.append((onset, nucleus, tail))
            else:
                onset, nucleus, coda = syllable_split(ph)
                syllables.append((onset, nucleus, coda))
    return syllables


def vowel_shape(key):
    key = key if key in PH else "e"
    return np.array(PH[key]["A"], dtype=float)


# ---------- durasi fonem ----------

DUR = {
    "consonant": 0.045,     # konsonan transien (b, d, g, ...)
    "fricative": 0.080,     # s, sh, f, v, z, h
    "nasal": 0.070,         # m, n, ng, ny
    "liquid": 0.055,        # l, r
    "glide": 0.055,         # w, y
    "burst": 0.012,         # ledakan plosif yang ditambahkan
}


def phone_duration(p):
    if p in ("s", "sh", "f", "v", "z", "h", "x", "dh"):
        return DUR["fricative"]
    if p in ("m", "n", "ng", "ny"):
        return DUR["nasal"]
    if p in ("l", "r", "rr"):
        return DUR["liquid"]
    if p in ("w", "y"):
        return DUR["glide"]
    return DUR["consonant"]
