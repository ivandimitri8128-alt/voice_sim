"""
Notasi lagu teks sederhana -> daftar nada.

Format:  <not><oktaf>:<durasi> [lirik]  ...

Contoh:
    # Lagu contoh: Burung Kakatua
    tempo 100
    C4:1 Bu  C4:1 rung  G4:2 Ka  G4:1 ka  A4:1 tu  A4:1 a
    G4:2 a   G4:1 a     F4:1 a    F4:1 a   E4:2 -
    C4:1 Bu  C4:1 rung  G4:2 Ka  G4:1 ka  A4:1 tu  A4:1 a
    G4:1 a   F4:1 a     E4:2 a    C4:4 -

Aturan:
  - Nada: C D E F G A B, boleh pakai # atau b  (C#4, Bb3). Oktaf 2..6.
  - Durasi dalam ketukan: 1 = seperempat, 2 = setengah, 4 = penuh, 0.5 = seperdelapan.
  - Lirik boleh satu suku kata ("ma"), beberapa suku kata dipisah '-' ("ma-kan"),
    atau beberapa kata dipisah '_' ("bu-_rung").  '~' atau '-' pada lirik
    (dan tidak ada satu kata pun) = satu nada dilanjutkan (slur/tie).
  - Tanda '-' tanpa lirik = perpanjangan nada sebelumnya (tie).
  - 'rest' atau 'r:1' = istirahat.  '|' = pemisah bar (diabaikan).
  - 'tempo 100' / 'tempo 96' = BPM per seperempat.
  - 'meter 4' = beat per birama; 'pickup 1.5' = beat anakrusis sebelum ketukan 1 (untuk aksen ketukan 1/3);
    'barreset 1' = hitung birama ulang di awal tiap frasa (untuk partitur yang tidak pas per birama).
  - 'q 1.2' = nilai ketegangan dasar pita suara (default 1.25 ~ 137 Hz).
  - 'breathe' = ambil napas singkat.
  - '#' di awal baris = komentar.
"""
import re

NOTE_INDEX = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(\d)?(?::(\d+(?:\.\d+)?|\.\d+))?$")
VOWELS = "aiueo"


def note_to_midi(name, octave):
    """'C', 4 -> 60 ;  'C#', 4 -> 61 ;  'Bb', 3 -> 58"""
    n = name.upper()
    m = NOTE_INDEX[n]
    if octave is None:
        octave = 4
    return 12 * (octave + 1) + m


def midi_to_freq(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


class Note:
    __slots__ = ("midi", "beats", "lyric", "tie")

    def __init__(self, midi=None, beats=1.0, lyric="", tie=False):
        self.midi = midi          # None = istirahat
        self.beats = beats
        self.lyric = lyric
        self.tie = tie

    @property
    def freq(self):
        return None if self.midi is None else midi_to_freq(self.midi)

    def __repr__(self):
        if self.midi is None:
            return f"Rest({self.beats})"
        n = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"][self.midi % 12]
        return f"{n}{self.midi // 12 - 1}:{self.beats}{(' ' + repr(self.lyric)) if self.lyric else ''}"


def _parse_note_token(tok):
    """'C4:2' -> ('note', midi, beats)."""
    t = tok.replace("|", "").strip()
    if not t:
        return None
    low = t.lower()
    if low in ("r", "rest", "istirahat"):
        return ("rest", None, 1.0)
    if low.startswith(("rest:", "r:")):          # 'r:1' / 'R:0.5' sesuai dokumentasi partitur
        return ("rest", None, float(low.split(":")[1]))
    m = NOTE_RE.match(t)
    if not m:
        return None
    name, acc, octv, dur = m.groups()
    octave = int(octv) if octv else 4
    midi = note_to_midi(name, octave) + (1 if acc == "#" else -1 if acc == "b" else 0)
    beats = float(dur) if dur else 1.0
    return ("note", midi, beats)


def parse_score(text):
    """
    Baca partitur -> (list[Note], dict config)
    config: tempo (BPM), q (ketegangan), vowel (vokal default untuk slur)
    """
    cfg = {"tempo": 100.0, "q": 1.25, "vowel": "a", "lead": 0.12, "tail": 0.25,
           "meter": 4.0, "pickup": 0.0, "barreset": 0.0}
    notes = []

    for raw in text.splitlines():
        line = raw.split("#")[0].strip()
        if not line:
            continue
        low = line.lower()
        # --- perintah ---
        if low.startswith("tempo"):
            try:
                cfg["tempo"] = float(low.split()[1])
            except (IndexError, ValueError):
                pass
            continue
        if low.startswith("q ") or low == "q":
            try:
                cfg["q"] = float(low.split()[1])
            except (IndexError, ValueError):
                pass
            continue
        if low.startswith(("meter", "pickup", "barreset")):
            try:
                cfg[low.split()[0]] = float(low.split()[1])
            except (IndexError, ValueError):
                pass
            continue
        if low.startswith("breathe") or low == "napas":
            notes.append(Note(None, 0.25, ""))
            continue

        # --- tokenisasi: pertahankan urutan not + lirik ---
        tokens = line.replace("|", " ").split()
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if "=" in tok:                       # lirik menempel tanpa spasi
                parts = tok.split("=")
                i += 1
                # satukan kembali bila lirik mengandung spasi tak sengaja
                while i < len(tokens) and "=" not in tokens[i] and _parse_note_token(tokens[i]) is None:
                    parts[-1] += " " + tokens[i]
                    i += 1
                tok = parts[0]
                nres = _parse_note_token(tok)
                if nres is None:
                    continue
                kind, midi, beats = nres
                if kind == "rest":
                    notes.append(Note(None, beats))
                else:
                    notes.append(Note(midi, beats, parts[1]))
                continue

            nres = _parse_note_token(tok)
            if nres is None:
                i += 1
                continue
            kind, midi, beats = nres
            if kind == "rest":
                notes.append(Note(None, beats))
                i += 1
                continue

            # format spasi: "C4:1 Bu" / "C4:1 -" (perpanjangan)
            lyric = ""
            if i + 1 < len(tokens):
                nxt = tokens[i + 1]
                if nxt in ("-", "~"):
                    lyric = nxt
                    i += 1
                elif _parse_note_token(nxt) is None and "=" not in nxt:
                    lyric = nxt
                    i += 1
            if "=" in lyric:
                lyric = lyric.split("=", 1)[1]
            notes.append(Note(midi, beats, lyric))
            i += 1

    # gabungkan tie/slur: nada dengan lirik '-'/'~'/'_' digabung ke nada sebelumnya
    merged = []
    for nt in notes:
        if merged and nt.lyric in ("-", "~", "_") and nt.midi is not None:
            prev = merged[-1]
            if prev.midi is not None:
                prev.beats += nt.beats
                continue
        merged.append(nt)

    return merged, cfg


def score_duration(notes, tempo):
    """Total durasi (detik) partitur."""
    sec_per_beat = 60.0 / tempo
    return sum(n.beats for n in notes) * sec_per_beat


def load_score(path):
    with open(path, "r", encoding="utf-8") as f:
        return parse_score(f.read())


# ---------- pemecahan suku kata untuk lirik pada satu nada ----------
def _split_kv(word):
    """'kakatua' -> ['ka','ka','tu','a']; 'nyanyi' -> ['nya','nyi'] secara heuristik."""
    from phonemes import spell_to_phonemes, _is_vowel
    ph = spell_to_phonemes(word)
    syls = []
    if not ph:
        return syls
    vidx = [k for k, p in enumerate(ph) if _is_vowel(p)]
    if not vidx:
        return [(ph, "e", [])]
    start = 0
    for s, vi in enumerate(vidx):
        end = vidx[s + 1] if s + 1 < len(vidx) else len(ph)
        # konsonan sebelum vokal berikutnya: sisakan satu untuk onset suku berikut
        cons = ph[vi + 1:end]
        if s + 1 < len(vidx):
            if len(cons) >= 2:
                take = len(cons) - 1 if len(cons) > 1 else 0
                cut = vi + 1 + take
            else:
                cut = vi + 1
        else:
            cut = len(ph)
        syls.append((ph[start:vi], ph[vi], ph[vi + 1:cut]))
        start = cut
    return syls


def lyric_syllables(lyric):
    """
    Lirik satu nada -> list suku kata (onset, nucleus, coda).

    Aturan penting: hanya vokal PERTAMA word yang menjadi inti nada ini.
    Sisa huruf setelahnya menjadi ONSET nada berikutnya (yang belum diketahui),
    sehingga kata yang terpotong di tengah tidak kehilangan konsonan.

    'Bu_rung'   -> [(['b'],'u',[], ['r'])]    kata 'rung' digabung sebagai onset 'r'
    'Bu'        -> [(['b'],'u',[], [])]
    'ma-kan'    -> [('m','a',[]), ('k','a',[])]   (penanda '-' = suku berikutnya)
    """
    words = lyric.replace("_", " ").split()
    if not words:
        return []

    # kata yang memakai '-' = pemisah suku eksplisit
    if any("-" in w for w in words):
        out = []
        for w in words:
            for part in w.split("-"):
                if not part:
                    continue
                out.extend(_sylls_from_word(part, all_=True))
        return out

    out = []
    for wi, w in enumerate(words):
        syls = _sylls_from_word(w, all_=(wi < len(words) - 1))
        out.extend(syls)
        # onset sisa kata sebelumnya -> gabung ke onset suku pertama kata ini
        if out and wi > 0:
            pass
    return _merge_pending(out)


def _sylls_from_word(word, all_=False):
    from phonemes import spell_to_phonemes, _is_vowel
    ph = spell_to_phonemes(word)
    if not ph:
        return []
    vidx = [k for k, p in enumerate(ph) if _is_vowel(p)]
    if not vidx:
        # tanpa vokal: seluruhnya menjadi onset "pending" (dipakai nada berikut)
        return [([], "e", [], list(ph), "pending")]
    if all_:
        # pecah semua KV
        syls = []
        start = 0
        for s, vi in enumerate(vidx):
            end = vidx[s + 1] if s + 1 < len(vidx) else len(ph)
            cons = ph[vi + 1:end]
            if s + 1 < len(vidx):
                cut = vi + 1 + (len(cons) - 1 if len(cons) >= 2 else 0)
            else:
                cut = len(ph)
            syls.append((ph[start:vi], ph[vi], ph[vi + 1:cut], [], "ok"))
            start = cut
        return syls
    vi = vidx[0]
    onset = ph[:vi]
    coda = ph[vi + 1:]
    # konsonan sisa (selain 1 coda) -> onset nada berikutnya
    if len(coda) >= 2:
        return [(onset, ph[vi], coda[:1], coda[1:], "rest_on_next")]
    return [(onset, ph[vi], [], coda, "rest_on_next")]


def _merge_pending(syls):
    """Gabungkan konsonan 'pending' (sisa kata terpotong) ke onset suku berikutnya."""
    out = []
    pend = []
    for s in syls:
        onset, nu, coda, rest, tag = s
        onset = list(onset) + pend if pend else list(onset)
        pend = list(rest)
        out.append((onset, nu, list(coda)))
    if pend and out:
        # tidak ada nada berikutnya: taruh sebagai coda suku terakhir
        out[-1][2].extend(pend)
    return out
