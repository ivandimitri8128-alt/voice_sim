# voice_sim — Simulasi Suara Manusia Bernyanyi

Sintesis suara manusia dari nol dengan Python: model pita suara, saluran suara (vocal tract) berbasis
waveguide Kelly-Lochbaum, lalu dipakai untuk **menyanyikan partitur teks** lengkap dengan lirik,
vibrato, dinamika, dan reverb ruang. Tanpa sampel suara dan tanpa model neural. Semuanya fisika dan pemrosesan sinyal.

## Fitur

- **Tahap 1:** pita suara model *two-mass* (fisika penuh).
- **Tahap 2:** vokal a, e, i, o, u lewat saluran suara 44 segmen.
- **Tahap 3 (`sing_v2.py`):** mesin menyanyi dari partitur teks.
  - Pitch datar per nada dengan luncuran antar nada, portamento, dan register falset.
  - Vibrato organik, dinamika per frasa, rubato, dan timing manusiawi.
  - Konsonan dengan tempat artikulasi yang benar, serta intonasi dan hembusan napas manusiawi.
  - Koreksi timbre (singer's formant dan EQ spektrum) serta reverb ruang, mono atau stereo.
- **A cappella (`acappella.py`):** aransemen 4 suara (lead, harmoni, arpeggio, bass) *The Sound of Silence*.
- **Verifikasi terukur (`check_v2.py`):** pitch, formant, HNR, level, dan klik diukur dari file audio.

## Instalasi

Butuh Python 3.10+.

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` berisi `numpy`, `scipy`, `matplotlib`, dan `numba`.
`numba` sangat disarankan: tanpanya render sekitar 3x durasi audio, atau jauh lebih lambat untuk Tahap 1-2.

## Cara pakai

```powershell
# Tahap 1: pita suara two-mass
python tahap1.py

# Tahap 2: vokal a-e-i-o-u
python tahap2.py

# Tahap 3: menyanyikan lagu (hasil WAV di folder output/)
python sing_v2.py songs/baby_persis.txt

# Verifikasi hasil render (opsi seed/bar/pickup harus sama dengan saat render)
python check_v2.py songs/baby_persis.txt output/baby_persis_v2.wav

# A cappella 4 suara (--bars 6 untuk uji cepat)
python acappella.py
```

### Opsi `sing_v2.py`

| Opsi | Default | Fungsi |
|---|---|---|
| `-o`, `--out` | otomatis | Nama file WAV keluaran |
| `--tempo` | dari partitur | Ganti BPM |
| `--vibrato` | 28 | Kedalaman vibrato (cents) |
| `--oq` | 0.56 | Open quotient glotis (lebih kecil = harmonik atas lebih kuat) |
| `--no-falsetto` | mati | Matikan register falset (>= Bb4) |
| `--no-human` | mati | Matikan timing acak (onset terkunci) |
| `--bar`, `--pickup`, `--barreset` | dari partitur | Atur meter, anakrusis, dan hitungan birama |
| `--detune` | 6 | Salah-nada manusiawi dalam cents (0 = mati) |
| `--breath` | 1 | Pengali hembusan napas (0 = mati) |
| `--singer` | 1.0 | Kekuatan singer's formant (~2.9 kHz) |
| `--timbre` | 1 | Koreksi spektrum (0 = datar) |
| `--rough` | 1 | Kekasaran sumber suara (0 = mati) |
| `--space` | 0.16 | Level reverb ruang |
| `--stereo` | mati | Tulis WAV stereo (dua IR reverb) |
| `--seed` | 1 | Seed acak, agar hasil bisa diulang |
| `--no-plot` | mati | Jangan buat grafik |

Opsi eksperimen: `--branch` (side branch waveguide, saat ini nonaktif karena tidak stabil) dan `--couple`.

## Format partitur

File teks di folder `songs/`. Satu nada ditulis `<nada><oktaf>:<durasi>=<lirik>`:

```text
tempo 65        # BPM
meter 4         # ketukan per birama
pickup 1.5      # ketukan anakrusis sebelum ketukan 1
barreset 1      # hitung birama ulang di tiap frasa

Bb3:0.5=Yu Bb3:0.5=know Eb4:1=love Bb3:0.5=me
R:0.5           # istirahat (napas)
```

- Nada: `C D E F G A B`, boleh `#` atau `b` (misalnya `C#4`, `Bb3`). Oktaf 2-6.
- Durasi dalam ketukan: 1 = seperempat, 0.5 = seperdelapan, 4 = penuh.
- Lirik dipisah `-` per suku kata dan `_` per kata. Tanda `-` tanpa lirik berarti nada ditahan (tie).
- `R:0.5` atau `rest` = istirahat. Baris berawalan `#` = komentar.

Aturan lengkap ada di docstring `score.py`.

## Lagu contoh

| File | Keterangan |
|---|---|
| `songs/baby_persis.txt` | Intro, pre-chorus, dan chorus, sekitar 60 detik |
| `songs/bohongi_hati.txt` | Contoh lagu Indonesia |
| `songs/sound_of_silence.txt` | Dipakai oleh `acappella.py` |

## Struktur proyek

| File | Isi |
|---|---|
| `sing_v2.py` | Mesin menyanyi utama (aktif) |
| `score.py` | Parser partitur teks |
| `check_v2.py` | Verifikasi pitch, formant, HNR, level, dan klik |
| `acappella.py` | Aransemen 4 suara |
| `vowels.py`, `phonemes.py` | Bentuk saluran suara dan fonem |
| `vocal_fold.py`, `voice.py` | Model pita suara dan sintesis dasar (Tahap 1-2) |
| `tahap1.py`, `tahap2.py` | Demo Tahap 1 dan 2 |
| `sing_song.py`, `timeline.py`, `song.py` | Mesin Tahap 3 lama (two-mass), tidak dipakai lagi |
| `tune_*.py`, `debug_*.py`, `_probe_*.py`, `_patch.py` | Skrip eksperimen dan debugging |
| `ANALISIS.md` | Analisis kenapa suara awal tidak mirip manusia |
| `CHANGELOG.md` | Riwayat teknis per versi (v2.1 sampai v2.6) |

## Alur pemrosesan (`sing_v2.py`)

```text
plan_song -> build_segments -> build_controls -> glottal_flow -> tract_run -> post_process
(waktu)      (fonem/segmen)    (pitch, vibrato,   (sumber        (waveguide   (EQ timbre,
                                dinamika, napas)   glotis)       Kelly-        level vokal,
                                                                 Lochbaum)     reverb)
```

## Batasan

- Verifikasi dilakukan lewat pengukuran, bukan pendengaran. Kesan "terdengar manusia" tetap perlu dinilai dengan telinga.
- Jalur nyanyi memakai sumber pita suara parametris (bukan two-mass), jadi belum ada interaksi sumber-filter.
- Nasal memakai murmur tiruan, belum cabang hidung waveguide penuh.
- Belum ada tes regresi otomatis.

Rencana berikutnya: pita suara yang berosilasi sendiri dan terkopel ke saluran suara, artikulator parametrik
(rahang, lidah, bibir, velum), derau turbulen dari aliran, tes regresi, dan akselerasi numba atau C.
