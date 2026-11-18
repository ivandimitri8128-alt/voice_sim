# Kenapa suara tidak mirip orang bernyanyi — analisis & perbaikan

Semua angka di bawah diukur dari file audio (bukan dari telinga). Lihat `output/perbandingan_lama_vs_baru.png`.

## Penyebab (diurutkan dari paling merusak)

1. **Pitch tidak pernah datar (siren, bukan nada).**
   `timeline._pitch_track` hanya memberi titik pitch di awal nada (+0,08 s), lalu `np.interp`
   menarik garis lurus ke nada berikutnya. Nada 1 ketukan (0,58 s) cuma datar ~0,08 s,
   sisanya meluncur. Hanya ~16% waktu pitch target stabil.
   Terhadap nada partitur: error median **127 cents**, terburuk **673 cents**, **29 dari 44 nada meleset >50 cents**.
   `check_pitch.py` lama lolos karena membandingkan dengan kurva yang sudah meluncur dan ambang 200 cents (= 1 nada penuh).

2. **Lirik salah tempat.** `"Bu_rung"` pada satu nada membuat dua suku kata dijejalkan ke satu nada,
   lalu nada sambungan (`C3:1` tanpa lirik) memperpanjang segmen TERAKHIR, yaitu konsonan **"ng"**:
   "bu-ru-nnnng" berdengung 0,67 detik, bukan vokal "u" yang ditahan.

3. **Pita suara two-mass sering mati.** Osilasi self-excited sangat sensitif (parameter harus dijepit q<=2,3, Ps>=900 dst).
   Saat saya render ulang, ~40% jendela bersuara tidak berosilasi sama sekali (hanya aliran DC). Hasil antar-run bahkan berbeda total
   (korelasi 0,07 dengan WAV yang Anda kirim), tanda sistem dekat batas bifurkasi.

4. **Vibrato terus-menerus.** `onset = clip(t/0.25)` memakai waktu absolut lagu, bukan awal nada, sehingga vibrato 5,4 Hz
   aktif penuh di semua nada, termasuk yang 0,2 detik. Penyanyi manusia baru ber-vibrato setelah ~0,3 s pada nada panjang.

5. **Konsonan salah tempat artikulasi & derau di tempat salah.** /t/ /s/ ditaruh di segmen 20 (rongga faring), padahal gusi ada di ~37,
   bibir ~42. Derau frikatif/ledakan diinjeksi di glotis (hasilnya mirip "napas berwarna vokal"), dan ledakan plosif dibunyikan di AWAL penutupan,
   bukan saat pelepasan.

6. **Level naik-turun.** /u/ jauh lebih keras daripada /a/ di model; kompresor global tidak meratakan per suku kata.

7. (Sumber derau tersembunyi) variasi acak bentuk pulsa per periode terlalu besar menurunkan HNR; sudah dikecilkan.

## Yang diubah (sing_v2.py)

| Masalah | Perbaikan |
|---|---|
| Pitch meluncur | Tiap nada datar; luncuran 60–140 ms di batas nada (bergantung besar interval) |
| Lirik | Suku kata ekstra dipindah ke nada berikutnya; tie memperpanjang VOKAL; konsonan awal jatuh sebelum ketukan |
| Fonasi mati | Sumber glotis parametris (Rosenberg/LF) + jitter 0,4% + shimmer 3%; selalu berosilasi |
| Vibrato | Per nada panjang, tunda 0,28 s, ramp 0,25 s, rate 5,5±0,25 Hz, berhenti 0,1 s sebelum akhir |
| Konsonan | Tempat artikulasi benar, koartikulasi dengan vokal tetangga, derau di titik penyempitan, ledakan saat pelepasan + aspirasi |
| Level | Gain per vokal (+0,25 dB/semiton) setelah radiasi bibir |
| Dinamika | Redup 45% pada batas nada vokal-tanpa-konsonan (artikulasi ulang), fade frasa, tremolo kecil |

Saluran suara tetap waveguide Kelly-Lochbaum 44 segmen dari `vowels.py` — formant hasil render cocok target
(/a/ 684/1074/2437, /i/ 264/2272/3004, /u/ 298/842/2232, /e/ 495/1805/2451 Hz).

## Hasil terukur (burung_kakatua)

| | Lama | Baru |
|---|---|---|
| Error pitch median vs nada partitur | 127 cents | **4 cents** |
| Nada terburuk | 673 cents | **13 cents** |
| Nada meleset >50 cents | 29/44 | **0/44** |
| Rentang level antar vokal | >12 dB (u vs a) | **3,2 dB** |
| HNR vokal | – | **23 dB** (suara manusia 15–25) |
| Kemiringan spektrum vokal | – | 500-1k: +1, 1-2k: −6, 2-4k: −12, 4-8k: −16 dB |

Lagu lain (twinkle, es_lilin): error pitch median 4 dan 8 cents, HNR 25 dan 24 dB.

## Batasan yang jujur

- **Saya tidak bisa mendengar.** Perbaikan diverifikasi lewat pengukuran (pitch, formant, HNR, level, spektrogram).
  Kemiripan "terdengar manusia" tetap perlu Anda nilai dengan telinga.
- Di jalur nyanyi, pita suara **bukan lagi two-mass** (diganti sumber parametris agar stabil). Tahap 1–2 tetap fisika penuh.
  Konsekuensi: tidak ada interaksi sumber-filter (riak formant pada pulsa glotis).
- Lirik baris 2 dan 4 di `burung_kakatua.txt` hanya `a` (placeholder), jadi terdengar "aaa". Ganti dengan suku kata asli.
- Nasal memakai "murmur" tiruan (tap di dekat velum), bukan cabang hidung waveguide penuh.
- Suara masih ber-karakter sintetis halus karena tidak ada formant penyanyi (2,8–3,2 kHz), tidak ada variasi vokal/timbre antar nada,
  dan tidak ada ekspresi (crescendo, portamento ekspresif). Itu langkah berikutnya, bukan penyebab "tidak mirip sama sekali".

## Knob penyetelan (sing_v2.synthesize / CLI)
`--vibrato` (cents), `--oq` (0,56; lebih kecil = glotis menutup lebih tajam, harmonik atas naik), `breath_gain` (napas), `jitter`, `shimmer`, `loss`/`rg` (lebar bandwidth formant).

## Pembaruan ekspresi (v2.1)
- **Portamento**: langkah < 2 semiton tetap luncur 80 ms. Lompatan >= 2 semiton pada nada legato (tanpa konsonan awal)
  memakai peluruhan eksponensial (tau 40-60 ms) yang mulai 175 ms sebelum ketukan; lompatan >= 5 semiton ditambah
  scoop 25-40 cents. Konstanta: `PORT_MIN`, `PORT_LEAD`, `PORT_SMALL_GL`, `SCOOP_MIN` di `sing_v2.py`.
- **Register falset** (>= Bb4 = MIDI 70; MIDI 67-70 dicampur, waktu campur 100 ms): oq 0,48, jitter 0,002, amplitudo -3 dB.
  Matikan dengan `--no-falsetto`. `check_v2.py` memakai `falsetto=True` (bawaan) agar level yang turun tidak dihitung vokal mati.
- **Penutupan glotis**: oq default 0,56, rasio tp tetap 0,74.

