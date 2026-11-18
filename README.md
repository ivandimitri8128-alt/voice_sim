# Simulasi Suara Manusia

Instalasi (Windows, venv aktif):
    pip install numpy scipy matplotlib numba

Tahap 1 : python tahap1.py   -> pita suara two-mass
Tahap 2 : python tahap2.py   -> vokal a-e-i-o-u
Tahap 3 (LAMA)  : python sing_song.py songs/burung_kakatua.txt   -> nyanyi, two-mass (sering "mati", pitch meluncur)
Tahap 3 (BARU)  : python sing_v2.py songs/burung_kakatua.txt     -> nyanyi v2 (disarankan)
Verifikasi      : python check_v2.py songs/burung_kakatua.txt output/burung_kakatua_v2.wav

Baca ANALISIS.md untuk penyebab suara tidak mirip manusia & apa yang diubah.
numba sangat disarankan (tanpa numba render bisa puluhan menit).

Opsi sing_v2: --oq 0.56 (default), --no-falsetto (matikan register falset >= Bb4), --vibrato, --tempo, --seed, --timbre, --rough, --space, --stereo, --branch.
check_v2: tambahkan --no-falsetto bila lagu dirender tanpa falset.

---

## HANDOFF (untuk AI/pengembang lain) — Tahap 3 v2.1 "ekspresi manusiawi"

**File aktif:** `sing_v2.py` (mesin nyanyi), `score.py` (parser partitur), `check_v2.py` (verifikasi), `vowels.py`, `songs/*.txt`.
`voice_sim_tahap2_fixed/` = salinan LAMA (jangan diedit; nomor baris di sana tidak sama dengan file utama).
`sing_song.py`/`timeline.py` = mesin tahap 3 lama (two-mass), tidak dipakai lagi.

**Alur:** `plan_song` (waktu + Mark) -> `build_segments` -> `build_controls` (pitch, vibrato, bentuk saluran, amp, dinamika)
-> `glottal_flow` -> `tract_run` (waveguide Kelly-Lochbaum) -> `post_process` (`level_vowels`, reverb).

**Perubahan v2.1**
1. Vibrato organik (`build_controls`): laju 5,2-5,8 Hz diundi per FRASA; tunda acak 0,35-0,45 s; ramp 0,3 s;
   10-20% nada panjang lurus; kedalaman = `vib_scale(panjang nada)` (0 <0,45 s | 60% <0,9 s | 100% >=0,9 s);
   menyusut sampai -40% di akhir frasa. Konstanta di blok `VIB_*` atas file. Catatan: nada 0,45-0,7 s praktis
   tanpa vibrato karena tunda + pelepasan 0,1 s (konsekuensi spesifikasi).
2. Dinamika: `plan_song` kini mengembalikan `Mark(t, midi)` (tetap bisa di-unpack 2 elemen) dengan field
   `beat, beats, dur, down, last`. `down` = ketukan 1 (default 4 beat/birama, `--bar`). `build_controls`:
   aksen +1,5-2 dB di `down`, messa di voce (+2,5 dB puncak 60%, nada > 0,5 s), busur frasa -3 dB di kedua ujung.
   Hasil disimpan di `ctl["dyn"]`; `level_vowels(..., dyn=)` hanya meratakan beda JENIS vokal dan mempertahankan dyn.
3. Timing manusiawi (`plan_song(rng=...)`): onset +-10-20 ms, istirahat = max(partitur, acak 0,35-0,8 s),
   nada akhir frasa +5-8%. RNG timing = `timing_rng(seed)`; `check_v2.py` memakainya (`--seed/--no-human/--bar`
   harus sama dengan saat render). `--no-human` mematikan timing acak.
4. Perbaikan: `score.py` kini mengenali `R:0.5`/`r:1` sebagai istirahat (dulu diam-diam dibuang, jadi
   `baby_persis.txt` tidak punya jeda napas sama sekali). Bug vibrato lama (offset konstan 0,28/0,10 s dan
   lantai kedalaman 35% sehingga nada pendek ikut bervibrato) digantikan fungsi berbasis panjang nada.

**Opsi baru:** `--no-human`, `--bar N`. **Tes cepat:** `python sing_v2.py songs/baby_persis.txt` lalu
`python check_v2.py songs/baby_persis.txt output/baby_persis_v2.wav`.
Tanpa numba render ~3,7x durasi audio (tes mini 7,6 s = 28 s).

**Ide berikutnya:** variasi tempo antar frasa, konsonan/vokal per-nada yang lebih ekspresif, meter selain 4/4 dari partitur.

---

## HANDOFF — Tahap 3 v2.2 "suara bernyanyi lebih hidup" (setelah v2.1)

**Perubahan (semua di `sing_v2.py` kecuali dicatat):**
1. Vibrato: tunda dibatasi 40% panjang nada (`VIB_DELAY_FRAC`), span minimal 0,15 s -> nada 0,5-0,9 s kini ikut bervibrato (di Baby 73/83 nada; dulu 2/83).
2. Ketukan: `score.py` mengenali `meter N`, `pickup P` (anakrusis, beat), `barreset 1` (hitung birama ulang tiap frasa). `Mark.half` = ketukan 3 (aksen setengah). Baby: 5 -> 11 nada di ketukan 1 (+9 di ketukan 3) dengan `pickup 1.5` + `barreset 1`. Override CLI: `--bar --pickup --barreset`.
3. Intonasi manusiawi (`build_controls`, `--detune C`, default 6, 0 = mati): salah-nada tetap per nada (sigma C cents), nada panjang melorot 0-8 cents, overshoot 8-18 cents di langkah legato 1-5 semiton, awal frasa mulai 20-40 cents di bawah lalu naik ~70 ms. Error pitch median Baby 3 -> 4 cents (terburuk 25).
4. Hembusan (`--breath G`, default 1, 0 = mati): `ctl["breath"]` menaikkan derau glotis di awal frasa/nada dan akhir frasa; `oq` berbeda per frasa dan sedikit mengikuti dinamika (`ctl["oq_off"]`).
5. Singer's formant (`--singer S`, default 0,8): `singer_formant()` = EQ puncak 2,9 kHz (+6 dB x S) hanya di vokal, di `post_process` sebelum `level_vowels`. Pita 2,4-3,4 kHz naik ~3,4 dB (S=0,8), formant F1-F3 tidak berubah.
   **Sudah dicoba dan GAGAL:** penyempitan epilaring fisik (segmen 1-9 ke 0,45 cm^2) di waveguide: pita 2,4-3,4 kHz tak berubah, F3 turun, /u/ rusak (F3 2291 -> 1328 Hz di S=1). Jangan diulang tanpa mengubah model akustik.
6. `check_v2.py`: opsi `--seed --no-human --bar --pickup --barreset` (harus sama dengan render) dan baris `[SINGER]`.

**Catatan penting:** `check_v2.py` belum mengukur vibrato/dinamika/intonasi; belum ada tes regresi. `voice_sim_tahap2_fixed/` (salinan lama) dikeluarkan dari zip v2.2 agar ringan; file lama ada di zip sebelumnya.
**Render:** tanpa numba ~3,2x durasi audio (Baby 66,5 s = 212 s). **Output terbaru:** `output/baby_persis_v2_2.wav`.
## HANDOFF — Tahap 3 v2.3 "timbre manusia" (setelah v2.2)

Tujuan: menutup jarak "suara vokal sintetis" ke manusia. Semua perubahan di `sing_v2.py`.

**Bukti awal (render `bohongi_hati`, v2.2):** spektrum 500-1k/1-2k/2-4k = +2/-3/-2 dB, padahal manusia +1/-6/-12.
HNR 25 dB (batas atas = terlalu bersih). Itu sumber kesan "logam/his".

**Perubahan**
1. `timbre_eq()` di `post_process` sebelum `level_vowels`: low shelf +1,5 dB @200 Hz, high shelf -8 dB @1,5 kHz,
   dan cekungan -16 dB @3,2 kHz (Q 2,2) yang diskalakan `g_hi` (0 di bawah f0 230 Hz, penuh di atas 420 Hz,
   karena spektrum model di nada rendah memang sudah gelap). LP 9 kHz diganti 18 kHz supaya /s/ /c/ tidak tumpul.
   Hasil: `bohongi_hati` 500-1k/1-2k/2-4k = +1/-5/-13 dB, `baby` -8/-13/-16 dB. High shelf dulu dicoba 1,2-2 kHz
   tapi band 1-2 kHz jadi terlalu redup; cekungan sempit lebih tepat sasaran.
   **Jangan** pakai `timbre_eq` sebelum `singer_formant` bila singer dipakai untuk mengukur, urutannya sudah benar.
2. `jaw_gain()`/`apply_jaw()`: nada di atas MIDI 60 membuka vokal (penyempitan digeser ke depan, luas dinaikkan
   2,8%/semiton, maksimum 1,75x) setelah `areas` dihaluskan. F1 /i/ pada Eb5 535 -> target ~450-500 (v2.2 641),
   /u/ 426, /o/ 420. Batas penting: maksimum >1,9x membuat /i/ /o/ jatuh ke puncak harmonik dan pengukur LPC salah
   memilih F1 (sudah dicoba 2,2x: /i/ 649, /o/ 773).
3. Opsi CLI `--timbre` (default 1, 0 = mati/datar seperti v2.2).

**Regresi:** `baby_persis` setelah perubahan: pitch median 4 cents (sama), 0 vokal mati, HNR 23,7, klik 13
(dari 0 di v2.2; semua di 17,2 s dan 55,37 s, nada pendek di sekitar napas, belum ditelusuri).
Formant baby tidak bergeser berarti karena median MIDI 58 < 60.

**Catatan:** dimensi suara masih kurang dua hal terbesar dan sengaja belum disentuh: sumber pita suara parametris
(tanpa interaksi sumber-filter) dan murmur hidung tiruan. Urutan berikutnya yang disarankan: modulasi kasar 20-40 Hz
di `amp` (1-3%), side branch 2,9 kHz untuk singer's formant, IR stereo 0,7-1,2 s.

## HANDOFF — Tahap 3 v2.4 "getaran otot & ruang" (setelah v2.3)

1. Kekasaran sumber (`--rough R`, default 1, 0 = mati): `build_controls` menambahkan derau lowpass 40 Hz ke
   `amp` (3%) dan derau lowpass 8 Hz ke `cents` (4 cents), keduanya digerbangi frasa. Ini mengembalikan
   "ketidaksempurnaan" yang dulu datang dari two-mass.
   **Terukur:** HNR turun hanya 0,3-0,5 dB (23,5 vs 23,2 pada rough 0), pitch & formant tak berubah.
   Efeknya harus dinilai dengan telinga; angka tidak akan menunjukkan apa pun.
2. `room_ir()` + `apply_space()` menggantikan IR 0,3 s lama: pre-delay 18 ms, 6 pantulan awal, ekor 0,75 s
   yang diredam HF, basah default 0,16 (dulu 0,08). `--space` mengatur basah, `--stereo` menulis WAV 2 kanal
   dengan dua IR berbeda (korelasi kanal 0,97, beda RMS 0,027 - cukup lebar, tidak pecah).
   `check_v2` merata-ratakan kanal bila WAV stereo.
3. `post_process` kini mengembalikan list saat stereo; `main` menulis int16 interleaved.

**Regresi:** `baby_persis` pitch 4 cents, 0 vokal mati, klik 0, HNR 22,3. Tidak ada regresi fungsional.

**Belum dikerjakan & tetap jadi penyebab utama sisa suara sintetis:** interaksi sumber-filter (pita suara parametris),
murmur hidung tiruan, konsonan seragam (VOT 22 ms & penutupan 60-65 ms tetap), rubato/dinamika sempit (2,5 dB),
dan singer's formant fisik (side branch) yang kini makin tertutup cekungan 3,2 kHz.

## HANDOFF — Tahap 3 v2.5/v2.6 "menghilangkan kesan sintetis" (setelah v2.4)

**v2.5 — singer's formant fisik (GAGAL, dimatikan).**
Dicoba cabang samping (side branch) di waveguide: `tract_run` diberi port ketiga di segmen `BRANCH_IDX`,
delay line `BRANCH_LEN` slot, ujung tertutup `BR_OPEN=+1` (puncak) atau terbuka `-1` (lekuk).
Pada SATU nada panjang hasilnya bagus dan stabil: idx 36, alpha 0,10 -> pita 2,4-3,4 kHz naik
**+5,9 dB** dengan F1-F3 hampir tak bergeser (/a/ 703->667, /i/ 269->356). Tidak ada pertumbuhan energi
pada nada 30 detik (rms per 6 bagian: 38/51/62/65/55/38 vs tanpa cabang 43/56/68/73/61/42).
**Tapi pada lagu penuh hasilnya rusak:** frame pitch terdeteksi 6-28% (normal 86%), HNR 8-14 dB,
60 klik, dan `level_vowels` meledak (rentang level 37 dB). Penyebab: `refl` berubah cepat (formant bergerak),
sehingga loop cabang bisa memompa energi; menggerbang cabang per vokal memperburuk (sambungan tak invarian waktu).
Uji ulang bila mau: butuh junction yang benar-benar pasif (cek konservasi energi per sub-langkah) dan
delay line yang ikut interpolasi blok seperti `refl`. **Jangan** hanya menaikkan alpha.
Kode tetap ada, `BRANCH_ALPHA = 0.0` (mati); `--branch X` untuk eksperimen.

**v2.6 — yang dipakai (semua terdengar & terukur lebih hidup):**
1. Rubato: `plan_song` memberi tiap FRASA skala tempo sendiri, `RUBATO_RANGE` (0,92-1,10).
2. Kekuatan frasa: `PHRASE_DYN_DB` = 5 dB; tiap frasa punya tingkat kenyaringan sendiri (jarak pp-ff lebih lebar).
3. Konsonan tidak seragam: `build_segments(events, rng)` mengacak penutupan plosif 35-90 ms,
   frikatif 60-140 ms, VOT 13-37 ms. `cons_rng(seed)` dipakai juga oleh `check_v2` (rencana waktu harus sama).
   Ada pengaman: onset <= 55% dan koda <= 45% ruang suku kata, supaya vokal selalu >= 0,08 s.
4. Spektrum: cekungan 3,2 kHz diringankan -16 -> -12 dB, `SINGER_DEFAULT` 0,8 -> 1,0.
   Hasil bohongi: 500-1k/1-2k/2-4k/4-8k = +1/-6/-10/-12 dB, singer -21,2 -> -18,1 dB.
5. `--rough`, `--space`, `--stereo`, `--branch` (lihat v2.4/v2.5).

**Hasil akhir (bohongi_hati, stereo):** pitch median 7 cents (terburuk 25), 85/85 nada terukur, 0 vokal mati,
0 klik, HNR 23,7, formant /a/ 681/1042/2428, /e/ 499/1775/2477.
**baby_persis:** 6 cents (terburuk 21), 0 vokal mati, 0 klik, HNR 21,3, spektrum -7/-13/-14/-13.
Catatan: rentang level naik ke 13-17 dB (dulu 10-11) karena dinamika frasa disengaja; `check_v2` masih
mencetak ambang lama 6 dB di docstring, jadi jangan dibaca sebagai regresi.

**Langkah berikutnya (peta ke suara "saluran napas tersimulasi penuh"):** sumber pita suara berosilasi sendiri terkopel ke saluran (tekanan subglotis), artikulator parametrik (jaw/lidah/bibir/velum/laring), derau turbulen dari aliran, tes regresi, numba/C.
