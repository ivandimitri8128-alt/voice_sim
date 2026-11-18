"""
Tahap 3 - Pita suara + saluran suara untuk lagu (nyanyi).

Perbedaan dengan Tahap 2 (voice.py):
  * pitch = kurva bebas (melodi), bukan q konstan
  * bentuk saluran = array penuh per sampel (konsonan vokal dinamis)
  * sumber derau (frikatif/aspirasi) + ledakan plosif ditambahkan di glotis
  * nasalisasi (cabang hidung) opsional

API utama:
  run_song(n, fs, ps, q, x0, a_cur, a_next, mix, voiced, uv, nas,
           burst, burst_lo, burst_hi, oversample) -> (p_lip, ug)
"""
import math

import numpy as np

try:
    from numba import njit
except ImportError:          # numba opsional
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

from vocal_fold import M1, M2, K1, K2, KC, ZETA1, ZETA2, ZETA_COL, L as FOLD_L, D1, D2, RHO

C_SOUND = 350.0
BLOCK = 64
KICK = True   # kick fonasi saat onset voiced (start osilasi pita suara)
RAMP = True   # ramp injeksi 20 ms saat onset (anti klik)


# ------------------------------------------------------------------ area helper
def area_at_block(a_cur, a_next, mix, idx, nas):
    """Log-interpolasi area (cm^2) untuk frame idx."""
    a = a_cur[idx]
    b = a_next[idx]
    m = mix[idx]
    out = np.exp(np.log(np.maximum(a, 1e-3)) * (1.0 - m) + np.log(np.maximum(b, 1e-3)) * m)
    if nas[idx] > 0:
        out = out.copy()
        out[0:14] *= (1.0 - 0.35 * nas[idx])
    return out


# ------------------------------------------------------------------ derau
def filter_noise(rng, ctl, fs, amp):
    """
    Derau pita sempit per segmen (frikatif) + derau lebar (aspirasi).
    Hasil: array float. Filter orde-1 time-varying (murah & cukup).
    """
    from scipy.signal import sosfilt, butter

    n = ctl.n
    white = rng.standard_normal(n)

    # derau lebar (aspirasi/breathiness) -> high-pass ringan
    sos_hp = butter(2, 400.0 / (fs / 2.0), btype="high", output="sos")
    wide = sosfilt(sos_hp, white)

    # derau pita untuk tiap segmen: pakai bandpass butter statis pada beberapa pita
    out = np.zeros(n)
    # pita-pita tetap; bobot dipilih per sampel dari nz_lo/nz_hi
    bands = [(500, 1500), (1500, 3000), (3000, 5000), (5000, 9000), (7000, 12000)]
    filt = []
    for lo, hi in bands:
        hi = min(hi, fs / 2.0 * 0.95)
        sos = butter(2, [lo / (fs / 2.0), hi / (fs / 2.0)], btype="band", output="sos")
        filt.append(sosfilt(sos, white))
    filt = np.array(filt)

    centers = np.array([(lo + hi) / 2.0 for lo, hi in bands])
    for k in range(5):
        w = np.exp(-0.5 * ((centers[k] - (ctl.nz_lo + ctl.nz_hi) / 2.0) /
                           np.maximum((ctl.nz_hi - ctl.nz_lo) / 2.0, 200.0)) ** 2)
        out += filt[k] * w * ctl.nz
    out += wide * ctl.nz * 0.35
    return out * amp


# ------------------------------------------------------------------ mesin
def run_song(n, fs, ps, q, x0, a_cur, a_next, mix, voiced, uv, nas,
             burst, burst_lo, burst_hi, oversample=4, noise_amp=0.10):
    """
    Jalankan two-mass + waveguide untuk seluruh n sampel.

    ps, q, x0      : array float64 panjang n
    a_cur,a_next   : (n,44) luas saluran (cm^2)
    mix            : (n,) posisi interpolasi 0..1
    voiced         : (n,) 0/1
    uv             : (n,) sumber derau (sudah difilter)
    nas            : (n,) 0..1 nasalisasi
    burst*,        : (n,) ledakan plosif + pita frekuensinya
    return         : (p_lip, ug)
    """
    n_seg = a_cur.shape[1]
    p_lip = np.zeros(n)
    ug_out = np.zeros(n)
    on_gain_dbg = np.zeros(n)
    globals()["_last_on_gain"] = on_gain_dbg

    R = np.zeros(n_seg)
    Lw = np.zeros(n_seg)
    fold = np.array([1e-5, 0.0, 0.0, 0.0])
    # state lintas-blok: posisi ramp onset, level burst, voiced sebelumnya
    state = np.zeros(3)                # [ramp_pos, br_prev, prev_vo]
    state[0] = 1e9                     # gain penuh di awal
    state[2] = voiced[0]
    refl = np.zeros(n_seg)
    z0 = 1.0
    loss = 0.9993
    lip_refl = -0.85

    # glotis: bentuk saluran diinterpolasi halus antar batas blok
    # (tanpa ini, refl & z0 melompat di batas blok -> klik periodik ~fs/BLOCK Hz)
    prev_areas = area_at_block(a_cur, a_next, mix, 0, nas)
    for s in range(0, n, BLOCK):
        nb = min(BLOCK, n - s)
        i_end = min(n - 1, s + nb - 1)
        end_areas = area_at_block(a_cur, a_next, mix, i_end, nas)

        refl_start = np.zeros(n_seg)
        refl_end = np.zeros(n_seg)
        refl_start[1:] = (prev_areas[:-1] - prev_areas[1:]) / (prev_areas[:-1] + prev_areas[1:])
        refl_end[1:] = (end_areas[:-1] - end_areas[1:]) / (end_areas[:-1] + end_areas[1:])
        refl = (refl_start, refl_end)
        z0 = (RHO * C_SOUND / (prev_areas[0] * 1e-4),
              RHO * C_SOUND / (end_areas[0] * 1e-4))

        _block(nb, s, ps, q, x0, voiced, uv, burst, burst_lo, burst_hi,
               refl, z0, R, Lw, fold, p_lip, ug_out, float(fs), oversample,
               loss, lip_refl, state, on_gain_dbg)
        prev_areas = end_areas
    return p_lip, ug_out


@njit(cache=True)
def _block(nb, s, ps, q, x0, voiced, uv, burst, burst_lo, burst_hi,
           refl, z0, R, Lw, fold, out, ugo, fs, oversample, loss, lip_refl,
           state, og):
    """Satu blok BLOCK sampel (loop Python; numba tidak dipakai karena array 2D)."""
    n_seg = R.shape[0]
    dt = 1.0 / (fs * oversample)
    x1, x2, v1, v2 = fold[0], fold[1], fold[2], fold[3]
    JR = np.empty(n_seg)
    JL = np.empty(n_seg + 1)
    voiced_prev = 0.0
    rs, re = refl
    dr = re - rs
    z0s, z0e = z0
    dz0 = z0e - z0s
    rc = np.empty(n_seg)               # refl halus per sampel
    # burst: batasi laju kenaikan (attack) TANPA bergantung batas blok
    br_step = 0.30 / (0.0025 * fs)     # skala penuh dalam 2.5 ms
    br_prev = state[1]
    ramp_len = int(0.020 * fs)  # onset voiced: 20 ms (anti klik), per sampel audio
    ramp_pos = state[0]
    prev_vo = state[2]

    for b in range(nb):
        i = s + b
        # smoothstep: turunan kontinu di batas blok (linier = kink -> dengung komb)
        u = b / max(nb - 1, 1)
        wss = 0.5 - 0.5 * math.cos(math.pi * u)
        rc[:] = rs + dr * wss
        z0 = z0s + dz0 * wss
        qq = q[i]
        k1 = K1 * qq * qq
        k2 = K2 * qq * qq
        kc = KC * qq * qq
        r1 = 2.0 * ZETA1 * math.sqrt(M1 * k1)
        r2 = 2.0 * ZETA2 * math.sqrt(M2 * k2)
        rc1 = 2.0 * ZETA_COL * math.sqrt(M1 * k1)
        rc2 = 2.0 * ZETA_COL * math.sqrt(M2 * k2)
        h1 = 3.0 * k1
        h2 = 3.0 * k2
        psg_in = ps[i]
        x0i = x0[i]
        vo = voiced[i]

        # onset ramp: 0->1 halus saat voiced mulai (anti klik pulsa pertama)
        if vo > 0.5 and prev_vo <= 0.5:
            ramp_pos = 0
            if KICK:
                # kick fonasi: mulsa dari posisi nyaris tertutup & bentuk konvergen,
                # sehingga Bernoulli menariknya menutup lalu tekanan mendorong buka
                # -> osilasi langsung tumbuh (tanpa ini, titik setimbang stabil = mati)
                x1 = 0.12 * x0i
                x2 = -0.02 * x0i
                v1 = 0.0
                v2 = 0.0
        prev_vo = vo
        if ramp_pos < ramp_len:
            ramp_pos += 1
            on_gain = 0.5 - 0.5 * math.cos(math.pi * ramp_pos / ramp_len) if RAMP else 1.0
        else:
            on_gain = 1.0

        l0 = Lw[0]
        ug_acc = 0.0
        for _o in range(oversample):
            a1 = x0i + x1
            a2 = x0i + x2
            ag1 = 2.0 * FOLD_L * (a1 if a1 > 0.0 else 0.0)
            ag2 = 2.0 * FOLD_L * (a2 if a2 > 0.0 else 0.0)
            amin = ag1 if ag1 < ag2 else ag2

            dp0 = psg_in - 2.0 * l0
            if vo > 0.5 and amin > 1e-12 and dp0 > 0.0:
                cq = RHO / (2.0 * amin * amin)
                ug = (-z0 + math.sqrt(z0 * z0 + 4.0 * cq * dp0)) / (2.0 * cq)
            else:
                ug = 0.0
            psg = 2.0 * l0 + ug * z0

            if ug > 0.0:
                if ag2 <= ag1:
                    p1 = psg_in - 0.5 * RHO * (ug / ag1) ** 2
                else:
                    p1 = psg
                p2 = psg
            else:
                p1 = psg_in if vo > 0.5 else psg
                p2 = psg

            f1 = p1 * FOLD_L * D1
            f2 = p2 * FOLD_L * D2

            c1 = 0.0
            c2 = 0.0
            r1e = r1
            r2e = r2
            if a1 < 0.0:
                c1 = -h1 * a1
                r1e = r1 + rc1
            if a2 < 0.0:
                c2 = -h2 * a2
                r2e = r2 + rc2

            acc1 = (-k1 * x1 - r1e * v1 - kc * (x1 - x2) + f1 + c1) / M1
            acc2 = (-k2 * x2 - r2e * v2 - kc * (x2 - x1) + f2 + c2) / M2
            v1 += acc1 * dt
            v2 += acc2 * dt
            x1 += v1 * dt
            x2 += v2 * dt
            ug_acc += ug

        ug_m = ug_acc / oversample
        ugo[i] = ug_m
        og[i] = on_gain

        # ---- waveguide, 2 langkah per sampel audio
        lip_acc = 0.0
        inj = (ug_m * z0 + (0.0 if vo > 0.5 else uv[i])) * on_gain
        br = burst[i]
        if br > 0:
            # ledakan plosif: attack dibatasi laju kenaikan per sampel
            br = min(br, br_prev + br_step)
            inj += br * 0.30 * (_pseudo_noise(i, burst_lo[i], burst_hi[i], fs)) * on_gain
        br_prev = br
        src = Lw[0] + inj
        for _t in range(2):
            JR[0] = src
            w_arr = rc[1:n_seg] * (R[:n_seg - 1] - Lw[1:n_seg])
            JR[1:n_seg] = R[:n_seg - 1] + w_arr
            JL[1:n_seg] = Lw[1:n_seg] + w_arr
            JL[n_seg] = lip_refl * R[n_seg - 1]
            R[:] = JR * loss
            Lw[:] = JL[1:] * loss
            lip_acc += R[n_seg - 1] * (1.0 + lip_refl)
        out[i] = 0.5 * lip_acc

    fold[0] = x1
    fold[1] = x2
    fold[2] = v1
    fold[3] = v2
    state[0] = ramp_pos
    state[1] = br_prev
    state[2] = prev_vo


@njit(cache=True)
def _pseudo_noise(i, lo, hi, fs):
    """Derau deterministik murah untuk ledakan."""
    h = (i * 2654435761) & 0xFFFFFFFF
    h ^= (h >> 13)
    h = (h * 1274126177) & 0xFFFFFFFF
    v = ((h & 0xFFFF) / 32768.0) - 1.0
    return v