"""
Tahap 2 - Pita suara (two-mass) + saluran suara (waveguide Kelly-Lochbaum), saling terhubung.

Alur fisika per sampel:
  paru-paru (Ps) -> glotis -> aliran udara Ug -> saluran suara (tabung 44 segmen) -> bibir -> suara
  Tekanan balik dari saluran suara memengaruhi aliran glotis dan getaran pita suara
  (interaksi sumber-filter, hal yang tidak ada di model source-filter biasa).
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
BLOCK = 32          # panjang blok sampel (bentuk saluran diperbarui tiap blok)


@njit(cache=True)
def run_block(nb, ps_arr, q_arr, x0_arr, refl, z0, R, L, fold, lip_out, ug_out,
              fs, oversample, loss, lip_refl):
    n = R.shape[0]
    dt = 1.0 / (fs * oversample)
    x1 = fold[0]; x2 = fold[1]; v1 = fold[2]; v2 = fold[3]
    JR = np.empty(n)
    JL = np.empty(n + 1)

    for b in range(nb):
        ps = ps_arr[b]; q = q_arr[b]; x0 = x0_arr[b]
        k1 = K1 * q * q; k2 = K2 * q * q; kc = KC * q * q
        r1 = 2.0 * ZETA1 * math.sqrt(M1 * k1)
        r2 = 2.0 * ZETA2 * math.sqrt(M2 * k2)
        rc1 = 2.0 * ZETA_COL * math.sqrt(M1 * k1)
        rc2 = 2.0 * ZETA_COL * math.sqrt(M2 * k2)
        h1 = 3.0 * k1; h2 = 3.0 * k2

        l0 = L[0]
        ug_acc = 0.0
        for _s in range(oversample):
            a1 = x0 + x1
            a2 = x0 + x2
            ag1 = 2.0 * FOLD_L * (a1 if a1 > 0.0 else 0.0)
            ag2 = 2.0 * FOLD_L * (a2 if a2 > 0.0 else 0.0)
            amin = ag1 if ag1 < ag2 else ag2

            # Aliran glotis: Bernoulli + beban saluran suara (solusi implisit, persamaan kuadrat)
            dp0 = ps - 2.0 * l0
            if amin > 1e-12 and dp0 > 0.0:
                cq = RHO / (2.0 * amin * amin)
                ug = (-z0 + math.sqrt(z0 * z0 + 4.0 * cq * dp0)) / (2.0 * cq)
            else:
                ug = 0.0
            psg = 2.0 * l0 + ug * z0          # tekanan tepat di atas glotis

            if ug > 0.0:
                if ag2 <= ag1:                # celah menyempit (convergent)
                    p1 = ps - 0.5 * RHO * (ug / ag1) ** 2
                else:                         # celah melebar: jet terlepas
                    p1 = psg
                p2 = psg
            else:                             # tertutup: tekanan paru penuh di bawah
                p1 = ps
                p2 = psg

            f1 = p1 * FOLD_L * D1
            f2 = p2 * FOLD_L * D2

            c1 = 0.0; c2 = 0.0; r1e = r1; r2e = r2
            if a1 < 0.0:
                c1 = -h1 * a1; r1e = r1 + rc1
            if a2 < 0.0:
                c2 = -h2 * a2; r2e = r2 + rc2

            acc1 = (-k1 * x1 - r1e * v1 - kc * (x1 - x2) + f1 + c1) / M1
            acc2 = (-k2 * x2 - r2e * v2 - kc * (x2 - x1) + f2 + c2) / M2
            v1 += acc1 * dt; v2 += acc2 * dt
            x1 += v1 * dt;   x2 += v2 * dt
            ug_acc += ug

        ug_m = ug_acc / oversample
        ug_out[b] = ug_m

        # ---- saluran suara: waveguide berjalan 2x lebih cepat dari sampling audio
        # (satu segmen = c/(2*fs) = 0.397 cm, jadi 44 segmen = 17.5 cm)
        lip_acc = 0.0
        for _t in range(2):
            JR[0] = L[0] + ug_m * z0              # sumber aliran di glotis
            w = refl[1:n] * (R[:n - 1] - L[1:n])      # Kelly-Lochbaum, gelombang tekanan
            JR[1:n] = R[:n - 1] + w
            JL[1:n] = L[1:n] + w
            JL[n] = lip_refl * R[n - 1]
            R[:] = JR * loss
            L[:] = JL[1:] * loss
            lip_acc += R[n - 1] * (1.0 + lip_refl)
        lip_out[b] = 0.5 * lip_acc

    fold[0] = x1; fold[1] = x2; fold[2] = v1; fold[3] = v2


def reflection_coeffs(areas):
    """areas: array luas (cm^2) -> koefisien refleksi antar segmen (indeks 1..n-1)."""
    a = np.asarray(areas, dtype=float)
    r = np.zeros_like(a)
    r[1:] = (a[:-1] - a[1:]) / (a[:-1] + a[1:])
    return r


def synthesize(n, fs, ps, q, x0, area_fn, loss=0.9992, lip_refl=-0.85, oversample=4):
    """
    n        : jumlah sampel
    ps,q,x0  : array kontrol per sampel (tekanan paru, ketegangan, celah istirahat)
    area_fn  : fungsi t(detik) -> array luas saluran (cm^2), panjang N segmen
    return   : (suara di bibir, aliran glotis)
    """
    N = len(area_fn(0.0))
    R = np.zeros(N); Lw = np.zeros(N)
    fold = np.array([1e-5, 0.0, 0.0, 0.0])
    out = np.zeros(n); ugo = np.zeros(n)
    ps = np.asarray(ps, float); q = np.asarray(q, float); x0 = np.asarray(x0, float)

    for s in range(0, n, BLOCK):
        nb = min(BLOCK, n - s)
        areas = area_fn((s + nb / 2) / fs)
        refl = reflection_coeffs(areas)
        z0 = RHO * C_SOUND / (areas[0] * 1e-4)
        lo = np.zeros(nb); uo = np.zeros(nb)
        run_block(nb, ps[s:s + nb], q[s:s + nb], x0[s:s + nb], refl, z0, R, Lw, fold,
                  lo, uo, float(fs), oversample, loss, lip_refl)
        out[s:s + nb] = lo; ugo[s:s + nb] = uo
    return out, ugo
