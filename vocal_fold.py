"""
Tahap 1 - Model pita suara two-mass (Ishizaka-Flanagan, disederhanakan)
Paru-paru (tekanan subglotal) -> pita suara -> aliran udara glotis (Ug)

Satuan: SI (meter, kg, detik, Pascal).
"""
import math

try:
    from numba import njit
except ImportError:          # numba opsional; tanpa numba tetap jalan (lebih lambat)
    def njit(*a, **k):
        if a and callable(a[0]):
            return a[0]
        return lambda f: f

RHO = 1.14          # kg/m^3, kerapatan udara hangat-lembap

# --- Parameter pita suara (nilai tipikal pria dewasa, Ishizaka & Flanagan 1972)
M1, M2 = 0.125e-3, 0.025e-3        # massa bawah/atas (kg)
K1, K2, KC = 80.0, 8.0, 25.0       # kekakuan pegas (N/m) & kopling
ZETA1, ZETA2, ZETA_COL = 0.1, 0.6, 1.0   # rasio redaman
L = 1.4e-2                          # panjang pita suara (m)
D1, D2 = 0.25e-2, 0.05e-2           # tebal massa bawah/atas (m)
X0 = 0.02e-2                        # setengah-celah istirahat (m)


@njit(cache=True)
def simulate(n_out, fs, ps_arr, q_arr, x0_arr, oversample=4, noise_amp=0.0):
    """
    n_out     : jumlah sampel keluaran
    fs        : sampling rate keluaran (Hz)
    ps_arr    : tekanan subglotal per sampel (Pa)
    q_arr     : faktor ketegangan per sampel (f0 ~ 110 Hz * q)
    x0_arr    : setengah-celah istirahat per sampel (m); kecil = pressed, besar = breathy
    return    : (aliran glotis Ug [m^3/s], area glotis min [m^2])
    """
    dt = 1.0 / (fs * oversample)
    ug_out = [0.0] * n_out
    a_out = [0.0] * n_out

    x1 = 1e-5
    x2 = 0.0
    v1 = 0.0
    v2 = 0.0

    for n in range(n_out):
        ps = ps_arr[n]
        q = q_arr[n]
        x0 = x0_arr[n]

        k1 = K1 * q * q
        k2 = K2 * q * q
        kc = KC * q * q
        r1 = 2.0 * ZETA1 * math.sqrt(M1 * k1)
        r2 = 2.0 * ZETA2 * math.sqrt(M2 * k2)
        h1 = 3.0 * k1
        h2 = 3.0 * k2

        ug_acc = 0.0
        a_acc = 0.0
        for _ in range(oversample):
            # lebar celah (>= 0)
            a1 = x0 + x1
            a2 = x0 + x2
            ag1 = 2.0 * L * (a1 if a1 > 0.0 else 0.0)
            ag2 = 2.0 * L * (a2 if a2 > 0.0 else 0.0)
            amin = ag1 if ag1 < ag2 else ag2

            # Aliran (Bernoulli kuasi-stasioner, tekanan supraglotal ~ 0)
            if amin > 1e-12 and ps > 0.0:
                ug = amin * math.sqrt(2.0 * ps / RHO)
            else:
                ug = 0.0

            # Tekanan pada tiap massa
            if ug > 0.0 and ag2 <= ag1:
                # celah menyempit (convergent): tekanan menurun menuju keluaran
                p1 = ps - 0.5 * RHO * (ug / ag1) ** 2
                p2 = 0.0
            else:
                # celah melebar (divergent): jet terlepas, tekanan ~ atmosfer
                p1 = 0.0
                p2 = 0.0
            if ug == 0.0:
                p1 = ps if ag1 <= 0.0 else 0.0   # tertutup: tekanan penuh di bawah
                p2 = 0.0

            f1 = p1 * L * D1
            f2 = p2 * L * D2

            # Gaya
            c1 = 0.0
            c2 = 0.0
            r1e = r1
            r2e = r2
            if a1 < 0.0:                 # tumbukan pita suara kiri-kanan (massa 1)
                c1 = -h1 * a1
                r1e = r1 + 2.0 * ZETA_COL * math.sqrt(M1 * k1)
            if a2 < 0.0:
                c2 = -h2 * a2
                r2e = r2 + 2.0 * ZETA_COL * math.sqrt(M2 * k2)

            acc1 = (-k1 * x1 - r1e * v1 - kc * (x1 - x2) + f1 + c1) / M1
            acc2 = (-k2 * x2 - r2e * v2 - kc * (x2 - x1) + f2 + c2) / M2

            # Euler semi-implisit (simplektik)
            v1 += acc1 * dt
            v2 += acc2 * dt
            x1 += v1 * dt
            x2 += v2 * dt

            ug_acc += ug
            a_acc += amin

        ug_out[n] = ug_acc / oversample
        a_out[n] = a_acc / oversample
    return ug_out, a_out
