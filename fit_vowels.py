import numpy as np
from scipy.optimize import minimize
from scipy.interpolate import PchipInterpolator

C = 350.0; N = 44; FS = 44100
DX = C/(2*FS)            # panjang segmen (m)
RHO = 1.14
NK = 9                    # titik kontrol
xk = np.linspace(0, 1, NK)
xs = (np.arange(N)+0.5)/N

def areas_from_ctrl(logA):
    return np.exp(PchipInterpolator(xk, logA)(xs))   # cm^2

def formants(A_cm2, fmax=3600, df=10.0):
    A = A_cm2*1e-4
    f = np.arange(100, fmax, df)
    k = 2*np.pi*f/C
    p = np.zeros_like(f, dtype=complex); U = np.ones_like(f, dtype=complex)   # di bibir: p=0, U=1
    for i in range(N-1, -1, -1):          # dari bibir ke glotis
        Z = RHO*C/A[i]
        c, s = np.cos(k*DX), np.sin(k*DX)
        p, U = c*p + 1j*Z*s*U, 1j*s/Z*p + c*U
    ug = U.real
    idx = np.where(np.sign(ug[:-1]) != np.sign(ug[1:]))[0]
    res = []
    for i in idx:
        res.append(f[i] - ug[i]*df/(ug[i+1]-ug[i]))
    return np.array(res)

TARGET = {  # F1, F2, F3 pria dewasa (Hz) - nilai tipikal
 'a': (730, 1090, 2440), 'e': (530, 1840, 2480), 'i': (270, 2290, 3010),
 'o': (570,  840, 2410), 'u': (300,  870, 2240)}

def cost(logA, tgt):
    A = areas_from_ctrl(logA)
    fm = formants(A)
    if len(fm) < 3: return 1e6
    err = sum(((fm[j]-tgt[j])/tgt[j])**2 for j in range(3))
    smooth = 0.02*np.sum(np.diff(logA)**2)
    bounds = np.sum(np.clip(np.abs(logA-np.log(3.0))-np.log(4.0), 0, None)**2)*10  # 0.75..12 cm2
    return err*100 + smooth + bounds

if __name__ == "__main__":
    import sys
    out = {}
    rng = np.random.default_rng(1)
    for v, tgt in [(sys.argv[1], TARGET[sys.argv[1]])]:
        best = None
        for trial in range(5):
            x0 = np.log(3.0) + rng.normal(0, 0.6, NK)
            r = minimize(cost, x0, args=(tgt,), method='Nelder-Mead',
                         options={'maxiter':1200,'xatol':1e-3,'fatol':1e-6})
            if best is None or r.fun < best.fun: best = r
        A = areas_from_ctrl(best.x)
        fm = formants(A)[:4]
        print(v, "target", tgt, "hasil", np.round(fm), "cost", round(best.fun,4), "A min/max", A.min().round(2), A.max().round(2))
        out[v] = best.x
    np.save(f"vowel_{sys.argv[1]}.npy", out[sys.argv[1]])
