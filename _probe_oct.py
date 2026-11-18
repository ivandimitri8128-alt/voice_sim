import numpy as np
import sing_v2 as S
from score import parse_score
from check_v2 import hnr_db

fs = S.FS
txt = "tempo 120\n"
for midi, nm in ((67, "G4"), (72, "C5"), (75, "Eb5")):
    for v in ("a", "i", "u"):
        txt += f"{nm}:8={v}\n"
notes, cfg = parse_score(txt)

for cp in (0.0, 0.03, 0.05, 0.07, 0.10):
    p, c, s, m, ph = S.synthesize(notes, cfg["tempo"], **{"human": False, "couple": cp})
    y = S.post_process(p, fs, s, c["f0"], fal=c["fal"], dyn=c["dyn"], singer=0.0, reverb=0.0)
    bad, hn = 0, []
    for k, x in enumerate([z for z in s if z[2] == "vowel"]):
        a, b = int((x[0] + 0.8) * fs), int((x[0] + 1.4) * fs)
        seg = y[a:b]
        # deteksi oktaf: autokorelasi puncak di 2*T0 lebih kuat dari T0?
        f0 = float(np.median(c["f0"][a:b]))
        w = seg * np.hanning(len(seg))
        ac = np.correlate(w, w, "full")[len(w) - 1:]
        ac /= ac[0] + 1e-12
        lag = int(fs / f0)
        r1 = ac[max(0, lag - 3):lag + 4].max()
        r2 = ac[max(0, 2 * lag - 4):2 * lag + 5].max()
        if r2 > 1.15 * r1:
            bad += 1
        hn.append(hnr_db(seg, fs, f0))
    print(f"couple {cp:4.2f}  oktaf-turun {bad}/{len(hn)}  hnr median {np.median(hn):5.1f}  p10 {np.percentile(hn, 10):5.1f}")
