import numpy as np
import sing_v2 as S
from score import parse_score
from check_v2 import hnr_db

fs = S.FS
notes, cfg = parse_score("tempo 90\nC4:6=a\nC4:6=i\nC4:6=u\nC4:6=e\nC4:6=o\n")
for cp in (0.0, 0.12, 0.15):
    p, c, s, m, ph = S.synthesize(notes, cfg["tempo"], **{"human": False, "couple": cp})
    y = S.post_process(p, fs, s, c["f0"], fal=c["fal"], dyn=c["dyn"], singer=0.0, reverb=0.0)
    rms = [round(float(np.sqrt((y[int(k * len(y) / 10):int((k + 1) * len(y) / 10)] ** 2).mean())), 3) for k in range(10)]
    hn = []
    for x in s:
        if x[2] == "vowel":
            a, b = int((x[0] + 0.6) * fs), int((x[0] + 0.9) * fs)
            hn.append((x[3], round(hnr_db(y[a:b], fs, float(np.median(c["f0"][a:b]))), 1)))
    print(f"couple {cp:4.2f} {hn}")
    print("   rms", rms)
