p = "sing_v2.py"
s = open(p, encoding="utf-8").read()
bad = ("        lp_hp += a_hp * (Lw[0] - lp_hp)\n"
       "        lw0 = Lw[0] - lp_hp                      # hanya riak frekuensi tinggi (formant)\n"
       "        fbe += 0.0005 * (abs(lw0) - fbe)")
good = ("        lp_hp += a_hp * (Lw[0] - lp_hp)\n"
        "        lw0 = Lw[0] - lp_hp                      # hanya riak frekuensi tinggi (formant)\n"
        "        fbe += 0.0005 * (abs(Lw[0]) - fbe)       # skala = amplitudo total (bukan pita sempit)")
assert bad in s
s = s.replace(bad, good)
open(p, "w", encoding="utf-8").write(s)
print("patched normalisasi")
