"""CLI wrapper; functions live in p7_tests_lib.py"""
import sys, numpy as np, pandas as pd
from common import load_panel, DATASET
from p7_tests_lib import per_origin, ci

df = load_panel(); rng = np.random.default_rng(0)
prop = sys.argv[1]
for base in sys.argv[2:]:
    pb, pt = per_origin(prop, df); bb, bt = per_origin(base, df)
    for lvl, A, B, k in [("bottom", pb, bb, "ay"), ("total", pt, bt, "y")]:
        res = [ci(A, B, "ae", k, rng, L=L) for L in (1, 4, 8)]
        d, w = res[0][0], res[0][3]
        cis = "  ".join(f"L{L}[{r[1]*100:+.2f},{r[2]*100:+.2f}]" for L, r in zip((1, 4, 8), res))
        print(f"{DATASET} {prop} vs {base:30s} {lvl:6s} dWAPE {d*100:+.2f} pp  {cis}  wins {w:.2f}")
