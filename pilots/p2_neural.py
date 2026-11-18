"""P2 trained neural models (neuralforecast), global, one fit per origin, on real-terms series.
usage: p2_neural.py NHITS|PatchTST|TFT|LSTM|DeepAR [max_steps]"""
import sys, time, os, numpy as np, pandas as pd, torch
import logging; logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, PatchTST, TFT, LSTM, DeepAR
from neuralforecast.losses.pytorch import MAE, DistributionLoss
from common import load_panel, ORIGINS, H, save_fc
from macro import ipca_index, focus, expected_path

def make(name, steps):
    kw = dict(h=H, input_size=36, max_steps=steps, scaler_type="robust", random_seed=0,
              enable_progress_bar=False, enable_model_summary=False, logger=False)
    if name == "NHITS":   return NHITS(loss=MAE(), **kw)
    if name == "PatchTST": return PatchTST(loss=MAE(), patch_len=6, stride=3, **kw)
    if name == "TFT":     return TFT(loss=MAE(), hidden_size=64, **kw)
    if name == "LSTM":
        kw["input_size"] = 36
        return LSTM(loss=MAE(), encoder_hidden_size=64, decoder_hidden_size=64, recurrent=False, **kw)
    if name == "DeepAR":  return DeepAR(loss=DistributionLoss("StudentT", level=[80]), **kw)
    raise ValueError(name)

def main(name, steps):
    torch.set_float32_matmul_precision("high")
    df = load_panel(); P = ipca_index(); F = focus("IPCA")
    df["y"] = df.y / df.ds.map(P)
    out = []; t0 = time.time()
    for o in ORIGINS:
        tr = df[df.ds <= o][["unique_id", "ds", "y"]]
        nf = NeuralForecast(models=[make(name, steps)], freq="MS")
        nf.fit(tr)
        p = nf.predict().reset_index(drop=False)
        col = [c for c in p.columns if c.startswith(name) and "lo" not in c and "hi" not in c][0]
        p = p.rename(columns={col: "yhat"})[["unique_id", "ds", "yhat"]]
        p["yhat"] = p.yhat * p.ds.map(expected_path(o, P, F, H, "naive"))
        p["origin"] = o; p["h"] = p.groupby("unique_id").cumcount() + 1
        out.append(p); print(name, o.date(), f"{time.time()-t0:.0f}s", flush=True)
    save_fc(pd.concat(out), f"{name}_real_naive")

if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 1000)
