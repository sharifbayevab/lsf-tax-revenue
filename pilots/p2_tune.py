"""P2t: leakage-free tuning of trained neural models + 3-seed ensemble on the test origins.
Validation origins 2016-03..2018-12 (quarterly, 12); all validation targets end <= 2019-12 = first test origin.
Selection metric: mean MASE (seasonal scale on each training window) on validation.
usage: p2_tune.py LSTM|NHITS|PatchTST [tune|final]"""
import sys, os, json, itertools, time, numpy as np, pandas as pd, torch
import logging; logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, PatchTST, LSTM
from neuralforecast.losses.pytorch import MAE
from common import load_panel, ORIGINS, H, save_fc, RES
from macro import ipca_index, focus, expected_path

VAL = pd.date_range("2016-03-01", "2018-12-01", freq="3MS")
GRID = {
    "LSTM": dict(input_size=[24, 36, 60], hidden=[32, 64, 128], max_steps=[500, 1500], learning_rate=[1e-3, 3e-3]),
    "NHITS": dict(input_size=[24, 36, 60], hidden=[256, 512], max_steps=[500, 1500], learning_rate=[1e-3, 3e-3]),
    "PatchTST": dict(input_size=[24, 36, 60], hidden=[64, 128], max_steps=[300, 1000], learning_rate=[1e-4, 1e-3]),
}


def make(name, c, seed):
    kw = dict(h=H, input_size=c["input_size"], max_steps=c["max_steps"], learning_rate=c["learning_rate"],
              scaler_type="robust", random_seed=seed, enable_progress_bar=False, enable_model_summary=False,
              logger=False, loss=MAE())
    if name == "LSTM":
        return LSTM(encoder_hidden_size=c["hidden"], decoder_hidden_size=c["hidden"], recurrent=False, **kw)
    if name == "NHITS":
        return NHITS(mlp_units=[[c["hidden"], c["hidden"]]] * 3, **kw)
    if name == "PatchTST":
        return PatchTST(hidden_size=c["hidden"], patch_len=6, stride=3, **kw)


def run(name, c, seed, origins, df):
    out = []
    for o in origins:
        tr = df[df.ds <= o][["unique_id", "ds", "y"]]
        nf = NeuralForecast(models=[make(name, c, seed)], freq="MS"); nf.fit(tr)
        p = nf.predict().reset_index(drop=False).rename(columns={name: "yhat"})[["unique_id", "ds", "yhat"]]
        p["origin"] = o; p["h"] = p.groupby("unique_id").cumcount() + 1; out.append(p)
    return pd.concat(out)


def val_mase(fc, df):
    W = df.pivot(index="ds", columns="unique_id", values="y")
    rows = []
    for o in fc.origin.unique():
        tr = W.loc[:o]; sc = (tr - tr.shift(12)).abs().iloc[12:].mean()
        rows.append(pd.DataFrame({"unique_id": sc.index, "origin": o, "scale": sc.values}))
    m = fc.merge(df[["unique_id", "ds", "y"]], on=["unique_id", "ds"]).merge(pd.concat(rows), on=["unique_id", "origin"])
    return ((m.y - m.yhat).abs() / m.scale.replace(0, np.nan)).mean()


def main(name, stage):
    torch.set_float32_matmul_precision("high")
    df = load_panel(); P = ipca_index(); F = focus("IPCA")
    df["y"] = df.y / df.ds.map(P)   # real terms; validation scored in real terms
    path = RES / f"tune_{name}.csv"
    if stage == "tune":
        keys = list(GRID[name]); res = []
        for vals in itertools.product(*GRID[name].values()):
            c = dict(zip(keys, vals)); t0 = time.time()
            s = val_mase(run(name, c, 0, VAL, df), df)
            res.append({**c, "val_mase": s, "sec": time.time() - t0}); print(c, round(s, 4), flush=True)
            pd.DataFrame(res).to_csv(path, index=False)
    t = pd.read_csv(path).sort_values("val_mase"); c = t.iloc[0].to_dict(); print("best", c)
    c = {k: (int(v) if k in ("input_size", "hidden", "max_steps") else v) for k, v in c.items()}
    fcs = [run(name, c, s, ORIGINS, df) for s in (0, 1, 2)]
    fc = fcs[0][["unique_id", "origin", "ds", "h"]].copy()
    fc["yhat"] = np.mean([f.yhat.values for f in fcs], axis=0)
    parts = []
    for o, g in fc.groupby("origin"):
        parts.append(g.assign(yhat=g.yhat * g.ds.map(expected_path(o, P, F, H, "naive"))))
    save_fc(pd.concat(parts), f"{name}_tuned_real_naive")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "tune")
