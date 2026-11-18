"""P1 data: parse Brazil RFB arrecadacao-estado.csv into a long panel (unique_id, ds, y).

The CSV mixes three number formats: '47,953,915' (comma thousands), '91.155.798'
(dot thousands) and '8441827,67' (comma decimal). Values are nominal BRL.
"""
import re, sys
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/arrecadacao-estado.csv"
OUT = ROOT / "data/processed"

MONTHS = ['Janeiro','Fevereiro','Março','Abril','Maio','Junho','Julho','Agosto',
          'Setembro','Outubro','Novembro','Dezembro']

TH_COMMA = re.compile(r'^-?\d{1,3}(,\d{3})+$')
TH_DOT = re.compile(r'^-?\d{1,3}(\.\d{3})+$')
DEC_COMMA = re.compile(r'^-?\d+(,\d+)?$')

def parse(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return np.nan
    s = str(v).strip()
    if s == '':
        return np.nan
    if TH_COMMA.match(s) and s.count(',') >= 2:
        return float(s.replace(',', ''))
    if TH_DOT.match(s):
        return float(s.replace('.', ''))
    if DEC_COMMA.match(s):
        return float(s.replace(',', '.'))
    raise ValueError(f"unparsed value {s!r}")

# short ids for tax columns kept in the panel (continuous from 2004-01)
TAXES = {
    'IMPOSTO SOBRE IMPORTAÇÃO': 'II',
    'IPI - VINCULADO À IMPORTACAO': 'IPI_IMP',
    'IPI - OUTROS': 'IPI_OUT',
    'IRPF': 'IRPF',
    'IRPJ - ENTIDADES FINANCEIRAS': 'IRPJ_FIN',
    'IRPJ - DEMAIS EMPRESAS': 'IRPJ_DEM',
    'IRRF - RENDIMENTOS DO TRABALHO': 'IRRF_TRAB',
    'IRRF - RENDIMENTOS DO CAPITAL': 'IRRF_CAP',
    'IRRF - REMESSAS P/ EXTERIOR': 'IRRF_REM',
    'IRRF - OUTROS RENDIMENTOS': 'IRRF_OUT',
    'IMPOSTO S/ OPERAÇÕES FINANCEIRAS': 'IOF',
    'COFINS - FINANCEIRAS': 'COFINS_FIN',
    'COFINS - DEMAIS': 'COFINS_DEM',
    'CONTRIBUIÇÃO PARA O PIS/PASEP - FINANCEIRAS': 'PIS_FIN',
    'CONTRIBUIÇÃO PARA O PIS/PASEP - DEMAIS': 'PIS_DEM',
    'CSLL - FINANCEIRAS': 'CSLL_FIN',
    'CSLL - DEMAIS': 'CSLL_DEM',
    'CIDE-COMBUSTÍVEIS': 'CIDE',
    'OUTRAS RECEITAS ADMINISTRADAS': 'OUTRAS',
}
START = '2005-01-01'   # CIDE starts 2005; COFINS/PIS/CSLL splits start 2004

def main():
    df = pd.read_csv(RAW, sep=';', encoding='latin1', dtype=str)
    df['ds'] = pd.to_datetime(dict(year=df['Ano'].astype(int),
                                   month=df['Mês'].map({m: i+1 for i, m in enumerate(MONTHS)}),
                                   day=1))
    for c in TAXES:
        df[c] = df[c].map(parse)
    long = df.melt(id_vars=['UF', 'ds'], value_vars=list(TAXES), var_name='tax', value_name='y')
    long['tax'] = long['tax'].map(TAXES)
    long = long[long.ds >= START]
    long['y'] = long['y'].fillna(0.0)  # blank cell = no collection reported
    long['unique_id'] = long['UF'] + '|' + long['tax']
    long = long.sort_values(['unique_id', 'ds']).reset_index(drop=True)

    # sanity: month-to-month jumps of >=500x in large series hint at a unit/format error
    nat = long.groupby(['tax', 'ds']).y.sum().unstack(0)
    ratio = (nat / nat.shift(1)).replace([np.inf, -np.inf], np.nan)
    flags = ratio[(ratio > 500) | (ratio < 1/500)].stack()
    print("national jump flags (>500x):", len(flags))
    print(flags.head(20))

    # series filter: drop bottom series with >10% zero/negative months or tiny mean
    st = long.groupby('unique_id').agg(zero=('y', lambda s: (s <= 0).mean()), mean=('y', 'mean'))
    keep = st[(st.zero <= 0.10)].index
    print(f"series total {st.shape[0]}, kept {len(keep)} (zero-share<=10%)")
    print(st.loc[~st.index.isin(keep)].groupby(st.loc[~st.index.isin(keep)].index.str.split('|').str[1]).size())
    long['kept'] = long.unique_id.isin(keep)
    # real-time filter: decide using data up to the first forecast origin only (no look-ahead)
    pre = long[long.ds <= '2019-12-01']
    st_rt = pre.groupby('unique_id').y.apply(lambda s: (s <= 0).mean())
    keep_rt = st_rt[st_rt <= 0.10].index
    long['kept_rt'] = long.unique_id.isin(keep_rt)
    print(f"real-time filter kept {len(keep_rt)}; differs from full-sample on",
          sorted(set(keep) ^ set(keep_rt)))
    OUT.mkdir(parents=True, exist_ok=True)
    long.to_parquet(OUT / 'brazil_panel.parquet', index=False)
    print(long.ds.min().date(), long.ds.max().date(), long.ds.nunique(), "months")
    print(nat.loc['2023-01':'2024-12', ['IRPF', 'IRPJ_DEM', 'COFINS_DEM']].div(1e9).round(2))

if __name__ == '__main__':
    main()
