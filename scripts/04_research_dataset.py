import pandas as pd
from src.config import *
from src.targets import past_realized_volatility, forward_realized_volatility, forward_average_correlation, forward_drawdown_indicator

base = pd.read_csv("data/processed/analysis_dataset.csv", index_col="Date", parse_dates=True)
midi = pd.read_csv("data/processed/midi.csv", index_col="Date", parse_dates=True)

r = base["MKT"]
out = pd.concat([
    base, midi,
    past_realized_volatility(r, 20),
    forward_realized_volatility(r, 5),
    forward_realized_volatility(r, 20),
    forward_average_correlation(base[ASSETS], 20),
    forward_drawdown_indicator(r, 20, -0.05),
], axis=1)

out.to_csv("data/processed/research_dataset.csv")
print(out.tail())
print("Saved data/processed/research_dataset.csv")
