from pathlib import Path
from src.config import *
from src.data import download_yahoo_prices, download_fred_series, build_analysis_dataset

Path("data/processed").mkdir(parents=True, exist_ok=True)

tickers = ASSETS + TRADED_FACTORS
prices = download_yahoo_prices(tickers, START_DATE, YF_END_DATE)
fred = download_fred_series(FRED_SERIES, START_DATE, FRED_END_DATE)
dataset = build_analysis_dataset(prices, fred, ASSETS)

dataset.to_csv("data/processed/analysis_dataset.csv")
print(dataset.head())
print(dataset.tail())
print(dataset.shape)
print("Saved data/processed/analysis_dataset.csv")
