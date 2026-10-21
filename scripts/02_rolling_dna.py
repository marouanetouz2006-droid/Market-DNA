from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from src.config import *
from src.models import estimate_rolling_dna

Path("data/processed/rolling_betas").mkdir(parents=True, exist_ok=True)
Path("output/figures").mkdir(parents=True, exist_ok=True)

df = pd.read_csv("data/processed/analysis_dataset.csv", index_col="Date", parse_dates=True)

for asset in ASSETS:
    betas = estimate_rolling_dna(df, asset, FACTOR_COLUMNS, ROLLING_WINDOW, MIN_OBS)
    betas.to_csv(f"data/processed/rolling_betas/{asset}.csv")

    if asset == "XLF":
        for col in [c for c in betas.columns if c.startswith("beta_")]:
            ax = betas[col].plot(figsize=(11,4), title=f"XLF financial DNA — {col}")
            ax.axhline(0, linewidth=1)
            plt.tight_layout()
            plt.savefig(f"output/figures/XLF_{col}.png", dpi=160)
            plt.close()

print("Saved rolling betas for all assets and XLF beta charts.")
