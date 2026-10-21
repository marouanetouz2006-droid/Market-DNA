import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from src.config import *
from src.models import mahalanobis_ids

Path("output/figures").mkdir(parents=True, exist_ok=True)

ids = {}
for asset in ASSETS:
    beta = pd.read_csv(f"data/processed/rolling_betas/{asset}.csv", index_col="Date", parse_dates=True)
    ids[f"IDS_{asset}"] = mahalanobis_ids(
        beta,
        CALIBRATION_START,
        CALIBRATION_END
    )

ids_df = pd.concat(ids, axis=1)
ids_df["MIDI"] = ids_df.mean(axis=1, skipna=False)
ids_df.to_csv("data/processed/midi.csv")

threshold = ids_df.loc[CALIBRATION_START:CALIBRATION_END, "MIDI"].quantile(0.95)
ax = ids_df["MIDI"].plot(figsize=(12,5), title="Market Identity Drift Index (MIDI)")
ax.axhline(threshold, linestyle="--", label="Calibration 95th percentile")
ax.legend()
plt.tight_layout()
plt.savefig("output/figures/MIDI.png", dpi=180)
plt.close()

print(ids_df.tail())
print("Calibration 95th percentile:", threshold)
print("Saved data/processed/midi.csv and output/figures/MIDI.png")
