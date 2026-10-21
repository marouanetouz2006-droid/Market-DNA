import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from src.config import ASSETS


# ============================================================
# SETTINGS
# ============================================================

REFERENCE_WINDOW = 504   # About 2 trading years
GAP = 252                # About 1 trading year

FACTOR_NAMES = [
    "MKT",
    "RATE",
    "CREDIT",
    "USD",
    "OIL",
    "VIX"
]


# ============================================================
# DYNAMIC MAHALANOBIS DISTANCE
# ============================================================

def dynamic_mahalanobis(beta_df):

    # Detect the six beta columns automatically
    factor_cols = []

    for factor in FACTOR_NAMES:

        if factor in beta_df.columns:
            factor_cols.append(factor)

        elif f"beta_{factor}" in beta_df.columns:
            factor_cols.append(f"beta_{factor}")

        else:
            raise ValueError(
                f"Could not find beta column for factor: {factor}"
            )

    betas = beta_df[factor_cols].dropna().copy()

    dynamic_ids = pd.Series(
        index=betas.index,
        dtype=float
    )

    first_valid_position = REFERENCE_WINDOW + GAP

    for i in range(first_valid_position, len(betas)):

        # Historical reference period:
        # 504 observations ending 252 days before today
        reference_end = i - GAP
        reference_start = reference_end - REFERENCE_WINDOW

        reference = betas.iloc[
            reference_start:reference_end
        ]

        current_beta = betas.iloc[i].to_numpy()

        # Historical average DNA
        mu = reference.mean().to_numpy()

        # Historical covariance matrix
        cov = reference.cov().to_numpy()

        # Small regularisation term for numerical stability
        avg_variance = np.mean(np.diag(cov))

        ridge = max(
            avg_variance * 1e-6,
            1e-12
        )

        cov_regularised = (
            cov
            + ridge * np.eye(cov.shape[0])
        )

        inv_cov = np.linalg.pinv(
            cov_regularised
        )

        difference = current_beta - mu

        distance_squared = (
            difference
            @ inv_cov
            @ difference
        )

        dynamic_ids.iloc[i] = np.sqrt(
            max(distance_squared, 0)
        )

    return dynamic_ids


# ============================================================
# CALCULATE DYNAMIC IDS FOR EACH ASSET
# ============================================================

dynamic_ids = {}

for asset in ASSETS:

    beta = pd.read_csv(
        f"data/processed/rolling_betas/{asset}.csv",
        index_col="Date",
        parse_dates=True
    )

    dynamic_ids[f"DIDS_{asset}"] = dynamic_mahalanobis(
        beta
    )


# ============================================================
# CREATE DYNAMIC MIDI
# ============================================================

dynamic_df = pd.concat(
    dynamic_ids,
    axis=1
)

dynamic_df["DMIDI"] = dynamic_df.mean(
    axis=1,
    skipna=False
)


# ============================================================
# SAVE DYNAMIC MIDI
# ============================================================

Path("data/processed").mkdir(
    parents=True,
    exist_ok=True
)

dynamic_df.to_csv(
    "data/processed/dynamic_midi.csv"
)


# ============================================================
# TRAINING-PERIOD THRESHOLD
# 2015-2021 only
# ============================================================

training_dmidi = dynamic_df.loc[
    "2015-01-01":"2021-12-31",
    "DMIDI"
].dropna()

threshold = training_dmidi.quantile(0.95)


# ============================================================
# GRAPH
# ============================================================

Path("output/figures").mkdir(
    parents=True,
    exist_ok=True
)

ax = dynamic_df["DMIDI"].plot(
    figsize=(12, 5),
    title="Dynamic Market Identity Drift Index (DMIDI)"
)

ax.axhline(
    threshold,
    linestyle="--",
    label="2015-2021 95th percentile"
)

ax.set_ylabel(
    "Dynamic Mahalanobis Distance"
)

ax.legend()

plt.tight_layout()

plt.savefig(
    "output/figures/Dynamic_MIDI.png",
    dpi=180
)

plt.close()


# ============================================================
# ADD DYNAMIC MIDI TO RESEARCH DATASET
# WITHOUT OVERWRITING VERSION 1
# ============================================================

research = pd.read_csv(
    "data/processed/research_dataset.csv",
    index_col="Date",
    parse_dates=True
)

research_v2 = research.join(
    dynamic_df,
    how="left"
)

research_v2.to_csv(
    "data/processed/research_dataset_v2.csv"
)


# ============================================================
# RESULTS
# ============================================================

print(dynamic_df.tail())

print(
    "2015-2021 DMIDI 95th percentile:",
    threshold
)

print(
    "Saved data/processed/dynamic_midi.csv"
)

print(
    "Saved data/processed/research_dataset_v2.csv"
)

print(
    "Saved output/figures/Dynamic_MIDI.png"
)