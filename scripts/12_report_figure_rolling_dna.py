import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

ASSET = "XLF"

FACTORS = [
    "MKT",
    "RATE",
    "CREDIT",
    "USD",
    "OIL",
    "VIX"
]


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

Path(
    "report/final_figures"
).mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD ROLLING BETAS
# ============================================================

beta = pd.read_csv(
    f"data/processed/rolling_betas/{ASSET}.csv",
    index_col="Date",
    parse_dates=True
)

beta = beta.sort_index()


# ============================================================
# DETECT COLUMN NAMES
#
# Works whether columns are named:
# MKT, RATE, ...
#
# or:
# beta_MKT, beta_RATE, ...
# ============================================================

beta_columns = {}

for factor in FACTORS:

    if f"beta_{factor}" in beta.columns:

        beta_columns[factor] = (
            f"beta_{factor}"
        )

    elif factor in beta.columns:

        beta_columns[factor] = factor

    else:

        raise ValueError(
            f"Could not find rolling beta "
            f"for factor {factor}"
        )


# ============================================================
# CREATE FIGURE
# ============================================================

fig, axes = plt.subplots(
    2,
    3,
    figsize=(13, 7),
    sharex=True
)

axes = axes.flatten()


for ax, factor in zip(
    axes,
    FACTORS
):

    column = beta_columns[
        factor
    ]

    ax.plot(
        beta.index,
        beta[column],
        linewidth=1.4
    )

    ax.axhline(
        0,
        linestyle="--",
        linewidth=0.8,
        alpha=0.7
    )

    ax.set_title(
        f"{factor} beta",
        fontsize=11,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Rolling beta",
        fontsize=9
    )

    ax.tick_params(
        axis="both",
        labelsize=8
    )

    ax.grid(
        alpha=0.15
    )


# ============================================================
# GLOBAL TITLE
# ============================================================

fig.suptitle(
    "XLF Financial DNA — "
    "Time-Varying Factor Exposures",
    fontsize=15,
    fontweight="bold",
    y=0.98
)


# ============================================================
# FORMAT DATES
# ============================================================

for ax in axes:

    ax.tick_params(
        axis="x",
        rotation=30
    )


plt.tight_layout(
    rect=[
        0,
        0,
        1,
        0.94
    ]
)


# ============================================================
# SAVE
# ============================================================

output_file = (
    "report/final_figures/"
    "Figure_1_XLF_Rolling_Financial_DNA.png"
)


plt.savefig(
    output_file,
    dpi=300,
    bbox_inches="tight"
)

plt.close()


print(
    "Saved:",
    output_file
)