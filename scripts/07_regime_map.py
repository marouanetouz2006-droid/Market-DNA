import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

ASSETS = [
    "QQQ",
    "XLF",
    "XLE",
    "FEZ",
    "TLT",
    "GLD"
]

FACTORS = [
    "MKT",
    "RATE",
    "CREDIT",
    "USD",
    "OIL",
    "VIX"
]

CASE_YEARS = [
    2020,
    2022,
    2025,
    2026
]


# ============================================================
# DIRECTORIES
# ============================================================

Path("output/figures").mkdir(
    parents=True,
    exist_ok=True
)

Path("output/tables").mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD QUANT DATA
# ============================================================

df = pd.read_csv(
    "data/processed/quant_research_dataset.csv",
    index_col="Date",
    parse_dates=True
)

df = df.sort_index()


# ============================================================
# TRAINING REFERENCE: 2015-2021
# ============================================================

reference = df.loc[
    "2015-01-01":"2021-12-31"
].copy()


dmidi_mean = reference["DMIDI"].mean()
dmidi_std = reference["DMIDI"].std()

breadth_mean = reference[
    "DRIFT_BREADTH"
].mean()

breadth_std = reference[
    "DRIFT_BREADTH"
].std()


# ============================================================
# STANDARDISED REGIME COORDINATES
# ============================================================

df["DMIDI_Z"] = (
    df["DMIDI"] - dmidi_mean
) / dmidi_std

df["BREADTH_Z"] = (
    df["DRIFT_BREADTH"] - breadth_mean
) / breadth_std


# ============================================================
# REGIME CLASSIFICATION
# ============================================================

def classify_regime(row):

    x = row["DMIDI_Z"]
    y = row["BREADTH_Z"]

    if pd.isna(x) or pd.isna(y):
        return np.nan

    if x < 0 and y < 0:
        return "Stable / Local"

    if x >= 0 and y < 0:
        return "Concentrated Shift"

    if x < 0 and y >= 0:
        return "Broad Mild Transition"

    return "Systemic Structural Shift"


df["REGIME"] = df.apply(
    classify_regime,
    axis=1
)


# ============================================================
# REGIME MAP
# ============================================================

plt.figure(
    figsize=(10, 8)
)

regimes = [
    "Stable / Local",
    "Concentrated Shift",
    "Broad Mild Transition",
    "Systemic Structural Shift"
]

for regime in regimes:

    subset = df[
        df["REGIME"] == regime
    ]

    plt.scatter(
        subset["DMIDI_Z"],
        subset["BREADTH_Z"],
        s=12,
        alpha=0.45,
        label=regime
    )


plt.axvline(
    0,
    linestyle="--"
)

plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Dynamic drift intensity (DMIDI z-score)"
)

plt.ylabel(
    "Drift breadth z-score"
)

plt.title(
    "Market DNA Regime Map"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "output/figures/"
    "Market_DNA_Regime_Map.png",
    dpi=180
)

plt.close()


# ============================================================
# REGIME FREQUENCY TABLE
# ============================================================

regime_summary = (
    df["REGIME"]
    .value_counts(
        dropna=True
    )
    .rename_axis(
        "Regime"
    )
    .reset_index(
        name="Observations"
    )
)

regime_summary[
    "Share"
] = (
    regime_summary["Observations"]
    /
    regime_summary[
        "Observations"
    ].sum()
)


regime_summary.to_csv(
    "output/tables/"
    "regime_frequency.csv",
    index=False
)


# ============================================================
# AUTOMATIC CASE STUDY SELECTION
# Highest SYSTEMIC_DRIFT in each year
# ============================================================

case_rows = []


for year in CASE_YEARS:

    yearly = df[
        df.index.year == year
    ].dropna(
        subset=[
            "SYSTEMIC_DRIFT"
        ]
    )

    if yearly.empty:
        continue

    event_date = (
        yearly[
            "SYSTEMIC_DRIFT"
        ]
        .idxmax()
    )

    row = df.loc[
        event_date
    ]

    case_rows.append(
        {
            "Year":
                year,

            "Date":
                event_date,

            "DMIDI":
                row["DMIDI"],

            "DriftBreadth":
                row[
                    "DRIFT_BREADTH"
                ],

            "SystemicDrift":
                row[
                    "SYSTEMIC_DRIFT"
                ],

            "Regime":
                row[
                    "REGIME"
                ],

            "DominantAsset":
                row[
                    "DOMINANT_ASSET"
                ],

            "DominantShare":
                row[
                    "DOMINANT_SHARE"
                ],

            "VIX":
                row[
                    "VIX_LEVEL"
                ]
        }
    )


case_table = pd.DataFrame(
    case_rows
)

case_table.to_csv(
    "output/tables/"
    "historical_case_studies.csv",
    index=False
)


# ============================================================
# CASE STUDY PLOTS
# ============================================================

for case in case_rows:

    year = case["Year"]

    event_date = pd.Timestamp(
        case["Date"]
    )

    dominant_asset = (
        case[
            "DominantAsset"
        ]
    )


    # --------------------------------------------------------
    # Contribution chart
    # --------------------------------------------------------

    contribution_values = []

    for asset in ASSETS:

        column = (
            f"SHARE_{asset}"
        )

        contribution_values.append(
            df.loc[
                event_date,
                column
            ] * 100
        )


    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        ASSETS,
        contribution_values
    )

    plt.ylabel(
        "Contribution to dynamic drift (%)"
    )

    plt.title(
        f"{year} Market DNA Episode — "
        f"{event_date.date()}"
    )

    plt.tight_layout()

    plt.savefig(
        f"output/figures/"
        f"Case_{year}_"
        f"Drift_Contributions.png",
        dpi=180
    )

    plt.close()


    # --------------------------------------------------------
    # Dominant asset rolling beta analysis
    # --------------------------------------------------------

    beta_file = (
        f"data/processed/"
        f"rolling_betas/"
        f"{dominant_asset}.csv"
    )

    beta = pd.read_csv(
        beta_file,
        index_col="Date",
        parse_dates=True
    )

    beta_cols = []

    for factor in FACTORS:

        if (
            f"beta_{factor}"
            in beta.columns
        ):

            beta_cols.append(
                f"beta_{factor}"
            )

        elif factor in beta.columns:

            beta_cols.append(
                factor
            )


    window_start = (
        event_date
        - pd.Timedelta(
            days=500
        )
    )

    window_end = (
        event_date
        + pd.Timedelta(
            days=250
        )
    )

    beta_window = (
        beta.loc[
            window_start:
            window_end,
            beta_cols
        ]
        .dropna()
        .copy()
    )


    # Standardise betas so factors with different units
    # can be compared visually.

    beta_mean = (
        beta_window.mean()
    )

    beta_std = (
        beta_window.std()
        .replace(
            0,
            np.nan
        )
    )

    beta_z = (
        beta_window
        - beta_mean
    ) / beta_std


    plt.figure(
        figsize=(12, 6)
    )

    for column in beta_z.columns:

        label = (
            column.replace(
                "beta_",
                ""
            )
        )

        plt.plot(
            beta_z.index,
            beta_z[column],
            label=label
        )


    plt.axvline(
        event_date,
        linestyle="--",
        label="Selected episode"
    )

    plt.axhline(
        0,
        linestyle="--"
    )

    plt.ylabel(
        "Standardised rolling beta"
    )

    plt.title(
        f"{year} — "
        f"{dominant_asset} Financial DNA"
    )

    plt.legend(
        ncol=3
    )

    plt.tight_layout()

    plt.savefig(
        f"output/figures/"
        f"Case_{year}_"
        f"{dominant_asset}_DNA.png",
        dpi=180
    )

    plt.close()


# ============================================================
# CURRENT MARKET POSITION
# ============================================================

latest = (
    df.dropna(
        subset=[
            "DMIDI_Z",
            "BREADTH_Z"
        ]
    )
    .iloc[-1]
)


current = pd.DataFrame(
    {
        "Metric": [
            "Date",
            "DMIDI",
            "DMIDI z-score",
            "Breadth",
            "Breadth z-score",
            "Systemic Drift",
            "Regime",
            "Dominant Asset",
            "Dominant Share"
        ],

        "Value": [
            latest.name,
            latest[
                "DMIDI"
            ],
            latest[
                "DMIDI_Z"
            ],
            latest[
                "DRIFT_BREADTH"
            ],
            latest[
                "BREADTH_Z"
            ],
            latest[
                "SYSTEMIC_DRIFT"
            ],
            latest[
                "REGIME"
            ],
            latest[
                "DOMINANT_ASSET"
            ],
            latest[
                "DOMINANT_SHARE"
            ]
        ]
    }
)


current.to_csv(
    "output/tables/"
    "current_regime_state.csv",
    index=False
)


# ============================================================
# SAVE DATASET WITH REGIMES
# ============================================================

df.to_csv(
    "data/processed/"
    "market_dna_regimes.csv"
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n======================================"
)

print(
    "MARKET DNA — REGIME ANALYSIS COMPLETE"
)

print(
    "======================================\n"
)


print(
    "Current date:",
    latest.name.date()
)

print(
    "Current regime:",
    latest["REGIME"]
)

print(
    "DMIDI z-score:",
    round(
        latest["DMIDI_Z"],
        3
    )
)

print(
    "Breadth z-score:",
    round(
        latest["BREADTH_Z"],
        3
    )
)

print(
    "Dominant asset:",
    latest[
        "DOMINANT_ASSET"
    ]
)

print(
    "Dominant share:",
    f"{latest['DOMINANT_SHARE'] * 100:.1f}%"
)


print(
    "\nSelected historical cases:\n"
)

print(
    case_table[
        [
            "Year",
            "Date",
            "DMIDI",
            "DriftBreadth",
            "SystemicDrift",
            "Regime",
            "DominantAsset",
            "DominantShare"
        ]
    ]
)


print(
    "\nSaved regime map:"
)

print(
    "output/figures/"
    "Market_DNA_Regime_Map.png"
)

print(
    "\nSaved case-study figures "
    "to output/figures/"
)