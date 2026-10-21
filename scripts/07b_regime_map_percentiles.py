import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

REFERENCE_START = "2015-01-01"
REFERENCE_END = "2021-12-31"

CASE_YEARS = [
    2020,
    2022,
    2025,
    2026
]

RECENT_WINDOW = 60


# ============================================================
# REGIME COLORS
# ============================================================

REGIME_COLORS = {
    "Stable / Local": "tab:blue",
    "Concentrated Shift": "tab:orange",
    "Broad Mild Transition": "tab:green",
    "Systemic Structural Shift": "tab:red"
}


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
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/processed/quant_research_dataset.csv",
    index_col="Date",
    parse_dates=True
)

df = df.sort_index()


# ============================================================
# REFERENCE PERIOD
# Percentiles defined ONLY using 2015-2021
# ============================================================

reference = df.loc[
    REFERENCE_START:
    REFERENCE_END
].copy()

reference_dmidi = (
    reference["DMIDI"]
    .dropna()
    .sort_values()
    .to_numpy()
)

reference_breadth = (
    reference["DRIFT_BREADTH"]
    .dropna()
    .sort_values()
    .to_numpy()
)


# ============================================================
# PERCENTILE FUNCTION
# ============================================================

def percentile_rank(value, reference_values):

    if pd.isna(value):
        return np.nan

    return (
        100
        * np.searchsorted(
            reference_values,
            value,
            side="right"
        )
        / len(reference_values)
    )


df["DMIDI_PERCENTILE"] = (
    df["DMIDI"]
    .apply(
        lambda x:
        percentile_rank(
            x,
            reference_dmidi
        )
    )
)

df["BREADTH_PERCENTILE"] = (
    df["DRIFT_BREADTH"]
    .apply(
        lambda x:
        percentile_rank(
            x,
            reference_breadth
        )
    )
)


# ============================================================
# PERCENTILE-BASED REGIME CLASSIFICATION
# ============================================================

def classify_percentile_regime(row):

    x = row["DMIDI_PERCENTILE"]
    y = row["BREADTH_PERCENTILE"]

    if pd.isna(x) or pd.isna(y):
        return np.nan

    if x < 50 and y < 50:
        return "Stable / Local"

    if x >= 50 and y < 50:
        return "Concentrated Shift"

    if x < 50 and y >= 50:
        return "Broad Mild Transition"

    return "Systemic Structural Shift"


df["REGIME_PERCENTILE"] = df.apply(
    classify_percentile_regime,
    axis=1
)


# ============================================================
# HISTORICAL CASE STUDIES
# Highest SYSTEMIC_DRIFT within each year
# ============================================================

case_rows = []

for year in CASE_YEARS:

    yearly = df[
        df.index.year == year
    ].dropna(
        subset=[
            "SYSTEMIC_DRIFT",
            "DMIDI_PERCENTILE",
            "BREADTH_PERCENTILE"
        ]
    )

    if yearly.empty:
        continue

    event_date = (
        yearly["SYSTEMIC_DRIFT"]
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

            "DMIDI_Percentile":
                row[
                    "DMIDI_PERCENTILE"
                ],

            "Breadth":
                row[
                    "DRIFT_BREADTH"
                ],

            "Breadth_Percentile":
                row[
                    "BREADTH_PERCENTILE"
                ],

            "SystemicDrift":
                row[
                    "SYSTEMIC_DRIFT"
                ],

            "Regime":
                row[
                    "REGIME_PERCENTILE"
                ],

            "DominantAsset":
                row[
                    "DOMINANT_ASSET"
                ],

            "DominantShare":
                row[
                    "DOMINANT_SHARE"
                ]
        }
    )


case_table = pd.DataFrame(
    case_rows
)

case_table.to_csv(
    "output/tables/"
    "historical_case_studies_percentiles.csv",
    index=False
)


# ============================================================
# CURRENT STATE
# ============================================================

latest = (
    df.dropna(
        subset=[
            "DMIDI_PERCENTILE",
            "BREADTH_PERCENTILE"
        ]
    )
    .iloc[-1]
)

latest_date = latest.name


# ============================================================
# RECENT 60-DAY TRAJECTORY
# ============================================================

recent = (
    df.dropna(
        subset=[
            "DMIDI_PERCENTILE",
            "BREADTH_PERCENTILE"
        ]
    )
    .tail(
        RECENT_WINDOW
    )
    .copy()
)


# ============================================================
# HELPER FOR VISUAL POSITION
#
# Actual percentile values remain unchanged.
# Only markers exactly near 0/100 are moved slightly inward.
# ============================================================

def plot_position(value):

    return np.clip(
        value,
        2,
        98
    )


# ============================================================
# MAIN FINAL REGIME MAP
# ============================================================

fig, ax = plt.subplots(
    figsize=(12, 8)
)


regimes = [
    "Stable / Local",
    "Concentrated Shift",
    "Broad Mild Transition",
    "Systemic Structural Shift"
]


# ------------------------------------------------------------
# Historical background
# ------------------------------------------------------------

for regime in regimes:

    subset = df[
        df["REGIME_PERCENTILE"]
        == regime
    ]

    ax.scatter(
        subset[
            "DMIDI_PERCENTILE"
        ],
        subset[
            "BREADTH_PERCENTILE"
        ],
        s=9,
        alpha=0.13,
        color=REGIME_COLORS[
            regime
        ],
        label=regime
    )


# ------------------------------------------------------------
# Regime boundaries
# ------------------------------------------------------------

ax.axvline(
    50,
    linestyle="--",
    linewidth=1.2
)

ax.axhline(
    50,
    linestyle="--",
    linewidth=1.2
)


# ------------------------------------------------------------
# Historical case-study points
# ------------------------------------------------------------

for case in case_rows:

    true_x = case[
        "DMIDI_Percentile"
    ]

    true_y = case[
        "Breadth_Percentile"
    ]

    x = plot_position(
        true_x
    )

    y = plot_position(
        true_y
    )

    year = case[
        "Year"
    ]

    regime = case[
        "Regime"
    ]


    ax.scatter(
        x,
        y,
        s=150,
        color=REGIME_COLORS[
            regime
        ],
        edgecolor="black",
        linewidth=1.4,
        zorder=8
    )


    if x > 90:

        xytext = (
            -10,
            9
        )

        horizontal_alignment = (
            "right"
        )

    else:

        xytext = (
            9,
            9
        )

        horizontal_alignment = (
            "left"
        )


    ax.annotate(
        str(year),
        xy=(
            x,
            y
        ),
        xytext=xytext,
        textcoords="offset points",
        fontsize=11,
        fontweight="bold",
        ha=horizontal_alignment
    )


# ------------------------------------------------------------
# Current market
# ------------------------------------------------------------

true_current_x = latest[
    "DMIDI_PERCENTILE"
]

true_current_y = latest[
    "BREADTH_PERCENTILE"
]

current_x = plot_position(
    true_current_x
)

current_y = plot_position(
    true_current_y
)


ax.scatter(
    current_x,
    current_y,
    s=320,
    marker="*",
    color="gold",
    edgecolor="black",
    linewidth=1.5,
    zorder=10,
    label="Current market"
)


# ------------------------------------------------------------
# Current market label
# ------------------------------------------------------------

if true_current_y == 0:

    breadth_text = (
        "Breadth < training minimum"
    )

else:

    breadth_text = (
        f"Breadth "
        f"{true_current_y:.1f}th pct."
    )


current_label = (
    f"CURRENT\n"
    f"DMIDI "
    f"{true_current_x:.1f}th pct.\n"
    f"{breadth_text}\n"
    f"{latest['DOMINANT_ASSET']} "
    f"{latest['DOMINANT_SHARE'] * 100:.1f}%"
)


ax.annotate(
    current_label,
    xy=(
        current_x,
        current_y
    ),
    xytext=(
        12,
        18
    ),
    textcoords="offset points",
    fontsize=10,
    fontweight="bold",
    ha="left"
)


# ============================================================
# QUADRANT LABELS
# ============================================================

ax.text(
    8,
    94,
    "Broad Mild Transition",
    fontsize=11,
    fontweight="bold",
    alpha=0.70
)

ax.text(
    62,
    94,
    "Systemic Structural Shift",
    fontsize=11,
    fontweight="bold",
    alpha=0.70
)

ax.text(
    8,
    8,
    "Stable / Local",
    fontsize=11,
    fontweight="bold",
    alpha=0.70
)

ax.text(
    63,
    8,
    "Concentrated Shift",
    fontsize=11,
    fontweight="bold",
    alpha=0.70
)


# ============================================================
# FORMAT MAIN GRAPH
# ============================================================

ax.set_xlim(
    0,
    100
)

ax.set_ylim(
    0,
    100
)

ax.set_xlabel(
    "Structural Drift Intensity — "
    "DMIDI Percentile",
    fontsize=11
)

ax.set_ylabel(
    "Structural Drift Breadth — "
    "Percentile",
    fontsize=11
)

ax.set_title(
    "Market DNA Regime Map\n"
    "Intensity vs Breadth "
    "of Structural Change",
    fontsize=15,
    fontweight="bold"
)


ax.legend(
    loc="center left",
    bbox_to_anchor=(
        1.02,
        0.5
    ),
    fontsize=8,
    frameon=True
)


plt.tight_layout()

plt.savefig(
    "output/figures/"
    "Market_DNA_Regime_Map_FINAL.png",
    dpi=250,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# SECOND GRAPH
# RECENT 60-DAY MARKET TRAJECTORY
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 8)
)


ax.axvline(
    50,
    linestyle="--"
)

ax.axhline(
    50,
    linestyle="--"
)


ax.plot(
    recent[
        "DMIDI_PERCENTILE"
    ],
    recent[
        "BREADTH_PERCENTILE"
    ],
    marker="o",
    markersize=3,
    linewidth=2,
    alpha=0.8
)


# ------------------------------------------------------------
# Current point
# ------------------------------------------------------------

ax.scatter(
    current_x,
    current_y,
    s=300,
    marker="*",
    color="gold",
    edgecolor="black",
    linewidth=1.5,
    zorder=10
)

ax.annotate(
    "CURRENT",
    xy=(
        current_x,
        current_y
    ),
    xytext=(
        10,
        8
    ),
    textcoords="offset points",
    fontweight="bold"
)


# ------------------------------------------------------------
# Start of trajectory
# ------------------------------------------------------------

trajectory_start = (
    recent.iloc[0]
)

start_x = plot_position(
    trajectory_start[
        "DMIDI_PERCENTILE"
    ]
)

start_y = plot_position(
    trajectory_start[
        "BREADTH_PERCENTILE"
    ]
)


ax.scatter(
    start_x,
    start_y,
    s=90,
    edgecolor="black",
    zorder=8
)

ax.annotate(
    "60 days ago",
    xy=(
        start_x,
        start_y
    ),
    xytext=(
        8,
        8
    ),
    textcoords="offset points"
)


# ------------------------------------------------------------
# Quadrant labels
# ------------------------------------------------------------

ax.text(
    10,
    10,
    "Stable / Local",
    alpha=0.7
)

ax.text(
    65,
    10,
    "Concentrated Shift",
    alpha=0.7
)

ax.text(
    8,
    93,
    "Broad Mild Transition",
    alpha=0.7
)

ax.text(
    62,
    93,
    "Systemic Structural Shift",
    alpha=0.7
)


# ------------------------------------------------------------
# Format trajectory graph
# ------------------------------------------------------------

ax.set_xlim(
    0,
    100
)

ax.set_ylim(
    0,
    100
)

ax.set_xlabel(
    "DMIDI Percentile"
)

ax.set_ylabel(
    "Drift Breadth Percentile"
)

ax.set_title(
    "Market DNA — "
    "Recent Regime Trajectory"
)

plt.tight_layout()

plt.savefig(
    "output/figures/"
    "Market_DNA_Recent_Trajectory.png",
    dpi=220
)

plt.close()


# ============================================================
# SAVE ENRICHED DATASET
# ============================================================

df.to_csv(
    "data/processed/"
    "market_dna_percentile_regimes.csv"
)


# ============================================================
# CURRENT STATE TABLE
# ============================================================

current_table = pd.DataFrame(
    {
        "Metric": [
            "Date",
            "DMIDI",
            "DMIDI Percentile",
            "Drift Breadth",
            "Breadth Percentile",
            "Systemic Drift",
            "Regime",
            "Dominant Asset",
            "Dominant Share"
        ],

        "Value": [
            latest_date,
            latest[
                "DMIDI"
            ],
            latest[
                "DMIDI_PERCENTILE"
            ],
            latest[
                "DRIFT_BREADTH"
            ],
            latest[
                "BREADTH_PERCENTILE"
            ],
            latest[
                "SYSTEMIC_DRIFT"
            ],
            latest[
                "REGIME_PERCENTILE"
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


current_table.to_csv(
    "output/tables/"
    "current_market_percentile_state.csv",
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n======================================"
)

print(
    "MARKET DNA — FINAL REGIME MAP"
)

print(
    "======================================\n"
)


print(
    "Current date:",
    latest_date.date()
)

print(
    "Current regime:",
    latest[
        "REGIME_PERCENTILE"
    ]
)

print(
    "DMIDI:",
    round(
        latest[
            "DMIDI"
        ],
        3
    )
)

print(
    "DMIDI percentile:",
    round(
        latest[
            "DMIDI_PERCENTILE"
        ],
        1
    )
)

print(
    "Breadth:",
    round(
        latest[
            "DRIFT_BREADTH"
        ],
        3
    )
)

print(
    "Breadth percentile:",
    round(
        latest[
            "BREADTH_PERCENTILE"
        ],
        1
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
    "\nHistorical cases:\n"
)

print(
    case_table[
        [
            "Year",
            "Date",
            "DMIDI_Percentile",
            "Breadth_Percentile",
            "Regime",
            "DominantAsset",
            "DominantShare"
        ]
    ]
)


print(
    "\nSaved final figures:"
)

print(
    "output/figures/"
    "Market_DNA_Regime_Map_FINAL.png"
)

print(
    "output/figures/"
    "Market_DNA_Recent_Trajectory.png"
)