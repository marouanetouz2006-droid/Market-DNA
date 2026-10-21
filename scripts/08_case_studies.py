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
# LOAD MARKET DNA DATA
# ============================================================

df = pd.read_csv(
    "data/processed/"
    "market_dna_percentile_regimes.csv",
    index_col="Date",
    parse_dates=True
)

df = df.sort_index()


# ============================================================
# SELECT CASE STUDIES
#
# Objective rule:
# highest SYSTEMIC_DRIFT within each year
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
                row[
                    "DMIDI"
                ],

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


# ============================================================
# SAVE SUMMARY TABLE
# ============================================================

case_table.to_csv(
    "output/tables/"
    "case_study_summary.csv",
    index=False
)


# ============================================================
# CREATE EACH CASE STUDY
# ============================================================

for case in case_rows:

    year = case[
        "Year"
    ]

    event_date = pd.Timestamp(
        case[
            "Date"
        ]
    )

    dominant_asset = case[
        "DominantAsset"
    ]

    regime = case[
        "Regime"
    ]


    # ========================================================
    # EVENT WINDOW
    #
    # 1 year before
    # 6 months after
    # ========================================================

    window_start = (
        event_date
        - pd.Timedelta(
            days=365
        )
    )

    window_end = (
        event_date
        + pd.Timedelta(
            days=180
        )
    )

    market_window = df.loc[
        window_start:
        window_end
    ].copy()


    # ========================================================
    # LOAD DOMINANT ASSET BETAS
    # ========================================================

    beta = pd.read_csv(
        f"data/processed/"
        f"rolling_betas/"
        f"{dominant_asset}.csv",
        index_col="Date",
        parse_dates=True
    )

    beta = beta.sort_index()


    # Detect beta column naming
    beta_columns = {}

    for factor in FACTORS:

        if (
            f"beta_{factor}"
            in beta.columns
        ):

            beta_columns[
                factor
            ] = (
                f"beta_{factor}"
            )

        elif factor in beta.columns:

            beta_columns[
                factor
            ] = factor


    if len(beta_columns) == 0:

        raise ValueError(
            f"No beta columns found "
            f"for {dominant_asset}"
        )


    # ========================================================
    # STANDARDISE BETAS USING 2015-2021 ONLY
    #
    # This gives:
    # 0  = normal relative to reference period
    # +2 = two standard deviations above normal
    # -2 = two standard deviations below normal
    # ========================================================

    beta_reference = beta.loc[
        REFERENCE_START:
        REFERENCE_END,
        list(
            beta_columns.values()
        )
    ].copy()


    beta_mean = (
        beta_reference.mean()
    )

    beta_std = (
        beta_reference.std()
        .replace(
            0,
            np.nan
        )
    )


    beta_z = (
        beta[
            list(
                beta_columns.values()
            )
        ]
        - beta_mean
    ) / beta_std


    # Rename columns back to clean factor names

    reverse_map = {
        column: factor
        for factor, column
        in beta_columns.items()
    }

    beta_z = beta_z.rename(
        columns=reverse_map
    )


    beta_window = beta_z.loc[
        window_start:
        window_end
    ].copy()


    # ========================================================
    # ASSET CONTRIBUTIONS AT EVENT DATE
    # ========================================================

    contribution_values = []

    for asset in ASSETS:

        contribution_values.append(
            df.loc[
                event_date,
                f"SHARE_{asset}"
            ]
            * 100
        )


    contributions = pd.Series(
        contribution_values,
        index=ASSETS
    )


    # ========================================================
    # CREATE FIGURE
    # ========================================================

    fig = plt.figure(
        figsize=(15, 10)
    )


    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[
            1,
            1.15
        ],
        hspace=0.32,
        wspace=0.25
    )


    ax1 = fig.add_subplot(
        grid[
            0,
            0
        ]
    )

    ax2 = fig.add_subplot(
        grid[
            0,
            1
        ]
    )

    ax3 = fig.add_subplot(
        grid[
            1,
            :
        ]
    )


    # ========================================================
    # TITLE
    # ========================================================

    fig.suptitle(
        f"Market DNA Case Study — {year}",
        fontsize=20,
        fontweight="bold",
        y=0.98
    )


    # ========================================================
    # SUMMARY LINE
    # ========================================================

    summary_text = (
        f"{event_date.date()}  |  "
        f"{regime}  |  "
        f"DMIDI {case['DMIDI_Percentile']:.1f}th pct.  |  "
        f"Breadth {case['Breadth_Percentile']:.1f}th pct.  |  "
        f"Dominant asset: {dominant_asset} "
        f"({case['DominantShare'] * 100:.1f}%)"
    )


    fig.text(
        0.5,
        0.935,
        summary_text,
        ha="center",
        fontsize=11
    )


    # ========================================================
    # PANEL 1
    # DMIDI + BREADTH PERCENTILES
    # ========================================================

    ax1.plot(
        market_window.index,
        market_window[
            "DMIDI_PERCENTILE"
        ],
        label="DMIDI percentile",
        linewidth=2
    )

    ax1.plot(
        market_window.index,
        market_window[
            "BREADTH_PERCENTILE"
        ],
        label="Breadth percentile",
        linewidth=2
    )


    ax1.axvline(
        event_date,
        linestyle="--",
        linewidth=1.5,
        label="Selected episode"
    )


    ax1.axhline(
        50,
        linestyle=":",
        linewidth=1
    )


    ax1.set_ylim(
        0,
        100
    )

    ax1.set_ylabel(
        "Reference-period percentile"
    )

    ax1.set_title(
        "Structural Drift Around the Episode",
        fontweight="bold"
    )

    ax1.legend(
        fontsize=8
    )


    # ========================================================
    # PANEL 2
    # CONTRIBUTIONS
    # ========================================================

    bar_colors = []

    for asset in ASSETS:

        if asset == dominant_asset:

            bar_colors.append(
                REGIME_COLORS[
                    regime
                ]
            )

        else:

            bar_colors.append(
                "lightgray"
            )


    bars = ax2.bar(
        contributions.index,
        contributions.values,
        color=bar_colors,
        edgecolor="black",
        linewidth=0.6
    )


    ax2.set_ylabel(
        "Contribution to total drift (%)"
    )

    ax2.set_title(
        "Asset Contribution to Structural Drift",
        fontweight="bold"
    )


    for bar, value in zip(
        bars,
        contributions.values
    ):

        ax2.text(
            bar.get_x()
            + bar.get_width() / 2,

            bar.get_height()
            + 0.8,

            f"{value:.1f}%",

            ha="center",
            va="bottom",
            fontsize=9
        )


    ax2.set_ylim(
        0,
        max(
            contributions.max()
            * 1.25,
            30
        )
    )


    # ========================================================
    # PANEL 3
    # DOMINANT ASSET FINANCIAL DNA
    # ========================================================

    for factor in FACTORS:

        if factor not in beta_window.columns:
            continue

        ax3.plot(
            beta_window.index,
            beta_window[
                factor
            ],
            linewidth=1.8,
            label=factor
        )


    ax3.axvline(
        event_date,
        linestyle="--",
        linewidth=1.5,
        label="Selected episode"
    )


    ax3.axhline(
        0,
        linestyle=":",
        linewidth=1
    )


    ax3.set_ylabel(
        "Rolling beta deviation "
        "(standard deviations)"
    )

    ax3.set_title(
        f"{dominant_asset} Financial DNA — "
        f"Factor Exposure Changes",
        fontweight="bold"
    )


    ax3.legend(
        ncol=4,
        fontsize=8
    )


    # ========================================================
    # FOOTNOTE
    # ========================================================

    fig.text(
        0.01,
        0.012,
        (
            "Notes: DMIDI measures the intensity of structural "
            "change in cross-asset factor exposures. "
            "Breadth measures how widely that drift is distributed "
            "across assets. Percentiles use the 2015–2021 reference "
            "distribution. Rolling betas are standardised relative "
            "to the same reference period."
        ),
        fontsize=8,
        alpha=0.75
    )


    # ========================================================
    # SAVE
    # ========================================================

    plt.tight_layout(
        rect=[
            0,
            0.04,
            1,
            0.91
        ]
    )


    output_file = (
        f"output/figures/"
        f"Case_{year}_FINAL.png"
    )


    plt.savefig(
        output_file,
        dpi=250,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# PRINT SUMMARY
# ============================================================

print(
    "\n======================================"
)

print(
    "MARKET DNA — CASE STUDIES COMPLETE"
)

print(
    "======================================\n"
)


print(
    case_table[
        [
            "Year",
            "Date",
            "Regime",
            "DMIDI_Percentile",
            "Breadth_Percentile",
            "DominantAsset",
            "DominantShare"
        ]
    ]
)


print(
    "\nSaved figures:"
)


for year in CASE_YEARS:

    print(
        f"output/figures/"
        f"Case_{year}_FINAL.png"
    )


print(
    "\nSaved table:"
)

print(
    "output/tables/"
    "case_study_summary.csv"
)