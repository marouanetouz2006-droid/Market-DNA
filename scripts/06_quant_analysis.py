import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from scipy.stats import pearsonr, spearmanr


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

DIDS_COLS = [
    f"DIDS_{asset}"
    for asset in ASSETS
]

HORIZONS = [
    5,
    10,
    20,
    40,
    60
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

Path("data/processed").mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(
    "data/processed/research_dataset_v2.csv",
    index_col="Date",
    parse_dates=True
)

df = df.sort_index()


required_columns = (
    ASSETS
    + DIDS_COLS
    + [
        "MKT",
        "MIDI",
        "DMIDI",
        "VIX_LEVEL"
    ]
)

missing = [
    col
    for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# 1. DRIFT CONTRIBUTIONS
# ============================================================

dids = df[DIDS_COLS].copy()

total_drift = dids.sum(
    axis=1,
    min_count=len(DIDS_COLS)
)

shares = dids.div(
    total_drift,
    axis=0
)

shares.columns = [
    f"SHARE_{asset}"
    for asset in ASSETS
]


# ============================================================
# 2. DRIFT BREADTH — ENTROPY
# ============================================================

def entropy_breadth(row):

    values = row.dropna().to_numpy()

    if len(values) != len(ASSETS):
        return np.nan

    values = values[
        values > 0
    ]

    if len(values) == 0:
        return np.nan

    entropy = -np.sum(
        values * np.log(values)
    )

    max_entropy = np.log(
        len(ASSETS)
    )

    return entropy / max_entropy


df["DRIFT_BREADTH"] = shares.apply(
    entropy_breadth,
    axis=1
)


# ============================================================
# 3. SYSTEMIC DRIFT SCORE
# ============================================================

df["SYSTEMIC_DRIFT"] = (
    df["DMIDI"]
    * df["DRIFT_BREADTH"]
)


# ============================================================
# 4. DOMINANT CONTRIBUTOR
# ============================================================

# ============================================================
# 4. DOMINANT CONTRIBUTOR
# ============================================================

df["DOMINANT_ASSET"] = pd.Series(
    index=df.index,
    dtype="object"
)

df["DOMINANT_SHARE"] = np.nan

valid_share_rows = shares.notna().any(axis=1)

df.loc[
    valid_share_rows,
    "DOMINANT_ASSET"
] = (
    shares.loc[
        valid_share_rows
    ]
    .idxmax(axis=1)
    .str.replace(
        "SHARE_",
        "",
        regex=False
    )
)

df.loc[
    valid_share_rows,
    "DOMINANT_SHARE"
] = (
    shares.loc[
        valid_share_rows
    ]
    .max(axis=1)
)


# Add contribution shares
df = pd.concat(
    [
        df,
        shares
    ],
    axis=1
)


# ============================================================
# 5. FORWARD MARKET METRICS
# ============================================================

market_returns = df[
    "MKT"
].to_numpy()

asset_returns = df[
    ASSETS
]


for horizon in HORIZONS:

    future_rv = np.full(
        len(df),
        np.nan
    )

    future_return = np.full(
        len(df),
        np.nan
    )

    future_corr = np.full(
        len(df),
        np.nan
    )

    future_crash = np.full(
        len(df),
        np.nan
    )


    for i in range(
        len(df) - horizon
    ):

        # ----------------------------------------
        # Future market window
        # ----------------------------------------

        market_window = market_returns[
            i + 1:
            i + 1 + horizon
        ]

        if not np.isnan(
            market_window
        ).any():

            # Annualised realised volatility
            future_rv[i] = np.sqrt(
                252
                * np.mean(
                    market_window ** 2
                )
            )

            # Cumulative market return
            future_return[i] = (
                np.exp(
                    np.sum(
                        market_window
                    )
                )
                - 1
            )

            # Maximum drawdown
            growth = np.exp(
                np.cumsum(
                    market_window
                )
            )

            path = np.concatenate(
                [
                    [1.0],
                    growth
                ]
            )

            running_peak = (
                np.maximum.accumulate(
                    path
                )
            )

            drawdowns = (
                path
                / running_peak
                - 1
            )

            maximum_drawdown = (
                drawdowns.min()
            )

            future_crash[i] = (
                1
                if maximum_drawdown <= -0.05
                else 0
            )


        # ----------------------------------------
        # Future cross-asset correlation
        # ----------------------------------------

        asset_window = (
            asset_returns.iloc[
                i + 1:
                i + 1 + horizon
            ]
        )

        if (
            len(asset_window) == horizon
            and not asset_window
                .isna()
                .any()
                .any()
        ):

            corr_matrix = (
                asset_window.corr()
            )

            corr_values = (
                corr_matrix
                .to_numpy()
            )

            upper_triangle = (
                np.triu_indices(
                    len(ASSETS),
                    k=1
                )
            )

            pairwise = (
                corr_values[
                    upper_triangle
                ]
            )

            future_corr[i] = (
                pairwise.mean()
            )


    df[
        f"future_RV_{horizon}"
    ] = future_rv

    df[
        f"future_return_{horizon}"
    ] = future_return

    df[
        f"future_avg_corr_{horizon}"
    ] = future_corr

    df[
        f"crash_{horizon}d_5pct"
    ] = future_crash


# ============================================================
# 6. DIDS CORRELATION MATRIX
# ============================================================

dids_corr = (
    df[DIDS_COLS]
    .corr()
)

dids_corr.to_csv(
    "output/tables/"
    "DIDS_correlation_matrix.csv"
)


# ============================================================
# 7. PCA OF STRUCTURAL DRIFT
# ============================================================

pca_data = (
    df[DIDS_COLS]
    .dropna()
    .copy()
)

means = pca_data.mean()

stds = pca_data.std(
    ddof=1
)

standardised = (
    (pca_data - means)
    / stds
)

covariance = np.cov(
    standardised,
    rowvar=False
)

eigenvalues, eigenvectors = (
    np.linalg.eigh(
        covariance
    )
)

order = np.argsort(
    eigenvalues
)[::-1]

eigenvalues = (
    eigenvalues[order]
)

eigenvectors = (
    eigenvectors[:, order]
)

explained_variance = (
    eigenvalues
    / eigenvalues.sum()
)


# Flip PC1 orientation if needed
# so average loading is positive.

if (
    eigenvectors[:, 0].mean()
    < 0
):

    eigenvectors[:, 0] *= -1


scores = (
    standardised.to_numpy()
    @ eigenvectors
)

pc1 = pd.Series(
    scores[:, 0],
    index=pca_data.index,
    name="PC1_STRUCTURAL_DRIFT"
)

df[
    "PC1_STRUCTURAL_DRIFT"
] = pc1


loadings = (
    eigenvectors
    * np.sqrt(
        eigenvalues
    )
)

loadings_df = pd.DataFrame(
    loadings,
    index=ASSETS,
    columns=[
        f"PC{i+1}"
        for i in range(
            len(ASSETS)
        )
    ]
)

loadings_df.to_csv(
    "output/tables/"
    "PCA_loadings.csv"
)


pca_variance_df = pd.DataFrame(
    {
        "Component": [
            f"PC{i+1}"
            for i in range(
                len(ASSETS)
            )
        ],
        "ExplainedVariance": (
            explained_variance
        ),
        "CumulativeVariance": (
            np.cumsum(
                explained_variance
            )
        )
    }
)

pca_variance_df.to_csv(
    "output/tables/"
    "PCA_explained_variance.csv",
    index=False
)


# ============================================================
# 8. PEARSON + SPEARMAN LEAD-LAG ANALYSIS
# ============================================================

signals = [
    "MIDI",
    "DMIDI",
    "DRIFT_BREADTH",
    "SYSTEMIC_DRIFT",
    "PC1_STRUCTURAL_DRIFT"
] + DIDS_COLS


correlation_results = []


def safe_correlations(
    x,
    y
):

    temp = pd.concat(
        [
            x,
            y
        ],
        axis=1
    ).dropna()

    if len(temp) < 30:

        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            len(temp)
        )

    x_clean = (
        temp.iloc[:, 0]
    )

    y_clean = (
        temp.iloc[:, 1]
    )

    if (
        x_clean.std() == 0
        or y_clean.std() == 0
    ):

        return (
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            len(temp)
        )

    pearson_r, pearson_p = (
        pearsonr(
            x_clean,
            y_clean
        )
    )

    spearman_r, spearman_p = (
        spearmanr(
            x_clean,
            y_clean
        )
    )

    return (
        pearson_r,
        pearson_p,
        spearman_r,
        spearman_p,
        len(temp)
    )


for signal in signals:

    for horizon in HORIZONS:

        outcomes = {

            "FutureVolatility":
                f"future_RV_{horizon}",

            "FutureCorrelation":
                f"future_avg_corr_{horizon}",

            "FutureReturn":
                f"future_return_{horizon}",

            "CrashRisk":
                f"crash_{horizon}d_5pct"
        }


        for outcome_name, column in outcomes.items():

            (
                pearson_r,
                pearson_p,
                spearman_r,
                spearman_p,
                n
            ) = safe_correlations(
                df[signal],
                df[column]
            )


            correlation_results.append(
                {
                    "Signal":
                        signal,

                    "Outcome":
                        outcome_name,

                    "Horizon":
                        horizon,

                    "N":
                        n,

                    "Pearson":
                        pearson_r,

                    "Pearson_p":
                        pearson_p,

                    "Spearman":
                        spearman_r,

                    "Spearman_p":
                        spearman_p
                }
            )


correlation_table = pd.DataFrame(
    correlation_results
)

correlation_table.to_csv(
    "output/tables/"
    "lead_lag_correlations.csv",
    index=False
)


# ============================================================
# 9. DMIDI DECILE ANALYSIS
# ============================================================

decile_data = df[
    [
        "DMIDI",
        "future_RV_20",
        "future_avg_corr_20",
        "future_return_20",
        "crash_20d_5pct"
    ]
].copy()


valid_dmidi = (
    decile_data["DMIDI"]
    .notna()
)


decile_data.loc[
    valid_dmidi,
    "DMIDI_DECILE"
] = (
    pd.qcut(
        decile_data.loc[
            valid_dmidi,
            "DMIDI"
        ],
        q=10,
        labels=False,
        duplicates="drop"
    )
    + 1
)


decile_stats = (
    decile_data
    .dropna(
        subset=[
            "DMIDI_DECILE"
        ]
    )
    .groupby(
        "DMIDI_DECILE"
    )
    .agg(
        observations=(
            "DMIDI",
            "size"
        ),

        mean_DMIDI=(
            "DMIDI",
            "mean"
        ),

        future_volatility_20=(
            "future_RV_20",
            "mean"
        ),

        future_correlation_20=(
            "future_avg_corr_20",
            "mean"
        ),

        future_return_20=(
            "future_return_20",
            "mean"
        ),

        crash_probability_20=(
            "crash_20d_5pct",
            "mean"
        )
    )
    .reset_index()
)


decile_stats.to_csv(
    "output/tables/"
    "DMIDI_decile_analysis.csv",
    index=False
)


# ============================================================
# 10. SAVE ENRICHED DATASET
# ============================================================

df.to_csv(
    "data/processed/"
    "quant_research_dataset.csv"
)


shares.to_csv(
    "output/tables/"
    "drift_contribution_shares.csv"
)


# ============================================================
# 11. GRAPH — DRIFT BREADTH
# ============================================================

ax = df[
    "DRIFT_BREADTH"
].plot(
    figsize=(12, 5),
    title=(
        "Market DNA — "
        "Structural Drift Breadth"
    )
)

ax.set_ylabel(
    "Entropy breadth [0–1]"
)

ax.set_ylim(
    0,
    1.05
)

plt.tight_layout()

plt.savefig(
    "output/figures/"
    "Drift_Breadth.png",
    dpi=180
)

plt.close()


# ============================================================
# 12. GRAPH — CONTRIBUTIONS
# ============================================================

contribution_plot = (
    shares
    * 100
)

ax = contribution_plot.plot.area(
    figsize=(12, 6),
    title=(
        "Market DNA — "
        "Asset Contributions to Dynamic Drift"
    )
)

ax.set_ylabel(
    "Contribution (%)"
)

ax.set_ylim(
    0,
    100
)

ax.legend(
    loc="upper left",
    ncol=3,
    fontsize=8
)

plt.tight_layout()

plt.savefig(
    "output/figures/"
    "Drift_Contributions.png",
    dpi=180
)

plt.close()


# ============================================================
# 13. GRAPH — DIDS CORRELATION HEATMAP
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 7)
)

image = ax.imshow(
    dids_corr.to_numpy(),
    vmin=-1,
    vmax=1
)

ax.set_xticks(
    range(
        len(ASSETS)
    )
)

ax.set_yticks(
    range(
        len(ASSETS)
    )
)

ax.set_xticklabels(
    ASSETS,
    rotation=45,
    ha="right"
)

ax.set_yticklabels(
    ASSETS
)

for i in range(
    len(ASSETS)
):

    for j in range(
        len(ASSETS)
    ):

        ax.text(
            j,
            i,
            f"{dids_corr.iloc[i, j]:.2f}",
            ha="center",
            va="center",
            fontsize=8
        )


fig.colorbar(
    image,
    ax=ax,
    label="Correlation"
)

ax.set_title(
    "Correlation of Dynamic Identity Drift Scores"
)

plt.tight_layout()

plt.savefig(
    "output/figures/"
    "DIDS_Correlation_Heatmap.png",
    dpi=180
)

plt.close()


# ============================================================
# 14. GRAPH — PCA EXPLAINED VARIANCE
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    pca_variance_df[
        "Component"
    ],
    pca_variance_df[
        "ExplainedVariance"
    ] * 100
)

plt.ylabel(
    "Explained variance (%)"
)

plt.title(
    "PCA of Market Identity Drift"
)

plt.tight_layout()

plt.savefig(
    "output/figures/"
    "PCA_Explained_Variance.png",
    dpi=180
)

plt.close()


# ============================================================
# 15. GRAPH — PC1
# ============================================================

ax = df[
    "PC1_STRUCTURAL_DRIFT"
].plot(
    figsize=(12, 5),
    title=(
        "Market DNA — "
        "Common Structural Drift Factor (PC1)"
    )
)

ax.axhline(
    0,
    linestyle="--"
)

plt.tight_layout()

plt.savefig(
    "output/figures/"
    "PC1_Structural_Drift.png",
    dpi=180
)

plt.close()


# ============================================================
# 16. GRAPH — DMIDI FORECAST HORIZON PROFILE
# ============================================================

dmidi_profile = (
    correlation_table[
        (
            correlation_table[
                "Signal"
            ] == "DMIDI"
        )
        &
        (
            correlation_table[
                "Outcome"
            ].isin(
                [
                    "FutureVolatility",
                    "FutureCorrelation"
                ]
            )
        )
    ]
)


plt.figure(
    figsize=(9, 5)
)


for outcome in [
    "FutureVolatility",
    "FutureCorrelation"
]:

    subset = (
        dmidi_profile[
            dmidi_profile[
                "Outcome"
            ] == outcome
        ]
    )

    plt.plot(
        subset[
            "Horizon"
        ],
        subset[
            "Spearman"
        ],
        marker="o",
        label=outcome
    )


plt.axhline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Forecast horizon (trading days)"
)

plt.ylabel(
    "Spearman correlation"
)

plt.title(
    "DMIDI — Lead-Lag Correlation Profile"
)

plt.legend()

plt.tight_layout()

plt.savefig(
    "output/figures/"
    "DMIDI_Lead_Lag_Profile.png",
    dpi=180
)

plt.close()


# ============================================================
# 17. GRAPH — DMIDI DECILES
# ============================================================

plt.figure(
    figsize=(9, 5)
)

plt.bar(
    decile_stats[
        "DMIDI_DECILE"
    ],
    decile_stats[
        "future_volatility_20"
    ]
)

plt.xlabel(
    "DMIDI decile"
)

plt.ylabel(
    "Average future 20-day volatility"
)

plt.title(
    "Future Volatility by DMIDI Decile"
)

plt.tight_layout()

plt.savefig(
    "output/figures/"
    "DMIDI_Decile_Future_Volatility.png",
    dpi=180
)

plt.close()


# ============================================================
# 18. CURRENT MARKET STATE
# ============================================================

latest = (
    df.dropna(
        subset=[
            "DMIDI",
            "DRIFT_BREADTH"
        ]
    )
    .iloc[-1]
)


current_state = pd.DataFrame(
    {
        "Metric": [
            "Date",
            "DMIDI",
            "Drift Breadth",
            "Systemic Drift",
            "Dominant Asset",
            "Dominant Share"
        ],

        "Value": [
            latest.name,
            latest["DMIDI"],
            latest["DRIFT_BREADTH"],
            latest["SYSTEMIC_DRIFT"],
            latest["DOMINANT_ASSET"],
            latest["DOMINANT_SHARE"]
        ]
    }
)


current_state.to_csv(
    "output/tables/"
    "current_market_state.csv",
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print(
    "\n======================================"
)

print(
    "MARKET DNA — QUANT ANALYSIS COMPLETE"
)

print(
    "======================================\n"
)


print(
    "Current date:",
    latest.name.date()
)

print(
    "Current DMIDI:",
    round(
        latest["DMIDI"],
        3
    )
)

print(
    "Current Drift Breadth:",
    round(
        latest["DRIFT_BREADTH"],
        3
    )
)

print(
    "Current Systemic Drift:",
    round(
        latest["SYSTEMIC_DRIFT"],
        3
    )
)

print(
    "Dominant contributor:",
    latest["DOMINANT_ASSET"]
)

print(
    "Dominant share:",
    f"{latest['DOMINANT_SHARE'] * 100:.1f}%"
)


print(
    "\nPC1 explained variance:",
    f"{explained_variance[0] * 100:.2f}%"
)


print(
    "\nPC1 loadings:"
)

print(
    loadings_df[
        "PC1"
    ]
    .round(3)
)


print(
    "\nSaved quantitative tables "
    "to output/tables/"
)

print(
    "Saved quantitative figures "
    "to output/figures/"
)

print(
    "Saved enriched dataset:"
)

print(
    "data/processed/"
    "quant_research_dataset.csv"
)