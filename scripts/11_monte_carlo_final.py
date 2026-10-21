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

WEIGHTS = np.array([
    1/6,
    1/6,
    1/6,
    1/6,
    1/6,
    1/6
])

N_SIMULATIONS = 10000
HORIZON = 20
BLOCK_SIZE = 5
CURRENT_NEIGHBORS = 250
RANDOM_SEED = 42


SCENARIO_COLORS = {
    "Stable / Local": "tab:blue",
    "Current-like": "tab:orange",
    "Systemic Shift": "tab:red"
}


# ============================================================
# OUTPUT
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
    "data/processed/"
    "market_dna_percentile_regimes.csv",
    index_col="Date",
    parse_dates=True
)

df = df.sort_index()


# ============================================================
# CURRENT MARKET DNA
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

current_dmidi = latest[
    "DMIDI_PERCENTILE"
]

current_breadth = latest[
    "BREADTH_PERCENTILE"
]


# ============================================================
# CURRENT-LIKE ANALOGUES
# ============================================================

analogue_data = (
    df.dropna(
        subset=[
            "DMIDI_PERCENTILE",
            "BREADTH_PERCENTILE"
        ]
    )
    .copy()
)


analogue_data[
    "DNA_DISTANCE"
] = np.sqrt(

    (
        analogue_data[
            "DMIDI_PERCENTILE"
        ]
        - current_dmidi
    ) ** 2

    +

    (
        analogue_data[
            "BREADTH_PERCENTILE"
        ]
        - current_breadth
    ) ** 2
)


analogue_data = (
    analogue_data.iloc[
        :-BLOCK_SIZE
    ]
)


current_analogues = (
    analogue_data
    .nsmallest(
        CURRENT_NEIGHBORS,
        "DNA_DISTANCE"
    )
    .copy()
)


current_distance_map = (
    current_analogues[
        "DNA_DISTANCE"
    ]
    .to_dict()
)


# ============================================================
# SCENARIO DATES
# ============================================================

stable_dates = set(

    df[
        df[
            "REGIME_PERCENTILE"
        ] == "Stable / Local"
    ].index
)


systemic_dates = set(

    df[
        df[
            "REGIME_PERCENTILE"
        ]
        == "Systemic Structural Shift"
    ].index
)


current_like_dates = set(
    current_analogues.index
)


# ============================================================
# POSITION LOOKUP
# ============================================================

dates = df.index

date_to_position = {
    date: i
    for i, date
    in enumerate(dates)
}


# ============================================================
# BUILD FORWARD 5-DAY BLOCKS
# ============================================================

def build_records(
    scenario_dates,
    distance_map=None
):

    records = []

    asset_returns = df[
        ASSETS
    ]


    for signal_date in sorted(
        scenario_dates
    ):

        if signal_date not in date_to_position:
            continue

        i = date_to_position[
            signal_date
        ]


        if (
            i + BLOCK_SIZE
            >= len(df)
        ):
            continue


        block = (
            asset_returns.iloc[
                i + 1:
                i + 1 + BLOCK_SIZE
            ]
        )


        if (
            len(block) != BLOCK_SIZE
            or block.isna().any().any()
        ):
            continue


        distance = np.nan

        if distance_map is not None:

            distance = distance_map.get(
                signal_date,
                np.nan
            )


        records.append(
            {
                "signal_date":
                    signal_date,

                "start_pos":
                    i + 1,

                "end_pos":
                    i + BLOCK_SIZE,

                "distance":
                    distance,

                "block":
                    block.to_numpy()
            }
        )


    return records


stable_records = build_records(
    stable_dates
)

current_records = build_records(
    current_like_dates,
    current_distance_map
)

systemic_records = build_records(
    systemic_dates
)


# ============================================================
# NON-OVERLAPPING SELECTION
# ============================================================

def intervals_overlap(
    first,
    second
):

    return not (

        first["end_pos"]
        < second["start_pos"]

        or

        first["start_pos"]
        > second["end_pos"]
    )


def chronological_nonoverlap(
    records
):

    ordered = sorted(
        records,
        key=lambda x:
            x["start_pos"]
    )

    selected = []

    last_end = -1


    for record in ordered:

        if record["start_pos"] > last_end:

            selected.append(
                record
            )

            last_end = (
                record["end_pos"]
            )


    return selected


def current_nonoverlap(
    records
):

    ordered = sorted(
        records,
        key=lambda x:
            x["distance"]
    )

    selected = []


    for record in ordered:

        overlap = any(

            intervals_overlap(
                record,
                existing
            )

            for existing
            in selected
        )


        if not overlap:

            selected.append(
                record
            )


    return selected


scenario_records = {

    "Stable / Local":
        chronological_nonoverlap(
            stable_records
        ),

    "Current-like":
        current_nonoverlap(
            current_records
        ),

    "Systemic Shift":
        chronological_nonoverlap(
            systemic_records
        )
}


# ============================================================
# MONTE CARLO
# ============================================================

def simulate(
    records,
    seed
):

    rng = np.random.default_rng(
        seed
    )

    blocks = [
        record["block"]
        for record
        in records
    ]


    blocks_needed = int(
        np.ceil(
            HORIZON
            / BLOCK_SIZE
        )
    )


    cumulative_returns = np.empty(
        N_SIMULATIONS
    )

    maximum_drawdowns = np.empty(
        N_SIMULATIONS
    )

    wealth_paths = np.empty(
        (
            N_SIMULATIONS,
            HORIZON + 1
        )
    )


    for simulation in range(
        N_SIMULATIONS
    ):

        selections = rng.integers(
            0,
            len(blocks),
            size=blocks_needed
        )


        log_returns = np.vstack(
            [
                blocks[index]
                for index
                in selections
            ]
        )[
            :HORIZON
        ]


        simple_returns = np.expm1(
            log_returns
        )


        portfolio_returns = (
            simple_returns
            @ WEIGHTS
        )


        wealth = np.concatenate(
            [
                [1.0],

                np.cumprod(
                    1
                    + portfolio_returns
                )
            ]
        )


        wealth_paths[
            simulation
        ] = wealth


        cumulative_returns[
            simulation
        ] = (
            wealth[-1]
            - 1
        )


        running_peak = (
            np.maximum.accumulate(
                wealth
            )
        )


        drawdowns = (
            wealth
            / running_peak
            - 1
        )


        maximum_drawdowns[
            simulation
        ] = (
            drawdowns.min()
        )


    return {
        "returns":
            cumulative_returns,

        "drawdowns":
            maximum_drawdowns,

        "paths":
            wealth_paths
    }


results = {}


for number, (
    scenario,
    records
) in enumerate(
    scenario_records.items()
):

    results[
        scenario
    ] = simulate(
        records,
        RANDOM_SEED
        + number
    )


# ============================================================
# RISK METRICS
# ============================================================

def metrics(
    returns,
    drawdowns
):

    q05 = np.quantile(
        returns,
        0.05
    )

    q01 = np.quantile(
        returns,
        0.01
    )


    return {

        "Mean_Return":
            returns.mean(),

        "Median_Return":
            np.median(
                returns
            ),

        "Volatility":
            returns.std(
                ddof=1
            ),

        "VaR_95":
            -q05,

        "VaR_99":
            -q01,

        "ES_95":
            -returns[
                returns <= q05
            ].mean(),

        "ES_99":
            -returns[
                returns <= q01
            ].mean(),

        "P_Loss_5pct":
            np.mean(
                returns <= -0.05
            ),

        "P_Loss_10pct":
            np.mean(
                returns <= -0.10
            ),

        "Mean_Max_Drawdown":
            drawdowns.mean()
    }


summary_rows = []


for scenario, result in results.items():

    row = metrics(
        result["returns"],
        result["drawdowns"]
    )

    row[
        "Scenario"
    ] = scenario

    row[
        "Historical_Blocks"
    ] = len(
        scenario_records[
            scenario
        ]
    )

    summary_rows.append(
        row
    )


summary = pd.DataFrame(
    summary_rows
)


summary = summary[
    [
        "Scenario",
        "Historical_Blocks",
        "Mean_Return",
        "Median_Return",
        "Volatility",
        "VaR_95",
        "VaR_99",
        "ES_95",
        "ES_99",
        "P_Loss_5pct",
        "P_Loss_10pct",
        "Mean_Max_Drawdown"
    ]
]


summary.to_csv(
    "output/tables/"
    "Monte_Carlo_FINAL_Risk_Summary.csv",
    index=False
)


# ============================================================
# FIGURE 1
# FINAL RETURN DISTRIBUTIONS
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 7)
)


for scenario, result in results.items():

    returns_pct = (
        result["returns"]
        * 100
    )


    ax.hist(
        returns_pct,
        bins=70,
        density=True,
        histtype="step",
        linewidth=2,
        color=SCENARIO_COLORS[
            scenario
        ],
        label=scenario
    )


ax.axvline(
    0,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Simulated 20-day portfolio return (%)"
)

ax.set_ylabel(
    "Density"
)

ax.set_title(
    "Market DNA Monte Carlo\n"
    "Non-Overlapping 20-Day Portfolio Return Scenarios",
    fontsize=15,
    fontweight="bold"
)


ax.legend()


plt.tight_layout()


plt.savefig(
    "output/figures/"
    "Monte_Carlo_FINAL_Return_Distributions.png",
    dpi=250
)

plt.close()


# ============================================================
# FIGURE 2
# FINAL TAIL RISK
# ============================================================

risk_plot = (
    summary
    .set_index(
        "Scenario"
    )
    [
        [
            "VaR_95",
            "ES_95",
            "VaR_99",
            "ES_99"
        ]
    ]
    * 100
)


fig, ax = plt.subplots(
    figsize=(11, 7)
)


risk_plot.plot(
    kind="bar",
    ax=ax
)


ax.set_ylabel(
    "Portfolio loss (%)"
)

ax.set_xlabel(
    ""
)

ax.set_title(
    "Market DNA Monte Carlo\n"
    "Tail Risk from Non-Overlapping Historical Blocks",
    fontsize=15,
    fontweight="bold"
)


ax.tick_params(
    axis="x",
    rotation=0
)


for container in ax.containers:

    ax.bar_label(
        container,
        fmt="%.1f",
        fontsize=8,
        padding=2
    )


plt.tight_layout()


plt.savefig(
    "output/figures/"
    "Monte_Carlo_FINAL_Tail_Risk.png",
    dpi=250
)

plt.close()


# ============================================================
# FIGURE 3
# FINAL PORTFOLIO PATHS
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 7)
)


days = np.arange(
    HORIZON + 1
)


for scenario, result in results.items():

    paths = result[
        "paths"
    ]


    p05 = np.quantile(
        paths,
        0.05,
        axis=0
    )

    p50 = np.quantile(
        paths,
        0.50,
        axis=0
    )

    p95 = np.quantile(
        paths,
        0.95,
        axis=0
    )


    color = (
        SCENARIO_COLORS[
            scenario
        ]
    )


    ax.plot(
        days,
        p50 * 100,
        linewidth=2,
        color=color,
        label=scenario
    )


    ax.fill_between(
        days,
        p05 * 100,
        p95 * 100,
        color=color,
        alpha=0.10
    )


ax.axhline(
    100,
    linestyle="--",
    linewidth=1
)


ax.set_xlabel(
    "Trading day"
)

ax.set_ylabel(
    "Portfolio value "
    "(initial value = 100)"
)

ax.set_title(
    "Market DNA Monte Carlo\n"
    "5th–95th Percentile Portfolio Paths",
    fontsize=15,
    fontweight="bold"
)


ax.legend()


plt.tight_layout()


plt.savefig(
    "output/figures/"
    "Monte_Carlo_FINAL_Portfolio_Paths.png",
    dpi=250
)

plt.close()


# ============================================================
# FIGURE 4
# COMPACT RISK DASHBOARD
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 6)
)


plot_data = (
    summary
    .set_index(
        "Scenario"
    )
    [
        [
            "P_Loss_5pct",
            "P_Loss_10pct"
        ]
    ]
    * 100
)


plot_data.columns = [
    "P(Return < -5%)",
    "P(Return < -10%)"
]


plot_data.plot(
    kind="bar",
    ax=ax
)


ax.set_ylabel(
    "Probability (%)"
)

ax.set_xlabel(
    ""
)

ax.set_title(
    "Market DNA Monte Carlo\n"
    "Probability of Large 20-Day Portfolio Losses",
    fontsize=15,
    fontweight="bold"
)


ax.tick_params(
    axis="x",
    rotation=0
)


for container in ax.containers:

    ax.bar_label(
        container,
        fmt="%.1f%%",
        fontsize=9,
        padding=2
    )


plt.tight_layout()


plt.savefig(
    "output/figures/"
    "Monte_Carlo_FINAL_Loss_Probabilities.png",
    dpi=250
)

plt.close()


# ============================================================
# PRINT SUMMARY
# ============================================================

print(
    "\n======================================"
)

print(
    "MARKET DNA — FINAL MONTE CARLO"
)

print(
    "======================================\n"
)


print(
    "Non-overlapping historical blocks:"
)


for scenario, records in scenario_records.items():

    print(
        scenario,
        ":",
        len(records)
    )


display = summary.copy()


percentage_columns = [
    "Mean_Return",
    "Median_Return",
    "Volatility",
    "VaR_95",
    "VaR_99",
    "ES_95",
    "ES_99",
    "P_Loss_5pct",
    "P_Loss_10pct",
    "Mean_Max_Drawdown"
]


for column in percentage_columns:

    display[
        column
    ] *= 100


print(
    "\nFinal risk summary:\n"
)


print(
    display.round(
        2
    ).to_string(
        index=False
    )
)


print(
    "\nSaved final figures:"
)


print(
    "output/figures/"
    "Monte_Carlo_FINAL_Return_Distributions.png"
)

print(
    "output/figures/"
    "Monte_Carlo_FINAL_Tail_Risk.png"
)

print(
    "output/figures/"
    "Monte_Carlo_FINAL_Portfolio_Paths.png"
)

print(
    "output/figures/"
    "Monte_Carlo_FINAL_Loss_Probabilities.png"
)