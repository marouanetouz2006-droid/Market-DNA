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

# Equal-weight portfolio
WEIGHTS = np.array([
    1/6,
    1/6,
    1/6,
    1/6,
    1/6,
    1/6
])

N_SIMULATIONS = 10000

HORIZON = 20       # trading days
BLOCK_SIZE = 5     # preserves short-run dependence

CURRENT_NEIGHBORS = 250

RANDOM_SEED = 42


# ============================================================
# OUTPUT DIRECTORIES
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


required = (
    ASSETS
    + [
        "DMIDI_PERCENTILE",
        "BREADTH_PERCENTILE",
        "REGIME_PERCENTILE"
    ]
)

missing = [
    col
    for col in required
    if col not in df.columns
]

if missing:

    raise ValueError(
        f"Missing columns: {missing}"
    )


# ============================================================
# CURRENT MARKET STATE
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
# FIND HISTORICAL STATES SIMILAR TO CURRENT MARKET DNA
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


# Remove most recent observations where subsequent
# returns are not available.

analogue_data = analogue_data.iloc[
    :-BLOCK_SIZE
]


current_analogues = (
    analogue_data
    .nsmallest(
        CURRENT_NEIGHBORS,
        "DNA_DISTANCE"
    )
)


current_analogues[
    [
        "DMIDI_PERCENTILE",
        "BREADTH_PERCENTILE",
        "REGIME_PERCENTILE",
        "DNA_DISTANCE"
    ]
].to_csv(
    "output/tables/"
    "current_market_analogues.csv"
)


# ============================================================
# SCENARIO SIGNAL DATES
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


SCENARIOS = {

    "Stable / Local":
        stable_dates,

    "Current-like":
        current_like_dates,

    "Systemic Shift":
        systemic_dates
}


# ============================================================
# BUILD FORWARD RETURN BLOCKS
#
# For every Market DNA state at date t,
# take asset returns from t+1 to t+5.
#
# This gives a scenario-conditioned historical block.
# ============================================================

def build_blocks(
    data,
    scenario_dates
):

    blocks = []

    dates = data.index

    returns = data[
        ASSETS
    ]


    for i in range(
        len(data) - BLOCK_SIZE
    ):

        signal_date = dates[i]

        if signal_date not in scenario_dates:
            continue


        block = (
            returns.iloc[
                i + 1:
                i + 1 + BLOCK_SIZE
            ]
        )


        if (
            len(block) == BLOCK_SIZE
            and not block
                .isna()
                .any()
                .any()
        ):

            blocks.append(
                block.to_numpy()
            )


    if len(blocks) == 0:

        raise ValueError(
            "No valid blocks found "
            "for scenario."
        )


    return np.array(
        blocks
    )


scenario_blocks = {}


for scenario, dates in SCENARIOS.items():

    scenario_blocks[
        scenario
    ] = build_blocks(
        df,
        dates
    )


# ============================================================
# MONTE CARLO SIMULATION
# ============================================================

rng = np.random.default_rng(
    RANDOM_SEED
)


def simulate_scenario(
    blocks,
    n_simulations,
    horizon,
    block_size
):

    n_blocks_needed = int(
        np.ceil(
            horizon
            / block_size
        )
    )


    cumulative_returns = np.empty(
        n_simulations
    )

    max_drawdowns = np.empty(
        n_simulations
    )

    wealth_paths = np.empty(
        (
            n_simulations,
            horizon + 1
        )
    )


    for simulation in range(
        n_simulations
    ):

        chosen = rng.integers(
            0,
            len(blocks),
            size=n_blocks_needed
        )


        simulated_log_returns = (
            np.vstack(
                [
                    blocks[index]
                    for index in chosen
                ]
            )
        )


        simulated_log_returns = (
            simulated_log_returns[
                :horizon
            ]
        )


        # Convert asset log returns
        # into simple returns.

        asset_simple_returns = (
            np.expm1(
                simulated_log_returns
            )
        )


        # Daily portfolio return

        portfolio_daily = (
            asset_simple_returns
            @ WEIGHTS
        )


        # Wealth path starting at 1

        wealth = np.concatenate(
            [
                [1.0],

                np.cumprod(
                    1
                    + portfolio_daily
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


        max_drawdowns[
            simulation
        ] = (
            drawdowns.min()
        )


    return {
        "returns":
            cumulative_returns,

        "drawdowns":
            max_drawdowns,

        "paths":
            wealth_paths
    }


simulation_results = {}


for scenario, blocks in scenario_blocks.items():

    simulation_results[
        scenario
    ] = simulate_scenario(
        blocks,
        N_SIMULATIONS,
        HORIZON,
        BLOCK_SIZE
    )


# ============================================================
# RISK METRICS
# ============================================================

def risk_metrics(
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


    var_95 = -q05
    var_99 = -q01


    es_95 = -returns[
        returns <= q05
    ].mean()


    es_99 = -returns[
        returns <= q01
    ].mean()


    probability_loss_5 = (
        np.mean(
            returns <= -0.05
        )
    )


    probability_loss_10 = (
        np.mean(
            returns <= -0.10
        )
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
            var_95,

        "VaR_99":
            var_99,

        "ES_95":
            es_95,

        "ES_99":
            es_99,

        "P_Loss_5pct":
            probability_loss_5,

        "P_Loss_10pct":
            probability_loss_10,

        "Mean_Max_Drawdown":
            drawdowns.mean(),

        "Median_Max_Drawdown":
            np.median(
                drawdowns
            )
    }


summary_rows = []


for scenario, result in simulation_results.items():

    metrics = risk_metrics(
        result["returns"],
        result["drawdowns"]
    )


    metrics[
        "Scenario"
    ] = scenario


    metrics[
        "Historical_Blocks"
    ] = len(
        scenario_blocks[
            scenario
        ]
    )


    summary_rows.append(
        metrics
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
        "Mean_Max_Drawdown",
        "Median_Max_Drawdown"
    ]
]


summary.to_csv(
    "output/tables/"
    "Monte_Carlo_Risk_Summary.csv",
    index=False
)


# ============================================================
# SAVE SIMULATED RETURNS
# ============================================================

simulated_returns_table = pd.DataFrame(
    {
        scenario:
            result[
                "returns"
            ]

        for scenario, result
        in simulation_results.items()
    }
)


simulated_returns_table.to_csv(
    "output/tables/"
    "Monte_Carlo_Simulated_Returns.csv",
    index=False
)


# ============================================================
# FIGURE 1
# SIMULATED 20-DAY RETURN DISTRIBUTIONS
# ============================================================

SCENARIO_COLORS = {

    "Stable / Local":
        "tab:blue",

    "Current-like":
        "tab:orange",

    "Systemic Shift":
        "tab:red"
}


fig, ax = plt.subplots(
    figsize=(11, 7)
)


for scenario, result in simulation_results.items():

    ax.hist(
        result[
            "returns"
        ] * 100,
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
    "Market DNA Monte Carlo — "
    "20-Day Portfolio Return Distribution",
    fontweight="bold"
)

ax.legend()


plt.tight_layout()


plt.savefig(
    "output/figures/"
    "Monte_Carlo_Return_Distributions.png",
    dpi=250
)

plt.close()


# ============================================================
# FIGURE 2
# VaR AND EXPECTED SHORTFALL
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

ax.set_title(
    "Market DNA Monte Carlo — "
    "Tail Risk by Scenario",
    fontweight="bold"
)

ax.tick_params(
    axis="x",
    rotation=0
)


plt.tight_layout()


plt.savefig(
    "output/figures/"
    "Monte_Carlo_Tail_Risk.png",
    dpi=250
)

plt.close()


# ============================================================
# FIGURE 3
# FAN CHART OF PORTFOLIO PATHS
# ============================================================

fig, ax = plt.subplots(
    figsize=(11, 7)
)


days = np.arange(
    HORIZON + 1
)


for scenario, result in simulation_results.items():

    paths = (
        result[
            "paths"
        ]
    )


    p10 = np.quantile(
        paths,
        0.10,
        axis=0
    )

    p50 = np.quantile(
        paths,
        0.50,
        axis=0
    )

    p90 = np.quantile(
        paths,
        0.90,
        axis=0
    )


    color = SCENARIO_COLORS[
        scenario
    ]


    ax.plot(
        days,
        p50 * 100,
        linewidth=2,
        color=color,
        label=scenario
    )


    ax.fill_between(
        days,
        p10 * 100,
        p90 * 100,
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
    "Market DNA Monte Carlo — "
    "Simulated Portfolio Paths",
    fontweight="bold"
)

ax.legend()


plt.tight_layout()


plt.savefig(
    "output/figures/"
    "Monte_Carlo_Portfolio_Paths.png",
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
    "MARKET DNA — MONTE CARLO RISK ENGINE"
)

print(
    "======================================\n"
)


print(
    f"Simulations per scenario: "
    f"{N_SIMULATIONS:,}"
)

print(
    f"Forecast horizon: "
    f"{HORIZON} trading days"
)

print(
    f"Bootstrap block size: "
    f"{BLOCK_SIZE} days"
)


print(
    "\nCurrent Market DNA:"
)

print(
    "DMIDI percentile:",
    round(
        current_dmidi,
        1
    )
)

print(
    "Breadth percentile:",
    round(
        current_breadth,
        1
    )
)


print(
    "\nHistorical blocks available:"
)

for scenario, blocks in scenario_blocks.items():

    print(
        scenario,
        ":",
        len(blocks)
    )


print(
    "\nMonte Carlo risk summary:\n"
)


display_summary = (
    summary.copy()
)


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
    "Mean_Max_Drawdown",
    "Median_Max_Drawdown"
]


for column in percentage_columns:

    display_summary[
        column
    ] = (
        display_summary[
            column
        ]
        * 100
    )


print(
    display_summary.round(
        2
    ).to_string(
        index=False
    )
)


print(
    "\nSaved tables:"
)

print(
    "output/tables/"
    "Monte_Carlo_Risk_Summary.csv"
)

print(
    "output/tables/"
    "current_market_analogues.csv"
)


print(
    "\nSaved figures:"
)

print(
    "output/figures/"
    "Monte_Carlo_Return_Distributions.png"
)

print(
    "output/figures/"
    "Monte_Carlo_Tail_Risk.png"
)

print(
    "output/figures/"
    "Monte_Carlo_Portfolio_Paths.png"
)