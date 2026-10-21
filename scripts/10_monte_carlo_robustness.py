import numpy as np
import pandas as pd

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

N_WORST_BLOCKS = 15


# ============================================================
# OUTPUT
# ============================================================

Path(
    "output/tables"
).mkdir(
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
# CURRENT-LIKE HISTORICAL ANALOGUES
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


# Remove final observations without a complete forward block

analogue_data = analogue_data.iloc[
    :-BLOCK_SIZE
]


current_analogues = (
    analogue_data
    .nsmallest(
        CURRENT_NEIGHBORS,
        "DNA_DISTANCE"
    )
    .copy()
)


# ============================================================
# CHECK 1
# YEAR DISTRIBUTION OF CURRENT-LIKE ANALOGUES
# ============================================================

current_analogues[
    "Year"
] = current_analogues.index.year


analogue_years = (
    current_analogues
    .groupby(
        "Year"
    )
    .size()
    .reset_index(
        name="Count"
    )
)


analogue_years[
    "Share"
] = (
    analogue_years[
        "Count"
    ]
    / analogue_years[
        "Count"
    ].sum()
)


analogue_years[
    "Share_pct"
] = (
    analogue_years[
        "Share"
    ]
    * 100
)


analogue_years.to_csv(
    "output/tables/"
    "current_analogue_year_distribution.csv",
    index=False
)


top_year_row = (
    analogue_years.loc[
        analogue_years[
            "Count"
        ].idxmax()
    ]
)


number_of_years = (
    analogue_years[
        "Year"
    ].nunique()
)


# Herfindahl concentration of analogue years

year_hhi = np.sum(
    analogue_years[
        "Share"
    ] ** 2
)


effective_years = (
    1 / year_hhi
)


# ============================================================
# BUILD SCENARIO DATES
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


current_distance_map = (
    current_analogues[
        "DNA_DISTANCE"
    ]
    .to_dict()
)


# ============================================================
# POSITION LOOKUP
# ============================================================

dates = df.index

date_to_position = {
    date: i
    for i, date
    in enumerate(
        dates
    )
}


# ============================================================
# BUILD BLOCK RECORDS
#
# Each record stores:
# - Market DNA signal date
# - Forward 5-day return window
# - Asset return matrix
# - Portfolio 5-day return
# ============================================================

def build_block_records(
    scenario_dates,
    distance_map=None
):

    records = []

    returns = df[
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
            returns.iloc[
                i + 1:
                i + 1 + BLOCK_SIZE
            ]
        )


        if (
            len(block) != BLOCK_SIZE
            or block
                .isna()
                .any()
                .any()
        ):
            continue


        log_returns = (
            block.to_numpy()
        )


        simple_returns = (
            np.expm1(
                log_returns
            )
        )


        portfolio_daily = (
            simple_returns
            @ WEIGHTS
        )


        portfolio_cumulative = (
            np.prod(
                1
                + portfolio_daily
            )
            - 1
        )


        distance = np.nan

        if distance_map is not None:

            distance = (
                distance_map.get(
                    signal_date,
                    np.nan
                )
            )


        records.append(
            {
                "signal_date":
                    signal_date,

                "start_date":
                    dates[
                        i + 1
                    ],

                "end_date":
                    dates[
                        i + BLOCK_SIZE
                    ],

                "start_pos":
                    i + 1,

                "end_pos":
                    i + BLOCK_SIZE,

                "portfolio_return":
                    portfolio_cumulative,

                "distance":
                    distance,

                "block":
                    log_returns
            }
        )


    return records


stable_records = (
    build_block_records(
        stable_dates
    )
)


systemic_records = (
    build_block_records(
        systemic_dates
    )
)


current_records = (
    build_block_records(
        current_like_dates,
        current_distance_map
    )
)


# ============================================================
# CHECK 2
# WORST STABLE / LOCAL FORWARD BLOCKS
# ============================================================

worst_stable = sorted(
    stable_records,
    key=lambda x:
        x[
            "portfolio_return"
        ]
)[
    :N_WORST_BLOCKS
]


worst_stable_table = pd.DataFrame(
    [
        {
            "Signal_Date":
                record[
                    "signal_date"
                ],

            "Forward_Start":
                record[
                    "start_date"
                ],

            "Forward_End":
                record[
                    "end_date"
                ],

            "Portfolio_Return":
                record[
                    "portfolio_return"
                ],

            "Portfolio_Return_pct":
                record[
                    "portfolio_return"
                ]
                * 100
        }

        for record
        in worst_stable
    ]
)


worst_stable_table.to_csv(
    "output/tables/"
    "worst_stable_forward_blocks.csv",
    index=False
)


# ============================================================
# NON-OVERLAPPING BLOCK SELECTION
# ============================================================

def intervals_overlap(
    a,
    b
):

    return not (
        a[
            "end_pos"
        ]
        <
        b[
            "start_pos"
        ]

        or

        a[
            "start_pos"
        ]
        >
        b[
            "end_pos"
        ]
    )


# ------------------------------------------------------------
# For Stable/Systemic:
# select blocks chronologically,
# then skip overlapping windows
# ------------------------------------------------------------

def select_non_overlapping_chronological(
    records
):

    ordered = sorted(
        records,
        key=lambda x:
            x[
                "start_pos"
            ]
    )

    selected = []

    last_end = -1


    for record in ordered:

        if (
            record[
                "start_pos"
            ]
            > last_end
        ):

            selected.append(
                record
            )

            last_end = (
                record[
                    "end_pos"
                ]
            )


    return selected


# ------------------------------------------------------------
# For Current-like:
# prioritise closest DNA analogues,
# while rejecting overlapping windows
# ------------------------------------------------------------

def select_non_overlapping_by_distance(
    records
):

    ordered = sorted(
        records,
        key=lambda x:
            x[
                "distance"
            ]
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


stable_nonoverlap = (
    select_non_overlapping_chronological(
        stable_records
    )
)


systemic_nonoverlap = (
    select_non_overlapping_chronological(
        systemic_records
    )
)


current_nonoverlap = (
    select_non_overlapping_by_distance(
        current_records
    )
)


# ============================================================
# MONTE CARLO SIMULATION
# ============================================================

def simulate(
    records,
    seed
):

    if len(records) == 0:

        raise ValueError(
            "No blocks available."
        )


    blocks = [
        record[
            "block"
        ]

        for record
        in records
    ]


    rng = np.random.default_rng(
        seed
    )


    blocks_needed = int(
        np.ceil(
            HORIZON
            / BLOCK_SIZE
        )
    )


    cumulative_returns = (
        np.empty(
            N_SIMULATIONS
        )
    )


    maximum_drawdowns = (
        np.empty(
            N_SIMULATIONS
        )
    )


    for simulation in range(
        N_SIMULATIONS
    ):

        choices = rng.integers(
            0,
            len(blocks),
            size=blocks_needed
        )


        simulated_log_returns = (
            np.vstack(
                [
                    blocks[index]
                    for index
                    in choices
                ]
            )
        )


        simulated_log_returns = (
            simulated_log_returns[
                :HORIZON
            ]
        )


        asset_simple_returns = (
            np.expm1(
                simulated_log_returns
            )
        )


        portfolio_daily = (
            asset_simple_returns
            @ WEIGHTS
        )


        wealth = np.concatenate(
            [
                [1.0],

                np.cumprod(
                    1
                    + portfolio_daily
                )
            ]
        )


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


    return (
        cumulative_returns,
        maximum_drawdowns
    )


# ============================================================
# RISK METRICS
# ============================================================

def calculate_metrics(
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


# ============================================================
# ORIGINAL VS NON-OVERLAPPING POOLS
# ============================================================

original_pools = {

    "Stable / Local":
        stable_records,

    "Current-like":
        current_records,

    "Systemic Shift":
        systemic_records
}


nonoverlap_pools = {

    "Stable / Local":
        stable_nonoverlap,

    "Current-like":
        current_nonoverlap,

    "Systemic Shift":
        systemic_nonoverlap
}


results = []


seed_counter = 0


for version, pools in [

    (
        "Original overlapping",
        original_pools
    ),

    (
        "Non-overlapping",
        nonoverlap_pools
    )

]:

    for scenario, records in pools.items():

        returns, drawdowns = simulate(
            records,
            RANDOM_SEED
            + seed_counter
        )


        seed_counter += 1


        metrics = (
            calculate_metrics(
                returns,
                drawdowns
            )
        )


        metrics[
            "Version"
        ] = version

        metrics[
            "Scenario"
        ] = scenario

        metrics[
            "Historical_Blocks"
        ] = len(
            records
        )


        results.append(
            metrics
        )


robustness_summary = (
    pd.DataFrame(
        results
    )
)


robustness_summary = robustness_summary[
    [
        "Version",
        "Scenario",
        "Historical_Blocks",
        "Mean_Return",
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


robustness_summary.to_csv(
    "output/tables/"
    "Monte_Carlo_Robustness_Summary.csv",
    index=False
)


# ============================================================
# PRINT RESULTS
# ============================================================

print(
    "\n======================================"
)

print(
    "MARKET DNA — MONTE CARLO ROBUSTNESS"
)

print(
    "======================================\n"
)


# ============================================================
# CHECK 1 OUTPUT
# ============================================================

print(
    "CHECK 1 — CURRENT-LIKE ANALOGUES BY YEAR\n"
)


display_years = (
    analogue_years.copy()
)


display_years[
    "Share_pct"
] = (
    display_years[
        "Share_pct"
    ].round(
        1
    )
)


print(
    display_years[
        [
            "Year",
            "Count",
            "Share_pct"
        ]
    ]
    .to_string(
        index=False
    )
)


print(
    "\nYears represented:",
    number_of_years
)


print(
    "Largest single-year share:",
    f"{top_year_row['Share_pct']:.1f}%"
)


print(
    "Effective number of years:",
    round(
        effective_years,
        2
    )
)


# ============================================================
# CHECK 2 OUTPUT
# ============================================================

print(
    "\nCHECK 2 — WORST STABLE / LOCAL "
    "FORWARD 5-DAY BLOCKS\n"
)


print(
    worst_stable_table[
        [
            "Signal_Date",
            "Forward_Start",
            "Forward_End",
            "Portfolio_Return_pct"
        ]
    ]
    .round(
        {
            "Portfolio_Return_pct":
                2
        }
    )
    .to_string(
        index=False
    )
)


# ============================================================
# CHECK 3 OUTPUT
# ============================================================

print(
    "\nCHECK 3 — BLOCK COUNTS\n"
)


print(
    "Stable / Local:",
    len(
        stable_records
    ),
    "->",
    len(
        stable_nonoverlap
    )
)


print(
    "Current-like:",
    len(
        current_records
    ),
    "->",
    len(
        current_nonoverlap
    )
)


print(
    "Systemic Shift:",
    len(
        systemic_records
    ),
    "->",
    len(
        systemic_nonoverlap
    )
)


print(
    "\nMONTE CARLO ROBUSTNESS SUMMARY\n"
)


display_summary = (
    robustness_summary.copy()
)


percentage_columns = [
    "Mean_Return",
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
    "\nSaved:"
)

print(
    "output/tables/"
    "current_analogue_year_distribution.csv"
)

print(
    "output/tables/"
    "worst_stable_forward_blocks.csv"
)

print(
    "output/tables/"
    "Monte_Carlo_Robustness_Summary.csv"
)