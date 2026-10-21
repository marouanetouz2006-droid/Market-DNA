# Market DNA

## Detecting Structural Identity Drift Across Financial Assets

**Market DNA** is a quantitative finance research project designed to measure how the structural behaviour of financial assets evolves through time.

Instead of describing an asset only through returns or volatility, the framework represents its financial identity through time-varying sensitivities to a set of common economic and market factors.

The objective is to detect when these relationships change, determine whether the change is concentrated in a small number of assets or broadly shared across markets, and assess whether structural drift contains useful information about future market risk.

---

## Research Question

**Can changes in cross-asset factor exposures provide a useful measure of structural market transformation?**

The project initially investigated whether structural identity drift could improve volatility forecasting.

Out-of-sample testing did not support this hypothesis.

The framework was therefore developed into a broader structural market-state model focused on:

- detecting changes in asset behaviour;
- distinguishing concentrated from systemic structural shifts;
- analysing cross-asset participation;
- assessing links with future volatility and correlation;
- conditioning portfolio risk simulations on structural market regimes.

---

## Asset Universe

The analysis covers six liquid assets representing distinct areas of the financial system:

| Asset | Representation |
|---|---|
| QQQ | US technology / growth equities |
| XLF | US financial sector |
| XLE | US energy sector |
| FEZ | Euro-area equities |
| TLT | Long-duration US Treasury bonds |
| GLD | Gold |

---

## Factor Architecture

Each asset is modelled against six common market factors:

- **MKT** — global equity market return
- **RATE** — change in the US 10-year Treasury yield
- **CREDIT** — credit-risk proxy
- **USD** — US dollar return
- **OIL** — crude oil return
- **VIX** — change in market-implied volatility

Rolling factor regressions generate a time-varying exposure vector for each asset.

This vector is interpreted as the asset's **Financial DNA**.

---

## Methodology

### 1. Rolling Financial DNA

For each asset, rolling regressions estimate sensitivities to the six market factors.

The resulting beta vector describes how the asset currently interacts with the broader financial environment.

### 2. Identity Drift

Changes in Financial DNA are measured using the **Mahalanobis distance**.

This approach accounts not only for the magnitude of changes in factor exposures, but also for their historical variability and covariance structure.

A dynamic asset-level measure, **DIDS — Dynamic Identity Drift Score**, compares the current exposure vector with a rolling historical reference distribution.

### 3. Dynamic Market Identity Drift Index

The six asset-level DIDS measures are aggregated into:

**DMIDI — Dynamic Market Identity Drift Index**

DMIDI measures the overall magnitude of structural displacement across the asset universe.

### 4. Drift Breadth

Magnitude alone does not indicate whether structural change is widespread.

An entropy-based **Drift Breadth** measure captures how evenly identity drift is distributed across the six assets.

High breadth indicates broad participation.

Low breadth indicates that aggregate drift is concentrated in a limited number of assets.

### 5. Market Regimes

DMIDI and Drift Breadth are combined to distinguish four structural regimes:

- **Stable / Local**
- **Concentrated Shift**
- **Broad Mild Transition**
- **Systemic Structural Shift**

---

## Main Findings

### Structural drift is multidimensional

Principal Component Analysis shows that the first component explains only approximately **25.4%** of total DIDS variation.

Structural transformations therefore cannot be reduced to a single common factor.

### Market DNA does not improve volatility forecasting

Out-of-sample tests show that adding MIDI or DMIDI does **not** improve volatility forecasts relative to models based on the VIX and past realised volatility.

This negative result is an important part of the research.

Market DNA should therefore not be interpreted as a volatility-prediction model.

### Future cross-asset correlation

A nonlinear relationship appears between DMIDI and future average cross-asset correlation.

Quadratic DMIDI specifications provide modest out-of-sample improvements in several years, but pooled statistical evidence is not strong enough to claim robust forecasting superiority.

The result is therefore interpreted as **suggestive structural information rather than a trading signal**.

### Structural regimes contain different information

The combination of drift magnitude and breadth allows the framework to distinguish between:

- large changes driven by a single asset;
- widespread structural transformations affecting the broader system.

This distinction is not visible from aggregate drift alone.

---

## Current Market DNA State

As of **30 September 2026**, the framework identifies a:

### Concentrated Shift

- **DMIDI:** 10.71
- **DMIDI percentile:** 83.8th
- **Drift Breadth:** 0.847
- **Breadth:** below the minimum of the 2015–2021 reference distribution
- **Dominant asset:** GLD
- **GLD contribution:** approximately 44.7%

The signal therefore represents elevated structural drift, but the transformation is unusually concentrated rather than broadly systemic.

---

## Historical Case Studies

The framework is illustrated using several historical structural episodes:

- **2020:** Concentrated post-shock transformation
- **2022:** Systemic structural transition
- **2025:** Broad systemic structural shift
- **2026:** Concentrated gold-led drift

These case studies show how similar levels of aggregate drift may represent very different market structures depending on cross-asset participation.

---

## Scenario-Conditioned Monte Carlo

The final stage of the project investigates whether Market DNA regimes can be used to condition portfolio risk simulations.

A historical block-bootstrap Monte Carlo framework generates 20-day portfolio paths using market observations associated with different structural states.

The results do not show a simple relationship such as:

> higher DMIDI = higher portfolio risk.

Instead, different regimes produce different combinations of:

- terminal returns;
- Value at Risk;
- Expected Shortfall;
- loss probabilities;
- maximum drawdowns.

This supports the interpretation of Market DNA as a **risk-context framework rather than a directional forecasting model**.

---

## Key Conclusion

The original forecasting hypothesis did not survive out-of-sample testing.

However, the project shows that changes in factor exposures contain useful information about the **structure of financial markets**.

Market DNA provides a framework for answering three different questions:

1. **How unusual is the current behaviour of financial assets?**
2. **Is structural change concentrated or broadly distributed?**
3. **What type of market environment does that structural configuration represent?**

The framework is therefore best interpreted as a tool for **structural market diagnostics, regime analysis and risk contextualisation**, rather than as a standalone return or volatility predictor.

---

## Full Research Report

The complete research report, including methodology, econometric tests, robustness analysis, historical case studies and Monte Carlo results, is available here:

[Market DNA — Full Research Report](report/Market_DNA_Full_Report.pdf)

---

## Technologies

The project was developed using:

- Python
- R
- pandas
- NumPy
- statsmodels
- scikit-learn
- matplotlib
- econometric regression analysis
- Newey–West standard errors
- Principal Component Analysis
- historical block-bootstrap Monte Carlo simulation

---

## Repository Structure

```text
market-dna/
│
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
│
├── scripts/
│   ├── 01_data_pipeline.py
│   ├── 02_rolling_dna.py
│   ├── 03_identity_drift.py
│   ├── 04_research_dataset.py
│   ├── 05_dynamic_midi.py
│   ├── 06_quant_analysis.py
│   ├── 07_regime_map.py
│   ├── 08_case_studies.py
│   └── ...
│
├── r/
│
├── output/
│   └── selected_figures/
│
└── report/
    └── Market_DNA_Full_Report.pdf
```

---

## Disclaimer

This project is an academic and research exercise.

It is not investment advice, and the results should not be interpreted as trading recommendations or forecasts of future market performance.
