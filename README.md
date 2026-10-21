# Market DNA — Starter Code

Run in order:
1. `python 01_data_pipeline.py`
2. `python 02_rolling_dna.py`
3. `python 03_identity_drift.py`
4. `python 04_research_dataset.py`
5. Open R and run `r/statistical_tests.R`

Core assets: QQQ, XLF, XLE, FEZ, TLT, GLD.
Factors: VT, DGS10, HYG-IEF, UUP, CL=F, VIXCLS.
Sample: 2010-01-04 to 2026-09-30.
Calibration: 2011-2014. Holdout: 2022-2026.

Important: `VIX` is the daily VIX change used as a DNA factor; `VIX_LEVEL` is retained separately as a forecasting control.
