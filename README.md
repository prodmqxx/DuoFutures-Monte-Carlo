# DuoFutures Monte Carlo (NQ & ES)

Correlated Monte Carlo engine for **Nasdaq-100 (NQ)** and **S&P 500 (ES)** futures.  
It auto-calibrates drift/vol/correlation from recent history (Yahoo Finance) or runs with your manual params, simulates correlated GBM paths, builds a weighted portfolio, and computes **risk metrics + $PnL**. Exports pretty charts, CSVs, and a single Excel bundle.

---

## Features
- **Auto-calibration** of μ, σ, and ρ from history (`NQ=F`, `ES=F`) or manual overrides
- **Correlated GBM** paths for NQ & ES (Cholesky)
- **Portfolio** blending with weights (rebased to 1.0 at t=0)
- **Futures P&L in USD** (contract counts & multipliers; E-mini / Micro friendly)
- **Risk stats**: VaR/ES (95%), Sharpe (median), max drawdown (median), loss probability
- **Exports**: PNG charts, CSVs, and **Excel `bundle.xlsx`** with summaries, quantiles, sample paths, and full P&L vector

---

## Install
```bash
pip install numpy pandas matplotlib yfinance xlsxwriter
