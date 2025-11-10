#!/usr/bin/env python3
"""
Two-asset Monte Carlo (GBM) for NQ & ES futures with:
- Auto-calibration from Yahoo Finance (μ, σ, ρ)
- Correlated simulation
- Portfolio paths by weights
- Futures P&L in dollars (contract multipliers)
- Charts, CSV exports, and Excel bundle

- Author : Maxim Konovalov

Examples:
  python mc_two_asset_futures.py --hist_years 5 --paths 10000 --w_nq 0.5 --w_es 0.5
  python mc_two_asset_futures.py --no_calibrate --mu_nq 0.10 --sigma_nq 0.30 --mu_es 0.07 --sigma_es 0.20 --rho 0.85
  python mc_two_asset_futures.py --contracts_nq 1 --contracts_es 1 --mult_nq 20 --mult_es 50
"""

import argparse, json
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import yfinance as yf

# --------- pandas console width for nicer printing ----------
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 100)

# ---------------------------
# Data & calibration helpers
# ---------------------------
def fetch_prices(tickers, hist_years: int):
    end = datetime.now().date()
    start = end - timedelta(days=int(365.25 * hist_years) + 10)
    data = {}
    for t in tickers:
        df = yf.download(t, start=start.isoformat(), end=end.isoformat(),
                         progress=False, auto_adjust=False)  # futures: no CF adjust
        if df.empty:
            raise RuntimeError(f"No data returned for {t}.")
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        data[t] = df[col].dropna()
    prices = pd.concat(data, axis=1).dropna()
    return prices

def fetch_last_prices(tickers):
    out = []
    for t in tickers:
        df = yf.download(t, period="7d", progress=False, auto_adjust=False)
        col = "Adj Close" if "Adj Close" in df.columns else "Close"
        px = df[col].dropna()
        if px.empty:
            out.append(np.nan)
        else:
            out.append(float(px.iloc[-1]))
    return np.array(out, dtype=float)

def calibrate_from_history(prices: pd.DataFrame):
    # daily log returns
    r = np.log(prices / prices.shift(1)).dropna()
    mean_d = r.mean(axis=0).values           # per day
    std_d  = r.std(axis=0, ddof=1).values    # per day
    corr   = np.corrcoef(r.values.T)

    # annualize
    sigma_ann = std_d * np.sqrt(252.0)
    mu_ann    = mean_d * 252.0 + 0.5 * sigma_ann**2  # GBM log-return mean consistency

    S0 = prices.iloc[-1].values.astype(float)
    rho_hist = float(r.corr().iloc[0, 1])
    return S0, mu_ann, sigma_ann, corr, rho_hist

# ---------------------------
# Simulation (correlated GBM)
# ---------------------------
def simulate_correlated_gbm(S0_vec, mu_vec, sigma_vec, corr, years, steps, n_paths, seed):
    dim = len(S0_vec)
    assert dim == 2, "This script is for 2 assets."

    rng = np.random.default_rng(seed)
    dt = years / steps
    chol = np.linalg.cholesky(corr)

    S = np.zeros((steps + 1, n_paths, dim), dtype=float)
    S[0, :, :] = S0_vec

    drift    = (mu_vec - 0.5 * sigma_vec**2) * dt
    step_vol = sigma_vec * np.sqrt(dt)

    for t in range(1, steps + 1):
        Z  = rng.standard_normal((dim, n_paths))
        Zc = chol @ Z
        shocks = (step_vol[:, None] * Zc).T
        step_log_ret = drift + shocks
        S[t] = S[t-1] * np.exp(step_log_ret)

    start = pd.Timestamp(datetime.now().date())
    dates = pd.bdate_range(start=start, periods=steps + 1)
    paths_nq = pd.DataFrame(S[:, :, 0], index=dates, columns=[f"path_{i}" for i in range(n_paths)])
    paths_es = pd.DataFrame(S[:, :, 1], index=dates, columns=[f"path_{i}" for i in range(n_paths)])
    return paths_nq, paths_es

# ---------------------------
# Stats
# ---------------------------
def stats_from_paths(paths_df: pd.DataFrame, S0: float, years: float, rf: float):
    terminal_prices = paths_df.iloc[-1].to_numpy()
    terminal_returns = terminal_prices / S0 - 1.0
    ann_returns = (terminal_prices / S0) ** (1.0 / years) - 1.0

    daily_rets = paths_df.pct_change().iloc[1:, :].to_numpy()
    ann_vols = np.nanstd(daily_rets, axis=0, ddof=1) * np.sqrt(252)

    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = (ann_returns - rf) / ann_vols
        sharpe = np.where(np.isfinite(sharpe), sharpe, np.nan)

    S = paths_df.to_numpy()
    run_max = np.maximum.accumulate(S, axis=0)
    dd = S / run_max - 1.0
    max_dd = dd.min(axis=0)

    qs = np.percentile(S, q=[5, 25, 50, 75, 95], axis=1).T
    qdf = pd.DataFrame(qs, index=paths_df.index, columns=["p05", "p25", "p50", "p75", "p95"])

    alpha = 0.95
    q = np.quantile(terminal_returns, 1 - alpha)
    VaR_95 = -q
    ES_95 = -terminal_returns[terminal_returns <= q].mean()

    summary = pd.DataFrame([{
        "S0": S0, "T_years": years, "n_paths": paths_df.shape[1], "steps": paths_df.shape[0] - 1,
        "Ann_Return_mean": float(np.mean(ann_returns)),
        "Ann_Return_median": float(np.median(ann_returns)),
        "Ann_Vol_median": float(np.nanmedian(ann_vols)),
        "Sharpe_median": float(np.nanmedian(sharpe)),
        "Max_Drawdown_median": float(np.median(max_dd)),
        "Prob_Loss_over_horizon": float(np.mean(terminal_prices < S0)),
        "VaR_95_horizon_loss_frac": float(VaR_95),
        "ES_95_horizon_loss_frac": float(ES_95),
        "Median_Terminal_Price": float(np.median(terminal_prices)),
        "Terminal_Price_p05": float(np.quantile(terminal_prices, 0.05)),
        "Terminal_Price_p95": float(np.quantile(terminal_prices, 0.95)),
    }])
    return summary, qdf, terminal_prices, ann_returns, max_dd

def portfolio_from_two(paths_nq: pd.DataFrame, paths_es: pd.DataFrame, w_nq: float, w_es: float):
    nq_rel = paths_nq / paths_nq.iloc[0]
    es_rel = paths_es / paths_es.iloc[0]
    port = w_nq * nq_rel + w_es * es_rel
    port.index = paths_nq.index
    port.columns = [f"path_{i}" for i in range(paths_nq.shape[1])]
    return port

# ---------------------------
# Plotting (1 fig each)
# ---------------------------
def _fan_plot(ax, qdf, title):
    x_num = mdates.date2num(qdf.index.to_pydatetime())
    p05, p25, p50, p75, p95 = [qdf[c].values.astype(float) for c in ["p05","p25","p50","p75","p95"]]
    ax.plot(x_num, p50, linewidth=2, label="Median")
    ax.fill_between(x_num, p25, p75, alpha=0.3, label="25–75%")
    ax.fill_between(x_num, p05, p95, alpha=0.15, label="5–95%")
    ax.set_title(title); ax.set_xlabel("Date"); ax.set_ylabel("Price"); ax.legend()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

def save_paths_plot(paths_df, title, outpath):
    plt.figure(figsize=(10, 6))
    n_show = min(100, paths_df.shape[1])
    for i in range(n_show):
        plt.plot(paths_df.index, paths_df.iloc[:, i])
    plt.title(title); plt.xlabel("Date"); plt.ylabel("Price")
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

def save_fan_plot(qdf, title, outpath):
    fig, ax = plt.subplots(figsize=(10, 6))
    _fan_plot(ax, qdf, title)
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

def save_hist(arr, title, xlabel, outpath, vline=None):
    plt.figure(figsize=(10, 6))
    plt.hist(arr, bins=60)
    if vline is not None:
        plt.axvline(vline, linestyle="--", linewidth=2)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel("Frequency")
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

def save_scatter(x, y, title, xlabel, ylabel, outpath):
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, s=10, alpha=0.5)
    plt.title(title); plt.xlabel(xlabel); plt.ylabel(ylabel)
    plt.tight_layout(); plt.savefig(outpath, dpi=150); plt.close()

# ---------------------------
# Excel bundle
# ---------------------------
def save_excel_bundle(outdir: Path, summary: pd.DataFrame,
                      q_nq: pd.DataFrame, q_es: pd.DataFrame, q_pf: pd.DataFrame,
                      paths_nq: pd.DataFrame, paths_es: pd.DataFrame, port: pd.DataFrame,
                      pnl_total: np.ndarray):
    xlsx = outdir / "bundle.xlsx"
    with pd.ExcelWriter(xlsx, engine="xlsxwriter") as writer:
        summary.round(6).to_excel(writer, sheet_name="summary", index=False)
        q_nq.to_excel(writer, sheet_name="nq_quantiles")
        q_es.to_excel(writer, sheet_name="es_quantiles")
        q_pf.to_excel(writer, sheet_name="portfolio_quantiles")
        # keep workbook size reasonable: first 100 paths each
        paths_nq.iloc[:, :100].to_excel(writer, sheet_name="nq_paths_sample")
        paths_es.iloc[:, :100].to_excel(writer, sheet_name="es_paths_sample")
        port.iloc[:, :100].to_excel(writer, sheet_name="portfolio_paths_sample")
        pd.DataFrame({"pnl_$": pnl_total}).to_excel(writer, sheet_name="futures_pnl_$", index=False)

# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker_nq", type=str, default="NQ=F")
    ap.add_argument("--ticker_es", type=str, default="ES=F")
    ap.add_argument("--hist_years", type=int, default=5, help="History length (years) for calibration")
    ap.add_argument("--years", type=float, default=1.0, help="Simulation horizon (years)")
    ap.add_argument("--steps", type=int, default=252, help="Steps over the horizon")
    ap.add_argument("--paths", type=int, default=10000)
    ap.add_argument("--rf", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--w_nq", type=float, default=0.5)
    ap.add_argument("--w_es", type=float, default=0.5)

    ap.add_argument("--no_calibrate", action="store_true", help="Skip history; use manual params below")
    ap.add_argument("--mu_nq", type=float, default=0.10)
    ap.add_argument("--sigma_nq", type=float, default=0.30)
    ap.add_argument("--mu_es", type=float, default=0.07)
    ap.add_argument("--sigma_es", type=float, default=0.20)
    ap.add_argument("--rho", type=float, default=0.85, help="Correlation between NQ and ES")

    ap.add_argument("--contracts_nq", type=float, default=1.0)
    ap.add_argument("--contracts_es", type=float, default=1.0)
    ap.add_argument("--mult_nq", type=float, default=20.0)  # E-mini NQ $20/point (MNQ=2)
    ap.add_argument("--mult_es", type=float, default=50.0)  # E-mini ES $50/point (MES=5)

    ap.add_argument("--outdir", type=str, default="outputs_nq_es")
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(exist_ok=True)
    tickers = [args.ticker_nq, args.ticker_es]

    # Calibrate or use manual params
    if not args.no_calibrate:
        prices = fetch_prices(tickers, hist_years=args.hist_years)
        S0_vec, mu_vec, sigma_vec, corr_mat, rho_hist = calibrate_from_history(prices)
        rho_used = rho_hist
        calib_note = {"source": "Yahoo Finance", "hist_years": args.hist_years, "rho_hist": float(rho_hist)}
    else:
        # Get last prices for realistic $ PnL (even if using manual μ/σ/ρ)
        S0_vec = fetch_last_prices(tickers)
        if np.any(np.isnan(S0_vec)):
            S0_vec = np.array([100.0, 100.0], dtype=float)  # fallback
        mu_vec = np.array([args.mu_nq, args.mu_es], dtype=float)
        sigma_vec = np.array([args.sigma_nq, args.sigma_es], dtype=float)
        corr_mat = np.array([[1.0, args.rho], [args.rho, 1.0]], dtype=float)
        rho_used = args.rho
        calib_note = {"source": "manual"}

    # Simulate
    paths_nq, paths_es = simulate_correlated_gbm(
        S0_vec=S0_vec, mu_vec=mu_vec, sigma_vec=sigma_vec, corr=corr_mat,
        years=args.years, steps=args.steps, n_paths=args.paths, seed=args.seed
    )

    # Portfolio (rebased to 1 at t=0)
    port = portfolio_from_two(paths_nq, paths_es, args.w_nq, args.w_es)

    # Stats
    sum_nq, q_nq, term_nq, ann_nq, dd_nq = stats_from_paths(paths_nq, S0=S0_vec[0], years=args.years, rf=args.rf)
    sum_es, q_es, term_es, ann_es, dd_es = stats_from_paths(paths_es, S0=S0_vec[1], years=args.years, rf=args.rf)
    sum_pf, q_pf, term_pf, ann_pf, dd_pf = stats_from_paths(port, S0=1.0, years=args.years, rf=args.rf)

    # Futures dollar P&L at horizon
    pnl_nq = (term_nq - S0_vec[0]) * args.mult_nq * args.contracts_nq
    pnl_es = (term_es - S0_vec[1]) * args.mult_es * args.contracts_es
    pnl_total = pnl_nq + pnl_es
    pnl_q05 = np.quantile(pnl_total, 0.05)
    pnl_es95 = pnl_total[pnl_total <= pnl_q05].mean()

    pnl_summary = pd.DataFrame([{
        "asset": "FUTURES_PNL($)",
        "Mean_PnL_$": float(np.mean(pnl_total)),
        "Median_PnL_$": float(np.median(pnl_total)),
        "VaR95_$ (5% quantile)": float(pnl_q05),
        "ES95_$ (avg worst 5%)": float(pnl_es95),
        "Prob_Neg_PnL": float(np.mean(pnl_total < 0.0)),
        "NQ_contracts": args.contracts_nq,
        "ES_contracts": args.contracts_es,
        "mult_NQ": args.mult_nq,
        "mult_ES": args.mult_es,
    }])

    # Export CSVs
    paths_nq.iloc[:, :250].to_csv(outdir / "nq_paths_sample.csv")
    paths_es.iloc[:, :250].to_csv(outdir / "es_paths_sample.csv")
    port.iloc[:, :250].to_csv(outdir / "portfolio_paths_sample.csv")

    q_nq.to_csv(outdir / "nq_quantiles.csv")
    q_es.to_csv(outdir / "es_quantiles.csv")
    q_pf.to_csv(outdir / "portfolio_quantiles.csv")

    summary = pd.concat([
        sum_nq.assign(asset="NQ"),
        sum_es.assign(asset="ES"),
        sum_pf.assign(asset=f"PORTFOLIO(wNQ={args.w_nq:.2f},wES={args.w_es:.2f})"),
        pnl_summary
    ], ignore_index=True)
    summary = summary[["asset"] + [c for c in summary.columns if c != "asset"]]
    summary.to_csv(outdir / "summary.csv", index=False)

    # Save raw PnL vector
    np.savetxt(outdir / "pnl_total.csv", pnl_total, delimiter=",")

    # Params snapshot
    with open(outdir / "run_params.json", "w") as f:
        json.dump({
            "tickers": tickers,
            "S0_vec": S0_vec.tolist(),
            "mu_vec": mu_vec.tolist(),
            "sigma_vec": sigma_vec.tolist(),
            "rho_used": float(rho_used),
            "weights": {"w_nq": args.w_nq, "w_es": args.w_es},
            "years": args.years, "steps": args.steps, "paths": args.paths,
            "rf": args.rf, "seed": args.seed, "calibration": calib_note
        }, f, indent=2)

    # Charts
    save_paths_plot(paths_nq, "NQ Simulated Price Paths (first 100)", outdir / "01_nq_paths.png")
    save_paths_plot(paths_es, "ES Simulated Price Paths (first 100)", outdir / "02_es_paths.png")
    save_paths_plot(port,    "Portfolio Paths (first 100)", outdir / "03_portfolio_paths.png")

    save_fan_plot(q_nq, "NQ Fan Chart (Percentile Bands)", outdir / "04_nq_fan.png")
    save_fan_plot(q_es, "ES Fan Chart (Percentile Bands)", outdir / "05_es_fan.png")
    save_fan_plot(q_pf, "Portfolio Fan Chart (Percentile Bands)", outdir / "06_portfolio_fan.png")

    save_hist(term_nq, "NQ Terminal Prices", "Terminal Price", outdir / "07_nq_terminal_prices.png", vline=S0_vec[0])
    save_hist(term_es, "ES Terminal Prices", "Terminal Price", outdir / "08_es_terminal_prices.png", vline=S0_vec[1])
    save_hist(term_pf, "Portfolio Terminal Value", "Terminal Value", outdir / "09_portfolio_terminal_value.png", vline=1.0)

    save_hist(ann_nq, "NQ Annualized Returns", "Annualized Return", outdir / "10_nq_ann_returns.png")
    save_hist(ann_es, "ES Annualized Returns", "Annualized Return", outdir / "11_es_ann_returns.png")
    save_hist(ann_pf, "Portfolio Annualized Returns", "Annualized Return", outdir / "12_portfolio_ann_returns.png")

    save_hist(dd_nq, "NQ Max Drawdown (per path)", "Max Drawdown", outdir / "13_nq_max_dd.png")
    save_hist(dd_es, "ES Max Drawdown (per path)", "Max Drawdown", outdir / "14_es_max_dd.png")
    save_hist(dd_pf, "Portfolio Max Drawdown (per path)", "Max Drawdown", outdir / "15_portfolio_max_dd.png")

    save_scatter(term_nq / S0_vec[0] - 1.0, term_es / S0_vec[1] - 1.0,
                 "Terminal Returns: NQ vs ES", "NQ Terminal Return", "ES Terminal Return",
                 outdir / "16_scatter_terminal_returns.png")

    save_hist(pnl_total, "Total P&L (NQ/ES futures, $)", "PnL ($)", outdir / "17_total_pnl_hist.png")

    # Excel bundle
    save_excel_bundle(outdir, summary, q_nq, q_es, q_pf, paths_nq, paths_es, port, pnl_total)

    # Console summary
    print("\n== SUMMARY ==")
    print(summary.round(4).to_string(index=False))
    print(f"\nFiles saved to: {outdir.resolve()}")
    for f in sorted(outdir.iterdir()):
        print(" -", f.name)

if __name__ == "__main__":
    main()

