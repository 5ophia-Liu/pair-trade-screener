# -*- coding: utf-8 -*-
"""
Pairs Trading Cointegration Screener
-------------------------------------
Screens a list of stock/ETF pairs for statistical arbitrage candidates using:
  - Engle-Granger cointegration testing (3M / 6M / 1Y windows)
  - OLS hedge ratios (price-based and return-based, multiple lookback windows)
  - Log-price spread Z-scores
  - Ornstein-Uhlenbeck half-life estimation (mean-reversion speed)

Pairs are read from pairs.csv (ticker_a, ticker_b columns).
Price data is pulled directly from Yahoo Finance via yfinance.

Outputs:
  - Console summary table of all screened pairs
  - Console + CSV table of "tradable" candidates (cointegrated + |Z| >= 2)
  - Full results text file per run in logfiles/
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint
import yfinance as yf
import time
import os
from datetime import datetime

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAIRS_CSV = os.path.join(SCRIPT_DIR, "pairs.csv")
LOG_DIR = os.path.join(SCRIPT_DIR, "logfiles")

pd.set_option('display.width', 400)
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.expand_frame_repr', False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def fmt_elapsed(seconds):
    """Format elapsed seconds as HH:MM:SS."""
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def log_step(name, start_time):
    """Print the elapsed time for a completed step."""
    elapsed = time.time() - start_time
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {name} completed in {fmt_elapsed(elapsed)}")
    return elapsed


def hedge_ratio_from_returns(price_a, price_b):
    """OLS hedge ratio computed on daily returns rather than price levels."""
    df = pd.DataFrame({"a": price_a, "b": price_b}).dropna()
    ret_a = df["a"].pct_change().dropna()
    ret_b = df["b"].pct_change().dropna()
    ret_a, ret_b = ret_a.align(ret_b, join="inner")

    X = sm.add_constant(ret_b)
    model = sm.OLS(ret_a, X).fit()
    beta = model.params.iloc[1]
    return beta, model


def log_price_beta(price_a, price_b):
    """OLS hedge ratio on log prices; returns the resulting log-spread series."""
    df = pd.DataFrame({"a": price_a, "b": price_b}).dropna()
    y = np.log(df["a"])
    x = np.log(df["b"])
    X = sm.add_constant(x)
    modellog = sm.OLS(y, X).fit()
    alphalog = modellog.params.iloc[0]
    betalog = modellog.params.iloc[1]
    spreadlog = y - (alphalog + betalog * x)
    return betalog, alphalog, spreadlog, modellog


def pair_correlation(price_a, price_b, use_log_returns=False):
    df = pd.DataFrame({"a": price_a, "b": price_b}).dropna()
    if use_log_returns:
        ret_a = np.log(df["a"]).diff()
        ret_b = np.log(df["b"]).diff()
    else:
        ret_a = df["a"].pct_change()
        ret_b = df["b"].pct_change()
    return ret_a.corr(ret_b)


def get_company_name(ticker):
    """Fetches the company/long name for a ticker directly via yfinance."""
    try:
        return yf.Ticker(ticker).info.get('longName', ticker)
    except Exception:
        return ticker


def extract_close(raw: pd.DataFrame) -> pd.DataFrame:
    """Reduce a yf.download result to one 'Close' column per ticker,
    handling both single- and multi-level column layouts."""
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw[["Close"]] if "Close" in raw.columns else raw
    levels = raw.columns.levels
    for lvl in (1, 0):
        if "Close" in levels[lvl]:
            return raw.xs("Close", axis=1, level=lvl)
    raise KeyError("Close")


def load_pairs(csv_path):
    df = pd.read_csv(csv_path)
    return list(df[["ticker_a", "ticker_b"]].itertuples(index=False, name=None))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pairs = load_pairs(PAIRS_CSV)
    tickers = sorted(set(t for pair in pairs for t in pair))

    print(f"Loaded {len(pairs)} pairs ({len(tickers)} unique tickers) from {PAIRS_CSV}")
    print("Fetching daily price data for tickers...")
    t0 = time.time()
    data = extract_close(yf.download(tickers, period="1y", progress=False))
    log_step("Fetching daily price data", t0)

    results = []
    tradables = []

    print("Processing pairs...")
    t0 = time.time()
    for stockA, stockB in pairs:
        if stockA not in data.columns or stockB not in data.columns:
            print(f"Skipping pair {stockA}/{stockB} due to missing data.")
            continue

        ratio = data[stockA] / data[stockB]
        rolling_mean = ratio.rolling(window=126).mean()
        rolling_std = ratio.rolling(window=126).std()
        z_score = (ratio - rolling_mean) / rolling_std
        current_z = z_score.iloc[-1]

        y = data[stockA].dropna()
        x = data[stockB].dropna()
        combined = pd.concat([y, x], axis=1, join='inner').dropna()

        if combined.empty:
            print(f"Skipping pair {stockA}/{stockB} due to no common data after dropping NaNs.")
            continue

        y_series = combined[stockA]
        x_series = combined[stockB]

        # Engle-Granger cointegration test across three lookback windows
        _, p_3m, _ = coint(y_series.iloc[-63:], x_series.iloc[-63:])
        _, p_6m, _ = coint(y_series.iloc[-126:], x_series.iloc[-126:])
        _, p_1y, _ = coint(y_series, x_series)

        # Hedge ratio (beta) via OLS on price levels, multiple windows
        beta_1y = sm.OLS(y_series, sm.add_constant(x_series)).fit().params.iloc[1]
        beta_6m = sm.OLS(y_series.iloc[-126:], sm.add_constant(x_series.iloc[-126:])).fit().params.iloc[1]
        beta_3m = sm.OLS(y_series.iloc[-63:], sm.add_constant(x_series.iloc[-63:])).fit().params.iloc[1]

        # Hedge ratio via OLS on returns (used for trade sizing)
        beta_ret_1y, _ = hedge_ratio_from_returns(y_series, x_series)
        beta_ret_6m, _ = hedge_ratio_from_returns(y_series.iloc[-126:], x_series.iloc[-126:])
        beta_ret_3m, _ = hedge_ratio_from_returns(y_series.iloc[-63:], x_series.iloc[-63:])

        betalog, alphalog, spreadlog, _ = log_price_beta(y_series, x_series)
        zscore = (spreadlog - spreadlog.rolling(60).mean()) / spreadlog.rolling(60).std()
        current_spread_zs = zscore.iloc[-1]
        corr = pair_correlation(y_series, x_series)

        # Use 6-month price beta for spread calculations
        beta = beta_6m
        spread = y_series - (beta * x_series)
        spread_mean = spread.rolling(window=60).mean()
        spread_std = spread.rolling(window=60).std()
        spread_zscore = (spread - spread_mean) / spread_std
        current_spread_z = spread_zscore.iloc[-1]

        # --- Half-life (Ornstein-Uhlenbeck mean reversion speed) ---
        spread_lag = spread.shift(1)
        spread_diff = spread - spread_lag
        df_ou = pd.DataFrame({'lag': spread_lag, 'diff': spread_diff}).dropna()
        ou_model = sm.OLS(df_ou['diff'], sm.add_constant(df_ou['lag'])).fit()
        lambda_param = ou_model.params.iloc[1]

        if lambda_param < 0:
            half_life = round(-np.log(2) / lambda_param, 1)
        else:
            half_life = np.nan  # not mean-reverting (trending)

        results.append({
            "Pair": f"{stockA} / {stockB}",
            "Price Z-Score": round(current_z, 2),
            "log Z-Score": round(current_spread_zs, 2),
            "Spread Z-Score": round(current_spread_z, 2),
            "Signal": (
                "LONG A/SHORT B" if current_spread_z <= -2
                else "SHORT A/LONG B" if current_spread_z >= 2
                else "NEUTRAL"
            ),
            '3MPV': round(p_3m, 4),
            '6MPV': round(p_6m, 4),
            '1YPV': round(p_1y, 4),
            'PB(3M)': round(beta_3m, 4),
            'PB(6M)': round(beta_6m, 4),
            'PB(1Y)': round(beta_1y, 4),
            'RB(3M)': round(beta_ret_3m, 4),
            'RB(6M)': round(beta_ret_6m, 4),
            'RB(1Y)': round(beta_ret_1y, 4),
            'Half-Life (Days)': half_life if not np.isnan(half_life) else "N/A (Trending)",
            'Correlation (1y)': round(corr, 4),
            'Cointegrated (80%)': "YES" if p_1y < 0.20 else "NO",
            'Cointegrated (95%)': "YES" if p_1y < 0.05 else "NO",
        })

        if (p_1y <= 0.10) and (abs(current_spread_z) >= 2):
            name_a = get_company_name(stockA)
            name_b = get_company_name(stockB)
            tradables.append({
                "Ticker A": stockA,
                "Ticker B": stockB,
                "Pair": f"{name_a} ({stockA}) / {name_b} ({stockB})",
                "Signal": (
                    "LONG A / SHORT B" if current_spread_z <= -1.5
                    else "SHORT A / LONG B" if current_spread_z >= 1.5
                    else "NEUTRAL"
                ),
                "P-value": round(p_1y, 4),
                "Spread Z": round(current_spread_z, 2),
                'Return Beta (3M)': round(beta_ret_3m, 4),
                'Return Beta (6M)': round(beta_ret_6m, 4),
                'Return Beta (1Y)': round(beta_ret_1y, 4),
                'Half-Life (Days)': half_life if not np.isnan(half_life) else "N/A (Trending)",
                'Correlation (1y)': round(corr, 4),
                "Cointegrated": p_1y < 0.10,
            })

    log_step("Processing pairs", t0)

    # -----------------------------------------------------------------
    # Display + persist results
    # -----------------------------------------------------------------
    print("Displaying results...")
    t0 = time.time()
    df_results = pd.DataFrame(results)
    df_results.index = df_results.index + 1
    df_results.index.name = "Line #"
    df_results = df_results.reset_index()

    print("\n--- RESULTS ---")
    console_cols = [
        "Line #", "Pair", "Spread Z-Score", "Signal", "1YPV",
        "Half-Life (Days)", "Correlation (1y)", "Cointegrated (95%)",
    ]
    console_cols = [c for c in console_cols if c in df_results.columns]
    print(df_results[console_cols].to_string(index=False, max_rows=None, max_cols=None))

    df_tradables = pd.DataFrame(tradables)
    print("\n--- Tradables ---")
    if not df_tradables.empty:
        print(df_tradables.drop(columns=["Ticker A", "Ticker B"], errors="ignore")
              .sort_values("P-value")
              .to_string(index=False, max_rows=None, max_cols=None))
    else:
        print("(none)")

    os.makedirs(LOG_DIR, exist_ok=True)
    results_txt = os.path.join(LOG_DIR, f"pair_results_{datetime.now().strftime('%Y%m%d_%H%M')}.txt")
    with open(results_txt, "w", encoding="utf-8") as f:
        f.write(df_results.to_string(index=False, max_rows=None, max_cols=None))
    print(f"\nFull results saved: {results_txt}")
    log_step("Displaying results", t0)

    # -----------------------------------------------------------------
    # Persist tradables (appendable CSV)
    # -----------------------------------------------------------------
    if not df_tradables.empty:
        csv_file = os.path.join(LOG_DIR, "tradables.csv")
        if os.path.exists(csv_file):
            try:
                with open(csv_file, encoding="utf-8") as f:
                    first_line = f.readline()
                if "Scan Time" not in first_line:
                    legacy = csv_file.replace(".csv", "_legacy.csv")
                    if not os.path.exists(legacy):
                        os.replace(csv_file, legacy)
                        print(f"Old-format tradables.csv archived as {legacy}")
            except Exception as exc:
                print(f"Schema check skipped ({exc})")

        df_tradables.insert(0, "Scan Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        header = not os.path.exists(csv_file)
        df_tradables.to_csv(csv_file, mode="a", header=header, index=False)
        print(f"\nSaved {len(df_tradables)} tradables to {csv_file} (appended).")
    else:
        print("\nNo tradables found this run; nothing saved to CSV.")


if __name__ == "__main__":
    main()
