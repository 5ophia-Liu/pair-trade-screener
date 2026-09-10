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
import time
import os
from datetime import datetime
from analytics import screen_pair
from data import load_pairs, fetch_price_data, get_company_name

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


def result_to_display_row(result):
    """Convert a screen_pair() result dict into the console/summary row format."""
    return {
        "Pair": f"{result['ticker_a']} / {result['ticker_b']}",
        "Price Z-Score": result["price_z_score"],
        "log Z-Score": result["log_z_score"],
        "Spread Z-Score": result["spread_z_score"],
        "Signal": result["signal"],
        '3MPV': result["pvalue_3m"],
        '6MPV': result["pvalue_6m"],
        '1YPV': result["pvalue_1y"],
        'PB(3M)': result["price_beta_3m"],
        'PB(6M)': result["price_beta_6m"],
        'PB(1Y)': result["price_beta_1y"],
        'RB(3M)': result["return_beta_3m"],
        'RB(6M)': result["return_beta_6m"],
        'RB(1Y)': result["return_beta_1y"],
        'Half-Life (Days)': result["half_life_days"] if result["half_life_days"] is not None else "N/A (Trending)",
        'Correlation (1y)': result["correlation_1y"],
        'Cointegrated (80%)': "YES" if result["cointegrated_80"] else "NO",
        'Cointegrated (95%)': "YES" if result["cointegrated_95"] else "NO",
    }


def result_to_tradable_row(result):
    """Convert a tradable screen_pair() result dict into the tradables table format."""
    name_a = get_company_name(result["ticker_a"])
    name_b = get_company_name(result["ticker_b"])
    spread_z = result["spread_z_score"]
    return {
        "Ticker A": result["ticker_a"],
        "Ticker B": result["ticker_b"],
        "Pair": f"{name_a} ({result['ticker_a']}) / {name_b} ({result['ticker_b']})",
        "Signal": (
            "LONG A / SHORT B" if spread_z <= -1.5
            else "SHORT A / LONG B" if spread_z >= 1.5
            else "NEUTRAL"
        ),
        "P-value": result["pvalue_1y"],
        "Spread Z": spread_z,
        'Return Beta (3M)': result["return_beta_3m"],
        'Return Beta (6M)': result["return_beta_6m"],
        'Return Beta (1Y)': result["return_beta_1y"],
        'Half-Life (Days)': result["half_life_days"] if result["half_life_days"] is not None else "N/A (Trending)",
        'Correlation (1y)': result["correlation_1y"],
        "Cointegrated": result["pvalue_1y"] < 0.10,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    pairs = load_pairs(PAIRS_CSV)
    tickers = sorted(set(t for pair in pairs for t in pair))

    print(f"Loaded {len(pairs)} pairs ({len(tickers)} unique tickers) from {PAIRS_CSV}")
    print("Fetching daily price data for tickers...")
    t0 = time.time()
    data = fetch_price_data(tickers)
    log_step("Fetching daily price data", t0)

    results = []
    tradables = []

    print("Processing pairs...")
    t0 = time.time()
    for stockA, stockB in pairs:
        result = screen_pair(stockA, stockB, data)
        if result is None:
            print(f"Skipping pair {stockA}/{stockB} due to missing/insufficient data.")
            continue

        results.append(result_to_display_row(result))

        if result["tradable"]:
            tradables.append(result_to_tradable_row(result))
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