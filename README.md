# Pairs Trading Cointegration Screener

A Python screener that scans a list of stock/ETF pairs for statistical arbitrage
candidates using cointegration testing, hedge-ratio regression, and mean-reversion
speed estimation.

## What it does

For each pair (e.g. `KO` / `PEP`), the screener computes:

- **Engle-Granger cointegration test** (3-month, 6-month, and 1-year windows) —
  tests whether the two price series share a stable long-run relationship
- **Hedge ratio (beta)** via OLS regression, computed two ways:
  - on price levels (for spread construction)
  - on daily returns (for trade sizing)
- **Spread Z-score** — how far the current price spread has drifted from its
  60-day rolling mean, in standard deviations
- **Half-life of mean reversion** — via an Ornstein-Uhlenbeck fit, estimating
  how many trading days it typically takes the spread to revert halfway back
  to its mean
- **Correlation** of daily returns over the trailing year

Pairs that are cointegrated (p-value ≤ 0.10) and currently stretched
(|Z-score| ≥ 2) are flagged as "tradable" candidates and written to
`logfiles/tradables.csv`.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

`pairs.csv` can be edited to add or remove pairs (two columns: `ticker_a`, `ticker_b`),
then run:

```bash
python pair_screener.py
```

Output:
- A console summary table of every screened pair
- A console table of "tradable" candidates
- `logfiles/pair_results_<timestamp>.txt` — full results for this run
- `logfiles/tradables.csv` — appended log of every tradable signal found across runs

## Note

- Price data is pulled live from Yahoo Finance via `yfinance`. Results will
  vary run to run as new market data comes in.
