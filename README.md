# Pairs Trading Cointegration Screener

This project explores quantitative tools that can be queried in natural language using AI.
The underlying logic is a Python screener that scans a list of stock/ETF pairs for
statistical arbitrage candidates using cointegration testing, hedge-ratio regression, and
mean-reversion speed estimation. It's available both as a CLI tool and as an MCP server,
so the same analysis can be queried in natural language through an AI client like Claude
Desktop.

## Demo

https://github.com/user-attachments/assets/d811314d-1ee6-4cd8-a8c3-efbbf56835eb

## What it does

For each pair (e.g. `KO` / `PEP`), the screener computes:

- **Engle-Granger cointegration test** (3-month, 6-month, and 1-year windows) — tests
  whether the two price series share a stable long-run relationship
- **Hedge ratio (beta)** via OLS regression, computed two ways:
  - on price levels (for spread construction)
  - on daily returns (for trade sizing)
- **Spread Z-score** — how far the current price spread has drifted from its 60-day
  rolling mean, in standard deviations
- **Half-life of mean reversion** — via an Ornstein-Uhlenbeck fit, estimating how many
  trading days it typically takes the spread to revert halfway back to its mean
- **Correlation** of daily returns over the trailing year

Pairs that are cointegrated (p-value ≤ 0.10) and currently stretched (|Z-score| ≥ 2) are
flagged as "tradable" candidates and written to `logfiles/tradables.csv`.

## Project structure

| File | Purpose |
|---|---|
| `data.py` | Fetches price data from Yahoo Finance based on the provided pairs list |
| `analytics.py` | Computes cointegration tests, beta, Z-score, half-life, and correlation for a given pair |
| `pair_screener.py` | CLI entry point — runs the full universe and prints/saves results |
| `mcp_server.py` | MCP server exposing `screen_pair()` as callable tools for AI clients like Claude Desktop |
| `manifest.json.example` | Template for connecting the MCP server to Claude Desktop as a local extension |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage: CLI

Edit `pairs.csv` to add or remove pairs (two columns: `ticker_a`, `ticker_b`), then run:

```bash
python pair_screener.py
```

**Output:**
- A console summary table of every screened pair
- A console table of "tradable" candidates
- `logfiles/pair_results_<timestamp>.txt` — full results for this run
- `logfiles/tradables.csv` — appended log of every tradable signal found across runs

## Usage: AI integration (MCP server)

This project exposes the `screen_pair()` logic as an MCP (Model Context Protocol) server,
so an AI client can answer questions like *"Is KO/PEP a good pairs trade candidate right
now?"* or *"Scan for any tradable pairs right now."* using the real analysis underneath,
not a guess.

**Tools exposed:**
- `screen_pair_tool(ticker_a, ticker_b)` — full analysis on a single pair
- `scan_universe_tool(max_results=10)` — scans `pairs.csv` and returns the top tradable
  candidates, ranked by cointegration p-value

**Test it standalone** with the MCP Inspector:
```bash
pip install "mcp[cli]"
mcp dev mcp_server.py
```

**Connect it to Claude Desktop** as a local extension:

1. Copy the manifest template and fill in your own venv path:
```bash
   cp manifest.json.example manifest.json
```
   Edit `manifest.json` and replace the placeholder command with the full path to your
   venv's Python, e.g. `/Users/yourname/Projects/pair-trade-screener/.venv/bin/python3`.
2. In Claude Desktop: **Settings → Extensions → Install unpacked extension**, then select
   this project folder.
3. Fully quit and reopen Claude Desktop (Cmd+Q, not just closing the window).
4. Start a new chat and ask a question — Claude will call the tool and return real,
   data-backed results.

`manifest.json` is gitignored since it contains a machine-specific path; only the
`.example` template is committed.

## Note

- Price data is pulled live from Yahoo Finance via `yfinance`. Results will vary run to
  run as new market data comes in.
- The MCP server runs locally via stdio — no data leaves the local machine except the
  existing Yahoo Finance price requests.
