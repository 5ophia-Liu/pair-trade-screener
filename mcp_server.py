# -*- coding: utf-8 -*-
"""
MCP Server for the Pairs Trading Cointegration Screener
---------------------------------------------------------
Exposes the existing screen_pair() quant logic as MCP tools, so an LLM
client (e.g. Claude Desktop) can call it in response to natural-language
questions like "Is KO/PEP a good pairs trade right now?"

Run locally with:
    mcp dev mcp_server.py

Or connect it to Claude Desktop as a local extension:
    1. Ensure manifest.json (in this repo) points to your venv's Python.
    2. Claude Desktop -> Settings -> Extensions -> Install unpacked extension
       -> select this project folder.
    3. Fully restart Claude Desktop.
"""

import os
from mcp.server.mcpserver import MCPServer

from analytics import screen_pair
from data import load_pairs, fetch_price_data

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PAIRS_CSV = os.path.join(SCRIPT_DIR, "pairs.csv")

mcp = MCPServer("pair-trade-screener")

# Sample Prompt: "Is KO/PEP a good pairs trade candidate right now?"
@mcp.tool()
def screen_pair_tool(ticker_a: str, ticker_b: str) -> dict:
    """
    Screen a single stock/ETF pair for statistical arbitrage potential.

    Runs Engle-Granger cointegration testing, OLS hedge-ratio regression,
    spread Z-score, and Ornstein-Uhlenbeck half-life estimation on the pair.

    Args:
        ticker_a: First ticker symbol (e.g. "KO")
        ticker_b: Second ticker symbol (e.g. "PEP")

    Returns:
        A dict with cointegration p-values, hedge ratios, spread Z-score,
        half-life estimate, correlation, and a tradable flag/signal.
    """
    data = fetch_price_data([ticker_a, ticker_b])
    result = screen_pair(ticker_a, ticker_b, data)

    if result is None:
        return {
            "error": f"Insufficient or missing data for {ticker_a}/{ticker_b}"
        }
    return result

# Sample Prompt: "Scan for any tradable pairs right now."
@mcp.tool()
def scan_universe_tool(max_results: int = 10) -> list[dict]:
    """
    Scan the full pairs.csv universe and return the top tradable candidates,
    ranked by cointegration p-value (most statistically significant first).

    Args:
        max_results: Maximum number of tradable candidates to return (default 10)

    Returns:
        A list of result dicts for pairs currently flagged as tradable
        (cointegrated with a stretched spread), sorted by p-value.
    """
    pairs = load_pairs(PAIRS_CSV)
    tickers = sorted(set(t for pair in pairs for t in pair))
    data = fetch_price_data(tickers)

    tradables = []
    for ticker_a, ticker_b in pairs:
        result = screen_pair(ticker_a, ticker_b, data)
        if result is not None and result["tradable"]:
            tradables.append(result)

    tradables.sort(key=lambda r: r["pvalue_1y"])
    return tradables[:max_results]


if __name__ == "__main__":
    mcp.run()