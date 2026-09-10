import pandas as pd
import yfinance as yf

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

def fetch_price_data(tickers, period="1y"):
    """
    Downloads daily close prices for a list of tickers from Yahoo Finance.
    Returns a DataFrame with one column per ticker.
    """
    raw = yf.download(tickers, period=period, progress=False)
    return extract_close(raw)