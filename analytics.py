import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

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


def screen_pair(stockA, stockB, data):
    """
    Screens a pair for cointegration and mean-reversion signal.
    `data` is the price DataFrame (Close prices, one column per ticker).
    Returns a dict of results, or None if data is missing/insufficient.
    """
    if stockA not in data.columns or stockB not in data.columns:
        return None
    
    ratio = data[stockA] / data[stockB]
    rolling_mean = ratio.rolling(window=126).mean()
    rolling_std = ratio.rolling(window=126).std()
    z_score = (ratio - rolling_mean) / rolling_std
    current_z = z_score.iloc[-1]

    y = data[stockA].dropna()
    x = data[stockB].dropna()
    combined = pd.concat([y, x], axis=1, join='inner').dropna()

    # Skipping pair {stockA}/{stockB} due to no common data after dropping NaNs.
    if combined.empty: return None

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

    tradable = bool((p_1y <= 0.10) and (abs(current_spread_z) >= 2))

    return {
            "ticker_a": stockA,
            "ticker_b": stockB,
            "price_z_score": round(current_z, 2),
            "log_z_score": round(current_spread_zs, 2),
            "spread_z_score": round(current_spread_z, 2),
            "signal": (
                "LONG A/SHORT B" if current_spread_z <= -2
                else "SHORT A/LONG B" if current_spread_z >= 2
                else "NEUTRAL"
            ),
            "pvalue_3m": round(p_3m, 4),
            "pvalue_6m": round(p_6m, 4),
            "pvalue_1y": round(p_1y, 4),
            "price_beta_3m": round(beta_3m, 4),
            "price_beta_6m": round(beta_6m, 4),
            "price_beta_1y": round(beta_1y, 4),
            "return_beta_3m": round(beta_ret_3m, 4),
            "return_beta_6m": round(beta_ret_6m, 4),
            "return_beta_1y": round(beta_ret_1y, 4),
            "half_life_days": half_life,
            "correlation_1y": round(corr, 4),
            "cointegrated_80": p_1y < 0.20,
            "cointegrated_95": p_1y < 0.05,
            "tradable": tradable,
        }