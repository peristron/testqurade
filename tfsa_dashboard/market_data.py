"""Historical market-data helpers isolated from the Streamlit interface."""

from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf

from .errors import DashboardError

ALLOWED_PERIODS = {"1mo", "3mo", "6mo", "1y", "2y", "5y"}


def historical_prices(symbol: str, period: str = "1y") -> pd.DataFrame:
    normalized = symbol.strip().upper()
    if not normalized:
        raise DashboardError("A symbol is required for historical data.")
    safe_period = period if period in ALLOWED_PERIODS else "1y"
    try:
        frame = yf.Ticker(normalized).history(period=safe_period, auto_adjust=True)
    except Exception as exc:
        raise DashboardError(f"Historical data could not be loaded for {normalized}.") from exc
    if frame.empty or "Close" not in frame:
        raise DashboardError(f"No historical prices were returned for {normalized}.")
    result = frame.reset_index()
    date_column = "Date" if "Date" in result else result.columns[0]
    result[date_column] = pd.to_datetime(result[date_column], utc=True).dt.strftime("%Y-%m-%d")
    return result[[date_column, "Close", "Volume"]].rename(columns={date_column: "Date"})


def history_as_records(symbol: str, period: str = "1y", limit: int = 180) -> dict[str, Any]:
    frame = historical_prices(symbol, period)
    sampled = frame.tail(max(1, min(int(limit), 365))).copy()
    sampled["Close"] = sampled["Close"].round(4)
    return {"symbol": symbol.upper(), "period": period, "prices": sampled.to_dict("records")}
