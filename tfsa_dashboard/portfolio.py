"""Normalize Questrade account payloads for display and core calculations."""

from __future__ import annotations

from typing import Any

from .errors import BrokerError
from .questrade import QuestradeClient
from .rebalancer import Holding, Quote


def _number(value: Any) -> float:
    try:
        return float(value or 0)
    except (TypeError, ValueError):
        return 0.0


def summarize_balances(payload: dict[str, Any]) -> dict[str, float]:
    per_currency = payload.get("perCurrencyBalances", [])
    combined = payload.get("combinedBalances", [])
    cad = next(
        (row for row in per_currency if str(row.get("currency", "")).upper() == "CAD"), {}
    )
    combined_cad = next(
        (row for row in combined if str(row.get("currency", "")).upper() == "CAD"), {}
    )
    return {
        "cash_cad": _number(cad.get("cash")),
        "total_equity_cad": _number(combined_cad.get("totalEquity", cad.get("totalEquity"))),
        "buying_power_cad": _number(combined_cad.get("buyingPower", cad.get("buyingPower"))),
        "maintenance_excess_cad": _number(
            combined_cad.get("maintenanceExcess", cad.get("maintenanceExcess"))
        ),
    }


def normalize_positions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "symbol": str(row.get("symbol", "")).upper(),
            "quantity": int(_number(row.get("openQuantity"))),
            "average_entry_price": _number(row.get("averageEntryPrice")),
            "current_price": _number(row.get("currentPrice")),
            "market_value": _number(row.get("currentMarketValue")),
            "open_pnl": _number(row.get("openPnl")),
        }
        for row in rows
        if row.get("symbol")
    ]


def holdings_for_rebalancer(positions: list[dict[str, Any]]) -> dict[str, Holding]:
    """Include only Toronto-listed holdings in automated allocation calculations."""
    return {
        row["symbol"]: Holding(
            symbol=row["symbol"],
            quantity=int(row["quantity"]),
            market_value=float(row["market_value"]),
            current_price=float(row["current_price"]),
            currency="CAD",
        )
        for row in positions
        if str(row.get("symbol", "")).endswith(".TO")
    }


def load_quotes(broker: QuestradeClient, symbols: list[str]) -> dict[str, Quote]:
    normalized = list(dict.fromkeys(symbol.strip().upper() for symbol in symbols if symbol.strip()))
    resolved = [broker.resolve_symbol(symbol) for symbol in normalized]
    if any(not item.get("isQuotable", True) for item in resolved):
        blocked = [str(item.get("symbol")) for item in resolved if not item.get("isQuotable", True)]
        raise BrokerError(f"These symbols are not currently quotable: {', '.join(blocked)}.")
    raw_quotes = broker.get_quotes([int(item["symbolId"]) for item in resolved])
    by_id = {int(row.get("symbolId", 0)): row for row in raw_quotes}
    result: dict[str, Quote] = {}
    for item in resolved:
        symbol_id = int(item["symbolId"])
        raw = by_id.get(symbol_id)
        if not raw:
            raise BrokerError(f"Questrade returned no quote for {item['symbol']}.")
        result[str(item["symbol"]).upper()] = Quote(
            symbol=str(item["symbol"]).upper(),
            symbol_id=symbol_id,
            currency=str(item.get("currency", "")).upper(),
            bid=_number(raw.get("bidPrice")),
            ask=_number(raw.get("askPrice")),
            last=_number(raw.get("lastTradePrice")),
        )
    return result
