"""Read-only tools exposed to the research assistant."""

from __future__ import annotations

from typing import Any

from .market_data import history_as_records
from .questrade import QuestradeClient
from .search import TavilySearch, results_as_tool_payload

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "get_portfolio_snapshot",
            "description": "Get the connected TFSA balances and current positions. Read-only.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_quotes",
            "description": "Get current Questrade bid, ask, last price, and currency for symbols.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {
                        "type": "array",
                        "items": {"type": "string"},
                        "maxItems": 10,
                    }
                },
                "required": ["symbols"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_historical_prices",
            "description": "Get adjusted daily historical closes and volume from yfinance.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string"},
                    "period": {
                        "type": "string",
                        "enum": ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
                    },
                },
                "required": ["symbol"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": "Search recent web or Canadian financial news. Return URLs for citations.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "topic": {"type": "string", "enum": ["general", "news", "finance"]},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_current_allocation",
            "description": "Calculate current position weights using Questrade-reported market values.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


class ResearchTools:
    def __init__(
        self,
        *,
        broker: QuestradeClient | None,
        account_number: str | None,
        portfolio_snapshot: dict[str, Any],
        web_search: TavilySearch | None,
    ) -> None:
        self.broker = broker
        self.account_number = account_number
        self.snapshot = portfolio_snapshot
        self.web_search = web_search

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        handlers = {
            "get_portfolio_snapshot": self._portfolio,
            "get_quotes": self._quotes,
            "get_historical_prices": self._history,
            "search_web": self._search,
            "calculate_current_allocation": self._allocation,
        }
        if name not in handlers:
            raise ValueError(f"Unknown or disallowed research tool: {name}")
        return handlers[name](**arguments)

    def _portfolio(self) -> dict[str, Any]:
        if not self.snapshot:
            return {"status": "No Questrade account is connected."}
        return self.snapshot

    def _quotes(self, symbols: list[str]) -> dict[str, Any]:
        if not self.broker or not self.account_number:
            return {"status": "Connect Questrade to retrieve broker quotes."}
        if not isinstance(symbols, list) or not all(isinstance(item, str) for item in symbols):
            raise ValueError("symbols must be a list of ticker strings")
        normalized = list(dict.fromkeys(str(symbol).strip().upper() for symbol in symbols))[:10]
        if not normalized:
            raise ValueError("At least one symbol is required")
        resolved = [self.broker.resolve_symbol(symbol) for symbol in normalized]
        quote_rows = self.broker.get_quotes([int(item["symbolId"]) for item in resolved])
        metadata = {int(item["symbolId"]): item for item in resolved}
        return {
            "quotes": [
                {
                    "symbol": metadata.get(int(row.get("symbolId", 0)), {}).get("symbol"),
                    "symbol_id": row.get("symbolId"),
                    "currency": metadata.get(int(row.get("symbolId", 0)), {}).get("currency"),
                    "bid": row.get("bidPrice"),
                    "ask": row.get("askPrice"),
                    "last": row.get("lastTradePrice"),
                    "delay_seconds": row.get("delay"),
                }
                for row in quote_rows
            ]
        }

    def _history(self, symbol: str, period: str = "1y") -> dict[str, Any]:
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        return history_as_records(symbol, period)

    def _search(self, query: str, topic: str = "general") -> dict[str, Any]:
        if not self.web_search:
            return {"status": "Tavily search is not configured."}
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")
        return results_as_tool_payload(self.web_search.search(query[:500], topic=topic))

    def _allocation(self) -> dict[str, Any]:
        positions = self.snapshot.get("positions", [])
        total = sum(max(float(row.get("market_value", 0) or 0), 0) for row in positions)
        if total <= 0:
            return {"allocations": []}
        return {
            "allocations": [
                {
                    "symbol": row.get("symbol"),
                    "weight_percent": round(float(row.get("market_value", 0) or 0) / total * 100, 2),
                }
                for row in positions
            ]
        }
