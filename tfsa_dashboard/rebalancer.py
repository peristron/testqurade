"""Pure-Python portfolio allocation and order safety logic."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

from .errors import RebalanceError


@dataclass(frozen=True)
class Quote:
    symbol: str
    symbol_id: int
    currency: str
    bid: float
    ask: float
    last: float

    @property
    def buy_reference(self) -> float:
        return self.ask if self.ask > 0 else self.last

    @property
    def sell_reference(self) -> float:
        return self.bid if self.bid > 0 else self.last


@dataclass(frozen=True)
class Holding:
    symbol: str
    quantity: int
    market_value: float
    current_price: float
    currency: str = "CAD"


@dataclass(frozen=True)
class SuggestedOrder:
    execute: bool
    action: str
    symbol: str
    symbol_id: int
    quantity: int
    limit_price: float
    estimated_value: float
    reason: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def is_cad_listed(symbol: str, currency: str = "CAD") -> bool:
    """Apply the project's deliberately strict Canadian-listing policy."""
    return symbol.strip().upper().endswith(".TO") and currency.strip().upper() == "CAD"


def normalize_targets(rows: Iterable[dict[str, Any]], tolerance: float = 0.5) -> dict[str, float]:
    """Validate editable target rows and return decimal weights."""
    targets: dict[str, float] = {}
    for row in rows:
        symbol = str(row.get("Symbol", "")).strip().upper()
        raw_weight = row.get("Target Weight %", 0)
        if not symbol and (raw_weight is None or float(raw_weight or 0) == 0):
            continue
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise RebalanceError(f"Target weight for {symbol or 'a blank row'} is not numeric.") from exc
        if not symbol:
            raise RebalanceError("Every non-zero target row needs a symbol.")
        if symbol in targets:
            raise RebalanceError(f"Duplicate target symbol: {symbol}.")
        if weight < 0:
            raise RebalanceError(f"Target weight for {symbol} cannot be negative.")
        if weight == 0:
            continue
        if not symbol.endswith(".TO"):
            raise RebalanceError(
                f"{symbol} is not a .TO symbol. New purchases are restricted to CAD-listed securities."
            )
        targets[symbol] = weight / 100.0
    if not targets:
        raise RebalanceError("Add at least one positive target weight.")
    total_percent = sum(targets.values()) * 100
    if abs(total_percent - 100.0) > tolerance:
        raise RebalanceError(
            f"Target weights total {total_percent:.2f}%. They must be within {tolerance:.1f}% of 100%."
        )
    total = sum(targets.values())
    return {symbol: weight / total for symbol, weight in targets.items()}


def _price_to_cents(price: float) -> float:
    return round(float(price) + 1e-9, 2)


def generate_rebalance_orders(
    *,
    targets: dict[str, float],
    holdings: dict[str, Holding],
    quotes: dict[str, Quote],
    cash_cad: float,
    cash_reserve: float,
    buying_power_cad: float,
    total_equity_cad: float,
    min_trade_value: float = 50.0,
) -> list[SuggestedOrder]:
    """Create whole-share CAD limit orders without relying on unsettled sale proceeds."""
    if cash_reserve < 0:
        raise RebalanceError("Cash reserve cannot be negative.")
    if total_equity_cad <= 0:
        raise RebalanceError("Total equity must be positive before rebalancing.")
    if cash_reserve >= total_equity_cad:
        raise RebalanceError("Cash reserve must be lower than total account equity.")
    if abs(sum(targets.values()) - 1.0) > 0.001:
        raise RebalanceError("Normalized target weights must sum to 100%.")

    investable_equity = max(total_equity_cad - cash_reserve, 0.0)
    threshold = max(float(min_trade_value), total_equity_cad * 0.005)
    all_symbols = set(targets) | {
        symbol for symbol, holding in holdings.items() if is_cad_listed(symbol, holding.currency)
    }
    missing_quotes = sorted(symbol for symbol in all_symbols if symbol not in quotes)
    if missing_quotes:
        raise RebalanceError(f"No usable quote was found for: {', '.join(missing_quotes)}.")

    orders: list[SuggestedOrder] = []
    current_values = {symbol: holdings.get(symbol, Holding(symbol, 0, 0, 0)).market_value for symbol in all_symbols}
    desired_values = {symbol: investable_equity * targets.get(symbol, 0.0) for symbol in all_symbols}

    for symbol in sorted(all_symbols):
        excess = current_values[symbol] - desired_values[symbol]
        if excess < threshold:
            continue
        holding = holdings.get(symbol)
        quote = quotes[symbol]
        if quote.bid <= 0:
            raise RebalanceError(f"{symbol} has no live bid; no sell order was created.")
        limit_price = _price_to_cents(quote.bid)
        if not holding or holding.quantity <= 0:
            continue
        quantity = min(holding.quantity, math.floor(excess / limit_price))
        if quantity <= 0 or quantity * limit_price < min_trade_value:
            continue
        orders.append(
            SuggestedOrder(
                execute=False,
                action="Sell",
                symbol=symbol,
                symbol_id=quote.symbol_id,
                quantity=quantity,
                limit_price=limit_price,
                estimated_value=round(quantity * limit_price, 2),
                reason="Above target allocation",
            )
        )

    # Sale proceeds are intentionally excluded. The user can regenerate after fills.
    buy_budget = min(max(cash_cad - cash_reserve, 0.0), max(buying_power_cad, 0.0))
    deficits = sorted(
        (
            (desired_values[symbol] - current_values[symbol], symbol)
            for symbol in targets
            if desired_values[symbol] - current_values[symbol] >= threshold
        ),
        reverse=True,
    )
    for shortfall, symbol in deficits:
        if buy_budget < min_trade_value:
            break
        quote = quotes[symbol]
        if not is_cad_listed(symbol, quote.currency):
            raise RebalanceError(f"Blocked non-CAD purchase: {symbol} ({quote.currency}).")
        if quote.ask <= 0:
            raise RebalanceError(f"{symbol} has no live ask; no buy order was created.")
        limit_price = _price_to_cents(quote.ask)
        quantity = math.floor(min(shortfall, buy_budget) / limit_price)
        if quantity <= 0 or quantity * limit_price < min_trade_value:
            continue
        estimated = round(quantity * limit_price, 2)
        orders.append(
            SuggestedOrder(
                execute=False,
                action="Buy",
                symbol=symbol,
                symbol_id=quote.symbol_id,
                quantity=quantity,
                limit_price=limit_price,
                estimated_value=estimated,
                reason="Below target allocation",
            )
        )
        buy_budget -= estimated
    return orders


def validate_execution_batch(
    orders: Iterable[SuggestedOrder],
    *,
    holdings: dict[str, Holding],
    cash_cad: float,
    cash_reserve: float,
    buying_power_cad: float,
) -> None:
    """Re-check core invariants immediately before broker submission."""
    selected = list(orders)
    if not selected:
        raise RebalanceError("Select at least one order to execute.")
    seen: set[tuple[str, str]] = set()
    total_buys = 0.0
    sell_quantities: dict[str, int] = {}
    for order in selected:
        key = (order.action, order.symbol)
        if key in seen:
            raise RebalanceError(f"Duplicate {order.action.lower()} order for {order.symbol}.")
        seen.add(key)
        if order.action not in {"Buy", "Sell"}:
            raise RebalanceError("Only Buy and Sell actions can be submitted.")
        if isinstance(order.quantity, bool) or order.quantity <= 0 or int(order.quantity) != order.quantity:
            raise RebalanceError(f"{order.symbol} quantity must be a positive whole number.")
        if order.limit_price <= 0:
            raise RebalanceError(f"{order.symbol} must have a positive limit price.")
        if order.action == "Buy":
            if not is_cad_listed(order.symbol, "CAD"):
                raise RebalanceError(f"Blocked non-CAD purchase: {order.symbol}.")
            total_buys += order.quantity * order.limit_price
        else:
            sell_quantities[order.symbol] = sell_quantities.get(order.symbol, 0) + order.quantity
    for symbol, quantity in sell_quantities.items():
        available = holdings.get(symbol, Holding(symbol, 0, 0, 0)).quantity
        if quantity > available:
            raise RebalanceError(f"Cannot sell {quantity} {symbol}; current holding is {available}.")
    if total_buys > max(buying_power_cad, 0.0) + 0.01:
        raise RebalanceError("Selected purchases exceed current CAD buying power.")
    if cash_cad - total_buys < cash_reserve - 0.01:
        raise RebalanceError(
            "Selected purchases would breach the CAD cash reserve. Sale proceeds are not assumed."
        )
