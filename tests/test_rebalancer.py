import pytest

from tfsa_dashboard.errors import RebalanceError
from tfsa_dashboard.rebalancer import (
    Holding,
    Quote,
    SuggestedOrder,
    generate_rebalance_orders,
    normalize_targets,
    validate_execution_batch,
)


def quote(symbol: str, price: float, currency: str = "CAD", symbol_id: int = 1) -> Quote:
    return Quote(symbol, symbol_id, currency, price - 0.01, price + 0.01, price)


def test_normalize_targets_accepts_near_100_and_normalizes() -> None:
    result = normalize_targets(
        [
            {"Symbol": "XIC.TO", "Target Weight %": 60},
            {"Symbol": "ZAG.TO", "Target Weight %": 40.2},
        ]
    )
    assert sum(result.values()) == pytest.approx(1.0)


def test_normalize_targets_rejects_us_listing() -> None:
    with pytest.raises(RebalanceError, match="not a .TO symbol"):
        normalize_targets([{"Symbol": "VTI", "Target Weight %": 100}])


def test_buys_are_whole_shares_and_preserve_cash_reserve() -> None:
    orders = generate_rebalance_orders(
        targets={"XIC.TO": 1.0},
        holdings={},
        quotes={"XIC.TO": quote("XIC.TO", 40.0)},
        cash_cad=1_000,
        cash_reserve=200,
        buying_power_cad=1_000,
        total_equity_cad=1_000,
    )
    assert len(orders) == 1
    assert orders[0].action == "Buy"
    assert isinstance(orders[0].quantity, int)
    assert orders[0].estimated_value <= 800


def test_non_cad_quote_blocks_purchase() -> None:
    with pytest.raises(RebalanceError, match="Blocked non-CAD purchase"):
        generate_rebalance_orders(
            targets={"XIC.TO": 1.0},
            holdings={},
            quotes={"XIC.TO": quote("XIC.TO", 40.0, currency="USD")},
            cash_cad=1_000,
            cash_reserve=200,
            buying_power_cad=1_000,
            total_equity_cad=1_000,
        )


def test_sale_proceeds_do_not_fund_same_batch_buys() -> None:
    orders = generate_rebalance_orders(
        targets={"XIC.TO": 1.0},
        holdings={"OLD.TO": Holding("OLD.TO", 10, 1_000, 100)},
        quotes={
            "OLD.TO": quote("OLD.TO", 100, symbol_id=1),
            "XIC.TO": quote("XIC.TO", 40, symbol_id=2),
        },
        cash_cad=200,
        cash_reserve=200,
        buying_power_cad=200,
        total_equity_cad=1_200,
    )
    assert any(order.action == "Sell" for order in orders)
    assert not any(order.action == "Buy" for order in orders)


def test_validate_batch_rejects_reserve_breach() -> None:
    order = SuggestedOrder(False, "Buy", "XIC.TO", 1, 20, 40.0, 800, "Below target")
    with pytest.raises(RebalanceError, match="cash reserve"):
        validate_execution_batch(
            [order], holdings={}, cash_cad=900, cash_reserve=200, buying_power_cad=2_000
        )


def test_validate_batch_rejects_oversell() -> None:
    order = SuggestedOrder(False, "Sell", "XIC.TO", 1, 11, 40.0, 440, "Above target")
    holdings = {"XIC.TO": Holding("XIC.TO", 10, 400, 40)}
    with pytest.raises(RebalanceError, match="current holding is 10"):
        validate_execution_batch(
            [order], holdings=holdings, cash_cad=500, cash_reserve=200, buying_power_cad=500
        )
