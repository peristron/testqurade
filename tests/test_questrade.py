import time

import pytest
import requests

from tfsa_dashboard.errors import BrokerError
from tfsa_dashboard.questrade import QuestradeClient


class FakeResponse:
    def __init__(self, status: int, payload: dict, reason: str = "") -> None:
        self.status_code = status
        self._payload = payload
        self.reason = reason
        self.ok = 200 <= status < 300

    def json(self) -> dict:
        return self._payload


class FakeSession:
    def __init__(self) -> None:
        self.token_requests = []
        self.requests = []

    def get(self, url, **kwargs):
        self.token_requests.append((url, kwargs))
        return FakeResponse(
            200,
            {
                "access_token": "access-2",
                "refresh_token": "refresh-2",
                "api_server": "https://api.example",
                "expires_in": 1800,
            },
        )

    def request(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        return FakeResponse(200, {"orderId": 123})


def test_refresh_rotates_token_without_logging_it() -> None:
    session = FakeSession()
    client = QuestradeClient("refresh-1", session=session)

    client.refresh_access_token()

    assert client.refresh_token == "refresh-2"
    assert client.access_token == "access-2"
    assert client.expires_at > time.time()

    assert len(session.token_requests) == 1
    url, request_kwargs = session.token_requests[0]
    assert url == QuestradeClient.TOKEN_URL
    assert request_kwargs["params"] == {
        "grant_type": "refresh_token",
        "refresh_token": "refresh-1",
    }


def test_order_payload_forces_limit_and_whole_shares() -> None:
    session = FakeSession()
    client = QuestradeClient(
        "refresh", access_token="access", api_server="https://api.example/v1", expires_at=time.time() + 60, session=session
    )
    client.place_limit_order("123", 42, "buy", 5, 10.25)
    payload = session.requests[0][2]["json"]
    assert payload["orderType"] == "Limit"
    assert payload["action"] == "Buy"
    assert payload["quantity"] == 5
    assert payload["timeInForce"] == "Day"


@pytest.mark.parametrize("quantity", [0, -1, 1.5, True])
def test_order_rejects_invalid_quantity(quantity) -> None:
    with pytest.raises(BrokerError, match="positive whole number"):
        QuestradeClient._order_payload("123", 42, "Buy", quantity, 10.0)


class TimeoutSession(FakeSession):
    def request(self, method, url, **kwargs):
        raise requests.Timeout("unknown submission state")


def test_post_timeout_is_not_retried() -> None:
    session = TimeoutSession()
    client = QuestradeClient(
        "refresh", access_token="access", api_server="https://api.example/v1", expires_at=time.time() + 60, session=session
    )
    with pytest.raises(BrokerError, match="No automatic retry"):
        client.place_limit_order("123", 42, "Buy", 5, 10.0)
