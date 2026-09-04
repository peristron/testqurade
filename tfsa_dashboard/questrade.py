"""Small Questrade REST client with in-session refresh-token rotation."""

from __future__ import annotations

import threading
import time
from typing import Any

import requests

from .errors import BrokerError


class QuestradeClient:
    """Questrade API client designed to live in one authenticated Streamlit session."""

    TOKEN_URL = "https://login.questrade.com/oauth2/token"

    def __init__(
        self,
        refresh_token: str,
        *,
        access_token: str = "",
        api_server: str = "",
        expires_at: float = 0,
        timeout: float = 15,
        session: requests.Session | None = None,
    ) -> None:
        if not refresh_token.strip():
            raise BrokerError("A Questrade refresh token is required.")
        self.refresh_token = refresh_token.strip()
        self.access_token = access_token
        self.api_server = api_server.rstrip("/")
        self.expires_at = expires_at
        self.timeout = timeout
        self._http = session or requests.Session()
        self._refresh_lock = threading.Lock()

    @property
    def connected(self) -> bool:
        return bool(self.access_token and self.api_server and time.time() < self.expires_at)

    def connection_state(self) -> dict[str, Any]:
        """Return serializable state. Never log or display this dictionary."""
        return {
            "refresh_token": self.refresh_token,
            "access_token": self.access_token,
            "api_server": self.api_server,
            "expires_at": self.expires_at,
        }

    def refresh_access_token(self) -> None:
        """Redeem the current single-use refresh token and retain its replacement."""
        with self._refresh_lock:
            try:
                response = self._http.get(
                    self.TOKEN_URL,
                    params={"grant_type": "refresh_token", "refresh_token": self.refresh_token},
                    timeout=self.timeout,
                )
            except requests.RequestException as exc:
                raise BrokerError("Could not reach Questrade to refresh the session.") from exc
            payload = self._json_or_error(response, "Questrade token refresh")
            required = ("access_token", "refresh_token", "api_server", "expires_in")
            if any(key not in payload for key in required):
                raise BrokerError("Questrade returned an incomplete token response.")
            self.access_token = str(payload["access_token"])
            self.refresh_token = str(payload["refresh_token"])
            api_server = str(payload["api_server"]).rstrip("/")
            self.api_server = api_server if api_server.endswith("/v1") else f"{api_server}/v1"
            self.expires_at = time.time() + max(int(payload["expires_in"]) - 45, 1)

    def _ensure_token(self) -> None:
        if not self.connected:
            self.refresh_access_token()

    @staticmethod
    def _json_or_error(response: requests.Response, operation: str) -> dict[str, Any]:
        try:
            payload = response.json()
        except ValueError:
            payload = {}
        if not response.ok:
            message = payload.get("message") or payload.get("error_description") or response.reason
            if response.status_code in (401, 403):
                message = (
                    f"{message}. The token may be expired, revoked, or missing the required API scope."
                )
            raise BrokerError(f"{operation} failed ({response.status_code}): {message}")
        if not isinstance(payload, dict):
            raise BrokerError(f"{operation} returned an unexpected response.")
        return payload

    def _request(
        self,
        method: str,
        path: str,
        *,
        retry_auth: bool = True,
        **kwargs: Any,
    ) -> dict[str, Any]:
        self._ensure_token()
        url = f"{self.api_server}/{path.lstrip('/')}"
        headers = dict(kwargs.pop("headers", {}))
        headers["Authorization"] = f"Bearer {self.access_token}"
        try:
            response = self._http.request(
                method, url, headers=headers, timeout=self.timeout, **kwargs
            )
        except requests.RequestException as exc:
            verb = method.upper()
            if verb == "POST":
                raise BrokerError(
                    "The broker connection ended during a submission. No automatic retry was made; "
                    "check Questrade order history before trying again."
                ) from exc
            raise BrokerError("Could not reach the Questrade API.") from exc

        if response.status_code == 401 and retry_auth and method.upper() == "GET":
            self.refresh_access_token()
            return self._request(method, path, retry_auth=False, **kwargs)
        return self._json_or_error(response, f"Questrade {method.upper()} {path}")

    def get_accounts(self) -> list[dict[str, Any]]:
        return list(self._request("GET", "accounts").get("accounts", []))

    def get_balances(self, account_number: str) -> dict[str, Any]:
        return self._request("GET", f"accounts/{account_number}/balances")

    def get_positions(self, account_number: str) -> list[dict[str, Any]]:
        return list(
            self._request("GET", f"accounts/{account_number}/positions").get("positions", [])
        )

    def search_symbols(self, prefix: str, offset: int = 0) -> list[dict[str, Any]]:
        payload = self._request(
            "GET", "symbols/search", params={"prefix": prefix, "offset": offset}
        )
        return list(payload.get("symbols", []))

    def resolve_symbol(self, symbol: str) -> dict[str, Any]:
        normalized = symbol.strip().upper()
        if not normalized:
            raise BrokerError("A symbol is required.")
        matches = self.search_symbols(normalized)
        exact = next(
            (item for item in matches if str(item.get("symbol", "")).upper() == normalized),
            None,
        )
        if exact is None:
            raise BrokerError(f"Questrade could not resolve symbol {normalized}.")
        symbol_id = int(exact["symbolId"])
        detailed = self._request("GET", f"symbols/{symbol_id}").get("symbols", [])
        resolved = dict(exact)
        if detailed:
            resolved.update(detailed[0])
        detailed_currency = str(resolved.get("currency", "") or "").strip()
        search_currency = str(exact.get("currency", "") or "").strip()
        resolved["currency"] = (detailed_currency or search_currency).upper()
        if not resolved["currency"]:
            raise BrokerError(
                f"Questrade returned no currency for {normalized}; "
                "the symbol cannot be used safely for rebalancing."
            )
        return resolved

    def get_quotes(self, symbol_ids: list[int]) -> list[dict[str, Any]]:
        if not symbol_ids:
            return []
        joined = ",".join(str(int(item)) for item in symbol_ids)
        return list(self._request("GET", f"markets/quotes/{joined}").get("quotes", []))

    @staticmethod
    def _order_payload(
        account_number: str,
        symbol_id: int,
        action: str,
        quantity: int,
        limit_price: float,
    ) -> dict[str, Any]:
        normalized_action = action.title()
        if normalized_action not in {"Buy", "Sell"}:
            raise BrokerError("Only Buy and Sell actions are supported.")
        if isinstance(quantity, bool) or int(quantity) != quantity or int(quantity) <= 0:
            raise BrokerError("Order quantity must be a positive whole number.")
        if float(limit_price) <= 0:
            raise BrokerError("Limit price must be positive.")
        return {
            "accountNumber": str(account_number),
            "symbolId": int(symbol_id),
            "quantity": int(quantity),
            "icebergQuantity": 0,
            "limitPrice": round(float(limit_price), 4),
            "isAllOrNone": False,
            "isAnonymous": False,
            "orderType": "Limit",
            "timeInForce": "Day",
            "action": normalized_action,
            "primaryRoute": "AUTO",
            "secondaryRoute": "AUTO",
        }

    def preview_limit_order(
        self,
        account_number: str,
        symbol_id: int,
        action: str,
        quantity: int,
        limit_price: float,
    ) -> dict[str, Any]:
        payload = self._order_payload(
            account_number, symbol_id, action, quantity, limit_price
        )
        return self._request(
            "POST", f"accounts/{account_number}/orders/impact", json=payload, retry_auth=False
        )

    def place_limit_order(
        self,
        account_number: str,
        symbol_id: int,
        action: str,
        quantity: int,
        limit_price: float,
    ) -> dict[str, Any]:
        payload = self._order_payload(
            account_number, symbol_id, action, quantity, limit_price
        )
        return self._request(
            "POST", f"accounts/{account_number}/orders", json=payload, retry_auth=False
        )
