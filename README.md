# Research & Rebalancing Dashboard

An authenticated Streamlit Community Cloud app for reviewing and
researching Canadian ETFs with one of five LLM providers, and preparing conservative
whole-share limit orders. No trade is sent until the signed-in user selects it, accepts
the confirmation, and clicks **Execute Selected Trades**.

> Not financial advice. You're solely responsible for every trade.

## What's included

- Read-only Brokerage account, CAD balance, position, symbol, and quote access.
- Brokerage access-token refresh with the replacement refresh token retained in the
  current authenticated session.
- Editable target allocations and CAD cash reserve (default `$200`).
- Whole-share, CAD-listed (`.TO`) limit-order suggestions.
- A broker impact check for every selected order before the first order is submitted.
- DeepSeek, Zhipu/GLM, Mistral, Cohere, and SEA-LION model selection.
- Read-only LLM tools for portfolio data, quotes, yfinance history, allocation math,
  and Tavily web/news search.
- Cited search context, inline portfolio charts, and an in-session audit log.

## Important Questrade limitation

QT's official scope table currently describes `POST accounts/:id/orders` as a
**trade scope for partner developers only**. A personal API application may therefore
be able to read the account but receive `401` or `403` when validating or placing an
order. This app implements the documented limit-order endpoints, but it cannot grant
your token a scope that the Brokerage has not issued. Confirm API trading eligibility with
Questrade before relying on the execution feature.

Official references (using QT as an example):

- [QT API authorization and scopes](https://www.questrade.com/api/documentation)
- [QT order placement](https://www.questrade.com/api/documentation/rest-operations/order-calls/accounts-id-orders)

## Why a thin Brokerage client

The project uses a small `requests`-based client in `tfsa_dashboard/questrade.py`
instead of `questrade-api`, `qtrade`, or `qt-api`. At review time, `qtrade` was the most
recently published of the named mature candidates (0.6.1 in May 2025), but its published
API covers account and market-data reads rather than documented order placement.
`questrade-api` persists tokens under the user's home directory, which is a poor fit for
ephemeral Streamlit Cloud, and its PyPI release is much older. The official REST surface
needed here is small, and the thin client makes these safety-critical behaviours explicit:

- use the API server returned by Brokerage during token redemption;
- rotate the single-use refresh token in memory;
- refresh an expired token for read requests;
- never retry an order `POST` after a network timeout, because its outcome may be
  unknown; and
- force `orderType="Limit"`, positive whole shares, and `timeInForce="Day"`.

This avoids depending on a third-party wrapper's release cadence or hidden retry
behaviour and is easier to audit before deployment.

Wrapper references: [qtrade on PyPI](https://pypi.org/project/qtrade/) and
[questrade-api on PyPI](https://pypi.org/project/questrade-api/).

## Repository layout

```text
streamlit_app.py                 Streamlit UI and human confirmation flow
tfsa_dashboard/config.py        Provider defaults and shared system prompt
tfsa_dashboard/questrade.py     OAuth rotation and Questrade REST operations
tfsa_dashboard/rebalancer.py    Pure-Python target and order calculations
tfsa_dashboard/portfolio.py     Questrade payload normalization
tfsa_dashboard/llm.py           Provider-neutral LLM interface and tool loop
tfsa_dashboard/research_tools.py Read-only tools available to the assistant
tfsa_dashboard/market_data.py   yfinance historical prices
tfsa_dashboard/search.py        Tavily web/news search
tests/                           Safety-focused unit tests
.streamlit/secrets.toml.example Secrets structure with placeholders only
```

## Local setup

Use Python 3.11 (Python 3.10+ is supported):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Generate a password hash. Do not put the plain password in TOML or shell history on a
shared computer:

```bash
python -c "import getpass; from streamlit_authenticator.utilities.hasher import Hasher; print(Hasher.hash(getpass.getpass('Password: ')))"
```

Paste the resulting bcrypt value into
`auth.credentials.usernames.<username>.password`. Generate a random cookie key:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Then run:

```bash
streamlit run streamlit_app.py
```

## QT refresh token

1. Sign in to QT and open the API Centre / personal applications area.
2. Create or select a personal API application and generate a manual refresh token.
3. Copy the token immediately and place it under `[questrade]` in Streamlit secrets,
   or paste it into the password-style field in the app sidebar.
4. Click **Connect / Refresh Questrade**.

QT access tokens are short-lived. Redeeming a refresh token returns a replacement
refresh token; the app keeps that replacement only in `st.session_state`. Streamlit
Community Cloud storage is ephemeral, so an app restart can lose the latest replacement
and make the original secret unusable. If reconnecting fails, generate a new manual
refresh token in Questrade and paste it into the sidebar. Treat every Questrade token as
a high-privilege secret and revoke it from QT if exposure is suspected.

## LLM and search configuration

The requested model IDs are defaults. Provider catalogs change, so each section accepts
optional `model` and `base_url` overrides without a code deployment. Configure one or
more providers; only providers with non-empty keys appear in the model selector.

| Secret section | Default model | Protocol |
| --- | --- | --- |
| `deepseek` | `deepseek-v4-flash` | OpenAI-compatible |
| `glm` | `glm-4.7-flash` | OpenAI-compatible |
| `mistral` | `mistral-small-latest` | OpenAI-compatible |
| `cohere` | `command-a-plus-05-2026` | Cohere Chat v2 |
| `sealion` | `aisingapore/Gemma-SEA-LION-v4-27B-IT` | OpenAI-compatible |

SEA-LION's documentation notes that the requested Gemma model does not use automatic
OpenAI-style `tool_calls`. Its adapter therefore uses a bounded JSON tool-request
protocol and validates every requested name against the same read-only allowlist.
Native tool calls remain enabled for the other providers. The app also supplies an
optional fresh Tavily search as context, and never exposes an order-placement tool to
any model.

Tavily was selected because its basic search endpoint returns a compact title, URL, and
content snippet, supports finance/news topics, and costs one basic search credit. The
model is explicitly told to cite URLs and ignore instructions embedded in search text.
Add its key under `[tavily]`.

Provider references:

- [Z.ai chat and tool API](https://docs.z.ai/api-reference/llm/chat-completion)
- [Mistral chat API](https://docs.mistral.ai/api/)
- [Cohere Chat v2](https://docs.cohere.com/v2/reference/chat)
- [SEA-LION API](https://docs.sea-lion.ai/guides/inferencing/api)
- [Tavily Search API](https://docs.tavily.com/documentation/api-reference/endpoint/search)

## Deploy to Streamlit Community Cloud

1. Create a private GitHub repository and commit this project. Confirm that
   `.streamlit/secrets.toml` is not tracked.
2. In Streamlit Community Cloud, create an app from the repository and select
   `streamlit_app.py` as the entry point.
3. Choose Python 3.11 in Advanced settings.
4. Paste the completed TOML from `.streamlit/secrets.toml.example` into **App settings
   > Secrets**. Use real values only in that Cloud settings screen.
5. Deploy, sign in, connect Questrade, and verify account reads before testing any order.

When rotating a key, update it in Streamlit Cloud Secrets and reboot the app. Revoke the
old provider key or Questrade token at its issuer.

## Rebalancing behaviour

- Targets must total within `0.5%` of 100%; they are normalized after validation.
- Purchases require both a `.TO` symbol and a QT `CAD` currency result.
- The calculation uses the lower of available CAD cash above the reserve and reported
  CAD buying power.
- Expected sale proceeds never fund purchases in the same batch. Refresh after sells
  fill, then generate a new batch.
- A live ask is required for a buy and a live bid is required for a sell; the generator
  will not substitute a potentially stale last-trade price.
- Trades under `$50` or `0.5%` of account equity (whichever threshold is larger) are
  skipped to reduce churn.
- Existing non-`.TO` holdings appear in the overview but are never automatically sold.
- The app does not deposit money and cannot exceed contribution room through a deposit.
- Every execution batch refreshes balances and positions, validates quantities and cash,
  checks that quotes are within 10% of proposed limits, and calls QT's impact
  endpoint before submission.

## Tests

```bash
python -m compileall streamlit_app.py tfsa_dashboard tests
pytest -q
ruff check .
```

Before production use, also test with a non-trading/read-only token, a missing provider
key, an expired Questrade token, an invalid target total, an illiquid symbol, insufficient
cash, and a deliberately unselected order. Never test trade submission with a quantity
or symbol you are not prepared to place.

## Known limitations

- QT/Brokerage may restrict trade scope to approved partner applications.
- The newest rotating refresh token exists only in the active Streamlit session.
- QT-reported combined CAD equity is used as the allocation base; non-`.TO`
  positions are not sold by the generator.
- yfinance data can be delayed or incomplete and is research context, not an execution
  quote. QT quotes are used for proposed orders.
- Limit orders may not fill, may partially fill, and may incur commissions or ECN fees.
- The session audit log disappears when the session ends unless manually downloaded.
- No background or autonomous trading is implemented.
