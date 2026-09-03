"""Application-specific exceptions with user-safe messages."""


class DashboardError(Exception):
    """Base error that is safe to show in the Streamlit interface."""


class ConfigurationError(DashboardError):
    """Required configuration is missing or invalid."""


class BrokerError(DashboardError):
    """A Questrade operation failed."""


class ProviderError(DashboardError):
    """An LLM provider request failed."""


class SearchError(DashboardError):
    """A web search request failed."""


class RebalanceError(DashboardError):
    """A target or proposed order violates a portfolio rule."""
