"""Account, API-key, and authentication exports.

The accounts package groups user records, API-key records, security helpers,
and service functions used by gateway routes. It should keep local/test
credential behavior explicit and avoid making production authentication
assumptions from examples or in-memory smoke checks.
"""
from . import security, service
from .models import ApiKey, User
from .service import AccountError

__all__ = ["User", "ApiKey", "service", "security", "AccountError"]
