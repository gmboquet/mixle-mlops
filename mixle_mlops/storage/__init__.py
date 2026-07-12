"""Persistence entry points for relational state and artifact storage.

This package owns database engine/session helpers and object-store integration
used by accounts, registry records, conversations, feedback, media, and
artifacts. Route code should treat storage configuration as an operational
dependency and document whether a release check used in-memory, SQLite,
Redis/object-store, or production-like backends.
"""
from .db import get_engine, get_session, init_db

__all__ = ["get_engine", "get_session", "init_db"]
