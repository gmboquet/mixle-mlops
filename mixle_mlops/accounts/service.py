"""Account and API-key service operations for gateway authentication.

The service layer owns user creation, password authentication, API-key minting,
lookup, listing, and revocation while keeping persistence details behind the
SQLModel session boundary.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlmodel import Session, select

from . import security
from .models import ApiKey, User


class AccountError(Exception):
    pass


def create_user(
    session: Session, email: str, password: str | None = None, *, is_admin: bool = False
) -> User:
    if session.exec(select(User).where(User.email == email)).first():
        raise AccountError("email already registered")
    hashed = security.hash_password(password) if password else None
    user = User(email=email, hashed_password=hashed, is_admin=is_admin)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def authenticate(session: Session, email: str, password: str) -> User:
    user = session.exec(select(User).where(User.email == email)).first()
    if not user or not security.verify_password(password, user.hashed_password):
        raise AccountError("invalid email or password")
    if not user.is_active:
        raise AccountError("account is disabled")
    return user


def create_api_key(session: Session, user: User, *, name: str = "default", kind: str = "api") -> tuple[ApiKey, str]:
    raw, prefix, hashed = security.generate_api_key()
    key = ApiKey(user_id=user.id, name=name, prefix=prefix, hashed_key=hashed, kind=kind)
    session.add(key)
    session.commit()
    session.refresh(key)
    return key, raw


def resolve_api_key(session: Session, raw: str) -> tuple[User, ApiKey] | None:
    key = session.exec(
        select(ApiKey).where(ApiKey.hashed_key == security.hash_key(raw), ApiKey.revoked == False)  # noqa: E712
    ).first()
    if key is None:
        return None
    user = session.get(User, key.user_id)
    if user is None or not user.is_active:
        return None
    key.last_used = datetime.now(timezone.utc)
    session.add(key)
    session.commit()
    return user, key


def list_keys(session: Session, user: User) -> list[ApiKey]:
    return list(session.exec(select(ApiKey).where(ApiKey.user_id == user.id, ApiKey.revoked == False)))  # noqa: E712


def revoke_key(session: Session, user: User, key_id: str) -> bool:
    key = session.get(ApiKey, key_id)
    if key is None or key.user_id != user.id:
        return False
    key.revoked = True
    session.add(key)
    session.commit()
    return True
