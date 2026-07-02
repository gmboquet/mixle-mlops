"""Database engine/session. Local = SQLite (file under data_dir); cloud = Postgres (MIXLE_DATABASE_URL).
Same ORM models, switched by config — no code change between laptop and cluster."""
from __future__ import annotations

from collections.abc import Iterator

from sqlmodel import Session, SQLModel, create_engine

from ..config import get_settings

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        url = get_settings().resolved_database_url()
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args)
    return _engine


def init_db() -> None:
    # import every module that declares SQLModel tables so create_all registers them
    from ..accounts import models as _accounts  # noqa: F401
    from ..conversations import models as _conversations  # noqa: F401
    from ..datasets import models as _datasets  # noqa: F401
    from ..evolve import models as _evolve  # noqa: F401
    from ..feedback import models as _feedback  # noqa: F401
    from ..rag import models as _rag  # noqa: F401
    from ..training import models as _training  # noqa: F401
    SQLModel.metadata.create_all(get_engine())


def get_session() -> Iterator[Session]:
    with Session(get_engine()) as session:
        yield session
