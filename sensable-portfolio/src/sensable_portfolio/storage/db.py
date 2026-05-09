"""Async SQLite engine + session helpers."""
from __future__ import annotations

from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel import SQLModel
from sqlmodel.ext.asyncio.session import AsyncSession


async def init_engine(url: str) -> AsyncEngine:
    engine = create_async_engine(url, echo=False, future=True)
    async with engine.begin() as conn:
        if url.startswith("sqlite") and ":memory:" not in url:
            await conn.exec_driver_sql("PRAGMA journal_mode=WAL")
        await conn.run_sync(SQLModel.metadata.create_all)
    return engine


@asynccontextmanager
async def get_session(engine: AsyncEngine):
    factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with factory() as s:
        yield s
