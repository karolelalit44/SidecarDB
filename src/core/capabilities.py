from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from core.plan import Statement

if TYPE_CHECKING:
    from core.catalog import SchemaGraph


@runtime_checkable
class Planner(Protocol):
    async def complete_json(
        self, system_prompt: str, user_prompt: str, temperature: float | None = None
    ) -> dict[str, Any]: ...


@runtime_checkable
class Embedder(Protocol):
    @property
    def enabled(self) -> bool: ...

    async def aencode(self, texts: list[str]) -> list[list[float]]: ...

    async def aclose(self) -> None: ...


@runtime_checkable
class DatabaseGateway(Protocol):
    async def introspect(self) -> "SchemaGraph": ...

    async def execute(self, statement: Statement) -> list[dict]: ...

    async def ping(self) -> bool: ...

    async def close(self) -> None: ...


__all__ = [
    "DatabaseGateway",
    "Embedder",
    "Planner",
]
