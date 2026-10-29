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
class Runner(Protocol):
    async def run(self, statement: Statement) -> list[dict]: ...


@runtime_checkable
class Catalog(Protocol):
    async def introspect(self) -> "SchemaGraph": ...

    async def ping(self) -> bool: ...
