import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import logging_config
from app.config import settings
from app.db import database
from app.embeddings import embedder

logging_config.configure()
logger = logging.getLogger("backfill")


def text_columns(graph, table: str) -> list[str]:
    info = graph.tables.get(table)
    if info is None:
        return []
    return [c.name for c in info.columns if c.is_string and not c.primary_key]


async def backfill(graph, table: str, column: str, batch: int) -> int:
    sources = text_columns(graph, table)
    if not sources:
        logger.warning("%s has no text column to embed; skipping", table)
        return 0

    projection = ", ".join(f'"{name}"' for name in sources)
    rows = await database.fetch(f'SELECT ctid::text AS locator, {projection} FROM "{table}"')

    updated = 0
    for start in range(0, len(rows), batch):
        chunk = rows[start : start + batch]
        texts = [" ".join(str(r[name]) for name in sources if r.get(name)) for r in chunk]
        usable = [i for i, text in enumerate(texts) if text.strip()]
        if not usable:
            continue

        vectors = await embedder.aencode([texts[i] for i in usable])
        for i, vector in zip(usable, vectors, strict=True):
            literal = "[" + ",".join(f"{v:.8f}" for v in vector) + "]"
            await database.fetch(
                f'UPDATE "{table}" SET "{column}" = {literal}::vector WHERE ctid = %s',
                chunk[i]["locator"],
            )
            updated += 1
        logger.info("%s: %d/%d", table, updated, len(rows))

    return updated


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    if not settings.embedding_enabled:
        logger.error("EMBEDDING_ENABLED must be true in .env")
        return 1

    await database.connect(readonly=False)
    graph = await database.introspect()

    targets = [(c.table, c.column) for c in graph.indexed_columns]
    if not targets:
        logger.error("No extension-backed vector columns found in %s", settings.db_schema)
        await database.close()
        return 1

    logger.info("Backfilling %d column(s)", len(targets))
    total = 0
    for table, column in targets:
        try:
            total += await backfill(graph, table, column, args.batch)
        except Exception as exc:
            logger.error("%s.%s failed: %s", table, column, exc)

    logger.info("Done. %d row(s) embedded.", total)
    await database.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
