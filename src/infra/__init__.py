"""External adapters: PostgreSQL access, the LLM provider, embeddings, execution.

``infra`` is the only place that knows about asyncpg, langchain and
sentence-transformers; everything in ``core`` consumes these capabilities
through small interfaces.
"""
