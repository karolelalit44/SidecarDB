import pytest

from sidecardb.config import Settings
from sidecardb.exceptions import EmbeddingError
from sidecardb.types import ColumnInfo, SchemaGraph, TableInfo

CATEGORY = {
    "uuid": "U",
    "varchar": "S",
    "text": "S",
    "numeric": "N",
    "integer": "N",
    "bigint": "N",
    "date": "D",
    "timestamp with time zone": "D",
    "boolean": "B",
}


def column(name, data_type, *, description=None):
    return ColumnInfo(
        name=name,
        data_type=data_type,
        type_category=CATEGORY.get(data_type, "U"),
        description=description,
    )


def table(name, columns, *, description=None, rows=100):
    return TableInfo(
        name=name,
        description=description,
        columns=list(columns),
        row_estimate=rows,
    )


ECOMMERCE = SchemaGraph(
    tables={
        "products": table(
            "products",
            [
                column("id", "uuid"),
                column("sku", "varchar"),
                column("title", "varchar"),
                column("category", "varchar"),
                column("unit_price", "numeric"),
                column("stock_on_hand", "integer"),
            ],
            description="Sellable catalogue items",
        ),
        "customers": table(
            "customers",
            [
                column("id", "uuid"),
                column("display_name", "varchar"),
                column("email_address", "varchar"),
                column("city", "varchar"),
                column("signed_up_on", "timestamp with time zone"),
            ],
            description="Registered buyers",
        ),
        "orders": table(
            "orders",
            [
                column("id", "uuid"),
                column("order_reference", "varchar"),
                column("customer_id", "uuid"),
                column("placed_at", "timestamp with time zone"),
                column("order_state", "varchar"),
                column("grand_total", "numeric"),
            ],
            description="Customer purchase orders",
        ),
        "order_lines": table(
            "order_lines",
            [
                column("id", "uuid"),
                column("order_id", "uuid"),
                column("product_id", "uuid"),
                column("quantity", "integer"),
                column("line_total", "numeric"),
            ],
        ),
    }
)

HEALTHCARE = SchemaGraph(
    tables={
        "clinicians": table(
            "clinicians",
            [
                column("id", "uuid"),
                column("registration_code", "varchar"),
                column("practitioner_name", "varchar"),
                column("specialty", "varchar"),
            ],
            description="Licensed practitioners",
        ),
        "patients": table(
            "patients",
            [
                column("id", "uuid"),
                column("mrn", "varchar"),
                column("full_name", "varchar"),
                column("date_of_birth", "date"),
                column("enrolment_state", "varchar"),
            ],
            description="Individuals under care",
        ),
        "encounters": table(
            "encounters",
            [
                column("id", "uuid"),
                column("encounter_code", "varchar"),
                column("patient_id", "uuid"),
                column("clinician_id", "uuid"),
                column("visit_datetime", "timestamp with time zone"),
                column("visit_status", "varchar"),
                column("billed_amount", "numeric"),
            ],
            description="Scheduled visits",
        ),
    }
)

LIBRARY = SchemaGraph(
    tables={
        "authors": table(
            "authors",
            [
                column("id", "uuid"),
                column("author_name", "varchar"),
                column("birth_year", "integer"),
            ],
            description="Book authors",
        ),
        "titles": table(
            "titles",
            [
                column("id", "uuid"),
                column("accession_code", "varchar"),
                column("headline", "varchar"),
                column("author_id", "uuid"),
                column("circulation_status", "varchar"),
                column("acquired_on", "date"),
            ],
            description="Bibliographic catalogue entries",
        ),
        "loans": table(
            "loans",
            [
                column("id", "uuid"),
                column("title_id", "uuid"),
                column("member_id", "uuid"),
                column("borrowed_at", "timestamp with time zone"),
                column("returned_at", "timestamp with time zone"),
            ],
            description="Items checked out by members",
        ),
    }
)


class FakeEmbedder:
    """Deterministic stand-in: every text embeds to the same unit vector.

    All scores tie at the maximum, which is enough to exercise the embedding
    code path without downloading a model.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    async def aencode(self, texts):
        if not self.enabled:
            raise EmbeddingError("Embeddings are disabled.")
        return [[1.0, 0.0] for _ in texts]

    async def aclose(self) -> None:
        return None


class FakeDatabase:
    """A database that never connects; used where the pipeline is bypassed."""


@pytest.fixture
def settings() -> Settings:
    """Hermetic settings: no .env, no environment, deterministic values."""
    return Settings(
        _env_file=None,
        database_url="postgresql://app:app@localhost:5432/app_test",
        llm_api_key="test-key",
        embedding_provider="none",
        expose_sql=True,
        log_level="DEBUG",
    )


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def ecommerce() -> SchemaGraph:
    return ECOMMERCE


@pytest.fixture
def healthcare() -> SchemaGraph:
    return HEALTHCARE


@pytest.fixture
def library() -> SchemaGraph:
    return LIBRARY


@pytest.fixture(params=["ecommerce", "healthcare", "library"])
def any_graph(request) -> SchemaGraph:
    return request.getfixturevalue(request.param)
