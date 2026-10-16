import pytest

from app.types import (
    ColumnInfo,
    ForeignKeyInfo,
    IndexedColumn,
    SchemaGraph,
    TableInfo,
)

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


def column(name, data_type, *, pk=False, description=None, distinct=None):
    return ColumnInfo(
        name=name,
        data_type=data_type,
        type_category=CATEGORY.get(data_type, "U"),
        nullable=not pk,
        primary_key=pk,
        description=description,
        distinct_estimate=distinct if distinct is not None else 0.0,
    )


def table(name, columns, *, fks=(), description=None, rows=100, embeddings=()):
    return TableInfo(
        name=name,
        description=description,
        columns=list(columns),
        primary_keys=[c.name for c in columns if c.primary_key],
        foreign_keys=list(fks),
        row_estimate=rows,
        embedding_columns=list(embeddings),
    )


def fk(from_table, from_column, to_table, to_column):
    return ForeignKeyInfo(
        from_table=from_table,
        from_column=from_column,
        to_table=to_table,
        to_column=to_column,
    )


ECOMMERCE = SchemaGraph(
    tables={
        "products": table(
            "products",
            [
                column("id", "uuid", pk=True),
                column("sku", "varchar", distinct=180),
                column("title", "varchar", distinct=180),
                column("category", "varchar", distinct=6),
                column("unit_price", "numeric"),
                column("stock_on_hand", "integer"),
            ],
            description="Sellable catalogue items",
        ),
        "customers": table(
            "customers",
            [
                column("id", "uuid", pk=True),
                column("display_name", "varchar", distinct=90),
                column("email_address", "varchar", distinct=90),
                column("city", "varchar", distinct=14),
                column("signed_up_on", "timestamp with time zone"),
            ],
            description="Registered buyers",
        ),
        "orders": table(
            "orders",
            [
                column("id", "uuid", pk=True),
                column("order_reference", "varchar", distinct=120),
                column("customer_id", "uuid"),
                column("placed_at", "timestamp with time zone"),
                column("order_state", "varchar", distinct=4),
                column("grand_total", "numeric"),
            ],
            fks=[fk("orders", "customer_id", "customers", "id")],
            description="Customer purchase orders",
        ),
        "order_lines": table(
            "order_lines",
            [
                column("id", "uuid", pk=True),
                column("order_id", "uuid"),
                column("product_id", "uuid"),
                column("quantity", "integer"),
                column("line_total", "numeric"),
            ],
            fks=[
                fk("order_lines", "order_id", "orders", "id"),
                fk("order_lines", "product_id", "products", "id"),
            ],
        ),
    },
    indexed_columns=[
        IndexedColumn(
            table="products",
            column="semantic_vector",
            data_type="vector(384)",
            extension="vector",
        )
    ],
)
ECOMMERCE.tables["products"].embedding_columns = ["semantic_vector"]

HEALTHCARE = SchemaGraph(
    tables={
        "clinicians": table(
            "clinicians",
            [
                column("id", "uuid", pk=True),
                column("registration_code", "varchar", distinct=40),
                column("practitioner_name", "varchar", distinct=40),
                column("specialty", "varchar", distinct=9),
            ],
            description="Licensed practitioners",
        ),
        "patients": table(
            "patients",
            [
                column("id", "uuid", pk=True),
                column("mrn", "varchar", distinct=70),
                column("full_name", "varchar", distinct=70),
                column("date_of_birth", "date"),
                column("enrolment_state", "varchar", distinct=5),
            ],
            description="Individuals under care",
        ),
        "encounters": table(
            "encounters",
            [
                column("id", "uuid", pk=True),
                column("encounter_code", "varchar", distinct=200),
                column("patient_id", "uuid"),
                column("clinician_id", "uuid"),
                column("visit_datetime", "timestamp with time zone"),
                column("visit_status", "varchar", distinct=3),
                column("billed_amount", "numeric"),
            ],
            fks=[
                fk("encounters", "patient_id", "patients", "id"),
                fk("encounters", "clinician_id", "clinicians", "id"),
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
                column("id", "uuid", pk=True),
                column("author_name", "varchar", distinct=300),
                column("birth_year", "integer"),
            ],
            description="Book authors",
        ),
        "titles": table(
            "titles",
            [
                column("id", "uuid", pk=True),
                column("accession_code", "varchar", distinct=900),
                column("headline", "varchar", distinct=900),
                column("author_id", "uuid"),
                column("circulation_status", "varchar", distinct=4),
                column("acquired_on", "date"),
            ],
            fks=[fk("titles", "author_id", "authors", "id")],
            description="Bibliographic catalogue entries",
        ),
        "loans": table(
            "loans",
            [
                column("id", "uuid", pk=True),
                column("title_id", "uuid"),
                column("member_id", "uuid"),
                column("borrowed_at", "timestamp with time zone"),
                column("returned_at", "timestamp with time zone"),
            ],
            fks=[fk("loans", "title_id", "titles", "id")],
            description="Items checked out by members",
        ),
    }
)


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
