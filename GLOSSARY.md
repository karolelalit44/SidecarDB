# sidecarDB

Single context: the sidecarDB package — a plug-and-play natural-language search engine for
PostgreSQL. It introspects a database schema at runtime, plans a search as a constrained JSON
object, and executes that plan against the live database in a sandbox; it returns rows with a
plain-English answer.

## Language

**SidecarDB**:
The product: a natural-language search engine for PostgreSQL, shipped as an embeddable library
and an HTTP service.
_Avoid_: sidecardb (prose), "the planner" (meaning the whole product)

**Naming convention**:
Prose and class names spell `SidecarDB`; machine surfaces (distribution name, CLI command, log
namespace) spell lowercase `sidecardb`. No other spelling of the product name is used anywhere.

**Search**:
A user question answered with an Answer: the matching rows plus a plain-English explanation.
_Avoid_: query, question (as nouns for the whole operation)

**Answer**:
The plain-English explanation returned together with the rows for a Search.
_Avoid_: summary, narration, response

**Plan**:
The constrained JSON description of a Search, produced by the language model; never raw SQL.
_Avoid_: query plan, response

**Plan pipeline**:
The stages that turn a Search into a Plan — retrieval, prompt building, normalisation, checking,
retry on model error.
_Avoid_: graph, orchestrator

**Statement**:
The executable SQL artifact the Guard produces from a Plan; the only SQL form the sandbox runs.
_Avoid_: SQL, rendered query

**Guard**:
The single validator a Plan passes through on its way to becoming a Statement.
_Avoid_: render, sanitizer, API validator

**Catalog**:
The live representation of a PostgreSQL schema used for planning and checking a Plan.
_Avoid_: schema (reserved for wire DTOs), metadata, introspection result

