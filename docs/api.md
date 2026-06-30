# facsimile — API Reference

**Version:** 0.1.0  
**Base URL:** `http://localhost:8000` (default when run with Uvicorn)  
**OpenAPI spec:** [`docs/openapi.json`](openapi.json) — importable into Postman, Insomnia, or any OpenAPI-compatible tool.

facsimile is a visual document RAG service. It accepts PDF uploads, indexes every page as a
late-interaction visual embedding, and answers natural-language questions by retrieving the
most relevant page images and passing them to a Claude vision model.

---

## Authentication

No authentication is required in the default configuration. When deploying externally, place a
reverse proxy (nginx, Caddy, API gateway) in front of the service and handle auth there.

The vision-synthesis endpoint (`POST /ask`) calls the Anthropic API internally. The API key
must be available to the server process via the `ANTHROPIC_API_KEY` environment variable.

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `QDRANT_URL` | *(in-memory)* | Qdrant server URL. Omit to run Qdrant in-process (dev / test). |
| `DATA_DIR` | `facsimile_data` | Directory where rasterized page PNGs are stored. |
| `ANTHROPIC_API_KEY` | *(required for `/ask`)* | Anthropic API key used by the vision synthesizer. |

---

## Endpoints

### `GET /health`

Liveness check. Returns immediately with no side effects.

**Response `200 application/json`**

```json
{ "status": "ok", "version": "0.1.0" }
```

---

### `POST /ingest`

Upload a PDF. facsimile rasterizes every page to a PNG, embeds each page as a late-interaction
multivector representation, and upserts the vectors into Qdrant.

The rasterized images are persisted under `{DATA_DIR}/{doc_id}/page_NNNN.png` so they can be
served by `GET /pages/{doc_id}/{page_num}` and displayed in the viewer.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|---|---|---|---|
| `file` | binary (PDF) | yes | The PDF file to ingest. Must have a `.pdf` extension. |

The `doc_id` used for all subsequent calls is derived from the uploaded filename stem
(e.g. uploading `annual_report_2024.pdf` creates `doc_id = "annual_report_2024"`).

**Response `200 application/json`**

```json
{
  "doc_id": "annual_report_2024",
  "pages": 42
}
```

| Field | Type | Description |
|---|---|---|
| `doc_id` | string | Stable identifier for this document (filename stem). |
| `pages` | integer | Number of pages rasterized, embedded, and indexed. |

**Error responses**

| Status | Condition |
|---|---|
| `422` | The uploaded file does not have a `.pdf` extension. |
| `503` | The vector store has not finished initialising (transient at startup). |

**Example**

```bash
curl -X POST http://localhost:8000/ingest \
     -F "file=@annual_report_2024.pdf"
```

---

### `POST /retrieve`

Embed a natural-language query and return the most relevant indexed pages, ranked by
MaxSim late-interaction score.

**Request** — `application/json`

```json
{
  "query": "What was the gross profit margin in Q3?",
  "top_k": 5
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | string | yes | — | Natural-language question or search string. |
| `top_k` | integer | no | `5` | Maximum number of pages to return. |

**Response `200 application/json`**

```json
{
  "pages": [
    {
      "doc_id": "annual_report_2024",
      "page_num": 17,
      "doc_path": "annual_report_2024",
      "score": 28.47
    }
  ]
}
```

**`PageInfo` fields**

| Field | Type | Description |
|---|---|---|
| `doc_id` | string | Document identifier (matches the value returned by `/ingest`). |
| `page_num` | integer | 1-based page number within the document. |
| `doc_path` | string | Document path / identifier used internally in the index. |
| `score` | float | MaxSim late-interaction relevance score. Higher is more relevant. |

**Error responses**

| Status | Condition |
|---|---|
| `503` | Store not initialised (transient at startup). |

**Example**

```bash
curl -X POST http://localhost:8000/retrieve \
     -H "Content-Type: application/json" \
     -d '{"query": "gross profit margin Q3", "top_k": 3}'
```

---

### `POST /ask`

Retrieve the top-`k` pages for the query, load their rasterized images from disk, and
pass them to a Claude vision model. Returns a grounded answer with `[Page N]` citation
markers and the list of pages that were shown to the model.

**Request** — `application/json`

```json
{
  "query": "What was the gross profit margin in Q3?",
  "top_k": 3
}
```

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `query` | string | yes | — | Natural-language question to answer. |
| `top_k` | integer | no | `3` | Number of pages to retrieve and pass to the vision model. |

**Response `200 application/json`**

```json
{
  "answer": "The gross profit margin in Q3 was 38.2% [Page 17].",
  "cited_pages": [17],
  "model": "claude-opus-4-8",
  "retrieved_pages": [
    {
      "doc_id": "annual_report_2024",
      "page_num": 17,
      "doc_path": "annual_report_2024",
      "score": 28.47
    }
  ]
}
```

| Field | Type | Description |
|---|---|---|
| `answer` | string | Vision-model response with inline `[Page N]` citation markers. |
| `cited_pages` | integer[] | Page numbers cited in the answer, in ascending order. |
| `model` | string | Claude model ID used for synthesis (e.g. `"claude-opus-4-8"`). |
| `retrieved_pages` | PageInfo[] | Pages passed to the vision model (see `POST /retrieve` for field docs). |

**Special cases**

If no pages are indexed yet, `answer` is a human-readable message and `cited_pages` is
empty; no Anthropic API call is made. Same if the rasterized PNGs are missing on disk
(re-ingesting the document resolves this).

**Error responses**

| Status | Condition |
|---|---|
| `503` | Store not initialised (transient at startup). |

**Example**

```bash
curl -X POST http://localhost:8000/ask \
     -H "Content-Type: application/json" \
     -d '{"query": "What was the gross profit margin in Q3?", "top_k": 3}'
```

---

### `GET /pages/{doc_id}/{page_num}`

Serve a rasterized page image. Used by the viewer to display retrieved pages alongside the
answer.

**Path parameters**

| Parameter | Type | Description |
|---|---|---|
| `doc_id` | string | Document identifier returned by `POST /ingest`. |
| `page_num` | integer | 1-based page number. |

**Response `200 image/png`**

The raw PNG bytes of the rasterized page.

**Error responses**

| Status | Condition |
|---|---|
| `404` | The page image does not exist (document not ingested, or wrong page number). |
| `503` | Store not initialised. |

**Example**

```bash
curl http://localhost:8000/pages/annual_report_2024/17 -o page17.png
```

---

### `GET /viewer`

Returns the interactive page-highlight viewer as a self-contained HTML page. Open it in a
browser; it presents a text field for questions, calls `POST /ask` internally, renders the
retrieved page images, and highlights cited pages with a gold border.

**Response `200 text/html`**

The full HTML single-page application. No additional server or build step required.

**Example**

```bash
open http://localhost:8000/viewer       # macOS
xdg-open http://localhost:8000/viewer   # Linux
```

---

## Schemas

### `IngestResponse`

```json
{
  "doc_id": "string",
  "pages": 0
}
```

### `RetrieveRequest`

```json
{
  "query": "string",
  "top_k": 5
}
```

### `RetrieveResponse`

```json
{
  "pages": [ /* PageInfo[] */ ]
}
```

### `AskRequest`

```json
{
  "query": "string",
  "top_k": 3
}
```

### `AskResponse`

```json
{
  "answer": "string",
  "cited_pages": [0],
  "model": "string",
  "retrieved_pages": [ /* PageInfo[] */ ]
}
```

### `PageInfo`

```json
{
  "doc_id": "string",
  "page_num": 0,
  "doc_path": "string",
  "score": 0.0
}
```

---

## Typical workflow

```
1. Start the server
   uvicorn facsimile.api.main:app --reload

2. Ingest a document
   POST /ingest  (multipart PDF upload)
   → { "doc_id": "report", "pages": 42 }

3. Ask a question
   POST /ask  { "query": "What was revenue in FY2024?" }
   → { "answer": "...[Page 7]...", "cited_pages": [7], ... }

4. Browse the viewer
   GET /viewer  → open in browser; page images with cited-page highlight
```

---

## Error envelope

Validation and 4xx/5xx errors follow FastAPI's standard envelope:

```json
{
  "detail": "Only PDF files are accepted"
}
```

For request-body validation failures (status `422`), the detail is an array of
[`ValidationError`](openapi.json) objects with `loc`, `msg`, and `type` fields.
