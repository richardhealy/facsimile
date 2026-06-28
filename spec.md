# facsimile — Implementation & Status Plan

**Stack:** Python 3.12. A late-interaction visual retriever (ColPali / ColQwen-style), a vision-capable LLM for synthesis, a store for page-image embeddings (Qdrant with multivector support), FastAPI.
**One-liner:** A document RAG system that retrieves over pages as images, using late-interaction visual embeddings, so tables, charts, and layout survive instead of being flattened by text extraction.
**The single problem it isolates:** Visual document understanding. Text extraction destroys exactly what carries meaning in real business documents (a table's structure, a chart, a form's layout), and `facsimile` retrieves on the visual document directly.

---

## Why this (the showcase)

Anyone with a folder of financial PDFs, pitch decks, or scanned forms has felt this: the answer is in a table or a chart, and the text pipeline either mangles it or loses it. The decisions on show:

- **Visual retrieval with late interaction:** embed page images as multi-vector representations (ColPali-style) and match at the patch/token level, getting cross-encoder-grade accuracy without a separate OCR-and-chunk pipeline.
- **No lossy OCR step:** skip the brittle "extract text, hope the table survives" stage entirely; the page image is the unit.
- **Vision-grounded synthesis:** feed the retrieved page images to a vision LLM, so the answer reads the table or chart as rendered, and cites the page.
- **Honest comparison:** the same chart/table questions run through a text pipeline (`quarry`) and `facsimile`, with the difference shown.

It is current, it demos beautifully on a real annual report, and it solves a pain that text-only RAG simply cannot.

---

## Scope

**In:**
- Page rasterization and late-interaction visual embedding of each page.
- A multivector index and late-interaction scoring at query time.
- Vision-LLM synthesis over retrieved page images, citing page numbers.
- A benchmark on table/chart questions vs a text pipeline.
- FastAPI plus a viewer that highlights the retrieved page.

**Explicitly out (for v1):**
- Pure text corpora (that is `quarry`'s job; `facsimile` earns its cost on visually rich documents).
- Editing or extracting structured data out of the documents (retrieval and Q&A only).

---

## Architecture

```
facsimile/
  ingest/        # PDF -> page images -> late-interaction embeddings
  index/         # multivector store (Qdrant), per-page vectors
  retrieve/      # late-interaction scoring, top pages
  answer/        # vision-LLM synthesis over page images + page citations
  bench/         # table/chart question set, visual-vs-text comparison
  api/ + viewer/
```

Design rule: the page image is the atomic unit end to end, so nothing about a table or chart is lost between ingestion and synthesis. Cost per page (embedding and storage are heavier than text) is tracked and stated, because that is the real trade-off.

---

## Best-in-class quality checklist

- [ ] Late-interaction visual retrieval implemented (multivector, patch-level scoring).
- [ ] A question whose answer is inside a table is answered correctly and cites the page.
- [ ] README benchmark: visual vs text pipeline on chart/table questions, with the gap shown.
- [ ] The viewer highlights the retrieved page so the answer is verifiable.
- [ ] Cost and storage per page tracked and compared to a text pipeline.
- [ ] Works on a real, public annual report or deck end to end.

---

## Milestones & status

| # | Milestone | Outcome | Status |
|---|-----------|---------|--------|
| M0 | Scaffold | rasterization, multivector store, CI green | ☐ Not started |
| M1 | Visual embedding | late-interaction page embeddings indexed | ☐ Not started |
| M2 | Retrieval | late-interaction scoring, top-page selection | ☐ Not started |
| M3 | Vision synthesis | vision-LLM answers over pages + citations | ☐ Not started |
| M4 | Benchmark | table/chart set, visual-vs-text numbers | ☐ Not started |
| M5 | Ship | viewer with page highlight, README, release | ☐ Not started |

Status legend: ☐ Not started, ◐ In progress, ☑ Done, ⊘ Blocked.

---

## Definition of done

1. A question answered only by a table or chart returns the right answer and cites the page.
2. The README shows `facsimile` beating a text pipeline on visual questions, with numbers.
3. The viewer shows the retrieved page so a reader can verify.
4. Cost per page is measured and the trade-off stated.

## Stretch goals
- Hybrid with text retrieval for mixed corpora (route visual pages to `facsimile`, text to `quarry`).
- Token pruning / quantization to cut the visual-embedding cost.
- Bounding-box highlight of the exact region used.

## Relationship to the portfolio
`facsimile` is the visual branch `dispatch` routes image-heavy documents to, complementing `quarry` (text) and `lattice` (relationships).
