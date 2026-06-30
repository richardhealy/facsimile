"""Simple text-extraction pipeline for benchmark comparison.

This pipeline deliberately avoids visual features: it extracts raw text from
each PDF page using pypdfium2 and ranks pages by keyword overlap with the
query.  It is the baseline against which the visual pipeline is compared in
the M4 benchmark.

Public API:
- :func:`extract_page_texts` — raw text per page.
- :func:`text_retrieve` — keyword-ranked top-k pages.
- :class:`TextPage` — one ranked result item.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


# Common English stop words that carry no discriminative signal.
_STOPWORDS = frozenset(
    [
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "of", "in", "on", "at",
        "to", "for", "with", "by", "from", "as", "or", "and", "but", "not",
        "what", "which", "who", "how", "when", "where", "this", "that",
        "its", "it", "their", "they", "them", "we", "i", "you",
    ]
)


@dataclass
class TextPage:
    """One page returned by :func:`text_retrieve`."""

    page_num: int
    text: str
    score: float


def _tokenize(text: str) -> list[str]:
    """Lower-case the text, split on non-alphanumeric runs, drop stop words."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in _STOPWORDS]


def _score(query: str, text: str) -> float:
    """Keyword-overlap score: fraction of distinct query terms found in page text.

    Returns a value in ``[0, 1]``.  Zero if the query has no content terms.
    """
    query_terms = set(_tokenize(query))
    if not query_terms:
        return 0.0
    page_words = set(_tokenize(text))
    hits = len(query_terms & page_words)
    return hits / len(query_terms)


def extract_page_texts(pdf_path: Path) -> list[tuple[int, str]]:
    """Extract the text layer from every page in *pdf_path*.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of ``(page_num, text)`` pairs, one per page (1-based page numbers).
        Pages with no text layer return an empty string.
    """
    import pypdfium2 as pdfium  # noqa: PLC0415

    doc = pdfium.PdfDocument(str(pdf_path))
    results: list[tuple[int, str]] = []
    for i, page in enumerate(doc):
        textpage = page.get_textpage()
        text = textpage.get_text_range()
        results.append((i + 1, text))
    return results


def text_retrieve(pdf_path: Path, query: str, top_k: int = 3) -> list[TextPage]:
    """Return the *top_k* pages most relevant to *query* by keyword overlap.

    Pages are scored with :func:`_score` and returned in descending order.
    Ties are broken by page number (earlier pages first).

    Args:
        pdf_path: Path to the PDF to search.
        query: Natural-language query string.
        top_k: Maximum number of pages to return.

    Returns:
        List of :class:`TextPage` objects, highest score first.
    """
    pages = extract_page_texts(pdf_path)
    scored = [
        TextPage(page_num=page_num, text=text, score=_score(query, text))
        for page_num, text in pages
    ]
    scored.sort(key=lambda p: (-p.score, p.page_num))
    return scored[:top_k]
