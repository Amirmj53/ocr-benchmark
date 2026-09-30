"""Page cache contract.

Darsio wraps these interfaces with its DB (SQLAlchemy model + service
functions). The engine depends only on the PageCache protocol, so tests use
InMemoryPageCache and production uses a thin DB-backed adapter.

Idempotency rule: a cached page is a hit only when document_id, page_no,
content checksum, dpi, source method AND engine_version all match. A bare
text lookup (get_page_text) is never used to skip OCR -- it only short-
circuits the text path when the caller knows the content matches.
"""

from __future__ import annotations

from typing import Protocol

from darsio_ingest.api_types import CachedPage


class PageCache(Protocol):
    """Storage-agnostic page cache. Implement with your DB session."""

    def get_page_text(self, document_id: str, page_no: int) -> str | None:
        """Raw text lookup (spec contract). May return stale content."""
        ...

    def get_cached_page(self, document_id: str, page_no: int) -> CachedPage | None:
        """Identity-aware lookup; the engine uses this to skip work."""
        ...

    def upsert_page_text(self, page: CachedPage) -> None:
        """Insert or update one page row."""
        ...


class InMemoryPageCache:
    """Dict-backed implementation for tests, the CLI, and examples."""

    def __init__(self) -> None:
        self._pages: dict[tuple[str, int], CachedPage] = {}

    def get_page_text(self, document_id: str, page_no: int) -> str | None:
        page = self._pages.get((document_id, page_no))
        return page.text if page else None

    def get_cached_page(self, document_id: str, page_no: int) -> CachedPage | None:
        return self._pages.get((document_id, page_no))

    def upsert_page_text(self, page: CachedPage) -> None:
        self._pages[(page.document_id, page.page_no)] = page

    def __len__(self) -> int:
        return len(self._pages)
