"""Small, dependency-free pagination helper for the History API."""

from __future__ import annotations

from math import ceil


def paginate(items: list, page: int = 1, page_size: int = 24) -> dict:
    """Return one page of ``items`` plus stable total/pages metadata.

    ``page`` is clamped to the last page so deleting the only item on the
    final page never leaves the UI on an empty, out-of-range page.
    """
    try:
        size = int(page_size)
    except (TypeError, ValueError):
        size = 24
    size = max(1, min(100, size))

    total = len(items)
    pages = int(ceil(total / size)) if total else 0

    try:
        current = int(page)
    except (TypeError, ValueError):
        current = 1
    if pages > 0:
        current = max(1, min(pages, current))
    else:
        current = 1

    start = (current - 1) * size
    return {
        "items": items[start:start + size],
        "total": total,
        "page": current,
        "page_size": size,
        "pages": pages,
    }
