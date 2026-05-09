"""pdf_create -- stdlib PDF generator.

Closes the PDF generation gap surfaced in the hard-press
(document_create only handles docx/pptx/xlsx; "make me a PDF" had no
direct path before). Pure Python, no external dependencies (reportlab
isn't pip-installable in the uv-managed venv on the host machine, and libreoffice
isn't installed).

Output is real PDF-1.4 with Helvetica text. Handles multi-page
automatically when content overflows. Sandboxed at ~/azriel-files/
just like document_create.

Input shape: "name|content"
  name: file basename (no extension; .pdf is appended)
  content: free text, paragraphs separated by blank lines

Output: full path to the created PDF + size, or ERROR on failure.
"""
from __future__ import annotations

import re
from pathlib import Path

SANDBOX = Path.home() / "azriel-files"

# Helvetica metrics at 12pt (we hold them as ratios so we can render
# any size). Source: standard Type1 font metrics built into PDF readers.
PAGE_WIDTH = 612 # US Letter, 8.5" x 72
PAGE_HEIGHT = 792
MARGIN = 72 # 1 inch all sides
LINE_HEIGHT = 16 # 12pt + 4pt leading
CHAR_WIDTH = 6.0 # average for 12pt Helvetica


def _wrap(text: str, max_chars: int = 80) -> list[str]:
    """Greedy word-wrap into lines no wider than max_chars chars."""
    words = text.split()
    lines = []
    cur = ""
    for w in words:
        cand = (cur + " " + w).strip() if cur else w
        if len(cand) <= max_chars:
            cur = cand
        else:
            if cur:
                lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines or [""]


def _escape_pdf_text(s: str) -> str:
    """Escape characters that are special in a PDF string literal."""
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _resolve_in_sandbox(name: str) -> Path | None:
    """Place the output PDF at ~/azriel-files/<name>.pdf (sandboxed)."""
    name = name.strip().lstrip("/")
    if not name:
        return None
    if not name.endswith(".pdf"):
        name = name + ".pdf"
    p = (SANDBOX / name).resolve()
    try:
        p.relative_to(SANDBOX.resolve())
    except ValueError:
        return None
    return p


def _build_pdf(title: str, paragraphs: list[str]) -> bytes:
    """Assemble a minimal PDF-1.4 with one or more pages of body text."""
    text_width_chars = int((PAGE_WIDTH - 2 * MARGIN) / CHAR_WIDTH)
    lines_per_page = int((PAGE_HEIGHT - 2 * MARGIN) / LINE_HEIGHT) - 3
    body_lines: list[str] = []
    for i, para in enumerate(paragraphs):
        if i > 0:
            body_lines.append("") # blank line between paragraphs
        body_lines.extend(_wrap(para, max_chars=text_width_chars))
    pages: list[list[str]] = []
    cur: list[str] = []
    for line in body_lines:
        if len(cur) >= lines_per_page:
            pages.append(cur)
            cur = []
        cur.append(line)
    if cur:
        pages.append(cur)
    if not pages:
        pages = [[""]]

    objects: list[bytes] = []
    n_pages = len(pages)

    catalog_id = 1
    pages_id = 2
    page_ids = list(range(3, 3 + n_pages))
    contents_ids = list(range(3 + n_pages, 3 + 2 * n_pages))
    font_id = 3 + 2 * n_pages

    # 1: catalog
    objects.append(f"<< /Type /Catalog /Pages {pages_id} 0 R >>".encode())

    # 2: pages root
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(
        f"<< /Type /Pages /Kids [{kids}] /Count {n_pages} >>".encode()
    )

    # page objects + content streams
    for i, lines in enumerate(pages):
        page_obj = page_ids[i]
        contents_obj = contents_ids[i]
        objects.append(
            (
                f"<< /Type /Page /Parent {pages_id} 0 R "
                f"/MediaBox [0 0 {PAGE_WIDTH} {PAGE_HEIGHT}] "
                f"/Contents {contents_obj} 0 R "
                f"/Resources << /Font << /F1 {font_id} 0 R >> >> >>"
            ).encode()
        )
    # content streams (must come after pages so the first ref order is
    # stable; but we wrote pages 3..3+n-1 already; now we write
    # contents 3+n..3+2n-1)
    title_y = PAGE_HEIGHT - MARGIN
    for i, lines in enumerate(pages):
        stream_lines = []
        # title only on page 1
        y = title_y
        if i == 0 and title:
            stream_lines.append(
                f"BT /F1 18 Tf {MARGIN} {y} Td ({_escape_pdf_text(title)}) Tj ET"
            )
            y -= LINE_HEIGHT * 2
        for line in lines:
            stream_lines.append(
                f"BT /F1 12 Tf {MARGIN} {y} Td ({_escape_pdf_text(line)}) Tj ET"
            )
            y -= LINE_HEIGHT
        stream = "\n".join(stream_lines).encode()
        objects.append(
            f"<< /Length {len(stream)} >>\nstream\n".encode()
            + stream
            + b"\nendstream"
        )

    # font
    objects.append(
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
    )

    # Assemble file
    body = bytearray(b"%PDF-1.4\n%\xc4\xe5\xf2\xe5\xeb\xa7\xf3\xa0\xd0\xc4\xc6\n")
    offsets = [0]
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(body))
        body.extend(f"{i} 0 obj\n".encode() + obj + b"\nendobj\n")
    xref_pos = len(body)
    body.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    body.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        body.extend(f"{off:010d} 00000 n \n".encode())
    body.extend(
        f"trailer\n<< /Size {len(objects) + 1} /Root {catalog_id} 0 R >>\n".encode()
    )
    body.extend(f"startxref\n{xref_pos}\n%%EOF\n".encode())
    return bytes(body)


def pdf_create(arg: str) -> str:
    """Generate a PDF in the sandbox.

    Format: "name|content"
      name -- basename (.pdf appended automatically)
      content -- free text, paragraphs separated by blank lines

    The first line of content is treated as the title (rendered larger);
    subsequent paragraphs are body text. Auto-paginated.
    """
    if not arg or "|" not in arg:
        return "ERROR: expected 'name|content'."
    name, content = arg.split("|", 1)
    name = name.strip()
    content = content.strip()
    if not name:
        return "ERROR: empty name."
    if not content:
        return "ERROR: empty content."

    out = _resolve_in_sandbox(name)
    if out is None:
        return f"ERROR: name '{name}' resolves outside the sandbox."
    out.parent.mkdir(parents=True, exist_ok=True)

    paragraphs = re.split(r"\n\s*\n", content)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    title = paragraphs[0] if paragraphs else name
    body = paragraphs[1:] if len(paragraphs) > 1 else paragraphs

    try:
        pdf_bytes = _build_pdf(title, body)
        out.write_bytes(pdf_bytes)
    except Exception as e:
        return f"ERROR: {type(e).__name__}: {e}"

    return (
        f"created {out.name} ({len(pdf_bytes)} bytes) -- "
        f"accessible via fs_read or download from {out}"
    )
