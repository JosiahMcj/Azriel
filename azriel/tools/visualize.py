"""visualize tool -- inline widget rendering in the dashboard.

Wraps SVG / sanitized HTML in <viz>...</viz> markers. The dashboard's
chat renderer detects these blocks and renders them as a sandboxed
inline widget rather than as text.

Usage:
  visualize("<svg viewBox='0 0 100 100'><circle cx='50' cy='50' r='40' fill='currentColor'/></svg>")
  visualize("<table><tr><td>Hi</td></tr></table>")

Safety: only a small whitelist of tags+attrs is rendered client-side; the
runtime + dashboard strip <script>, event handlers, and external URL
references before display.
"""
import re

MAX_CHARS = 8000
SCRIPT_RE = re.compile(r"<script[^>]*>.*?</script>", re.S | re.I)
EVENT_RE = re.compile(r"\s*on[a-z]+\s*=\s*('|\")[^'\"]*\1", re.I)


def visualize(markup: str) -> str:
    if not isinstance(markup, str):
        return "ERROR: visualize expects a string of HTML/SVG."
    s = markup.strip()
    if not s:
        return "ERROR: empty markup."
    if len(s) > MAX_CHARS:
        return f"ERROR: markup too long ({len(s)} chars; max {MAX_CHARS})."
    s = SCRIPT_RE.sub("", s)
    s = EVENT_RE.sub("", s)
    # Marker the dashboard recognizes and lifts into an inline widget.
    return "<viz>" + s + "</viz>"


if __name__ == "__main__":
    print(visualize("<svg viewBox='0 0 24 24'><circle cx='12' cy='12' r='10' fill='gold'/></svg>"))
